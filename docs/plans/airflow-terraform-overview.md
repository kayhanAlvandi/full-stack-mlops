# Overview: Airflow & Terraform — Concepts and Use Cases (details TBD)

General ideas and use cases for the two tools planned *after* the Kubernetes
work in `kubernetes-plan.md`. Deliberately kept at a conceptual level — we'll
come back and flesh out concrete DAGs / Terraform modules once Phases 1–2
(CD + Kubernetes) are implemented and this project actually has K8s workloads
to orchestrate and cloud infrastructure to provision.

Order: **Kubernetes (see `kubernetes-plan.md`) → Airflow → Terraform (cloud).**

---

## Part 1 — Airflow

### What it actually is (and isn't)

Airflow is a **workflow orchestrator**, not an infrastructure manager. It does
not run or scale live services and does not watch a cluster's Pods on its own.
It defines *what tasks run, in what order, on what schedule, and what happens
on success/failure/branch*. It can *launch* a K8s Pod as one way to execute a
task (`KubernetesPodOperator`), but the orchestration logic (dependencies,
branching, retries, scheduling) is Airflow's job, not K8s's.

Division of labor to keep straight:
- **K8s** = "run and scale this workload, keep it healthy." (the *how/where*)
- **Airflow** = "run task A, and only if it succeeds run B and C in parallel,
  then if C's result exceeds a threshold run D, else run E — every night at
  2am, retrying failed tasks 3 times." (the *what/when/in-what-order*)

Airflow is **not**:
- For serving / long-running processes (that's K8s Deployments). It orchestrates
  finite tasks that end.
- For real-time, per-request reactions (that's streaming infra like
  Kafka/Flink, or the K8s serving path). Airflow is batch/scheduled
  (seconds-to-days granularity).
- A CI/CD tool. GitHub Actions builds/tests/ships *code and images*; Airflow
  runs *data/ML workflows* on a schedule, on top of already-built images.

### Core vocabulary

- **DAG** (Directed Acyclic Graph): the pipeline. Nodes = tasks, edges = dependencies.
- **Task / Operator**: one unit of work (`PythonOperator`, `BashOperator`, `KubernetesPodOperator`, `SQLOperator`, ...).
- **Dependencies**: e.g. `a >> [b, c] >> d` = "a, then b and c in parallel, then d."
- **Branching** (`BranchPythonOperator` / `@task.branch`): pick a downstream path based on a computed result.
- **XCom**: how tasks pass small results to each other (e.g. "candidate F1 = 0.91"), enabling branching decisions downstream.
- **Sensors**: a task that waits for an external condition (a file lands, a DB row appears, an MLflow run gets tagged).
- **Scheduling + backfill**: DAGs run on a cron-like schedule; Airflow can re-run a DAG "as of" any past date (backfill historical windows).
- **Retries / trigger rules**: per-task retry policy; trigger rules (`all_success`, `one_failed`, `none_failed`, ...) control when a task fires relative to its parents.

### Concrete real-world ML use cases

1. **Scheduled retraining pipeline (the classic).**
   `extract_new_data → validate_data → build_features → train_model → evaluate_on_holdout → BRANCH: if new_metric > current_prod_metric → register_model + trigger_deploy, else → alert("candidate underperformed")`.
2. **Feature engineering / ETL into a feature store.**
   `pull_raw_events → dedup/clean → aggregate_features → write_to_feature_store → validate_freshness`, hourly. (Airbnb/Uber-style feature pipelines.)
3. **Batch scoring pipelines.**
   `load_todays_records → dynamic fan-out over chunks → score each chunk → merge_predictions → write_to_warehouse → data_quality_check`. Fan-out width decided at runtime from data size (dynamic task mapping).
4. **Monitoring / data-quality pipelines.**
   `compute_drift → compute_quality → BRANCH: if drift_detected OR quality_dropped → alert + open_ticket, else → log_ok`.
5. **This project's promotion pipeline** (the strongest fit for this repo):
   `find_versions_tagged(benchmark_test) → filter_out_already_scored (idempotency) → dynamic fan-out: score_candidate_i (KubernetesPodOperator running compute_predictions_references) → compute_benchmark_quality → compare_against_champion → BRANCH: if best_candidate beats champion by threshold → set_champion_alias + trigger_redeploy, else → notify/no-op`.
   Every Airflow feature earns its keep here: dependencies, dynamic fan-out, XCom (passing scores), branching on result, retries on flaky scoring.

### How it combines with Kubernetes

The production-grade pattern (and this project's direction): **Airflow-on-K8s
dispatching tasks as Pods.**
- Airflow itself runs as Pods in the cluster.
- Each DAG task uses `KubernetesPodOperator` (or the `KubernetesExecutor`) to
  launch an ephemeral Pod running one of the job images from
  `kubernetes-plan.md`, wait for it, capture its result, and branch on it.
- K8s provides isolated, resource-limited, GPU-schedulable execution; Airflow
  provides the graph/branching/scheduling/retries on top.

### Alternatives (same job, different flavor — for awareness, not adopting now)

- **Prefect / Dagster** — more modern, Python-native, nicer local dev; Dagster leans into data-asset lineage.
- **Argo Workflows** — K8s-native (every step *is* a Pod by definition), YAML-defined; natural once you're all-in on K8s.
- **Flyte / Kubeflow Pipelines** — ML-specialized, strong typing/caching of step outputs, popular at larger ML orgs.

Airflow is the most widely-adopted and most transferable to learn first, which is why it's the chosen tool here.

### This project's planned DAGs (sketch — refine once Phases 1–2 land)

1. **DAG 1 — weekly monitoring:** `label_backfill → run_quality_report (depends on backfill)` and `run_drift_report` (parallel) → branch (alert vs. no-op) on regression thresholds. Alerting channel still TBD.
   - **Open decision (moved from `kubernetes-plan.md`):** the scheduling mechanism for these weekly jobs — K8s `CronJob`s vs. this Airflow DAG. The jobs already exist as plain `Job` manifests (`k8s/jobs/{label-backfill,drift-report,quality-report}.yaml`). Leaning Airflow, so the dependency/branch/alert logic lives in one place rather than being split across CronJob schedules + a DAG; decide when Airflow is actually stood up.
2. **DAG 2 — promotion:** discover MLflow versions tagged `benchmark_test` → skip any already scored (idempotent: check `benchmark_quality` for the run_id) → fan out `compute_predictions_references(--target benchmark)` per new candidate → `compute_benchmark_quality` (upsert) → compare all candidates + current `@champion` in DB with **threshold + best** rule → conditional `set_champion_alias` → downstream redeploy/reload trigger.
3. Practice-oriented features to deliberately exercise: backfill/reprocessing over date-partitioned monitoring windows (`window_start`/`window_end`), sensors polling MLflow for new `benchmark_test` tags, dynamic task mapping for the candidate fan-out.
4. Prereqs this surfaces for other code (tracked, not built yet):
   - Add `@champion` alias resolution to `api/predictor.py` (`get_model_version_by_alias`) — today it only handles `Name/<version>`/`latest`.
   - `benchmark_quality` table (one row per model, `UNIQUE(run_id)`) + `quality_report` FK refactor + a `compute_benchmark_quality` step.
   - A mechanism for the served API to pick up a promoted champion (poll+reload vs. redeploy).

---

## Part 2 — Terraform

### What it actually is

**Infrastructure as Code (IaC).** Instead of clicking around a cloud console
(or running one-off `aws`/`gcloud` CLI commands) to create a cluster,
database, bucket, etc., you write declarative config describing the desired
infrastructure, and Terraform figures out the API calls to create/update/
destroy real cloud resources to match. Version-controlled, reviewable (a plan
shows exactly what will be created/changed/destroyed before it happens), and
reproducible across environments.

**Where it sits relative to the other tools:**
- **Terraform** provisions the *cloud resources themselves* — the K8s
  cluster, VPC/networking, managed Postgres, object storage, IAM roles, DNS,
  container registry.
- **Kubernetes manifests** (`kubernetes-plan.md`'s `k8s/` dir) configure *what
  runs inside* that cluster once it exists.
- **Airflow** orchestrates ML workflows running on that cluster.
- **GitHub Actions/CD** builds and ships application images.

Four layers, each the right tool for its job.

### Concrete ML-project use cases

1. **Provisioning the K8s cluster itself.** A Terraform module (e.g.
   `terraform-aws-modules/eks`) defines the EKS cluster, node groups
   (including a GPU node group with the right instance type/autoscaling),
   networking, and IAM — as reviewable code instead of manual `eksctl` runs.
2. **Managed databases.** Postgres becomes RDS (AWS) / Cloud SQL (GCP) instead
   of a self-managed StatefulSet — Terraform provisions it, sets backup
   policies, security groups, and exports connection info for app config.
3. **Object storage for MLflow artifacts / datasets.** An S3/GCS bucket for
   `mlruns` artifacts, with lifecycle policies (e.g. auto-delete old
   experiment artifacts after 90 days), replacing the local `mlruns/` volume.
4. **Container registry.** If moving off GHCR to cloud-native (ECR/Artifact
   Registry), Terraform provisions it with retention/scanning policies.
5. **IAM/secrets plumbing.** Roles/service-accounts so K8s Pods can pull
   images, read secrets, and write to storage — least-privilege access
   defined as code (reviewable/auditable once real credentials are involved).
6. **GPU node pools with autoscaling.** A Terraform-defined node group that
   scales 0→N GPU nodes on demand — what makes "training as a K8s Job" (from
   `kubernetes-plan.md`) actually cost-effective in the cloud.
7. **Multi-environment parity.** `environments/staging/` and
   `environments/prod/` configs sharing the same modules but different
   sizes/settings — lets a "local kind → cloud" jump also become "cloud
   staging → cloud prod" cleanly, using the same infra definitions.

### Is it usable for this project? Yes — after local K8s + some Airflow practice

Terraform has nothing to provision while everything runs locally on kind (no
cloud account, no billable resources). It earns its place once there's:
- A real cloud account with billable resources not worth hand-clicking.
- More than one environment (even "my cloud sandbox" benefits — tear down and
  recreate identically for cost reasons).
- Resources beyond the cluster itself (managed DB, storage, IAM) that need
  coordinating together.

So: **Phase 4, after Kubernetes (Phases 1–2) and Airflow (Phase 3) are working
locally**, when the "cloud-managed later" step from the original K8s decision
is actually taken. At that point Terraform provisions the EKS/GKE cluster +
RDS/Cloud SQL + storage bucket + registry, and the existing `k8s/` manifests
apply into that cluster largely unchanged — which is the whole point of having
kept them portable.

---

## Suggested execution order (overall)
1. `kubernetes-plan.md` Phases 1–2 (CD → GHCR, then local kind K8s).
2. Airflow (Part 1 above), orchestrating the K8s jobs via `KubernetesPodOperator` — detail once Phase 2 lands.
3. Terraform (Part 2 above), provisioning real cloud infra for the cluster + stateful services — detail once Airflow is working locally.

## Progress log
- [ ] Airflow — DAG 1 (weekly monitoring)
- [ ] Airflow — DAG 2 (promotion)
- [ ] Terraform — cloud provisioning (cluster, DB, storage, registry, IAM)
