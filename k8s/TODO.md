# Kubernetes TODO

K8s-specific follow-ups. Project-wide items live in the repo-root `TODO.md`.

## imagePullPolicy + Kustomize overlays

`api/api-deployement.yaml` and `jobs/debug-simulate-live.yaml` use
`imagePullPolicy: IfNotPresent` for both production and dev. These want
different behavior:

- **Production:** `Always` — every rollout should pick up the latest CI-pushed
  `:latest` from GHCR.
- **Dev:** `IfNotPresent` + `kind load docker-image` — trust the locally built,
  not-yet-pushed image instead of silently re-pulling `:latest`.

Today `bootstrap` happens to work only because it doesn't `kind load` after
`kind-up`, so the node pulls from GHCR; `bootstrap-dev` loads the local image
first. This is fragile and couples the pull behavior to which make target ran.

**Fix:** introduce a Kustomize base + `local` / `prod` overlays that patch
`imagePullPolicy` (and fold in the build/load step), replacing the current
`bootstrap` vs `bootstrap-dev` split. This is the natural place to also move the
API/job manifests under overlay management if desired.

**Status:** not started.

## ghcr-pull secret must exist in both namespaces

Both the `api` and `monitoring` default ServiceAccounts reference a `ghcr-pull`
imagePullSecret, but Secrets are namespaced. If the GHCR package is private,
ensure `ghcr-pull` is created in **both** namespaces (the
`config/secret-examples/ghcr-pull-secret.example.yaml` template / setup docs
should cover both). Harmless no-op while the package is public.

**Status:** verify.

## Minor: manifest/plan alignment

- `api/api-deployement.yaml` has no `replicas:` (defaults to 1; the plan mentions
  2 — HPA `minReplicas: 1` makes it moot, but align for clarity).
- `kind-cluster.yaml` is single-node; the plan mentions control-plane + worker.
  Single-node is the right default and **not** a missed optimization: in kind
  every node is just a Docker container on the same host, so multi-node gives no
  extra CPU/RAM — it only *simulates* a topology. Multi-node is worth it only to
  practice node-level scheduling (pod anti-affinity / `topologySpreadConstraints`
  to spread API replicas, taints/tolerations + affinity for a future GPU node,
  node drain + `PodDisruptionBudget`, DaemonSets). Caveat: our volumes are
  `hostPath` mounts tied to the control-plane node's `extraMounts`
  (`/mnt/data`, `/mnt/o-drive`, `/mnt/mlruns`, `/mnt/monitoring-reports`); a pod
  landing on a worker wouldn't see them, so going multi-node must be paired with
  moving those to PVCs. Revisit near the cloud phase.
- `make kind-up` could `mkdir -p` the reports dir so the extraMount never fails
  on a fresh checkout.

**Status:** low priority.
