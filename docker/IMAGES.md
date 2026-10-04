# Images & CD

Where each image is built, published, and pulled from. Registry:
`ghcr.io/kayhanalvandi/full-stack-mlops/<name>` (GHCR names are
lowercase; `docker/metadata-action` lowercases the owner automatically).

## Images

| Image (`<name>`) | Dockerfile | Built by | Trigger |
|---|---|---|---|
| `api` | `docker/services/api/Dockerfile` | `ci_serving.yml` (`serving-build-push`, `needs: test-push-serving`) | push to `main` on `api/**` etc., or `workflow_dispatch` |
| `monitoring` | `docker/jobs/monitoring/Dockerfile` | `ci_monitoring.yml` (`monitoring-build-push`, `needs:` all monitoring test jobs) | push to `main` on `monitoring/**` etc., or `workflow_dispatch` |
| `mlflow` | `docker/services/mlflow/Dockerfile` | `cd_mlflow.yml` (no tests to gate on) | push to `main` on `docker/services/mlflow/**`, or `workflow_dispatch` |
| `training` | `docker/jobs/train/Dockerfile` | `ci_training.yml` (`training-build-push`) | **`workflow_dispatch` only** (`run_tests: all` + optional `image_tag`); never on push — see `docs/plans/kubernetes-plan.md` |
| `build_dataset` | `docker/jobs/build_dataset/Dockerfile` | — (local only) | not published; local dev tool |

Build/push logic is shared via the composite action
`.github/actions/build-push-image`.

## Tags (per `docker/metadata-action`)

- `latest` — only on the default branch (`main`).
- `sha-<shortsha>` — every build (immutable reference; use this to pin a K8s Deployment or roll back).
- `<version>` — on `vX.Y.Z` git tag pushes (semver).
- custom `extra_tag` — `training` only, from the `image_tag` dispatch input (e.g. an experiment/config name).

Every image also carries OCI labels incl. `org.opencontainers.image.revision`
(full commit SHA) — so any image is traceable to its exact commit via
`docker inspect`, even one tagged with a human-readable name.

## Auth

CI authenticates to GHCR with the `GHCR_TOKEN` repo secret (a PAT with
`write:packages`), passed into the composite action as its `token` input
(the `secrets` context is not available inside composite actions, so it must
be passed explicitly). Ensure that secret exists before triggering the
workflows.

## Pulling

```bash
docker pull ghcr.io/kayhanalvandi/full-stack-mlops/api:latest
# private packages: authenticate first
echo $GHCR_TOKEN | docker login ghcr.io -u <username> --password-stdin
```

## Compose

`docker/services/api`, `docker/services/mlflow`, `docker/jobs/monitoring`, and
`docker/jobs/compute_references` reference the GHCR images (pull, no local
build). Local builds are kept for iteration where pushing isn't wanted:
`docker/jobs/train` (training), `docker/jobs/build_dataset`, and the api
`docker-compose.dev.yaml` overlay (`serve-dev`, tagged `api:dev`).
