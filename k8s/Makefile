# kind cluster lifecycle.
#
# kind does not expand environment variables in its own config file, so
# host-machine paths (../data, plus the externally-supplied O_DRIVE_PATH /
# TOOLS_LIB_PATH from .env) are rendered into kind-cluster.yaml via envsubst
# before being handed to `kind create cluster`.
#
# Requires envsubst (GNU gettext). On Windows, install via Git Bash's package
# manager or `choco install gettext`; it ships by default on Linux/macOS.

# leading '-' -- don't hard-fail if .env doesn't exist yet (see .env.example)
-include .env
export

# Always "<repo>/data" / "<repo>/monitoring/reports" / "<repo>/mlflow-data"
# relative to this Makefile -- never hand-set, so they're correct on every
# machine without editing .env. reports needs to exist upfront: kind's
# extraMounts (and Docker bind mounts generally) require the host path to
# already exist.
DATA_PATH := $(abspath ../data)
REPORTS_PATH := $(abspath ../monitoring/reports)
MLFLOW_DATA_PATH := $(abspath ../mlflow-data/mlruns)

export DATA_PATH REPORTS_PATH MLFLOW_DATA_PATH

CLUSTER_NAME := image-classifier

# Must match the image: in k8s/api/api-deployement.yaml -- bootstrap-dev
# builds and loads exactly this tag so the Deployment manifest never needs
# to change between "local dev" and "the real pushed image".
API_IMAGE := ghcr.io/kayhanalvandi/full-stack-mlops/api:latest

kind-up:
	python -c "import os,string,sys; sys.stdout.write(string.Template(open('kind-cluster.yaml').read()).substitute(os.environ))" | kind create cluster --name $(CLUSTER_NAME) --config -

kind-down:
	kind delete cluster --name $(CLUSTER_NAME)

# ── Bootstrap: everything needed after a fresh kind-up ───────────────────
# kind wipes the node's image store on every kind-down/kind-up, so
# imagePullPolicy: IfNotPresent will otherwise silently pull the real
# :latest from GHCR instead of using local changes -- load-api-image exists
# so that can never happen by accident in dev.
#
# NOT covered here (gitignored / credential-bearing, apply once yourself):
#   kubectl create secret generic api-secrets -n api --from-literal=API_DB_URI=...
#   kubectl create secret docker-registry ghcr-pull -n api ...   (only if GHCR package is private)
build-api-image:
	docker build -f ../docker/services/api/Dockerfile -t $(API_IMAGE) ..

load-api-image: build-api-image
	kind load docker-image $(API_IMAGE) --name $(CLUSTER_NAME)

addons:
	kubectl apply -f addons/metrics-server.yaml
	kubectl apply -f addons/ingress-nginx.yaml
	kubectl wait --namespace ingress-nginx --for=condition=ready pod \
		--selector=app.kubernetes.io/component=controller --timeout=180s



# Final/stable: no local build/load -- relies on the real image already
# pushed to GHCR and imagePullPolicy: IfNotPresent pulling it normally.
bootstrap: addons
	kubectl apply -f namespaces/ || kubectl apply -f namespaces/
	kubectl apply -k .

bootstrap-dev: load-api-image addons
	kubectl apply -f namespaces/ || kubectl apply -f namespaces/
	kubectl apply -k .

# ── Simulate live prediction traffic ─────────────────────────────────────
# Exercise the served API with the manifest-driven traffic generator
# (scripts/simulate_live_predictions.py) two ways:
#
#   simulate-internal -- from a pod *inside* the cluster, hitting the API's
#                        in-cluster Service DNS. Proves pod-to-pod serving
#                        (Service + readiness gating) independent of ingress.
#                        Runs in the debug-simulate-live pod, which carries the
#                        data/O-drive host mounts; `requests` is already in the
#                        api image (pulled in by mlflow), so no extra install.
#   simulate-external -- from the host, hitting the ingress hostname. Proves
#                        ingress-nginx routes external traffic to the API.
#
# Override the traffic args per run, e.g.:
#   make simulate-internal INTERNAL_ARGS="--manifest /mnt/data/custom_v1/dataset_manifest.json --count 20"
#   make simulate-external EXTERNAL_ARGS="--manifest ../data/custom_v1/dataset_manifest.json --loop"
SIM_POD := simulate-live
SIM_SCRIPT := ../scripts/simulate_live_predictions.py
INTERNAL_ARGS ?= --manifest /mnt/data/custom_v1/dataset_manifest.json --count 10
EXTERNAL_ARGS ?= --manifest ../data/custom_v1/dataset_manifest.json --count 10

simulate-internal:
	kubectl apply -f jobs/debug-simulate-live.yaml
	kubectl wait --for=condition=Ready pod/$(SIM_POD) -n api --timeout=180s
	kubectl cp $(SIM_SCRIPT) api/$(SIM_POD):/tmp/simulate_live_predictions.py
	kubectl exec -it -n api $(SIM_POD) -- \
		python /tmp/simulate_live_predictions.py \
		--base-url http://api-service.api.svc.cluster.local:8000 $(INTERNAL_ARGS)

# Remove the debug pod when done (it just sleeps infinity otherwise).
simulate-internal-clean:
	kubectl delete -f jobs/debug-simulate-live.yaml --ignore-not-found

simulate-external:
	python $(SIM_SCRIPT) --base-url http://api.localtest.me $(EXTERNAL_ARGS)

