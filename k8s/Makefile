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

# Always "<repo>/data" / "<repo>/monitoring/reports" relative to this
# Makefile -- never hand-set, so they're correct on every machine without
# editing .env. reports needs to exist upfront: kind's extraMounts (and
# Docker bind mounts generally) require the host path to already exist.
DATA_PATH := $(abspath ../data)
REPORTS_PATH := $(abspath ../monitoring/reports)

export DATA_PATH REPORTS_PATH

CLUSTER_NAME := image-classifier

kind-up:
	python -c "import os,string,sys; sys.stdout.write(string.Template(open('kind-cluster.yaml').read()).substitute(os.environ))" | kind create cluster --name $(CLUSTER_NAME) --config -

kind-down:
	kind delete cluster --name $(CLUSTER_NAME)
