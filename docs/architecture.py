"""Render the project's architecture diagram as a PNG (diagrams-as-code).

This produces `docs/architecture.png`, the end-to-end view used at the top of the
README: training -> MLflow registry -> serving -> Postgres -> monitoring, all on
Kubernetes with external stateful services.

Prerequisites (one-time):
    pip install diagrams          # https://diagrams.mingrammer.com
    # Graphviz must also be installed and on PATH:
    #   Windows:  choco install graphviz   (or https://graphviz.org/download/)
    #   macOS:    brew install graphviz
    #   Linux:    apt-get install graphviz

Render:
    python docs/architecture.py      # writes docs/architecture.png

The .py script is the source of truth and is diffable in git; the generated PNG
is what the README embeds. Re-run after changing the architecture.
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.k8s.clusterconfig import HPA
from diagrams.k8s.compute import Deployment, Job
from diagrams.k8s.network import Ingress
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.client import Users
from diagrams.onprem.container import Docker
from diagrams.onprem.database import Mongodb, PostgreSQL
from diagrams.onprem.mlops import Mlflow
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

GRAPH_ATTR = {
    "fontsize": "22",
    "bgcolor": "white",
    "pad": "0.6",
    "splines": "spline",
}

with Diagram(
    "Image Classifier - MLOps Platform",
    filename="docs/architecture",
    show=False,
    direction="LR",
    graph_attr=GRAPH_ATTR,
):
    client = Users("Client / acquisition")

    # Stateful services that live outside the cluster (host now, managed cloud later).
    with Cluster("External stateful services"):
        mlflow = Mlflow("MLflow\ntracking + registry")
        postgres = PostgreSQL("Postgres\npredictions + monitoring")
        mongo = Mongodb("MongoDB\nground-truth labels")

    # Build & publish images.
    with Cluster("CI/CD (GitHub Actions -> GHCR)"):
        ci = GithubActions("5 scoped workflows\nlint + test + build/push")
        registry = Docker("GHCR images\napi / monitoring / mlflow")
        ci >> Edge(label="build & push") >> registry

    # The training side (human-in-the-loop, not auto-scheduled).
    with Cluster("Training (Lightning + Hydra)"):
        train = Python("train.py")
        train >> Edge(label="params / metrics / model + dataset version") >> mlflow

    # Everything that runs in-cluster.
    with Cluster("Kubernetes (kind)"):
        ingress = Ingress("ingress-nginx\napi.localtest.me")

        with Cluster("api namespace"):
            api = Fastapi("FastAPI /predict\nTilePredictor")
            hpa = HPA("HPA (CPU)")
            hpa >> Edge(style="dashed", label="scale") >> api

        with Cluster("monitoring namespace (batch jobs)"):
            compute = Job("compute-references")
            drift = Job("drift-report")
            quality = Job("quality-report")
            backfill = Job("label-backfill")
            benchmark = Job("register-benchmark")

        api_deploy = Deployment("rolling restart\non model change")
        api_deploy >> Edge(style="dashed") >> api

    # Serving request path.
    client >> Edge(label="upload image") >> ingress >> api
    api >> Edge(label="load model by name/run") >> mlflow
    api >> Edge(label="log predictions") >> postgres

    # Monitoring data flows.
    compute >> Edge(label="score val + benchmark") >> postgres
    compute >> Edge(style="dashed", label="load model") >> mlflow
    benchmark >> Edge(label="register benchmark set") >> postgres
    backfill << Edge(label="resolve labels") << mongo
    backfill >> Edge(label="backfill t_label") >> postgres
    postgres >> Edge(label="live vs reference (unsupervised)") >> drift
    postgres >> Edge(label="benchmark vs labeled live (supervised)") >> quality

    # Images consumed by the in-cluster workloads.
    registry >> Edge(style="dotted", color="grey") >> api
    registry >> Edge(style="dotted", color="grey") >> compute
