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
from diagrams.generic.storage import Storage
from diagrams.k8s.clusterconfig import HPA
from diagrams.k8s.compute import Job
from diagrams.k8s.network import Ingress
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.client import Users
from diagrams.onprem.container import Docker
from diagrams.onprem.database import PostgreSQL
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
    "MLOps Platform",
    filename="docs/architecture",
    show=False,
    direction="TB",
    graph_attr=GRAPH_ATTR,
):
    # Smaller, secondary nodes. Height stays a bit taller than width so the
    # label has room to sit *below* the scaled-down icon instead of being
    # drawn over it; labels on these must be short (narrow nodes clip wide
    # text onto the icon).
    small = {"width": "0.95", "height": "1.35", "fontsize": "11"}

    client = Users("Client / acquisition")

    # Stateful services live outside the cluster (host now, managed cloud
    # later). Kept as separate free-standing nodes -- not boxed together --
    # so each sits next to whatever it talks to instead of tangling lines.
    mlflow = Mlflow("MLflow (external)\ntracking + registry")
    postgres = PostgreSQL("Postgres (external)\npredictions + monitoring")

    # Monitoring writes its rendered reports to a mounted volume on the host.
    # Short label: shrunk nodes clip multi-line text into the icon.
    reports = Storage("reports", **small)

    # Build & publish images. Keep the cluster title short and wrapped -- a
    # long one-line title forces the whole box as wide as the text.
    with Cluster("CI/CD -> GHCR\nlint · test · build/push"):
        ci = GithubActions("GitHub Actions", width="1.8", height="2.2", fontsize="11")
        # Wider/taller than `small` so the image list fits as a label under
        # the icon (narrow nodes clip wide text onto the image).
        registry = Docker(
            "GHCR\napi / monitoring / mlflow",
            width="1.8",
            height="2.3",
            fontsize="11",
        )
        ci >> Edge(label="build & push") >> registry

    # The training side (human-in-the-loop, not auto-scheduled).
    with Cluster("Training (Lightning + Hydra)"):
        train = Python("train.py")
        train >> Edge(label="model + dataset version") >> mlflow

    # Everything that runs in-cluster.
    with Cluster("Kubernetes (kind)"):
        with Cluster("api namespace"):
            api = Fastapi("FastAPI /predict\nTilePredictor")
            ingress = Ingress("ingress", **small)
            hpa = HPA("HPA", **small)
            ingress >> Edge(style="dashed") >> api
            hpa >> Edge(style="dashed", label="autoscale") >> api

        with Cluster("monitoring namespace"):
            monitor = Job("batch jobs\ndrift + quality reports")

    # Serving request path.
    client >> Edge(label="upload image") >> ingress
    mlflow >> Edge(label="load model") >> api
    api >> Edge(label="log predictions") >> postgres

    # Monitoring reads predictions / writes report rows, and renders reports to disk.
    monitor >> Edge(label="read predictions / write reports", forward=True, reverse=True) >> postgres
    monitor >> Edge(label="render") >> reports

    # One image per workload built & pushed to GHCR (api / monitoring / mlflow).
    registry >> Edge(style="dotted", color="grey") >> api
    registry >> Edge(style="dotted", color="grey") >> monitor
    registry >> Edge(style="dotted", color="grey") >> mlflow
