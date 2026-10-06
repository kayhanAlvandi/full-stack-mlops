"""Render the project's architecture diagram as a PNG (diagrams-as-code).

This produces `docs/diagram/architecture.png`, the end-to-end view used at the
top of the README: training -> MLflow registry -> serving -> Postgres ->
monitoring, all on Kubernetes with external stateful services.

Prerequisites (one-time):
    pip install diagrams          # https://diagrams.mingrammer.com
    # Graphviz must also be installed and on PATH:
    #   Windows:  choco install graphviz   (or https://graphviz.org/download/)
    #   macOS:    brew install graphviz
    #   Linux:    apt-get install graphviz

Render (from the repo root):
    python docs/diagram/architecture.py      # writes docs/diagram/architecture.png

The .py script is the source of truth and is diffable in git; the generated PNG
is what the README embeds. Re-run after changing the architecture.
"""

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.generic.storage import Storage
from diagrams.k8s.clusterconfig import HPA
from diagrams.k8s.compute import Job
from diagrams.k8s.network import Ingress
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.client import Users
from diagrams.onprem.container import Docker
from diagrams.onprem.database import PostgreSQL
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

# Absolute, forward-slash path so Graphviz finds the icon regardless of the
# shell's working directory (a relative image path that misses is silently
# rendered as a label-only node with no error).
MLFLOW_ICON = (Path(__file__).parent / "assets" / "mlflow.png").as_posix()

GRAPH_ATTR = {
    "fontsize": "22",
    "bgcolor": "white",
    "pad": "0.6",
    "splines": "spline",
}

with Diagram(
    "Full-Stack MLOps Platform",
    filename="docs/diagram/architecture",
    show=False,
    direction="TB",
    graph_attr=GRAPH_ATTR,
):
    # Smaller, secondary nodes. Height stays a bit taller than width so the
    # label has room to sit *below* the scaled-down icon instead of being
    # drawn over it; labels on these must be short (narrow nodes clip wide
    # text onto the icon).
    small = {"width": "0.95", "height": "1.35", "fontsize": "11"}

    # Main data-flow spine (train -> register -> serve -> log -> monitor),
    # drawn bold so the eye follows the real pipeline first; CI/CD (dotted)
    # and autoscale (dashed) stay muted as secondary annotations.
    spine = {"color": "#1a3e72", "penwidth": "2.2"}

    client = Users("Client / acquisition")

    # Stateful services live outside the cluster (host now, managed cloud
    # later). Kept as separate free-standing nodes -- not boxed together --
    # so each sits next to whatever it talks to instead of tangling lines.
    # MLflow uses a local high-contrast asset (the built-in icon renders faint).
    mlflow = Custom(
        "MLflow (external)\ntracking + registry",
        MLFLOW_ICON,
        width="2.0",
        height="1.6",
    )
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
    with Cluster("① Training (Lightning + Hydra)"):
        train = Python("train.py")
        train >> Edge(label="model + dataset version", **spine) >> mlflow

    # Everything that runs in-cluster.
    with Cluster("Kubernetes (kind)"):
        with Cluster("② Serving — api namespace"):
            api = Fastapi("FastAPI /predict\nTilePredictor")
            ingress = Ingress("ingress", **small)
            hpa = HPA("HPA", **small)
            ingress >> Edge(style="dashed") >> api
            hpa >> Edge(style="dashed", label="autoscale") >> api

        with Cluster("③ Monitoring — monitoring namespace"):
            monitor = Job("batch jobs\ndrift + quality reports")

    # Serving request path.
    client >> Edge(label="upload image", **spine) >> ingress
    mlflow >> Edge(label="load model", **spine) >> api
    api >> Edge(label="log predictions", **spine) >> postgres

    # Monitoring reads predictions / writes report rows, and renders to disk.
    monitor >> Edge(label="read predictions / write reports", forward=True, reverse=True, **spine) >> postgres
    monitor >> Edge(label="render") >> reports

    # Close the loop: monitoring informs the human-triggered retraining.
    monitor >> Edge(style="dashed", label="retrain trigger (human review)") >> train

    # One image per workload built & pushed to GHCR (api / monitoring / mlflow).
    registry >> Edge(style="dotted", color="grey") >> api
    registry >> Edge(style="dotted", color="grey") >> monitor
    registry >> Edge(style="dotted", color="grey") >> mlflow
