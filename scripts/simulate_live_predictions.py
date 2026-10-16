"""Simulate live production traffic against a running prediction API.

Reads a dataset manifest (built with ``scripts/build_dataset.py``, the same
artifact format used for the benchmark set) and, at randomized intervals,
sends one sample at a time to the API's ``/predict`` endpoint over HTTP --
one file-per-channel multipart POST, exactly like ``tests/diagnostics/
test_predict_api.py`` and a real acquisition pipeline would send one scanned
image at a time.

Talks to the API's host:port directly (default ``http://127.0.0.1:8000``),
not through the Docker network: this stands in for a real *external* client
(e.g. the acquisition software), not another job inside docker-compose.
Requires the ``api``/``postgres`` services to already be running
(``make serve`` or ``make serve-dev``).

Each request goes through the normal ``/predict`` path with no
``is_reference``/``benchmark_id``/``label`` in the metadata, so rows land as
plain production traffic (``is_reference=FALSE``, ``benchmark_id IS NULL``,
``t_label IS NULL`` until ``backfill_labels.py`` resolves it later).

Requires the third-party ``requests`` package (``pip install requests``) --
this is a host-side script, not part of any Docker image.

Usage:
    # Send the whole manifest once, one sample every 1-3 minutes, shuffled.
    python scripts/simulate_live_predictions.py --manifest data/live_v1/dataset_manifest.json

    # Keep looping forever (reshuffling each pass) until Ctrl+C.
    python scripts/simulate_live_predictions.py --manifest data/live_v1/dataset_manifest.json --loop

    # Fixed number of requests, custom interval and API host.
    python scripts/simulate_live_predictions.py --manifest data/live_v1/dataset_manifest.json \\
        --count 20 --min-interval 30 --max-interval 90 --base-url http://192.168.1.10:8000
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import re
import time
from contextlib import ExitStack
from pathlib import Path

import requests

_MNT_DRIVE_RE = re.compile(r"^/mnt/([A-Za-z])(/.*)?$")


def local_root_path(root_path: str, data_root: str | None = None) -> Path:
    """Resolve a manifest's ``root_path`` to a path readable on this machine.

    With ``data_root``, the manifest's ``/mnt/<letter>`` mount prefix is
    replaced by that host directory (the mounted drive's contents live
    directly under it), keeping any path remainder intact. Without it:
    on Windows, translate ``/mnt/<letter>/...`` to a drive letter; elsewhere
    the manifest path is already correct as-is.
    """
    match = _MNT_DRIVE_RE.match(root_path.replace("\\", "/"))
    if data_root:
        rest = (match.group(2) or "").lstrip("/") if match else root_path.lstrip("/")
        return Path(data_root, rest) if rest else Path(data_root)
    if platform.system() != "Windows" or not match:
        return Path(root_path)
    drive, rest = match.group(1).upper(), match.group(2) or ""
    return Path(f"{drive}:{rest}")


def load_manifest_samples(manifest_path: str) -> list[dict]:
    """Read a dataset manifest (flat ``samples`` or training's ``val_samples``)."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    samples = manifest.get("samples")
    if samples is None:
        samples = manifest.get("train_samples", []) + manifest.get("val_samples", [])
    if not samples:
        raise ValueError(f"No samples found in manifest: {manifest_path}")
    return samples


def send_sample(base_url: str, sample: dict, timeout: float,
                data_root: str | None = None) -> dict:
    """POST one sample's channel files to /predict, one file per channel.

    Mirrors tests/diagnostics/test_predict_api.py's multipart shape: files
    sent in ascending channel order (matches training/predict input order),
    root_path as a plain form field. The manifest's own root_path (e.g.
    /mnt/O/...) is sent as-is -- it's just metadata the API stores, and should
    stay consistent with reference/benchmark rows written from inside Docker;
    only the *local* file read is translated for this host.
    """
    manifest_root_path = sample["root_path"]
    read_root = local_root_path(manifest_root_path, data_root)
    channel_items = sorted(sample["channel_files"].items(), key=lambda kv: int(kv[0]))

    with ExitStack() as stack:
        multipart = []
        for _channel, file_name in channel_items:
            fh = stack.enter_context(open(read_root / file_name, "rb"))
            suffix = Path(file_name).suffix.lstrip(".")
            multipart.append(("files", (file_name, fh, f"image/{suffix}")))
        response = requests.post(
            f"{base_url}/predict",
            files=multipart,
            data={"root_path": manifest_root_path},
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate live production traffic against a running prediction API."
    )
    parser.add_argument("--manifest", required=True,
                        help="Path to a dataset manifest built by scripts/build_dataset.py.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="Base URL of the running API (default: %(default)s).")
    parser.add_argument("--data-root", default=None,
                    help="Host directory holding what the manifest's /mnt/<letter> "
                         "mount root points at (e.g. D:\\personal_project\\microscopy_data).")
    parser.add_argument("--min-interval", type=float, default=10.0,
                        help="Minimum seconds between requests (default: %(default)s).")
    parser.add_argument("--max-interval", type=float, default=50.0,
                        help="Maximum seconds between requests (default: %(default)s).")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of requests to send. Default: one pass over the manifest.")
    parser.add_argument("--loop", action="store_true",
                        help="Keep looping over the manifest (reshuffling each pass) until "
                             "--count is reached or the process is interrupted.")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Send samples in manifest order instead of a random order.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling and interval sampling.")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-request HTTP timeout in seconds (default: %(default)s).")
    args = parser.parse_args()

    if args.min_interval > args.max_interval:
        parser.error("--min-interval cannot be greater than --max-interval.")

    samples = load_manifest_samples(args.manifest)
    print(f"Loaded {len(samples)} samples from {args.manifest}")

    rng = random.Random(args.seed)
    if not args.no_shuffle:
        rng.shuffle(samples)

    try:
        health = requests.get(f"{args.base_url}/health", timeout=10).json()
        print(f"API health: {health}")
    except requests.RequestException as e:
        print(f"WARNING: could not reach {args.base_url}/health ({e}); continuing anyway.")

    n_sent, n_failed = 0, 0
    try:
        while True:
            for sample in samples:
                if args.count is not None and n_sent + n_failed >= args.count:
                    break

                key = (sample["plate"], sample["well"], sample["field"])
                try:
                    result = send_sample(args.base_url, sample, args.timeout, args.data_root)
                    n_sent += 1
                    print(f"[{n_sent + n_failed}] {key} -> {result.get('predicted_class')} "
                          f"(run_id={result.get('run_id')})")
                except Exception as e:  # noqa: BLE001
                    n_failed += 1
                    print(f"[{n_sent + n_failed}] {key} FAILED: {e}")

                if args.count is not None and n_sent + n_failed >= args.count:
                    break
                sleep_s = rng.uniform(args.min_interval, args.max_interval)
                print(f"  sleeping {sleep_s:.0f}s...")
                time.sleep(sleep_s)

            reached_count = args.count is not None and n_sent + n_failed >= args.count
            if reached_count or not args.loop:
                break
            if not args.no_shuffle:
                rng.shuffle(samples)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"Done. {n_sent} sent, {n_failed} failed.")


if __name__ == "__main__":
    main()
