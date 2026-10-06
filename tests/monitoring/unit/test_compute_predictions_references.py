"""Tests for monitoring/compute_predictions_references.py helper functions.

The module only imports api.predictor.TilePredictor lazily (inside main(), see
the module docstring/comment there), so this runs with plain
requirements/monitoring_req.txt like the rest of tests/monitoring/unit -- no
torch/mlflow needed. The helper functions tested here (_group_members) are
pure data-transformation functions that don't load models or images.
"""
from __future__ import annotations

from monitoring.compute_predictions_references import _group_members


class TestGroupMembers:
    def test_empty(self):
        assert _group_members([]) == []

    def test_single_sample_multiple_channels(self):
        """One benchmark sample with 3 channels -> one group with 3 channel dicts."""
        rows = [
            {"benchmark_id": 1, "plate": "P1", "well": "K07", "field": 1,
             "t_label": "ClassA", "channel_index": 0, "channel": 1,
             "root_path": "/data", "file_name": "f1.jxl"},
            {"benchmark_id": 1, "plate": "P1", "well": "K07", "field": 1,
             "t_label": "ClassA", "channel_index": 1, "channel": 2,
             "root_path": "/data", "file_name": "f2.jxl"},
            {"benchmark_id": 1, "plate": "P1", "well": "K07", "field": 1,
             "t_label": "ClassA", "channel_index": 2, "channel": 3,
             "root_path": "/data", "file_name": "f3.jxl"},
        ]
        grouped = _group_members(rows)
        assert len(grouped) == 1
        sample = grouped[0]
        assert sample["benchmark_id"] == 1
        assert sample["plate"] == "P1"
        assert sample["well"] == "K07"
        assert sample["field"] == 1
        assert sample["t_label"] == "ClassA"
        assert len(sample["channels"]) == 3
        assert sample["channels"][0]["channel"] == 1
        assert sample["channels"][0]["file_name"] == "f1.jxl"
        assert sample["channels"][0]["root_path"] == "/data"

    def test_multiple_samples(self):
        """Two benchmark samples -> two groups."""
        rows = [
            {"benchmark_id": 1, "plate": "P1", "well": "K07", "field": 1,
             "t_label": "ClassA", "channel_index": 0, "channel": 1,
             "root_path": "/data", "file_name": "f1.jxl"},
            {"benchmark_id": 2, "plate": "P1", "well": "K08", "field": 1,
             "t_label": "ClassB", "channel_index": 0, "channel": 1,
             "root_path": "/data", "file_name": "f2.jxl"},
        ]
        grouped = _group_members(rows)
        assert len(grouped) == 2
        assert grouped[0]["benchmark_id"] == 1
        assert grouped[1]["benchmark_id"] == 2

    def test_channels_preserve_input_order(self):
        """Channels appear in the order they were in the input rows."""
        rows = [
            {"benchmark_id": 1, "plate": "P1", "well": "K07", "field": 1,
             "t_label": "ClassA", "channel_index": 2, "channel": 3,
             "root_path": "/data", "file_name": "f3.jxl"},
            {"benchmark_id": 1, "plate": "P1", "well": "K07", "field": 1,
             "t_label": "ClassA", "channel_index": 0, "channel": 1,
             "root_path": "/data", "file_name": "f1.jxl"},
            {"benchmark_id": 1, "plate": "P1", "well": "K07", "field": 1,
             "t_label": "ClassA", "channel_index": 1, "channel": 2,
             "root_path": "/data", "file_name": "f2.jxl"},
        ]
        grouped = _group_members(rows)
        channels = grouped[0]["channels"]
        assert channels[0]["channel"] == 3
        assert channels[1]["channel"] == 1
        assert channels[2]["channel"] == 2

    def test_sample_metadata_consistent(self):
        """All rows for a sample share the same plate/well/field/t_label."""
        rows = [
            {"benchmark_id": 5, "plate": "Exp1", "well": "C12", "field": 3,
             "t_label": "treatment_x", "channel_index": 0, "channel": 1,
             "root_path": "/mnt/data", "file_name": "img.jxl"},
            {"benchmark_id": 5, "plate": "Exp1", "well": "C12", "field": 3,
             "t_label": "treatment_x", "channel_index": 1, "channel": 2,
             "root_path": "/mnt/data", "file_name": "img2.jxl"},
        ]
        grouped = _group_members(rows)
        s = grouped[0]
        assert s["plate"] == "Exp1"
        assert s["well"] == "C12"
        assert s["field"] == 3
        assert s["t_label"] == "treatment_x"
