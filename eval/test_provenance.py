"""Ensure replay output describes its actual sources rather than a fixed SHA."""

import hashlib
from pathlib import Path

from eval.provenance import source_provenance


def test_provenance_hashes_the_executed_simulator():
    data = source_provenance()
    root = Path(__file__).resolve().parents[1]
    name = "eval/detector_pipeline.py"
    assert data["source_file_sha256"][name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert data["android_runtime_executed_by_replay"] is False
    assert data["source_commit_sha"] is None or len(data["source_commit_sha"]) == 40
    assert "tested_detector_head_sha" not in data
