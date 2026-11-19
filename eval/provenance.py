"""Identify the simulator actually executed, including uncommitted source edits."""

import hashlib
from pathlib import Path
import subprocess


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "eval").glob("*.py"))
    paths += sorted((root / "app/src/main/java/com/echoshield/echonode/service").glob("*.kt"))
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in paths}
    model = root / "app/src/main/assets/yamnet.tflite"
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    return {
        "execution_kind": "python_simulator_with_real_tflite_inference",
        "source_commit_sha": head,
        "working_tree_dirty": dirty,
        "source_file_sha256": sources,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "android_runtime_executed_by_replay": False,
    }
