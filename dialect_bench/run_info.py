import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version


def _command(args, cwd=None):
    """Output of a command, or None if it isn't available (e.g. no GPU or no git)."""
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(input_files, repo_dir):
    """Everything about this run that doesn't depend on the model."""
    gpus = _command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    commit = _command(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    changes = _command(["git", "status", "--porcelain"], cwd=repo_dir)
    return {
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpus": gpus.splitlines() if gpus else [],
        "git_commit": commit,
        # if this is true, the code differs from the commit above
        "uncommitted_changes": bool(changes) if changes is not None else None,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["pandas", "pyyaml", "requests", "tqdm"]},
        # the hash shows if an input file changed between runs
        "input_files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in input_files},
    }


def save(folder, info):
    """Adds one line to run_info.jsonl. A resumed run adds a new line, so every session is kept."""
    folder.mkdir(parents=True, exist_ok=True)
    line = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}
    with open(folder / "run_info.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
