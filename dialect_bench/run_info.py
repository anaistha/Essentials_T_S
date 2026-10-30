import json
import subprocess
from datetime import datetime, timezone


def gpu():
    """Name of the GPU, or None if there is none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        return out.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def save(folder, info):
    """Adds one line to run_info.jsonl. A resumed run adds a new line, so every session is kept."""
    folder.mkdir(parents=True, exist_ok=True)
    line = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}
    with open(folder / "run_info.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
