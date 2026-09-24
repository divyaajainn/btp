from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def code_version(repo_dir: str | Path = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for name in ("pandas", "scipy", "sklearn"):
        try:
            module = __import__(name)
            versions[name] = module.__version__
        except Exception:
            versions[name] = "unavailable"
    return versions


def write_run_metadata(path: str | Path, config: dict[str, Any], seed: int) -> None:
    payload = {
        "seed": seed,
        "config": config,
        "code_version": code_version(Path.cwd()),
        "dependencies": dependency_versions(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "python_executable": sys.executable,
        "pid": os.getpid(),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

