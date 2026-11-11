"""Provenance helpers: record exactly which code produced a result."""

from pathlib import Path
import subprocess
from typing import Any, Dict


def git_info() -> Dict[str, Any]:
    """Current commit of this repo, and whether it has uncommitted changes."""
    repo_dir = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo_dir, stdout=subprocess.PIPE, check=True
        ).stdout.decode()

    return {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git("status", "--porcelain").strip()),
    }
