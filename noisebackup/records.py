"""Provenance helpers: record exactly which code produced a result."""

from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Union

REPO_DIR = Path(__file__).resolve().parents[1]

PathLike = Union[str, Path]


def git_info(repo_dir: PathLike = REPO_DIR, ignore: Optional[PathLike] = None) -> Dict[str, Any]:
    """Current commit of the repo, and whether it has uncommitted changes.

    `ignore` is a folder whose contents don't count as changes: a run's own output folder,
    so that the results being written don't make every run look "dirty".
    """
    repo_dir = Path(repo_dir).resolve()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo_dir, stdout=subprocess.PIPE, check=True
        ).stdout.decode()

    # --untracked-files=all lists each untracked file, not just "results/", so the
    # exclude below can match files inside the ignored folder.
    status_args = ["status", "--porcelain", "--untracked-files=all"]
    if ignore is not None:
        ignore = Path(ignore).resolve()
        if repo_dir in ignore.parents:
            status_args += ["--", ".", f":(exclude){ignore.relative_to(repo_dir)}"]

    return {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git(*status_args).strip()),
    }
