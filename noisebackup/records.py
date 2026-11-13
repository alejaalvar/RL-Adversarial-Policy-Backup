"""Provenance helpers: record exactly which code produced a result."""

from pathlib import Path
import subprocess
from typing import Any, Dict, Optional, Union

REPO_DIR = Path(__file__).resolve().parents[1]

PathLike = Union[str, Path]


# Where runs write their outputs. New (untracked) files here are other runs' outputs, not code.
OUTPUT_FOLDERS = ("models", "results")


def git_info(repo_dir: PathLike = REPO_DIR, ignore: Optional[PathLike] = None) -> Dict[str, Any]:
    """Current commit of the repo, and whether it has uncommitted changes ("dirty").

    Dirty means: any change to a tracked file, or any new untracked file outside
    OUTPUT_FOLDERS. New files inside OUTPUT_FOLDERS don't count, so runs can go back to back
    without a commit in between.

    `ignore` is a folder whose contents don't count at all: a run's own output folder, so
    that the results being written (even into an already committed file, when a sweep
    resumes) don't make the run look dirty.
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

    changes = [line for line in git(*status_args).splitlines() if not _new_output(line)]
    return {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(changes),
    }


def _new_output(status_line: str) -> bool:
    """Is this `git status --porcelain` line a new untracked file inside OUTPUT_FOLDERS?"""
    # Untracked lines look like "?? models/run/weights.pkl" (path relative to the repo root).
    if not status_line.startswith("?? "):
        return False
    return status_line[3:].strip('"').split("/")[0] in OUTPUT_FOLDERS
