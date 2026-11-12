"""Tests for noisebackup.records.git_info: which code produced a result, and was it committed."""

import subprocess

import pytest

from noisebackup.records import git_info


@pytest.fixture
def repo(tmp_path):
    """A throwaway git repo with one committed file and nothing else."""

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    git("init")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-m", "initial")
    return tmp_path


def test_a_run_s_own_output_folder_does_not_count_as_uncommitted_changes(repo):
    (repo / "results" / "run").mkdir(parents=True)
    (repo / "results" / "run" / "episodes.csv").write_text("sigma\n")

    assert git_info(repo, ignore=repo / "results" / "run")["git_dirty"] is False
    assert git_info(repo)["git_dirty"] is True


def test_changes_outside_the_ignored_folder_still_count(repo):
    (repo / "results" / "run").mkdir(parents=True)
    (repo / "code.py").write_text("x = 2\n")

    assert git_info(repo, ignore=repo / "results" / "run")["git_dirty"] is True


def test_an_ignored_folder_outside_the_repo_is_harmless(repo, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")

    info = git_info(repo, ignore=elsewhere)

    assert info["git_dirty"] is False
    assert len(info["git_commit"]) == 40
