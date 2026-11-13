"""Tests for noisebackup.records.git_info: which code produced a result, and was it committed."""

import subprocess

import pytest

from noisebackup.records import git_info


@pytest.fixture
def git(tmp_path):
    """Run a git command in the throwaway repo."""

    def run(*args):
        subprocess.run(
            ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    return run


@pytest.fixture
def repo(tmp_path, git):
    """A throwaway git repo with one committed file and nothing else."""
    git("init")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-m", "initial")
    return tmp_path


def test_a_run_s_own_output_folder_does_not_count_as_uncommitted_changes(repo, git):
    # A resumed sweep appends to its own results file, which may already be committed.
    run = repo / "results" / "run"
    run.mkdir(parents=True)
    (run / "episodes.csv").write_text("sigma\n")
    git("add", "results")
    git("commit", "-m", "first sigmas")
    (run / "episodes.csv").write_text("sigma\n0.5\n")

    assert git_info(repo, ignore=run)["git_dirty"] is False
    assert git_info(repo)["git_dirty"] is True


def test_other_runs_new_outputs_in_models_and_results_do_not_count(repo):
    # Runs back to back without a commit in between: earlier runs' new output folders are
    # not code, and nothing reads them unless a config points at them.
    (repo / "models" / "earlier_run").mkdir(parents=True)
    (repo / "models" / "earlier_run" / "weights.pkl").write_bytes(b"weights")
    (repo / "results" / "earlier_sweep").mkdir(parents=True)
    (repo / "results" / "earlier_sweep" / "episodes.csv").write_text("sigma\n")

    assert git_info(repo, ignore=repo / "models" / "this_run")["git_dirty"] is False


def test_a_new_file_outside_models_and_results_counts(repo):
    (repo / "new_module.py").write_text("y = 1\n")  # e.g. code you forgot to git add

    assert git_info(repo)["git_dirty"] is True


def test_a_change_to_committed_weights_or_results_counts(repo, git):
    (repo / "models" / "victim").mkdir(parents=True)
    (repo / "models" / "victim" / "weights.pkl").write_bytes(b"trained")
    git("add", "models")
    git("commit", "-m", "weights")
    (repo / "models" / "victim" / "weights.pkl").write_bytes(b"edited")

    assert git_info(repo, ignore=repo / "results" / "sweep")["git_dirty"] is True


def test_changes_outside_the_ignored_folder_still_count(repo):
    (repo / "results" / "run").mkdir(parents=True)
    (repo / "code.py").write_text("x = 2\n")

    assert git_info(repo, ignore=repo / "results" / "run")["git_dirty"] is True


def test_an_ignored_folder_outside_the_repo_is_harmless(repo, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")

    info = git_info(repo, ignore=elsewhere)

    assert info["git_dirty"] is False
    assert len(info["git_commit"]) == 40
