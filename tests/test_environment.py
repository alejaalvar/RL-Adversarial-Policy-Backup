"""Smoke tests: the Docker test loop works and the original code is importable."""

import sys


def test_runs_on_python_37():
    assert sys.version_info[:2] == (3, 7)


def test_original_repo_and_zoo_envs_are_importable():
    import aprl.envs.wrappers  # noqa: F401
    import gym_compete  # noqa: F401


def test_backup_package_is_importable():
    import noisebackup

    assert noisebackup.__doc__
