"""Tests for noisebackup.sweep: running every sigma and recording one row per game."""

import csv
import json

import numpy as np
import pytest

from noisebackup import obs_stats, sweep
from noisebackup.evaluate import Episode

CONFIG = {
    "run_name": "test_run",
    "env_name": "multicomp/SumoAnts-v0",
    "victim": {"label": "zoo1", "type": "zoo", "path": "1"},
    "opponent": {"type": "zoo", "path": "1"},
    "victim_index": 0,
    "obs_stats": "results/obs_stats/SumoAnts.json",
    "sigmas": [0.0, 0.5],
    "episodes": 3,
    "num_envs": 2,
    "seed": 0,
    "noise_seed": 1,
}

# Every sigma point: victim (agent 0) wins, opponent (agent 1) wins, then a tie.
FAKE_EPISODES = [
    Episode(env_index=0, winner=0, length=10),
    Episode(env_index=1, winner=1, length=20),
    Episode(env_index=0, winner=None, length=500),
]


class FakeEvaluator:
    """Stands in for real games; records which sigmas it was asked to play."""

    def __init__(self):
        self.sigmas_played = []

    def __call__(self, config, sigma):
        self.sigmas_played.append(sigma)
        return FAKE_EPISODES


def read_rows(out_dir):
    with open(out_dir / "episodes.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_every_sigma_is_played_and_each_game_becomes_one_row(tmp_path):
    evaluator = FakeEvaluator()

    sweep.run_sweep(CONFIG, evaluator, tmp_path)

    assert evaluator.sigmas_played == [0.0, 0.5]
    rows = read_rows(tmp_path)
    assert [(r["sigma"], r["episode"], r["winner"], r["length"]) for r in rows] == [
        ("0.0", "0", "victim", "10"),
        ("0.0", "1", "opponent", "20"),
        ("0.0", "2", "tie", "500"),
        ("0.5", "0", "victim", "10"),
        ("0.5", "1", "opponent", "20"),
        ("0.5", "2", "tie", "500"),
    ]
    assert {r["env_name"] for r in rows} == {"multicomp/SumoAnts-v0"}
    assert {r["victim"] for r in rows} == {"zoo1"}
    assert [r["env_index"] for r in rows[:3]] == ["0", "1", "0"]


def test_config_and_a_record_of_the_run_are_saved_next_to_the_results(tmp_path):
    sweep.run_sweep(CONFIG, FakeEvaluator(), tmp_path)

    assert json.loads((tmp_path / "config.json").read_text()) == CONFIG
    run_info = json.loads((tmp_path / "run_info.json").read_text())
    (invocation,) = run_info["invocations"]
    assert invocation["sigmas_played"] == [0.0, 0.5]
    assert invocation["started_at"]
    assert len(invocation["git_commit"]) == 40  # full commit hash of this repo
    assert isinstance(invocation["git_dirty"], bool)


class CrashingEvaluator(FakeEvaluator):
    """Like FakeEvaluator, but crashes (e.g. power loss) when asked to play `crash_at`."""

    def __init__(self, crash_at):
        super().__init__()
        self.crash_at = crash_at

    def __call__(self, config, sigma):
        if sigma == self.crash_at:
            raise KeyboardInterrupt("simulated crash")
        return super().__call__(config, sigma)


def test_rerunning_after_a_crash_plays_only_the_unfinished_sigmas(tmp_path):
    config = dict(CONFIG, sigmas=[0.0, 0.5, 1.0])
    with pytest.raises(KeyboardInterrupt):
        sweep.run_sweep(config, CrashingEvaluator(crash_at=0.5), tmp_path)

    evaluator = FakeEvaluator()
    sweep.run_sweep(config, evaluator, tmp_path)

    assert evaluator.sigmas_played == [0.5, 1.0]
    sigmas_in_csv = [r["sigma"] for r in read_rows(tmp_path)]
    assert sigmas_in_csv == ["0.0"] * 3 + ["0.5"] * 3 + ["1.0"] * 3
    invocations = json.loads((tmp_path / "run_info.json").read_text())["invocations"]
    assert [i["sigmas_played"] for i in invocations] == [[0.0], [0.5, 1.0]]


def test_refuses_to_add_results_from_a_different_config_and_leaves_files_untouched(tmp_path):
    sweep.run_sweep(CONFIG, FakeEvaluator(), tmp_path)
    before = {p.name: p.read_text() for p in tmp_path.iterdir()}

    evaluator = FakeEvaluator()
    with pytest.raises(ValueError, match="config"):
        sweep.run_sweep(dict(CONFIG, episodes=5), evaluator, tmp_path)

    assert evaluator.sigmas_played == []
    assert {p.name: p.read_text() for p in tmp_path.iterdir()} == before


@pytest.mark.slow
def test_tiny_real_sumo_ants_sweep_records_every_game(tmp_path):
    stats_path = tmp_path / "SumoAnts.json"
    stats = obs_stats.compute(np.random.RandomState(0).normal(size=(10, 137)))
    obs_stats.save(stats, stats_path, {"note": "placeholder stats for this test"})
    config = dict(CONFIG, obs_stats=str(stats_path), episodes=4, num_envs=2)

    sweep.run_sweep(config, sweep.play_sigma, tmp_path / "out")

    rows = read_rows(tmp_path / "out")
    assert [r["sigma"] for r in rows] == ["0.0"] * 4 + ["0.5"] * 4
    assert {r["winner"] for r in rows} <= {"victim", "opponent", "tie"}
    assert all(int(r["length"]) > 0 for r in rows)
