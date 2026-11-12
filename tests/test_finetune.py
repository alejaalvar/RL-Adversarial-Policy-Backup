"""Tests for noisebackup.finetune: noise-augmented fine-tuning of a Zoo victim (real games)."""

import csv
import json

from aprl.envs.gym_compete import load_zoo_agent_params
import numpy as np
import pytest

from noisebackup import evaluate, finetune, weights

# A tiny but complete training setup on real Sumo Ants games.
TINY = {
    "run_name": "tiny_test",
    "env_name": "multicomp/SumoAnts-v0",
    "victim": {"type": "zoo", "path": "1"},
    "opponent": {"type": "zoo", "path": "1"},
    "victim_index": 0,
    "obs_stats": "results/obs_stats/SumoAnts.json",
    "sigma_range": [0.0, 1.0],
    "total_timesteps": 2048,
    "num_envs": 4,
    "seed": 1000,
    "noise_seed": 1001,
    "ppo": {
        "batch_size": 256,
        "learning_rate": 3e-4,
        "ent_coef": 0.0,
        "nminibatches": 4,
        "noptepochs": 4,
        "cliprange": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
    },
}


def first_training_observation(tmp_path, sigma_range):
    config = dict(TINY, sigma_range=sigma_range)
    venv = finetune.make_training_env(config, log_path=tmp_path / "log.csv")
    try:
        return venv.reset()
    finally:
        venv.close()


@pytest.mark.slow
def test_the_agent_being_trained_sees_the_victim_s_observations_with_the_noise_on_them(tmp_path):
    clean_games = evaluate.make_venv(TINY["env_name"], TINY["num_envs"], TINY["seed"])
    try:
        victim_clean, _ = clean_games.reset()
    finally:
        clean_games.close()

    # sigma fixed at 0: exactly the victim's (seat 0's) observations, so we train the victim.
    np.testing.assert_array_equal(first_training_observation(tmp_path, [0.0, 0.0]), victim_clean)
    # sigma fixed at 1: the same observations with noise on them.
    noisy = first_training_observation(tmp_path, [1.0, 1.0])
    assert noisy.shape == victim_clean.shape
    assert np.all(noisy != victim_clean)


@pytest.mark.slow
def test_a_tiny_training_run_writes_weights_a_training_record_and_a_game_log(tmp_path):
    out = tmp_path / "model"

    finetune.run(TINY, out)

    params = weights.load(out / "weights.pkl")  # Zoo format, same size as Zoo1's
    zoo1 = load_zoo_agent_params("1", TINY["env_name"], 0)
    assert params.shape == zoo1.shape
    assert not np.array_equal(params, zoo1)  # training really changed the weights

    info = json.loads((out / "training_info.json").read_text())
    assert info["config"] == TINY
    assert info["timesteps"] == 2048
    assert len(info["git_commit"]) == 40 and isinstance(info["git_dirty"], bool)
    assert info["started_at"] and info["finished_at"] and info["steps_per_second"] > 0

    with open(out / "training_episodes.csv", newline="") as f:
        games = list(csv.DictReader(f))
    assert games, "4 parallel games x 512 steps should finish some Sumo Ants games"
    assert all(0.0 <= float(g["sigma"]) <= 1.0 for g in games)
    assert {g["winner"] for g in games} <= {"victim", "opponent", "tie"}
    assert all(0 < int(g["length"]) and 0 < int(g["timestep"]) <= 2048 for g in games)


@pytest.mark.slow
def test_with_learning_rate_zero_the_saved_weights_are_exactly_zoo1(tmp_path):
    # Training runs end to end but can't change anything, so this checks the wiring alone:
    # we start from Zoo1, and the right variables are extracted and saved in the right order.
    config = dict(TINY, ppo=dict(TINY["ppo"], learning_rate=0.0))

    finetune.run(config, tmp_path / "model")

    np.testing.assert_array_equal(
        weights.load(tmp_path / "model" / "weights.pkl"),
        load_zoo_agent_params("1", TINY["env_name"], 0),
    )


def test_refuses_to_train_into_an_existing_model_folder(tmp_path):
    existing = tmp_path / "model"
    existing.mkdir()
    (existing / "weights.pkl").write_text("committed weights")

    with pytest.raises(FileExistsError):
        finetune.run(TINY, existing)

    assert (existing / "weights.pkl").read_text() == "committed weights"
    assert [f.name for f in existing.iterdir()] == ["weights.pkl"]  # refused before writing
