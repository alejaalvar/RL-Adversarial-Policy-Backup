"""Tests for noisebackup.obs_stats: per-input observation statistics."""

import numpy as np
import pytest

from fakes import ScriptedGamesVecMultiEnv, ZeroPolicy
from noisebackup import evaluate, obs_stats


def test_compute_gives_per_input_mean_and_std():
    # Input 0 takes values 1, 3, 5 -> mean 3, std sqrt(8/3).
    # Input 1 is constant 10 -> mean 10, std 0.
    observations = [[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]]

    stats = obs_stats.compute(observations)

    np.testing.assert_allclose(stats.mean, [3.0, 10.0])
    np.testing.assert_allclose(stats.std, [1.632993161855452, 0.0])
    assert stats.n_samples == 3


@pytest.mark.parametrize(
    "bad_observations",
    [
        np.zeros((0, 3)),  # no samples: mean/std would silently be NaN
        [1.0, 2.0, 3.0],  # one flat observation, not a (samples, inputs) table
    ],
)
def test_compute_rejects_input_that_is_not_a_nonempty_2d_table(bad_observations):
    with pytest.raises(ValueError):
        obs_stats.compute(bad_observations)


def test_saved_stats_load_back_exactly_with_their_metadata(tmp_path):
    stats = obs_stats.compute([[0.1, -2.5, 7.0], [0.3, 4.25, 7.0]])
    metadata = {"env_name": "multicomp/SumoAnts-v0", "victim": "zoo:1", "episodes": 100}
    path = tmp_path / "SumoAnts.json"

    obs_stats.save(stats, path, metadata)
    loaded, loaded_metadata = obs_stats.load(path)

    np.testing.assert_array_equal(loaded.mean, stats.mean)
    np.testing.assert_array_equal(loaded.std, stats.std)
    assert loaded.n_samples == 2
    assert loaded_metadata == metadata


def test_collect_returns_the_victims_observation_at_every_step_of_its_share_of_games():
    # Victim (agent 0) always sees [1, 2]; the opponent sees [9]. Game 0's share is one 3-step
    # game and game 1's is one 5-step game, so 3 + 5 = 8 observations.
    env = ScriptedGamesVecMultiEnv([[(3, 0)], [(5, None)]], base_obs=([1.0, 2.0], [9.0]))

    observations = obs_stats.collect(env, [ZeroPolicy(), ZeroPolicy()], agent_index=0, episodes=2)

    np.testing.assert_array_equal(observations, np.tile([1.0, 2.0], (8, 1)))


@pytest.mark.slow
def test_collect_from_real_sumo_ants_games_gives_137_victim_inputs_per_step():
    env_name = "multicomp/SumoAnts-v0"
    venv = evaluate.make_venv(env_name, num_envs=2, seed=0)
    try:
        policies = evaluate.load_policies(venv, env_name, [("zoo", "1"), ("zoo", "1")])
        observations = obs_stats.collect(venv, policies, agent_index=0, episodes=2)
    finally:
        venv.close()

    assert observations.ndim == 2
    assert observations.shape[1] == 137
    assert observations.shape[0] > 2  # at least one step per game
