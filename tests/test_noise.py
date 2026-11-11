"""Tests for noisebackup.noise: Gaussian noise on the victim's observations only."""

import numpy as np
import pytest

from fakes import ConstantObsVecMultiEnv
from noisebackup.noise import ObservationNoise

VICTIM_OBS = [1.0, -2.0, 3.0]
OPPONENT_OBS = [5.0, 6.0]


def make_env(num_envs=1):
    return ConstantObsVecMultiEnv([VICTIM_OBS, OPPONENT_OBS], num_envs=num_envs)


def test_noise_changes_victim_observations_but_not_opponents():
    env = ObservationNoise(make_env(), agent_index=0, input_std=[1.0, 1.0, 1.0], sigma=0.5, seed=0)

    victim_obs, opponent_obs = env.reset()

    assert np.all(victim_obs != np.array([VICTIM_OBS]))
    np.testing.assert_array_equal(opponent_obs, [OPPONENT_OBS])


def test_noise_has_mean_zero_and_std_sigma_times_input_std():
    # sigma 0.5 with input stds [2.0, 0.1, 0.0] -> noise stds [1.0, 0.05, 0.0].
    env = ObservationNoise(
        make_env(num_envs=20000), agent_index=0, input_std=[2.0, 0.1, 0.0], sigma=0.5, seed=0
    )

    victim_obs, _ = env.reset()
    noise = victim_obs - np.array(VICTIM_OBS)

    np.testing.assert_allclose(noise.std(axis=0), [1.0, 0.05, 0.0], rtol=0.05)
    np.testing.assert_allclose(noise.mean(axis=0), [0.0, 0.0, 0.0], atol=0.03)


def test_inputs_with_zero_spread_get_no_noise():
    env = ObservationNoise(
        make_env(num_envs=100), agent_index=0, input_std=[1.0, 0.0, 1.0], sigma=2.0, seed=0
    )

    victim_obs, _ = env.reset()

    np.testing.assert_array_equal(victim_obs[:, 1], VICTIM_OBS[1])


def test_sigma_zero_leaves_victim_observations_exactly_unchanged():
    env = ObservationNoise(
        make_env(num_envs=10), agent_index=0, input_std=[2.0, 0.1, 5.0], sigma=0.0, seed=0
    )

    victim_obs, _ = env.reset()
    np.testing.assert_array_equal(victim_obs, np.tile(VICTIM_OBS, (10, 1)))

    env.step_async(None)
    (victim_obs, _), _, _, _ = env.step_wait()
    np.testing.assert_array_equal(victim_obs, np.tile(VICTIM_OBS, (10, 1)))


def test_each_step_gets_fresh_noise_on_victim_only():
    env = ObservationNoise(make_env(), agent_index=0, input_std=[1.0, 1.0, 1.0], sigma=0.5, seed=0)
    env.reset()

    env.step_async(None)
    (first_victim, first_opponent), _, _, _ = env.step_wait()
    env.step_async(None)
    (second_victim, _), _, _, _ = env.step_wait()

    assert np.all(first_victim != np.array([VICTIM_OBS]))
    assert np.all(first_victim != second_victim)
    np.testing.assert_array_equal(first_opponent, [OPPONENT_OBS])


def noisy_victim_obs(seed):
    env = ObservationNoise(
        make_env(num_envs=4), agent_index=0, input_std=[1.0, 1.0, 1.0], sigma=0.5, seed=seed
    )
    victim_obs, _ = env.reset()
    return victim_obs


def test_same_seed_gives_same_noise_and_different_seed_gives_different_noise():
    np.testing.assert_array_equal(noisy_victim_obs(seed=7), noisy_victim_obs(seed=7))
    assert np.all(noisy_victim_obs(seed=7) != noisy_victim_obs(seed=8))


def test_rejects_input_std_whose_length_does_not_match_victim_observations():
    # e.g. Sumo Ants stats (137 inputs) used with Sumo Humans (395 inputs)
    with pytest.raises(ValueError, match="input_std"):
        ObservationNoise(make_env(), agent_index=0, input_std=[1.0, 1.0], sigma=0.5, seed=0)


def test_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma"):
        ObservationNoise(make_env(), agent_index=0, input_std=[1.0, 1.0, 1.0], sigma=-0.1, seed=0)
