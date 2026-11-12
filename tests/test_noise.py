"""Tests for noisebackup.noise: Gaussian noise on the victim's observations only."""

import numpy as np
import pytest

from fakes import ConstantObsVecMultiEnv, ScriptedGamesVecMultiEnv
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


# --- Random sigma per game (for noise-augmented training) ---------------------------------

MANY_INPUTS = 5000  # enough inputs to measure each game's noise std from a single observation


def make_wide_env(num_envs):
    return ConstantObsVecMultiEnv([np.zeros(MANY_INPUTS), OPPONENT_OBS], num_envs=num_envs)


def test_with_a_sigma_range_each_parallel_game_gets_its_own_sigma_within_the_range():
    env = ObservationNoise(
        make_wide_env(num_envs=4),
        agent_index=0,
        input_std=np.ones(MANY_INPUTS),
        sigma=(0.2, 1.0),
        seed=0,
    )

    victim_obs, _ = env.reset()

    assert env.sigmas.shape == (4,)
    assert np.all((env.sigmas >= 0.2) & (env.sigmas <= 1.0))
    assert len(set(env.sigmas)) == 4  # drawn separately for each game
    # Each game's noise really has that game's sigma (observations are all zero underneath).
    np.testing.assert_allclose(victim_obs.std(axis=1), env.sigmas, rtol=0.05)


def test_a_game_that_ends_gets_a_new_sigma_and_the_others_keep_theirs():
    # Game 0 ends after 2 steps; game 1 is still running after 2 steps (it lasts 5).
    games = ScriptedGamesVecMultiEnv(
        [[(2, 0)], [(5, 1)]], base_obs=(np.zeros(MANY_INPUTS), OPPONENT_OBS)
    )
    env = ObservationNoise(
        games, agent_index=0, input_std=np.ones(MANY_INPUTS), sigma=(0.0, 1.0), seed=0
    )
    env.reset()
    first_sigmas = env.sigmas.copy()

    env.step_async(None)
    env.step_wait()
    np.testing.assert_array_equal(env.sigmas, first_sigmas)  # mid-game: no change
    env.step_async(None)
    (victim_obs, _), _, dones, _ = env.step_wait()

    assert list(dones) == [True, False]
    assert env.sigmas[0] != first_sigmas[0]
    assert env.sigmas[1] == first_sigmas[1]
    # The observation returned when a game ends is the first one of the next game (VecEnvs
    # reset automatically), so it already carries the new game's noise level.
    np.testing.assert_allclose(victim_obs.std(axis=1), env.sigmas, rtol=0.05)


def sigma_sequence(seed):
    """Sigmas of the first few games in each of two parallel slots (games of length 1)."""
    env = ObservationNoise(
        ScriptedGamesVecMultiEnv([[(1, 0)], [(1, 1)]], base_obs=(VICTIM_OBS, OPPONENT_OBS)),
        agent_index=0,
        input_std=[1.0, 1.0, 1.0],
        sigma=(0.0, 1.0),
        seed=seed,
    )
    env.reset()
    sigmas = [env.sigmas.copy()]
    for _ in range(3):
        env.step_async(None)
        env.step_wait()
        sigmas.append(env.sigmas.copy())
    return np.array(sigmas)


def test_same_seed_gives_same_sigmas_and_different_seed_gives_different_sigmas():
    np.testing.assert_array_equal(sigma_sequence(seed=7), sigma_sequence(seed=7))
    assert np.all(sigma_sequence(seed=7) != sigma_sequence(seed=8))


@pytest.mark.parametrize("sigma", [(-0.1, 1.0), (1.0, 0.5), (0.1, 0.5, 1.0)])
def test_rejects_a_sigma_range_that_is_not_low_high_with_0_le_low_le_high(sigma):
    with pytest.raises(ValueError, match="sigma"):
        ObservationNoise(make_env(), agent_index=0, input_std=[1.0, 1.0, 1.0], sigma=sigma, seed=0)
