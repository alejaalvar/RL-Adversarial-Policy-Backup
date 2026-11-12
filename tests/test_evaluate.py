"""Tests for noisebackup.evaluate: playing a fixed number of games and recording outcomes."""

import collections

from aprl.envs.gym_compete import load_zoo_agent_params
import numpy as np
import pytest

from fakes import ScriptedGamesVecMultiEnv, ZeroPolicy
from noisebackup import evaluate, weights
from noisebackup.noise import ObservationNoise

POLICIES = [ZeroPolicy(), ZeroPolicy()]


def test_single_game_plays_requested_episodes_and_records_winner_and_length():
    # One parallel game: agent 0 wins in 3 steps, a tie after 5, agent 1 wins in 2.
    env = ScriptedGamesVecMultiEnv([[(3, 0), (5, None), (2, 1)]])

    episodes = evaluate.run_episodes(env, POLICIES, episodes=3)

    assert [(e.winner, e.length) for e in episodes] == [(0, 3), (None, 5), (1, 2)]


def test_each_parallel_game_contributes_its_fixed_share_even_if_slower():
    # Game 0: quick 2-step wins for agent 0. Game 1: slow 10-step ties.
    # Stopping at "the first 4 finished" would take all 4 from game 0 and record no ties.
    env = ScriptedGamesVecMultiEnv([[(2, 0)], [(10, None)]])

    episodes = evaluate.run_episodes(env, POLICIES, episodes=4)

    assert sorted((e.env_index, e.winner, e.length) for e in episodes) == [
        (0, 0, 2),
        (0, 0, 2),
        (1, None, 10),
        (1, None, 10),
    ]


@pytest.mark.parametrize(
    "episodes, expected_per_game",
    [
        (5, {0: 3, 1: 2}),  # remainder goes to the first games
        (1, {0: 1}),  # fewer episodes than parallel games
    ],
)
def test_uneven_totals_are_split_as_evenly_as_possible(episodes, expected_per_game):
    env = ScriptedGamesVecMultiEnv([[(2, 0)], [(3, 1)]])

    results = evaluate.run_episodes(env, POLICIES, episodes=episodes)

    assert dict(collections.Counter(e.env_index for e in results)) == expected_per_game


@pytest.mark.parametrize("episodes", [0, -1])
def test_rejects_non_positive_episode_count(episodes):
    env = ScriptedGamesVecMultiEnv([[(2, 0)]])

    with pytest.raises(ValueError, match="episodes"):
        evaluate.run_episodes(env, POLICIES, episodes=episodes)


@pytest.mark.slow
def test_real_sumo_ants_match_with_noise_plays_fixed_shares():
    env_name = "multicomp/SumoAnts-v0"
    venv = evaluate.make_venv(env_name, num_envs=2, seed=0)
    venv = ObservationNoise(venv, agent_index=0, input_std=np.ones(137), sigma=0.1, seed=0)
    policies = evaluate.load_policies(venv, env_name, [("zoo", "1"), ("zoo", "1")])
    try:
        results = evaluate.run_episodes(venv, policies, episodes=4)
    finally:
        venv.close()

    assert dict(collections.Counter(e.env_index for e in results)) == {0: 2, 1: 2}
    assert all(e.winner in (0, 1, None) for e in results)
    assert all(e.length > 0 for e in results)


def play_real_ants_games(policy_seed, specs=(("zoo", "1"), ("zoo", "1"))):
    """4 real Sumo Ants games (fixed game seed), agents' action sampling seeded."""
    env_name = "multicomp/SumoAnts-v0"
    venv = evaluate.make_venv(env_name, num_envs=2, seed=0)
    try:
        policies = evaluate.load_policies(venv, env_name, list(specs), seed=policy_seed)
        return evaluate.run_episodes(venv, policies, episodes=4)
    finally:
        venv.close()


@pytest.mark.slow
def test_same_seed_replays_the_same_games_even_though_actions_are_sampled():
    # Zoo agents sample each action at random (as in the paper); the seed makes it repeatable.
    first = play_real_ants_games(policy_seed=0)
    again = play_real_ants_games(policy_seed=0)
    other = play_real_ants_games(policy_seed=1)

    assert first == again
    assert first != other  # the seed really controls the sampling


@pytest.mark.slow
@pytest.mark.parametrize("zoo_version", ["1", "2"])
def test_a_finetuned_victim_file_is_played_exactly_like_the_zoo_agent_it_holds(
    tmp_path, zoo_version
):
    # A "finetuned" victim is built by aprl's own Zoo-agent code with its weights read from our
    # file, so Zoo weights saved to a file must replay exactly that Zoo agent's games. Version 2
    # shows the file is really used (a loader that ignored it would play Zoo1's games).
    path = tmp_path / "weights.pkl"
    weights.save(load_zoo_agent_params(zoo_version, "multicomp/SumoAnts-v0", 0), path)

    from_file = play_real_ants_games(0, specs=[("finetuned", str(path)), ("zoo", "1")])
    original = play_real_ants_games(0, specs=[("zoo", zoo_version), ("zoo", "1")])

    assert from_file == original
