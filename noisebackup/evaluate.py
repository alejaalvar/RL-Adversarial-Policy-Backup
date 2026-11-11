"""Play games between two policies and record the outcome of each one."""

import functools
from typing import List, NamedTuple, Optional, Sequence, Tuple

from aprl.envs.gym_compete import game_outcome
from aprl.envs.multi_agent import make_dummy_vec_multi_env, make_subproc_vec_multi_env
from aprl.envs.wrappers import make_env, simulate
from aprl.policies.loader import load_policy


class Episode(NamedTuple):
    """Outcome of one finished game."""

    env_index: int  # which parallel game it was played in
    winner: Optional[int]  # winning agent's index, or None for a tie
    length: int  # number of timesteps


def run_episodes(venv, policies, episodes: int) -> List[Episode]:
    """Play `episodes` games in total and return their outcomes.

    Each parallel game plays a fixed share of the total. Simply stopping once `episodes` games
    have finished would be biased: the games still running at that moment are dropped, and those
    are the long ones (often ties).
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    shares = split_evenly(episodes, venv.num_envs)
    results = []
    finished = [0] * venv.num_envs
    steps = [0] * venv.num_envs
    for _, _, dones, infos in simulate(venv, policies, record=False):
        for i, (done, info) in enumerate(zip(dones, infos)):
            steps[i] += 1
            if done:
                if finished[i] < shares[i]:
                    results.append(Episode(env_index=i, winner=game_outcome(info), length=steps[i]))
                    finished[i] += 1
                steps[i] = 0
        if len(results) == episodes:
            return results
    raise AssertionError("simulate() stopped before all episodes finished")


def split_evenly(total: int, parts: int) -> List[int]:
    """Split `total` into `parts` near-equal shares, the remainder going to the first parts."""
    base, remainder = divmod(total, parts)
    return [base + 1 if i < remainder else base for i in range(parts)]


def make_venv(env_name: str, num_envs: int, seed: int):
    """Create `num_envs` parallel copies of a two-player game (aprl VecMultiEnv).

    Parallel game i is seeded with seed + i, so the same seed gives the same set of games.
    More than one game runs in separate processes.
    """
    env_fns = [functools.partial(make_env, env_name, seed, i, None) for i in range(num_envs)]
    if num_envs > 1:
        return make_subproc_vec_multi_env(env_fns)
    return make_dummy_vec_multi_env(env_fns)


def load_policies(venv, env_name: str, specs: Sequence[Tuple[str, str]]) -> list:
    """Load one policy per agent; specs[i] = (policy_type, policy_path), e.g. ("zoo", "1")."""
    return [
        load_policy(policy_type, policy_path, venv, env_name, index)
        for index, (policy_type, policy_path) in enumerate(specs)
    ]
