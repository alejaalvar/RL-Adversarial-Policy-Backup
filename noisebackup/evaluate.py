"""Play games between two policies and record the outcome of each one."""

import contextlib
import functools
from typing import List, NamedTuple, Optional, Sequence, Tuple
from unittest import mock

from aprl.envs import gym_compete as aprl_gym_compete
from aprl.envs.gym_compete import game_outcome
from aprl.envs.multi_agent import make_dummy_vec_multi_env, make_subproc_vec_multi_env
from aprl.envs.wrappers import make_env, simulate
from aprl.policies.loader import load_policy

from noisebackup import weights


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


def load_policies(
    venv, env_name: str, specs: Sequence[Tuple[str, str]], seed: Optional[int] = None
) -> list:
    """Load one policy per agent; specs[i] = (policy_type, policy_path), e.g. ("zoo", "1").

    Policy type "finetuned" is a fine-tuned victim: path to a Zoo-format weights file (see
    noisebackup.weights). It is built by aprl's own Zoo-agent code with those weights, so it is
    evaluated exactly like an original Zoo agent.

    Zoo agents pick each action at random from their policy (as in the paper). With `seed`,
    that randomness is repeatable: agent i's sampling is seeded with seed + i (different per
    agent, so victim and opponent don't draw the same numbers). Without it, every run differs.
    """
    policies = []
    for index, (policy_type, policy_path) in enumerate(specs):
        if seed is None:
            policies.append(_load_policy(venv, env_name, policy_type, policy_path, index))
            continue
        if policy_type not in ("zoo", "finetuned"):
            raise NotImplementedError(
                f"seeding is only implemented for zoo and finetuned policies, not {policy_type!r}"
            )
        with _seeded_zoo_graph(seed + index) as seeded:
            policies.append(_load_policy(venv, env_name, policy_type, policy_path, index))
        if seeded != [True]:
            raise RuntimeError(
                "aprl did not build the zoo policy the expected way; seed not applied"
            )
    return policies


def _load_policy(venv, env_name: str, policy_type: str, policy_path: str, index: int):
    if policy_type != "finetuned":
        return load_policy(policy_type, policy_path, venv, env_name, index)
    params = weights.load(policy_path)
    with _zoo_params(params) as used:
        policy = load_policy("zoo", "finetuned", venv, env_name, index)
    if used != [True]:
        raise RuntimeError("aprl did not load the zoo policy the expected way; weights not used")
    return policy


@contextlib.contextmanager
def _zoo_params(params):
    """While active, aprl's Zoo-agent loader uses `params` instead of a packaged Zoo file.

    load_zoo_agent(tag, ...) gets its weights from load_zoo_agent_params(tag, ...) and then
    builds and restores the policy; swapping only that lookup means everything else (network,
    observation normalization, session, our seeding hook) is exactly as for an original Zoo
    agent. Yields a list that gets one True per use, so the caller can check it was used.
    """
    used = []

    def fine_tuned_params(tag, env_name, index):
        used.append(True)
        return params

    with mock.patch.object(aprl_gym_compete, "load_zoo_agent_params", fine_tuned_params):
        yield used


@contextlib.contextmanager
def _seeded_zoo_graph(seed: int):
    """While active, the TF graph aprl builds a zoo policy in gets a fixed random seed.

    aprl's load_zoo_agent creates a fresh tf.Graph, then calls make_session(graph) *before*
    building the network, so that call is where we can seed the graph. Every random op built
    in it (including the action sampling) then has a fixed seed. Yields a list that gets one
    True per seeded graph, so the caller can check the hook was actually used.
    """
    seeded = []
    original = aprl_gym_compete.make_session

    def make_seeded_session(graph=None):
        graph.seed = seed
        seeded.append(True)
        return original(graph)

    with mock.patch.object(aprl_gym_compete, "make_session", make_seeded_session):
        yield seeded
