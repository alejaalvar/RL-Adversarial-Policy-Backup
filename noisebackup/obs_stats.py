"""Per-input statistics of the victim's observations during clean (noise-free) play.

The noise attack is scaled per input: noise on input i is sigma * std[i] * N(0, 1). Raw input
spreads differ by about four orders of magnitude, so a single raw sigma would swamp some inputs
and barely touch others. These stats are measured once per environment and reused for both the
original and the defended victim, so a given sigma means the same noise for both.
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple, Union

from aprl.envs.wrappers import simulate
import numpy as np

from noisebackup import evaluate
from noisebackup.evaluate import split_evenly
from noisebackup.records import git_info

PathLike = Union[str, Path]


class ObsStats(NamedTuple):
    """Mean and standard deviation of each observation input."""

    mean: np.ndarray
    std: np.ndarray
    n_samples: int


def compute(observations) -> ObsStats:
    """Compute per-input mean and (population) std from an (n_samples, n_inputs) array."""
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError(
            "Expected a non-empty (n_samples, n_inputs) array of observations, "
            f"got shape {observations.shape}"
        )
    return ObsStats(
        mean=observations.mean(axis=0),
        std=observations.std(axis=0),
        n_samples=observations.shape[0],
    )


def collect(venv, policies, agent_index: int, episodes: int) -> np.ndarray:
    """Play clean games and return every observation agent `agent_index` sees.

    Returns an (n_steps, n_inputs) array with one row per timestep of the counted games. Like
    evaluate.run_episodes, each parallel game contributes a fixed share of the episodes, so long
    games aren't under-represented.
    """
    shares = split_evenly(episodes, venv.num_envs)
    finished = [0] * venv.num_envs
    rows = []
    for observations, _, dones, _ in simulate(venv, policies, record=False):
        for i, done in enumerate(dones):
            if finished[i] < shares[i]:
                rows.append(observations[agent_index][i])
                if done:
                    finished[i] += 1
        if sum(finished) == episodes:
            return np.array(rows, dtype=np.float64)
    raise AssertionError("simulate() stopped before all episodes finished")


def save(stats: ObsStats, path: PathLike, metadata: Dict[str, Any]) -> None:
    """Write stats to a JSON file, with metadata recording how they were measured."""
    record = {
        "metadata": metadata,
        "n_samples": stats.n_samples,
        "mean": stats.mean.tolist(),
        "std": stats.std.tolist(),
    }
    Path(path).write_text(json.dumps(record, indent=2))


def load(path: PathLike) -> Tuple[ObsStats, Dict[str, Any]]:
    """Read stats written by `save`. Returns (stats, metadata)."""
    record = json.loads(Path(path).read_text())
    stats = ObsStats(
        mean=np.array(record["mean"], dtype=np.float64),
        std=np.array(record["std"], dtype=np.float64),
        n_samples=record["n_samples"],
    )
    return stats, record["metadata"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure per-input stats of the victim's observations in clean games."
    )
    parser.add_argument("--env", required=True, help="e.g. multicomp/SumoAnts-v0")
    parser.add_argument("--out", required=True, help="e.g. results/obs_stats/SumoAnts.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=8)
    # A different seed from the sweeps (seed 0), so the stats don't come from the same games
    # that are later evaluated.
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--victim-path", default="1", help="zoo version of the victim")
    parser.add_argument("--opponent-path", default="1", help="zoo version of the opponent")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(
            f"{out_path} already exists. Stats are frozen once measured, because sweeps (and "
            "the defended victim) must all use the same ones; delete it deliberately to redo."
        )

    victim_index = 0
    specs = [("zoo", args.victim_path), ("zoo", args.opponent_path)]
    venv = evaluate.make_venv(args.env, args.num_envs, args.seed)
    try:
        policies = evaluate.load_policies(venv, args.env, specs)
        observations = collect(venv, policies, victim_index, args.episodes)
    finally:
        venv.close()

    stats = compute(observations)
    metadata = {
        "env_name": args.env,
        "victim": f"zoo:{args.victim_path}",
        "opponent": f"zoo:{args.opponent_path}",
        "victim_index": victim_index,
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **git_info(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(stats, out_path, metadata)
    print(
        f"Wrote {out_path}: {stats.n_samples} observations x {len(stats.std)} inputs; "
        f"std min {stats.std.min():.3g}, median {np.median(stats.std):.3g}, "
        f"max {stats.std.max():.3g}, zero-spread inputs {int((stats.std == 0).sum())}"
    )


if __name__ == "__main__":
    main()
