"""Per-input statistics of the victim's observations during clean (noise-free) play.

The noise attack is scaled per input: noise on input i is sigma * std[i] * N(0, 1). Raw input
spreads differ by about four orders of magnitude, so a single raw sigma would swamp some inputs
and barely touch others. These stats are measured once per environment and reused for both the
original and the defended victim, so a given sigma means the same noise for both.
"""

import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple, Union

import numpy as np

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
