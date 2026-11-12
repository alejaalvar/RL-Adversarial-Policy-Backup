"""Fine-tuned victim weights, stored exactly like gym_compete's Zoo parameter files.

A Zoo agent's parameters (e.g. agent_zoo/SumoAnts-v0/agent_parameters-v1.pkl) are one pickled,
flat float32 array: every variable of the policy, in creation order. Saving fine-tuned weights
the same way lets them be loaded through aprl's own Zoo-agent code, so a fine-tuned victim is
built and evaluated exactly like the original.
"""

from pathlib import Path
import pickle
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def save(params: np.ndarray, path: PathLike) -> None:
    """Write a flat float32 parameter vector as a Zoo-format parameter file."""
    if params.ndim != 1:
        raise ValueError(f"weights must be one flat vector, got shape {params.shape}")
    if params.dtype != np.float32:
        raise ValueError(f"weights must be float32 like the Zoo files, got {params.dtype}")
    if not np.all(np.isfinite(params)):
        raise ValueError("weights must all be finite (NaN/inf: did training diverge?)")
    # "x": create only; never overwrite (committed weights are the canonical trained victims).
    with open(path, "xb") as f:
        pickle.dump(params, f, protocol=3)  # protocol 3, as the Zoo files use


def load(path: PathLike) -> np.ndarray:
    """Read a Zoo-format parameter file (ours or gym_compete's)."""
    with open(path, "rb") as f:
        return pickle.load(f)
