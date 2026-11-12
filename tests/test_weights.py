"""Tests for noisebackup.weights: fine-tuned victim weights in the Zoo parameter-file format."""

import hashlib
import os
import pickle

from aprl.envs.gym_compete import load_zoo_agent_params
import gym_compete
import numpy as np
import pytest

from noisebackup import weights

PARAMS = np.array([0.5, -1.25, 3.0, 1e-7], dtype=np.float32)


def test_saved_weights_load_back_exactly_in_the_zoo_file_format(tmp_path):
    path = tmp_path / "weights.pkl"

    weights.save(PARAMS, path)

    loaded = weights.load(path)
    np.testing.assert_array_equal(loaded, PARAMS)
    assert loaded.dtype == np.float32
    # Same format as gym_compete's agent_parameters-v1.pkl: a pickled flat float32 array.
    with open(path, "rb") as f:
        raw = pickle.load(f)
    assert isinstance(raw, np.ndarray) and raw.dtype == np.float32 and raw.ndim == 1


ZOO1_ANTS_FILE = os.path.join(
    os.path.dirname(gym_compete.__file__), "agent_zoo", "SumoAnts-v0", "agent_parameters-v1.pkl"
)


def test_reads_gym_compete_s_own_zoo1_file_exactly_as_aprl_does():
    zoo1 = load_zoo_agent_params("1", "multicomp/SumoAnts-v0", 0)

    np.testing.assert_array_equal(weights.load(ZOO1_ANTS_FILE), zoo1)


def test_saving_zoo1_s_parameters_reproduces_gym_compete_s_file_byte_for_byte(tmp_path):
    path = tmp_path / "weights.pkl"

    weights.save(load_zoo_agent_params("1", "multicomp/SumoAnts-v0", 0), path)

    # Compare fingerprints: on failure pytest prints two short hashes instead of trying to
    # diff two 1.2 MB byte strings (which takes practically forever).
    assert sha256(path) == sha256(ZOO1_ANTS_FILE)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.parametrize(
    "bad_params, problem",
    [
        (PARAMS.reshape(2, 2), "flat"),  # not a single vector
        (PARAMS.astype(np.float64), "float32"),  # wrong precision for the Zoo policy
        (np.array([0.5, np.nan], dtype=np.float32), "finite"),  # e.g. training diverged
        (np.array([0.5, np.inf], dtype=np.float32), "finite"),
    ],
)
def test_refuses_to_save_anything_but_a_flat_finite_float32_vector(tmp_path, bad_params, problem):
    path = tmp_path / "weights.pkl"

    with pytest.raises(ValueError, match=problem):
        weights.save(bad_params, path)

    assert not path.exists()


def test_never_overwrites_existing_weights(tmp_path):
    path = tmp_path / "weights.pkl"
    weights.save(PARAMS, path)

    with pytest.raises(FileExistsError):
        weights.save(PARAMS * 2, path)

    np.testing.assert_array_equal(weights.load(path), PARAMS)
