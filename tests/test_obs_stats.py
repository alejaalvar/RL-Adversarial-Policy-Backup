"""Tests for noisebackup.obs_stats: per-input observation statistics."""

import numpy as np
import pytest

from noisebackup import obs_stats


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
