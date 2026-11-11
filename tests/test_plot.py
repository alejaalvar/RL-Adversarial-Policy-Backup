"""Tests for noisebackup.plot.summarize: per-sigma win rates with Wilson intervals.

Expected interval bounds are standard Wilson 95% values (e.g. 8/10 -> [0.490, 0.943]).
"""

import pandas as pd
import pytest

from noisebackup import plot


def games(env_name, victim, sigma, victim_wins, opponent_wins, ties):
    """Rows as they appear in episodes.csv, for one sigma point."""
    winners = ["victim"] * victim_wins + ["opponent"] * opponent_wins + ["tie"] * ties
    return [
        {"env_name": env_name, "victim": victim, "sigma": sigma, "winner": w, "length": 100}
        for w in winners
    ]


def test_summarize_gives_counts_win_rate_with_wilson_interval_and_tie_rate_per_sigma():
    results = pd.DataFrame(
        games("multicomp/SumoAnts-v0", "zoo1", 0.0, victim_wins=8, opponent_wins=1, ties=1)
        + games("multicomp/SumoAnts-v0", "zoo1", 0.5, victim_wins=2, opponent_wins=1, ties=1)
    )

    summary = plot.summarize(results)

    assert list(summary["sigma"]) == [0.0, 0.5]
    assert list(summary["games"]) == [10, 4]
    assert list(summary["victim_wins"]) == [8, 2]
    assert list(summary["opponent_wins"]) == [1, 1]
    assert list(summary["ties"]) == [1, 1]
    assert list(summary["win_rate"]) == pytest.approx([0.8, 0.5])
    assert list(summary["win_ci_low"]) == pytest.approx([0.49016, 0.15004], abs=1e-5)
    assert list(summary["win_ci_high"]) == pytest.approx([0.94332, 0.84996], abs=1e-5)
    assert list(summary["tie_rate"]) == pytest.approx([0.1, 0.25])


def test_summarize_keeps_environments_and_victims_apart_at_the_same_sigma():
    results = pd.DataFrame(
        games("multicomp/SumoAnts-v0", "zoo1", 0.0, victim_wins=8, opponent_wins=1, ties=1)
        + games("multicomp/SumoHumans-v0", "zoo1", 0.0, victim_wins=0, opponent_wins=10, ties=0)
        + games("multicomp/SumoAnts-v0", "defended", 0.0, victim_wins=10, opponent_wins=0, ties=0)
    )

    summary = plot.summarize(results)

    assert list(zip(summary["env_name"], summary["victim"], summary["win_rate"])) == [
        ("multicomp/SumoAnts-v0", "defended", 1.0),
        ("multicomp/SumoAnts-v0", "zoo1", 0.8),
        ("multicomp/SumoHumans-v0", "zoo1", 0.0),
    ]
    # Wilson bounds stay inside [0, 1] at the extremes: 10/10 -> [0.7225, 1], 0/10 -> [0, 0.2775]
    assert list(summary["win_ci_low"]) == pytest.approx([0.72247, 0.49016, 0.0], abs=1e-5)
    assert list(summary["win_ci_high"]) == pytest.approx([1.0, 0.94332, 0.27753], abs=1e-5)
