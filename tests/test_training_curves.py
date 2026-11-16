"""Tests for noisebackup.training_curves: victim win rate over training time, per run."""

import json

import pandas as pd
import pytest

from noisebackup import training_curves


def training_games(timesteps_and_winners, env_name="multicomp/SumoAnts-v0", victim="defended"):
    """Rows as load_run returns them: training_episodes.csv plus the run's game and victim."""
    return pd.DataFrame(
        [
            {
                "env_name": env_name,
                "victim": victim,
                "timestep": t,
                "env_index": 0,
                "sigma": 0.5,
                "winner": w,
                "length": 100,
            }
            for t, w in timesteps_and_winners
        ]
    )


def test_summarize_bins_games_by_when_they_ended_with_win_rate_interval_and_tie_rate():
    episodes = training_games(
        [(100, "victim"), (500, "opponent")]  # bin (0, 500]: a game ending at 500 is in it
        + [(501, "victim"), (900, "victim"), (1000, "tie")]  # bin (500, 1000]
    )

    curve = training_curves.summarize(episodes, bin_steps=500)

    assert list(curve["timestep"]) == [500, 1000]  # end of each bin
    assert list(curve["games"]) == [2, 3]
    assert list(curve["victim_wins"]) == [1, 2]
    assert list(curve["win_rate"]) == pytest.approx([0.5, 2 / 3])
    # Wilson 95%: 1/2 -> [0.0945, 0.9055], 2/3 -> [0.2077, 0.9385]
    assert list(curve["win_ci_low"]) == pytest.approx([0.09453, 0.20765], abs=1e-5)
    assert list(curve["win_ci_high"]) == pytest.approx([0.90547, 0.93851], abs=1e-5)
    assert list(curve["tie_rate"]) == pytest.approx([0.0, 1 / 3])


def test_summarize_keeps_runs_apart_and_skips_bins_without_games():
    episodes = pd.concat(
        [
            training_games([(100, "victim"), (1200, "opponent")], victim="defended"),
            training_games([(300, "tie")], victim="finetuned_clean"),
            training_games([(400, "victim")], env_name="multicomp/SumoHumans-v0"),
        ]
    )

    curve = training_curves.summarize(episodes, bin_steps=500)

    assert list(zip(curve["env_name"], curve["victim"], curve["timestep"])) == [
        ("multicomp/SumoAnts-v0", "defended", 500),
        ("multicomp/SumoAnts-v0", "defended", 1500),  # (500, 1000] had no games: no row
        ("multicomp/SumoAnts-v0", "finetuned_clean", 500),
        ("multicomp/SumoHumans-v0", "defended", 500),
    ]


def write_run(folder, sigma_range, rows):
    folder.mkdir()
    config = {"run_name": folder.name, "env_name": "multicomp/SumoAnts-v0"}
    config["sigma_range"] = sigma_range
    (folder / "training_info.json").write_text(json.dumps({"config": config}))
    (folder / "training_episodes.csv").write_text(
        "timestep,env_index,sigma,winner,length\n" + "".join(rows)
    )


def test_load_run_labels_games_with_the_run_s_game_and_victim_type(tmp_path):
    write_run(tmp_path / "ants_defended", [0.0, 1.0], ["16,0,0.3,victim,90\n"])
    write_run(tmp_path / "ants_finetuned_clean", [0.0, 0.0], ["32,1,0.0,tie,120\n"])

    noisy = training_curves.load_run(tmp_path / "ants_defended")
    clean = training_curves.load_run(tmp_path / "ants_finetuned_clean")

    assert noisy.to_dict("records") == [
        {
            "timestep": 16,
            "env_index": 0,
            "sigma": 0.3,
            "winner": "victim",
            "length": 90,
            "env_name": "multicomp/SumoAnts-v0",
            "victim": "defended",
        }
    ]
    assert list(clean["victim"]) == ["finetuned_clean"]  # trained with sigma fixed at 0
