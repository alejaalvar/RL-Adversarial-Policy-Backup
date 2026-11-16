"""Victim win rate over training time, from the fine-tuning runs' per-game training logs.

Noise-trained runs drew a new sigma for every training game (uniform over the config's
`sigma_range`), so their curve averages over noise levels and is not comparable to one sweep
point; the sweep (plot.py) is the measurement at fixed sigma. Threat model: *direct
observation* (see noise.py).
"""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from noisebackup.plot import ENV_TITLES, INK, SECONDARY_INK, VICTIM_STYLES, style_axes


def load_run(model_dir) -> pd.DataFrame:
    """A run's training_episodes.csv, with columns naming its game and victim type."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "training_info.json").read_text())["config"]
    episodes = pd.read_csv(model_dir / "training_episodes.csv")
    episodes["env_name"] = config["env_name"]
    clean = [float(s) for s in config["sigma_range"]] == [0.0, 0.0]
    episodes["victim"] = "finetuned_clean" if clean else "defended"
    return episodes


def summarize(episodes: pd.DataFrame, bin_steps: int) -> pd.DataFrame:
    """One row per (environment, victim, training-time bin) that has games: game count,
    victim win rate with Wilson 95% interval, and tie rate.

    A game belongs to the bin in which it ended: (0, bin_steps], (bin_steps, 2*bin_steps], ...
    `timestep` is the end of the bin. Sorted by environment, victim, then timestep.
    """
    bin_end = np.ceil(episodes["timestep"] / bin_steps).astype(int) * bin_steps
    rows = []
    for (env_name, victim, timestep), group in episodes.groupby(
        ["env_name", "victim", bin_end.rename("bin_end")]
    ):
        games = len(group)
        wins = int((group["winner"] == "victim").sum())
        low, high = proportion_confint(wins, games, alpha=0.05, method="wilson")
        rows.append(
            {
                "env_name": env_name,
                "victim": victim,
                "timestep": timestep,
                "games": games,
                "victim_wins": wins,
                "win_rate": wins / games,
                "win_ci_low": low,
                "win_ci_high": high,
                "tie_rate": int((group["winner"] == "tie").sum()) / games,
            }
        )
    return pd.DataFrame(rows)


def draw(curves: pd.DataFrame, out_path: Path) -> None:
    """Win rate (with 95% band) and tie rate vs. training steps; one column per environment."""
    env_names = [env for env in ENV_TITLES if env in set(curves["env_name"])]
    fig = Figure(figsize=(7.0, 4.6), dpi=300)
    grid = fig.add_gridspec(2, len(env_names), height_ratios=[3, 1.3], hspace=0.12, wspace=0.08)

    for col, env_name in enumerate(env_names):
        win_ax = fig.add_subplot(grid[0, col])
        tie_ax = fig.add_subplot(grid[1, col], sharex=win_ax)
        env_rows = curves[curves["env_name"] == env_name]
        for victim, rows in env_rows.groupby("victim"):
            style = VICTIM_STYLES[victim]
            steps = rows["timestep"] / 1e6
            line = dict(
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.5,
                markersize=3,
            )
            win_ax.fill_between(
                steps,
                rows["win_ci_low"],
                rows["win_ci_high"],
                color=style["color"],
                alpha=0.15,
                linewidth=0,
            )
            win_ax.plot(steps, rows["win_rate"], label=style["label"], **line)
            tie_ax.plot(steps, rows["tie_rate"], **line)

        win_ax.set_title(ENV_TITLES[env_name], color=INK, fontsize=9)
        win_ax.set_ylim(0, 1)
        tie_ax.set_ylim(0, 1)
        tie_ax.set_xlim(0, None)
        tie_ax.set_xlabel("training steps (millions)", color=SECONDARY_INK)
        for ax in (win_ax, tie_ax):
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            style_axes(ax)
        win_ax.tick_params(labelbottom=False)
        if col == 0:
            win_ax.set_ylabel("victim win rate", color=SECONDARY_INK)
            tie_ax.set_ylabel("tie rate", color=SECONDARY_INK)
        else:
            win_ax.tick_params(labelleft=False)
            tie_ax.tick_params(labelleft=False)

    handles = dict(zip(*reversed(win_ax.get_legend_handles_labels())))
    labels = [s["label"] for s in VICTIM_STYLES.values() if s["label"] in handles]
    fig.legend(
        [handles[label] for label in labels],
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(labels),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.2)
    fig.suptitle(
        "Victim win rate during fine-tuning against Zoo1\n(noise-trained: each game's σ drawn "
        "from U[0, 1]; control: σ = 0; shaded band: 95% Wilson interval)",
        color=INK,
        fontsize=9,
    )
    fig.savefig(out_path, facecolor="white")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dirs", nargs="+", help="fine-tuned model folders, e.g. models/*/")
    parser.add_argument("--out", required=True, help="figure path, e.g. figures/training.png")
    parser.add_argument("--bin-steps", type=int, default=500_000, help="training steps per point")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = pd.concat([load_run(d) for d in args.model_dirs], ignore_index=True)
    curves = summarize(episodes, args.bin_steps)
    curves.to_csv(out_path.with_suffix(".csv"), index=False)
    draw(curves, out_path)
    print(f"Wrote {out_path} and {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
