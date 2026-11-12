"""Summarize sweep results (victim win rate vs. sigma) and draw the figure.

Threat model: *direct observation* (noise added to the victim's observations); see noise.py.
"""

import argparse
from pathlib import Path
from typing import Sequence

from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ENV_TITLES = {
    "multicomp/SumoAnts-v0": "Sumo Ants",
    "multicomp/SumoHumans-v0": "Sumo Humans",
}

# Color follows the victim, never its position in the data. Dashes and marker shapes repeat
# the distinction for grayscale printing. Colors pass the dataviz palette validator
# (colorblind-safe); the aqua is just under 3:1 contrast on white, which is allowed because the
# line style, marker, legend and summary CSV identify it too.
VICTIM_STYLES = {
    "zoo1": {"color": "#2a78d6", "linestyle": "-", "marker": "o", "label": "original (Zoo1)"},
    "defended": {"color": "#eb6834", "linestyle": "--", "marker": "s", "label": "noise-trained"},
    "finetuned_clean": {
        "color": "#1baf7a",
        "linestyle": "-.",
        "marker": "^",
        "label": "clean-fine-tuned (control)",
    },
}

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
AXIS = "#c3c2b7"
GRID = "#e1e0d9"


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (environment, victim, sigma): game counts, victim win rate with Wilson 95%
    interval, and tie rate. Sorted by environment, victim, then sigma.

    Win rate = victim wins / all games, so ties count as non-wins (as in the paper).
    """
    rows = []
    for (env_name, victim, sigma), group in results.groupby(["env_name", "victim", "sigma"]):
        games = len(group)
        wins = int((group["winner"] == "victim").sum())
        ties = int((group["winner"] == "tie").sum())
        low, high = proportion_confint(wins, games, alpha=0.05, method="wilson")
        rows.append(
            {
                "env_name": env_name,
                "victim": victim,
                "sigma": sigma,
                "games": games,
                "victim_wins": wins,
                "opponent_wins": int((group["winner"] == "opponent").sum()),
                "ties": ties,
                "win_rate": wins / games,
                "win_ci_low": low,
                "win_ci_high": high,
                "tie_rate": ties / games,
            }
        )
    return pd.DataFrame(rows)


def load_results(csv_paths: Sequence[str]) -> pd.DataFrame:
    """Concatenate one or more episodes.csv files."""
    return pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)


def draw(summary: pd.DataFrame, out_path: Path) -> None:
    """Win rate (with 95% band) and tie rate vs. sigma; one column per environment."""
    env_names = [env for env in ENV_TITLES if env in set(summary["env_name"])]
    unknown = set(summary["victim"]) - set(VICTIM_STYLES)
    if unknown:
        raise ValueError(f"No plot style for victim label(s) {sorted(unknown)}")

    fig = Figure(figsize=(7.0, 4.6), dpi=300)
    grid = fig.add_gridspec(2, len(env_names), height_ratios=[3, 1.3], hspace=0.12, wspace=0.08)
    sigmas = sorted(summary["sigma"].unique())

    for col, env_name in enumerate(env_names):
        win_ax = fig.add_subplot(grid[0, col])
        tie_ax = fig.add_subplot(grid[1, col], sharex=win_ax)
        env_rows = summary[summary["env_name"] == env_name]
        for victim, rows in env_rows.groupby("victim"):
            style = VICTIM_STYLES[victim]
            line = dict(
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.5,
                markersize=4.5,
            )
            win_ax.fill_between(
                rows["sigma"],
                rows["win_ci_low"],
                rows["win_ci_high"],
                color=style["color"],
                alpha=0.15,
                linewidth=0,
            )
            win_ax.plot(rows["sigma"], rows["win_rate"], label=style["label"], **line)
            tie_ax.plot(rows["sigma"], rows["tie_rate"], **line)

        win_ax.set_title(ENV_TITLES[env_name], color=INK, fontsize=9)
        win_ax.set_ylim(0, 1)
        tie_ax.set_ylim(0, 1)
        tie_ax.set_xscale("symlog", linthreshx=0.05)
        tie_ax.xaxis.set_major_locator(FixedLocator(sigmas))
        tie_ax.xaxis.set_minor_locator(NullLocator())
        tie_ax.set_xticklabels([f"{s:g}" for s in sigmas], rotation=45)
        tie_ax.set_xlabel("noise level σ (× each input's clean std)", color=SECONDARY_INK)
        for ax in (win_ax, tie_ax):
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            _style_axes(ax)
        win_ax.tick_params(labelbottom=False)
        if col == 0:
            win_ax.set_ylabel("victim win rate", color=SECONDARY_INK)
            tie_ax.set_ylabel("tie rate", color=SECONDARY_INK)
        else:
            win_ax.tick_params(labelleft=False)
            tie_ax.tick_params(labelleft=False)

    has_legend = summary["victim"].nunique() > 1
    if has_legend:
        # Legend in VICTIM_STYLES order (original first), below the panels.
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
    # Fixed margins instead of bbox_inches="tight": in matplotlib 3.1 the tight crop drops the
    # figure-level legend and title.
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.2 if has_legend else 0.15)
    title = "Victim win rate under Gaussian observation noise"
    if not has_legend:  # one line: name it in the title instead of a legend
        title += f" (victim: {VICTIM_STYLES[summary['victim'].iloc[0]]['label']})"
    fig.suptitle(
        title + "\n(direct-observation threat model; shaded band: 95% Wilson interval)",
        color=INK,
        fontsize=9,
    )
    fig.savefig(out_path, facecolor="white")


def _style_axes(ax) -> None:
    """Recessive axes: hairline grid, no top/right spines, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SECONDARY_INK, labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", help="episodes.csv file(s) from sweeps")
    parser.add_argument("--out", required=True, help="figure path, e.g. figures/attack.png")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(load_results(args.csv))
    summary.to_csv(out_path.with_suffix(".csv"), index=False)
    draw(summary, out_path)
    print(f"Wrote {out_path} and {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
