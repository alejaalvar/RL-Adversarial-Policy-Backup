"""Draw figures/followup_checks.png: does the defense's effect hold up in the follow-up runs?

For Sumo Ants and Sumo Humans, the gap in victim win rate between the noise-trained victim
and the clean-trained control (noise-trained minus control, in percentage points) at sigma = 0
and sigma = 1, in five conditions: the main runs vs. Zoo1, a second training seed, 20M training
steps, and the main victims against held-out opponents Zoo2 and Zoo3. Error bars: 95% interval
for the difference of two proportions (normal approximation, 1,000 games per point).

Reads results/<game>_{defended,finetuned_clean}<suffix>/episodes.csv. Threat model: *direct
observation* (noise added straight to the victim's observations).

Run inside the container: ./run.sh python followup/plot_checks.py
"""

import argparse
import math
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pandas as pd

from noisebackup.plot import AXIS, INK, SECONDARY_INK, VICTIM_STYLES, style_axes

GAMES = {"ants": "Sumo Ants", "humans": "Sumo Humans"}
# (results-folder suffix, x-axis label), in display order.
CONDITIONS = [
    ("", "main run\nvs Zoo1"),
    ("_seed2", "seed 2\nvs Zoo1"),
    ("_20M", "20M steps\nvs Zoo1"),
    ("_opp2", "held-out\nZoo2"),
    ("_opp3", "held-out\nZoo3"),
]
# sigma = 1 (the edge of the training range) carries the noise-trained color; sigma = 0 is
# neutral and hollow, so the two also differ by fill and marker shape in grayscale.
SIGMA_STYLES = {
    0.0: {"color": SECONDARY_INK, "marker": "o", "fill": "white", "label": "σ = 0 (no noise)"},
    1.0: {
        "color": VICTIM_STYLES["defended"]["color"],
        "marker": "s",
        "fill": VICTIM_STYLES["defended"]["color"],
        "label": "σ = 1",
    },
}


def win_counts(results_dir, run_name, sigma):
    """(victim wins, games) for one sweep at one sigma."""
    episodes = pd.read_csv(Path(results_dir) / run_name / "episodes.csv")
    games = episodes[episodes["sigma"] == sigma]
    return int((games["winner"] == "victim").sum()), len(games)


def gap(results_dir, game, suffix, sigma):
    """Noise-trained minus control win rate, and its 95% half-width, in points."""
    wins_a, n_a = win_counts(results_dir, f"{game}_defended{suffix}", sigma)
    wins_b, n_b = win_counts(results_dir, f"{game}_finetuned_clean{suffix}", sigma)
    p_a, p_b = wins_a / n_a, wins_b / n_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    return 100 * (p_a - p_b), 100 * 1.96 * se


def draw(results_dir, out_path):
    fig = Figure(figsize=(7.0, 3.6), dpi=300)
    axes = fig.subplots(1, len(GAMES), sharey=True)
    offsets = {0.0: -0.14, 1.0: 0.14}

    for ax, (game, title) in zip(axes, GAMES.items()):
        ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
        # Light divider between the runs against Zoo1 and the held-out opponents.
        ax.axvline(2.5, color=AXIS, linewidth=0.6, linestyle=":", zorder=1)
        for sigma, style in SIGMA_STYLES.items():
            for x, (suffix, _) in enumerate(CONDITIONS):
                diff, half_width = gap(results_dir, game, suffix, sigma)
                ax.errorbar(
                    x + offsets[sigma],
                    diff,
                    yerr=half_width,
                    color=style["color"],
                    marker=style["marker"],
                    markerfacecolor=style["fill"],
                    markeredgecolor=style["color"],
                    markeredgewidth=1.2,
                    markersize=5,
                    elinewidth=1.2,
                    capsize=0,
                    linestyle="none",
                    zorder=3,
                )
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([label for _, label in CONDITIONS])
        ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
        style_axes(ax)

    axes[0].set_ylabel("noise-trained − control\n(win-rate points)", color=SECONDARY_INK)
    axes[0].set_ylim(-16, 16)
    axes[1].tick_params(labelleft=False)

    handles = [
        Line2D(
            [],
            [],
            color=s["color"],
            marker=s["marker"],
            markerfacecolor=s["fill"],
            markeredgewidth=1.2,
            markersize=5,
            linestyle="none",
        )
        for s in SIGMA_STYLES.values()
    ]
    fig.legend(
        handles,
        [s["label"] for s in SIGMA_STYLES.values()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles),
        frameon=False,
        fontsize=8,
    )
    # Fixed margins instead of bbox_inches="tight" (matplotlib 3.1 drops figure-level text).
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.21, wspace=0.08)
    fig.suptitle(
        "Defense effect in the follow-up runs: noise-trained minus clean-trained control\n"
        "(direct-observation threat model; above 0: noise training helped; "
        "error bars: 95% interval)",
        color=INK,
        fontsize=9,
    )
    fig.savefig(out_path, facecolor="white")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results", help="folder holding results/<run>/")
    parser.add_argument("--out", default="figures/followup_checks.png")
    args = parser.parse_args(argv)
    draw(args.results, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
