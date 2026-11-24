"""Summarize the follow-up experiments (queued 2026-11-21) into results/followup_summary.md.

For every follow-up sweep (configs/followup/*.json): victim win rate with Wilson 95% interval at
each sigma, and the difference from the matching committed run (training seed 1000, 10M steps,
opponent Zoo1) with a two-sided two-proportion z test (pooled, as in the PREDICTIONS.md
addendum). Also, within each follow-up condition, the noise-trained victim vs. the control.
A Holm correction runs over the addendum's 42 comparisons plus all the new ones (both kinds).
Failed or missing runs are listed from results/followup/status.tsv and the results folders.

Threat model: direct observation (noise added straight to the victim's observations).

Run inside the container: ./run.sh python followup/summarize.py
"""

import argparse
import datetime
import json
import math
from pathlib import Path
import subprocess

import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

GAMES = {"ants": "Sumo Ants", "humans": "Sumo Humans"}
VICTIMS = ["zoo1_original", "defended", "finetuned_clean"]
# Comparison families, all in one Holm correction.
ADDENDUM, VS_COMMITTED, WITHIN = "addendum", "vs. committed", "within condition"
# The addendum's 42 comparisons: these three pairs, at each of 7 sigmas, in both games.
PAPER_PAIRS = [
    ("defended", "finetuned_clean"),
    ("defended", "zoo1_original"),
    ("finetuned_clean", "zoo1_original"),
]
GROUPS = [
    ("_opp2", "1. Held-out opponent Zoo2", "The victim is unchanged; only the opponent differs."),
    ("_opp3", "1. Held-out opponent Zoo3", "The victim is unchanged; only the opponent differs."),
    (
        "_seed2",
        "2. Extra training seed",
        "Training seed 2000 / noise seed 2001 instead of 1000 / 1001.",
    ),
    ("_20M", "3. Twice as long (20M steps)", "Trained from Zoo1 for 20M steps instead of 10M."),
]


def ztest(wins_a, n_a, wins_b, n_b):
    """Two-sided pooled two-proportion z test: (difference in win rate, p-value)."""
    p_a, p_b = wins_a / n_a, wins_b / n_b
    pooled = (wins_a + wins_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se if se else 0.0
    return p_a - p_b, math.erfc(abs(z) / math.sqrt(2))


def load_wins(results_dir, run_name, expected=None):
    """{sigma: (wins, games)} for a sweep, or None if it is missing or incomplete.

    `expected` is the run's config: every sigma must have all its games.
    """
    path = Path(results_dir) / run_name / "episodes.csv"
    if not path.exists():
        return None
    episodes = pd.read_csv(path)
    counts = {
        float(sigma): (int((group["winner"] == "victim").sum()), len(group))
        for sigma, group in episodes.groupby("sigma")
    }
    if expected is not None:
        if any(counts.get(float(s), (0, 0))[1] != expected["episodes"] for s in expected["sigmas"]):
            return None
    return counts


def base_run(run_name):
    """The committed run a follow-up run is compared with: its name without the suffix."""
    for suffix, _, _ in GROUPS:
        if run_name.endswith(suffix):
            return run_name[: -len(suffix)], suffix
    raise ValueError(f"{run_name}: no follow-up suffix")


def comparisons(results_dir, configs):
    """All comparisons: the addendum's 42; each follow-up sweep vs. its committed run; and,
    within each follow-up condition, noise-trained vs. control. Holm runs over all of them."""
    rows = []
    for game in GAMES:
        runs = {v: load_wins(results_dir, f"{game}_{v}") for v in VICTIMS}
        for a, b in PAPER_PAIRS:
            for sigma in sorted(runs[a]):
                rows.append(
                    compare(ADDENDUM, f"{game}_{a}", runs[a], f"{game}_{b}", runs[b], sigma)
                )
    complete = {}
    for run_name, config in configs.items():
        new = load_wins(results_dir, run_name, config)
        if new is None:
            continue
        complete[run_name] = new
        base, _ = base_run(run_name)
        committed = load_wins(results_dir, base)
        for sigma in sorted(new):
            rows.append(compare(VS_COMMITTED, run_name, new, base, committed, sigma))
    # Added 2026-11-24, after the results were in: does the defense still beat the control?
    for suffix, _, _ in GROUPS:
        for game in GAMES:
            a, b = f"{game}_defended{suffix}", f"{game}_finetuned_clean{suffix}"
            if a in complete and b in complete:
                for sigma in sorted(complete[a]):
                    rows.append(compare(WITHIN, a, complete[a], b, complete[b], sigma))
    table = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(table["p"], alpha=0.05, method="holm")
    table["p_holm"] = p_holm
    table["significant"] = reject
    return table


def compare(family, name_a, runs_a, name_b, runs_b, sigma):
    (wins_a, n_a), (wins_b, n_b) = runs_a[sigma], runs_b[sigma]
    diff, p = ztest(wins_a, n_a, wins_b, n_b)
    return {
        "family": family,
        "a": name_a,
        "b": name_b,
        "sigma": sigma,
        "wins_a": wins_a,
        "n_a": n_a,
        "wins_b": wins_b,
        "n_b": n_b,
        "diff": diff,
        "p": p,
    }


def pct(x):
    return f"{100 * x:.1f}"


def fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def sweep_table(run_name, table):
    rows = table[(table["family"] == VS_COMMITTED) & (table["a"] == run_name)]
    rows = rows.sort_values("sigma")
    lines = [
        "| σ | wins / games | win rate [95% CI] | committed | difference (points) | p | Holm p |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in rows.iterrows():
        low, high = proportion_confint(r["wins_a"], r["n_a"], alpha=0.05, method="wilson")
        star = " ✱" if r["significant"] else ""
        lines.append(
            f"| {r['sigma']:g} | {r['wins_a']} / {r['n_a']} "
            f"| {pct(r['wins_a'] / r['n_a'])}% [{pct(low)}, {pct(high)}] "
            f"| {pct(r['wins_b'] / r['n_b'])}% | {100 * r['diff']:+.1f} "
            f"| {fmt_p(r['p'])} | {fmt_p(r['p_holm'])}{star} |"
        )
    return lines


def within_table(rows):
    """Noise-trained vs. control in one condition and game, one row per sigma."""
    lines = [
        "| σ | noise-trained | control | gap (points) | p | Holm p |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in rows.sort_values("sigma").iterrows():
        star = " ✱" if r["significant"] else ""
        lines.append(
            f"| {r['sigma']:g} | {pct(r['wins_a'] / r['n_a'])}% | {pct(r['wins_b'] / r['n_b'])}% "
            f"| {100 * r['diff']:+.1f} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])}{star} |"
        )
    return lines


def problems(results_dir, configs, status_path):
    """Lines describing failed, skipped, missing or incomplete runs."""
    lines = []
    if Path(status_path).exists():
        status = pd.read_csv(
            status_path, sep="\t", names=["time", "step", "result", "detail"], dtype=str
        )
        # A step may have been retried after a restart; its latest line counts.
        latest = status.groupby("step").tail(1)
        for _, r in latest[latest["result"] != "OK"].iterrows():
            lines.append(f"- **{r['step']}**: {r['result']} ({r['detail']}, {r['time']})")
    else:
        lines.append(f"- No step log found at `{status_path}`.")
    for run_name, config in configs.items():
        if load_wins(results_dir, run_name, config) is None:
            lines.append(f"- **{run_name}**: no complete results in `results/{run_name}/`.")
    return lines


def provenance(results_dir, configs):
    lines = [
        "| run | trained (steps, hours, git_dirty) | swept at commit | sweep git_dirty |",
        "|---|---|---|---|",
    ]
    for run_name, config in configs.items():
        trained = ""
        if config["victim"]["type"] == "finetuned":
            info_path = Path(config["victim"]["path"]).parent / "training_info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
                trained = (
                    f"{info['timesteps']:,}, {info['training_seconds'] / 3600:.1f} h, "
                    f"{str(info['git_dirty']).lower()}"
                )
        commits, dirty = "", ""
        info_path = Path(results_dir) / run_name / "run_info.json"
        if info_path.exists():
            invocations = json.loads(info_path.read_text())["invocations"]
            commits = ", ".join(sorted({i["git_commit"][:7] for i in invocations}))
            dirty = ", ".join(sorted({str(i["git_dirty"]).lower() for i in invocations}))
        lines.append(f"| {run_name} | {trained} | {commits} | {dirty} |")
    return lines


def report(results_dir, configs_dir, status_path):
    configs = {p.stem: json.loads(p.read_text()) for p in sorted(Path(configs_dir).glob("*.json"))}
    order = {suffix: i for i, (suffix, _, _) in enumerate(GROUPS)}
    configs = dict(sorted(configs.items(), key=lambda kv: (order[base_run(kv[0])[1]], kv[0])))
    table = comparisons(results_dir, configs)
    counts = table["family"].value_counts()
    n_paper, n_committed, n_within = (counts.get(f, 0) for f in (ADDENDUM, VS_COMMITTED, WITHIN))
    commit = (
        subprocess.run(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE, check=False)
        .stdout.decode()
        .strip()
    )

    out = [
        "# Follow-up experiments: summary",
        "",
        f"Generated {datetime.date.today().isoformat()} at commit {commit or '?'} by "
        "`followup/summarize.py` from the raw `episodes.csv` files. **Threat model: direct "
        "observation** (noise added straight to the victim's observations).",
        "",
        "**How to read this.**",
        "- Win rate = victim wins ÷ all games (ties count as non-wins), with a Wilson 95% "
        "interval.",
        "- *Committed* is the matching committed run: same victim and σ, trained with seed 1000 "
        "for 10M steps, against opponent Zoo1 (`results/<run>/` without the suffix).",
        "- *Difference* = follow-up − committed, in percentage points. *p*: two-sided pooled "
        "two-proportion z test, as in the PREDICTIONS.md addendum.",
        f"- *Holm p*: Holm-corrected over **{len(table)} comparisons**: the addendum's "
        f"{n_paper} (noise-trained vs. control, noise-trained vs. Zoo1, control vs. Zoo1, at "
        f"each σ in both games), {n_committed} follow-up sweeps vs. their committed runs, and "
        f"{n_within} noise-trained vs. control within each follow-up condition (section 4; "
        "added to the correction on 2026-11-24, after the results were in). ✱ = significant at "
        "0.05 after correction.",
        "- For the held-out opponents, the difference measures the *opponent* change (the "
        "victim is the same), so a drop means Zoo2/Zoo3 is a harder opponent for that victim.",
        "",
        "## Failed, skipped or missing runs",
        "",
    ]
    issues = problems(results_dir, configs, status_path)
    out += issues or ["None: every step finished and every sweep is complete."]
    out += [""]

    for suffix, title, note in GROUPS:
        runs = [r for r in configs if base_run(r)[1] == suffix]
        out += [f"## {title}", "", note, ""]
        for run_name in runs:
            out += [f"### {run_name} (vs. {base_run(run_name)[0]})", ""]
            if ((table["family"] == VS_COMMITTED) & (table["a"] == run_name)).any():
                out += sweep_table(run_name, table)
            else:
                out += ["No complete results (see above)."]
            out += [""]

    out += [
        "## 4. Noise-trained vs. control within each condition",
        "",
        "Does the defense still beat the control? Both victims from the same condition, at each σ.",
        "",
    ]
    within = table[table["family"] == WITHIN]
    for suffix, title, _ in GROUPS:
        for game, game_title in GAMES.items():
            rows = within[within["a"] == f"{game}_defended{suffix}"]
            out += [f"### {title[3:]}: {game_title}", ""]
            out += within_table(rows) if len(rows) else ["No complete results (see above)."]
            out += [""]

    out += [
        f"## Holm-corrected list: all {len(table)} comparisons",
        "",
        "Sorted by raw p. *A − B* is the difference in victim win rate.",
        "",
        "| # | source | A − B | σ | difference (points) | p | Holm p | significant |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rank, (_, r) in enumerate(table.sort_values("p", kind="mergesort").iterrows(), 1):
        out.append(
            f"| {rank} | {r['family']} | {r['a']} − {r['b']} | {r['sigma']:g} "
            f"| {100 * r['diff']:+.1f} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} "
            f"| {'yes' if r['significant'] else 'no'} |"
        )
    out += ["", "## Provenance", ""] + provenance(results_dir, configs) + [""]
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", default="configs/followup", help="follow-up sweep configs")
    parser.add_argument("--results", default="results", help="folder holding results/<run>/")
    parser.add_argument("--status", default="results/followup/status.tsv", help="step log")
    parser.add_argument("--out", default="results/followup_summary.md")
    args = parser.parse_args(argv)

    text = report(args.results, args.configs, args.status)
    Path(args.out).write_text(text)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
