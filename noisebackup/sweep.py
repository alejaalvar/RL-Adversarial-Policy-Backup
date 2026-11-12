"""Evaluate a victim at every noise level (sigma) and record one row per game.

Threat model: *direct observation* (noise added to the victim's observations); see noise.py.
"""

import argparse
import csv
import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from noisebackup import evaluate, obs_stats
from noisebackup.evaluate import Episode
from noisebackup.noise import ObservationNoise
from noisebackup.records import git_info

Config = Dict[str, Any]
EvaluateFn = Callable[[Config, float], List[Episode]]

COLUMNS = ["env_name", "victim", "sigma", "episode", "env_index", "winner", "length"]


def run_sweep(config: Config, evaluate_fn: EvaluateFn, out_dir: Union[str, Path]) -> None:
    """Play `config["episodes"]` games at each sigma; append them to out_dir/episodes.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "episodes.csv"

    run_info_path = out_dir / "run_info.json"
    config_path = out_dir / "config.json"

    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(
            f"{out_dir} already holds results from a different config; use a new output "
            "folder (or delete this one) instead of mixing results from two setups."
        )
    _write_json(config_path, config)
    invocation = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **git_info(ignore=out_dir),  # this run's own results aren't "uncommitted code"
        "sigmas_played": [],
    }
    run_info = json.loads(run_info_path.read_text()) if run_info_path.exists() else {}
    run_info.setdefault("invocations", []).append(invocation)
    _write_json(run_info_path, run_info)

    if csv_path.exists():
        finished = _finished_sigmas(csv_path)
    else:
        finished = set()
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    for sigma in config["sigmas"]:
        if float(sigma) in finished:
            continue
        # Rows are only written once all games at this sigma have finished, so a crash
        # mid-sigma leaves no partial rows and the sigma is simply replayed next time.
        episodes = evaluate_fn(config, sigma)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            for number, episode in enumerate(episodes):
                writer.writerow(
                    {
                        "env_name": config["env_name"],
                        "victim": config["victim"]["label"],
                        "sigma": sigma,
                        "episode": number,
                        "env_index": episode.env_index,
                        "winner": _winner_label(episode.winner, config["victim_index"]),
                        "length": episode.length,
                    }
                )
        invocation["sigmas_played"].append(sigma)
        _write_json(run_info_path, run_info)


def play_sigma(config: Config, sigma: float) -> List[Episode]:
    """Real games: the victim, with noise at `sigma` on its observations, vs. the opponent.

    The game seeds and the noise seed are the same at every sigma, so sigma is the only
    thing that changes between points.
    """
    stats, _ = obs_stats.load(config["obs_stats"])
    victim_index = config["victim_index"]
    specs = [None, None]
    specs[victim_index] = (config["victim"]["type"], config["victim"]["path"])
    specs[1 - victim_index] = (config["opponent"]["type"], config["opponent"]["path"])

    venv = evaluate.make_venv(config["env_name"], config["num_envs"], config["seed"])
    try:
        venv = ObservationNoise(venv, victim_index, stats.std, sigma, config["noise_seed"])
        policies = evaluate.load_policies(venv, config["env_name"], specs)
        return evaluate.run_episodes(venv, policies, config["episodes"])
    finally:
        venv.close()


def _finished_sigmas(csv_path: Path) -> set:
    """Sigmas that already have rows in the results file."""
    with open(csv_path, newline="") as f:
        return {float(row["sigma"]) for row in csv.DictReader(f)}


def _winner_label(winner, victim_index: int) -> str:
    """Name the winner from the victim's point of view."""
    if winner is None:
        return "tie"
    return "victim" if winner == victim_index else "opponent"


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="JSON config, e.g. configs/ants.json")
    parser.add_argument("--episodes", type=int, help="override games per sigma (quick tests)")
    parser.add_argument("--num-envs", type=int, help="override number of parallel games")
    parser.add_argument("--out", help="output folder (default: results/<run_name>)")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    overridden = False
    if args.episodes is not None:
        config["episodes"] = args.episodes
        overridden = True
    if args.num_envs is not None:
        config["num_envs"] = args.num_envs
        overridden = True

    if args.out:
        out_dir = Path(args.out)
    elif overridden:
        # Quick test runs never land in the real (committed) results folder.
        out_dir = Path("results") / "scratch" / config["run_name"]
    else:
        out_dir = Path("results") / config["run_name"]

    print(f"Sweeping sigmas {config['sigmas']} on {config['env_name']} -> {out_dir}")
    run_sweep(config, play_sigma, out_dir)
    print(f"Done: {out_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()
