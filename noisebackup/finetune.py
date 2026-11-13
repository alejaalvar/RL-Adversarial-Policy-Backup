"""Fine-tune a copy of a Zoo victim with Gaussian noise on its observations (the defense).

The victim starts from its pretrained Zoo weights and keeps learning with PPO against the frozen
Zoo opponent, whose actions are sampled exactly as in the evaluation sweep. While training,
every game gets its own noise level sigma, drawn from the config's `sigma_range` ([0, 0] for the
clean control). Threat model: *direct observation* (see noise.py).

Built from the same aprl pieces as aprl.train, but without Sacred, with observation noise on
the trained agent, a sampling (not deterministic) opponent, and seeded TensorFlow graphs.
"""

import contextlib
import csv
import datetime
import json
from pathlib import Path
import time
from unittest import mock

from aprl.envs.gym_compete import game_outcome, get_policy_type_for_zoo_agent, load_zoo_agent_params
from aprl.envs.multi_agent import FlattenSingletonVecEnv, VecMultiWrapper
from aprl.training.embedded_agents import CurryVecEnv
from gym_compete import policy as gym_compete_policy
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common import tf_util
from stable_baselines.common.vec_env import VecEnvWrapper, VecNormalize
import tensorflow as tf

from noisebackup import evaluate, obs_stats, weights
from noisebackup.noise import ObservationNoise
from noisebackup.records import git_info
from noisebackup.sweep import winner_label

LOG_COLUMNS = ["timestep", "env_index", "sigma", "winner", "length"]


def make_training_env(config, log_path):
    """The single-agent environment PPO trains the victim in.

    Parallel two-player games -> noise on the victim's observations -> per-game training log
    -> the frozen opponent plays its seat -> a plain single-agent VecEnv of the victim's view
    -> win/loss reward only -> PPO's usual reward scaling.
    """
    victim_index = config["victim_index"]
    opponent_index = 1 - victim_index
    stats, _ = obs_stats.load(config["obs_stats"])

    venv = evaluate.make_venv(config["env_name"], config["num_envs"], config["seed"])
    noise = ObservationNoise(
        venv, victim_index, stats.std, tuple(config["sigma_range"]), config["noise_seed"]
    )
    log = TrainingLog(noise, noise, victim_index, log_path)
    opponent = evaluate.load_agent(
        log,
        config["env_name"],
        (config["opponent"]["type"], config["opponent"]["path"]),
        opponent_index,
        seed=config["seed"] + opponent_index,
    )
    # deterministic=False: the opponent samples its actions, as in the evaluation sweep.
    venv = CurryVecEnv(log, opponent, agent_idx=opponent_index, deterministic=False)
    venv = FlattenSingletonVecEnv(venv)
    # Win/loss reward only (the paper's sparse reward), then PPO's usual reward scaling. The
    # Zoo policy normalizes its own observations, so VecNormalize must leave them alone.
    venv = SparseReward(venv, victim_index)
    return VecNormalize(venv, norm_obs=False, gamma=config["ppo"]["gamma"])


def run(config, out_dir) -> None:
    """Fine-tune the victim as configured; write weights.pkl, training_info.json and
    training_episodes.csv to `out_dir`."""
    out_dir = Path(out_dir)
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    provenance = git_info(ignore=out_dir)
    out_dir.mkdir(parents=True)

    venv = make_training_env(config, out_dir / "training_episodes.csv")
    model = make_model(config, venv)

    start = time.time()
    model.learn(total_timesteps=config["total_timesteps"], seed=config["seed"])
    seconds = time.time() - start

    weights.save(_policy_params(model), out_dir / "weights.pkl")
    info = {
        "config": config,
        "started_at": started_at,
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **provenance,
        "timesteps": model.num_timesteps,
        "training_seconds": round(seconds, 1),
        "steps_per_second": round(model.num_timesteps / seconds, 1),
    }
    (out_dir / "training_info.json").write_text(json.dumps(info, indent=2) + "\n")
    venv.close()
    model.sess.close()


def make_model(config, venv) -> PPO2:
    """PPO2 with the configured settings, in a seeded graph, starting from the Zoo victim's
    pretrained weights."""
    ppo = config["ppo"]
    env_name, victim_index = config["env_name"], config["victim_index"]
    policy_cls, policy_kwargs = get_policy_type_for_zoo_agent(env_name, transparent_params=None)
    with _seeded_ppo_graph(config["seed"]) as seeded, _game_major_lstm_outputs() as fixed:
        model = PPO2(
            policy=policy_cls,
            env=venv,
            n_steps=ppo["batch_size"] // config["num_envs"],
            nminibatches=ppo["nminibatches"],
            noptepochs=ppo["noptepochs"],
            learning_rate=ppo["learning_rate"],
            ent_coef=ppo["ent_coef"],
            cliprange=ppo["cliprange"],
            gamma=ppo["gamma"],
            lam=ppo["lam"],
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
    if seeded != [True]:
        raise RuntimeError("stable-baselines did not build PPO2 the expected way; seed not applied")
    if not fixed:
        raise RuntimeError(
            "gym_compete's LSTM was not built the expected way; output fix not applied"
        )
    # Start from the victim's pretrained weights (as aprl.train does for Zoo fine-tuning; the
    # training model shares these variables).
    if config["victim"]["type"] != "zoo":
        raise ValueError("fine-tuning starts from a Zoo victim, e.g. {'type': 'zoo', 'path': '1'}")
    model.act_model.restore(load_zoo_agent_params(config["victim"]["path"], env_name, victim_index))
    return model


def _policy_params(model) -> np.ndarray:
    """The trained policy's variables as one flat vector, in the Zoo parameter-file order."""
    values = model.sess.run(model.act_model.get_trainable_variables())
    return np.concatenate([v.ravel() for v in values])


@contextlib.contextmanager
def _seeded_ppo_graph(seed: int):
    """While active, the TF graph stable-baselines builds PPO2 in gets a fixed random seed.

    PPO2.setup_model creates a fresh tf.Graph and calls tf_util.make_session(graph=...) before
    building the network, so seeding the graph there fixes PPO's own action sampling. Yields a
    list that gets one True per seeded graph, so the caller can check the hook was used.
    """
    seeded = []
    original = tf_util.make_session

    def make_seeded_session(*args, graph=None, **kwargs):
        graph.seed = seed
        seeded.append(True)
        return original(*args, graph=graph, **kwargs)

    with mock.patch.object(tf_util, "make_session", make_seeded_session):
        yield seeded


@contextlib.contextmanager
def _game_major_lstm_outputs():
    """While active, gym_compete's LSTMPolicy returns its outputs in the order PPO expects.

    Fixes a bug in gym_compete/policy.py (LSTMPolicy). PPO feeds a batch of several games game
    by game (game 1's steps, then game 2's, ...). The LSTM reads it that way, but stacks its
    outputs time step by time step (step 1 of every game, then step 2, ...) and flattens them
    with seq_to_batch without transposing back. So whenever a PPO minibatch holds more than one
    game, each action is scored against another game's output: approxkl ~1000 before any
    learning. Playing (one step at a time) is unaffected; this only changes the training graph.

    Yields a list that gets one True per patched call, so the caller can check it was used.
    """
    fixed = []

    def game_major_seq_to_batch(outputs):
        # outputs: [n_steps, n_games, n_hidden] -> rows ordered game by game, as PPO's batch.
        fixed.append(True)
        return tf.reshape(tf.transpose(outputs, (1, 0, 2)), [-1, outputs.shape[-1].value])

    with mock.patch.object(gym_compete_policy, "seq_to_batch", game_major_seq_to_batch):
        yield fixed


class SparseReward(VecEnvWrapper):
    """Reward = 0.01 x the game's win/loss reward for the victim (the paper's sparse reward).

    gym_compete reports it as info[victim]["reward_remaining"]: +1000 for a win, -1000 for a
    loss or a timeout, at the step the game ends (0 otherwise). Replaces the environment's
    default reward, which also contains dense movement terms.
    """

    def __init__(self, venv, victim_index):
        super().__init__(venv)
        self.victim_index = victim_index

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        observations, _, dones, infos = self.venv.step_wait()
        rewards = np.array([0.01 * info[self.victim_index]["reward_remaining"] for info in infos])
        return observations, rewards, dones, infos


class TrainingLog(VecMultiWrapper):
    """Append one CSV row per finished training game: timestep, env, sigma, winner, length.

    `noise` is the ObservationNoise wrapper underneath, read for the sigma each game was played
    with (it redraws a game's sigma as soon as that game ends, so it is read before stepping).
    """

    def __init__(self, venv, noise, victim_index, path):
        super().__init__(venv)
        self.noise = noise
        self.victim_index = victim_index
        self.path = Path(path)
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_COLUMNS).writeheader()
        self.timestep = 0
        self.lengths = np.zeros(self.num_envs, dtype=int)

    def reset(self):
        self.lengths[:] = 0
        return self.venv.reset()

    def step_wait(self):
        sigmas = self.noise.sigmas.copy()  # the sigma of each game being played this step
        observations, rewards, dones, infos = self.venv.step_wait()
        self.timestep += self.num_envs
        self.lengths += 1
        rows = []
        for i in np.flatnonzero(dones):
            rows.append(
                {
                    "timestep": self.timestep,
                    "env_index": i,
                    "sigma": sigmas[i],
                    "winner": winner_label(game_outcome(infos[i]), self.victim_index),
                    "length": self.lengths[i],
                }
            )
            self.lengths[i] = 0
        if rows:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=LOG_COLUMNS).writerows(rows)
        return observations, rewards, dones, infos


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="e.g. configs/finetune/ants_defended.json")
    parser.add_argument("--timesteps", type=int, help="override training length (quick tests)")
    parser.add_argument("--out", help="output folder (default: models/<run_name>)")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.out:
        out_dir = Path(args.out)
    elif args.timesteps is not None:
        # Shortened runs never land in models/, which holds only the real trained victims.
        out_dir = Path("results") / "scratch" / "models" / config["run_name"]
    else:
        out_dir = Path("models") / config["run_name"]

    print(f"Fine-tuning {config['run_name']} for {config['total_timesteps']} steps -> {out_dir}")
    run(config, out_dir)
    print(f"Done: {out_dir / 'weights.pkl'}")


if __name__ == "__main__":
    main()
