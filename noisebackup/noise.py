"""Gaussian noise on one agent's (the victim's) observations.

Threat model: *direct observation*. The noise is added straight to the victim's inputs; the
opponent and the physics are untouched. This is NOT the paper's natural-observation attack, in
which an adversary can only influence the victim by acting in the shared environment.
"""

from aprl.envs.multi_agent import VecMultiWrapper
import numpy as np


class ObservationNoise(VecMultiWrapper):
    """Add sigma * input_std[i] * N(0, 1) to input i of one agent's observations, every step.

    Noise is drawn fresh at every reset and step, independently for each input and each
    parallel game. Other agents see their observations unchanged.

    :param venv: aprl VecMultiEnv (observations are a tuple with one entry per agent).
    :param agent_index: which agent's observations get noise (the victim).
    :param input_std: per-input standard deviation from clean play (see obs_stats).
    :param sigma: noise level relative to input_std.
    :param seed: seed for the noise generator (separate from the game seeds).
    """

    def __init__(self, venv, agent_index, input_std, sigma, seed):
        super().__init__(venv)
        input_std = np.asarray(input_std, dtype=np.float64)
        obs_shape = venv.observation_space.spaces[agent_index].shape
        if input_std.shape != obs_shape:
            raise ValueError(
                f"input_std has shape {input_std.shape} but agent {agent_index}'s observations "
                f"have shape {obs_shape}; were the stats measured on a different environment?"
            )
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")

        self.agent_index = agent_index
        self.input_std = input_std
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

    def _add_noise(self, observations):
        observations = list(observations)
        clean = observations[self.agent_index]
        noise = self.rng.standard_normal(clean.shape)
        observations[self.agent_index] = clean + self.sigma * self.input_std * noise
        return tuple(observations)

    def reset(self):
        return self._add_noise(self.venv.reset())

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        return self._add_noise(observations), rewards, dones, infos
