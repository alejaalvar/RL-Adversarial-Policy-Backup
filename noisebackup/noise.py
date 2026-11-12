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

    `sigma` is either one fixed level (the attack sweep) or a (low, high) range (noise-augmented
    training): then every game gets its own sigma, drawn uniformly from the range when the game
    starts. `self.sigmas` holds the sigma of the game currently running in each parallel slot.

    :param venv: aprl VecMultiEnv (observations are a tuple with one entry per agent).
    :param agent_index: which agent's observations get noise (the victim).
    :param input_std: per-input standard deviation from clean play (see obs_stats).
    :param sigma: noise level relative to input_std, or a (low, high) range of levels.
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
        if np.isscalar(sigma):
            if sigma < 0:
                raise ValueError(f"sigma must be >= 0, got {sigma}")
        elif len(sigma) != 2 or not 0 <= sigma[0] <= sigma[1]:
            raise ValueError(
                f"a sigma range must be (low, high) with 0 <= low <= high, got {sigma}"
            )

        self.agent_index = agent_index
        self.input_std = input_std
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)
        # Per-game sigmas come from their own stream, so fixed-sigma noise is exactly what it
        # was before sigma ranges existed (round-1 results stay reproducible).
        self.sigma_rng = np.random.RandomState([seed, 1])
        self.sigmas = np.full(venv.num_envs, sigma if np.isscalar(sigma) else np.nan)

    def _add_noise(self, observations):
        observations = list(observations)
        clean = observations[self.agent_index]
        noise = self.rng.standard_normal(clean.shape)
        observations[self.agent_index] = clean + self.sigmas[:, None] * self.input_std * noise
        return tuple(observations)

    def _draw_sigmas(self, games):
        """Give each listed game a new sigma, if sigma is a (low, high) range."""
        if not np.isscalar(self.sigma):
            low, high = self.sigma
            self.sigmas[games] = self.sigma_rng.uniform(low, high, size=len(games))

    def reset(self):
        self._draw_sigmas(np.arange(self.num_envs))
        return self._add_noise(self.venv.reset())

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        # A finished game has already been reset, so `observations` holds the next game's first
        # observation: draw that game's sigma before adding noise.
        self._draw_sigmas(np.flatnonzero(dones))
        return self._add_noise(observations), rewards, dones, infos
