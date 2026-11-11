"""Test doubles that follow aprl's VecMultiEnv interface without running MuJoCo."""

from aprl.envs.multi_agent import VecMultiEnv
import gym
import numpy as np


class ConstantObsVecMultiEnv(VecMultiEnv):
    """Two-player vectorized env whose observations never change.

    Agent i always observes `base_obs[i]` in every parallel game, so any difference in what an
    agent sees after wrapping must have been added by the wrapper. Follows aprl's convention:
    observations are a num_agents-length tuple, entry i of shape (num_envs, obs_dim_i).
    """

    def __init__(self, base_obs, num_envs=1):
        self.base_obs = [np.asarray(obs, dtype=np.float64) for obs in base_obs]
        obs_space = gym.spaces.Tuple(
            [gym.spaces.Box(-np.inf, np.inf, shape=obs.shape) for obs in self.base_obs]
        )
        act_space = gym.spaces.Tuple([gym.spaces.Box(-1.0, 1.0, shape=(1,)) for _ in self.base_obs])
        super().__init__(num_envs, len(self.base_obs), obs_space, act_space)

    def _observations(self):
        return tuple(np.tile(obs, (self.num_envs, 1)) for obs in self.base_obs)

    def reset(self):
        return self._observations()

    def step_async(self, actions):
        pass

    def step_wait(self):
        rewards = np.zeros((self.num_agents, self.num_envs))
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        return self._observations(), rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *method_args, **method_kwargs):
        raise NotImplementedError
