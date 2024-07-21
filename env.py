import gym
import numpy as np
from gym import spaces


class VanDeploymentEnv(gym.Env):
    def __init__(self, vans, goods):
        super(VanDeploymentEnv, self).__init__()
        self.vans = vans
        self.goods = goods
        self.num_vans = len(vans)

        # Action space: which van to assign to which good
        self.action_space = spaces.Discrete(self.num_vans * self.num_vans)

        # Observation space: remaining vans and goods
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_vans * 2,)
        )

    def reset(self):
        self.remaining_vans = self.vans.copy()
        self.remaining_goods = self.goods.copy()
        return self._get_observation()

    def step(self, action):
        van_idx = action // self.num_vans
        good_idx = action % self.num_vans

        if self.remaining_vans[van_idx] == 0 or self.remaining_goods[good_idx] == 0:
            reward = -10  # Penalty for invalid action
        else:
            van_capacity = self.remaining_vans[van_idx]
            good_weight = self.remaining_goods[good_idx]
            efficiency = good_weight / van_capacity
            reward = efficiency * 10  # Reward based on efficiency

            self.remaining_vans[van_idx] = 0
            self.remaining_goods[good_idx] = 0

        done = all(v == 0 for v in self.remaining_vans) or all(
            g == 0 for g in self.remaining_goods
        )
        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        return np.concatenate([self.remaining_vans, self.remaining_goods])
