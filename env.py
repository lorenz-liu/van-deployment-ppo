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

        van_capacity = self.remaining_vans[van_idx]
        good_weight = self.remaining_goods[good_idx]

        if van_capacity == 0 or good_weight == 0:
            reward = -10  # Penalty for invalid action (already assigned)
        elif good_weight > van_capacity:
            reward = -15  # Penalty for overloading the van
        else:
            efficiency = good_weight / van_capacity
            reward = efficiency * 10  # Reward based on efficiency

            self.remaining_vans[van_idx] = 0
            self.remaining_goods[good_idx] = 0

        reward -= 0.1  # Small penalty for each step

        done = all(v == 0 for v in self.remaining_vans) or all(
            g == 0 for g in self.remaining_goods
        )

        if (
            done
            and all(v == 0 for v in self.remaining_vans)
            and all(g == 0 for g in self.remaining_goods)
        ):
            reward += 50  # Bonus for completing all assignments

        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        return np.concatenate([self.remaining_vans, self.remaining_goods])
