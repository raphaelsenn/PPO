from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, k_repeats: int) -> None:
        super().__init__(env)

        if k_repeats < 1:
            raise ValueError("k_repeats must be >= 1")
        
        self.k_repeats = k_repeats

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        reward_sum = 0.0
        
        terminated = False
        truncated = False
        info: dict[str, Any] = {} 
        
        for _ in range(self.k_repeats):
            obs_nxt,  reward, terminated, truncated, info = self.env.step(action) 
            reward_sum += reward
        
            if terminated or truncated:
                break 

        return obs_nxt, reward_sum, terminated, truncated, info