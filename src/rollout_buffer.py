from typing import Tuple

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
            self, 
            obs_shape: Tuple[int, ...],
            action_dim: int,
            horizon: int,
            n_envs: int, 
            batch_size: int,
            norm_advantages: bool,
            is_continuous: bool,
            device: torch.device
    ) -> None:
        self.T = horizon
        self.N = n_envs 

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.norm_advantages = norm_advantages
        self.is_continuous = is_continuous 
        self.device = device
        self.position = 0

        self.states = np.zeros(shape=(horizon, n_envs, *obs_shape), dtype=np.float32)
        self.rewards = np.zeros(shape=(horizon, n_envs), dtype=np.float32)
        self.dones = np.zeros(shape=(horizon, n_envs), dtype=np.float32)

        self.log_probs = np.zeros(shape=(horizon, n_envs), dtype=np.float32)
        self.values = np.zeros(shape=(horizon, n_envs), dtype=np.float32)
        self.values_nxt = np.zeros(shape=(horizon, n_envs), dtype=np.float32)
 
        self.rtgs = np.zeros(shape=(horizon, n_envs), dtype=np.float32)
        self.advantages = np.zeros(shape=(horizon, n_envs), dtype=np.float32)

        if self.is_continuous:
            self.actions = np.zeros(shape=(horizon, n_envs, self.action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(shape=(horizon, n_envs), dtype=np.int32)

    def compute_rtgs(self, gamma: float) -> None:
        rtg = np.zeros(self.N, dtype=np.float32)    # [n_envs]
        for t in reversed(range(self.T)): 
            not_done = 1.0 - self.dones[t]          # [n_envs]
            reward = self.rewards[t]                # [n_envs]
            rtg = reward + gamma * not_done * rtg   # [n_envs]
            self.rtgs[t] = rtg                      # [n_envs]

    def compute_advantages(self, gamma: float, gae_lambda: float) -> None:
        adv = np.zeros(self.N, np.float32)          # [n_envs]
        for t in reversed(range(self.T)): 
            not_done = 1.0 - self.dones[t]          # [n_envs]
            reward = self.rewards[t]                # [n_envs]
            value = self.values[t]                  # [n_envs]
            value_nxt = self.values_nxt[t]          # [n_envs]

            td_error = reward + gamma * not_done * value_nxt - value    # [n_envs]
            adv = td_error + gamma * gae_lambda * not_done * adv        # [n_envs]
            self.advantages[t] = adv                                    # [n_envs]

    def push(
            self, 
            s: np.ndarray, 
            a: np.ndarray, 
            r: np.ndarray, 
            log_prob: np.ndarray, 
            value: np.ndarray,
            value_nxt: np.ndarray, 
            done: np.ndarray
    ) -> None:
        i = self.position 
        horizon = self.T
        
        if self.is_continuous: 
            self.actions[i] = a.astype(np.float32)          # [n_envs, action_dim]
        else:
            self.actions[i] = a.astype(np.int32)            # [n_envs]

        self.states[i] = s.astype(np.float32)               # [n_envs, *obs_shape]
        self.rewards[i] = r.astype(np.float32)              # [n_envs]
        self.dones[i] = done.astype(np.float32)             # [n_envs]
        
        self.log_probs[i] = log_prob.astype(np.float32)     # [n_envs]
        self.values[i] = value.astype(np.float32)           # [n_envs]
        self.values_nxt[i] = value_nxt.astype(np.float32)   # [n_envs]

        self.position = (self.position + 1) % horizon

    def _flatten(self):
        # [T, N, ...] -> [T*N, ...]

        if self.is_continuous: 
            actions = self.actions.reshape(self.T * self.N, self.action_dim)    # [horizon * n_envs, action_dim]
        else:
            actions = self.actions.reshape(self.T * self.N)                     # [horizon * n_envs, action_dim]
        
        obs = self.states.reshape(self.T * self.N, *self.obs_shape)             # [horizon * n_envs, *obs_shape]
        log_probs = self.log_probs.reshape(self.T * self.N)                     # [horizon * n_envs]
        advantages = self.advantages.reshape(self.T * self.N)                   # [horizon * n_envs]
        rtgs = self.rtgs.reshape(self.T * self.N)                               # [horizon * n_envs]

        return obs, actions, log_probs, advantages, rtgs

    def minibatches(self):
        obs, actions, log_probs, advantages, rtgs = self._flatten()
        
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = np.random.permutation(self.T * self.N)
        for start in range(0, self.T * self.N, self.batch_size):
            mb = idx[start:start + self.batch_size]

            if self.is_continuous: 
                yield (
                    torch.as_tensor(obs[mb], dtype=torch.float32, device=self.device),              # [batch_size, *obs_shape]
                    torch.as_tensor(actions[mb], dtype=torch.float32, device=self.device),          # [batch_size, action_dim]
                    torch.as_tensor(log_probs[mb], dtype=torch.float32, device=self.device),        # [batch_size]
                    torch.as_tensor(advantages[mb], dtype=torch.float32, device=self.device),       # [batch_size]
                    torch.as_tensor(rtgs[mb], dtype=torch.float32, device=self.device),             # [batch_size]
                )
            else:
                yield (
                    torch.as_tensor(obs[mb], dtype=torch.float32, device=self.device),              # [batch_size, *obs_shape]
                    torch.as_tensor(actions[mb], dtype=torch.long, device=self.device),             # [batch_size]
                    torch.as_tensor(log_probs[mb], dtype=torch.float32, device=self.device),        # [batch_size]
                    torch.as_tensor(advantages[mb], dtype=torch.float32, device=self.device),       # [batch_size]
                    torch.as_tensor(rtgs[mb], dtype=torch.float32, device=self.device),             # [batch_size]
                )

    def reset(self) -> None:
        self.position = 0