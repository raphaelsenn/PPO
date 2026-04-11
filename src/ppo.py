import os
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import gymnasium as gym

from src.actor_critic.actor_critic_base import ActorCritic 
from src.rollout_buffer import RolloutBuffer
from src.utils.factory import Factory
from src.utils import to_tensor


class PPO:
    """
    Implementation of the PPO algorithm.
    NOTE: Parallelized implementation for both discrete and continuous action spaces. 

    Reference:
    ---------- 
    Proximal Policy Optimization Algorithms, Schulman et al., 2017
    https://arxiv.org/abs/1707.06347 
    """ 
    def __init__(
            self,
            actor_critic: ActorCritic,
            n_envs: int,
            device: str,
            time_steps: int,
            horizon: int,
            batch_size: int,
            n_epochs: int,
            learning_rate: float,
            gamma: float,
            gae_lambda: float,
            clip_range: float,
            vf_coef: float,
            entropy_coef: float,
            norm_advantages: bool=True,
            clip_grad_norm: float | None=None,
            decay_learning_rate: bool=True,
            weight_decay: float=0.0,
            obs_scale: float|None=None,
            reward_clip: float|None=None,
            n_eval_runs: int=10,
            eval_every: int=5_000,
            save_every: int=10_000,
            seed: int=0,
            verbose: bool=True, 
    ) -> None:
        self.device = torch.device(device)

        # Init neural networks
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        # Cache actor-critic settings
        self.obs_shape = actor_critic.obs_shape
        self.state_dim = actor_critic.state_dim
        self.is_continuous = actor_critic.is_continuous
        self.action_dim = actor_critic.action_dim
        self.action_scale = actor_critic.action_scale

        # Optimizer        
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Criterion
        self.criterion_vf = nn.MSELoss()

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            self.obs_shape, self.action_dim, horizon,  n_envs, batch_size, norm_advantages, self.is_continuous, self.device
        )

        # Number of environments, total env steps and horizon
        self.n_envs = n_envs
        self.horizon = horizon 
        self.time_steps = time_steps

        # Hyperparameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_deacay = weight_decay

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_grad_norm = clip_grad_norm
        self.vf_coef = vf_coef 
        self.entropy_coef = entropy_coef
        self.norm_advantages = norm_advantages
        
        # Decay learning rate linearly 
        self.decay_learning_rate = decay_learning_rate
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1, end_factor=0, total_iters=int(time_steps / (horizon * n_envs))
        )

        # Evaluation and checkpoint settings 
        self.n_eval_runs = n_eval_runs
        self.eval_every = eval_every
        self.save_every = save_every
        self.seed = seed 
        self.verbose = verbose

        self.obs_scale = obs_scale
        self.reward_clip = reward_clip

        self.factory = None
        self._obs = None

        # Stats
        self.stats = {"t": [], "average_return": [], "std_return": []}

    @torch.no_grad()
    def get_action(self, s: np.ndarray) -> np.ndarray:
        s_t = to_tensor(s, self.obs_shape, self.device)

        if self.obs_scale is not None:
            s_t.div_(self.obs_scale) 
        
        action = self.actor_critic.pi(s_t)
        
        return action.cpu().numpy()

    @torch.no_grad()
    def get_value(self, s: np.ndarray) -> np.ndarray:
        s_t = to_tensor(s, self.obs_shape, self.device)

        if self.obs_scale is not None:
            s_t.div_(self.obs_scale) 
        
        value = self.actor_critic.value(s_t)
        return value.cpu().view(-1).numpy()

    @torch.no_grad()
    def sample(self, s: np.ndarray) -> Tuple[np.ndarray, ...]:
        s_t = to_tensor(s, self.obs_shape, self.device)

        if self.obs_scale is not None:
            s_t.div_(self.obs_scale) 

        a, log_probs, values = self.actor_critic.sample(s_t)          # [n_envs, action_dim], [n_envs]

        return a, log_probs, values

    @torch.no_grad()
    def collect_data(self) -> None:
        self.rollout_buffer.reset()

        if self._obs is None:
            self._obs, info = self.env.reset(seed=self.seed)            # [n_envs, *obs_shape], dict

        for _ in range(self.horizon): 
            s = self._obs                                               # [n_envs, *obs_shape]

            a, log_probs, values = self.sample(s)                       # [n_envs, action_dim], [n_envs]
            s_nxt, reward, terminated, truncated, info = self.env.step(a)

            # Handle reward clipping
            reward = self._handle_reward(reward)                        # [n_envs]
            # done = terminated                                           # [n_envs]
            done = np.logical_or(terminated, truncated)

            value_nxt = self.get_value(s_nxt)                           # [n_envs]
            value_nxt = value_nxt * (1.0 - done.astype(np.float32))     # [n_envs]

            self.rollout_buffer.push(
                s, a, reward, log_probs, values, value_nxt, done
            )
            self._obs = s_nxt                                           # [n_envs, *obs_shape]
            
    def optimize_actor_critic(self) -> None:
        self.actor_critic.train()
        clip_range = self.clip_range
        for _ in range(self.n_epochs):
            for s, a, log_prob_old, adv, rtg in self.rollout_buffer.minibatches(): 
                if self.obs_scale is not None: 
                    s = s.div(self.obs_scale) 

                # Compute logprobs, entropy and state values
                log_probs, entropy, values = self.actor_critic.evaluate_action(s, a)

                # Compute VF loss
                loss_vf = self.vf_coef * self.criterion_vf(values.view(-1), rtg)

                # Compute entropy bonus (encourages exploration) 
                entropy_bonus = self.entropy_coef * entropy.mean()

                # Compute CLIP loss
                ratio = torch.exp(log_probs - log_prob_old)                  # [batch_size]
                clip = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)   # [batch_size]
                objective = torch.min(ratio * adv, clip * adv)              # [batch_size]

                loss_clip = -torch.mean(objective) 
                loss = loss_clip + loss_vf - entropy_bonus

                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()

    def train(self, env: gym.Env) -> None:
        self.factory = Factory(env, self.n_envs)
        self.env = self.factory.make_vec_env()
        self.actor_critic.train()

        self._next_eval_step = self.eval_every
        self._next_save_step = self.save_every

        for t in range(0, self.time_steps, self.n_envs * self.horizon):
            # Run policy in environment for T steps 
            self.collect_data()

            # Compute rewards-to-go
            self._compute_rtgs()

            # Compute advantage estimates
            self._compute_advantages()

            # Update actor-critic
            self.optimize_actor_critic()

            # Evaluation, checkpoints, etc...
            current_step = min(t + self.n_envs * self.horizon, self.time_steps) 
            self._handle_periodic_tasks(current_step)

        self._evaluate(self.time_steps)
        self._checkpoint(self.time_steps)
        self.env.close()

    @torch.no_grad()
    def _evaluate(self, step: int) -> None: 
        self.actor_critic.eval()
        env = self.factory.make_env(None)
        rewards = np.zeros(self.n_eval_runs) 
        for ep in range(self.n_eval_runs): 
            done = False 
            s, _ = env.reset(seed=100 + ep)
            while not done:
                if self.obs_scale is not None:
                    s = s / self.obs_scale 
                
                a = self.actor_critic.act(s, deterministic=True).flatten()
                if not self.is_continuous: 
                    a = int(a.item())
                s_nxt, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated 
                s = s_nxt
                rewards[ep] += reward
        env.close()

        self.stats["t"].append(step)
        self.stats["average_return"].append(float(np.mean(rewards)))
        self.stats["std_return"].append(float(np.std(rewards)))
    
    def _handle_periodic_tasks(self, step: int) -> None:
        if self.decay_learning_rate:
            self.scheduler.step()

        if step >= self._next_eval_step:
            self._evaluate(step)
            average_return = self.stats["average_return"][-1]

            if self.verbose:
                print(
                    f"Total T: {step:8d} | "
                    f"Average Return: {average_return:10.3f}"
                )

            while step >= self._next_eval_step:
                self._next_eval_step += self.eval_every

        if step >= self._next_save_step:
            self._checkpoint(step)

            while step >= self._next_save_step:
                self._next_save_step += self.save_every

    def _compute_rtgs(self) -> None:
        self.rollout_buffer.compute_rtgs(self.gamma)

    def _compute_advantages(self) -> None:
        self.rollout_buffer.compute_advantages(self.gamma, self.gae_lambda)

    def _handle_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.reward_clip is not None:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)        # [n_envs]
        return reward

    def _checkpoint(self, step: int) -> None:
        env_id = self.factory.get_env_id() 
        
        save_dir = f"{env_id}-PPO-Checkpoints-Seed{self.seed}"
        os.makedirs(save_dir, exist_ok=True)

        file_name = f"{env_id}-PPO-ActorCritic-Lr{self.learning_rate}-t{step}-Seed{self.seed}.pt"
        file_name = os.path.join(save_dir, file_name) 
        torch.save(self.actor_critic.state_dict(), file_name)
        
        file_name = f"{env_id}-PPO-Seed{self.seed}.csv"
        pd.DataFrame.from_dict(self.stats).to_csv(file_name, index=False)