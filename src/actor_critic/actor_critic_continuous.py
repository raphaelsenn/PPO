from typing import Tuple

import torch
import torch.nn as nn

from src.actor_critic.actor_critic_base import ActorCriticContinuous
from src.utils.utils import LOG_STD_MIN, LOG_STD_MAX


class ActorCriticContinuousMLP(ActorCriticContinuous):
    def __init__(
            self, 
            state_dim: int,
            h1_dim: int,
            h2_dim: int,
            action_dim: int,
            action_scale: float 
    ) -> None:
        super().__init__((state_dim,), action_dim, action_scale)
        self.action_scale = action_scale

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, h1_dim),
            nn.ReLU(True),
            
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(h2_dim, action_dim)
        self.log_std = nn.Linear(h2_dim, action_dim)
        self.critic = nn.Linear(h2_dim, 1)

    def forward(self, s: torch.Tensor) -> Tuple:
        hs = self.mlp(s)            # [B, h2_dim]
        value = self.critic(hs)     # [B, 1]
        
        mu = self.mu(hs)            # [B, action_dim]
        log_std = self.log_std(hs)  # [B, action_dim]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) 
        std = torch.exp(log_std)    # [B, action_dim]
        
        return mu, std, value

    def pi(self, s: torch.Tensor) -> torch.Tensor:
        hs = self.mlp(s)            # [B, h2_dim]
        mu = self.mu(hs)            # [B, action_dim]
        return self.action_scale * torch.tanh(mu)

    def value(self, s: torch.Tensor) -> torch.Tensor:
        hs = self.mlp(s)            # [B, h2_dim]
        v_value = self.critic(hs)   # [B, 1]
        return v_value


class ActorCriticContinuousCNN(ActorCriticContinuous):
    def __init__(
            self, 
            obs_shape: Tuple[int, ...],     # [in_channels, width, height]
            action_dim: int, 
            action_scale: float,
            fc_in_dim: int=64*7*7,
            hidden_dim: int=512
    ) -> None:
        super().__init__(obs_shape, action_dim, action_scale)
        self.state_ch = obs_shape[0]
        self.action_scale = action_scale
        self.fc_in_dim = fc_in_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(self.state_ch, 32, 8, 4),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),

            nn.Flatten(start_dim=1),
            nn.Linear(fc_in_dim, hidden_dim),
            nn.ReLU(True),
        )

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, s: torch.Tensor) -> Tuple:
        hs = self.cnn(s)            # [B, hidden_dim]

        mu = self.mu(hs)            # [B, action_dim]
        log_std = self.log_std(hs)  # [B, action_dim]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) 
        std = torch.exp(log_std)   
        
        value = self.critic(hs)     # [B, 1]
        return mu, std, value

    def pi(self, s: torch.Tensor) -> torch.Tensor:
        hs = self.cnn(s)            # [B, hidden_dim]
        mu = self.mu(hs)            # [B, action_dim]
        return self.action_scale * torch.tanh(mu)

    def value(self, s: torch.Tensor) -> torch.Tensor:
        hs = self.cnn(s)            # [B, h2_dim]
        value = self.critic(hs)     # [B, 1]
        return value