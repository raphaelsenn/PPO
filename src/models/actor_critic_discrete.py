import math
from typing import Tuple

import torch
import torch.nn as nn

from src.models import ActorCriticDiscrete


class ActorCriticDiscreteMLP(ActorCriticDiscrete):
    def __init__(
            self, 
            state_dim: int,
            h1_dim: int,
            h2_dim: int,
            action_dim: int, 
    ) -> None:
        super().__init__((state_dim,), action_dim)
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, h1_dim),
            nn.ReLU(True),
            
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(True),
        )

        self.actor = nn.Linear(h2_dim, action_dim)
        self.critic = nn.Linear(h2_dim, 1)

    def forward(self, s: torch.Tensor) -> Tuple:
        hs = self.mlp(s)            # [B, h2_dim]
        log_probs = self.actor(hs)  # [B, action_dim]
        value = self.critic(hs)     # [B, 1]
        return log_probs, value

    def pi(self, s: torch.Tensor) -> torch.Tensor:
        hs = self.mlp(s)                # [B, h2_dim]
        logits = self.actor(hs)         # [B, action_dim]
        action = torch.argmax(logits, dim=-1)   # [B, 1]
        return action

    def value(self, s: torch.Tensor) -> torch.Tensor:
        hs = self.mlp(s)                # [B, h2_dim]
        value = self.critic(hs)         # [B, 1]
        return value


class ActorCriticDiscreteCNN(ActorCriticDiscrete):
    def __init__(
            self, 
            obs_shape: Tuple[int, ...],
            action_dim: int, 
            fc_in_dim: int=64*7*7,
            hidden_dim: int=512 
    ) -> None:
        super().__init__(obs_shape, action_dim)
        self.state_ch = obs_shape[0]

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

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, s: torch.Tensor) -> Tuple:
        hs = self.cnn(s)            # [B, hidden_dim]
        log_probs = self.actor(hs)  # [B, action_dim]
        value = self.critic(hs)     # [B, 1]
        return log_probs, value

    def pi(self, s: torch.Tensor) -> torch.Tensor:
        hs = self.cnn(s)            # [B, hidden_dim]
        logits = self.actor(hs)     # [B, action_dim]
        action = torch.argmax(logits, dim=-1) 
        return action

    def value(self, s: torch.Tensor) -> torch.Tensor:
        hs = self.cnn(s)            # [B, hidden_dim]
        value = self.critic(hs)     # [B, 1]
        return value