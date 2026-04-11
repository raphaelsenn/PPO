import copy
import math
from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import to_tensor, EPS


class ActorCritic(nn.Module, ABC):
    """Actor-Critic interface for deterministic/stochastic policies and a state-value function.""" 
    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            action_dim: int,
            is_continuous: bool
    ) -> None:
        super().__init__()
        if len(obs_shape) == 0:
            raise ValueError(f"obs_dim must be non-empty, got: {obs_shape}")

        self.obs_shape = tuple(int(element) for element in obs_shape)
        self.state_dim = math.prod(obs_shape)
        self.action_dim = action_dim
        self.is_continuous = is_continuous

        self.state_ch = None
        self.action_scale = None

    @abstractmethod
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Needs to be differentiable.""" 
        raise NotImplementedError

    @abstractmethod
    def pi(self, s: torch.Tensor) -> torch.Tensor:
        """Needs to be differentiable.""" 
        raise NotImplementedError
    
    @abstractmethod
    def value(self, s: torch.Tensor) -> torch.Tensor:
        """Needs to be differentiable.""" 
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool=True) -> np.ndarray:
        """Does not need to be differentiable.""" 
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def sample(self, s: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Does not need to be differentiable.""" 
        raise NotImplementedError

    @abstractmethod
    def evaluate_action(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[np.ndarray, ...]:
        """Needs to be differentiable.""" 
        raise NotImplementedError

    def copy(self) -> 'ActorCritic':
        actor_critic = copy.deepcopy(self)
        return actor_critic
    

class ActorCriticDiscrete(ActorCritic, ABC):
    """Discrete actor-critic with categorical policy.""" 
    def __init__(
            self, 
            obs_shape: Tuple[int, ...],
            action_dim: int, 
    ) -> None:
        super().__init__(obs_shape, action_dim, is_continuous=False)

    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool=True) -> np.ndarray:
        s_t = to_tensor(s, self.obs_shape, next(self.parameters()).device)
        logits, _ = self(s_t)
        
        if deterministic: 
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits) 
            action = dist.sample()

        return action.cpu().numpy()

    @torch.no_grad()
    def sample(self, s: np.ndarray) -> Tuple[np.ndarray, ...]:
        s_t = to_tensor(s, self.obs_shape, next(self.parameters()).device)
        logits, value = self(s_t)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() 
        log_probs = dist.log_prob(action)

        action = action.cpu().numpy()
        log_probs = log_probs.cpu().numpy()
        value = value.view(-1).cpu().numpy()

        return action, log_probs, value

    def evaluate_action(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        logits, values = self(s)
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(a)
        entropy = dist.entropy()

        return log_probs, entropy, values


class ActorCriticContinuous(ActorCritic, ABC):
    def __init__(
            self, 
            obs_shape: Tuple[int, ...],
            action_dim: int,
            action_scale: float
    ) -> None:
        super().__init__(obs_shape, action_dim, is_continuous=True)
        self.action_scale = action_scale
    
    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool=True) -> np.ndarray:
        s_t = to_tensor(s, self.obs_shape, next(self.parameters()).device)

        mu, std, _ = self(s_t)
        
        if deterministic:
            a_pre_tanh = mu
        else:
            dist = torch.distributions.Normal(mu, std)
            a_pre_tanh = dist.rsample()                             # [B, action_dim]
        
        a_tanh = torch.tanh(a_pre_tanh)                             # [B, action_dim] 
        a = self.action_scale * a_tanh                              # [B, action_dim]

        return a.cpu().numpy()

    @torch.no_grad()
    def sample(self, s: np.ndarray) -> Tuple:
        s_t = to_tensor(s, self.obs_shape, next(self.parameters()).device)
        mu, std, value = self(s_t)

        # Reparametrization trick 
        dist = torch.distributions.Normal(mu, std)
        a_pre_tanh = dist.rsample()                                 # [B, action_dim]
        a_tanh = torch.tanh(a_pre_tanh)                             # [B, action_dim] 
        a = self.action_scale * a_tanh                              # [B, action_dim]

        # Compute correct log-probs
        log_probs = dist.log_prob(a_pre_tanh).sum(dim=-1)                                       # [B]
        log_probs -= (2*(np.log(2) - a_pre_tanh - F.softplus(-2*a_pre_tanh))).sum(dim=-1)       # [B]
        log_probs -= self.action_dim * np.log(self.action_scale)

        # Convert to ndarray
        a = a.cpu().numpy()
        log_probs = log_probs.cpu().numpy()
        value = value.view(-1).cpu().numpy()

        return a, log_probs, value

    def evaluate_action(self, s: torch.Tensor, a: torch.Tensor) -> Tuple:
        mu, std, values = self(s)
        dist = torch.distributions.Normal(mu, std)

        # Undue reparametrization trick
        a_tanh = (1.0 / self.action_scale) * a
        a_tanh = torch.clamp(a_tanh, -1.0 + EPS, 1.0 - EPS) 
        a_pre_tanh = torch.atanh(a_tanh)

        # Original logprobs
        log_probs = dist.log_prob(a_pre_tanh).sum(dim=-1)
        log_probs -= (2*(np.log(2) - a_pre_tanh - F.softplus(-2*a_pre_tanh))).sum(dim=-1)
        log_probs -= self.action_dim * np.log(self.action_scale)

        # Compute entropy of policy
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, entropy, values