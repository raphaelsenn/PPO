import math
import torch
import numpy as np

import pytest
from src import ActorCriticContinuousCNN


@pytest.fixture
def actor_critic() -> ActorCriticContinuousCNN:
    return ActorCriticContinuousCNN((4, 84, 84), 10, 1.0, 64*7*7, 512)


@pytest.fixture
def state_batch() -> torch.Tensor:
    return torch.randn((64, 4, 84, 84), dtype=torch.float32)


@pytest.fixture
def state() -> torch.Tensor:
    return torch.randn(1, 4, 84, 84, dtype=torch.float32)


@pytest.fixture
def state_np() -> np.ndarray:
    return np.random.rand(4, 84, 84)


class TestActorCriticContinuousCNN:
    def test_init(self, actor_critic: ActorCriticContinuousCNN) -> None:
        assert actor_critic.obs_shape == (4, 84, 84)
        assert actor_critic.state_dim == 4 * 84 * 84
        assert actor_critic.state_ch == 4
        assert actor_critic.action_dim == 10
        assert actor_critic.is_continuous is True
        assert actor_critic.action_scale == 1.0
    
    def test_forward(self, actor_critic: ActorCriticContinuousCNN, state_batch: torch.Tensor) -> None:
        mu, std, values = actor_critic(state_batch)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert isinstance(values, torch.Tensor)

        assert mu.shape == (64, 10)
        assert std.shape == (64, 10)
        assert values.shape == (64, 1)
    
    def test_sample(self, actor_critic: ActorCriticContinuousCNN, state_batch: torch.Tensor) -> None:
        action, logprobs, values = actor_critic.sample(state_batch) 

        assert isinstance(action, np.ndarray)        
        assert isinstance(logprobs, np.ndarray)        
        assert isinstance(values, np.ndarray)        

        assert action.dtype == np.float32
        assert logprobs.dtype == np.float32
        assert values.dtype == np.float32

        assert action.shape == (64, 10)
        assert logprobs.shape == (64,)
        assert values.shape == (64,)

    def test_act(self, actor_critic, state_np) -> None:
        action = actor_critic.act(state_np)
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.float32
        assert action.shape == (1, 10)
        