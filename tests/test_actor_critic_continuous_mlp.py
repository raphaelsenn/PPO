import torch
import numpy as np

import pytest
from src.actor_critic_continuous import ActorCriticContinuousMLP


@pytest.fixture
def actor_critic() -> ActorCriticContinuousMLP:
    return ActorCriticContinuousMLP(784, 256, 256, 10, 1.0)


@pytest.fixture
def state_batch() -> torch.Tensor:
    return torch.randn(64, 784, dtype=torch.float32)


@pytest.fixture
def state() -> torch.Tensor:
    return torch.randn(784, dtype=torch.float32)


@pytest.fixture
def state_np() -> np.ndarray:
    return np.random.rand(784)


class TestActorCriticContinuousMLP:
    def test_init(self, actor_critic: ActorCriticContinuousMLP) -> None:
        assert actor_critic.obs_shape == (784,)
        assert actor_critic.state_dim == 784
        assert actor_critic.action_dim == 10
        assert actor_critic.is_continuous is True
        assert actor_critic.action_scale == 1.0
    
    def test_forward(self, actor_critic: ActorCriticContinuousMLP, state_batch: torch.Tensor) -> None:
        mu, std, values = actor_critic(state_batch)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert isinstance(values, torch.Tensor)

        assert mu.shape == (64, 10)
        assert std.shape == (64, 10)
        assert values.shape == (64, 1)
    
    def test_sample(self, actor_critic: ActorCriticContinuousMLP, state_batch: torch.Tensor) -> None:
        action, logprobs, values = actor_critic.sample(state_batch) 

        assert isinstance(action, np.ndarray)        
        assert isinstance(logprobs, np.ndarray)        
        assert isinstance(values, np.ndarray)        

        assert action.dtype == np.float32
        assert logprobs.dtype == np.float32
        assert values.dtype == np.float32

        assert action.shape == (64, 10)
        assert logprobs.shape == (64,)
        assert values.shape == (64, 1)

    def test_act(self, actor_critic, state_np) -> None:
        action = actor_critic.act(state_np)
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.float32
        assert action.shape == (1, 10)
        