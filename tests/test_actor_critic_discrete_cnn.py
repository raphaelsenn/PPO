import math
import torch
import numpy as np

import pytest
from src.actor_critic_discrete import ActorCriticDiscreteCNN


@pytest.fixture
def actor_critic() -> ActorCriticDiscreteCNN:
    return ActorCriticDiscreteCNN((4, 84, 84), 10, 64*7*7, 512)


@pytest.fixture
def state_batch() -> torch.Tensor:
    return torch.randn((64, 4, 84, 84), dtype=torch.float32)


@pytest.fixture
def state() -> torch.Tensor:
    return torch.randn(1, 4, 84, 84, dtype=torch.float32)


@pytest.fixture
def state_np() -> np.ndarray:
    return np.random.rand(4, 84, 84)


class TestActorCriticDiscreteCNN:
    def test_init(self, actor_critic: ActorCriticDiscreteCNN) -> None:
        assert actor_critic.obs_shape == (4, 84, 84)
        assert actor_critic.state_dim == 4 * 84 * 84
        assert actor_critic.state_ch == 4
        assert actor_critic.action_dim == 10
        assert actor_critic.is_continuous is False
        assert actor_critic.action_scale is None
    
    def test_forward(self, actor_critic: ActorCriticDiscreteCNN, state_batch: torch.Tensor) -> None:
        logits, values = actor_critic(state_batch)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(values, torch.Tensor)

        assert logits.shape == (64, 10)
        assert values.shape == (64, 1)
    
    def test_sample(self, actor_critic: ActorCriticDiscreteCNN, state_batch: torch.Tensor) -> None:
        action, logprobs, values = actor_critic.sample(state_batch) 

        assert isinstance(action, np.ndarray)        
        assert isinstance(logprobs, np.ndarray)        
        assert isinstance(values, np.ndarray)        

        assert action.dtype == np.int64
        assert logprobs.dtype == np.float32
        assert values.dtype == np.float32

        assert action.shape == (64,)
        assert logprobs.shape == (64,)
        assert values.shape == (64, 1)

    def test_act(self, actor_critic, state_np) -> None:
        action = actor_critic.act(state_np)
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.int64
        assert action.shape == (1,)
        