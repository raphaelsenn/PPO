import torch
import numpy as np

import pytest
from src import ActorCriticDiscreteMLP


@pytest.fixture
def actor_critic() -> ActorCriticDiscreteMLP:
    return ActorCriticDiscreteMLP(784, 256, 256, 10)


@pytest.fixture
def state_batch() -> torch.Tensor:
    return torch.randn(64, 784, dtype=torch.float32)


@pytest.fixture
def state() -> torch.Tensor:
    return torch.randn(784, dtype=torch.float32)


@pytest.fixture
def state_np() -> np.ndarray:
    return np.random.rand(784)


class TestActorCriticDiscreteMLP:
    def test_init(self, actor_critic: ActorCriticDiscreteMLP) -> None:
        assert actor_critic.obs_shape == (784,)
        assert actor_critic.state_dim == 784
        assert actor_critic.action_dim == 10
        assert actor_critic.is_continuous is False
        assert actor_critic.action_scale is None
    
    def test_forward(self, actor_critic: ActorCriticDiscreteMLP, state_batch: torch.Tensor) -> None:
        logits, values = actor_critic(state_batch)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(values, torch.Tensor)

        assert logits.shape == (64, 10)
        assert values.shape == (64, 1)
    
    def test_sample(self, actor_critic: ActorCriticDiscreteMLP, state_batch: torch.Tensor) -> None:
        action, logprobs, values = actor_critic.sample(state_batch) 

        assert isinstance(action, np.ndarray)        
        assert isinstance(logprobs, np.ndarray)        
        assert isinstance(values, np.ndarray)        

        assert action.dtype == np.int64
        assert logprobs.dtype == np.float32
        assert values.dtype == np.float32

        assert action.shape == (64,)
        assert logprobs.shape == (64,)
        assert values.shape == (64,)

    def test_act(self, actor_critic, state_np) -> None:
        action = actor_critic.act(state_np)
        assert isinstance(action, np.ndarray)
        assert action.dtype == np.int64
        assert action.shape == (1,)
        