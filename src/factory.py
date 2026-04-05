from argparse import Namespace

import gymnasium as gym
from gymnasium import spaces

from src.models.actor_critic_base import ActorCritic
from src.models.actor_critic_continuous import (
    ActorCriticContinuousMLP,
    ActorCriticContinuousCNN
)
from src.models.actor_critic_discrete import (
    ActorCriticDiscreteMLP,
    ActorCriticDiscreteCNN
)
from src.wrappers import ActionRepeat


class Factory:
    def __init__(self, env: gym.Env, n_envs: int=1) -> None:
        self.env_id = env.spec.id
        self.observation_space = env.observation_space
        self.obs_shape = env.observation_space.shape 
        self.action_space = env.action_space
        self.n_envs = n_envs

        self.action_repeat = None
        self.frame_stack = None
        
        curr = env
        while hasattr(curr, "env"):
            if hasattr(curr, "k_repeats"):
                self.action_repeat = curr.k_repeats
            
            if hasattr(curr, "stack_size"): 
                self.frame_stack = curr.stack_size

            curr = curr.env
    
    def make_env(self, render_mode: str|None=None) -> gym.Env:
        if len(self.obs_shape) > 1: 
            env = gym.make(self.env_id, render_mode=render_mode)
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, stack_size=self.frame_stack)
            env = ActionRepeat(env, self.action_repeat) 
            return env  
        env = gym.make(self.env_id, render_mode=render_mode)
        return env

    def make_vec_env(self) -> None:
        """Vectorizes the given env.""" 
        vec_env = gym.vector.SyncVectorEnv(
            [lambda env_id=self.env_id: self.make_env() for _ in range(self.n_envs)]
        )
        return vec_env

    def get_env_id(self) -> str:
        return self.env_id


def make_env(env_id: str, render_mode: str|None=None, frame_stack: int=4, action_repeat: int=4) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    obs_shape = env.observation_space.shape
    if len(obs_shape) > 1: 
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
        env = ActionRepeat(env, action_repeat)
        return env  
    return env


def make_actor_critic(env: gym.Env, args: Namespace) -> ActorCritic:
    obs_space = env.observation_space
    act_space = env.action_space
    obs_shape = obs_space.shape

    is_discrete = isinstance(act_space, spaces.Discrete)
    is_continuous = isinstance(act_space, spaces.Box)

    if not (is_discrete or is_continuous):
        raise ValueError(f"Unsupported action space: {act_space}")

    # CNN path
    if len(obs_shape) > 1:
        if not isinstance(obs_space, spaces.Box):
            raise ValueError(f"CNN expects Box observation space, got: {obs_space}")

        if is_discrete:
            return ActorCriticDiscreteCNN(
                obs_shape=obs_shape,
                action_dim=act_space.n,
                fc_in_dim=args.cnn_fc_in_dim
            )

        return ActorCriticContinuousCNN(
            obs_shape=obs_shape,
            action_dim=act_space.shape[0],
            action_scale=float(act_space.high[0]),
            fc_in_dim=args.cnn_fc_in_dim
        )

    # MLP path
    if not isinstance(obs_space, spaces.Box):
        raise ValueError(f"MLP expects Box observation space, got: {obs_space}")

    if len(obs_space.shape) != 1:
        raise ValueError(
            f"MLP expects 1D observations, got shape: {obs_space.shape}. "
            f"Use --use_cnn for image observations."
        )

    state_dim = obs_space.shape[0]

    if is_discrete:
        return ActorCriticDiscreteMLP(
            state_dim=state_dim,
            action_dim=act_space.n,
            h1_dim=args.h1_dim,
            h2_dim=args.h2_dim,
        )

    return ActorCriticContinuousMLP(
        state_dim=state_dim,
        action_dim=act_space.shape[0],
        h1_dim=args.h1_dim,
        h2_dim=args.h2_dim,
        action_scale=float(act_space.high[0]),
    )
