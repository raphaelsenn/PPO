from argparse import Namespace, ArgumentParser

import numpy as np
import torch

from src import PPO
from src import make_env, make_actor_critic


def parse_args() -> Namespace:
    """Settings for LunarLander-v3"""
    parser = ArgumentParser(description="PPO training")

    parser.add_argument("--env_id", type=str, default="LunarLander-v3")
    parser.add_argument("--num_timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--decay_learning_rate", type=bool, default=True)   # Decays learning rate from `learing_rate` to 0.0

    parser.add_argument("--norm_advantages", type=bool, default=True)
    parser.add_argument("--clip_grad_norm", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--obs_scale", type=float, default=None)
    parser.add_argument("--reward_clip", type=float, default=None)

    # MLP settings
    parser.add_argument("--h1_dim", type=int, default=256)
    parser.add_argument("--h2_dim", type=int, default=256)

    # CNN settings
    parser.add_argument("--cnn_fc_in_dim", type=int, default=4096)          # Only for CNN
    parser.add_argument("--frame_stack", type=int, default=4)               # Only for CNN
    parser.add_argument("--action_repeat", type=int, default=1)             # Only for CNN

    # Logging stuff
    parser.add_argument("--n_eval_runs", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--verbose", type=bool, default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    env = make_env(args.env_id, None, args.frame_stack, args.action_repeat)
    actor_critic = make_actor_critic(env, args)

    ppo = PPO(
        actor_critic=actor_critic,
        n_envs=args.n_envs,
        device=args.device,
        time_steps=args.num_timesteps,
        horizon=args.horizon,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        entropy_coef=args.entropy_coef,
        norm_advantages=args.norm_advantages,
        clip_grad_norm=args.clip_grad_norm,
        weight_decay=args.weight_decay,
        obs_scale=args.obs_scale,
        reward_clip=args.reward_clip,
        decay_learning_rate=args.decay_learning_rate,
        n_eval_runs=args.n_eval_runs,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        verbose=args.verbose,
    )

    ppo.train(env)


if __name__ == "__main__":
    main()