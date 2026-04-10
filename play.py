from argparse import Namespace, ArgumentParser

import torch

import gymnasium as gym

from src import ActorCritic
from src.factory import make_env, make_actor_critic


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Online Evaluation")

    parser.add_argument("--env_id", type=str, default="BipedalWalker-v3")
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--h1_dim", type=int, default=256)
    parser.add_argument("--h2_dim", type=int, default=256)
    parser.add_argument("--cnn_fc_in_dim", type=int, default=4096)
    # parser.add_argument("--weights", type=str, default="CarRacing-v3-PPO-ActorCritic-Lr0.0003-t3555328-Seed0.pt")
    parser.add_argument("--weights", type=str, default="BipedalWalker-v3-PPO-Checkpoints-Seed0/BipedalWalker-v3-PPO-ActorCritic-Lr0.0003-t1000000-Seed0.pt")
    # parser.add_argument("--weights", type=str, default="LunarLander-v3-PPO-Checkpoints-Seed5/LunarLander-v3-PPO-ActorCritic-Lr0.0003-t1000000-Seed5.pt")

    parser.add_argument("--verbose", default=True)

    return parser.parse_args()


def play(env: gym.Env, actor: ActorCritic, n_episodes: int=100) -> None:
    for ep in range(n_episodes):
        done = False
        s, _ = env.reset()
        t = 0 
        reward_sum = 0.0
        while not done:
            #a = env.action_space.sample() 
            a = actor.act(s, deterministic=True)
            a = a.flatten()
            #if len(a) == 1: a = a.item() 
            #else: a = a.flatten()

            s_nxt, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = s_nxt
            reward_sum += reward
            t += 1 
    env.close()


def main() -> None:
    args = parse_args()

    env = make_env(args.env_id, render_mode="human")
    actor_critic = make_actor_critic(env, args)
    actor_critic.load_state_dict(torch.load(args.weights, weights_only=True, map_location="cpu")) 
    play(env, actor_critic)


if __name__ == "__main__":
    main()