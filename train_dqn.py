"""Simple training harness for the dueling DQN agent.

This script runs episodes of MinesweeperEnv and trains
`agents.dueling_cnn_agent.Agent` using its `remember` and `optimize`
methods. It's intentionally minimal — intended for quick experiments
and smoke tests.

Usage examples:
  python train_dqn.py --episodes 100 --save-path ./dueling_agent.pth
  python train_dqn.py --episodes 2 --no-render
"""

import argparse
import os
import time

import numpy as np

from backend.environment import MinesweeperEnv

try:
    from agents.dueling_cnn_agent import Agent
except Exception as e:
    raise


def train(
    episodes: int = 100,
    env_kwargs: dict = None,
    save_path: str = "dueling_agent.pth",
    save_interval: int = 50,
    render: bool = True,
):
    env_kwargs = env_kwargs or {"height": 5, "width": 5, "depth": 5, "num_mines": 15, "render_mode": "ansi"}

    env = MinesweeperEnv(**env_kwargs)
    agent = Agent(env.action_space)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        start = time.time()

        while not done:
            if render:
                print(env.render())

            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # store experience and train
            agent.remember(obs, action, reward, next_obs, done)
            agent.optimize(num_iters=1)

            obs = next_obs
            total_reward += reward
            steps += 1

            # sanity cap (avoid pathological infinite loops)
            if steps > 1000:
                break

        elapsed = time.time() - start
        print(f"Episode {ep:4d} | reward {total_reward:6.1f} | steps {steps:3d} | time {elapsed:.2f}s")

        if ep % save_interval == 0 or ep == episodes:
            try:
                agent.save(save_path)
                print(f"Saved checkpoint to {save_path}")
            except Exception as e:
                print("Failed to save checkpoint:", e)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--save-path", type=str, default="dueling_agent.pth")
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--height", type=int, default=5)
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--num-mines", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env_kwargs = {"height": args.height, "width": args.width, "depth": args.depth, "num_mines": args.num_mines, "render_mode": "ansi"}
    train(episodes=args.episodes, env_kwargs=env_kwargs, save_path=args.save_path, save_interval=args.save_interval, render=args.render)
