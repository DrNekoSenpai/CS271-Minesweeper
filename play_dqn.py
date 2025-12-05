"""Play a trained dueling DQN agent against the Minesweeper environment.

This script loads a checkpoint saved by `agents.dueling_cnn_agent.Agent.save`
and runs episodes using `Agent.select_action`. It's intended for evaluation /
visual play. The checkpoint should contain the `policy` state dict.

Usage:
  python play_dqn.py --model ./dueling_agent.pth --episodes 10 --render
"""

import argparse
import os
import time

import numpy as np

from backend.environment import MinesweeperEnv
from agents.dueling_cnn_agent import Agent


def _decode_action(action: int, height: int, width: int):
    # same decoding convention used elsewhere in the repo
    z, rem = divmod(action, height * width)
    y, x = divmod(rem, width)
    return int(z), int(y), int(x)


def play(model_path: str, episodes: int = 10, render: bool = True, env_kwargs: dict = None, out_csv: str = None):
    env_kwargs = env_kwargs or {"height": 5, "width": 5, "depth": 5, "num_mines": 15, "render_mode": "ansi"}

    env = MinesweeperEnv(**env_kwargs)
    agent = Agent(env.action_space)

    # Build networks to match observation shape before loading weights
    obs, _ = env.reset()
    if agent.policy_net is None:
        agent._build_networks(obs.shape)

    # Load weights
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file not found: {model_path}. Running with untrained policy.")

    # CSV header
    if out_csv:
        write_header = not os.path.exists(out_csv)
        if write_header:
            with open(out_csv, "w") as f:
                f.write("episode,total_reward,steps,good_moves,mine_hits,win\n")

    # aggregate stats
    total_rewards = []
    total_wins = 0
    total_good_moves = 0
    total_mine_hits = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        good_moves = 0
        mine_hits = 0
        start = time.time()

        while not done:
            if render:
                print(env.render())

            action = agent.select_action(obs)
            z, y, x = _decode_action(action, env.height, env.width)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Good move = any positive reward (environment awards number of newly revealed tiles)
            if reward > 0:
                good_moves += 1
            # Mine hit detection: environment encodes revealed mine as -10 at the revealed cell
            elif reward < 0 and next_obs[z, y, x] == -10:
                mine_hits += 1

            obs = next_obs
            total_reward += reward
            steps += 1

            # safety cap
            if steps > 2000:
                break

        elapsed = time.time() - start
        win = getattr(env.game, "win", False)
        total_rewards.append(total_reward)
        total_wins += 1 if win else 0
        total_good_moves += good_moves
        total_mine_hits += mine_hits

        print(f"Episode {ep:3d} | reward {total_reward:7.1f} | steps {steps:3d} | good_moves {good_moves:3d} | mine_hits {mine_hits:3d} | win {win} | time {elapsed:.2f}s")
        print(env.render())

        if out_csv:
            with open(out_csv, "a") as f:
                f.write(f"{ep},{total_reward},{steps},{good_moves},{mine_hits},{int(win)}\n")

    # summary
    avg_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    win_rate = total_wins / episodes if episodes > 0 else 0.0
    print("\n=== SUMMARY ===")
    print(f"Episodes: {episodes}")
    print(f"Win rate: {win_rate:.2%} ({total_wins}/{episodes})")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Total good moves: {total_good_moves}")
    print(f"Total mine hits: {total_mine_hits}")
    if out_csv:
        print(f"Per-episode results appended to {out_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="dueling_agent.pth", help="Path to model checkpoint")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--height", type=int, default=5)
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--num-mines", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env_kwargs = {"height": args.height, "width": args.width, "depth": args.depth, "num_mines": args.num_mines, "render_mode": "ansi"}
    play(args.model, episodes=args.episodes, render=args.render, env_kwargs=env_kwargs)
