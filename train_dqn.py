import argparse
import numpy as np
import gymnasium as gym
import os
import matplotlib.pyplot as plt 
import re 
import pickle

from agents.dueling_cnn_agent import DuelingDCNNAgent
from backend.environment import MinesweeperEnv

def make_env(depth, height, width, num_mines):
    def _thunk():
        return MinesweeperEnv(
            depth=depth,
            height=height,
            width=width,
            num_mines=num_mines,
            render_mode=None
        )
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    # Vectorized envs
    envs = gym.vector.SyncVectorEnv([
        make_env(5, 5, 5, 15) for _ in range(args.num_envs)
    ])

    obs, info = envs.reset()  # obs shape: (num_envs, H, W, D)

    agent = DuelingDCNNAgent(
        height=5, width=5, depth=5,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup=10000,
        target_update=2000, 
        buffer_size=500000
    )

    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    episode_safe_moves = np.zeros(args.num_envs, dtype=np.float32)
    episode_safe_tiles = np.zeros(args.num_envs, dtype=np.float32)
    completed = 0

    checkpoint_path = "dqn-checkpoint"
    pattern = re.compile(rf"{checkpoint_path}-(\d+)\.pth")
    candidates = []
    for fname in os.listdir("."):
        m = pattern.match(fname)
        if m:
            candidates.append((int(m.group(1)), fname))

    load_path = candidates[-1][1] if candidates else f"{checkpoint_path}-0.pth"
    save_every = 2500
    start_step = 0
    metrics = []

    if not args.fresh and os.path.exists(load_path): 
        start_step = agent.load_checkpoint(load_path)
        print(f"Loaded checkpoint {load_path} from step {start_step}")

    else: 
        with open("metrics.log", "w", encoding="utf-8") as file: 
            file.write("")

    for step in range(start_step, args.steps):
        actions = agent.select_actions(obs)

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        done = np.logical_or(terminated, truncated)

        safe_batch = infos.get("safe_move", np.zeros_like(rewards))
        tile_batch = infos.get("safe_tiles", np.zeros_like(rewards))

        # Store transitions
        for i in range(args.num_envs):
            agent.remember(obs[i], int(actions[i]), float(rewards[i]), next_obs[i], bool(done[i]))

            episode_rewards[i] += rewards[i]
            episode_safe_moves[i] += safe_batch[i]
            episode_safe_tiles[i] += tile_batch[i]

            if done[i]:
                completed += 1
                if completed % 100 == 0: 
                    print(f"[step={step}] reward={np.mean(episode_rewards[i]):.1f} safe_moves={np.mean(episode_safe_moves[i]):.1f} safe_tiles={np.mean(episode_safe_tiles[i]):.1f}")
                    metrics.append(f"[step={step}] reward={np.mean(episode_rewards[i]):.1f} safe_moves={np.mean(episode_safe_moves[i]):.1f} safe_tiles={np.mean(episode_safe_tiles[i]):.1f}")

                episode_rewards[i] = 0.0
                episode_safe_moves[i] = 0.0
                episode_safe_tiles[i] = 0.0

        obs = next_obs

        # Train
        stats = agent.optimize()

        if step % save_every == 0 and step != start_step: 
            save_path = f"{checkpoint_path}-{step}.pth"
            agent.save_checkpoint(save_path, step)

            print(f"Saved checkpoint: {save_path}")
            with open('metrics.log', 'a', encoding='utf-8') as file: 
                for m in metrics: 
                    file.write(f"{m}\n")
                file.write(f"Saved checkpoint: {save_path}\n")
                metrics = []

            pattern = re.compile(rf"{checkpoint_path}-(\d+)\.pth")

            # Delete all other checkpoints in directory that are not the most recent
            for fname in os.listdir("."): 
                match = pattern.match(fname)
                if not match: continue 

                file_step = int(match.group(1))
                if file_step < step: os.remove(fname)

    agent.save(f"dueling_cnn.pt")
    print(f"Saved model to dueling_cnn.pt")

    # plt.figure(figsize=(10,6))

    # plt.xlabel("Log Step (index x 50)")
    # plt.ylabel("Average Reward")
    # plt.title("Training Reward Curve")
    # plt.grid(True)
    # plt.legend()

    # plt.savefig("train_results.png", dpi=600)
    # print("Saved reward curve to train_results.png")

if __name__ == "__main__":
    main()
