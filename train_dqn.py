import argparse
import numpy as np
import gymnasium as gym
import os
import matplotlib.pyplot as plt 
import re 

from agents.dueling_cnn_agent import DuelingDCNNAgent
from backend.environment import MinesweeperEnv
from matplotlib.ticker import MaxNLocator

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

def main(size:int, mines:int):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    directory = f"./metrics/s{size}-m{mines}"
    if not os.path.exists(directory): os.makedirs(directory)

    # Vectorized envs
    envs = gym.vector.SyncVectorEnv([
        make_env(size, size, size, mines) for _ in range(args.num_envs)
    ])

    obs, info = envs.reset()  # obs shape: (num_envs, H, W, D)

    agent = DuelingDCNNAgent(
        height=size, width=size, depth=size,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup=5000,
        target_update=1000, 
        buffer_size=200000, 
        eps_decay_steps=150000
    )

    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    episode_safe_moves = np.zeros(args.num_envs, dtype=np.float32)
    episode_safe_tiles = np.zeros(args.num_envs, dtype=np.float32)
    completed = 0

    checkpoint_path = f"dqn-checkpoint-s{size}-m{mines}"
    checkpoints = sorted([f for f in os.listdir(".") if checkpoint_path in f and f.endswith(".pth")], key=lambda x: int(x.split("-")[-1].split(".")[0]))
    load_path = checkpoints[-1] if checkpoints else f"dqn-checkpoint-0.pth"

    save_every = 5000
    start_step = 0
    logs = []

    metrics = {
        "steps": [], 
        "reward": [], 
        "safe_moves": [], 
        "safe_tiles": []
    }

    if not args.fresh and os.path.exists(load_path): 
        start_step = agent.load_checkpoint(load_path)
        print(f"Loaded checkpoint {load_path} from step {start_step}")

        with open("metrics.log", "r", encoding="utf-8") as file: 
            lines = file.readlines() 
        
        metrics_pattern = r"\[step=(\d+)\] reward=(.*) safe_moves=(.*) safe_tiles=(.*)"
        for line in lines: 
            match = re.search(metrics_pattern, line)
            if match: 
                steps, reward, safe_moves, safe_tiles = match.groups()
                metrics["steps"].append(int(steps))
                metrics["reward"].append(float(reward))
                metrics["safe_moves"].append(float(safe_moves))
                metrics["safe_tiles"].append(float(safe_tiles))

    else: 
        with open(f"./metrics/s{size}-m{mines}/metrics.log", "w", encoding="utf-8") as file: 
            file.write("")

        with open(f"./metrics/s{size}-m{mines}/loss.log", "w", encoding="utf-8") as file: 
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
                    logs.append((step, np.mean(episode_rewards[i]), np.mean(episode_safe_moves[i]), np.mean(episode_safe_tiles[i])))

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
            with open(f'./metrics/s{size}-m{mines}/metrics.log', 'a', encoding='utf-8') as file: 
                for l in logs: 
                    s, reward, safe_moves, safe_tiles = l
                    file.write(f"[step={s}] reward={reward} safe_moves={safe_moves} safe_tiles={safe_tiles}\n")
                    if safe_moves > safe_tiles: print("Warning: moves greater than tiles revealed")

                    metrics["steps"].append(s)
                    metrics["reward"].append(reward)
                    metrics["safe_moves"].append(safe_moves)
                    metrics["safe_tiles"].append(safe_tiles)
                    
                file.write(f"Saved checkpoint: {save_path}\n")
                logs = []

            with open(f'./metrics/s{size}-m{mines}/loss.log', 'a', encoding="utf-8") as file: 
                file.write(f"[step={step}]: {stats['loss']:.6f}\n")

            # Remove old checkpoints and models
            pattern = re.compile(rf"{checkpoint_path}-(\d+)\.pth")
            for fname in os.listdir("."): 
                match = pattern.match(fname)
                if not match: continue 

                file_step = int(match.group(1))
                if file_step < step: os.remove(fname)

            plt.figure(figsize=(10, 6))
            plt.plot(metrics["steps"], metrics["reward"], label="Rewards", linestyle="-")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.xlabel("Step")
            plt.ylabel("Average Reward")
            plt.title("Reward Curve")
            plt.grid(True)
            plt.savefig(f"./metrics/s{size}-m{mines}/rewards-{step}.png", dpi=600)
            plt.close()
            print(f"Saved reward curve to rewards-{step}.png")

            plt.figure(figsize=(10, 6))
            plt.plot(metrics["steps"], metrics["safe_moves"], label="Safe Moves", linestyle="-")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.xlabel("Step")
            plt.ylabel("Average Safe Moves")
            plt.title("Safe Moves")
            plt.grid(True)
            plt.savefig(f"./metrics/s{size}-m{mines}/safe_moves-{step}.png", dpi=600)
            plt.close()
            print(f"Saved safe moves curve to safe_moves-{step}.png")

            plt.figure(figsize=(10, 6))
            plt.plot(metrics["steps"], metrics["safe_tiles"], label="Safe Tiles", linestyle="-")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.xlabel("Step")
            plt.ylabel("Safe Tiles")
            plt.title("Safe Tiles")
            plt.grid(True)
            plt.savefig(f"./metrics/s{size}-m{mines}/safe_tiles-{step}.png", dpi=600)
            plt.close()
            print(f"Saved safe tiles curve to safe_tiles-{step}.png")

            pattern = re.compile(r"(rewards|safe_moves|safe_tiles)-(\d+)\.png")
            for fname in os.listdir(f"./metrics/s{size}-m{mines}/"): 
                match = pattern.match(fname)
                if not match: continue 

                file_step = int(match.group(2))
                if file_step < step: os.remove(f"./metrics/s{size}-m{mines}/{fname}")

            # pattern = re.compile(rf"{checkpoint_path}-(\d+)\.pth")
            # for fname in os.listdir("."): 
            #     match = pattern.match(fname)
            #     if not match: continue 

            #     file_step = int(match.group(1))
            #     if file_step < step: os.remove(fname)

if __name__ == "__main__":
    # main(size=4, mines=5)
    main(size=5, mines=5)
    main(size=5, mines=10)