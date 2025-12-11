"""
Batch-evaluate a trained Dueling Double-DQN 3D CNN Minesweeper agent.

This rewrite intentionally matches the output/print structure of
run_agent_batch.py:
  - progress update every 100 episodes
  - summary blocks with min/max/mean/median/var
  - Win / Loss counts and win rate

The loader supports:
  1) Training checkpoints saved via DuelingDCNNAgent.save_checkpoint
     (expects key: "online_state")
  2) A raw state_dict (if you later save inference-only weights)

Usage:
  python play_dqn.py --model checkpoint_step_50000.pth --episodes 1000
  python play_dqn.py --model dueling_cnn.pt --episodes 200 --render
"""

import argparse
import os
import numpy as np
import time
import torch

from backend.environment import MinesweeperEnv
from agents.dueling_cnn_agent import DuelingDCNNAgent
# from tqdm import tqdm

def summarize(values, name):
    values = np.array(values, dtype=np.float32)
    print(f"\n{name}:")
    print(f"  min     = {values.min():.2f}")
    print(f"  max     = {values.max():.2f}")
    print(f"  mean    = {values.mean():.2f}")
    print(f"  median  = {np.median(values):.2f}")
    print(f"  var     = {values.var():.2f}")
    print(f"  stdev   = {values.std():.2f}")


def load_weights(agent: DuelingDCNNAgent, path: str) -> bool:
    """
    Robust loader:
      - If file is a checkpoint dict with "online_state", load it.
      - Else try to treat file as a raw state_dict.
    """
    if not os.path.exists(path):
        return False

    payload = torch.load(path, map_location=agent.device)

    # Case 1: checkpoint-style dict
    if isinstance(payload, dict) and "online_state" in payload:
        agent.online.load_state_dict(payload["online_state"])

        # Prefer target_state if present; otherwise mirror online
        if "target_state" in payload:
            agent.target.load_state_dict(payload["target_state"])
        else:
            agent.target.load_state_dict(agent.online.state_dict())

        agent.target.eval()
        return True

    # Case 2: maybe someone saved {"state_dict": ...}
    if isinstance(payload, dict) and "state_dict" in payload:
        agent.online.load_state_dict(payload["state_dict"])
        agent.target.load_state_dict(agent.online.state_dict())
        agent.target.eval()
        return True

    # Case 3: raw state_dict
    try:
        agent.online.load_state_dict(payload)
        agent.target.load_state_dict(agent.online.state_dict())
        agent.target.eval()
        return True
    except Exception:
        return False

def select_action_greedy(agent: DuelingDCNNAgent, obs: np.ndarray) -> int:
    """
    Deterministic masked argmax policy.
    Avoids epsilon exploration during evaluation.
    """
    legal = agent.legal_action_indices(obs)
    if legal.size == 0:
        return int(np.random.randint(0, agent.n_actions))

    obs_b = np.expand_dims(obs, axis=0)  # (1,H,W,D)
    x = agent.obs_to_tensor(obs_b)

    with torch.no_grad():
        q = agent.online(x)[0]  # (n_actions,)

        mask = torch.zeros(agent.n_actions, dtype=torch.bool, device=agent.device)
        mask[torch.from_numpy(legal).to(agent.device)] = True

        q_masked = q.clone()
        q_masked[~mask] = -1e9

        return int(torch.argmax(q_masked).item())

def run_episode(env: MinesweeperEnv, agent: DuelingDCNNAgent, render: bool = False, seed: int | None = None):
    obs, _ = env.reset(seed=seed)
    done = False

    total_reward = 0.0
    total_moves = 0
    won = True

    while not done:
        if render and env.render_mode == "ansi":
            print(env.render())

        action = select_action_greedy(agent, obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_moves += 1

        # any negative reward implies a mine hit / loss path.
        if reward < 0:
            won = False

        done = terminated or truncated

    return {
        "reward": total_reward,
        "moves": total_moves,
        "won": won,
        "seed": seed,
    }

def animate_camera(env: MinesweeperEnv, frame_idx: int, num_frames:int):
    plotter = getattr(env, "_plotter", None)
    if plotter is None:
        return  # plotter not created yet

    step_deg = 0.25
    plotter.camera.azimuth += step_deg

def save_win_frames(
    size: int,
    mines: int,
    agent: DuelingDCNNAgent,
    seed: int,
    out_root: str,
    num_frames: int,
    frames_per_step: int = 5,
):
    # Unique folder per saved win
    out_dir = os.path.join(out_root, f"s{size}_m{mines}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    env3d = MinesweeperEnv(
        height=size,
        width=size,
        depth=size,
        num_mines=mines,
        render_mode="3d",
    )

    obs, _ = env3d.reset(seed=seed)

    done = False
    frame_idx = 0  # <--- NEW: global frame counter over the whole episode

    # Capture initial state (before any move)
    env3d.render_frame(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"), off_screen=True)
    frame_idx += 1

    while not done:
        action = select_action_greedy(agent, obs)
        obs, reward, terminated, truncated, info = env3d.step(action)

        # For smoother playback, emit multiple frames for this single logical step
        for _ in range(frames_per_step):
            animate_camera(env3d, frame_idx, num_frames)
            env3d.render_frame(
                os.path.join(out_dir, f"frame_{frame_idx:04d}.png"),
                off_screen=True,
            )
            frame_idx += 1

        done = terminated or truncated

    # Best-effort cleanup
    try:
        if env3d._plotter is not None:
            env3d._plotter.close()
    except Exception:
        pass

    print(f"Saved winning 3D frames to: {out_dir}, total frames: {frame_idx}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--render", action="store_true", help="Print ANSI board each step (keeps batch summary format)")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda (defaults to agent auto-choice)")
    return p.parse_args()

def main(size:int, mines:int):
    # Delete all previous win_frames output with the matching size/mines
    # i.e. win_frames/s4_m5_seed12345

    out_root = "./win_frames"
    if os.path.exists(out_root):
        import shutil 
        for entry in os.listdir(out_root):
            if entry.startswith(f"s{size}_m{mines}_"):
                shutil.rmtree(os.path.join(out_root, entry))

    args = parse_args()
    render_mode = "3d" if args.render else None

    env = MinesweeperEnv(
        height=size,
        width=size,
        depth=size,
        num_mines=mines,
        render_mode=render_mode
    )

    agent = DuelingDCNNAgent(
        height=size,
        width=size,
        depth=size,
        device=args.device
    )

    checkpoint_path = f"dqn-checkpoint-s{size}-m{mines}"
    checkpoints = sorted([f for f in os.listdir(".") if checkpoint_path in f and f.endswith(".pth")], key=lambda x: int(x.split("-")[-1].split(".")[0]))
    load_path = checkpoints[-1] if checkpoints else "dqn-checkpoint-0.pth"
    AGENT_NAME = load_path.split(".")[0]

    loaded = load_weights(agent, load_path)
    if loaded:
        print(f"Loaded model from {load_path}")
    else:
        print(f"WARNING: Could not load model from {load_path}. ")
        exit(1)

    rewards = []
    moves = []
    wins = 0
    losses = 0
    saved = 0
    max_saves = 1 if (size, mines) != (5, 10) else 5  # Save up to 5 wins for (5,10), else just 1

    print(f"\nRunning {args.episodes} episodes with agent '{AGENT_NAME}'...")

    start_time = time.time()

    for i in range(args.episodes):
        seed = int(np.random.randint(0, 2**31-1))
        result = run_episode(env, agent, render=args.render, seed=seed)

        rewards.append(result["reward"])
        moves.append(result["moves"])

        if result["won"]:
            wins += 1

            if saved < max_saves and result["moves"] >= 30:
                save_win_frames(size, mines, agent, seed, "./win_frames", result["moves"])
                saved += 1
            elif saved >= max_saves:
                break
        else:
            losses += 1

            if (size == 5 and mines == 10) and (saved < max_saves and result["moves"] >= 30):
                save_win_frames(size, mines, agent, seed, "./win_frames", result["moves"])
                saved += 1
            elif saved >= max_saves:
                break

        # Progress print every 100 episodes
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{args.episodes}")

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / max(1, wins + losses)

    # ---------- SUMMARY ----------
    summarize(rewards, "Total Reward")
    summarize(moves, "Number of Moves")

    print("\nWin / Loss:")
    print(f"  Wins   = {wins}")
    print(f"  Losses = {losses}")
    print(f"  Win rate = {wins / args.episodes:.3f}")

    print("\nTiming:")
    print(f"  Total time   = {total_time:.2f} seconds")
    print(f"  Avg per ep   = {avg_time:.4f} seconds")
    print(f"  Episodes/sec = {1.0 / avg_time:.2f}")

if __name__ == "__main__":
    main(4, 5)
    main(5, 5)
    main(5, 8)
    main(5, 10)