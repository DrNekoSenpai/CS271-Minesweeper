import argparse
import numpy as np
import torch
import time
import os
from tqdm import tqdm
from backend.environment import MinesweeperEnv
from agents.dueling_cnn_agent import DuelingDCNNAgent


def load_checkpoint(agent, checkpoint_path):
    """Load model weights from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    payload = torch.load(checkpoint_path, map_location=agent.device)
    
    if isinstance(payload, dict) and "online_state" in payload:
        agent.online.load_state_dict(payload["online_state"])
        if "target_state" in payload:
            agent.target.load_state_dict(payload["target_state"])
        else:
            agent.target.load_state_dict(agent.online.state_dict())
    elif isinstance(payload, dict) and "state_dict" in payload:
        agent.online.load_state_dict(payload["state_dict"])
        agent.target.load_state_dict(agent.online.state_dict())
    else:
        agent.online.load_state_dict(payload)
        agent.target.load_state_dict(agent.online.state_dict())
    
    agent.target.eval()
    print(f"✓ Loaded checkpoint: {checkpoint_path}")


def run_episode(env, agent):
    """Run a single episode and return results."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    moves = 0
    
    while not done:
        # Use agent's built-in action selection with hybrid mode enabled
        action = agent.select_action(obs, use_hybrid=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        moves += 1
        done = terminated or truncated
    
    # Win if final reward is exactly 100 (all non-mine tiles revealed)
    won = (reward == 100.0)
    
    return {
        "reward": total_reward,
        "moves": moves,
        "won": won
    }


def print_statistics(rewards, moves, wins, losses, elapsed_time, num_episodes):
    """Print statistics in the standard format."""
    rewards = np.array(rewards, dtype=np.float32)
    moves = np.array(moves, dtype=np.float32)
    
    print("\nTotal Reward:")
    print(f"  min     = {rewards.min():.2f}")
    print(f"  max     = {rewards.max():.2f}")
    print(f"  mean    = {rewards.mean():.2f}")
    print(f"  median  = {np.median(rewards):.2f}")
    print(f"  var     = {rewards.var():.2f}")
    print(f"  stdev   = {rewards.std():.2f}")
    
    print(f"\nNumber of Moves:")
    print(f"  min     = {moves.min():.2f}")
    print(f"  max     = {moves.max():.2f}")
    print(f"  mean    = {moves.mean():.2f}")
    print(f"  median  = {np.median(moves):.2f}")
    print(f"  var     = {moves.var():.2f}")
    print(f"  stdev   = {moves.std():.2f}")
    
    print(f"\nWin / Loss:")
    print(f"  Wins   = {wins}")
    print(f"  Losses = {losses}")
    print(f"  Win rate = {wins / num_episodes:.3f}")
    
    print(f"\nTiming:")
    print(f"  Total time   = {elapsed_time:.2f} seconds")
    print(f"  Avg per ep   = {elapsed_time / num_episodes:.4f} seconds")
    print(f"  Episodes/sec = {num_episodes / elapsed_time:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN agent on Minesweeper")
    parser.add_argument("--size", type=int, default=5, help="Board size (cube dimension)")
    parser.add_argument("--mines", type=int, default=8, help="Number of mines")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to run")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (auto-selects latest if not provided)")
    args = parser.parse_args()
    
    # Auto-select checkpoint if not provided
    if args.checkpoint is None:
        checkpoint_pattern = f"dqn-checkpoint-s{args.size}-m{args.mines}"
        checkpoints = sorted(
            [f for f in os.listdir(".") if checkpoint_pattern in f and f.endswith(".pth")],
            key=lambda x: int(x.split("-")[-1].split(".")[0])
        )
        if checkpoints:
            args.checkpoint = checkpoints[-1]
        else:
            pretrained = f"dqn-pretrained-s{args.size}-m{args.mines}.pth"
            if os.path.exists(pretrained):
                args.checkpoint = pretrained
            else:
                raise FileNotFoundError(f"No checkpoint found for size={args.size}, mines={args.mines}")
    
    print(f"Configuration:")
    print(f"  Board: {args.size}x{args.size}x{args.size}")
    print(f"  Mines: {args.mines}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Checkpoint: {args.checkpoint}")
    
    # Initialize environment and agent
    env = MinesweeperEnv(
        height=args.size,
        width=args.size,
        depth=args.size,
        num_mines=args.mines,
        render_mode=None
    )
    
    agent = DuelingDCNNAgent(
        height=args.size,
        width=args.size,
        depth=args.size
    )
    
    load_checkpoint(agent, args.checkpoint)
    
    # Use EXACT training configuration to match training behavior
    # During training: safe_action_prob=0.3, epsilon decays to 0.05
    agent.total_steps = agent.eps_decay_steps  # Force epsilon to minimum (0.05)
    
    print(f"Evaluation mode: hybrid={agent.use_hybrid}, safe_prob={agent.safe_action_prob}, epsilon={agent.epsilon():.3f}")
    print(f"  (matching training configuration)")
    
    # Run evaluation
    print(f"\nRunning {args.episodes} episodes...")
    rewards = []
    moves_list = []
    wins = 0
    losses = 0
    
    start_time = time.time()
    
    for _ in tqdm(range(args.episodes)):
        result = run_episode(env, agent)
        rewards.append(result["reward"])
        moves_list.append(result["moves"])
        
        if result["won"]:
            wins += 1
        else:
            losses += 1
    
    elapsed_time = time.time() - start_time
    
    # Print statistics
    print_statistics(rewards, moves_list, wins, losses, elapsed_time, args.episodes)


if __name__ == "__main__":
    main()
