"""
Imitation Learning Bootstrap for DQN Agent (#2)

Pre-trains the DQN agent by filling the replay buffer with trajectories 
from the Bayesian agent (expert policy), then performing supervised learning
on this data before starting standard RL training.

This gives the DQN a warm start with good behavior patterns.
"""

import argparse
import numpy as np
from tqdm import tqdm

from agents.dueling_cnn_agent import DuelingDCNNAgent
from agents.bayesian_approximation_agent import Agent as BayesianAgent
from backend.environment import MinesweeperEnv


def collect_expert_trajectories(env, expert_agent, num_episodes, verbose=True):
    """
    Collect trajectories from expert agent (Bayesian).
    
    Args:
        env: MinesweeperEnv instance
        expert_agent: Bayesian agent
        num_episodes: Number of episodes to collect
        verbose: Print progress
        
    Returns:
        trajectories: List of (obs, action, reward, next_obs, done) tuples
    """
    trajectories = []
    wins = 0
    losses = 0
    episode_lengths = []
    
    iterator = tqdm(range(num_episodes)) if verbose else range(num_episodes)
    for episode in iterator:
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            action = expert_agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            trajectories.append((obs.copy(), action, reward, next_obs.copy(), done))
            obs = next_obs
            
            # Track episode outcome
            if done:
                episode_lengths.append(steps)
                # Terminal reward is +100 for win, -100 for loss
                if reward >= 100:  # More explicit check for win
                    wins += 1
                else:
                    losses += 1
    
    if verbose:
        win_rate = (wins / num_episodes) * 100
        avg_moves = np.mean(episode_lengths)
        print(f"\nExpert Agent Performance:")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Wins: {wins} ({win_rate:.1f}%)")
        print(f"  Losses: {losses} ({100-win_rate:.1f}%)")
        print(f"  Average moves per episode: {avg_moves:.1f}")
        print(f"  Total transitions collected: {len(trajectories)}")
    
    return trajectories


def pretrain_from_expert(agent, trajectories, num_updates, batch_size=128, verbose=True):
    """
    Pre-train DQN agent on expert trajectories.
    
    Args:
        agent: DuelingDCNNAgent
        trajectories: List of transitions from expert
        num_updates: Number of gradient updates to perform
        batch_size: Training batch size
        verbose: Print progress
        
    Returns:
        losses: List of training losses
    """
    # Fill replay buffer with expert data
    print(f"Filling replay buffer with {len(trajectories)} expert transitions...")
    for obs, action, reward, next_obs, done in trajectories:
        agent.remember(obs, action, reward, next_obs, done)
    
    print(f"Buffer size: {len(agent.buffer)}")
    print(f"Performing {num_updates} pre-training updates...")
    
    losses = []
    iterator = tqdm(range(num_updates)) if verbose else range(num_updates)
    
    for update in iterator:
        stats = agent.optimize()
        if stats:
            losses.append(stats['loss'])
            
            if verbose and (update + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                iterator.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
    
    avg_loss = np.mean(losses) if losses else 0.0
    print(f"\nPre-training complete! Average loss: {avg_loss:.4f}")
    
    return losses


def main():
    parser = argparse.ArgumentParser(description="Pre-train DQN with imitation learning")
    parser.add_argument("--size", type=int, default=5, help="Board size (creates size x size x size cube)")
    parser.add_argument("--mines", type=int, default=10, help="Number of mines")
    parser.add_argument("--expert-episodes", type=int, default=500, help="Number of episodes to collect from expert")
    parser.add_argument("--num-updates", type=int, default=5000, help="Number of training updates")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default=None, help="Path to save pretrained checkpoint")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DQN Imitation Learning Pre-training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Board size: {args.size}x{args.size}x{args.size}")
    print(f"  Mines: {args.mines}")
    print(f"  Expert episodes: {args.expert_episodes}")
    print(f"  Training updates: {args.num_updates}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Create environment
    env = MinesweeperEnv(
        height=args.size,
        width=args.size,
        depth=args.size,
        num_mines=args.mines,
        render_mode=None
    )
    
    # Create expert agent (Bayesian)
    print("\nCreating expert agent (Bayesian)...")
    expert = BayesianAgent(
        env.action_space,
        height=args.size,
        width=args.size,
        depth=args.size,
        num_mines=args.mines
    )
    
    # Create DQN agent
    print("Creating DQN agent...")
    agent = DuelingDCNNAgent(
        height=args.size,
        width=args.size,
        depth=args.size,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup=0,  # No warmup needed, we're filling buffer manually
        target_update=1000,
        buffer_size=200000,
        eps_decay_steps=150000,
        # !!!!! 5070 pytorch issue workaround !!!!!
        # device="cpu"  # Force CPU for compatibility - comment this out to use GPU if available
    )
    print(f"Using device: {agent.device}")
    
    # Collect expert trajectories
    print(f"\nCollecting {args.expert_episodes} episodes from expert...")
    trajectories = collect_expert_trajectories(env, expert, args.expert_episodes)
    
    # Pre-train on expert data
    print("\nPre-training DQN on expert trajectories...")
    losses = pretrain_from_expert(agent, trajectories, args.num_updates, args.batch_size)
    
    # Save checkpoint
    if args.output is None:
        args.output = f"dqn-pretrained-s{args.size}-m{args.mines}.pth"
    
    print(f"\nSaving pretrained checkpoint to: {args.output}")
    agent.save_checkpoint(args.output, step=0)
    
    print("\n" + "=" * 60)
    print("Pre-training complete!")
    print(f"Checkpoint saved to: {args.output}")
    print("\nYou can now use this checkpoint with train_dqn.py:")
    print(f"  python train_dqn.py --size {args.size} --mines {args.mines}")
    print("(It will automatically load the latest checkpoint)")
    print("=" * 60)


if __name__ == "__main__":
    main()
