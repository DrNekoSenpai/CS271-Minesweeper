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
from multiprocessing import Pool, cpu_count

from agents.dueling_cnn_agent import DuelingDCNNAgent
from agents.bayesian_approximation_agent import Agent as BayesianAgent
from backend.environment import MinesweeperEnv


def _collect_single_episode(args_tuple):
    """
    Worker function for parallel episode collection.
    
    Args:
        args_tuple: (size, mines, episode_id) tuple
        
    Returns:
        (trajectories, steps, attempts): Episode data, episode length, and number of attempts needed
    """
    size, mines, episode_id = args_tuple
    
    # Create fresh env and agent in this worker process
    env = MinesweeperEnv(
        height=size,
        width=size,
        depth=size,
        num_mines=mines,
        render_mode=None
    )
    
    expert = BayesianAgent(
        env.action_space,
        height=size,
        width=size,
        depth=size,
        num_mines=mines
    )
    
    # Keep trying until we get a win
    attempts = 0
    while True:
        attempts += 1
        obs, _ = env.reset()
        done = False
        steps = 0
        episode_transitions = []
        
        while not done:
            action = expert.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            episode_transitions.append((obs.copy(), action, reward, next_obs.copy(), done))
            obs = next_obs
        
        # Only return winning episodes
        if reward == 100.0:
            return episode_transitions, steps, attempts
        # Otherwise, loop and try again


def collect_expert_trajectories(env, expert_agent, num_episodes, verbose=True, num_workers=None):
    """
    Collect trajectories from expert agent (Bayesian) using parallel workers.
    
    Args:
        env: MinesweeperEnv instance (used for extracting config)
        expert_agent: Bayesian agent (not used in parallel version, kept for API compat)
        num_episodes: Number of winning episodes to collect
        verbose: Print progress
        num_workers: Number of parallel workers (default: cpu_count())
        
    Returns:
        trajectories: List of (obs, action, reward, next_obs, done) tuples
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    if num_workers == 1:
        # Fall back to single-threaded for debugging or when explicitly requested
        return _collect_expert_trajectories_sequential(env, expert_agent, num_episodes, verbose)
    
    # Extract environment config
    size = env.height
    mines = env.num_mines
    
    trajectories = []
    episode_lengths = []
    total_attempts = 0
    
    if verbose:
        print(f"Collecting {num_episodes} winning episodes using {num_workers} parallel workers...")
    
    # Prepare arguments for each episode
    episode_args = [(size, mines, i) for i in range(num_episodes)]
    
    # Collect episodes in parallel with progress bar
    with Pool(processes=num_workers) as pool:
        if verbose:
            results = list(tqdm(
                pool.imap_unordered(_collect_single_episode, episode_args),
                total=num_episodes,
                desc="Collecting episodes"
            ))
        else:
            results = pool.map(_collect_single_episode, episode_args)
    
    # Aggregate results
    for episode_transitions, steps, attempts in results:
        trajectories.extend(episode_transitions)
        episode_lengths.append(steps)
        total_attempts += attempts
    
    if verbose:
        win_rate = (num_episodes / total_attempts) * 100 if total_attempts > 0 else 0
        avg_moves = np.mean(episode_lengths)
        print(f"\nExpert Agent Performance:")
        print(f"  Winning episodes collected: {num_episodes}")
        print(f"  Total attempts needed: {total_attempts}")
        print(f"  Expert win rate: {win_rate:.1f}%")
        print(f"  Parallel workers: {num_workers}")
        print(f"  Average moves per winning episode: {avg_moves:.1f}")
        print(f"  Total transitions collected: {len(trajectories)}")
        print(f"  [INFO] Training ONLY on winning trajectories!")
    
    return trajectories


def _collect_expert_trajectories_sequential(env, expert_agent, num_episodes, verbose=True):
    """
    Single-threaded version of trajectory collection (fallback).
    """
    trajectories = []
    wins = 0
    losses = 0
    episode_lengths = []
    
    iterator = tqdm(range(num_episodes), desc="Collecting episodes") if verbose else range(num_episodes)
    episodes_collected = 0
    
    while episodes_collected < num_episodes:
        obs, _ = env.reset()
        done = False
        steps = 0
        episode_transitions = []
        
        while not done:
            action = expert_agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            episode_transitions.append((obs.copy(), action, reward, next_obs.copy(), done))
            obs = next_obs
        
        if reward == 100.0:
            trajectories.extend(episode_transitions)
            wins += 1
            episode_lengths.append(steps)
            episodes_collected += 1
            if verbose:
                iterator.update(1)
        else:
            losses += 1
    
    if verbose:
        total_attempts = wins + losses
        win_rate = (wins / total_attempts) * 100 if total_attempts > 0 else 0
        avg_moves = np.mean(episode_lengths)
        print(f"\nExpert Agent Performance:")
        print(f"  Winning episodes collected: {wins}")
        print(f"  Total attempts needed: {total_attempts}")
        print(f"  Expert win rate: {win_rate:.1f}%")
        print(f"  Average moves per winning episode: {avg_moves:.1f}")
        print(f"  Total transitions collected: {len(trajectories)}")
        print(f"  [INFO] Training ONLY on winning trajectories!")
    
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
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel workers for data collection (default: CPU count)")
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
    print(f"\nCollecting {args.expert_episodes} winning episodes from expert...")
    trajectories = collect_expert_trajectories(env, expert, args.expert_episodes, num_workers=args.num_workers)
    
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
