"""
Imitation Learning Bootstrap for DQN Agent (#2)

Pre-trains the DQN agent by filling the replay buffer with trajectories 
from the Bayesian agent (expert policy), then performing supervised learning
on this data before starting standard RL training.

This gives the DQN a warm start with good behavior patterns.
"""

import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import pickle
import os

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
        
        # Only return winning episodes (win reward is 10.0)
        if reward == 10.0:
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
    
    # Collect episodes in parallel with progress updates
    with Pool(processes=num_workers) as pool:
        if verbose:
            import time
            start_time = time.time()
            print(f"Progress: 0/{num_episodes} episodes collected (0.0 eps/sec)", end='', flush=True)
            completed = 0
            results = []
            # Use imap_unordered for incremental results
            for result in pool.imap_unordered(_collect_single_episode, episode_args):
                results.append(result)
                completed += 1
                # Print progress every 10 episodes or at completion
                if completed % 10 == 0 or completed == num_episodes:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"\rProgress: {completed}/{num_episodes} episodes collected ({rate:.1f} eps/sec)", end='', flush=True)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"\rProgress: {completed}/{num_episodes} episodes collected ({rate:.1f} eps/sec) - Complete!")
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
    
    if verbose:
        import time
        start_time = time.time()
        print(f"Progress: 0/{num_episodes} episodes collected (0.0 eps/sec)", end='', flush=True)
    
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
        
        if reward == 10.0:  # Win (updated reward)
            trajectories.extend(episode_transitions)
            wins += 1
            episode_lengths.append(steps)
            episodes_collected += 1
            if verbose and (episodes_collected % 10 == 0 or episodes_collected == num_episodes):
                elapsed = time.time() - start_time
                rate = episodes_collected / elapsed if elapsed > 0 else 0
                print(f"\rProgress: {episodes_collected}/{num_episodes} episodes collected ({rate:.1f} eps/sec)", end='', flush=True)
        else:
            losses += 1
    
    if verbose:
        elapsed = time.time() - start_time
        rate = episodes_collected / elapsed if elapsed > 0 else 0
        print(f"\rProgress: {episodes_collected}/{num_episodes} episodes collected ({rate:.1f} eps/sec) - Complete!")
    
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


def pretrain_from_expert(agent, trajectories, num_updates, batch_size=128, verbose=True, checkpoint_prefix="dqn-pretrain", save_every=5000, start_update=0):
    """
    Pre-train DQN agent on expert trajectories.
    
    Args:
        agent: DuelingDCNNAgent
        trajectories: List of transitions from expert
        num_updates: Number of gradient updates to perform
        batch_size: Training batch size
        verbose: Print progress
        checkpoint_prefix: Prefix for checkpoint filenames
        save_every: Save checkpoint every N updates
        start_update: Starting update number (for resuming)
        
    Returns:
        losses: List of training losses
    """
    # Fill replay buffer with expert data (if starting fresh or buffer is empty)
    if len(agent.buffer) < len(trajectories):
        print(f"Filling replay buffer with {len(trajectories)} expert transitions...")
        for obs, action, reward, next_obs, done in trajectories:
            agent.remember(obs, action, reward, next_obs, done)
    else:
        print(f"Buffer already filled with {len(agent.buffer)} transitions (resuming from checkpoint)")
    
    print(f"Buffer size: {len(agent.buffer)}")
    print(f"Performing {num_updates} pre-training updates (starting from update {start_update})...")
    print(f"Saving checkpoints every {save_every} updates to {checkpoint_prefix}-XXXXX.pth")
    
    import time
    start_time = time.time()
    print(f"Progress: {start_update}/{start_update + num_updates} updates (0.0 upd/sec)", end='', flush=True)
    
    losses = []
    
    for update in range(num_updates):
        current_update = start_update + update
        stats = agent.optimize()
        if stats:
            losses.append(stats['loss'])
            
            if verbose and (current_update + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                elapsed = time.time() - start_time
                rate = (update + 1) / elapsed if elapsed > 0 else 0
                total_target = start_update + num_updates
                print(f"\rProgress: {current_update + 1}/{total_target} updates ({rate:.1f} upd/sec, avg_loss: {avg_loss:.4f})", end='', flush=True)
        
        # Save checkpoint periodically (using absolute update number)
        if (current_update + 1) % save_every == 0:
            checkpoint_path = f"{checkpoint_prefix}-{current_update + 1}.pth"
            agent.save_checkpoint(checkpoint_path, step=current_update + 1)
            avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if losses else 0.0)
            rate = (update + 1) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            print(f"\n[Checkpoint] Saved {checkpoint_path} (avg_loss: {avg_loss:.4f})")
            total_target = start_update + num_updates
            print(f"Progress: {current_update + 1}/{total_target} updates ({rate:.1f} upd/sec, avg_loss: {avg_loss:.4f})", end='', flush=True)
    
    if verbose:
        elapsed = time.time() - start_time
        rate = num_updates / elapsed if elapsed > 0 else 0
        avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else (np.mean(losses) if losses else 0.0)
        total_target = start_update + num_updates
        print(f"\rProgress: {total_target}/{total_target} updates ({rate:.1f} upd/sec, avg_loss: {avg_loss:.4f}) - Complete!")
    
    avg_loss = np.mean(losses[-10000:]) if len(losses) >= 10000 else np.mean(losses) if losses else 0.0
    print(f"\nPre-training complete! Average loss (last 10k): {avg_loss:.4f}")
    
    return losses


def main():
    parser = argparse.ArgumentParser(description="Pre-train DQN with imitation learning")
    parser.add_argument("--size", type=int, default=5, help="Board size (creates size x size x size cube)")
    parser.add_argument("--mines", type=int, default=10, help="Number of mines")
    parser.add_argument("--expert-episodes", type=int, default=500, help="Number of episodes to collect from expert")
    parser.add_argument("--num-updates", type=int, default=500000, help="Number of training updates (~3 epochs for 100k episodes)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (lower for stable pretraining)")
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
    
    # Check for cached expert trajectories
    expert_cache_path = f"expert_trajectories_s{args.size}_m{args.mines}_n{args.expert_episodes}.pkl"
    
    if os.path.exists(expert_cache_path):
        print(f"\n[CACHE] Found cached expert trajectories: {expert_cache_path}")
        print(f"Loading cached data...")
        with open(expert_cache_path, 'rb') as f:
            trajectories = pickle.load(f)
        print(f"Loaded {len(trajectories)} transitions from cache")
    else:
        # Create expert agent (Bayesian)
        print("\nCreating expert agent (Bayesian)...")
        expert = BayesianAgent(
            env.action_space,
            height=args.size,
            width=args.size,
            depth=args.size,
            num_mines=args.mines
        )
        
        # Collect expert trajectories
        print(f"\nCollecting {args.expert_episodes} winning episodes from expert...")
        trajectories = collect_expert_trajectories(env, expert, args.expert_episodes, num_workers=args.num_workers)
        
        # Cache the trajectories for future use
        print(f"\n[CACHE] Saving expert trajectories to: {expert_cache_path}")
        with open(expert_cache_path, 'wb') as f:
            pickle.dump(trajectories, f)
        print(f"Expert data cached successfully!")
    
    # Create DQN agent
    print("\nCreating DQN agent...")
    agent = DuelingDCNNAgent(
        height=args.size,
        width=args.size,
        depth=args.size,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup=0,  # No warmup needed, we're filling buffer manually
        target_update=10000,  # Infrequent updates during pretraining (or keep target frozen)
        buffer_size=2_100_000,  # Large enough to hold ~2M transitions from 100k winning expert episodes
        eps_decay_steps=150000,
        # !!!!! 5070 pytorch issue workaround !!!!!
        # device="cpu"  # Force CPU for compatibility - comment this out to use GPU if available
    )
    print(f"Using device: {agent.device}")
    
    # Check for existing pretrain checkpoint to resume from
    checkpoint_prefix = f"dqn-pretrain-s{args.size}-m{args.mines}"
    existing_checkpoints = sorted(
        [f for f in os.listdir('.') if checkpoint_prefix in f and f.endswith('.pth') and f != f"{checkpoint_prefix}.pth"],
        key=lambda x: int(x.split('-')[-1].split('.')[0])
    )
    
    start_update = 0
    if existing_checkpoints:
        latest_checkpoint = existing_checkpoints[-1]
        start_update = int(latest_checkpoint.split('-')[-1].split('.')[0])
        print(f"\n[RESUME] Found existing checkpoint: {latest_checkpoint}")
        print(f"Loading checkpoint to resume from update {start_update}...")
        try:
            agent.load_checkpoint(latest_checkpoint)
            print(f"Successfully loaded checkpoint!")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print(f"Starting from scratch instead.")
            start_update = 0
    
    # Pre-train on expert data
    if start_update >= args.num_updates:
        print(f"\n[COMPLETE] Pretraining already completed ({start_update}/{args.num_updates} updates)")
        print(f"Use --force-pretrain to restart from scratch")
    else:
        remaining_updates = args.num_updates - start_update
        print(f"\nPre-training DQN on expert trajectories...")
        print(f"Starting from update {start_update}, performing {remaining_updates} more updates (target: {args.num_updates})")
        losses = pretrain_from_expert(
            agent, trajectories, remaining_updates, args.batch_size, 
            checkpoint_prefix=checkpoint_prefix, save_every=5000, start_update=start_update
        )
    
    # Save final checkpoint
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
