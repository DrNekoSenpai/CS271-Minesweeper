"""
Test if pretraining actually helps by comparing pretrained vs random initialization.

This script evaluates both a pretrained and a fresh (random) agent on the same
set of test games to see if pretraining provides a meaningful advantage.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

import numpy as np
from agents.dueling_cnn_agent import DuelingDCNNAgent
from backend.environment import MinesweeperEnv

def evaluate_agent(agent, env, num_episodes=100, verbose=True):
    """Evaluate agent performance."""
    rewards = []
    safe_moves = []
    safe_tiles = []
    wins = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_safe_moves = 0
        ep_safe_tiles = 0
        
        while not done:
            # Use greedy policy (no exploration)
            action = agent.select_action(obs, use_hybrid=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            ep_safe_moves += info.get('safe_move', 0)
            ep_safe_tiles += info.get('safe_tiles', 0)
        
        rewards.append(ep_reward)
        safe_moves.append(ep_safe_moves)
        safe_tiles.append(ep_safe_tiles)
        
        if reward == 100.0:  # Win
            wins += 1
        
        if verbose and (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: {wins} wins so far")
    
    return {
        'avg_reward': np.mean(rewards),
        'avg_safe_moves': np.mean(safe_moves),
        'avg_safe_tiles': np.mean(safe_tiles),
        'win_rate': wins / num_episodes * 100,
        'wins': wins
    }


def main():
    print("=" * 70)
    print("Pretraining Validation Test")
    print("=" * 70)
    
    size = 5
    mines = 5
    num_test_episodes = 100
    
    print(f"\nConfiguration:")
    print(f"  Board: {size}x{size}x{size}")
    print(f"  Mines: {mines}")
    print(f"  Test episodes: {num_test_episodes}")
    
    # Create environment
    env = MinesweeperEnv(
        height=size,
        width=size,
        depth=size,
        num_mines=mines,
        render_mode=None
    )
    
    # Test 1: Pretrained agent
    print(f"\n{'-'*70}")
    print("Test 1: Pretrained Agent")
    print(f"{'-'*70}")
    
    pretrained_agent = DuelingDCNNAgent(
        height=size, width=size, depth=size,
        eps_start=0.0, eps_end=0.0  # No exploration for testing
    )
    
    # Find pretrained checkpoint
    import os
    import re
    
    # Match both old format (dqn-pretrain-s5-m5-500000.pth) and new format (dqn-pretrain-s5-m5-500000-loss1.234.pth)
    checkpoint_pattern = re.compile(rf"dqn-pretrain-s{size}-m{mines}-(\d+)(?:-loss[\d\.]+)?\.pth")
    
    pretrain_files = []
    for f in os.listdir('.'):
        match = checkpoint_pattern.match(f)
        if match:
            step = int(match.group(1))
            pretrain_files.append((step, f))
    
    if not pretrain_files:
        print("\nError: No pretrained checkpoint found!")
        print("Run pretraining first.")
        return
    
    # Sort by step number and get latest
    pretrain_files.sort(key=lambda x: x[0])
    _, pretrain_checkpoint = pretrain_files[-1]
    print(f"Loading: {pretrain_checkpoint}")
    pretrained_agent.load_checkpoint(pretrain_checkpoint)
    
    print("Evaluating pretrained agent...")
    pretrained_results = evaluate_agent(pretrained_agent, env, num_test_episodes)
    
    # Test 2: Random (untrained) agent
    print(f"\n{'-'*70}")
    print("Test 2: Random Initialization (No Training)")
    print(f"{'-'*70}")
    
    random_agent = DuelingDCNNAgent(
        height=size, width=size, depth=size,
        eps_start=0.0, eps_end=0.0  # No exploration for testing
    )
    
    print("Evaluating randomly initialized agent...")
    random_results = evaluate_agent(random_agent, env, num_test_episodes)
    
    # Compare results
    print(f"\n{'='*70}")
    print("Results Comparison")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'Pretrained':<20} {'Random':<20} {'Improvement':<15}")
    print(f"{'-'*70}")
    
    metrics = [
        ('Avg Reward', 'avg_reward', '.1f'),
        ('Avg Safe Moves', 'avg_safe_moves', '.1f'),
        ('Avg Safe Tiles', 'avg_safe_tiles', '.1f'),
        ('Win Rate', 'win_rate', '.1f'),
        ('Total Wins', 'wins', 'd')
    ]
    
    for name, key, fmt in metrics:
        pretrain_val = pretrained_results[key]
        random_val = random_results[key]
        
        if random_val > 0:
            improvement = ((pretrain_val - random_val) / random_val) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{name:<20} {pretrain_val:<20{fmt}} {random_val:<20{fmt}} {improvement_str:<15}")
    
    print(f"\n{'='*70}")
    print("Conclusion:")
    print(f"{'='*70}")
    
    if pretrained_results['avg_safe_tiles'] > random_results['avg_safe_tiles'] * 1.5:
        print("[PASS] Pretraining is WORKING - significant improvement over random!")
    elif pretrained_results['avg_safe_tiles'] > random_results['avg_safe_tiles'] * 1.1:
        print("[WARN] Pretraining shows SOME benefit, but marginal")
    else:
        print("[FAIL] Pretraining shows NO clear benefit - may need to revise approach")
    
    print(f"\nPretrained agent reveals {pretrained_results['avg_safe_tiles']:.1f} tiles on average")
    print(f"Random agent reveals {random_results['avg_safe_tiles']:.1f} tiles on average")
    print(f"Target for wins: ~120 tiles (125 total - 5 mines)")
    
    if pretrained_results['wins'] > 0:
        print(f"\n[SUCCESS] Pretrained agent can already win games! ({pretrained_results['wins']}/{num_test_episodes})")
    else:
        print(f"\n[INFO] Neither agent wins yet - RL training should help explore endgame strategies")


if __name__ == "__main__":
    main()
