"""
Diagnostic script to understand why DQN has 0 wins.

This script:
1. Loads both pretrained and trained checkpoints
2. Compares their Q-value distributions
3. Runs test episodes to see behavior
"""

import torch
import numpy as np
from agents.dueling_cnn_agent import DuelingDCNNAgent
from backend.environment import MinesweeperEnv


def analyze_q_values(agent, env, num_samples=20):
    """Analyze Q-value distributions on random initial states."""
    q_values_all = []
    
    for _ in range(num_samples):
        obs, _ = env.reset()
        obs_b = np.expand_dims(obs, axis=0)
        x = agent.obs_to_tensor(obs_b)
        
        with torch.no_grad():
            q = agent.online(x)[0]
            q_values_all.append(q.cpu().numpy())
    
    q_values_all = np.concatenate(q_values_all)
    
    return {
        'min': q_values_all.min(),
        'max': q_values_all.max(),
        'mean': q_values_all.mean(),
        'std': q_values_all.std(),
        'negative_ratio': (q_values_all < 0).mean(),
        'positive_ratio': (q_values_all > 0).mean(),
    }


def test_performance(agent, env, num_episodes=100, verbose=False):
    """Test agent performance."""
    wins = 0
    losses = 0
    total_moves = 0
    rewards_list = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        moves = 0
        total_reward = 0
        
        while not done and moves < 200:  # Safety limit
            legal = agent.legal_action_indices(obs)
            if legal.size == 0:
                break
            
            obs_b = np.expand_dims(obs, axis=0)
            x = agent.obs_to_tensor(obs_b)
            
            with torch.no_grad():
                q = agent.online(x)[0]
                mask = torch.zeros(agent.n_actions, dtype=torch.bool, device=agent.device)
                mask[torch.from_numpy(legal).to(agent.device)] = True
                q_masked = q.clone()
                q_masked[~mask] = -1e9
                action = int(torch.argmax(q_masked).item())
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            moves += 1
            done = terminated or truncated
        
        rewards_list.append(total_reward)
        total_moves += moves
        
        if reward == 100.0:
            wins += 1
        else:
            losses += 1
        
        if verbose and (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}: {wins} wins so far")
    
    return {
        'wins': wins,
        'losses': losses,
        'win_rate': wins / num_episodes,
        'avg_moves': total_moves / num_episodes,
        'avg_reward': np.mean(rewards_list),
        'rewards_std': np.std(rewards_list),
    }


def main():
    print("=" * 70)
    print("DQN DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    
    # Setup
    size = 5
    mines = 8
    env = MinesweeperEnv(height=size, width=size, depth=size, num_mines=mines, render_mode=None)
    
    # Test pretrained model
    print("\n1. PRETRAINED MODEL (after imitation learning)")
    print("-" * 70)
    pretrained_agent = DuelingDCNNAgent(height=size, width=size, depth=size, device='cpu')
    pretrained_path = f"dqn-pretrained-s{size}-m{mines}.pth"
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    pretrained_agent.online.load_state_dict(checkpoint['online_state'])
    pretrained_agent.target.load_state_dict(checkpoint['target_state'])
    
    print(f"\nCheckpoint step: {checkpoint.get('step', 0)}")
    
    print("\nQ-value Analysis (20 random initial states):")
    q_stats = analyze_q_values(pretrained_agent, env, num_samples=20)
    for key, val in q_stats.items():
        print(f"  {key:20s}: {val:.4f}")
    
    print("\nPerformance Test (100 episodes):")
    perf = test_performance(pretrained_agent, env, num_episodes=100, verbose=False)
    for key, val in perf.items():
        print(f"  {key:20s}: {val:.4f}" if isinstance(val, float) else f"  {key:20s}: {val}")
    
    # Test trained model
    print("\n\n2. TRAINED MODEL (after RL training)")
    print("-" * 70)
    trained_agent = DuelingDCNNAgent(height=size, width=size, depth=size, device='cpu')
    trained_path = f"dqn-checkpoint-s{size}-m{mines}-400000.pth"
    
    checkpoint = torch.load(trained_path, map_location='cpu')
    trained_agent.online.load_state_dict(checkpoint['online_state'])
    trained_agent.target.load_state_dict(checkpoint['target_state'])
    
    print(f"\nCheckpoint step: {checkpoint.get('step', 0)}")
    print(f"Total training steps: {checkpoint.get('total_steps', 0)}")
    
    print("\nQ-value Analysis (20 random initial states):")
    q_stats = analyze_q_values(trained_agent, env, num_samples=20)
    for key, val in q_stats.items():
        print(f"  {key:20s}: {val:.4f}")
    
    print("\nPerformance Test (100 episodes):")
    perf = test_performance(trained_agent, env, num_episodes=100, verbose=False)
    for key, val in perf.items():
        print(f"  {key:20s}: {val:.4f}" if isinstance(val, float) else f"  {key:20s}: {val}")
    
    print("\n\n3. DIAGNOSIS")
    print("=" * 70)
    print("""
The likely cause of 0 wins is:

1. Q-VALUE COLLAPSE: After RL training, all Q-values became negative,
   indicating the model learned that all actions lead to bad outcomes.

2. NEGATIVE REWARD DOMINATION: The model experienced many more losses 
   than wins during training, causing it to become pessimistic.

3. CATASTROPHIC FORGETTING: The good behaviors learned from imitation
   were overwritten by random exploration experiences during RL training.

SOLUTIONS:
- Use a lower initial epsilon (0.1 instead of 1.0) since we have pretraining
- Increase the ratio of expert demonstrations during training
- Use behavior cloning loss alongside Q-learning
- Adjust reward structure (e.g., -10 for loss instead of -100)
- Increase pretraining steps before RL
    """)


if __name__ == "__main__":
    main()
