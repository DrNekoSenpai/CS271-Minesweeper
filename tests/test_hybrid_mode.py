"""Test if hybrid mode and safe action detection works"""
import torch
import numpy as np
import os
import sys
from agents.dueling_cnn_agent import DuelingDCNNAgent
from backend.environment import MinesweeperEnv

# Check if checkpoint exists
checkpoint_path = "dqn-checkpoint-s5-m8-400000.pth"
if not os.path.exists(checkpoint_path):
    print(f"[SKIP] Checkpoint file not found: {checkpoint_path}")
    print("This test requires a trained DQN checkpoint to run.")
    sys.exit(2)  # Exit code 2 = skipped

# Load agent
agent = DuelingDCNNAgent(height=5, width=5, depth=5, device='cpu')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
agent.online.load_state_dict(checkpoint['online_state'])
agent.target.load_state_dict(checkpoint['target_state'])
agent.total_steps = agent.eps_decay_steps  # No exploration

# Test safe action detection
env = MinesweeperEnv(height=5, width=5, depth=5, num_mines=8, render_mode=None)

print("Testing hybrid mode on 10 episodes:\n")

total_moves = 0
safe_moves_used = 0
dqn_moves = 0
wins = 0

for ep in range(10):
    obs, _ = env.reset()
    done = False
    moves = 0
    episode_safe = 0
    episode_dqn = 0
    
    while not done:
        # Check what's available
        legal = agent.legal_action_indices(obs)
        safe_actions = agent.get_safe_actions(obs)
        
        # Select action with hybrid
        action = agent.select_action(obs, use_hybrid=True)
        
        # Track if we used safe action
        if safe_actions.size > 0 and action in safe_actions:
            episode_safe += 1
        else:
            episode_dqn += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        moves += 1
        done = terminated or truncated
    
    total_moves += moves
    safe_moves_used += episode_safe
    dqn_moves += episode_dqn
    
    if reward == 100.0:
        wins += 1
    
    print(f"Episode {ep+1}: {moves} moves, {episode_safe} safe, {episode_dqn} DQN, "
          f"{'WIN' if reward == 100.0 else 'LOSS'}")

print(f"\nSummary:")
print(f"  Total moves: {total_moves}")
print(f"  Safe moves used: {safe_moves_used} ({safe_moves_used/total_moves*100:.1f}%)")
print(f"  DQN moves: {dqn_moves} ({dqn_moves/total_moves*100:.1f}%)")
print(f"  Wins: {wins}/10 ({wins/10*100:.1f}%)")
print(f"\n{'[PASS] Hybrid mode working!' if safe_moves_used > 0 else '[FAIL] Safe moves not being used! Is hybrid mode enabled?'}")
