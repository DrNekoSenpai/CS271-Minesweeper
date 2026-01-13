"""Test if hybrid mode and safe action detection works"""
import torch
import numpy as np
import os
import sys
import re
from agents.dueling_cnn_agent import DuelingDCNNAgent
from backend.environment import MinesweeperEnv

# Find any training checkpoint (dqn-checkpoint-s*-m*-*.pth)
checkpoint_pattern = re.compile(r"dqn-checkpoint-s(\d+)-m(\d+)-(\d+)\.pth")
checkpoints = []
for f in os.listdir('.'):
    match = checkpoint_pattern.match(f)
    if match:
        size, mines, step = match.groups()
        checkpoints.append((int(step), int(size), int(mines), f))

if not checkpoints:
    print("[SKIP] No training checkpoint found (dqn-checkpoint-*.pth)")
    print("This test requires a trained DQN checkpoint to run.")
    sys.exit(2)  # Exit code 2 = skipped

# Use latest checkpoint
checkpoints.sort(key=lambda x: x[0])
step, size, mines, checkpoint_path = checkpoints[-1]
print(f"Using checkpoint: {checkpoint_path} (size={size}, mines={mines}, step={step})")

# Load agent
agent = DuelingDCNNAgent(height=size, width=size, depth=size, device='cpu')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
agent.online.load_state_dict(checkpoint['online_state'])
agent.target.load_state_dict(checkpoint['target_state'])
agent.total_steps = agent.eps_decay_steps  # No exploration

# Test safe action detection
env = MinesweeperEnv(height=size, width=size, depth=size, num_mines=mines, render_mode=None)

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
