"""Test if safe action detection is working correctly"""
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
agent.total_steps = agent.eps_decay_steps
agent.safe_action_prob = 1.0  # Always use safe moves

env = MinesweeperEnv(height=size, width=size, depth=size, num_mines=mines, render_mode=None)

print("Running 5 episodes with detailed logging:\n")

for ep in range(5):
    obs, _ = env.reset()
    done = False
    moves = 0
    safe_moves = 0
    dqn_moves = 0
    
    print(f"=== Episode {ep+1} ===")
    
    while not done and moves < 50:
        legal = agent.legal_action_indices(obs)
        safe_actions = agent.get_safe_actions(obs)
        
        # How many zeros are revealed?
        num_zeros = np.sum(obs == 0)
        num_unrevealed = np.sum(obs == -1)
        
        action = agent.select_action(obs, use_hybrid=True)
        
        used_safe = action in safe_actions if safe_actions.size > 0 else False
        
        obs, reward, terminated, truncated, info = env.step(action)
        moves += 1
        done = terminated or truncated
        
        if used_safe:
            safe_moves += 1
        else:
            dqn_moves += 1
            print(f"  Move {moves}: DQN choice (zeros={num_zeros}, unrevealed={num_unrevealed}, safe_avail={safe_actions.size}, reward={reward:.1f})")
    
    result = "WIN" if reward == 100.0 else "LOSS"
    print(f"Result: {result} - {moves} moves ({safe_moves} safe, {dqn_moves} DQN)\n")
