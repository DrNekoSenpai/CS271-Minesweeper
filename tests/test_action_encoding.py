"""
Test to verify action encoding/decoding is consistent between 
Bayesian agent and environment.
"""

import numpy as np
from backend.environment import MinesweeperEnv
from agents.bayesian_approximation_agent import Agent as BayesianAgent

# Create environment
env = MinesweeperEnv(height=5, width=5, depth=5, num_mines=5, render_mode=None)
obs, _ = env.reset(seed=42)

print("Observation shape:", obs.shape)
print("\nTesting action encoding/decoding consistency:\n")

# Test a few specific positions
test_positions = [
    (0, 0, 0),
    (4, 4, 4),
    (1, 2, 3),
    (2, 1, 3),
]

for pos in test_positions:
    x, y, z = pos
    H, W, D = obs.shape
    
    # How Bayesian agent encodes
    action = z * (H * W) + y * W + x
    
    # How environment decodes
    decoded_pos = env._decode_action(action)
    
    print(f"Position {pos}:")
    print(f"  Bayesian encodes to action: {action}")
    print(f"  Environment decodes to: {decoded_pos}")
    print(f"  Match: {pos == decoded_pos}")
    print()

# Now test the actual agent on a simple observation
print("\n" + "="*60)
print("Testing Bayesian agent action selection:")
print("="*60)

# Create a simple test observation with some revealed tiles
test_obs = np.full((5, 5, 5), -1, dtype=np.int32)
# Set borders to -1 (exposed)
test_obs[0, :, :] = -1
test_obs[-1, :, :] = -1
test_obs[:, 0, :] = -1
test_obs[:, -1, :] = -1
test_obs[:, :, 0] = -1
test_obs[:, :, -1] = -1

# Reveal a zero at position (0, 0, 0)
test_obs[0, 0, 0] = 0

print("\nTest observation with zero at (0, 0, 0)")
print(f"test_obs[0, 0, 0] = {test_obs[0, 0, 0]}")

# Create agent
agent = BayesianAgent(env.action_space, height=5, width=5, depth=5, num_mines=5)

# Get action from agent
action = agent.select_action(test_obs)
decoded = env._decode_action(action)

print(f"\nAgent selected action: {action}")
print(f"Environment decodes to position: {decoded}")
print(f"Observation value at that position: {test_obs[decoded]}")

# Check if it's picking a neighbor of the zero
neighbors_of_zero = []
for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
    nx, ny, nz = 0 + dx, 0 + dy, 0 + dz
    if 0 <= nx < 5 and 0 <= ny < 5 and 0 <= nz < 5:
        neighbors_of_zero.append((nx, ny, nz))

print(f"\nNeighbors of zero at (0,0,0): {neighbors_of_zero}")
print(f"Agent picked neighbor: {decoded in neighbors_of_zero}")
