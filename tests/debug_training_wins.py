"""
Debug script to check if training win tracking matches reality.
Runs a simulated training loop and verifies win conditions.
"""

import numpy as np
import gymnasium as gym
from backend.environment import MinesweeperEnv
from agents.dueling_cnn_agent import DuelingDCNNAgent

def make_env(depth, height, width, num_mines):
    def _thunk():
        return MinesweeperEnv(
            depth=depth,
            height=height,
            width=width,
            num_mines=num_mines,
            render_mode=None
        )
    return _thunk

# Test with actual vectorized envs like training
size = 5
mines = 5
num_envs = 8

envs = gym.vector.SyncVectorEnv([
    make_env(size, size, size, mines) for _ in range(num_envs)
])

obs, info = envs.reset()

# Create agent
agent = DuelingDCNNAgent(height=size, width=size, depth=size, device='cpu')

recent_wins = []
recent_rewards = []
completed = 0

episode_rewards = np.zeros(num_envs, dtype=np.float32)

print("Running 1000 steps to check win tracking...\n")

terminal_rewards_seen = []

for step in range(1000):
    actions = agent.select_actions(obs)
    next_obs, rewards, terminated, truncated, infos = envs.step(actions)
    done = np.logical_or(terminated, truncated)
    
    for i in range(num_envs):
        episode_rewards[i] += rewards[i]
        
        if done[i]:
            completed += 1
            
            # Record terminal reward
            terminal_rewards_seen.append(rewards[i])
            
            # Check win condition (win=10.0, loss=-10.0)
            if rewards[i] == 10.0:
                recent_wins.append(1)
                print(f"[WIN] Win detected! Env {i}, step {step}, terminal_reward={rewards[i]:.1f}, episode_total={episode_rewards[i]:.1f}")
            elif rewards[i] == -10.0:
                recent_wins.append(0)
            else:
                recent_wins.append(0)
                print(f"[ERROR] Terminal but reward is {rewards[i]:.1f}, not 10.0/-10.0! Env {i}, step {step}")
            
            recent_rewards.append(episode_rewards[i])
            episode_rewards[i] = 0.0
    
    obs = next_obs
    
    # Print summary every 100 completed episodes
    if completed >= 100 and completed % 100 == 0:
        actual_wins = sum(recent_wins[-100:])
        avg_reward = np.mean(recent_rewards[-100:])
        print(f"\n[{completed} episodes] wins={actual_wins}/100, avg_reward={avg_reward:.1f}")

# Final summary
print(f"\n{'='*60}")
print(f"Total episodes completed: {completed}")
print(f"Total wins: {sum(recent_wins)}")
print(f"Win rate: {sum(recent_wins)/max(1,completed)*100:.1f}%")
print(f"\nTerminal rewards seen: {len(terminal_rewards_seen)}")
print(f"  +100.0 (win) count: {sum(1 for r in terminal_rewards_seen if r == 100.0)}")
print(f"  -50.0 (loss) count: {sum(1 for r in terminal_rewards_seen if r == -50.0)}")
print(f"  Other count: {sum(1 for r in terminal_rewards_seen if r != 100.0 and r != -50.0)}")

# Check for terminal rewards (win=100.0, loss=-50.0 in current environment)
if any(r != 100.0 and r != -50.0 for r in terminal_rewards_seen):
    print(f"\n[WARNING] BUG FOUND: Terminal rewards that aren't 10.0/-10.0!")
    weird_rewards = [r for r in terminal_rewards_seen if r != 10.0 and r != -10.0]
    print(f"  Weird rewards: {weird_rewards[:10]}")
    import sys
    sys.exit(1)
else:
    print(f"\n[PASS] All terminal rewards are correctly 10.0/-10.0")
    print(f"[INFO] No bugs found in win tracking!")
    import sys
    sys.exit(0)
