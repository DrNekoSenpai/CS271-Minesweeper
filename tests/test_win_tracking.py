"""Test if win tracking is working correctly"""
import numpy as np
from backend.environment import MinesweeperEnv
from agents.bayesian_approximation_agent import Agent as BayesianAgent

env = MinesweeperEnv(height=5, width=5, depth=5, num_mines=8, render_mode=None)
agent = BayesianAgent(env.action_space, height=5, width=5, depth=5, num_mines=8)

wins = 0
mismatches = 0

for ep in range(50):
    obs, _ = env.reset()
    done = False
    final_reward = 0
    
    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            final_reward = reward
    
    # Check consistency
    game_win = env.game.win
    reward_win = (final_reward == 100.0)
    
    if game_win != reward_win:
        print(f"MISMATCH! Episode {ep}: game.win={game_win}, reward={final_reward}")
        mismatches += 1
    
    if game_win:
        wins += 1

print(f"\nResults:")
print(f"  Wins: {wins}/50 ({wins/50*100:.1f}%)")
print(f"  Mismatches: {mismatches}")
print(f"\n{'[PASS] Win tracking is correct!' if mismatches == 0 else '[FAIL] Win tracking has bugs!'}")
