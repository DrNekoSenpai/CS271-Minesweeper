"""Check if safe neighbors are actually legal actions"""
import numpy as np
from backend.environment import MinesweeperEnv
from agents.dueling_cnn_agent import DuelingDCNNAgent

env = MinesweeperEnv(height=5, width=5, depth=5, num_mines=8, render_mode=None)
agent = DuelingDCNNAgent(height=5, width=5, depth=5, device='cpu')

# Try multiple games until we find one with zeros
for attempt in range(20):
    obs, _ = env.reset()
    
    # Make a random move
    legal = agent.legal_action_indices(obs)
    action = np.random.choice(legal)
    obs, reward, term, trunc, info = env.step(action)
    
    num_zeros = np.sum(obs == 0)
    if num_zeros > 0:
        print(f"Found a game with {num_zeros} zeros after first move!\n")
        
        print(f"Unrevealed (-1): {np.sum(obs == -1)}")
        print(f"Buried (-2): {np.sum(obs == -2)}")
        
        # Check safe actions
        safe_actions = agent.get_safe_actions(obs)
        legal_actions = agent.legal_action_indices(obs)
        
        print(f"\nSafe actions found: {len(safe_actions)}")
        print(f"Legal actions: {len(legal_actions)}")
        
        # Are safe actions actually legal?
        safe_set = set(safe_actions)
        legal_set = set(legal_actions)
        valid_safe = safe_set.intersection(legal_set)
        
        print(f"Safe actions that are LEGAL: {len(valid_safe)}")
        print(f"Safe actions that are NOT legal: {len(safe_actions) - len(valid_safe)}")
        
        if len(safe_actions) > 0 and len(valid_safe) < len(safe_actions):
            print("\n[FAIL] BUG CONFIRMED: Some safe actions are NOT legal (buried tiles)!")
        elif len(safe_actions) > 0 and len(valid_safe) == len(safe_actions):
            print("\n[PASS] Safe action detection working correctly!")
        elif len(safe_actions) == 0:
            print("\n[INFO] No safe actions found even with zeros present - investigating...")
            # Show where the zeros are
            zero_coords = np.argwhere(obs == 0)
            print(f"Zero locations: {zero_coords[:5]}")  # Show first 5
        
        break
else:
    print("Could not find a game with zeros in 20 attempts")
