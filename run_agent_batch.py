import importlib
import os
import numpy as np
from backend.environment import MinesweeperEnv

# ========== CONFIG ==========
AGENT_NAME = "bayesian_approximation_agent" # Name of agent to run
NUM_EPISODES = 5000         # Number of episodes to run
SIZE = 5                    # Size of the Minesweeper grid (height, width, depth)
NUM_MINES = 10              # Number of mines in the grid
# ============================

def load_agent(agent_name, action_space):
    try:
        module = importlib.import_module(f"agents.{agent_name}")
        AgentClass = getattr(module, "Agent")
        return AgentClass(action_space)
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Agent '{agent_name}' not found in agents/")

def run_episode(env, agent):
    obs, _ = env.reset()
    done = False

    total_reward = 0.0
    total_moves = 0
    won = True

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_moves += 1

        # Loss condition: mine hit
        if reward < 0:
            won = False

        done = terminated or truncated

    return {
        "reward": total_reward,
        "moves": total_moves,
        "won": won
    }


def summarize(values, name):
    values = np.array(values)
    print(f"\n{name}:")
    print(f"  min     = {values.min():.2f}")
    print(f"  max     = {values.max():.2f}")
    print(f"  mean    = {values.mean():.2f}")
    print(f"  median  = {np.median(values):.2f}")
    print(f"  var     = {values.var():.2f}")
    print(f"  stdev   = {values.std():.2f}")

def main():
    env = MinesweeperEnv(
        height=SIZE,
        width=SIZE,
        depth=SIZE,
        num_mines=NUM_MINES,
        render_mode=None
    )

    agent = load_agent(AGENT_NAME, env.action_space)

    rewards = []
    moves = []
    wins = 0
    losses = 0

    print(f"Running {NUM_EPISODES} episodes with agent '{AGENT_NAME}'...\n")

    for i in range(NUM_EPISODES):
        result = run_episode(env, agent)

        rewards.append(result["reward"])
        moves.append(result["moves"])

        if result["won"]:
            wins += 1
        else:
            losses += 1

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{NUM_EPISODES}")

    # ---------- SUMMARY ----------
    summarize(rewards, "Total Reward")
    summarize(moves, "Number of Moves")

    print("\nWin / Loss:")
    print(f"  Wins   = {wins}")
    print(f"  Losses = {losses}")
    print(f"  Win rate = {wins / NUM_EPISODES:.3f}")


if __name__ == "__main__":
    main()
