import importlib, argparse, os 
from environment import MinesweeperEnv

def load_agent(agent_name, action_space): 
    try: 
        module = importlib.import_module(f"agents.{agent_name}")
        AgentClass = getattr(module, "Agent")
        return AgentClass(action_space)
    except (ModuleNotFoundError, AttributeError): 
        raise ValueError(f"Agent '{agent_name}' not found in agents/")

def list_agents():
    """
    Lists all agent python files in the agents/ directory.
    """
    agent_dir = os.path.join(os.path.dirname(__file__), "agents")
    agents = []

    for f in os.listdir(agent_dir):
        if f.endswith(".py") and f != "__init__.py":
            agents.append(f[:-3])  # remove .py extension

    return agents

def decode_action(action, depth, height, width):
    z, rem = divmod(action, height * width)
    y, x = divmod(rem, width)
    return z, y, x

def main():
    # ----- Parse arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", "-a", type=str, default=None, help="Name of agent (matching agents/<name>.py)")
    parser.add_argument("--size", "-s", type=int, default=5, help="Determine dimensions of Minesweeper board")
    parser.add_argument("--num_mines", "-m", type=int, default=10, help="Determine number of mines present on board")
    args = parser.parse_args()

    size = args.size
    num_mines = args.num_mines

    # ----- List agents -----
    available = list_agents()

    if args.agent is None:
        print("Available agents:")
        for a in available:
            print("  -", a)
        print("\nUse --agent <name> to run one of them.")
        return

    if args.agent not in available:
        print("ERROR: Agent not found!")
        print("Available agents are:", ", ".join(available))
        return

    # ----- Create environment -----
    env = MinesweeperEnv(height=size, width=size, depth=size, num_mines=num_mines, render_mode="ansi")

    # ----- Load agent -----
    agent = load_agent(args.agent, env.action_space)

    # ----- Run -----
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        print(env.render())
        print("----")

        action = agent.select_action(obs)

        z, y, x = decode_action(action, env.depth, env.height, env.width)
        print(f"Agent clicked tile: (x={x}, y={y}, z={z})")

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        print(f"Reward: {reward}")
        print()

        done = terminated or truncated

    print(env.render())
    print("----\nFINAL EPISODE REWARD:", episode_reward)

if __name__ == "__main__":
    main()