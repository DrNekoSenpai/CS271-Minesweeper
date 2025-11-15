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

def main():
    # ----- Parse arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", "-a", type=str, default=None, help="Name of agent (matching agents/<name>.py)")
    args = parser.parse_args()

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
    env = MinesweeperEnv(depth=5, height=5, width=5, num_mines=10, render_mode="ansi")

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

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        print(f"Action: {action}, Reward: {reward}")
        print()

        done = terminated or truncated

    print("FINAL EPISODE REWARD:", episode_reward)

if __name__ == "__main__":
    main()