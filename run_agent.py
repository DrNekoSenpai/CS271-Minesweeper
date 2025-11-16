import importlib, argparse, os 
from backend.environment import MinesweeperEnv

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

def decode_action(action, height, width):
    z, rem = divmod(action, height * width)
    y, x = divmod(rem, width)
    return z, y, x

def main():
    # ----- Parse arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", "-a", type=str, default=None, help="Name of agent (matching agents/<name>.py)")
    parser.add_argument("--render", "-r", type=str, choices=["ansi", "3d"], default='ansi')
    args = parser.parse_args()

    size = 5
    num_mines = 15

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
    env = MinesweeperEnv(height=size, width=size, depth=size, num_mines=num_mines, render_mode=args.render)

    # ----- Load agent -----
    agent = load_agent(args.agent, env.action_space)

    # ----- Run -----
    obs, info = env.reset()
    done = False
    episode_reward = 0

    good_moves = 0        # moves that revealed at least one safe tile
    total_moves = 0       # total moves made
    mine_hits = 0         # moves that hit a mine (should always be one - sanity check)

    while not done:
        if args.render == 'ansi': 
            print(env.render())
            print("----")

        elif args.render == '3d': 
            env.render()
            cmd = input("Press Enter to step, 'n' for new game, 'q' to quit: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "n":
                obs, info = env.reset()
                episode_reward = 0.0
                done = False
                good_moves = 0
                total_moves = 0
                mine_hits = 0
                continue
                    
        action = agent.select_action(obs)
        z, y, x = decode_action(action, env.height, env.width)

        # ----- Step and evaluate move -----
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        total_moves += 1

        # Good move = revealed at least one tile
        if reward > 0:
            good_moves += 1
        # Hit a mine
        elif reward < 0 and obs[z, y, x] == -10:
            mine_hits += 1

        print(f"Agent clicked tile: (x={x}, y={y}, z={z})")
        print(f"Reward: {reward}")
        print(f"Total reward: {episode_reward}")
        print(f"Good moves so far: {good_moves}, Total moves: {total_moves}, Mine hits: {mine_hits}")
        print("----")

        done = terminated or truncated

    print(env.render())
    print("----\nFINAL EPISODE REWARD:", episode_reward)
    print(f"Total moves: {total_moves}")
    print(f"Good moves: {good_moves}")
    print(f"Mine hits: {mine_hits}")


if __name__ == "__main__":
    main()