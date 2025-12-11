import importlib, numpy as np, time, os
from backend.environment import MinesweeperEnv
from tqdm import tqdm

def load_agent(agent_name, action_space):
    try:
        module = importlib.import_module(f"agents.{agent_name}")
        AgentClass = getattr(module, "Agent")
        return AgentClass(action_space)
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Agent '{agent_name}' not found in agents/")

def run_episode(env, agent, seed: int | None = None):
    obs, _ = env.reset(seed=seed)
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

def animate_camera(env, frame_idx, num_frames):
    plotter = getattr(env, "_plotter", None)
    if plotter is None: return

    step_deg = 0.25
    plotter.camera.azimuth += step_deg

def save_win_frames(size:int, mines:int, seed:int, out_root, num_frames:int, frames_per_step:int=5, agent:object=None, agent_name:str=""): 
    agent_name = agent_name.split("_")[0]
    out_dir = os.path.join(out_root, f"s{size}_m{mines}_{agent_name}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    env3d = MinesweeperEnv(height=size, width=size, depth=size, num_mines=mines, render_mode="3d")
    obs, _ = env3d.reset(seed=seed)

    done = False
    frame_idx = 0

    env3d.render_frame(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"), off_screen=True)
    frame_idx += 1

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env3d.step(action)

        for _ in range(frames_per_step):
            animate_camera(env3d, frame_idx, num_frames)
            env3d.render_frame(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"), off_screen=True)
            frame_idx += 1

        done = terminated or truncated

    try: env3d.close()
    except Exception: pass

    print(f"Saved winning 3D frames to: {out_dir}, total frames: {frame_idx}")

def summarize(values, name):
    values = np.array(values)
    print(f"\n{name}:")
    print(f"  min     = {values.min():.2f}")
    print(f"  max     = {values.max():.2f}")
    print(f"  mean    = {values.mean():.2f}")
    print(f"  median  = {np.median(values):.2f}")
    print(f"  var     = {values.var():.2f}")
    print(f"  stdev   = {values.std():.2f}")

def main(size:int, num_mines:int, agent_name:str, record_wins:bool=False):
    num_episodes = 5000
    
    env = MinesweeperEnv(
        height=size,
        width=size,
        depth=size,
        num_mines=num_mines,
        render_mode=None
    )

    agent = load_agent(agent_name, env.action_space)

    rewards = []
    moves = []
    wins = 0
    losses = 0
    saved = 0
    max_saves = 3

    print(f"Running {num_episodes} episodes with agent '{agent_name}'...\n")

    start_time = time.time()     # <-- NEW: start timer

    for i in range(num_episodes): # tqdm(range(num_episodes)):
        seed = int(np.random.randint(0, 2**31-1))
        result = run_episode(env, agent, seed=seed)

        rewards.append(result["reward"])
        moves.append(result["moves"])

        if result["won"]:
            wins += 1
            if record_wins: 
                if saved < max_saves and record_wins and result["moves"] >= 30:
                    save_win_frames(size, num_mines, seed, "win_frames", result["moves"], agent=agent, agent_name=agent_name) 
                    saved += 1
                elif saved >= max_saves: break
        else:
            losses += 1
    
    # Special case: no wins, we still need to output a loss 
    if wins == 0:
        print(f"No wins recorded, running episodes to capture losses...")
        for i in range(num_episodes): 
            seed = int(np.random.randint(0, 2**31-1))
            result = run_episode(env, agent)

            if saved < max_saves and result["moves"] >= 30:
                save_win_frames(size, num_mines, seed, "win_frames", result["moves"], agent=agent, agent_name=agent_name) 
                saved += 1

            elif saved >= max_saves: break

    end_time = time.time()       # <-- NEW: stop timer
    total_time = end_time - start_time
    avg_time = total_time / num_episodes

    if not record_wins: 
        # ---------- SUMMARY ----------
        summarize(rewards, "Total Reward")
        summarize(moves, "Number of Moves")

        print("\nWin / Loss:")
        print(f"  Wins   = {wins}")
        print(f"  Losses = {losses}")
        print(f"  Win rate = {wins / num_episodes:.3f}")

        # ---------- TIMING SUMMARY ----------
        print("\nTiming:")
        print(f"  Total time   = {total_time:.2f} seconds")
        print(f"  Avg per ep   = {avg_time:.4f} seconds")
        print(f"  Episodes/sec = {1.0 / avg_time:.2f}")

if __name__ == "__main__":
    main(5, 10, "bayesian_approximation_agent", True)