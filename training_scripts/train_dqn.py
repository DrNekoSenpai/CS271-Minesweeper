import argparse
import numpy as np
import gymnasium as gym
import os
import sys
import matplotlib.pyplot as plt 
import re
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from agents.dueling_cnn_agent import DuelingDCNNAgent
from backend.environment import MinesweeperEnv
from matplotlib.ticker import MaxNLocator

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

def main(size:int, mines:int, num_steps:int):
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=size, help="Board size (creates size x size x size cube)")
    parser.add_argument("--mines", type=int, default=mines, help="Number of mines")
    parser.add_argument("--num-steps", type=int, default=num_steps, help="Total training steps")
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel environments (higher = more GPU usage)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (higher = more GPU usage)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    
    # Use command line args if provided
    size = args.size
    mines = args.mines
    num_steps = args.num_steps

    directory = f"./metrics/s{size}-m{mines}"
    if not os.path.exists(directory): os.makedirs(directory)

    # Vectorized envs
    envs = gym.vector.SyncVectorEnv([
        make_env(size, size, size, mines) for _ in range(args.num_envs)
    ])

    obs, info = envs.reset()  # obs shape: (num_envs, H, W, D)

    # Initialize agent
    # NOTE: Hybrid mode (safe-action heuristic) is DISABLED by default in agent.__init__
    # This forces the DQN to actually learn instead of relying on the heuristic as a crutch
    agent = DuelingDCNNAgent(
        height=size, width=size, depth=size,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup=5000,
        target_update=200, 
        buffer_size=200000, 
        eps_decay_steps=150000,
        # !!!!! 5070 pytorch issue workaround !!!!!
        # device="cpu"  # Force CPU for compatibility - comment this out to use GPU if available
    )
    
    print(f"Using device: {agent.device}")
    print(f"Batch size: {args.batch_size}, Num envs: {args.num_envs}")

    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    episode_safe_moves = np.zeros(args.num_envs, dtype=np.float32)
    episode_safe_tiles = np.zeros(args.num_envs, dtype=np.float32)
    completed = 0
    
    # Track last 100 completed episodes for metrics
    recent_rewards = []
    recent_safe_moves = []
    recent_safe_tiles = []
    recent_wins = []  # Track actual wins (reward == 500.0)

    # Look for training checkpoints first, then pretrained as fallback
    checkpoint_path = f"dqn-checkpoint-s{size}-m{mines}"
    checkpoints = sorted([f for f in os.listdir(".") if checkpoint_path in f and f.endswith(".pth")], key=lambda x: int(x.split("-")[-1].split(".")[0]))
    
    is_pretrained = False  # Track if loading from pretraining
    if checkpoints:
        load_path = checkpoints[-1]  # Latest training checkpoint
        print(f"Found training checkpoint: {load_path}")
    else:
        # Check for both naming conventions: dqn-pretrained-* and dqn-pretrain-*
        pretrained_patterns = [
            f"dqn-pretrained-s{size}-m{mines}.pth",
            f"dqn-pretrain-s{size}-m{mines}"  # Matches dqn-pretrain-s5-m5-*.pth
        ]
        
        load_path = None
        for pattern in pretrained_patterns:
            # Look for exact match first
            if os.path.exists(pattern):
                load_path = pattern
                is_pretrained = True
                print(f"Found pretrained checkpoint: {load_path}")
                break
            
            # Look for pattern matches (e.g., dqn-pretrain-s5-m5-30000.pth)
            pretrain_checkpoints = sorted(
                [f for f in os.listdir(".") if pattern in f and f.endswith(".pth")],
                key=lambda x: int(x.split("-")[-1].split(".")[0])
            )
            if pretrain_checkpoints:
                load_path = pretrain_checkpoints[-1]  # Latest pretrain checkpoint
                is_pretrained = True
                print(f"Found pretrained checkpoint: {load_path}")
                break
        
        if load_path is None:
            load_path = f"dqn-checkpoint-0.pth"

    save_every = 5000
    start_step = 0
    logs = []

    metrics_dict = {
        "steps": [], 
        "reward": [], 
        "safe_moves": [], 
        "safe_tiles": []
    }

    loss_dict = {
        "steps": [],
        "loss": [] 
    }
    
    wins_dict = {
        "steps": [],
        "wins": []
    }

    if not args.fresh and os.path.exists(load_path):
        try:
            start_step = agent.load_checkpoint(load_path)
            print(f"Loaded checkpoint {load_path} from step {start_step}")
            
            # if loading pretrained model, then we need
            # to preserve learned policy instead of overwriting with random exploration
            if is_pretrained:
                agent.eps_start = 0.2  # Start at 20% exploration instead of 100%
                agent.eps_decay_steps = 300000  # Slower decay (2x longer)
                agent.total_steps = 0  # Reset for new epsilon schedule
                
                # Lower learning rate for fine-tuning to preserve pretrained knowledge
                for param_group in agent.optim.param_groups:
                    param_group['lr'] = args.lr * 0.5  # Half the learning rate
                
                print(f"[PRETRAINED] Adjusted hyperparameters:")
                print(f"  - Epsilon: {agent.eps_start} -> {agent.eps_end} over {agent.eps_decay_steps} steps")
                print(f"  - Learning rate: {args.lr * 0.5} (50% of normal for fine-tuning)")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"Warning: Checkpoint architecture mismatch (old 1-channel vs new 3-channel model)")
                print(f"Starting fresh training instead.")
                start_step = 0
            else:
                raise

    if start_step > 0:
        print(f"Loading historical metrics from previous training...")
        with open(f"./metrics/s{size}-m{mines}/metrics.log", "r", encoding="utf-8") as file: 
            lines = file.readlines() 
        
        # Parse complete metrics format: step, reward, safe_moves, safe_tiles, wins
        metrics_pattern = r"\[step=(\d+)/(\d+)\] reward=([\d\.\-]+) safe_moves=([\d\.\-]+) safe_tiles=([\d\.\-]+) wins=(\d+)/100"
        metrics_pattern_old = r"\[step=(\d+)\] reward=([\d\.\-]+) safe_moves=([\d\.\-]+) safe_tiles=([\d\.\-]+)"
        
        for line in lines: 
            # Try new complete format first (with wins)
            match = re.search(metrics_pattern, line)
            if match:
                steps, _, reward, safe_moves, safe_tiles, wins = match.groups()
                metrics_dict["steps"].append(int(steps))
                metrics_dict["reward"].append(float(reward))
                metrics_dict["safe_moves"].append(float(safe_moves))
                metrics_dict["safe_tiles"].append(float(safe_tiles))
                wins_dict["steps"].append(int(steps))
                wins_dict["wins"].append(int(wins))
            else:
                # Fall back to old format (no wins)
                match = re.search(metrics_pattern_old, line)
                if match: 
                    steps, reward, safe_moves, safe_tiles = match.groups()
                    metrics_dict["steps"].append(int(steps))
                    metrics_dict["reward"].append(float(reward))
                    metrics_dict["safe_moves"].append(float(safe_moves))
                    metrics_dict["safe_tiles"].append(float(safe_tiles))
        
        print(f"Loaded {len(metrics_dict['steps'])} historical metric data points")
        print(f"Loaded {len(wins_dict['steps'])} historical win rate data points")

        with open(f"./metrics/s{size}-m{mines}/loss.log", "r", encoding="utf-8") as file: 
            lines = file.readlines() 

        loss_pattern = r"\[step=(\d+)\]: (.*)"
        for line in lines: 
            match = re.search(loss_pattern, line)
            if match: 
                steps, loss_value = match.groups()
                loss_dict["steps"].append(int(steps))
                loss_dict["loss"].append(float(loss_value))

    else: 
        # Don't clear log files even with --fresh to preserve training history
        # Just start with empty in-memory data structures
        # Create files if they don't exist
        if not os.path.exists(f"./metrics/s{size}-m{mines}/metrics.log"):
            with open(f"./metrics/s{size}-m{mines}/metrics.log", "w", encoding="utf-8") as file: 
                file.write("")

        if not os.path.exists(f"./metrics/s{size}-m{mines}/loss.log"):
            with open(f"./metrics/s{size}-m{mines}/loss.log", "w", encoding="utf-8") as file: 
                file.write("")

    # Timing for progress tracking
    start_time = time.time()
    last_print_time = start_time
    last_print_step = start_step

    for step in range(start_step, num_steps + 1):
        actions = agent.select_actions(obs)

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        done = np.logical_or(terminated, truncated)

        safe_batch = infos.get("safe_move", np.zeros_like(rewards))
        tile_batch = infos.get("safe_tiles", np.zeros_like(rewards))

        # Store transitions
        for i in range(args.num_envs):
            agent.remember(obs[i], int(actions[i]), float(rewards[i]), next_obs[i], bool(done[i]))

            episode_rewards[i] += rewards[i]
            episode_safe_moves[i] += safe_batch[i]
            episode_safe_tiles[i] += tile_batch[i]
            
            # Track actual wins: terminal reward == 500.0 (revealed all non-mine tiles)
            if done[i] and rewards[i] == 500.0:
                recent_wins.append(1)
            elif done[i]:
                recent_wins.append(0)

            if done[i]:
                completed += 1
                
                # Track this completed episode
                recent_rewards.append(episode_rewards[i])
                recent_safe_moves.append(episode_safe_moves[i])
                recent_safe_tiles.append(episode_safe_tiles[i])
                
                # Print stats every 100 episodes (average of last 100)
                if completed % 100 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    steps_done = step - start_step
                    steps_remaining = num_steps - step
                    
                    # Calculate rate and ETA
                    if steps_done > 0:
                        steps_per_sec = steps_done / elapsed
                        eta_seconds = steps_remaining / steps_per_sec
                        eta_hours = eta_seconds / 3600
                        
                        # Recent rate (since last print)
                        time_since_last = current_time - last_print_time
                        steps_since_last = step - last_print_step
                        recent_rate = steps_since_last / time_since_last if time_since_last > 0 else 0
                        
                        last_print_time = current_time
                        last_print_step = step
                    else:
                        steps_per_sec = 0
                        eta_hours = 0
                        recent_rate = 0
                    
                    avg_reward = np.mean(recent_rewards[-100:])
                    avg_safe_moves = np.mean(recent_safe_moves[-100:])
                    avg_safe_tiles = np.mean(recent_safe_tiles[-100:])
                    
                    # True wins: episodes that ended with reward == 500.0
                    actual_wins = sum(recent_wins[-100:]) if len(recent_wins) >= 100 else sum(recent_wins)
                    episodes_counted = min(100, len(recent_wins))
                    
                    print(f"[step={step}/{num_steps}] reward={avg_reward:.1f} wins={actual_wins}/{episodes_counted} | "
                          f"{recent_rate:.1f} steps/s | ETA: {eta_hours:.1f}h")
                    logs.append((step, avg_reward, avg_safe_moves, avg_safe_tiles, actual_wins))

                episode_rewards[i] = 0.0
                episode_safe_moves[i] = 0.0
                episode_safe_tiles[i] = 0.0

        obs = next_obs

        # train every n steps (helps with efficiency with vectorized envs)
        stats = None
        if step % 4 == 0: # train every 4 steps
            stats = agent.optimize()

        if step % save_every == 0 and step != start_step: 
            save_path = f"{checkpoint_path}-{step}.pth"
            agent.save_checkpoint(save_path, step)

            print(f"Saved checkpoint: {save_path}")
            with open(f'./metrics/s{size}-m{mines}/metrics.log', 'a', encoding='utf-8') as file: 
                for l in logs: 
                    s, reward, safe_moves, safe_tiles, wins = l
                    file.write(f"[step={s}/{num_steps}] reward={reward:.1f} safe_moves={safe_moves:.1f} safe_tiles={safe_tiles:.1f} wins={wins}/100\n")

                    metrics_dict["steps"].append(s)
                    metrics_dict["reward"].append(reward)
                    metrics_dict["safe_moves"].append(safe_moves)
                    metrics_dict["safe_tiles"].append(safe_tiles)
                    wins_dict["steps"].append(s)
                    wins_dict["wins"].append(wins)
                    
                file.write(f"Saved checkpoint: {save_path}\n")
                logs = []

            # Only log loss when training occurred
            if stats is not None:
                with open(f'./metrics/s{size}-m{mines}/loss.log', 'a', encoding="utf-8") as file: 
                    file.write(f"[step={step}]: loss={stats['loss']:.6f} grad_norm={stats.get('grad_norm', 0):.6f} q_mean={stats.get('q_mean', 0):.6f} target_mean={stats.get('target_mean', 0):.6f}\n")
                    loss_dict["loss"].append(np.round(stats['loss'], 6))
                    loss_dict["steps"].append(step)

            # rm old checkpoints except 100k milestones
            pattern = re.compile(rf"{checkpoint_path}-(\d+)\.pth")
            for fname in os.listdir("."): 
                match = pattern.match(fname)
                if not match: continue 

                file_step = int(match.group(1))
                is_milestone = (file_step % 100000 == 0) and (file_step > 0)
                if file_step < step and not is_milestone:
                    os.remove(fname)
                    print(f"Removed old checkpoint: {fname}")

        if step % save_every == 0: 
            plt.figure(figsize=(10, 6))
            plt.plot(loss_dict["steps"], loss_dict["loss"], label="Loss", linestyle="-") 
            plt.xlabel("Step")
            plt.ylabel("Loss Values")
            plt.title(f"Loss Values for size={size}, mines={mines}")
            plt.grid(True)
            plt.savefig(f"./metrics/s{size}-m{mines}/loss-{step}.png", dpi=600)
            plt.close()
            print(f"Saved loss curve to loss-{step}.png")

            plt.figure(figsize=(10, 6))
            plt.plot(metrics_dict["steps"], metrics_dict["reward"], label="Rewards", linestyle="-")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.xlabel("Step")
            plt.ylabel("Average Reward")
            plt.title(f"Reward Curve for size={size}, mines={mines}")
            plt.grid(True)
            plt.savefig(f"./metrics/s{size}-m{mines}/rewards-{step}.png", dpi=600)
            plt.close()
            print(f"Saved reward curve to rewards-{step}.png")

            plt.figure(figsize=(10, 6))
            plt.plot(metrics_dict["steps"], metrics_dict["safe_moves"], label="Safe Moves", linestyle="-")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.xlabel("Step")
            plt.ylabel("Average Safe Moves")
            plt.title(f"Safe Moves for size={size}, mines={mines}")
            plt.grid(True)
            plt.savefig(f"./metrics/s{size}-m{mines}/safe_moves-{step}.png", dpi=600)
            plt.close()
            print(f"Saved safe moves curve to safe_moves-{step}.png")

            plt.figure(figsize=(10, 6))
            plt.plot(metrics_dict["steps"], metrics_dict["safe_tiles"], label="Safe Tiles", linestyle="-")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.xlabel("Step")
            plt.ylabel("Safe Tiles")
            plt.title(f"Safe Tiles for size={size}, mines={mines}")
            plt.grid(True)
            plt.savefig(f"./metrics/s{size}-m{mines}/safe_tiles-{step}.png", dpi=600)
            plt.close()
            print(f"Saved safe tiles curve to safe_tiles-{step}.png")            
            
            plt.figure(figsize=(10, 6))
            plt.plot(wins_dict["steps"], wins_dict["wins"], label="Win Rate", linestyle="-", color="green")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            plt.xlabel("Step")
            plt.ylabel("Wins (out of 100)")
            plt.ylim(0, 100)
            plt.title(f"Win Rate Curve for size={size}, mines={mines}")
            plt.grid(True)
            plt.savefig(f"./metrics/s{size}-m{mines}/winrate-{step}.png", dpi=600)
            plt.close()
            print(f"Saved win rate curve to winrate-{step}.png")
            
            pattern = re.compile(r"(loss|rewards|safe_moves|safe_tiles|winrate)-(\d+)\.png")
            for fname in os.listdir(f"./metrics/s{size}-m{mines}/"): 
                match = pattern.match(fname)
                if not match: continue 

                file_step = int(match.group(2))
                if file_step < step: os.remove(f"./metrics/s{size}-m{mines}/{fname}")

if __name__ == "__main__":
    # Default: just run with command line args or defaults
    main(size=5, mines=10, num_steps=250000)