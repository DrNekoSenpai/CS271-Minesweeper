import argparse, numpy as np, gymnasium as gym
from agents.dueling_cnn_agent import DuelingDCNNAgent
from backend.environment import MinesweeperEnv

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--mines", type=int, default=20)

    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    D, H, W = args.depth, args.height, args.width

    # Vectorized envs
    envs = gym.vector.SyncVectorEnv([
        make_env(D, H, W, args.mines) for _ in range(args.num_envs)
    ])

    obs, info = envs.reset()  # obs shape: (num_envs, H, W, D)

    agent = DuelingDCNNAgent(
        height=H, width=W, depth=D,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup=2000,
        target_update=1000
    )

    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    completed = 0

    for step in range(args.steps):
        actions = agent.select_actions(obs)

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        done = np.logical_or(terminated, truncated)

        # Store transitions
        for i in range(args.num_envs):
            agent.remember(obs[i], int(actions[i]), float(rewards[i]), next_obs[i], bool(done[i]))
            episode_rewards[i] += rewards[i]

            if done[i]:
                completed += 1
                if completed % 50 == 0:
                    print(f"[episodes={completed}] recent_reward={episode_rewards[i]:.1f}")
                episode_rewards[i] = 0.0

        obs = next_obs

        # Train
        stats = agent.optimize()
        if stats and step % 500 == 0:
            print(f"[step={step}] loss={stats['loss']:.4f}")

    agent.save("dueling_dcnn_3d.pt")
    print("Saved model to dueling_dcnn_3d.pt")


if __name__ == "__main__":
    main()
