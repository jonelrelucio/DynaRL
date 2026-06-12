import yaml
import numpy as np
from car_racing import build_env, make_agent, run_episode


def evaluate(n_episodes: int = 5, render: bool = True):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Force human rendering for watching the agent
    if render:
        cfg["environment"]["render_mode"] = "human"

    env, mode_cfg, mode_label = build_env(cfg)
    agent = make_agent(mode_cfg, env)
    obs_cfg = cfg.get("observation", {})

    # Load trained weights
    save_path = cfg.get("save_path")
    if not save_path:
        raise ValueError("No save_path set in config.yaml")
    agent.load(save_path)

    # Disable exploration completely
    agent.epsilon = 0.0

    print(f"Evaluating: {mode_label}")
    rewards = []

    for ep in range(n_episodes):
        # training=False → no update(), no replay buffer writes
        reward = run_episode(env, agent, obs_cfg, training=False, seed=ep)
        rewards.append(reward)
        print(f"  Episode {ep+1}: reward = {reward:.2f}")

    print(f"\n  Mean: {np.mean(rewards):.2f}  Std: {np.std(rewards):.2f}")
    env.close()


if __name__ == "__main__":
    evaluate(n_episodes=5, render=True)
