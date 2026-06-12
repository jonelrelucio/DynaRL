"""Evaluate a trained agent.

Usage::

    python evaluate.py                           # 5 episodes, rendered
    python evaluate.py --config my_cfg.yaml
    python evaluate.py --episodes 10 --no-render
"""

import argparse
import numpy as np
import yaml

from agents import make_agent
from envs import build_env
from trainer import run_episode


def evaluate(config: str = "config.yaml",
             n_episodes: int = 5,
             render: bool = True) -> None:
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    if render:
        cfg["environment"]["render_mode"] = "human"

    env, mode_cfg, mode_label = build_env(cfg)
    agent = make_agent(mode_cfg, env)
    obs_cfg = cfg.get("observation", {})

    save_path = cfg.get("save_path")
    if not save_path:
        raise ValueError("No save_path set in config.yaml")
    agent.load(save_path)

    # Disable exploration completely during evaluation
    agent.epsilon = 0.0

    print(f"Evaluating: {mode_label}")
    rewards = []

    for ep in range(n_episodes):
        reward = run_episode(env, agent, obs_cfg, training=False, seed=ep)
        rewards.append(reward)
        print(f"  Episode {ep+1}: reward = {reward:.2f}")

    print(f"\n  Mean: {np.mean(rewards):.2f}  Std: {np.std(rewards):.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CarRacing agent.")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--episodes",  type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    evaluate(config=args.config, n_episodes=args.episodes, render=not args.no_render)
