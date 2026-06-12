"""Entry point for training.

Usage::

    python train.py                        # uses config.yaml
    python train.py --config my_cfg.yaml   # custom config
"""

import argparse
import random as rnd
import numpy as np
import yaml

from agents import make_agent
from agents.dqn import TORCH_AVAILABLE
from envs import build_env
from trainer import Trainer
from utils.logging import CSVLogger, ConsoleLogger, CompositeLogger

try:
    import torch
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Train a RL agent on CarRacing.")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    rnd.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

    # ── Setup ─────────────────────────────────────────────────────────────────
    env, mode_cfg, mode_label = build_env(cfg)
    agent = make_agent(mode_cfg, env)

    save_path = cfg.get("save_path")
    if save_path:
        agent.load(save_path)

    # ── Loggers ───────────────────────────────────────────────────────────────
    log_every = cfg["environment"].get("log_interval", 25)
    backends = [ConsoleLogger(interval=log_every)]
    if save_path:
        backends.append(CSVLogger(save_path + "_log.csv"))
    logger = CompositeLogger(*backends)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        cfg=cfg, env=env, agent=agent,
        loggers=[logger], mode_label=mode_label,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
