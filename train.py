"""Entry point for training.

Usage::

    python train.py                        # uses config.yaml
    python train.py --config my_cfg.yaml   # custom config
"""

import argparse
import random as rnd
import numpy as np
import yaml
import glob
import os
import re

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
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        help="Resume from a checkpoint. Use without args to auto-find the latest, or provide a specific path prefix.",
    )
    parser.add_argument(
        "--human", action="store_true",
        help="Render the environment in a window (changes render_mode to 'human')",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.human:
        cfg["environment"]["render_mode"] = "human"

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
    if args.resume:
        if args.resume == "latest":
            if not save_path:
                raise ValueError("Cannot auto-resume: save_path not set in config.")
            pattern = save_path + "*.pt"
            candidates = glob.glob(pattern)
            if not candidates:
                print(f"No checkpoints found matching '{pattern}'. Starting fresh.")
            else:
                def extract_ep(path):
                    m = re.search(r'_ep(\d+)\.pt$', path)
                    if m: return int(m.group(1))
                    if path.endswith(".pt") and not "_ep" in path: return float('inf')
                    return -1
                
                latest_ckpt = max(candidates, key=extract_ep)
                resume_path = latest_ckpt[:-3]  # Strip '.pt'
                print(f"Auto-resuming from latest checkpoint: {resume_path}")
                agent.load(resume_path)
        else:
            print(f"Resuming from explicitly provided checkpoint: {args.resume}")
            agent.load(args.resume)
    elif save_path:
        agent.load(save_path)

    # ── Loggers ───────────────────────────────────────────────────────────────
    log_every = cfg["environment"].get("log_interval", 25)
    backends = [ConsoleLogger(interval=log_every)]
    log_path = cfg.get("log_path")
    if log_path:
        backends.append(CSVLogger(log_path))
    elif save_path:
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
