"""Entry point for training.

Usage::

    # Single run (default)
    python train.py                          # uses config.yaml
    python train.py --config my_cfg.yaml     # custom config

    # Parallel group run — runs all configs/<GROUP>*.yaml in parallel
    python train.py --group A                # all A*.yaml configs
    python train.py --group B                # all B*.yaml configs
    python train.py --group A B C            # multiple groups at once
    python train.py --group all              # every config in configs/
    python train.py --group A --workers 2    # limit parallel workers
"""

import argparse
import glob
import multiprocessing
import os
import random as rnd
import re
import sys

import numpy as np
import yaml

from agents import make_agent
from envs import build_env
from trainer import Trainer
from utils.logging import CSVLogger, ConsoleLogger, CompositeLogger


# ── Single-config training ────────────────────────────────────────────────────

def _train_one(config_path: str, human: bool = False, resume: str | None = None) -> None:
    """Load one config and run the full training loop.

    Designed to be called either directly or from a worker process.
    All output is prefixed with the config name so parallel runs are
    easy to tell apart in the terminal.
    """
    tag = f"[{os.path.splitext(os.path.basename(config_path))[0]}]"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if human:
        cfg["environment"]["render_mode"] = "human"

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    rnd.seed(seed)
    np.random.seed(seed)
    try:
        from agents.dqn import TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            import torch
            torch.manual_seed(seed)
    except Exception:
        pass

    # ── Setup ─────────────────────────────────────────────────────────────────
    env, mode_cfg, mode_label = build_env(cfg)
    agent = make_agent(mode_cfg, env)

    save_path = cfg.get("save_path")
    if resume:
        if resume == "latest":
            if not save_path:
                print(f"{tag} WARNING: Cannot auto-resume — save_path not set. Starting fresh.")
            else:
                pattern = save_path + "*.pt"
                candidates = glob.glob(pattern)
                if not candidates:
                    print(f"{tag} No checkpoints found matching '{pattern}'. Starting fresh.")
                else:
                    def _extract_ep(path):
                        m = re.search(r'_ep(\d+)\.pt$', path)
                        if m:
                            return int(m.group(1))
                        return float("inf") if (path.endswith(".pt") and "_ep" not in path) else -1

                    latest_ckpt = max(candidates, key=_extract_ep)
                    resume_path = latest_ckpt[:-3]
                    print(f"{tag} Auto-resuming from: {resume_path}")
                    agent.load(resume_path)
        else:
            print(f"{tag} Resuming from: {resume}")
            agent.load(resume)
    elif save_path:
        agent.load(save_path)

    # ── Loggers ───────────────────────────────────────────────────────────────
    log_every = cfg["environment"].get("log_interval", 25)
    backends = [ConsoleLogger(interval=log_every, prefix=tag)]
    log_path = cfg.get("log_path")
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
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
    print(f"\n{tag} Done.")


def _worker(args: tuple) -> None:
    """Multiprocessing entry point — unpacks args for _train_one."""
    config_path, human, resume = args
    try:
        _train_one(config_path, human=human, resume=resume)
    except Exception as exc:
        import traceback
        tag = f"[{os.path.splitext(os.path.basename(config_path))[0]}]"
        print(f"\n{tag} WORKER CRASHED:\n{traceback.format_exc()}", flush=True)
        raise exc


# ── Group resolution ─────────────────────────────────────────────────────────

def _resolve_configs(groups: list[str], configs_dir: str = "configs") -> list[str]:
    """Return sorted list of config paths matching the requested groups.

    Config files are organised into group subdirectories whose names follow
    the pattern ``<LETTER>_<description>`` (e.g. ``A_algorithms``)::

        configs/
          A_algorithms/   A1_....yaml  A2_....yaml  ...
          B_action_sets/  B1_....yaml  ...
          C_frame_stack/  ...

    The ``--group`` flag matches by the **leading letter** of the directory name,
    so ``--group A`` resolves to whichever directory starts with ``A`` (e.g.
    ``A_algorithms``). This means the folders can be renamed freely without
    changing the CLI flag.

    Special value ``'all'`` matches every YAML in every subdirectory.
    """
    if not os.path.isdir(configs_dir):
        sys.exit(f"ERROR: configs directory '{configs_dir}' not found. "
                 f"Create it or run from the project root.")

    # All subdirectories in configs_dir
    subdirs = sorted(
        d for d in os.listdir(configs_dir)
        if os.path.isdir(os.path.join(configs_dir, d))
    )

    if "all" in [g.lower() for g in groups]:
        all_cfgs = sorted(glob.glob(os.path.join(configs_dir, "*", "*.yaml")))
        if not all_cfgs:
            sys.exit(f"ERROR: No YAML files found under '{configs_dir}/'.") 
        return all_cfgs

    matched: list[str] = []
    for group in groups:
        prefix = group.upper()
        # Find any subdir whose name starts with the group letter
        hit_dirs = [d for d in subdirs if d.upper().startswith(prefix)]
        if not hit_dirs:
            print(f"WARNING: No directory starting with '{prefix}' found in "
                  f"'{configs_dir}' — skipping '{group}'.")
            continue
        for group_dir_name in hit_dirs:
            group_dir = os.path.join(configs_dir, group_dir_name)
            hits = sorted(glob.glob(os.path.join(group_dir, "*.yaml")))
            if not hits:
                print(f"WARNING: No YAML files found in '{group_dir}'.")
            matched.extend(hits)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = [p for p in matched if not (p in seen or seen.add(p))]
    return unique


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a RL agent on CarRacing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Single-config mode ────────────────────────────────────────────────────
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config (default: config.yaml). Ignored when --group is used.",
    )
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None,
        help="Resume from checkpoint. Omit value → auto-find latest.",
    )
    parser.add_argument(
        "--human", action="store_true",
        help="Render environment in a window (sets render_mode='human').",
    )

    # ── Group / parallel mode ─────────────────────────────────────────────────
    parser.add_argument(
        "--group", nargs="+", metavar="GROUP",
        help=(
            "Run all configs/GROUP*.yaml files in parallel. "
            "Pass one or more group letters (A, B, C …) or 'all'. "
            "Example: --group A B"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help=(
            "Max parallel workers for --group mode. "
            "Defaults to the number of matched configs (one per config)."
        ),
    )
    parser.add_argument(
        "--configs-dir", default="configs", metavar="DIR",
        help="Directory to scan for group configs (default: configs/).",
    )

    args = parser.parse_args()

    # ── Group mode ────────────────────────────────────────────────────────────
    if args.group:
        configs = _resolve_configs(args.group, configs_dir=args.configs_dir)
        if not configs:
            sys.exit("No matching configs found. Nothing to run.")

        n_workers = args.workers or len(configs)
        print(f"{'─'*62}")
        print(f"  Group mode  : {', '.join(args.group)}")
        print(f"  Configs     : {len(configs)}")
        print(f"  Workers     : {n_workers}")
        print(f"{'─'*62}")
        for p in configs:
            print(f"  • {p}")
        print(f"{'─'*62}\n")

        worker_args = [(p, args.human, args.resume) for p in configs]

        # Use 'spawn' to avoid CUDA / gym fork issues
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            pool.map(_worker, worker_args)

        print(f"\n{'─'*62}")
        print(f"  All {len(configs)} runs finished.")
        print(f"{'─'*62}")
        return

    # ── Single-config mode (original behaviour) ───────────────────────────────
    _train_one(args.config, human=args.human, resume=args.resume)


if __name__ == "__main__":
    main()
