"""Live plot for training results, supporting group-based layouts.

Usage::

    # Original single-CSV mode
    python plot_live.py --csv logs/A_algorithms/A1_tabular_discrete_qlearning_log.csv

    # One window per group (each shows all runs in that group overlaid)
    python plot_live.py --group A
    python plot_live.py --group A B C

    # One combined window with every run across all groups
    python plot_live.py --group all

    # Change refresh rate (milliseconds)
    python plot_live.py --group A --interval 3000
"""

import argparse
import glob
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# -- Colour palette ------------------------------------------------------------
# tab10 gives 10 distinct colours; we cycle for larger groups.
_CMAP = matplotlib.colormaps["tab10"]


def _color(i: int):
    return _CMAP(i % 10)


def _label(csv_path: str) -> str:
    """Human-readable label: strip directory and trailing '_log' suffix."""
    base = os.path.splitext(os.path.basename(csv_path))[0]
    if base.endswith("_log"):
        base = base[:-4]
    return base


# -- Log discovery -------------------------------------------------------------

def _resolve_group_csvs(groups: list[str], logs_dir: str = "logs") -> dict[str, list[str]]:
    """Return ``{group_label: [csv_paths]}`` for the requested groups.

    Directory layout expected::

        logs/
          A_algorithms/   A1_..._log.csv  A2_..._log.csv  ...
          B_action_sets/  B1_..._log.csv  ...

    ``--group A`` matches any sub-directory starting with ``A``.
    ``--group all`` collapses everything into one entry keyed ``"all"``.
    """
    if not os.path.isdir(logs_dir):
        sys.exit(f"ERROR: logs directory '{logs_dir}' not found.")

    subdirs = sorted(
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    )

    # Special case: combine everything into one window
    if "all" in [g.lower() for g in groups]:
        all_csvs = sorted(glob.glob(os.path.join(logs_dir, "*", "*.csv")))
        if not all_csvs:
            sys.exit(f"ERROR: No CSV files found under '{logs_dir}/'.")
        return {"all (every group)": all_csvs}

    result: dict[str, list[str]] = {}
    for group in groups:
        prefix = group.upper()
        hit_dirs = [d for d in subdirs if d.upper().startswith(prefix)]
        if not hit_dirs:
            print(f"WARNING: No log directory starting with '{prefix}' found in '{logs_dir}'.")
            continue
        csvs: list[str] = []
        group_label = hit_dirs[0]          # e.g. "A_algorithms"
        for d in hit_dirs:
            csvs.extend(sorted(glob.glob(os.path.join(logs_dir, d, "*.csv"))))
        if not csvs:
            print(f"WARNING: No CSV files found for group '{group}'.")
        else:
            result[group_label] = csvs

    if not result:
        sys.exit("ERROR: No CSV files resolved. Nothing to plot.")
    return result


# -- Figure builder ------------------------------------------------------------

def _build_figure(group_label: str, csv_paths: list[str]) -> tuple:
    """Create a matplotlib figure for *group_label* watching *csv_paths*.

    Returns ``(fig, update_fn)`` where ``update_fn(frame)`` refreshes the plot.
    """
    n = len(csv_paths)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.canvas.manager.set_window_title(f"Live Training — {group_label}")
    fig.suptitle(f"Group: {group_label}  ({n} run{'s' if n != 1 else ''})",
                 fontsize=13, fontweight="bold")

    def update(_frame):
        ax1.clear()
        ax2.clear()

        any_data = False
        for i, csv_path in enumerate(csv_paths):
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if df.empty:
                continue

            any_data = True
            color = _color(i)
            name  = _label(csv_path)

            # -- Reward -------------------------------------------------------
            ax1.plot(df["episode"], df["reward"],
                     alpha=0.2, color=color, linewidth=1)
            window = min(50, len(df))
            if window > 0:
                smoothed = df["reward"].rolling(window=window, min_periods=1).mean()
                ax1.plot(df["episode"], smoothed,
                         color=color, linewidth=2, label=name)

            # -- TD Error -----------------------------------------------------
            ax2.plot(df["episode"], df["avg_td_error"],
                     color=color, linewidth=1.5, alpha=0.85, label=name)

        # -- Axes decoration ---------------------------------------------------
        ax1.set_title("Reward  (faint = raw,  solid = 50-ep moving avg)",
                      fontsize=11)
        ax1.set_ylabel("Total Reward")
        ax1.grid(True, linestyle="--", alpha=0.5)
        if any_data:
            ax1.legend(loc="upper left", fontsize=8,
                       ncol=max(1, n // 6), framealpha=0.7)

        ax2.set_title("Avg TD Error", fontsize=11)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Error magnitude")
        ax2.grid(True, linestyle="--", alpha=0.5)
        if any_data:
            ax2.legend(loc="upper right", fontsize=8,
                       ncol=max(1, n // 6), framealpha=0.7)

        plt.tight_layout()

    # Initial draw so the window isn't blank at startup
    update(0)
    return fig, update


# -- Single-CSV mode (original behaviour) -------------------------------------

def _build_single_figure(csv_path: str) -> tuple:
    """Original single-CSV live plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title("Live Training Metrics")

    def update(_frame):
        if not os.path.exists(csv_path):
            return
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return
        if df.empty:
            return

        ax1.clear()
        ax2.clear()

        ax1.plot(df["episode"], df["reward"],
                 alpha=0.3, color="royalblue", label="Raw Reward")
        window = min(50, len(df))
        if window > 0:
            smoothed = df["reward"].rolling(window=window, min_periods=1).mean()
            ax1.plot(df["episode"], smoothed,
                     color="darkblue", linewidth=2,
                     label=f"{window}-Ep Moving Avg")
        ax1.set_title("Live Training Reward", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Total Reward")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.7)

        ax2.plot(df["episode"], df["avg_td_error"],
                 color="crimson", linewidth=2, label="Avg TD Error")
        ax2.set_title("Live Learning Error (TD Error)", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Error magnitude")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()

    update(0)
    return fig, update


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live plot training results from CSV logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to a single CSV log (original single-run mode).",
    )
    parser.add_argument(
        "--group", nargs="+", metavar="GROUP",
        help=(
            "Group letter(s) to watch (e.g. A, B, C) or 'all'. "
            "One window is opened per group. "
            "Example: --group A B"
        ),
    )
    parser.add_argument(
        "--logs-dir", default="logs", metavar="DIR",
        help="Root directory containing group log sub-folders (default: logs/).",
    )
    parser.add_argument(
        "--interval", type=int, default=5000, metavar="MS",
        help="Refresh interval in milliseconds (default: 5000).",
    )
    args = parser.parse_args()

    animations: list[FuncAnimation] = []   # keep refs to avoid GC

    if args.group:
        # -- Group mode --------------------------------------------------------
        group_map = _resolve_group_csvs(args.group, logs_dir=args.logs_dir)

        print(f"{'-'*60}")
        print(f"  Live plot - {len(group_map)} window(s)")
        for label, csvs in group_map.items():
            print(f"  [{label}]  {len(csvs)} CSV(s):")
            for c in csvs:
                status = "exists" if os.path.exists(c) else "waiting..."
                print(f"      {c}  ({status})")
        print(f"{'-'*60}\n")

        for group_label, csv_paths in group_map.items():
            fig, update_fn = _build_figure(group_label, csv_paths)
            ani = FuncAnimation(fig, update_fn, interval=args.interval,
                                cache_frame_data=False)
            animations.append(ani)

    else:
        # -- Single-CSV mode (original behaviour) ------------------------------
        csv_path = args.csv or "logs/agent_weights_log.csv"
        if not os.path.exists(csv_path):
            print(f"Waiting for {csv_path} to be created...")
        else:
            print(f"Starting live plot for {csv_path}")

        fig, update_fn = _build_single_figure(csv_path)
        ani = FuncAnimation(fig, update_fn, interval=args.interval,
                            cache_frame_data=False)
        animations.append(ani)

    print("Close all graph windows to stop.")
    plt.show()


if __name__ == "__main__":
    main()
