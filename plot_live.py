import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Set up command line arguments
parser = argparse.ArgumentParser(description="Live plot training results from a CSV log.")
parser.add_argument("--csv", default="logs/agent_weights_log.csv", help="Path to the training log CSV.")
parser.add_argument("--interval", type=int, default=5000, help="Update interval in milliseconds (default: 5000).")
args = parser.parse_args()

csv_path = args.csv

if not os.path.exists(csv_path):
    print(f"Waiting for {csv_path} to be created...")

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.canvas.manager.set_window_title('Live Training Metrics')

def update(frame):
    """This function is called repeatedly by FuncAnimation to update the graph."""
    if not os.path.exists(csv_path):
        return
    
    try:
        # Read the latest data
        df = pd.read_csv(csv_path)
    except Exception:
        # Failsafe in case we try to read exactly when the trainer is writing
        return 
        
    if df.empty:
        return
        
    # Clear the old lines
    ax1.clear()
    ax2.clear()
    
    # ─── Plot 1: Reward ──────────────────────────────────────────────────────────
    ax1.plot(df.index, df['reward'], alpha=0.3, color='royalblue', label='Raw Reward')
    
    # Moving average
    window = min(50, len(df))
    if window > 0:
        smoothed_reward = df['reward'].rolling(window=window, min_periods=1).mean()
        ax1.plot(df.index, smoothed_reward, color='darkblue', linewidth=2, label=f'{window}-Ep Moving Avg')
    
    ax1.set_title("Live Training Reward", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Total Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # ─── Plot 2: TD Error ────────────────────────────────────────────────────────
    ax2.plot(df.index, df['avg_td_error'], color='crimson', linewidth=2, label='Avg TD Error')
    ax2.set_title("Live Learning Error (TD Error)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Total Episodes Trained")
    ax2.set_ylabel("Error magnitude")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()

# Start the live animation loop
ani = FuncAnimation(fig, update, interval=args.interval, cache_frame_data=False)

print(f"Starting live plot for {csv_path} (updates every {args.interval/1000} seconds)")
print("Close the graph window to stop.")
plt.show()
