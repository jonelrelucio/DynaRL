import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_path, save_file):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        print("Make sure you are pointing to the correct log file.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("The CSV file is empty. Wait for training to complete a few episodes.")
        return

    # Create a professional-looking figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # ─── Plot 1: Reward ──────────────────────────────────────────────────────────
    # Plot raw reward lightly in the background
    ax1.plot(df.index, df['reward'], alpha=0.3, color='royalblue', label='Raw Reward')
    
    # Add a moving average to show the trend clearly
    window = min(50, len(df))
    if window > 0:
        smoothed_reward = df['reward'].rolling(window=window, min_periods=1).mean()
        ax1.plot(df.index, smoothed_reward, color='darkblue', linewidth=2, label=f'{window}-Ep Moving Avg')
    
    ax1.set_title("Training Reward over Time", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Total Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # ─── Plot 2: TD Error ────────────────────────────────────────────────────────
    # TD Error shows how surprised the network is. It usually spikes and then drops.
    ax2.plot(df.index, df['avg_td_error'], color='crimson', linewidth=2, label='Avg TD Error')
    ax2.set_title("Learning Error (TD Error) over Time", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Total Episodes Trained")
    ax2.set_ylabel("Error magnitude")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Finalize and save
    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved as: {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training results from a CSV log file.")
    parser.add_argument("--csv", default="logs/agent_weights_log.csv", help="Path to the training log CSV.")
    parser.add_argument("--out", default="training_results.png", help="Filename to save the graph image.")
    
    args = parser.parse_args()
    plot_results(args.csv, args.out)
