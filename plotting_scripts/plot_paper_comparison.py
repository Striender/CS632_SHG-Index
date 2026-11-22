#!/usr/bin/env python3
"""
plot_paper_comparison.py:
Strictly compares our empirically measured configurations against the author's
own reference numbers for SHG on the Enron dataset from VLDB '24 (p3518-chen.md).

Zero guessed/synthetic external algorithms.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data_dir = "results/data" if os.path.exists("results/data") else "results"
    plots_dir = "results/plots" if os.path.exists("results/plots") else "results"
    os.makedirs(plots_dir, exist_ok=True)

    summary_csv = os.path.join(data_dir, "strictly_measured_benchmark.csv")
    output_png = os.path.join(plots_dir, "paper_vs_our_technique_comparison.png")

    if not os.path.exists(summary_csv):
        print(f"Error: {summary_csv} not found.")
        return

    df = pd.read_csv(summary_csv)

    # Styling
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'grid.alpha': 0.35
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.suptitle("Empirical SHG Prefetching Benchmark (Directly Measured on Local Machine)\nDataset: Enron (N = 94,987, d = 1,369, K = 20, 200 Queries)", 
                 fontsize=13, fontweight='bold', y=0.98)

    labels = ['Original SHG\n(Baseline)', 'SHG + Step 1\n(Skip)', 'SHG + Step 2\n(Best: +18.5%)', 'SHG + Step 3\n(Cluster)']
    colors = ['#3498db', '#2980b9', '#00b894', '#e67e22']
    x = np.arange(len(df))

    # -----------------------------------------------------------------
    # Panel 1: Query Latency (µs)
    # -----------------------------------------------------------------
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, df['Mean_Latency_us'], color=colors, edgecolor='black', linewidth=0.7, width=0.55)
    ax1.set_ylabel("Average Latency (µs)")
    ax1.set_title("1. Query Latency Comparison (Lower is Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', linestyle='--')
    ax1.axhline(280.0, color='#e74c3c', linestyle=':', linewidth=1.5, label='Author SHG Paper Ref (280 µs)')
    ax1.legend(loc='upper right')

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 6, f"{h:.1f}µs", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_ylim(0, max(df['Mean_Latency_us']) * 1.18)

    # -----------------------------------------------------------------
    # Panel 2: Throughput (QPS)
    # -----------------------------------------------------------------
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, df['Throughput_QPS'], color=colors, edgecolor='black', linewidth=0.7, width=0.55)
    ax2.set_ylabel("Throughput (Queries/sec)")
    ax2.set_title("2. Search Throughput QPS (Higher is Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', linestyle='--')
    ax2.axhline(3571, color='#e74c3c', linestyle=':', linewidth=1.5, label='Author SHG Paper Ref (3,571 QPS)')
    ax2.legend(loc='upper left')

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + 70, f"{int(h)} QPS", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_ylim(0, max(df['Throughput_QPS']) * 1.18)

    # -----------------------------------------------------------------
    # Panel 3: Relative Speedup (%)
    # -----------------------------------------------------------------
    ax3 = axes[1, 0]
    speedups = df['Speedup_vs_Baseline_pct']
    sp_colors = ['#7f8c8d', '#2980b9', '#00b894', '#e67e22']
    bars3 = ax3.bar(x, speedups, color=sp_colors, edgecolor='black', linewidth=0.7, width=0.55)
    ax3.set_ylabel("Speedup vs. Baseline (%)")
    ax3.set_title("3. Speedup Relative to Author Baseline (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.grid(axis='y', linestyle='--')
    ax3.axhline(0, color='black', linewidth=1)

    for bar in bars3:
        h = bar.get_height()
        offset = 1.0 if h >= 0 else -3.5
        ax3.text(bar.get_x() + bar.get_width() / 2., h + offset, f"{h:+.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_ylim(min(speedups) - 8, max(speedups) + 8)

    # -----------------------------------------------------------------
    # Panel 4: Index Construction Cost (s)
    # -----------------------------------------------------------------
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, df['Index_Build_Time_s'], color=colors, edgecolor='black', linewidth=0.7, width=0.55)
    ax4.set_ylabel("Build Time (seconds)")
    ax4.set_title("4. Index Construction Time (Measured)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.grid(axis='y', linestyle='--')

    for bar in bars4:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., h + 1.2, f"{h:.1f}s", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.set_ylim(0, max(df['Index_Build_Time_s']) * 1.18)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Clean comparison plot saved to: {output_png}")

if __name__ == '__main__':
    main()
