#!/usr/bin/env python3
"""
plot_results.py:
Reads strictly measured benchmark data from CSV and plots a clean 4-panel figure
focusing entirely on the techniques we actually executed on our local machine.

No external guessed/synthetic algorithms.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data_dir = "results/data"
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)

    summary_csv = os.path.join(data_dir, "strictly_measured_benchmark.csv")
    query_csv = os.path.join(data_dir, "query_level_comparison.csv")
    output_png = os.path.join(plots_dir, "shg_prefetching_benchmark.png")

    if not os.path.exists(summary_csv) or not os.path.exists(query_csv):
        print(f"Error: Missing CSV files in {results_dir}/")
        return

    df_summary = pd.read_csv(summary_csv)
    df_queries = pd.read_csv(query_csv)

    # Set up matplotlib aesthetics
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'figure.titlesize': 14,
        'grid.alpha': 0.3
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Shortcut-Enabled Hierarchical Graph (SHG) Prefetching Benchmark\nStrictly Measured Results on Enron Dataset (N=94,987, d=1369, K=20, 200 Queries)", 
                 fontsize=13, fontweight='bold', y=0.98)

    labels = ['Original SHG\n(Baseline)', 'SHG + Step 1\n(Skip)', 'SHG + Step 2\n(Best: +18.5%)', 'SHG + Step 3\n(Cluster)']
    colors = ['#3498db', '#2980b9', '#00b894', '#e67e22']

    # -------------------------------------------------------------
    # Panel 1: Average Latency Comparison (µs)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    x = np.arange(len(df_summary))
    bars1 = ax1.bar(x, df_summary['Mean_Latency_us'], color=colors, edgecolor='black', linewidth=0.8, width=0.55)
    ax1.set_ylabel("Average Query Latency (µs)")
    ax1.set_title("1. Query Latency Comparison (Lower is Better)")
    ax1.grid(axis='y', linestyle='--')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    # Add reference line for Paper SHG reported average
    ax1.axhline(280.0, color='#e74c3c', linestyle=':', linewidth=1.5, label='Author Reported SHG Ref (280 µs)')
    ax1.legend(loc='upper right', frameon=True)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 6, f"{h:.1f} µs", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_ylim(0, max(df_summary['Mean_Latency_us']) * 1.18)

    # -------------------------------------------------------------
    # Panel 2: Throughput (QPS - Queries Per Second)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, df_summary['Throughput_QPS'], color=colors, edgecolor='black', linewidth=0.8, width=0.55)
    ax2.set_ylabel("Throughput (Queries / Sec)")
    ax2.set_title("2. Search Throughput QPS (Higher is Better)")
    ax2.grid(axis='y', linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    base_qps = df_summary.loc[0, 'Throughput_QPS']
    ax2.axhline(base_qps, color='#34495e', linestyle=':', linewidth=1.5, label=f'Baseline ({base_qps:.0f} QPS)')
    ax2.legend(loc='upper left', frameon=True)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + 70, f"{int(h)} QPS", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_ylim(0, max(df_summary['Throughput_QPS']) * 1.18)

    # -------------------------------------------------------------
    # Panel 3: Relative Speedup (%) vs. Baseline
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    speedups = df_summary['Speedup_vs_Baseline_pct']
    sp_colors = ['#7f8c8d', '#2980b9', '#00b894', '#e67e22']
    bars3 = ax3.bar(x, speedups, color=sp_colors, edgecolor='black', linewidth=0.8, width=0.55)
    ax3.set_ylabel("Speedup vs. Baseline SHG (%)")
    ax3.set_title("3. Prefetching Ablation Speedup (Relative to Baseline)")
    ax3.grid(axis='y', linestyle='--')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)

    for bar in bars3:
        h = bar.get_height()
        offset = 1.0 if h >= 0 else -3.5
        ax3.text(bar.get_x() + bar.get_width() / 2., h + offset, f"{h:+.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_ylim(min(speedups) - 8, max(speedups) + 8)

    # -------------------------------------------------------------
    # Panel 4: Cumulative Distribution Function (CDF) of Query Latency
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    modes_to_plot = [
        ('baseline_latency_us', 'Baseline SHG (266.9 µs)', '#3498db', '--'),
        ('step1_latency_us', 'Step 1: Skip (259.1 µs)', '#2980b9', ':'),
        ('step2_latency_us', 'Step 2: Lookahead (217.4 µs)', '#00b894', '-'),
        ('step3_latency_us', 'Step 3: Cluster (325.9 µs)', '#e67e22', '-.')
    ]

    for col, label, color, style in modes_to_plot:
        sorted_lat = np.sort(df_queries[col].dropna())
        cdf = np.linspace(0, 1, len(sorted_lat))
        ax4.plot(sorted_lat, cdf, label=label, color=color, linestyle=style, linewidth=2.2)

    ax4.axvline(280.0, color='#e74c3c', linestyle=':', label='Author SHG Ref (280 µs)')

    ax4.set_xlabel("Query Latency (µs)")
    ax4.set_ylabel("Fraction of Queries (CDF)")
    ax4.set_title("4. Latency CDF Across 200 Queries")
    ax4.grid(True, linestyle='--')
    ax4.legend(loc='lower right', frameon=True)
    ax4.set_xlim(30, 700)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Empirical benchmark plot saved to: {output_png}")

if __name__ == '__main__':
    main()
