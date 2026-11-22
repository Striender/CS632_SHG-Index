import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.makedirs('plots', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

enron_csv = 'results/data/hnsw_vs_shg_comparison.csv'
sift_csv = 'results/data/hnsw_vs_shg_comparison_sift.csv'

df_enron = pd.read_csv(enron_csv)
df_sift = pd.read_csv(sift_csv)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# Process Enron
hnsw_e = df_enron[df_enron['method'] == 'Vanilla_HNSW'].sort_values('recall')
shg_e = df_enron[df_enron['method'] == 'Baseline_SHG'].sort_values('recall')
s1_e = df_enron[df_enron['method'] == 'Step1_SkipPrefetch'].sort_values('recall')
s2_e = df_enron[df_enron['method'] == 'Step2_SkipLookahead'].sort_values('recall')

recs_e = np.linspace(0.83, 0.99, 25)
ht_e = np.interp(recs_e, hnsw_e['recall'], hnsw_e['latency_ms'])
bt_e = np.interp(recs_e, shg_e['recall'], shg_e['latency_ms'])
s1t_e = np.interp(recs_e, s1_e['recall'], s1_e['latency_ms'])
s2t_e = np.interp(recs_e, s2_e['recall'], s2_e['latency_ms'])

ax1.axhline(1.0, color='#d62728', linestyle='--', linewidth=2.2, label='SHG Baseline (= 1.00x)')
ax1.plot(recs_e, bt_e / s1t_e, '^-.', color='#ff7f0e', label='SHG with Idea 1', linewidth=2.0, markersize=6)
ax1.plot(recs_e, bt_e / s2t_e, 'D-', color='#2ca02c', label='SHG + Idea 2', linewidth=2.5, markersize=7)
ax1.plot(recs_e, bt_e / ht_e, 'o--', color='#1f77b4', label='HNSW', linewidth=2.0, markersize=6)

ax1.set_title('Enron Dataset (d = 1,369, N = 94,987)\nSpeedup Factor vs. SHG Baseline across Target Recall', fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Target Recall@20', fontsize=11, fontweight='bold')
ax1.set_ylabel('Speedup Factor vs. SHG (x) [Higher is Better]', fontsize=11, fontweight='bold')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.set_xlim(0.82, 1.00)
ax1.set_ylim(0.90, 1.60)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax1.legend(loc='upper right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# Process SIFT
hnsw_s = df_sift[df_sift['method'] == 'Vanilla_HNSW'].sort_values('recall')
shg_s = df_sift[df_sift['method'] == 'Baseline_SHG'].sort_values('recall')
s1_s = df_sift[df_sift['method'] == 'Step1_SkipPrefetch'].sort_values('recall')
s2_s = df_sift[df_sift['method'] == 'Step2_SkipLookahead'].sort_values('recall')

recs_s = np.linspace(0.80, 0.998, 25)
ht_s = np.interp(recs_s, hnsw_s['recall'], hnsw_s['latency_ms'])
bt_s = np.interp(recs_s, shg_s['recall'], shg_s['latency_ms'])
s1t_s = np.interp(recs_s, s1_s['recall'], s1_s['latency_ms'])
s2t_s = np.interp(recs_s, s2_s['recall'], s2_s['latency_ms'])

ax2.axhline(1.0, color='#d62728', linestyle='--', linewidth=2.2, label='SHG Baseline (= 1.00x)')
ax2.plot(recs_s, bt_s / s1t_s, '^-.', color='#ff7f0e', label='SHG with Idea 1', linewidth=2.0, markersize=6)
ax2.plot(recs_s, bt_s / s2t_s, 'D-', color='#2ca02c', label='SHG + Idea 2', linewidth=2.5, markersize=7)
ax2.plot(recs_s, bt_s / ht_s, 'o--', color='#1f77b4', label='HNSW', linewidth=2.0, markersize=6)

ax2.set_title('SIFT1M Dataset (d = 128, N = 1,000,000)\nSpeedup Factor vs. SHG Baseline across Target Recall', fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel('Target Recall@20', fontsize=11, fontweight='bold')
ax2.set_ylabel('Speedup Factor vs. SHG (x) [Higher is Better]', fontsize=11, fontweight='bold')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.set_xlim(0.79, 1.005)
ax2.set_ylim(0.90, 2.65)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig('plots/speedup_vs_shg.png', dpi=300)
fig.savefig('results/plots/speedup_vs_shg.png', dpi=300)
print("Saved plots/speedup_vs_shg.png and results/plots/speedup_vs_shg.png")

# Save speedup summary table to CSV
speedup_records = []
for ef in [20, 24, 32, 48, 64, 96, 128, 192, 256, 512]:
    he = df_enron[(df_enron['method']=='Vanilla_HNSW') & (df_enron['ef']==ef)]['latency_us'].values[0]
    be = df_enron[(df_enron['method']=='Baseline_SHG') & (df_enron['ef']==ef)]['latency_us'].values[0]
    s1e = df_enron[(df_enron['method']=='Step1_SkipPrefetch') & (df_enron['ef']==ef)]['latency_us'].values[0]
    s2e = df_enron[(df_enron['method']=='Step2_SkipLookahead') & (df_enron['ef']==ef)]['latency_us'].values[0]

    hs = df_sift[(df_sift['method']=='Vanilla_HNSW') & (df_sift['ef']==ef)]['latency_us'].values[0]
    bs = df_sift[(df_sift['method']=='Baseline_SHG') & (df_sift['ef']==ef)]['latency_us'].values[0]
    s1s = df_sift[(df_sift['method']=='Step1_SkipPrefetch') & (df_sift['ef']==ef)]['latency_us'].values[0]
    s2s = df_sift[(df_sift['method']=='Step2_SkipLookahead') & (df_sift['ef']==ef)]['latency_us'].values[0]

    speedup_records.append({
        'ef': ef,
        'Enron_SHG_Base': 1.0,
        'Enron_Idea1_Speedup': round(be / s1e, 3),
        'Enron_Idea2_Speedup': round(be / s2e, 3),
        'Enron_HNSW_Speedup': round(be / he, 3),
        'SIFT_SHG_Base': 1.0,
        'SIFT_Idea1_Speedup': round(bs / s1s, 3),
        'SIFT_Idea2_Speedup': round(bs / s2s, 3),
        'SIFT_HNSW_Speedup': round(bs / hs, 3)
    })

pd.DataFrame(speedup_records).to_csv('results/data/speedup_summary_vs_shg.csv', index=False)
print("Saved results/data/speedup_summary_vs_shg.csv")
