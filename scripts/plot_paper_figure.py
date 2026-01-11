#!/usr/bin/env python3
"""Generate paper Figure 6: Execution horizon and Inference delay vs. solve rate."""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

METHOD_CONFIG = {
    'oracle': {'label': 'Sync ($\Delta=0$)', 'color': '#f39c12', 'marker': '*', 'linestyle': '--'},
    'vlash': {'label': 'Ours', 'color': '#2ecc71', 'marker': '^', 'linestyle': '-'},
    'vlash_w_noise': {'label': 'Ours (w/ noise)', 'color': '#2ecc71', 'marker': 'o', 'linestyle': '-'},
    'realtime': {'label': 'RTC', 'color': '#3498db', 'marker': 's', 'linestyle': '-'},
    'naive': {'label': 'Naive', 'color': '#e74c3c', 'marker': 'D', 'linestyle': '-'},
}


def compute_wilson_interval(p_hat, n, confidence=0.95):
    """Compute Wilson score 95% confidence interval."""
    if n == 0:
        return 0, 0
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def plot_panel(df, ax, x_col, x_filter_fn, xlabel, xlim, xticks, ylim):
    """Generic plotting function for both panels."""
    df_filtered = x_filter_fn(df)
    grouped = df_filtered.groupby(['method', x_col])['returned_episode_solved'].agg(['mean', 'count']).reset_index()
    
    for method, cfg in METHOD_CONFIG.items():
        if method not in df['method'].unique():
            continue
        data = grouped[grouped['method'] == method].sort_values(x_col)
        if len(data) == 0:
            continue
        
        x, y = data[x_col].values, data['mean'].values
        ci = [compute_wilson_interval(sr, cnt * 1024) for sr, cnt in zip(y, data['count'].values)]
        
        ax.plot(x, y, label=cfg['label'], color=cfg['color'], marker=cfg['marker'],
                linestyle=cfg['linestyle'], linewidth=2.5, markersize=9,
                markeredgewidth=1.2, markeredgecolor='white', alpha=0.85)
        ax.fill_between(x, [c[0] for c in ci], [c[1] for c in ci], alpha=0.15, color=cfg['color'])
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Solve Rate', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=11)


def main(input_file: str, output_file: str, dpi: int = 300):
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows, methods: {df['method'].unique().tolist()}")
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, 3.15))
    
    # Left: Execution horizon vs solve rate (fixed delay=1)
    plot_panel(df, ax_left, 'execute_horizon',
               lambda d: d[(d['delay'] == 1) & (d['execute_horizon'] != 8)],
               'Execution Horizon, K', [0.6, 7.4], [1, 2, 3, 4, 5, 6, 7], [0.73, 0.92])
    
    # Right: Inference delay vs solve rate (horizon = max(delay, 1))
    def filter_right(d):
        d = d[d['delay'] <= 4].copy()
        d['expected'] = d['delay'].apply(lambda x: max(x, 1))
        return d[d['execute_horizon'] == d['expected']]
    
    plot_panel(df, ax_right, 'delay', filter_right,
               'Inference Delay, $\Delta$', [-0.3, 4.3], [0, 1, 2, 3, 4], [0.48, 1.02])
    ax_right.set_ylabel('')
    
    # Shared legend
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=11, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}, {output_file.replace('.png', '.pdf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper Figure 6")
    parser.add_argument("--input-file", type=str, default="eval_output/results.csv")
    parser.add_argument("--output-file", type=str, default="eval_outputs/paper_figure6.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.dpi)
