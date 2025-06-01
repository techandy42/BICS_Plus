"""
Module: src.visualize_benchmark

Load benchmark result JSONL files, compute average accuracies across context and depth dimensions, and generate a saved heatmap visualization.

Functions:
- load_results(result_dir): Load JSONL result files and aggregate accuracy by (context_length, depth_percentage).
- compute_matrix(stats): Compute an average accuracy matrix and axis values from aggregated stats.
- visualize(matrix, context_lengths, depth_percentages, provider, model): Create and save a heatmap of the accuracy data for a given provider/model.
- main(): Parse CLI arguments, load data, compute metrics, and invoke visualization.

Authors: Andrew Kim, Derek Sheen, Hokyung (Andy) Lee
Emails: hyojaekim03@gmail.com (A. Kim), derek.s.prog@gmail.com (D. Sheen), techandy42@gmail.com (H. Lee)
Date: May 3, 2025
"""


import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(result_dir):
    """Load JSONL result files and aggregate accuracy by (context_length, depth_percentage)."""
    stats = {}
    for i in range(20):
        path = os.path.join(result_dir, f"bics_result_{i}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                cl = item.get("context_length")
                dp = item.get("depth_percentage")
                acc = item.get("accuracy")
                if cl is None or dp is None or acc is None:
                    continue
                stats.setdefault((cl, dp), []).append(acc)
    return stats


def compute_matrix(stats):
    """Compute average accuracy matrix from aggregated stats."""
    context_lengths = sorted({cl for (cl, _) in stats.keys()})
    depth_percentages = sorted({dp for (_, dp) in stats.keys()})
    matrix = []
    for dp in depth_percentages:
        row = []
        for cl in context_lengths:
            vals = stats.get((cl, dp), [])
            avg = np.mean(vals) if vals else np.nan
            row.append(avg)
        matrix.append(row)
    return matrix, context_lengths, depth_percentages


def visualize(matrix, context_lengths, depth_percentages, provider, model):
    """Plot and save heatmap of the accuracy matrix."""
    # Prepare axis labels
    x_labels = [str(cl) if cl < 1000 else f"{cl//1000}K" for cl in context_lengths]
    y_labels = [str(dp/100) for dp in depth_percentages]

    df = pd.DataFrame(matrix, index=y_labels, columns=x_labels)
    numeric_df = df.applymap(lambda x: np.nan if isinstance(x, str) else x)

    # Annotation matrix with trimmed decimals
    def fmt(x):
        if isinstance(x, (int, float)) and not np.isnan(x):
            s = f"{x:.2f}"
            if s.endswith("0"):
                s = s[:-1]
            return s
        return ""
    annot_matrix = df.applymap(fmt)

    # Create heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        numeric_df,
        annot=annot_matrix.values,
        fmt="",
        cmap="YlGnBu",
        annot_kws={"size": 20},  # larger accuracy labels
        vmin=0,
        vmax=100
    )

    # Add title with spacing
    plt.title(f"{provider}/{model} Benchmark Accuracy", fontsize=24, pad=15)
    plt.xlabel('Context Length (tokens)', fontsize=20)
    plt.ylabel('Target Depth', fontsize=20)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.tick_params(labelsize=20)

    # Save figure
    output_dir = os.path.join('data', 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{provider}_{model}_benchmark.png"
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and save benchmark results for a given provider and model."
    )
    parser.add_argument(
        '--provider', required=True,
        help="LLM provider (e.g., openai, anthropic)"
    )
    parser.add_argument(
        '--model', required=True,
        help="Model name (e.g., gpt-4.1-mini)"
    )
    args = parser.parse_args()

    result_dir = os.path.join(
        'data', 'result', f"{args.provider}_{args.model}"
    )
    stats = load_results(result_dir)
    matrix, cls, dps = compute_matrix(stats)
    visualize(matrix, cls, dps, args.provider, args.model)


if __name__ == '__main__':
    main()
