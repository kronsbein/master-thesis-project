import logging
import os
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parent_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
sys.path.append(parent_dir)
sys.path.append(os.path.join(os.path.abspath(""), ".."))

from src.utils import create_dirs

name_mapping: Dict[str, Any] = {
    "kmeans": "K-Means",
    "grep": "Grep",
    "pagerank": "PageRank",
    "sgd": "SGD",
    "sort": "Sort",
    "bayes": "Bayes",
    "join": "Join",
    "lr": "LR",
    "regression": "Regression"
}


def plot_eval_data_with_metric(
    df: Dict[str, Any],
    datatype: str,
    experiment: str,
    target: str,
    workload: str,
    error_metric: str,
) -> None:
    """Function to plot evaluation data with error metric
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(22, 18))
    plt.ylim(0.0, 1.0)
    ax = sns.lineplot(data=df,
                      x="Data Points",
                      y=error_metric,
                      hue="Model Type",
                      lw=1,
                      errorbar="sd")
    ax.set_title(f"{name_mapping[workload]}", fontsize=72, pad=20)
    ax.set_xlabel("Data Points", fontsize=56, labelpad=20)
    plt.xticks(fontsize=56)
    ax.set_ylabel(f"{error_metric.upper()}", fontsize=56, labelpad=20)
    plt.yticks(fontsize=56)
    plt.grid(True, which='both', axis='both', linestyle='-', color='lightgray', alpha=0.7)
    plt.legend(fontsize=46,
               bbox_to_anchor=(0.5, -0.2),
               loc="upper center",
               ncol=len(df["Model Type"].unique()))
    plt.tight_layout()
    path: str = f"./{datatype}/figures/{experiment}/{error_metric}/{target}/"
    create_dirs(path=path)
    plt.savefig(os.path.join(path, f"{experiment}_{target}_{workload}_{error_metric}.pdf"))
    plt.close()
    logging.info(f"Plotting for {workload}_{target}_{error_metric} done.")


def plot_resource_ratio_by_model(
    df: pd.DataFrame,
    datatype: str,
    experiment: str,
    models: List[str] | np.ndarray,
) -> None:
    """Function to plot resource ratio by model
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(32, 24))
    # customize outliers to be less visible
    # flierprops = dict(marker='o', markerfacecolor='gray', markersize=2, linestyle='none', alpha=0.5)
    custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    if models:
        df = df[df['model'].isin(models)]
    ax = sns.boxplot(x="model",
                     y="noised_gt_runtime_resource_ratio",
                     data=df,
                     palette=custom_colors,
                     hue="model",
                     showfliers=False,
                     legend="full")
    ax.set_title("Noised Ground Truth Runtime Resource Ratio by Model", fontsize=72, pad=20)
    ax.set_xlabel("Model", fontsize=64, labelpad=20)
    plt.xticks(fontsize=64)
    ax.set_ylabel("Resource Ratio (%)", fontsize=64, labelpad=20)
    plt.yticks(fontsize=64)
    plt.grid(True, which='both', axis='both', linestyle='-', color='lightgray', alpha=0.7)
    plt.legend(fontsize=50,
               loc="upper center",
               bbox_to_anchor=(0.5, -0.15),
               ncol=len(df['model'].unique()))
    plt.ylim(top=100)
    plt.tight_layout()
    path: str = f"./{datatype}/figures/{experiment}/noised_gt_comparison/"
    create_dirs(path=path)
    plt.savefig(os.path.join(path, f"{experiment}_resource_ratio_by_model.pdf"))
    plt.close()
    logging.info(f"Plotting for resource_ratio_by_model done.")


def plot_runtime_target_met_by_model(
    df: pd.DataFrame,
    datatype: str,
    experiment: str,
    models: List[str] | np.ndarray,
    scale_to_one: bool = True,
) -> None:
    """Function to plot runtime target met by model for each workload"""
    # group by workload and model sum counts
    grouped_df = df.groupby(["workload", "model"]).agg({
        "target_met_count": "sum",
        "target_not_met_count": "sum"
    }).reset_index()

    # normalize the counts
    if scale_to_one:
        grouped_df["total"] = grouped_df["target_met_count"] + grouped_df["target_not_met_count"]
        grouped_df["target_met_count"] = grouped_df["target_met_count"] / grouped_df["total"]
        grouped_df[
            "target_not_met_count"] = grouped_df["target_not_met_count"] / grouped_df["total"]
    else:
        max_value = max(grouped_df["target_met_count"].max(),
                        grouped_df["target_not_met_count"].max())
        grouped_df["target_met_count"] = grouped_df["target_met_count"] / max_value
        grouped_df["target_not_met_count"] = grouped_df["target_not_met_count"] / max_value

    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # unique workloads and models
    workloads = grouped_df["workload"].unique()
    workloads_name = [name_mapping[key] for key, _ in name_mapping.items() if key in workloads]
    if len(models) == 0 or len(models) > len(grouped_df["model"].unique()):
        models = grouped_df["model"].unique()

    bar_width = 0.25
    x = np.arange(len(workloads))
    sns.set_style("whitegrid")
    _, ax = plt.subplots(figsize=(34, 24))

    # Plotting the bars
    for i, model in enumerate(models):
        model_data = grouped_df[grouped_df["model"] == model]
        target_met = model_data["target_met_count"].values
        target_not_met = model_data["target_not_met_count"].values
        ax.bar(x + i * bar_width,
               target_met,
               width=bar_width,
               color=color_palette[i],
               label=f"{model} - Met")
        ax.bar(x + i * bar_width,
               target_not_met,
               bottom=target_met,
               width=bar_width,
               color=color_palette[i],
               alpha=0.3,
               label=f"{model} - Not Met")

    ax.set_title("Synthetic Runtime Target Comparison", fontsize=72, pad=20)
    ax.set_xlabel("Workload", fontsize=64, labelpad=20)
    ax.set_ylabel("Normalized Count", fontsize=64, labelpad=20)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(workloads_name, fontsize=64)
    ax.tick_params(axis='y', labelsize=58)
    plt.grid(True, which='both', axis='both', linestyle='-', color='lightgray', alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(),
              by_label.keys(),
              title="Synthetic Target",
              loc="upper center",
              ncol=len(models),
              bbox_to_anchor=(0.5, -0.15),
              fontsize=50,
              title_fontsize=50)

    plt.tight_layout()

    # Save the plot
    path: str = f"./{datatype}/figures/{experiment}/noised_gt_comparison/"
    create_dirs(path=path)
    plt.savefig(os.path.join(path, f"{experiment}_runtime_target_met_by_model_and_workload.pdf"))
    plt.close()

    logging.info(f"Plotting for runtime_target_met_by_model and workload done.")
