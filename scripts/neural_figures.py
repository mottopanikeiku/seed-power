"""Make legacy/PPO comparison SVGs only from committed summary JSON files.

Intervals describe heterogeneous fixed-comparison detection rates. PPO outcomes
are conditional on its fixed evaluation reset sets, not episode replicates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_groups(ax, groups, labels, colors):
    for index, (group, color) in enumerate(zip(groups, colors, strict=True)):
        rate = group["detection_rate"]
        if rate is None:
            ax.text(index, .04, "No executable\ncomparisons", ha="center", fontsize=9)
            continue
        ax.bar(index, rate, color=color, width=.62)
        ax.errorbar(index, rate,
                    yerr=[[max(0, rate - group["wilson_95_low"])],
                          [max(0, group["wilson_95_high"] - rate)]],
                    fmt="none", color="#162b3e", capsize=4)
        ax.text(index, min(1.035, group["wilson_95_high"] + .03),
                f'{100 * rate:.1f}% ({group["detections"]}/{group["plans"]})',
                ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def make_overview(legacy, neural):
    tasks = list(neural["per_task"])
    fig, axes = plt.subplots(1, len(tasks) + 1, figsize=(4 * (len(tasks) + 1), 4.5),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    labels = ["Linear search", "Neural PPO"]
    colors = ["#8b8f97", "#286a9b"]
    plot_groups(axes[0], [legacy["nonnull"], neural["nonnull"]], labels, colors)
    axes[0].set_title("All executable different-variant plans")
    for ax, task in zip(axes[1:], tasks, strict=True):
        plot_groups(ax, [legacy["per_task"][task], neural["per_task"][task]], labels, colors)
        ax.set_title(task)
    for ax in axes:
        ax.axhline(neural["target_power"], color="#b34d30", linestyle="--",
                   label=f'{neural["target_power"]:.0%} pilot planning target')
    axes[0].set_ylabel("Fresh-root-seed detection frequency")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Pilot-derived budgets: linear search and neural PPO\n"
                 "Descriptive Wilson 95% intervals; PPO conditional on fixed evaluation resets")
    return fig


def make_comparisons(neural):
    tasks = list(neural["per_task"])
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks, strict=True):
        groups = [group for group in neural["per_comparison"] if group["task"] == task]
        labels = [f'{group["a"]}\nvs {group["b"]}'
                  + ("\n(same-variant control)" if group["null_control"] else "")
                  + f'\n{group["plans"]}/{group["attempted_plans"]} executable'
                  for group in groups]
        plot_groups(ax, groups, labels,
                    ["#8b8f97" if group["null_control"] else "#286a9b" for group in groups])
        ax.axhline(neural["target_power"], color="#b34d30", linestyle="--")
        ax.set_title(task)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Fresh-root-seed detection frequency")
    fig.suptitle("All planned PPO comparisons, including ineligible cells and controls\n"
                 "Descriptive Wilson 95% intervals; different variants are not known alternatives")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", default="results/summary.json")
    parser.add_argument("--neural", default="results/neural/summary.json")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()
    legacy = json.loads(Path(args.legacy).read_text())
    neural = json.loads(Path(args.neural).read_text())
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    for filename, fig in [("neural_vs_linear.svg", make_overview(legacy, neural)),
                          ("neural_comparisons.svg", make_comparisons(neural))]:
        fig.savefig(destination / filename)
        plt.close(fig)


if __name__ == "__main__":
    main()
