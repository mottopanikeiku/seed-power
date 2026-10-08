"""Make SVGs only from the committed confirmation summaries and outcomes."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    summary = json.loads(Path("results/summary.json").read_text())
    with Path("results/outcomes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    destination = Path("figures")
    destination.mkdir(exist_ok=True)
    # A fixed hash salt and no date keep regenerated SVGs byte-stable.
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "seed-power"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, task in zip(axes, ("CartPole-v1", "Acrobot-v1"), strict=True):
        groups = [row for row in summary["per_comparison"]
                  if row["task"] == task and not row["null_control"]]
        rates = np.asarray([group["detection_rate"] for group in groups])
        low = np.asarray([group["wilson_95_low"] for group in groups])
        high = np.asarray([group["wilson_95_high"] for group in groups])
        x = np.arange(len(groups))
        ax.bar(x, rates, color="#286a9b", width=.65)
        ax.errorbar(x, rates, yerr=[rates - low, high - rates], fmt="none", color="#162b3e", capsize=3)
        ax.axhline(summary["target_power"], color="#b34d30", linestyle="--", label="80% plan")
        ax.set_xticks(x, [f'{row["a"]}\nvs {row["b"]}\n(n={row["plans"]})' for row in groups])
        ax.set_title(task)
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Fresh-seed detection frequency (Wilson 95% interval)")
    axes[1].legend(loc="lower right")
    fig.suptitle("An 80% pilot calculation is not an 80% detection guarantee")
    fig.savefig(destination / "calibration.svg", metadata={"Date": None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, task in zip(axes, ("CartPole-v1", "Acrobot-v1"), strict=True):
        points = [row for row in rows if row["task"] == task and row["null_control"] == "False"]
        x = [float(row["pilot_effect_a_minus_b"]) for row in points]
        y = [float(row["confirmation_effect_a_minus_b"]) for row in points]
        ax.scatter(x, y, s=22, alpha=.65, color="#286a9b")
        bound = max([abs(value) for value in x + y] + [1]) * 1.05
        ax.plot([-bound, bound], [-bound, bound], linestyle="--", color="#b34d30", label="Same gap")
        ax.axhline(0, color="grey", linewidth=.5)
        ax.axvline(0, color="grey", linewidth=.5)
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), title=task,
               xlabel="Pilot mean gap: A − B", ylabel="Fresh-seed mean gap: A − B")
        ax.grid(alpha=.15)
        ax.legend()
    fig.suptitle("Pilot gaps and independent confirmation gaps (raw return points)")
    fig.savefig(destination / "pilot_transfer.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
