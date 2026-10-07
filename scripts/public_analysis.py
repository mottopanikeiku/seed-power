"""Describe every task/algorithm pair in the pinned released score cohort.

Observed-gap power is a plug-in Gaussian/Welch calculation, not true power,
not a significance result, and not the fraction of all published RL comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import nct, t

from seed_power.stats import plan_seeds, power_at_n

ROOT = Path(__file__).resolve().parents[1]
EFFECTS = (0.2, 0.5, 0.8, 1.0)
PAIR_FIELDS = [
    "claim_type", "task", "algorithm_a", "variant_a", "algorithm_b", "variant_b",
    "observed_seeds_a", "observed_seeds_b", "mean_a", "mean_b", "variance_a",
    "variance_b", "mean_difference", "standardized_mean_difference", "status",
    "plugin_power_observed_allocation", "below_80_percent", "planned_equal_n_observed_gap",
    "planned_equal_n_d_0.2", "planned_equal_n_d_0.5", "planned_equal_n_d_0.8",
    "planned_equal_n_d_1.0",
]
CAVEATS = [
    "Observed-gap calculations reuse the data to estimate both gap and variance; they are not "
    "prospective power or evidence of significance.",
    "This is the complete final cohort of one owner-authored public GitHub benchmark, not a "
    "representative survey or a fraction of all published RL comparisons.",
    "Scores are undiscounted episode-return means at each run's stopping policy, not scores at "
    "equal training steps. Repeated evaluation controls stopping, including at initialization.",
    "The same 100 evaluation episode seeds are reused across checkpoints, runs and algorithms. "
    "Episodes are not training seeds; inference is conditional on that fixed evaluation set.",
    "The release describes independent training runs; matching seed labels do not justify a "
    "paired test. Independence of every RNG/hardware source cannot be established from scores.",
    "Gaussian seed-level scores and fixed plug-in variances are assumptions, not verified facts. "
    "Ceiling effects and qualification thresholds can distort these distributions.",
    "All pairs sharing algorithms are dependent; their descriptive below-threshold fraction "
    "has no binomial confidence interval. No familywise significance claims are made.",
    "A zero total sample variance is reported as non-estimable, not replaced by a tiny SD.",
]


def plugin_power(effect: float, va: float, vb: float, na: int, nb: int, alpha=0.05) -> float:
    """Fixed-df noncentral-t approximation at the actual observed allocation."""
    if na == nb:
        return power_at_n(effect, va, vb, na, alpha=alpha)
    ca, cb = va / na, vb / nb
    df = (ca + cb) ** 2 / (ca * ca / (na - 1) + cb * cb / (nb - 1))
    critical = t.isf(alpha / 2, df)
    nc = abs(effect) / math.sqrt(ca + cb)
    return float(nct.sf(critical, df, nc) + nct.cdf(-critical, df, nc))


def load_scores(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        key = (row["task"], row["algorithm"], row["variant"], int(row["seed"]))
        if key in seen:
            raise ValueError("Duplicate training seed identity")
        seen.add(key)
        row["score"] = float(row["score"])
        row["seed"] = int(row["seed"])
        if not math.isfinite(row["score"]):
            raise ValueError("Nonfinite score")
        if row["score_units"] != "undiscounted_episode_return_at_stopping":
            raise ValueError("Incompatible score units")
    if not rows:
        raise ValueError("No observed public scores")
    return rows


def analyze(rows: list[dict], alpha=0.05, target=0.8) -> tuple[list[dict], dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["algorithm"], row["variant"])].append(row["score"])
    tasks = sorted({key[0] for key in groups})
    pairs = []
    for task in tasks:
        task_keys = sorted(k for k in groups if k[0] == task)
        for ka, kb in itertools.combinations(task_keys, 2):
            a, b = np.asarray(groups[ka]), np.asarray(groups[kb])
            if min(a.size, b.size) < 2:
                raise ValueError("At least two training runs per task/algorithm required")
            va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
            ma, mb = float(np.mean(a)), float(np.mean(b))
            gap = ma - mb
            valid = va + vb > 0
            pooled_sd = math.sqrt((va + vb) / 2)
            power = plugin_power(gap, va, vb, a.size, b.size, alpha) if valid else None
            pair = {
                "claim_type": "plugin_model", "task": task, "algorithm_a": ka[1],
                "variant_a": ka[2], "algorithm_b": kb[1], "variant_b": kb[2],
                "observed_seeds_a": int(a.size), "observed_seeds_b": int(b.size),
                "mean_a": ma, "mean_b": mb, "variance_a": va, "variance_b": vb,
                "mean_difference": gap,
                "standardized_mean_difference": gap / pooled_sd if valid else None,
                "status": "estimated" if valid else "zero_total_sample_variance",
                "plugin_power_observed_allocation": power,
                "below_80_percent": power < target if valid else None,
                "planned_equal_n_observed_gap": plan_seeds(gap, va, vb, target, alpha),
            }
            for d in EFFECTS:
                pair[f"planned_equal_n_d_{d}"] = plan_seeds(d * pooled_sd, va, vb, target, alpha)
            pairs.append(pair)
    eligible = [p for p in pairs if p["status"] == "estimated"]

    def describe(selected):
        estimable = [p for p in selected if p["status"] == "estimated"]
        below = sum(p["below_80_percent"] for p in estimable)
        return {
            "enumerated_pairs": len(selected), "estimable_pairs": len(estimable),
            "non_estimable_pairs": len(selected) - len(estimable),
            "below_target_pairs": below,
            "fraction_below_target_plugin_model": below / len(estimable) if estimable else None,
        }

    standardized = []
    for d in EFFECTS:
        n = plan_seeds(d, 1.0, 1.0, target, alpha)
        standardized.append({
            "claim_type": "plugin_model", "standardized_mean_effect": d, "variance_a": 1.0,
            "variance_b": 1.0, "alpha": alpha, "target": target, "seeds_per_arm": n,
            "total_training_runs": 2 * n,
            "power_at_n": power_at_n(d, 1.0, 1.0, n, alpha),
            "power_at_n_minus_one": power_at_n(d, 1.0, 1.0, n - 1, alpha),
        })
    summary = {
        "dataset": "control-clock-final", "source_manifest": "data/public_sources.json",
        "observed_count_claim_type": "observed_training_run", "model_claim_type": "plugin_model",
        "observed_records": len(rows), "task_count": len(tasks), "tasks": tasks,
        "algorithm_count": len({k[1] for k in groups}),
        "algorithms": sorted({k[1] for k in groups}),
        "cohorts": [{"task": k[0], "algorithm": k[1], "variant": k[2],
                     "observed_training_runs": len(groups[k])} for k in sorted(groups)],
        "alpha": alpha, "target_plugin_power": target,
        "score_units": "undiscounted_episode_return_at_stopping",
        "sample_variance_ddof": 1,
        "power_model": "two-sided noncentral-t with fixed Welch-Satterthwaite df; "
                       "observed gap and sample variances; actual unequal allocation",
        "planning_model": "same approximation; smallest n >= 2; equal allocation",
        "effect_definition": "d = mean gap / sqrt((variance_a + variance_b)/2); "
                             "standardized-effect budgets retain each pair's variance ratio",
        "all_pairs": describe(pairs),
        "per_task": {task: describe([p for p in pairs if p["task"] == task]) for task in tasks},
        "per_algorithm_pair": {},
        "plugin_power_quantiles": dict(zip(
            ["min", "q25", "median", "q75", "max"],
            map(float, np.quantile([p["plugin_power_observed_allocation"] for p in eligible],
                                  [0, 0.25, 0.5, 0.75, 1]))
        )) if eligible else None,
        "standardized_effect_plans_equal_variance": standardized,
        "caveats": CAVEATS,
    }
    for aa, ab in itertools.combinations(summary["algorithms"], 2):
        summary["per_algorithm_pair"][f"{aa}/{ab}"] = describe([
            p for p in pairs if (p["algorithm_a"], p["algorithm_b"]) == (aa, ab)
        ])
    return pairs, summary


def write_results(rows: list[dict], summary: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plans = summary["standardized_effect_plans_equal_variance"]
    with output.with_name(output.name + "_standardized.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plans[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(plans)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/public_scores.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/public")
    args = parser.parse_args()
    pairs, summary = analyze(load_scores(args.input))
    summary["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    write_results(pairs, summary, args.output)
    print(json.dumps(summary["all_pairs"]))


if __name__ == "__main__":
    main()
