"""Create the pilot plan or recompute confirmation outcomes from raw scores."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from seed_power.experiment import build_plans, plans_match, run_specs
from seed_power.stats import welch_test, wilson_interval

OUTCOME_FIELDS = [
    "task", "comparison_index", "repetition", "a", "b", "null_control",
    "pilot_mean_a", "pilot_mean_b", "pilot_variance_a", "pilot_variance_b",
    "pilot_effect_a_minus_b", "required_seeds_per_arm", "status", "modeled_power",
    "confirmation_mean_a", "confirmation_mean_b", "confirmation_variance_a",
    "confirmation_variance_b", "confirmation_effect_a_minus_b", "pvalue", "degeneracy",
    "detected", "opposite_direction",
]


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_records(design, stage, records, plans=None):
    expected = {spec["seed"]: spec for spec in run_specs(design, stage, plans)}
    seen = set()
    for record in records:
        seed = record["seed"]
        if seed in seen or seed not in expected:
            raise ValueError("duplicate or unallocated training seed")
        seen.add(seed)
        for key, value in expected[seed].items():
            if record[key] != value:
                raise ValueError(f"run metadata mismatch for seed {seed}: {key}")
        wanted_config = {**design["variants"][record["variant"]],
                         "population": design["population"]}
        if record["configuration"] != wanted_config:
            raise ValueError("optimizer differs from the committed design")
        returns = record["episode_returns"]
        if len(returns) != design["evaluation_episodes_per_training_seed"]:
            raise ValueError("wrong evaluation episode count")
        if record["score"] != float(np.mean(returns)) or not np.isfinite(record["score"]):
            raise ValueError("score is not the mean of finite evaluation returns")
    if seen != set(expected):
        raise ValueError("missing allocated training seeds")


def summary_group(rows):
    total = len(rows)
    detected = sum(row["detected"] for row in rows)
    low, high = wilson_interval(detected, total) if total else (None, None)
    counts = [row["required_seeds_per_arm"] for row in rows]
    return {"plans": total, "detections": detected,
            "detection_rate": detected / total if total else None,
            "wilson_95_low": low, "wilson_95_high": high,
            "opposite_pilot_direction_detections": sum(row["opposite_direction"] for row in rows),
            "seeds_per_arm_median": float(np.median(counts)) if counts else None,
            "seeds_per_arm_q25": float(np.quantile(counts, .25)) if counts else None,
            "seeds_per_arm_q75": float(np.quantile(counts, .75)) if counts else None}


def confirmation_outcomes(design, plans, records):
    validate_records(design, "confirmation", records, plans)
    by_arm = defaultdict(list)
    for record in records:
        by_arm[record["comparison_index"], record["repetition"], record["arm"]].append(record["score"])
    outcomes = []
    for cell in plans["cells"]:
        if cell["status"] != "planned":
            continue
        a, b = [by_arm[cell["comparison_index"], cell["repetition"], arm] for arm in (0, 1)]
        test = welch_test(a, b)
        detected = test["pvalue"] < design["alpha"]
        outcomes.append({**cell, "confirmation_mean_a": float(np.mean(a)),
                         "confirmation_mean_b": float(np.mean(b)),
                         "confirmation_variance_a": float(np.var(a, ddof=1)),
                         "confirmation_variance_b": float(np.var(b, ddof=1)),
                         "confirmation_effect_a_minus_b": test["mean_difference"],
                         "pvalue": test["pvalue"], "degeneracy": test["degeneracy"],
                         "detected": detected,
                         "opposite_direction": bool(detected and test["mean_difference"]
                                                   * cell["pilot_effect_a_minus_b"] < 0)})
    return outcomes


def summarize(design, plans, outcomes, records):
    nonnull = [row for row in outcomes if not row["null_control"]]
    nulls = [row for row in outcomes if row["null_control"]]
    groups = defaultdict(list)
    for row in outcomes:
        groups[row["task"], row["a"], row["b"], row["null_control"]].append(row)
    original_nonnull = [cell for cell in plans["cells"] if not cell["null_control"]]
    return {"target_power": design["target_power"],
            "interpretation": "Marginal detection frequency across executable fixed comparisons; not true power at a known effect or a general RL population.",
            "attempted_nonnull_plans": len(original_nonnull),
            "nonnull_status_counts": dict(Counter(cell["status"] for cell in original_nonnull)),
            "eligible_nonnull_fraction": len(nonnull) / len(original_nonnull),
            "nonnull": summary_group(nonnull), "null_controls": summary_group(nulls),
            "training_seeds_confirmation": len(records),
            "training_seeds_pilot": len(plans["cells"]) * 2 * design["pilot_seeds_per_arm"],
            "per_task": {task: summary_group([row for row in nonnull if row["task"] == task])
                         for task in design["tasks"]},
            "per_comparison": [{"task": task, "a": a, "b": b, "null_control": null,
                                **summary_group(rows)}
                               for (task, a, b, null), rows in sorted(groups.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "confirmation"])
    parser.add_argument("--design", default="configs/design.json")
    parser.add_argument("--pilot", default="results/pilot/runs.jsonl")
    parser.add_argument("--plan", default="results/pilot_plan.json")
    parser.add_argument("--confirmation", default="results/confirmation/runs.jsonl")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    design = json.loads(Path(args.design).read_text())
    pilot = load_records(args.pilot)
    validate_records(design, "pilot", pilot)
    plans = build_plans(design, pilot)
    plans["pilot_sha256"] = hashlib.sha256(Path(args.pilot).read_bytes()).hexdigest()
    plans["design_sha256"] = hashlib.sha256(Path(args.design).read_bytes()).hexdigest()
    if args.stage == "plan":
        Path(args.plan).write_text(json.dumps(plans, indent=2, allow_nan=False) + "\n")
        print(json.dumps(Counter(cell["status"] for cell in plans["cells"])))
        return
    committed = json.loads(Path(args.plan).read_text())
    if not plans_match(plans, committed):
        raise ValueError("committed plan does not reproduce from pilot scores")
    plans = committed  # Report the committed floats, not this platform's last bits.
    records = load_records(args.confirmation)
    if {row["seed"] for row in records} & {row["seed"] for row in pilot}:
        raise ValueError("pilot and confirmation training seeds overlap")
    outcomes = confirmation_outcomes(design, plans, records)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "outcomes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTCOME_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(outcomes)
    result = summarize(design, plans, outcomes, records)
    (destination / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["nonnull"]))


if __name__ == "__main__":
    main()
