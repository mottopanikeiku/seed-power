"""Plan and analyze PPO root-seed runs conditional on fixed evaluation resets.

Configuration resolution is variant, design population, then task overrides.
The adjacent confirmation environment.json records the committed plan revision;
PLAN_COMMIT or --plan-commit can supply it when analyzing standalone fixtures.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from seed_power.experiment import build_plans, plans_match, run_specs

if __package__:
    from .analyze import (
        OUTCOME_FIELDS, confirmation_outcomes as legacy_confirmation_outcomes,
        summarize as legacy_summarize, summary_group,
    )
else:
    from analyze import (
        OUTCOME_FIELDS, confirmation_outcomes as legacy_confirmation_outcomes,
        summarize as legacy_summarize, summary_group,
    )

ROOT = Path(__file__).resolve().parents[1]


def resolve_configuration(design, task, variant):
    return {**design["variants"][variant], "population": design["population"],
            **design["task_overrides"][task]}


def raw_path(path):
    path = Path(path)
    if not path.exists() and path.suffix != ".gz":
        compressed = Path(str(path) + ".gz")
        if compressed.exists():
            return compressed
    return path


def plan_provenance(design_path, plan_path, confirmation_path, explicit=None):
    environment_path = Path(confirmation_path).parent / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        committed = environment.get("plan_commit")
        if explicit and explicit != committed:
            raise ValueError("explicit plan revision differs from confirmation environment")
        for key, path in [("plan_sha256", plan_path), ("design_sha256", design_path)]:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if environment.get(key) != digest:
                raise ValueError(f"confirmation environment {key} does not match committed file")
    else:
        committed = explicit
    if not isinstance(committed, str) or not committed.strip():
        raise ValueError("supply plan_commit in confirmation environment, PLAN_COMMIT or --plan-commit")
    return committed


def load_records(path):
    path = raw_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_design(design, legacy_design=None):
    """Check fixed evaluation and separation of reserved training-seed ranges."""
    if design["evaluation_episodes_per_training_seed"] != 16:
        raise ValueError("neural evaluation requires exactly 16 fixed reset seeds")
    for task in design["tasks"]:
        seeds = design["evaluation_reset_seeds"][task]
        if (len(seeds) != 16 or len(set(seeds)) != 16
                or any(type(seed) is not int or seed < 0 for seed in seeds)):
            raise ValueError("each task needs 16 distinct nonnegative evaluation reset seeds")
    if design["maximum_executable_seeds_per_arm"] != 512:
        raise ValueError("the neural executable limit must be 512 seeds per arm")
    if legacy_design is None:
        legacy_design = json.loads((ROOT / "configs/design.json").read_text())

    def ranges(configuration):
        width = len(configuration["tasks"]) * len(configuration["comparisons"]) * 1_000_000
        return [(phase, base, base + width)
                for phase, base in configuration["phase_bases"].items()]

    neural_ranges = ranges(design)
    if set(design["phase_bases"]) != {"development", "pilot", "confirmation"}:
        raise ValueError("neural design must reserve development, pilot and confirmation seeds")
    for index, (phase, start, end) in enumerate(neural_ranges):
        if type(start) is not int or start < 0:
            raise ValueError("phase bases must be nonnegative integers")
        for other, low, high in neural_ranges[index + 1:] + ranges(legacy_design):
            if max(start, low) < min(end, high):
                raise ValueError(f"training seed ranges overlap: {phase} and {other}")


def validate_records(design, stage, records, plans=None):
    validate_design(design)
    specs = list(run_specs(design, stage, plans))
    expected = {spec["seed"]: spec for spec in specs}
    if len(expected) != len(specs):
        raise ValueError("allocated training seeds overlap")
    seen = set()
    for record in records:
        seed = record.get("seed")
        if type(seed) is not int or seed in seen or seed not in expected:
            raise ValueError("duplicate or unallocated training seed")
        seen.add(seed)
        for key, value in expected[seed].items():
            if record.get(key) != value:
                raise ValueError(f"run metadata mismatch for seed {seed}: {key}")
        configuration = resolve_configuration(design, record["task"], record["variant"])
        if record.get("configuration") != configuration:
            raise ValueError("optimizer differs from the committed neural design")
        if configuration.get("method") != "ppo":
            raise ValueError("neural configurations must use PPO")
        if record.get("evaluation_reset_seeds") != design["evaluation_reset_seeds"][record["task"]]:
            raise ValueError("evaluation reset seeds differ from the committed task list")
        returns = np.asarray(record.get("episode_returns"))
        if (returns.shape != (16,) or returns.dtype.kind not in "iuf"
                or not np.isfinite(returns).all()):
            raise ValueError("evaluation returns must contain 16 finite real values")
        score = record.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not np.isfinite(score) or score != float(np.mean(returns))):
            raise ValueError("score is not the mean of finite evaluation returns")
        transitions = record.get("training_transitions")
        if type(transitions) is not int or transitions != configuration["total_steps"]:
            raise ValueError("training transitions differ from the full committed budget")
    if seen != set(expected):
        raise ValueError("missing allocated training seeds")


def confirmation_outcomes(design, plans, records):
    """Reuse the existing Welch calculation after task-specific validation."""
    validate_records(design, "confirmation", records, plans)
    outcomes = []
    for task in design["tasks"]:
        task_design = {**design, "variants": {
            variant: resolve_configuration(design, task, variant)
            for variant in design["variants"]}}
        task_plans = {**plans, "cells": [cell for cell in plans["cells"] if cell["task"] == task]}
        task_records = [record for record in records if record["task"] == task]
        outcomes.extend(legacy_confirmation_outcomes(task_design, task_plans, task_records))
    return outcomes


def convention_counts(rows):
    return dict(Counter(row["degeneracy"] or "ordinary_welch" for row in rows))


def summarize(design, plans, outcomes, records, plan_commit):
    result = legacy_summarize(design, plans, outcomes, records)
    result.update(
        alpha=design["alpha"],
        plan_commit=plan_commit,
        interpretation=("Marginal detection frequency across executable fixed PPO variant "
                        "comparisons, conditional on the committed task-specific evaluation "
                        "reset sets. Different variants are not known alternatives. Evaluation "
                        "episodes are not independent training replicates; this is not true "
                        "power at a known effect or a general RL population."),
        interval_interpretation=("Wilson 95% intervals are descriptive pooled summaries of "
                                 "heterogeneous fixed comparison detection rates, not confidence "
                                 "intervals for one shared power parameter."),
        evaluation_reset_seeds=design["evaluation_reset_seeds"],
        training_transitions_confirmation=sum(record["training_transitions"] for record in records),
    )
    nonnull = [row for row in outcomes if not row["null_control"]]
    nulls = [row for row in outcomes if row["null_control"]]
    result["welch_conventions"] = {
        "nonnull": convention_counts(nonnull),
        "null_controls": convention_counts(nulls),
        "definitions": {
            "identical_constants": "No estimable t distribution; declared p=1 convention.",
            "distinct_constants": "No estimable t distribution; declared p=0 limiting convention.",
        },
    }
    result["nonnull_ordinary_welch"] = summary_group([
        row for row in nonnull if row["degeneracy"] is None
    ])
    groups = defaultdict(list)
    observed = defaultdict(list)
    for cell in plans["cells"]:
        groups[cell["task"], cell["a"], cell["b"], cell["null_control"]].append(cell)
    for row in outcomes:
        observed[row["task"], row["a"], row["b"], row["null_control"]].append(row)
    result["per_comparison"] = [
        {"task": task, "a": a, "b": b, "null_control": null,
         **summary_group(observed[key]), "welch_conventions": convention_counts(observed[key]),
         "attempted_plans": len(cells),
         "status_counts": dict(Counter(cell["status"] for cell in cells))}
        for key, cells in sorted(groups.items())
        for task, a, b, null in [key]
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "confirmation"])
    parser.add_argument("--design", default="configs/neural_design.json")
    parser.add_argument("--pilot", default="results/neural/pilot/runs.jsonl")
    parser.add_argument("--plan", default="results/neural/pilot_plan.json")
    parser.add_argument("--confirmation", default="results/neural/confirmation/runs.jsonl")
    parser.add_argument("--output", default="results/neural")
    parser.add_argument("--plan-commit", default=os.environ.get("PLAN_COMMIT"))
    args = parser.parse_args()
    design = json.loads(Path(args.design).read_text())
    pilot = load_records(args.pilot)
    validate_records(design, "pilot", pilot)
    plans = build_plans(design, pilot)
    plans["pilot_sha256"] = hashlib.sha256(raw_path(args.pilot).read_bytes()).hexdigest()
    plans["design_sha256"] = hashlib.sha256(Path(args.design).read_bytes()).hexdigest()
    if args.stage == "plan":
        destination = Path(args.plan)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plans, indent=2, allow_nan=False) + "\n")
        print(json.dumps(Counter(cell["status"] for cell in plans["cells"])))
        return
    plan_commit = plan_provenance(args.design, args.plan, args.confirmation, args.plan_commit)
    committed = json.loads(Path(args.plan).read_text())
    if not plans_match(plans, committed):
        raise ValueError("committed plan does not reproduce from pilot scores and design")
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
    result = summarize(design, plans, outcomes, records, plan_commit)
    (destination / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["nonnull"]))


if __name__ == "__main__":
    main()
