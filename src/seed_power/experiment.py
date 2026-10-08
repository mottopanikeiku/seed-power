"""Explicit independent-seed allocation and pilot-derived confirmation plans."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from seed_power.stats import plan_seeds, power_at_n


def comparison_cells(design):
    for task_index, task in enumerate(design["tasks"]):
        for pair_index, pair in enumerate(design["comparisons"]):
            comparison_index = task_index * len(design["comparisons"]) + pair_index
            for repetition in range(design["repetitions_per_comparison"]):
                yield {"task": task, "comparison_index": comparison_index,
                       "repetition": repetition, **pair}


def allocate_seed(design, phase, comparison_index, repetition, arm, index):
    if not (0 <= index < 1000 and arm in (0, 1) and 0 <= repetition < 100):
        raise ValueError("seed coordinate outside the committed allocation")
    return (design["phase_bases"][phase] + comparison_index * 1000000
            + repetition * 10000 + arm * 1000 + index)


def run_specs(design, stage, plans=None):
    if stage == "development":
        # Rate measurement only: fixed seed range, no changes to comparisons.
        for task_index, task in enumerate(design["tasks"]):
            for variant_index, variant in enumerate(("ars4", "cem4")):
                for index in range(32):
                    yield {"task": task, "variant": variant, "phase": stage,
                           "seed": design["phase_bases"][stage] + task_index * 1000
                                   + variant_index * 100 + index}
        return
    if stage not in ("pilot", "confirmation"):
        raise ValueError("stage must be development, pilot or confirmation")
    if stage == "confirmation":
        if plans is None:
            raise ValueError("confirmation requires an explicit pilot plan")
        cells = [cell for cell in plans["cells"] if cell["status"] == "planned"]
    else:
        cells = comparison_cells(design)
    for cell in cells:
        count = (design["pilot_seeds_per_arm"] if stage == "pilot"
                 else cell["required_seeds_per_arm"])
        for arm, variant in enumerate((cell["a"], cell["b"])):
            for index in range(count):
                yield {"task": cell["task"], "variant": variant, "phase": stage,
                       "comparison_index": cell["comparison_index"],
                       "repetition": cell["repetition"], "arm": arm, "index": index,
                       "seed": allocate_seed(design, stage, cell["comparison_index"],
                                             cell["repetition"], arm, index)}


def make_jobs(design, stage, plans=None):
    grouped = defaultdict(list)
    for spec in run_specs(design, stage, plans):
        grouped[spec["task"], spec["variant"]].append(spec)
    for (task, variant), specs in grouped.items():
        size = design["batch_size"]
        for start in range(0, len(specs), size):
            yield {"task": task, "variant": variant,
                   "configuration": {**design["variants"][variant],
                                     "population": design["population"]},
                   "evaluation_episodes": design["evaluation_episodes_per_training_seed"],
                   "specs": specs[start:start + size]}


def build_plans(design, pilot_records):
    by_arm = defaultdict(list)
    expected = list(run_specs(design, "pilot"))
    expected_seeds = {spec["seed"] for spec in expected}
    actual_seeds = [record["seed"] for record in pilot_records]
    if len(actual_seeds) != len(set(actual_seeds)) or set(actual_seeds) != expected_seeds:
        raise ValueError("pilot records must contain every allocated seed exactly once")
    for record in pilot_records:
        by_arm[record["comparison_index"], record["repetition"], record["arm"]].append(
            record["score"]
        )
    cells = []
    for cell in comparison_cells(design):
        a, b = [np.asarray(by_arm[cell["comparison_index"], cell["repetition"], arm])
                for arm in (0, 1)]
        effect = float(a.mean() - b.mean())
        va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
        required = plan_seeds(effect, va, vb, design["target_power"], design["alpha"])
        if required is None:
            status = "zero_effect" if effect == 0 else "zero_variance"
        elif required > design["maximum_executable_seeds_per_arm"]:
            status = "over_budget"
        else:
            status = "planned"
        cells.append({**cell, "pilot_mean_a": float(a.mean()), "pilot_mean_b": float(b.mean()),
                      "pilot_variance_a": va, "pilot_variance_b": vb,
                      "pilot_effect_a_minus_b": effect, "required_seeds_per_arm": required,
                      "status": status,
                      "modeled_power": (power_at_n(effect, va, vb, required, design["alpha"])
                                        if required is not None else None)})
    return {"target_power": design["target_power"], "alpha": design["alpha"],
            "pilot_seeds_per_arm": design["pilot_seeds_per_arm"], "cells": cells}


def plans_match(rebuilt, committed, rel_tol=1e-9):
    """Return whether a rebuilt plan reproduces a committed one.

    Counts, statuses, coordinates and hashes must match exactly. Floats such
    as ``modeled_power`` may differ in their last bits because SciPy's
    noncentral-t tails are not bitwise identical across CPU architectures, so
    floats are compared with a tight relative tolerance instead of ``==``.
    """
    if type(rebuilt) is not type(committed):
        return False
    if isinstance(rebuilt, float):
        return math.isclose(rebuilt, committed, rel_tol=rel_tol, abs_tol=0.0)
    if isinstance(rebuilt, dict):
        return rebuilt.keys() == committed.keys() and all(
            plans_match(rebuilt[key], committed[key], rel_tol) for key in rebuilt)
    if isinstance(rebuilt, list):
        return len(rebuilt) == len(committed) and all(
            plans_match(a, b, rel_tol) for a, b in zip(rebuilt, committed))
    return rebuilt == committed
