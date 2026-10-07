import json
from pathlib import Path

import numpy as np
import pytest

from seed_power.experiment import build_plans, comparison_cells, make_jobs, run_specs
from scripts.analyze import (
    confirmation_outcomes, load_records, main as analysis_main, summarize, validate_records,
)

ROOT = Path(__file__).resolve().parents[1]


def design():
    return json.loads((ROOT / "configs/design.json").read_text())


def small_design():
    result = design()
    result.update(tasks=["CartPole-v1"], comparisons=[result["comparisons"][0]],
                  repetitions_per_comparison=1, pilot_seeds_per_arm=4)
    return result


def fixture_records(configuration, stage, plans=None):
    records = []
    for spec in run_specs(configuration, stage, plans):
        score = float((10 if spec["arm"] == 0 else 0) + spec["index"] % 4)
        records.append({**spec, "score": score, "episode_returns": [score] * 16,
                        "configuration": {**configuration["variants"][spec["variant"]],
                                          "population": configuration["population"]}})
    return records


def test_all_pilot_and_confirmation_training_seeds_are_unique_and_disjoint():
    config = design()
    pilot = list(run_specs(config, "pilot"))
    maximum_plans = {"cells": [{**cell, "status": "planned", "required_seeds_per_arm": 256}
                                for cell in comparison_cells(config)]}
    confirmation = list(run_specs(config, "confirmation", maximum_plans))
    development = list(run_specs(config, "development"))
    all_seeds = [spec["seed"] for spec in pilot + confirmation + development]
    assert len(all_seeds) == len(set(all_seeds))
    assert len(pilot) == 2 * 7 * 12 * 2 * 8
    assert sum(len(job["specs"]) for job in make_jobs(config, "pilot")) == len(pilot)


def test_plan_and_confirmation_use_the_exact_planned_fresh_counts():
    config = small_design()
    pilot = fixture_records(config, "pilot")
    validate_records(config, "pilot", pilot)
    plans = build_plans(config, pilot)
    assert plans["cells"][0]["status"] == "planned"
    fresh = fixture_records(config, "confirmation", plans)
    assert len(fresh) == 2 * plans["cells"][0]["required_seeds_per_arm"]
    outcomes = confirmation_outcomes(config, plans, fresh)
    result = summarize(config, plans, outcomes, fresh)
    assert result["nonnull"]["plans"] == 1
    assert result["nonnull"]["detections"] == 1
    assert result["training_seeds_confirmation"] == len(fresh)
    with pytest.raises(ValueError, match="missing"):
        validate_records(config, "confirmation", fresh[:-1], plans)
    with pytest.raises(ValueError, match="duplicate"):
        validate_records(config, "confirmation", fresh + fresh[:1], plans)


def test_over_budget_is_not_truncated_or_called_a_planned_comparison():
    config = small_design()
    pilot = fixture_records(config, "pilot")
    for record in pilot:
        record["score"] = float(record["index"] + 0.0001 * record["arm"])
    plans = build_plans(config, pilot)
    cell = plans["cells"][0]
    assert cell["status"] == "over_budget"
    assert cell["required_seeds_per_arm"] > 256
    assert list(run_specs(config, "confirmation", plans)) == []


def test_analysis_cli_reports_an_empty_confirmation_when_every_plan_is_ineligible(
    tmp_path, monkeypatch,
):
    config = small_design()
    pilot = fixture_records(config, "pilot")
    for record in pilot:
        record.update(score=3.0, episode_returns=[3.0] * 16)
    design_path, pilot_path = tmp_path / "design.json", tmp_path / "pilot.jsonl"
    plan_path, fresh_path = tmp_path / "plan.json", tmp_path / "fresh.jsonl"
    design_path.write_text(json.dumps(config))
    pilot_path.write_text("".join(json.dumps(record) + "\n" for record in pilot))
    fresh_path.write_text("")
    shared = ["--design", str(design_path), "--pilot", str(pilot_path),
              "--plan", str(plan_path), "--confirmation", str(fresh_path),
              "--output", str(tmp_path)]
    monkeypatch.setattr("sys.argv", ["analyze.py", "plan", *shared])
    analysis_main()
    monkeypatch.setattr("sys.argv", ["analyze.py", "confirmation", *shared])
    analysis_main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["nonnull"]["plans"] == 0
    assert summary["nonnull"]["detection_rate"] is None
    assert summary["eligible_nonnull_fraction"] == 0
    header = (tmp_path / "outcomes.csv").read_text().splitlines()
    assert len(header) == 1
    assert "pvalue" in header[0].split(",")


def test_committed_confirmation_reproduces_the_reported_results():
    # This project ships actual runs; this is a consistency test, not a training run.
    config = design()
    plans = json.loads((ROOT / "results/pilot_plan.json").read_text())
    pilot = load_records(ROOT / "results/pilot/runs.jsonl")
    fresh = load_records(ROOT / "results/confirmation/runs.jsonl")
    validate_records(config, "pilot", pilot)
    rebuilt = build_plans(config, pilot)
    assert rebuilt["cells"] == plans["cells"]
    outcomes = confirmation_outcomes(config, plans, fresh)
    result = summarize(config, plans, outcomes, fresh)
    assert result == json.loads((ROOT / "results/summary.json").read_text())
    assert not {run["seed"] for run in pilot} & {run["seed"] for run in fresh}
    for run in pilot + fresh:
        bounds = (0, 500) if run["task"] == "CartPole-v1" else (-500, 0)
        assert bounds[0] <= run["score"] <= bounds[1]
        assert np.isfinite(run["score"])
