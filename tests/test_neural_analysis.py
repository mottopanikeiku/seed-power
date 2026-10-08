"""Small synthetic analysis fixtures; no policies are trained by these tests."""

from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from scripts import neural_analyze as analysis
from seed_power.experiment import build_plans, plans_match, run_specs
from seed_power.stats import welch_test


def small_design():
    base = {
        "method": "ppo", "num_envs": 16, "num_steps": 64, "minibatches": 4,
        "epochs": 4, "learning_rate": .0003, "entropy_coefficient": .01,
        "hidden_size": 32, "total_steps": 32768, "gamma": .99, "gae_lambda": .95,
        "clip_epsilon": .2, "value_coefficient": .5, "max_gradient_norm": .5,
    }
    return {
        "tasks": ["CartPole-v1", "Acrobot-v1"],
        "variants": {"base": base, "lr_fast": {**base, "learning_rate": .001}},
        "task_overrides": {"CartPole-v1": {"total_steps": 32768},
                           "Acrobot-v1": {"total_steps": 65536}},
        "population": 32, "pilot_seeds_per_arm": 4,
        "repetitions_per_comparison": 1, "target_power": .8, "alpha": .05,
        "maximum_executable_seeds_per_arm": 512,
        "evaluation_episodes_per_training_seed": 16,
        "evaluation_reset_seeds": {"CartPole-v1": list(range(100, 116)),
                                   "Acrobot-v1": list(range(200, 216))},
        "phase_bases": {"development": 2_000_000_000, "pilot": 3_000_000_000,
                        "confirmation": 4_000_000_000},
        "comparisons": [{"a": "base", "b": "lr_fast", "null_control": False},
                        {"a": "base", "b": "base", "null_control": True}],
    }


def fixture_records(design, stage, plans=None, gap=10.0, constant=False):
    records = []
    for spec in run_specs(design, stage, plans):
        configuration = analysis.resolve_configuration(design, spec["task"], spec["variant"])
        score = float((gap if spec["arm"] == 0 else 0) + (0 if constant else spec["index"] % 4))
        records.append({**spec, "configuration": configuration, "score": score,
                        "episode_returns": [score] * 16,
                        "evaluation_reset_seeds": design["evaluation_reset_seeds"][spec["task"]],
                        "training_transitions": configuration["total_steps"]})
    return records


def test_task_resolution_keeps_every_ppo_field_and_full_training_budget():
    design = small_design()
    records = fixture_records(design, "pilot")
    analysis.validate_records(design, "pilot", records)
    for record in records:
        wanted_steps = 32768 if record["task"] == "CartPole-v1" else 65536
        assert record["configuration"]["total_steps"] == wanted_steps
        assert record["configuration"]["population"] == 32
        assert record["training_transitions"] == wanted_steps
        assert record["configuration"]["learning_rate"] == (
            .001 if record["variant"] == "lr_fast" else .0003)
        assert set(record["configuration"]) == set(design["variants"]["base"]) | {"population"}


@pytest.mark.parametrize("change, message", [
    (lambda row: row["configuration"].update(epochs=1), "optimizer"),
    (lambda row: row["evaluation_reset_seeds"].reverse(), "evaluation reset"),
    (lambda row: row.update(training_transitions=1024), "training transitions"),
    (lambda row: row.update(episode_returns=[1.0] * 15), "16 finite"),
    (lambda row: row.update(episode_returns=[float("nan")] * 16), "16 finite"),
    (lambda row: row.update(episode_returns=[True] * 16), "16 finite"),
    (lambda row: row.update(score=1000.0), "score"),
    (lambda row: row.update(arm=1), "metadata"),
])
def test_raw_contract_mutations_are_rejected(change, message):
    design = small_design()
    records = deepcopy(fixture_records(design, "pilot"))
    change(records[0])
    with pytest.raises(ValueError, match=message):
        analysis.validate_records(design, "pilot", records)


def test_missing_and_duplicate_seeds_are_rejected():
    design = small_design()
    records = fixture_records(design, "pilot")
    with pytest.raises(ValueError, match="missing"):
        analysis.validate_records(design, "pilot", records[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        analysis.validate_records(design, "pilot", records + records[:1])


@pytest.mark.parametrize("phase, base", [("pilot", 100000), ("confirmation", 3_000_000_000)])
def test_neural_and_legacy_phase_ranges_cannot_overlap(phase, base):
    design = small_design()
    design["phase_bases"][phase] = base
    with pytest.raises(ValueError, match="ranges overlap"):
        analysis.validate_design(design)


def test_exact_planned_counts_and_welch_outcomes_reuse_independent_root_seeds():
    design = small_design()
    pilot = fixture_records(design, "pilot")
    plans = build_plans(design, pilot)
    fresh = fixture_records(design, "confirmation", plans)
    assert len(fresh) == sum(2 * cell["required_seeds_per_arm"] for cell in plans["cells"])
    assert not {row["seed"] for row in pilot} & {row["seed"] for row in fresh}
    outcomes = analysis.confirmation_outcomes(design, plans, fresh)
    assert len(outcomes) == len(plans["cells"])
    for outcome in outcomes:
        arms = [[row["score"] for row in fresh
                 if (row["comparison_index"], row["repetition"], row["arm"]) ==
                 (outcome["comparison_index"], outcome["repetition"], arm)] for arm in (0, 1)]
        assert outcome["pvalue"] == welch_test(*arms)["pvalue"]
    result = analysis.summarize(design, plans, outcomes, fresh, "committed-plan")
    assert result["plan_commit"] == "committed-plan"
    assert result["nonnull"]["plans"] == 2
    assert result["null_controls"]["plans"] == 2
    assert result["training_seeds_confirmation"] == len(fresh)
    assert result["training_transitions_confirmation"] == sum(
        row["training_transitions"] for row in fresh)
    assert "not known alternatives" in result["interpretation"]
    assert "heterogeneous" in result["interval_interpretation"]
    assert result["evaluation_reset_seeds"] == design["evaluation_reset_seeds"]
    with pytest.raises(ValueError, match="missing"):
        analysis.confirmation_outcomes(design, plans, fresh[:-1])


def test_constant_score_conventions_are_disclosed_and_not_called_ordinary_welch():
    design = small_design()
    plans = build_plans(design, fixture_records(design, "pilot"))
    fresh = fixture_records(design, "confirmation", plans, constant=True)
    for row in fresh:
        if row["comparison_index"] == 0:
            row["score"] = 0.0
            row["episode_returns"] = [0.0] * 16
    outcomes = analysis.confirmation_outcomes(design, plans, fresh)
    result = analysis.summarize(design, plans, outcomes, fresh, "committed-plan")
    assert result["welch_conventions"]["nonnull"] == {
        "identical_constants": 1, "distinct_constants": 1,
    }
    assert result["welch_conventions"]["null_controls"] == {"distinct_constants": 2}
    assert result["nonnull"]["detections"] == 1
    assert result["nonnull_ordinary_welch"]["plans"] == 0
    assert result["nonnull_ordinary_welch"]["detection_rate"] is None
    assert all(sum(group["welch_conventions"].values()) == group["plans"]
               for group in result["per_comparison"])


@pytest.mark.parametrize("gap, constant, status", [
    (.01, False, "over_budget"), (0.0, False, "zero_effect"),
    (10.0, True, "zero_variance"),
])
def test_all_ineligible_cells_and_uncapped_requirements_are_retained(gap, constant, status):
    design = small_design()
    pilot = fixture_records(design, "pilot", gap=gap, constant=constant)
    plans = build_plans(design, pilot)
    assert len(plans["cells"]) == 4
    assert {cell["status"] for cell in plans["cells"]} == {status}
    assert list(run_specs(design, "confirmation", plans)) == []
    if status == "over_budget":
        assert all(cell["required_seeds_per_arm"] > 512 for cell in plans["cells"])
    outcomes = analysis.confirmation_outcomes(design, plans, [])
    result = analysis.summarize(design, plans, outcomes, [], "committed-plan")
    assert result["attempted_nonnull_plans"] == 2
    assert result["nonnull_status_counts"] == {status: 2}
    assert result["nonnull"]["detection_rate"] is None
    assert result["eligible_nonnull_fraction"] == 0
    assert len(result["per_comparison"]) == 4
    assert all(group["attempted_plans"] == 1 and group["plans"] == 0
               for group in result["per_comparison"])


def test_no_pilot_significance_selection_is_applied():
    design = small_design()
    pilot = fixture_records(design, "pilot", gap=2.0)
    for comparison_index in range(4):
        arms = [[row["score"] for row in pilot
                 if row["comparison_index"] == comparison_index and row["arm"] == arm]
                for arm in (0, 1)]
        assert welch_test(*arms)["pvalue"] > design["alpha"]
    plans = build_plans(design, pilot)
    assert len(plans["cells"]) == 4
    assert all(cell["status"] == "planned" for cell in plans["cells"])


def test_cli_gzip_and_empty_confirmation_produce_header_and_plan_provenance(tmp_path, monkeypatch):
    design = small_design()
    pilot = fixture_records(design, "pilot", gap=0.0)
    design_path, pilot_path = tmp_path / "design.json", tmp_path / "pilot.jsonl"
    plan_path, fresh_path = tmp_path / "plan.json", tmp_path / "fresh.jsonl"
    design_path.write_text(json.dumps(design))
    with gzip.open(str(pilot_path) + ".gz", "wt") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in pilot))
    fresh_path.write_text("")
    assert analysis.load_records(pilot_path) == pilot
    monkeypatch.setenv("PLAN_COMMIT", "a" * 40)
    shared = ["--design", str(design_path), "--pilot", str(pilot_path),
              "--plan", str(plan_path), "--confirmation", str(fresh_path),
              "--output", str(tmp_path)]
    monkeypatch.setattr("sys.argv", ["neural_analyze.py", "plan", *shared])
    analysis.main()
    monkeypatch.setattr("sys.argv", ["neural_analyze.py", "confirmation", *shared])
    analysis.main()
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["plan_commit"] == "a" * 40
    assert result["nonnull"]["detection_rate"] is None
    assert len((tmp_path / "outcomes.csv").read_text().splitlines()) == 1
    assert "pvalue" in (tmp_path / "outcomes.csv").read_text()
    plan = json.loads(plan_path.read_text())
    plan["cells"][0]["required_seeds_per_arm"] = 512
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="does not reproduce"):
        analysis.main()


def test_confirmation_environment_supplies_revision_and_checks_exact_hashes(tmp_path):
    design_path, plan_path = tmp_path / "design.json", tmp_path / "plan.json"
    fresh_path = tmp_path / "runs.jsonl.gz"
    design_path.write_text("{}\n")
    plan_path.write_text("{}\n")
    environment = {
        "plan_commit": "b" * 40,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "design_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "environment.json").write_text(json.dumps(environment))
    assert analysis.plan_provenance(design_path, plan_path, fresh_path) == "b" * 40
    with pytest.raises(ValueError, match="explicit plan revision"):
        analysis.plan_provenance(design_path, plan_path, fresh_path, "a" * 40)
    plan_path.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="plan_sha256"):
        analysis.plan_provenance(design_path, plan_path, fresh_path)


def test_figures_take_rates_from_summaries_not_a_hardcoded_legacy_result():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.neural_figures import make_comparisons, make_overview

    design = small_design()
    pilot = fixture_records(design, "pilot")
    plans = build_plans(design, pilot)
    fresh = fixture_records(design, "confirmation", plans)
    neural = analysis.summarize(design, plans, analysis.confirmation_outcomes(design, plans, fresh),
                                fresh, "committed-plan")
    legacy = deepcopy(neural)
    legacy["nonnull"].update(detection_rate=.25, wilson_95_low=.1, wilson_95_high=.6,
                             plans=4, detections=1)
    fig = make_overview(legacy, neural)
    assert [bar.get_height() for bar in fig.axes[0].patches] == [.25, 1.0]
    plt.close(fig)
    empty_plans = build_plans(design, fixture_records(design, "pilot", gap=0.0))
    empty = analysis.summarize(design, empty_plans, [], [], "committed-plan")
    fig = make_comparisons(empty)
    assert len(fig.axes) == len(design["tasks"])
    assert all(any("No executable" in text.get_text() for text in ax.texts) for ax in fig.axes)
    plt.close(fig)


def test_committed_neural_roots_reproduce_counts_outcomes_and_summary():
    import csv
    import io

    root = analysis.ROOT
    design_path = root / "configs/neural_design.json"
    plan_path = root / "results/neural/pilot_plan.json"
    pilot_path = root / "results/neural/pilot/runs.jsonl.gz"
    fresh_path = root / "results/neural/confirmation/runs.jsonl.gz"
    design = json.loads(design_path.read_text())
    plans = json.loads(plan_path.read_text())
    pilot = analysis.load_records(pilot_path)
    fresh = analysis.load_records(fresh_path)
    analysis.validate_records(design, "pilot", pilot)
    rebuilt = build_plans(design, pilot)
    rebuilt["pilot_sha256"] = hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    rebuilt["design_sha256"] = hashlib.sha256(design_path.read_bytes()).hexdigest()
    assert plans_match(rebuilt, plans)
    plan_commit = analysis.plan_provenance(design_path, plan_path, fresh_path)
    outcomes = analysis.confirmation_outcomes(design, plans, fresh)
    result = analysis.summarize(design, plans, outcomes, fresh, plan_commit)
    assert result == json.loads((root / "results/neural/summary.json").read_text())
    assert len(pilot) == 8000 and len(fresh) == 65744 and len(outcomes) == 381
    assert not {record["seed"] for record in pilot} & {record["seed"] for record in fresh}
    legacy_seeds = set()
    for phase in ("pilot", "confirmation"):
        with (root / f"results/{phase}/runs.jsonl").open() as stream:
            legacy_seeds.update(json.loads(line)["seed"] for line in stream if line.strip())
    assert not legacy_seeds & {record["seed"] for record in pilot + fresh}
    for record in pilot + fresh:
        bounds = (0, 500) if record["task"] == "CartPole-v1" else (-500, 0)
        assert bounds[0] <= record["score"] <= bounds[1]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=analysis.OUTCOME_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(outcomes)
    assert stream.getvalue() == (root / "results/neural/outcomes.csv").read_text()
    environment = json.loads((fresh_path.parent / "environment.json").read_text())
    assert sum(batch["useful_seeds"] for batch in environment["batches"]) == len(fresh)
    assert {batch["index"] for batch in environment["batches"]} == set(range(1031))
    assert environment["training_identity"]["source_tree"] == (
        "23cd8f7b5b1bd5df4405f3b2e5224bb2d2ce51e3"
    )
    assert {batch["producer_code_commit"] for batch in environment["batches"]} == {
        "277c08f1203c0b1468037edccfe363136b717e6e",
        "37e9e4646acacea702034f8ff90d0aafad40a67c",
    }
