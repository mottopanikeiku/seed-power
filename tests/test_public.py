"""Network-free invariants over the actual committed public scores and source record."""

import hashlib
import importlib.util
import io
import itertools
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nct, t

from seed_power.stats import plan_seeds, power_at_n

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPORTER = script("import_public")
ANALYSIS = script("public_analysis")
MANIFEST = json.loads((ROOT / "data/public_sources.json").read_text())


def scores():
    return ANALYSIS.load_scores(ROOT / "data/public_scores.csv")


def test_source_manifest_and_complete_seed_cohorts():
    rows = scores()
    assert len(rows) == len(MANIFEST["records"]) == 140
    expected = {(s["identity"][0], s["identity"][1], s["identity"][2], s["identity"][3])
                for s in MANIFEST["records"]}
    observed = {(r["task"], r["algorithm"], r["variant"], r["seed"]) for r in rows}
    assert observed == expected
    counts = Counter((r["task"], r["algorithm"]) for r in rows)
    assert len(counts) == 10
    for (task, algorithm), count in counts.items():
        assert task in {"Acrobot-v1", "CartPole-v1"}
        assert count == (5 if algorithm in {"cleanrl", "sb3-zoo"} else 20)
        assert {r["seed"] for r in rows if (r["task"], r["algorithm"]) == (task, algorithm)} == (
            set(range(count))
        )
    for row in rows:
        source = next(s for s in MANIFEST["records"] if s["path"] == row["source_path"])
        assert row["source_sha256"] == source["sha256"]
        assert MANIFEST["revision"] in row["source_url"]
        assert len(source["sha256"]) == 64
        assert int(row["evaluation_episodes"]) == 100
    assert MANIFEST["license"]["identifier"] == "MIT"
    assert MANIFEST["rliable_access"]["status"] == "not_imported"


def test_real_source_record_import_and_raw_episode_mean():
    payload = (ROOT / "data/public_record_fixture.json").read_bytes()
    record = json.loads(payload)
    identity = [record["task"], record["method"], record["variant"], record["seed"]]
    source = next(s for s in MANIFEST["records"] if s["identity"] == identity)
    row = IMPORTER.parse_record(payload, source)
    assert row["score"] == pytest.approx(-95.57)
    assert row["evaluation_steps"] == 0  # Qualification at initialization is not removed.
    assert row["score"] == pytest.approx(np.mean(record["evaluations"][-1]["returns"]))
    committed = next(r for r in scores() if [r["task"], r["algorithm"], r["variant"], r["seed"]]
                     == identity)
    assert row["score"] == committed["score"]
    with pytest.raises(ValueError, match="SHA256"):
        IMPORTER.parse_record(payload + b"\n", source)


def test_archive_import_reads_real_record_without_filesystem_extraction():
    payload = (ROOT / "data/public_record_fixture.json").read_bytes()
    source = next(s for s in MANIFEST["records"] if s["identity"] == (
        ["Acrobot-v1", "ppo-cpu", "default", 19]
    ))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(MANIFEST["archive"]["member_prefix"] + source["path"])
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    archive_bytes = buffer.getvalue()
    subset = {"archive": dict(MANIFEST["archive"]), "records": [source]}
    subset["archive"]["sha256"] = hashlib.sha256(archive_bytes).hexdigest()
    assert IMPORTER.import_archive(archive_bytes, subset)[0]["score"] == pytest.approx(-95.57)
    with pytest.raises(ValueError, match="SHA256"):
        IMPORTER.import_archive(archive_bytes + b"changed", subset)


def test_all_pairs_and_saved_derived_result_cover_actual_observations():
    rows = scores()
    pairs, summary = ANALYSIS.analyze(rows)
    expected_pairs = set(itertools.combinations(summary["algorithms"], 2))
    assert len(pairs) == 20
    for task in summary["tasks"]:
        task_pairs = [p for p in pairs if p["task"] == task]
        assert {(p["algorithm_a"], p["algorithm_b"]) for p in task_pairs} == expected_pairs
        for p in task_pairs:
            a = [r["score"] for r in rows if r["task"] == task
                 and r["algorithm"] == p["algorithm_a"]]
            b = [r["score"] for r in rows if r["task"] == task
                 and r["algorithm"] == p["algorithm_b"]]
            assert p["observed_seeds_a"] == len(a)
            assert p["observed_seeds_b"] == len(b)
            assert p["mean_difference"] == pytest.approx(np.mean(a) - np.mean(b))
            assert p["variance_a"] == pytest.approx(np.var(a, ddof=1))
            assert p["variance_b"] == pytest.approx(np.var(b, ddof=1))
    saved = json.loads((ROOT / "results/public.json").read_text())
    assert summary["all_pairs"] == saved["all_pairs"]
    assert summary["per_task"] == saved["per_task"]
    assert summary["per_algorithm_pair"] == saved["per_algorithm_pair"]
    assert saved["all_pairs"]["below_target_pairs"] == 19
    assert saved["input_sha256"] == hashlib.sha256((ROOT / "data/public_scores.csv").read_bytes()).hexdigest()
    assert summary["model_claim_type"] == "plugin_model"


def test_observed_unequal_allocation_uses_both_arm_counts():
    rows = scores()
    pairs, _ = ANALYSIS.analyze(rows)
    pair = next(p for p in pairs if p["observed_seeds_a"] != p["observed_seeds_b"])
    na, nb = pair["observed_seeds_a"], pair["observed_seeds_b"]
    ca, cb = pair["variance_a"] / na, pair["variance_b"] / nb
    df = (ca + cb) ** 2 / (ca ** 2 / (na - 1) + cb ** 2 / (nb - 1))
    nc = abs(pair["mean_difference"]) / math.sqrt(ca + cb)
    critical = t.ppf(0.975, df)
    reference = nct.sf(critical, df, nc) + nct.cdf(-critical, df, nc)
    assert pair["plugin_power_observed_allocation"] == pytest.approx(reference)
    equal = next(p for p in pairs if p["observed_seeds_a"] == p["observed_seeds_b"])
    assert equal["plugin_power_observed_allocation"] == pytest.approx(power_at_n(
        equal["mean_difference"], equal["variance_a"], equal["variance_b"],
        equal["observed_seeds_a"]
    ))


def test_real_pair_plans_and_standardized_budgets_are_minimal():
    pairs, summary = ANALYSIS.analyze(scores())
    assert [p["seeds_per_arm"] for p in summary["standardized_effect_plans_equal_variance"]] == (
        [394, 64, 26, 17]
    )
    for pair in pairs:
        pooled_sd = math.sqrt((pair["variance_a"] + pair["variance_b"]) / 2)
        for d in ANALYSIS.EFFECTS:
            n = pair[f"planned_equal_n_d_{d}"]
            assert n == plan_seeds(d * pooled_sd, pair["variance_a"], pair["variance_b"])
            assert power_at_n(d * pooled_sd, pair["variance_a"], pair["variance_b"], n) >= 0.8
            if n > 2:
                assert power_at_n(d * pooled_sd, pair["variance_a"], pair["variance_b"], n - 1) < 0.8


def test_duplicate_and_nonfinite_real_score_mutations_are_rejected(tmp_path):
    payload = (ROOT / "data/public_scores.csv").read_text().splitlines()
    bad = tmp_path / "scores.csv"
    bad.write_text("\n".join(payload + [payload[1]]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        ANALYSIS.load_scores(bad)
    rows = scores()
    rows[0]["score"] = float("nan")
    IMPORTER.write_csv(rows, bad)
    with pytest.raises(ValueError, match="Nonfinite"):
        ANALYSIS.load_scores(bad)


def test_all_pairs_ignore_row_order_and_preserve_non_estimable_groups():
    rows = scores()
    forward, forward_summary = ANALYSIS.analyze(rows)
    reverse, reverse_summary = ANALYSIS.analyze(list(reversed(rows)))
    assert [(p["task"], p["algorithm_a"], p["algorithm_b"]) for p in forward] == (
        [(p["task"], p["algorithm_a"], p["algorithm_b"]) for p in reverse]
    )
    assert forward_summary["all_pairs"] == reverse_summary["all_pairs"]
    for a, b in zip(forward, reverse):
        assert a["plugin_power_observed_allocation"] == pytest.approx(
            b["plugin_power_observed_allocation"]
        )
    for row in rows:
        row["score"] = 0.0
    pairs, summary = ANALYSIS.analyze(rows)
    assert len(pairs) == 20
    assert summary["all_pairs"]["estimable_pairs"] == 0
    assert summary["all_pairs"]["non_estimable_pairs"] == 20
    assert summary["all_pairs"]["fraction_below_target_plugin_model"] is None
    assert all(p["status"] == "zero_total_sample_variance" for p in pairs)
    assert all(p["planned_equal_n_observed_gap"] is None for p in pairs)
    assert all(p["plugin_power_observed_allocation"] is None for p in pairs)
