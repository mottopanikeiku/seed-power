"""Independent distribution references, not just planner self-consistency."""

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2, nct, norm, t, ttest_ind

from seed_power.stats import plan_seeds, power_at_n, welch_test, wilson_interval


def reference_power(effect, variance_a, variance_b, n, alpha=0.05):
    """Direct textbook formula, separate from implementation's scaled/log path."""
    n = np.asarray(n)
    standard_error = np.sqrt(variance_a / n + variance_b / n)
    df = (variance_a / n + variance_b / n) ** 2 / (
        (variance_a / n) ** 2 / (n - 1) + (variance_b / n) ** 2 / (n - 1)
    )
    critical = t.ppf(1.0 - alpha / 2.0, df)
    noncentrality = abs(effect) / standard_error
    return nct.cdf(-critical, df, noncentrality) + nct.sf(critical, df, noncentrality)


@pytest.mark.parametrize(
    "effect,variance_a,variance_b,n,alpha",
    [
        (0.5, 1.0, 1.0, 5, 0.05),
        (0.5, 1.0, 1.0, 64, 0.05),
        (1382.0, 1341.0**2, 990.0**2, 10, 0.05),
        (-1.5, 0.25, 9.0, 23, 0.01),
        (0.3, 0.0, 1.5, 7, 0.1),
        (0.0, 2.0, 5.0, 19, 0.05),
    ],
)
def test_power_matches_direct_scipy_reference(effect, variance_a, variance_b, n, alpha):
    expected = reference_power(effect, variance_a, variance_b, n, alpha)
    assert power_at_n(effect, variance_a, variance_b, n, alpha) == pytest.approx(
        expected, rel=2e-12, abs=2e-14
    )


@pytest.mark.parametrize("effect,variance_a,variance_b,n", [(0.7, 1.0, 1.0, 9), (1.2, 0.5, 5.0, 13)])
def test_power_matches_normal_chi_square_integral(effect, variance_a, variance_b, n):
    # A noncentral t is (Z + noncentrality) / sqrt(V/df), with independent
    # standard normal Z and chi-square V. Integrate those primitives directly;
    # this reference does not call the implementation or scipy.stats.nct.
    df = (n - 1) * (variance_a + variance_b) ** 2 / (variance_a**2 + variance_b**2)
    nc = effect / math.sqrt((variance_a + variance_b) / n)
    critical = t.ppf(0.975, df)

    def integrand(v):
        boundary = critical * math.sqrt(v / df)
        rejection = norm.sf(boundary - nc) + norm.cdf(-boundary - nc)
        return rejection * chi2.pdf(v, df)

    expected, error = quad(integrand, 0.0, math.inf, epsabs=1e-10, epsrel=1e-10)
    assert error < 1e-8
    assert power_at_n(effect, variance_a, variance_b, n) == pytest.approx(expected, abs=2e-9)


@pytest.mark.parametrize(
    "effect,variance_a,variance_b,target,alpha",
    [
        (0.5, 1.0, 1.0, 0.8, 0.05),
        (0.8, 1.0, 1.0, 0.8, 0.05),
        (1.0, 0.25, 4.0, 0.9, 0.01),
        (0.5, 0.0, 2.0, 0.8, 0.05),
        (20.0, 1.0, 1.0, 0.8, 0.05),
        (0.1, 1.0, 1.0, 0.01, 0.05),
    ],
)
def test_planner_matches_brute_force_minimum(effect, variance_a, variance_b, target, alpha):
    counts = np.arange(2, 1001)
    powers = reference_power(effect, variance_a, variance_b, counts, alpha)
    acceptable = counts[powers >= target]
    assert acceptable.size > 0
    expected = int(acceptable[0])
    assert plan_seeds(effect, variance_a, variance_b, target, alpha) == expected
    assert reference_power(effect, variance_a, variance_b, expected, alpha) >= target
    if expected > 2:
        assert reference_power(effect, variance_a, variance_b, expected - 1, alpha) < target


def test_equal_variance_known_seed_count():
    assert plan_seeds(0.5, 1.0, 1.0) == 64
    assert plan_seeds(0.8, 1.0, 1.0) == 26


def test_planner_has_no_artificial_cap():
    n = plan_seeds(0.01, 1.0, 1.0)
    assert n is not None and n > 100_000
    assert reference_power(0.01, 1.0, 1.0, n) >= 0.8
    assert reference_power(0.01, 1.0, 1.0, n - 1) < 0.8


def test_sign_arm_and_units_symmetry():
    baseline = power_at_n(0.6, 1.0, 4.0, 20)
    assert power_at_n(-0.6, 1.0, 4.0, 20) == baseline
    assert power_at_n(0.6, 4.0, 1.0, 20) == baseline
    assert power_at_n(6.0, 100.0, 400.0, 20) == pytest.approx(baseline)
    n = plan_seeds(0.6, 1.0, 4.0)
    assert plan_seeds(-0.6, 4.0, 1.0) == n
    assert plan_seeds(6.0, 100.0, 400.0) == n


@pytest.mark.parametrize("scale", [1e-280, 1e280])
def test_variance_scaling_does_not_overflow_or_underflow(scale):
    effect = 0.5 * math.sqrt(scale)
    assert power_at_n(effect, scale, 2.0 * scale, 20) == pytest.approx(
        reference_power(0.5, 1.0, 2.0, 20), rel=2e-12
    )
    assert plan_seeds(effect, scale, 2.0 * scale) == plan_seeds(0.5, 1.0, 2.0)


def test_unplannable_inputs_and_deterministic_power_limits():
    assert plan_seeds(0.0, 1.0, 2.0) is None
    assert plan_seeds(0.0, 1.0, 2.0, target=0.01) is None
    assert plan_seeds(1.0, 0.0, 0.0) is None
    assert plan_seeds(0.0, 0.0, 0.0) is None
    assert power_at_n(0.0, 0.0, 0.0, 2) == 0.0
    assert power_at_n(1.0, 0.0, 0.0, 2) == 1.0
    assert power_at_n(-1.0, 0.0, 0.0, 2) == 1.0
    assert power_at_n(0.0, 1.0, 2.0, 2) == 0.05


@pytest.mark.parametrize(
    "a,b",
    [
        ([1.0, 2.0, 4.0, 8.0], [0.0, 3.0, 4.0, 5.0, 11.0]),
        ([-10.0, -1.0, 7.0], [2.0, 2.5, 2.75, 3.0, 3.25]),
        ([2.0, 2.0, 2.0], [0.0, 1.0, 3.0, 7.0]),
        ([0.0, 1.0, 3.0, 7.0], [2.0, 2.0, 2.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ],
)
def test_welch_matches_independent_scipy(a, b):
    expected = ttest_ind(a, b, equal_var=False)
    actual = welch_test(a, b)
    assert actual["statistic"] == pytest.approx(expected.statistic, abs=1e-14)
    assert actual["pvalue"] == pytest.approx(expected.pvalue, abs=1e-14)
    assert actual["df"] == pytest.approx(expected.df)
    assert actual["mean_difference"] == pytest.approx(np.mean(a) - np.mean(b))
    assert actual["degeneracy"] is None
    reversed_result = welch_test(b, a)
    assert reversed_result["pvalue"] == actual["pvalue"]
    assert reversed_result["statistic"] == -actual["statistic"]


@pytest.mark.parametrize("a,b", [([0.1] * 7, [0.1] * 9), ([2.0] * 3, [2.0] * 4)])
def test_identical_constants_have_no_rejection(a, b):
    assert welch_test(a, b) == {
        "pvalue": 1.0,
        "statistic": 0.0,
        "mean_difference": 0.0,
        "df": None,
        "degeneracy": "identical_constants",
    }


@pytest.mark.parametrize("left,right,sign", [(2.0, 1.0, 1.0), (1.0, 2.0, -1.0)])
def test_distinct_constants_use_labeled_deterministic_limit(left, right, sign):
    result = welch_test([left] * 3, [right] * 7)
    assert result["pvalue"] == 0.0
    assert result["statistic"] == math.copysign(math.inf, sign)
    assert result["mean_difference"] == left - right
    assert result["df"] is None
    assert result["degeneracy"] == "distinct_constants"


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, True, "1", 1j])
def test_planning_rejects_nonfinite_and_nonreal_values(invalid):
    for key in ("effect", "variance_a", "variance_b", "alpha"):
        arguments = {"effect": 1.0, "variance_a": 1.0, "variance_b": 2.0, "alpha": 0.05}
        arguments[key] = invalid
        with pytest.raises(ValueError):
            power_at_n(**arguments, n=3)
        with pytest.raises(ValueError):
            plan_seeds(**arguments)
    with pytest.raises(ValueError):
        plan_seeds(1.0, 1.0, 1.0, target=invalid)


@pytest.mark.parametrize("invalid", [0, 1, -1, 2.0, 2.5, True, math.nan, math.inf, "3"])
def test_power_requires_integer_n_at_least_two(invalid):
    with pytest.raises(ValueError):
        power_at_n(1.0, 1.0, 1.0, invalid)


def test_numpy_scalar_inputs_are_valid():
    assert power_at_n(np.float64(0.5), 1.0, 1.0, np.int64(64)) == pytest.approx(
        reference_power(0.5, 1.0, 1.0, 64)
    )
    assert wilson_interval(np.int64(5), np.int64(10)) == wilson_interval(5, 10)


@pytest.mark.parametrize("invalid", [-1.0, 0.0, 1.0, 2.0, math.nan, math.inf, True])
def test_invalid_probability_bounds(invalid):
    with pytest.raises(ValueError):
        power_at_n(1.0, 1.0, 1.0, 2, alpha=invalid)
    with pytest.raises(ValueError):
        plan_seeds(1.0, 1.0, 1.0, target=invalid)
    with pytest.raises(ValueError):
        wilson_interval(1, 2, confidence=invalid)


def test_unrepresentable_real_and_two_sided_alpha_are_rejected():
    with pytest.raises(ValueError):
        power_at_n(10**400, 1.0, 1.0, 2)
    with pytest.raises(ValueError):
        plan_seeds(1.0, 10**400, 1.0)
    smallest_positive_float = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError):
        power_at_n(1.0, 1.0, 1.0, 2, alpha=smallest_positive_float)
    with pytest.raises(ValueError):
        plan_seeds(1.0, 1.0, 1.0, alpha=smallest_positive_float)


@pytest.mark.parametrize("variance_a,variance_b", [(-1.0, 1.0), (1.0, -1.0)])
def test_negative_variances_are_invalid(variance_a, variance_b):
    with pytest.raises(ValueError):
        power_at_n(1.0, variance_a, variance_b, 2)
    with pytest.raises(ValueError):
        plan_seeds(1.0, variance_a, variance_b)


@pytest.mark.parametrize(
    "invalid", [[], [1.0], [[1.0, 2.0]], [1.0, math.nan], [1.0, math.inf], [True, False], ["1", "2"], [1j, 2j]]
)
def test_invalid_welch_samples(invalid):
    with pytest.raises(ValueError):
        welch_test(invalid, [1.0, 2.0])
    with pytest.raises(ValueError):
        welch_test([1.0, 2.0], invalid)


def test_wilson_known_values_and_boundary_reference():
    assert wilson_interval(5, 10) == pytest.approx((0.236593090512564, 0.763406909487436))
    z_squared = norm.ppf(0.975) ** 2
    assert wilson_interval(0, 10) == pytest.approx((0.0, z_squared / (10 + z_squared)))
    assert wilson_interval(10, 10) == pytest.approx((10 / (10 + z_squared), 1.0))


@pytest.mark.parametrize("successes,total,confidence", [(3, 17, 0.8), (20, 25, 0.99), (1, 1, 0.95)])
def test_wilson_endpoints_solve_score_equation(successes, total, confidence):
    lower, upper = wilson_interval(successes, total, confidence)
    z = norm.ppf((1 + confidence) / 2)
    proportion = successes / total
    for endpoint in (lower, upper):
        if endpoint not in (0.0, 1.0):
            assert total * (proportion - endpoint) ** 2 == pytest.approx(
                z * z * endpoint * (1 - endpoint)
            )
    complementary = wilson_interval(total - successes, total, confidence)
    assert complementary == pytest.approx((1 - upper, 1 - lower))


@pytest.mark.parametrize("successes,total", [(-1, 10), (11, 10), (0, 0), (0, -1), (0.5, 10), (1, 2.0), (True, 2)])
def test_invalid_wilson_counts(successes, total):
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def run_plan(*arguments):
    script = Path(__file__).resolve().parents[1] / "scripts" / "plan.py"
    return subprocess.run([sys.executable, str(script), *arguments], capture_output=True, text=True)


def test_cli_prints_per_arm_json_and_assumptions():
    completed = run_plan("--effect", "-0.5", "--sd-a", "1", "--sd-b", "1", "--power", "0.8", "--alpha", "0.05")
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["n"] == result["n_per_arm"] == 64
    assert result["total_training_seeds"] == 128
    assert result["effect_absolute"] == 0.5
    assert result["modeled_power"] == pytest.approx(reference_power(0.5, 1.0, 1.0, 64))
    assert result["unplannable_reason"] is None
    assert any("approximate" in assumption for assumption in result["assumptions"])
    assert any("not paired" in assumption for assumption in result["assumptions"])


@pytest.mark.parametrize("effect,sd_a,sd_b,reason", [("0", "1", "1", "zero_effect"), ("1", "0", "0", "zero_total_variance")])
def test_cli_unplannable_budget_is_json_null(effect, sd_a, sd_b, reason):
    completed = run_plan("--effect", effect, "--sd-a", sd_a, "--sd-b", sd_b)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["n_per_arm"] is None
    assert result["modeled_power"] is None
    assert result["unplannable_reason"] == reason


@pytest.mark.parametrize("flag,value", [("--sd-a", "-1"), ("--sd-b", "nan"), ("--effect", "inf"), ("--alpha", "0"), ("--power", "1"), ("--sd-a", "1e-300"), ("--sd-b", "1e300")])
def test_cli_rejects_invalid_inputs(flag, value):
    arguments = {"--effect": "1", "--sd-a": "1", "--sd-b": "1", "--power": "0.8", "--alpha": "0.05"}
    arguments[flag] = value
    completed = run_plan(*(item for pair in arguments.items() for item in pair))
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert not completed.stdout
