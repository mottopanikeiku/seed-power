"""Established power planning for independent training seeds, not episode pairs.

Colas et al., *How Many Random Seeds?* (2018), sections 3.1 and 4, motivate
Welch comparisons and pilot-based seed budgets: https://arxiv.org/abs/1806.08295.
Here I use a noncentral-t alternative rather than their shifted-central-t
illustration. Holding the Welch-Satterthwaite degrees of freedom fixed at the
supplied variances gives an established approximation, not exact unequal-
variance Welch power. Normal seed-level returns and accurate pilot estimates
are assumptions; a nominal budget does not guarantee prospective power.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.stats import chi2, nct, norm, t


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be representable as a finite float") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _probability(value: float, name: str) -> float:
    value = _finite_real(value, name)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return value


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _planning_inputs(effect: float, variance_a: float, variance_b: float, alpha: float):
    effect = abs(_finite_real(effect, "effect"))
    variance_a = _finite_real(variance_a, "variance_a")
    variance_b = _finite_real(variance_b, "variance_b")
    if variance_a < 0.0 or variance_b < 0.0:
        raise ValueError("variances must be nonnegative")
    alpha = _probability(alpha, "alpha")
    if alpha / 2.0 == 0.0:
        raise ValueError("alpha / 2 must be representable as a positive float")
    return effect, variance_a, variance_b, alpha


def _power(effect: float, variance_a: float, variance_b: float, n: int, alpha: float):
    scale = max(variance_a, variance_b)
    if scale == 0.0:
        # Deterministic limits, not ordinary t distributions.
        return float(effect != 0.0)
    if effect == 0.0:
        return alpha

    # Normalize before squaring: neither large nor tiny variances need a fake SD.
    a, b = variance_a / scale, variance_b / scale
    df_factor = (a + b) ** 2 / (a * a + b * b)
    try:
        df = float(n - 1) * df_factor
    except OverflowError:
        df = math.inf
    log_nc = math.log(effect) + 0.5 * (
        math.log(n) - math.log(scale) - math.log(a + b)
    )
    if log_nc > math.log(np.finfo(float).max):
        return 1.0
    nc = math.exp(log_nc)
    if math.isinf(df):
        # The genuine infinite-df t limit, only beyond float representability.
        critical = norm.isf(alpha / 2.0)
        power = norm.sf(critical - nc) + norm.cdf(-critical - nc)
    else:
        critical = t.isf(alpha / 2.0, df)
        power = nct.sf(critical, df, nc) + nct.cdf(-critical, df, nc)
        if not math.isfinite(power):
            # Some SciPy noncentral-t opposite tails return NaN at large nc.
            # Integrate the same normal/chi-square representation, not a
            # substituted distribution or a discarded rejection tail.
            def conditional_rejection(quantile):
                threshold = critical * math.sqrt(chi2.ppf(quantile, df) / df)
                return norm.cdf(nc - threshold) + norm.cdf(-nc - threshold)

            power, error = quad(
                conditional_rejection, 0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=200
            )
            if error > 1e-7:
                raise FloatingPointError("noncentral-t integration did not converge")
    if not math.isfinite(power):
        raise FloatingPointError("SciPy could not evaluate planning power for these inputs")
    return min(1.0, max(0.0, float(power)))


def power_at_n(
    effect: float, variance_a: float, variance_b: float, n: int, alpha: float = 0.05
) -> float:
    """Approximate two-sided Welch power with ``n`` independent seeds per arm.

    The raw mean gap is used in absolute value. With equal allocation,
    df = (n-1)*(variance_a+variance_b)**2/(variance_a**2+variance_b**2)
    and noncentrality = abs(effect)/sqrt((variance_a+variance_b)/n).
    Both rejection tails count. Zero total variance is a labeled-in-this-docstring
    deterministic limit: power 0 for identical means, 1 for distinct means.
    ``plan_seeds`` declines to infer a budget from either zero-variance case.
    """
    effect, variance_a, variance_b, alpha = _planning_inputs(
        effect, variance_a, variance_b, alpha
    )
    n = _integer(n, "n", 2)
    return _power(effect, variance_a, variance_b, n, alpha)


def plan_seeds(
    effect: float,
    variance_a: float,
    variance_b: float,
    target: float = 0.8,
    alpha: float = 0.05,
) -> int | None:
    """Return the smallest integer seeds per arm meeting approximate power.

    Return None for zero effect or zero total variance, even when target <=
    alpha. Otherwise use an uncapped exponential bracket and integer bisection
    of the increasing equal-allocation power curve, always starting at n=2.
    Pilot estimates are treated as fixed; their uncertainty is not integrated.
    Invalid inputs raise ValueError; numerical distribution failures are explicit.
    """
    effect, variance_a, variance_b, alpha = _planning_inputs(
        effect, variance_a, variance_b, alpha
    )
    target = _probability(target, "target")
    if effect == 0.0 or max(variance_a, variance_b) == 0.0:
        return None
    if _power(effect, variance_a, variance_b, 2, alpha) >= target:
        return 2
    lower, upper = 2, 4
    while _power(effect, variance_a, variance_b, upper, alpha) < target:
        lower, upper = upper, upper * 2
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if _power(effect, variance_a, variance_b, middle, alpha) >= target:
            upper = middle
        else:
            lower = middle
    return upper


def _sample(values: ArrayLike, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.size < 2 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional real sample with >= 2 seeds")
    values = values.astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def welch_test(a: ArrayLike, b: ArrayLike) -> dict[str, float | str | None]:
    """Two-sided unpaired Welch test; the mean difference is mean(a)-mean(b).

    No assumption pairs equal seed numbers across arms. Two constant groups
    have no estimable t distribution: identical constants are defined to have
    statistic=0 and p=1 (no rejection); distinct constants use the deterministic
    limit statistic=+/-inf and p=0. ``degeneracy`` labels these conventions and
    ``df`` is None. A single constant arm still permits an ordinary Welch test.
    """
    a, b = _sample(a, "a"), _sample(b, "b")
    constant_a, constant_b = bool(np.all(a == a[0])), bool(np.all(b == b[0]))
    if constant_a and constant_b:
        difference = float(a[0]) - float(b[0])
        identical = bool(a[0] == b[0])
        return {
            "pvalue": 1.0 if identical else 0.0,
            "statistic": 0.0 if identical else math.copysign(math.inf, difference),
            "mean_difference": difference,
            "df": None,
            "degeneracy": "identical_constants" if identical else "distinct_constants",
        }
    mean_a = float(a[0]) if constant_a else float(np.mean(a))
    mean_b = float(b[0]) if constant_b else float(np.mean(b))
    difference = mean_a - mean_b
    variance_a = 0.0 if constant_a else float(np.var(a, ddof=1))
    variance_b = 0.0 if constant_b else float(np.var(b, ddof=1))
    component_a, component_b = variance_a / a.size, variance_b / b.size
    scale = max(component_a, component_b)
    if (
        not math.isfinite(variance_a)
        or not math.isfinite(variance_b)
        or scale == 0.0
        or not math.isfinite(difference)
    ):
        raise FloatingPointError("sample moments cannot be represented for the Welch test")
    x, y = component_a / scale, component_b / scale
    df = (x + y) ** 2 / (x * x / (a.size - 1) + y * y / (b.size - 1))
    statistic = difference / (math.sqrt(scale) * math.sqrt(x + y))
    return {
        "pvalue": float(2.0 * t.sf(abs(statistic), df)),
        "statistic": statistic,
        "mean_difference": difference,
        "df": float(df),
        "degeneracy": None,
    }


def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for independent Bernoulli confirmation outcomes.

    At least one trial is required; no interval is invented for zero trials.
    """
    total = _integer(total, "total", 1)
    successes = _integer(successes, "successes", 0)
    if successes > total:
        raise ValueError("successes cannot exceed total")
    confidence = _probability(confidence, "confidence")
    z = float(norm.isf((1.0 - confidence) / 2.0))
    proportion = successes / total
    adjustment = z * z / total
    center = (proportion + adjustment / 2.0) / (1.0 + adjustment)
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / total + adjustment / (4.0 * total)
    ) / (1.0 + adjustment)
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == total else min(1.0, center + half_width)
    return lower, upper
