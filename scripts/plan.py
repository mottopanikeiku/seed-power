"""Print an equal-allocation independent-training-seed budget as JSON."""

from __future__ import annotations

import argparse
import json
import math

from seed_power.stats import plan_seeds, power_at_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect", type=float, required=True, help="Raw pilot mean gap; sign ignored")
    parser.add_argument("--sd-a", type=float, required=True, help="Pilot seed-level SD in arm A")
    parser.add_argument("--sd-b", type=float, required=True, help="Pilot seed-level SD in arm B")
    parser.add_argument("--power", type=float, default=0.8, help="Target power (default: 0.8)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided size (default: 0.05)")
    args = parser.parse_args()
    variances = []
    for name, sd in (("sd-a", args.sd_a), ("sd-b", args.sd_b)):
        if not math.isfinite(sd) or sd < 0.0:
            parser.error(f"--{name} must be finite and nonnegative")
        variance = sd * sd
        if not math.isfinite(variance) or (sd > 0.0 and variance == 0.0):
            parser.error(f"--{name} squared must be representable as a finite positive variance")
        variances.append(variance)
    try:
        n = plan_seeds(args.effect, *variances, target=args.power, alpha=args.alpha)
        modeled_power = (
            power_at_n(args.effect, *variances, n=n, alpha=args.alpha) if n is not None else None
        )
    except (ValueError, FloatingPointError) as error:
        parser.error(str(error))
    reason = None
    if n is None:
        reason = "zero_effect" if args.effect == 0.0 else "zero_total_variance"
    result = {
        "n": n,
        "n_per_arm": n,
        "total_training_seeds": 2 * n if n is not None else None,
        "effect_absolute": abs(args.effect),
        "sd_a": args.sd_a,
        "sd_b": args.sd_b,
        "target_power": args.power,
        "alpha": args.alpha,
        "modeled_power": modeled_power,
        "unplannable_reason": reason,
        "assumptions": [
            "Two-sided independent Welch comparison, not paired training seeds.",
            "Equal allocation; n and n_per_arm both count training seeds in each arm.",
            "Noncentral-t approximation with variance-based Welch-Satterthwaite degrees of freedom.",
            "Approximately normal independent seed-level returns; unequal-variance power is approximate.",
            "Absolute pilot mean gap and pilot variances treated as fixed, without pilot uncertainty.",
            "No budget cap or substituted standard deviation; zero effect or total variance is unplannable.",
            "Nominal modeled power is not guaranteed prospective detection frequency.",
        ],
    }
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
