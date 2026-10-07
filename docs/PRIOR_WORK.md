# What I build on

Seed-count planning in RL is not new. I add a small **prospective calibration test**: estimate a budget from pilot training runs, commit it, run fresh independent seeds, and measure detection frequency. An 80% calculation assumes a population effect, variance and score distribution that a small pilot cannot establish.

## Closest work

- **Henderson et al. (AAAI 2018), [Deep Reinforcement Learning that Matters](https://arxiv.org/pdf/1709.06560).** Their “Random Seeds and Trials”, reporting and supplemental significance sections document variability and use bootstrap power analysis with an artificial score lift. They motivate uncertainty reporting rather than a universal seed budget.
- **Colas, Sigaud and Oudeyer (2018), [How Many Random Seeds?](https://arxiv.org/pdf/1806.08295v1), sections 3–5.** They already give statistical seed-count guidance, a pilot-to-additional-runs workflow, and examples of small-pilot and distributional failures. Merely calculating a count from a pilot is not new. My narrower extension is repeated prospective calibration with an explicitly committed plan and independent future training runs. I use established noncentral-t calculations with approximate Welch degrees of freedom, not a new power formula.
- **Agarwal et al. (NeurIPS 2021), [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/pdf/2108.13264), sections 2–4 and Appendix A, [rliable](https://github.com/google-research/rliable).** Stratified bootstrap intervals, IQM and performance profiles answer aggregate benchmark questions. Their Atari 100k analysis uses 100 training runs per game, each averaging 100 evaluation episodes. My per-task mean-return budgets are narrower and do not replace rliable's aggregate methods.
- **Patterson et al., [Empirical Design in Reinforcement Learning](https://arxiv.org/pdf/2304.01315v1) (2023 preprint; [JMLR 2024](https://jmlr.org/papers/v25/23-0183.html)), sections 2–6.** Their experimental-unit, hyperparameter-selection and designer-bias discussion motivates fixed comparisons and separate pilots. Equal numeric seed labels do not establish a paired experiment. Evaluation episodes are not independent training runs.
- **Albers and Lakens (2018), [When power analyses based on pilot data are biased](https://doi.org/10.1016/j.jesp.2017.09.004).** Pilot-effect misplanning is already known. A meaningful-effect target fixed in advance avoids estimating the target from a noisy pilot, but the pilot variance and model assumptions can still be wrong. I test the observed-gap rule here; I do not claim to have prospectively validated that alternative.

[Evaluation-Aware Reinforcement Learning (2025)](https://arxiv.org/abs/2509.19464) concerns training for more accurate policy evaluation, a different objective. A literature search cannot prove that prospective seed-budget calibration has never been studied. I claim an executable small replication and extension, not priority for a statistical idea.

I follow [eval-power](https://github.com/mottopanikeiku/eval-power)'s separation of historical calculations and new measurements. I adapt affine ARS/CEM and Gymnasium batches from [control-clock](https://github.com/mottopanikeiku/control-clock/tree/ee9c303de8e7698ca25569fc585e173e3e172bb8), MIT. [Environment notices](ENVIRONMENTS.md) retain Gymnasium MIT and Acrobot/RLPy BSD terms. These short-budget affine policies do not establish seed requirements for deep RL generally.

## Public data and permissions

The rliable Atari 100k route was unavailable on 2026-10-07. Anonymous requests to its [SPR object](https://storage.googleapis.com/rl-benchmark-data/atari_100k/SPR.json) and bucket listings returned HTTP 404; [issue 31](https://github.com/google-research/rliable/issues/31) records an earlier access problem. The code/notebook's Apache-2.0 license does not establish separate bucket-data terms. I import no unavailable Atari scores and make no Atari power claim.

Instead I reanalyse my already-public MIT [Control Clock release](https://github.com/mottopanikeiku/control-clock/tree/ee9c303de8e7698ca25569fc585e173e3e172bb8), not an external RL-paper survey. I import every final record linked by its pinned summary: 140 training runs, two tasks and five implementations. ARS, CEM and batched PPO have 20 runs per task; adapted CleanRL PPO and Zoo SB3 PPO have five. The source archive, every raw-record URL and SHA256, selection rule and license are in [public_sources.json](../data/public_sources.json). The importer verifies each last-evaluation score against its 100 raw episode returns.

Those scores are last-evaluation means at stopping policies, not equal-training-budget outcomes. A shared fixed evaluation set is reused for stopping and evaluation, including qualification at initialization. Bounded, ceiling-heavy and qualification-selected scores make population extrapolation especially weak.

I enumerate all ten unordered implementation pairs per task, with actual unequal allocation where applicable. In [public.json](../results/public.json), 19 of 20 are below 80% **plug-in model power for their observed gaps**. That is neither true power nor a significance result, and is not the fraction of all published RL comparisons. Dependent pairs receive no binomial interval. Illustrative equal-variance standardized effects 0.2, 0.5, 0.8 and 1.0 require 394, 64, 26 and 17 independent seeds per arm, respectively; [the table](../results/public_standardized.csv) records assumptions and achieved modeled power.

```sh
uv run python scripts/import_public.py
uv run python scripts/public_analysis.py
```

## Neural PPO extension

I extend the same prospective question to small neural policies. I adapt
[Control Clock's JAX PPO](https://github.com/mottopanikeiku/control-clock/blob/b090dad8a3b53f937782676c6033ba3557cedfa1/control_clock/jax_ppo.py)
(MIT), which follows [PureJaxRL](https://github.com/luchris429/purejaxrl)'s
JAX/vmap/scan design (Apache-2.0; retained notice in `third_party/`).
I replace qualification stopping with a fixed transition budget and evaluate
once in Gymnax on a common fixed reset set. Optimizers and training randomness
remain independent across root training seeds. This removes the affine-only
limitation, not the small-task limitation, and introduces no new power formula.
The historical and neural cohorts are separate experiments, not an algorithm
ranking or a controlled causal comparison of policy classes.
