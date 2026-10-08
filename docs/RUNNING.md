# Reproduce the measurements

## Recompute committed results

These commands use the committed per-training-seed scores. They do not train policies or spend cloud compute. Each analyzer rebuilds the pilot plan and requires identical counts, statuses and hashes; floats such as modeled power may differ in their last bits across CPU architectures, and outputs keep the committed values. Regenerated PNGs and the last digits of `results/public*` can still differ across machines.

```sh
uv sync --locked --python 3.12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/public_analysis.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/analyze.py confirmation
uv run python scripts/figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/neural_analyze.py confirmation
uv run python scripts/neural_figures.py
uv run ruff check .
uv run python -m pytest -q
```

The optional `uv run python scripts/import_public.py` downloads the pinned 0.7 MB Control Clock archive and checks the archive and raw-record hashes before regenerating `data/public_scores.csv`. The archive's MIT license is retained separately. No unavailable Atari data is required.

## Repeat the original fresh-seed confirmation

I committed and pushed the complete pilot and confirmation counts at
[`3186fed41eddb6e58f4ab34f9c9c30ef00ad1f2b`](https://github.com/mottopanikeiku/seed-power/commit/3186fed41eddb6e58f4ab34f9c9c30ef00ad1f2b)
before starting confirmation. The runner requires that exact commit and plan; it refuses to overwrite an existing result directory. A repeat uses the same pseudorandom training seeds, so it is a reproducibility check, not new independent statistical evidence.

```sh
git switch --detach 3186fed41eddb6e58f4ab34f9c9c30ef00ad1f2b
uv sync --locked --python 3.12
SEED_POWER_TIMEOUT_SECONDS=600 SEED_POWER_CONTAINERS=8 uv run modal run modal_app.py --stage confirmation --output results/replication/confirmation --plan-commit 3186fed41eddb6e58f4ab34f9c9c30ef00ad1f2b
uv run python scripts/analyze.py confirmation --confirmation results/replication/confirmation/runs.jsonl --output results/replication
```

A Modal account is required only for training. Each container requests one CPU core and 1 GiB RAM; numerical libraries use one thread, with at most eight concurrent containers. The exact cloud environments and per-batch execution durations are in `results/*/environment.json`. Modal's sandbox reports the CPU model as unknown. I report cloud costs conservatively, not laptop training times or a throughput ranking.

## Export or reproduce neural PPO confirmation

I pushed the complete neural count plan at
[`260bebb4b1ab2d827ef06371b3681a0293f2b999`](https://github.com/mottopanikeiku/seed-power/commit/260bebb4b1ab2d827ef06371b3681a0293f2b999)
before any confirmation results. Transport commit `37e9e46` adds source-versioned, durable, resumable batches without changing the training code, design or count plan. Use a Python 3.12 client:

```sh
git switch --detach 37e9e46
uv sync --locked --python 3.12
SEED_POWER_TIMEOUT_SECONDS=3600 uv run --python 3.12 modal run neural_modal.py --stage confirmation --output results/replication/neural --plan-commit 260bebb4b1ab2d827ef06371b3681a0293f2b999
uv run python scripts/neural_analyze.py confirmation --confirmation results/replication/neural/runs.jsonl --output results/replication/neural
uv run python scripts/neural_figures.py --neural results/replication/neural/summary.json --output figures/replication
```

Each GPU container requests one L4, two CPU cores and 8 GiB RAM; the concurrency limit is one. Roots are vmapped in batches of 64. Every complete batch is committed to the dedicated `seed-power-day-results` Volume before being returned. Existing batches for the same design, count plan, committed training-source tree and immutable runtime image are exported rather than retrained. Five sequential fresh-client attempts share the overall deadline, and rerunning resumes only missing batches. An empty cache trains the same pseudorandom roots again; that is reproducibility, not additional independent evidence. No model weights are downloaded. Per-batch persistence adds substantial cloud wall time; the throughput forecast measures training only, not total export time.

The CPU-only neural correctness tests are optional locally (`uv sync --locked --python 3.12 --extra neural`); CI installs that extra and exercises them. [Neural protocol and limitations](NEURAL_PROTOCOL.md).


## Run a new pilot and confirmation

Edit the design and phase seed bases before collecting any outcomes. Keep every cell, choose the meaningful comparisons beforehand, and commit the design. Use a new output directory for each phase. The pilot's `--output` path must match the analyzer's `--pilot` argument. The original commands were:

```sh
SEED_POWER_TIMEOUT_SECONDS=300 uv run modal run modal_app.py --stage development --output results/development
SEED_POWER_TIMEOUT_SECONDS=300 uv run modal run modal_app.py --stage pilot --output results/pilot
uv run python scripts/analyze.py plan
# Commit and push the pilot, design and results/pilot_plan.json before continuing.
SEED_POWER_TIMEOUT_SECONDS=600 uv run modal run modal_app.py --stage confirmation --output results/confirmation --plan-commit YOUR_PUSHED_PLAN_COMMIT
```

The maximum executable count is a design constraint, not a reduced count advertised as 80% power. Ineligible cells stay in the plan. For a prespecified meaningful raw mean difference, the stand-alone planner is `uv run python scripts/plan.py --effect 0.5 --sd-a 1 --sd-b 1`; its answer is per arm. Fix the target effect before looking at data. The planner does not model future variance drift or account for pilot-variance uncertainty.
