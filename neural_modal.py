"""Run seed-vmapped neural PPO on one L4; preserve every planned raw score."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import modal

TIMEOUT = int(os.environ.get("SEED_POWER_TIMEOUT_SECONDS", "300"))
app = modal.App("seed-power-neural")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda12]==0.6.2", "flax==0.10.6", "optax==0.2.5", "gymnax==0.0.9",
                 "numpy==2.2.6", "scipy==1.15.3", "gymnasium==1.2.1")
    .env({"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
          "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    .add_local_python_source("seed_power")
)


def jobs_for(design, stage, plans=None):
    from seed_power.experiment import make_jobs

    if stage == "development":
        for task_index, task in enumerate(design["tasks"]):
            for repeat in range(2):
                specs = [{"task": task, "variant": "base", "phase": stage,
                          "seed": design["phase_bases"][stage] + task_index * 10000
                                  + repeat * design["batch_size"] + index}
                         for index in range(design["batch_size"])]
                yield {"task": task, "variant": "base", "specs": specs,
                       "configuration": {**design["variants"]["base"],
                                         "population": design["population"],
                                         **design["task_overrides"][task]},
                       "evaluation_reset_seeds": design["evaluation_reset_seeds"][task]}
        return
    for job in make_jobs(design, stage, plans):
        job["configuration"].update(design["task_overrides"][job["task"]])
        job["evaluation_reset_seeds"] = design["evaluation_reset_seeds"][job["task"]]
        yield job


@app.function(image=image, gpu="L4", cpu=2, memory=8192, timeout=TIMEOUT, max_containers=1)
def run_batches(jobs, batch_size):
    import importlib.metadata
    import platform
    import time
    import jax
    from seed_power.neural import train_batch

    if jax.default_backend() != "gpu":
        raise RuntimeError("neural experiment requires a GPU, not a silent CPU fallback")
    environment = {"python": platform.python_version(), "os": platform.platform(),
                   "devices": [str(device) for device in jax.devices()],
                   "device_kind": [device.device_kind for device in jax.devices()],
                   "libraries": {name: importlib.metadata.version(name)
                                 for name in ("jax", "jaxlib", "flax", "optax", "gymnax",
                                              "gymnasium", "numpy", "scipy")}}
    for batch_index, job in enumerate(jobs):
        seeds = [spec["seed"] for spec in job["specs"]]
        padding = batch_size - len(seeds)
        # Duplicate padding is discarded, not treated as independent observations.
        # Stable shapes reuse compilation; each useful seed's optimizer remains isolated.
        padded = seeds + [seeds[-1]] * padding
        started = time.perf_counter()
        results = train_batch(job["task"], job["configuration"], padded,
                              job["evaluation_reset_seeds"])
        seconds = time.perf_counter() - started
        records = [{**record, **spec} for record, spec in
                   zip(results[:len(seeds)], job["specs"], strict=True)]
        yield {"records": records, "batch": {"index": batch_index, "task": job["task"],
               "variant": job["variant"], "useful_seeds": len(seeds), "padding_seeds": padding,
               "seconds_including_first_compile": seconds}, "environment": environment}


@app.local_entrypoint()
def main(stage: str = "development", output: str = "results/neural/development",
         design_path: str = "configs/neural_design.json",
         plan_path: str = "results/neural/pilot_plan.json", plan_commit: str = ""):
    design_bytes = Path(design_path).read_bytes()
    design = json.loads(design_bytes)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    plans = None
    plan_sha = None
    if stage == "confirmation":
        if not plan_commit or commit != plan_commit:
            raise ValueError("confirmation requires the exact pushed plan commit")
        source_paths = ["src/seed_power", "neural_modal.py", design_path, plan_path]
        subprocess.run(["git", "diff", "--quiet", plan_commit, "--", *source_paths], check=True)
        committed = subprocess.check_output(["git", "show", f"{plan_commit}:{plan_path}"])
        if committed != Path(plan_path).read_bytes():
            raise ValueError("plan differs from the committed count plan")
        if subprocess.check_output(["git", "show", f"{plan_commit}:{design_path}"]) != design_bytes:
            raise ValueError("design differs from the committed design")
        plans = json.loads(committed)
        plan_sha = hashlib.sha256(committed).hexdigest()
    jobs = list(jobs_for(design, stage, plans))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "runs.jsonl.gz"
    if raw_path.exists():
        raise ValueError("refusing to overwrite an existing result directory")
    metadata = {"stage": stage, "code_commit": commit, "plan_commit": plan_commit or None,
                "plan_sha256": plan_sha, "design_sha256": hashlib.sha256(design_bytes).hexdigest(),
                "hardware": "Modal NVIDIA L4", "cpu_cores_per_container": 2,
                "memory_gib_per_container": 8, "max_containers": 1,
                "timeout_seconds": TIMEOUT, "numerical_threads": 2, "batches": []}
    with gzip.open(raw_path, "wt", encoding="utf-8") as stream:
        for result in run_batches.remote_gen(jobs, design["batch_size"]):
            for record in result["records"]:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            metadata["batches"].append(result["batch"])
            metadata["environment"] = result["environment"]
            (destination / "environment.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
            print(f'{stage}: {len(metadata["batches"])}/{len(jobs)} batches', flush=True)
    print(f"Completed {stage}; raw per-training-seed records in {raw_path}")
