"""CPU-only independent RL training. Run after committing the confirmation plan."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import modal

TIMEOUT = int(os.environ.get("SEED_POWER_TIMEOUT_SECONDS", "300"))
CONTAINERS = int(os.environ.get("SEED_POWER_CONTAINERS", "8"))
app = modal.App("seed-power")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "gymnasium==1.2.1")
    .env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    .add_local_python_source("seed_power")
)


@app.function(image=image, cpu=1, memory=1024, timeout=TIMEOUT, max_containers=CONTAINERS)
def train(job):
    import importlib.metadata
    import platform
    import time
    from seed_power.training import train_batch

    started = time.perf_counter()
    records = train_batch(
        job["task"], job["configuration"], [spec["seed"] for spec in job["specs"]],
        job["evaluation_episodes"],
    )
    records = [{**record, **spec} for record, spec in zip(records, job["specs"], strict=True)]
    elapsed = time.perf_counter() - started
    model = next((line.split(":", 1)[1].strip()
                  for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "not reported")
    return {"records": records, "seconds": elapsed, "cpu_model": model,
            "os": platform.platform(), "python": platform.python_version(),
            "libraries": {name: importlib.metadata.version(name)
                          for name in ("numpy", "scipy", "gymnasium")}}


@app.local_entrypoint()
def main(stage: str = "development", output: str = "results/development",
         design_path: str = "configs/design.json", plan_path: str = "results/pilot_plan.json",
         plan_commit: str = ""):
    from seed_power.experiment import make_jobs

    design_bytes = Path(design_path).read_bytes()
    design = json.loads(design_bytes)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    plans = None
    plan_sha = None
    if stage == "confirmation":
        if not plan_commit or commit != plan_commit:
            raise ValueError("run confirmation from the exact pushed plan commit")
        committed = subprocess.check_output(["git", "show", f"{plan_commit}:{plan_path}"])
        if committed != Path(plan_path).read_bytes():
            raise ValueError("confirmation plan differs from the committed plan")
        committed_design = subprocess.check_output(["git", "show", f"{plan_commit}:{design_path}"])
        if committed_design != design_bytes:
            raise ValueError("experiment design differs from the committed design")
        plans = json.loads(committed)
        plan_sha = hashlib.sha256(committed).hexdigest()
    jobs = list(make_jobs(design, stage, plans))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "runs.jsonl"
    if raw_path.exists():
        raise ValueError("refusing to overwrite an existing run; choose a new output directory")
    metadata = {"stage": stage, "code_commit": commit, "plan_commit": plan_commit or None,
                "plan_sha256": plan_sha, "design_sha256": hashlib.sha256(design_bytes).hexdigest(),
                "hardware": "Modal CPU, no GPU", "cpu_cores_per_container": 1,
                "memory_gib_per_container": 1, "max_containers": CONTAINERS,
                "timeout_seconds": TIMEOUT, "numerical_threads": 1, "batches": []}
    with raw_path.open("w") as stream:
        for result in train.map(jobs, order_outputs=False):
            for record in result.pop("records"):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            metadata["batches"].append(result)
            (destination / "environment.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
    print(f"Completed {stage}: {len(jobs)} batches; raw scores in {raw_path}")
