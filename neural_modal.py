"""Seed-vmapped PPO with durable batches and bounded fresh-client retries."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import modal

TIMEOUT = int(os.environ.get("SEED_POWER_TIMEOUT_SECONDS", "300"))
VOLUME_NAME = "seed-power-day-results"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
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


def batch_name(index):
    return f"batch-{index:06}.json.gz"


def completed_indices(entries, count):
    result = set()
    for entry in entries:
        name = Path(entry.path).name
        if name.startswith("batch-") and name.endswith(".json.gz"):
            text = name[len("batch-"):-len(".json.gz")]
            if text.isdigit():
                index = int(text)
                if not 0 <= index < count:
                    raise ValueError("saved batch index lies outside the committed jobs")
                result.add(index)
    return result


def write_batch(path, result):
    """Only a complete compressed file can be listed as a finished batch."""
    temporary = path.with_suffix(path.suffix + ".partial")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)
    temporary.replace(path)


@app.function(image=image, gpu="L4", cpu=2, memory=8192, timeout=TIMEOUT,
              max_containers=1, volumes={"/results": volume})
def run_batches(indexed_jobs, batch_size, run_key):
    import importlib.metadata
    import platform
    import jax
    from seed_power.neural import train_batch

    if jax.default_backend() != "gpu":
        raise RuntimeError("neural experiment requires a GPU, not a silent CPU fallback")
    volume.reload()
    destination = Path("/results") / run_key
    destination.mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "os": platform.platform(),
                   "devices": [str(device) for device in jax.devices()],
                   "device_kind": [device.device_kind for device in jax.devices()],
                   "libraries": {name: importlib.metadata.version(name)
                                 for name in ("jax", "jaxlib", "flax", "optax", "gymnax",
                                              "gymnasium", "numpy", "scipy")}}
    for batch_index, job in indexed_jobs:
        path = destination / batch_name(batch_index)
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                yield json.load(stream)
            continue
        seeds = [spec["seed"] for spec in job["specs"]]
        padding = batch_size - len(seeds)
        # Duplicate padding preserves the execution shape; it is never evidence.
        started = time.perf_counter()
        results = train_batch(job["task"], job["configuration"],
                              seeds + [seeds[-1]] * padding, job["evaluation_reset_seeds"])
        records = [{**record, **spec} for record, spec in
                   zip(results[:len(seeds)], job["specs"], strict=True)]
        result = {"records": records, "batch": {"index": batch_index, "task": job["task"],
                  "variant": job["variant"], "useful_seeds": len(seeds),
                  "padding_seeds": padding,
                  "seconds_including_first_compile": time.perf_counter() - started},
                  "environment": environment}
        write_batch(path, result)
        volume.commit()
        yield result


def collect_once(stage, output, design_path, plan_path, plan_commit):
    design_bytes = Path(design_path).read_bytes()
    design = json.loads(design_bytes)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    plans = None
    plan_sha = None
    if stage == "confirmation":
        if not plan_commit:
            raise ValueError("confirmation requires the pushed plan commit")
        # The transport runner may change; training code, design and counts may not.
        subprocess.run(["git", "diff", "--quiet", plan_commit, "--",
                        "src/seed_power", design_path, plan_path], check=True)
        committed = subprocess.check_output(["git", "show", f"{plan_commit}:{plan_path}"])
        if committed != Path(plan_path).read_bytes():
            raise ValueError("plan differs from the committed count plan")
        if subprocess.check_output(["git", "show", f"{plan_commit}:{design_path}"]) != design_bytes:
            raise ValueError("design differs from the committed design")
        plans = json.loads(committed)
        plan_sha = hashlib.sha256(committed).hexdigest()
    design_sha = hashlib.sha256(design_bytes).hexdigest()
    run_key = f"{stage}-{plan_sha or design_sha}"
    jobs = list(jobs_for(design, stage, plans))
    try:
        finished = completed_indices(volume.listdir(run_key), len(jobs))
    except modal.exception.NotFoundError:
        finished = set()
    missing = [(index, job) for index, job in enumerate(jobs) if index not in finished]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "environment.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("design_sha256") != design_sha or previous.get("plan_sha256") != plan_sha:
            raise ValueError("output directory belongs to another design or count plan")
    metadata = {"stage": stage, "code_commit": commit, "plan_commit": plan_commit or None,
                "plan_sha256": plan_sha, "design_sha256": design_sha,
                "hardware": "Modal NVIDIA L4", "cpu_cores_per_container": 2,
                "memory_gib_per_container": 8, "max_containers": 1,
                "timeout_seconds": TIMEOUT, "numerical_threads": 2,
                "volume": VOLUME_NAME, "volume_directory": run_key, "batches": []}
    raw_path = destination / "runs.jsonl.gz"
    temporary = destination / "runs.jsonl.gz.partial"
    seen = set()
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        def save(result):
            index = result["batch"]["index"]
            if index in seen:
                raise ValueError("duplicate saved batch")
            if [record["seed"] for record in result["records"]] != [
                spec["seed"] for spec in jobs[index]["specs"]
            ]:
                raise ValueError("saved batch seeds differ from the committed job")
            seen.add(index)
            for record in result["records"]:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            metadata["batches"].append(result["batch"])
            metadata["environment"] = result["environment"]
            metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

        for index in sorted(finished):
            compressed = b"".join(volume.read_file(f"{run_key}/{batch_name(index)}"))
            save(json.loads(gzip.decompress(compressed)))
        print(f"Resuming {stage}: {len(finished)} saved, {len(missing)} missing batches", flush=True)
        if missing:
            for result in run_batches.remote_gen(missing, design["batch_size"], run_key):
                save(result)
                print(f"{stage}: {len(seen)}/{len(jobs)} batches", flush=True)
    if len(seen) != len(jobs):
        raise ValueError("collection ended before every committed batch was exported")
    temporary.replace(raw_path)
    print(f"Completed {stage}; raw per-training-seed records in {raw_path}")


@app.local_entrypoint()
def main(stage: str = "development", output: str = "results/neural/development",
         design_path: str = "configs/neural_design.json",
         plan_path: str = "results/neural/pilot_plan.json", plan_commit: str = "",
         single_attempt: bool = False, attempts: int = 5):
    if single_attempt:
        collect_once(stage, output, design_path, plan_path, plan_commit)
        return
    if not 1 <= attempts <= 5:
        raise ValueError("attempts must lie between one and five")
    deadline = time.monotonic() + TIMEOUT
    command = [sys.executable, "-m", "modal", "run", "neural_modal.py", "--stage", stage,
               "--output", output, "--design-path", design_path, "--plan-path", plan_path,
               "--single-attempt"]
    if plan_commit:
        command += ["--plan-commit", plan_commit]
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the total client retry deadline expired")
        print(f"Fresh Modal client attempt {attempt}/{attempts}", flush=True)
        with subprocess.Popen(command) as process:
            try:
                code = process.wait(timeout=remaining)
            except BaseException:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code == 0:
            return
        if attempt < attempts:
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("all bounded client attempts failed; completed GPU batches remain in the volume")
