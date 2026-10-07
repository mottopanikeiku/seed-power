"""Import the complete published Control Clock final cohort, without running training.

Usage: python scripts/import_public.py [--archive PATH] [--output data/public_scores.csv]
Source URLs, hashes, endpoint and permissions are in data/public_sources.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/public_sources.json"
MAX_BYTES = 100_000_000
FIELDS = [
    "dataset", "claim_type", "task", "algorithm", "variant", "seed", "score", "score_units",
    "evaluation_episodes", "evaluation_steps", "solved", "stop_reason",
    "source_path", "source_url", "source_sha256",
]

def checked_bytes(payload: bytes, expected: str) -> bytes:
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Source SHA256 mismatch")
    return payload


def fetch_archive(source: dict) -> bytes:
    # Only the committed immutable public URL is requested. No accounts or tokens.
    with urllib.request.urlopen(source["url"], timeout=60) as response:
        buffer = io.BytesIO()
        while chunk := response.read(65536):
            if buffer.tell() + len(chunk) > MAX_BYTES:
                raise ValueError("Source archive exceeds 100 MB limit")
            buffer.write(chunk)
    return checked_bytes(buffer.getvalue(), source["sha256"])


def parse_record(payload: bytes, source: dict) -> dict:
    checked_bytes(payload, source["sha256"])
    record = json.loads(payload)
    task, method, variant, seed = source["identity"]
    if (record["task"], record["method"], record["variant"], record["seed"]) != (
        task, method, variant, seed
    ) or record["phase"] != "final":
        raise ValueError("Source record identity or phase mismatch")
    if record["error"] is not None or record["returncode"] != 0:
        raise ValueError("Source record has no valid observed score")
    evaluation = record["evaluations"][-1]
    returns = evaluation["returns"]
    episodes = record["evaluation_episodes"]
    score = float(evaluation["mean_return"])
    if len(returns) != episodes or episodes != 100:
        raise ValueError("Unexpected episode count")
    if not all(math.isfinite(x) for x in returns) or not math.isfinite(score):
        raise ValueError("Nonfinite source score")
    if not math.isclose(sum(returns) / episodes, score, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Reported episode mean does not match raw returns")
    return {
        "dataset": "control-clock-final", "claim_type": "observed_training_run", "task": task,
        "algorithm": method, "variant": variant, "seed": seed, "score": score,
        "score_units": "undiscounted_episode_return_at_stopping",
        "evaluation_episodes": episodes, "evaluation_steps": evaluation["steps"],
        "solved": record["solved"], "stop_reason": record["stop_reason"],
        "source_path": source["path"], "source_url": source["url"],
        "source_sha256": source["sha256"],
    }


def import_archive(payload: bytes, manifest: dict) -> list[dict]:
    checked_bytes(payload, manifest["archive"]["sha256"])
    rows = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        prefix = manifest["archive"]["member_prefix"]
        # Read members directly; never extract potentially unsafe filesystem paths.
        for source in manifest["records"]:
            member = archive.getmember(prefix + source["path"])
            if not member.isfile() or member.size > MAX_BYTES:
                raise ValueError("Invalid source archive member")
            with archive.extractfile(member) as stream:
                rows.append(parse_record(stream.read(), source))
    identities = {(r["task"], r["algorithm"], r["variant"], r["seed"]) for r in rows}
    if len(identities) != len(rows):
        raise ValueError("Duplicate training seed identity")
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an exact pinned archive offline")
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_scores.csv")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    payload = (
        args.archive.read_bytes() if args.archive else fetch_archive(manifest["archive"])
    )
    rows = import_archive(payload, manifest)
    write_csv(rows, args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
