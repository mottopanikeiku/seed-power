import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_modal import (
    batch_name, cache_identity, cache_key, completed_indices, write_batch, write_json_atomic,
)


def test_resume_lists_only_complete_in_range_batches():
    entries = [SimpleNamespace(path="confirmation-hash/" + name) for name in (
        "batch-000000.json.gz", "batch-000003.json.gz",
        "batch-000001.json.gz.partial", "batch-bad.json.gz", "environment.json",
    )]
    finished = completed_indices(entries, 5)
    assert finished == {0, 3}
    jobs = list(enumerate(["a", "b", "c", "d", "e"]))
    assert [(index, job) for index, job in jobs if index not in finished] == [
        (1, "b"), (2, "c"), (4, "e"),
    ]
    with pytest.raises(ValueError, match="outside"):
        completed_indices([SimpleNamespace(path="batch-000005.json.gz")], 5)


def test_complete_batch_is_atomic_and_recoverable_without_retraining(tmp_path):
    path = tmp_path / batch_name(12)
    result = {"records": [{"seed": 41, "score": 25.0, "episode_returns": [25.0] * 16}],
              "batch": {"index": 12, "useful_seeds": 1}, "environment": {"device_kind": ["L4"]}}
    write_batch(path, result)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".partial").exists()
    with gzip.open(path, "rt") as stream:
        assert json.load(stream) == result
    assert completed_indices([SimpleNamespace(path=str(path))], 20) == {12}


def test_cache_identity_changes_with_training_or_runtime_not_transport():
    identity = cache_identity("tree-one", "image-one")
    key = cache_key("confirmation", "design", "counts", identity)
    assert key == cache_key("confirmation", "design", "counts", identity)
    assert key != cache_key(
        "confirmation", "design", "counts", cache_identity("tree-two", "image-one"),
    )
    assert key != cache_key(
        "confirmation", "design", "counts", cache_identity("tree-one", "image-two"),
    )
    assert key != cache_key("confirmation", "design", "other-counts", identity)
    assert key != cache_key("pilot", "design", None, identity)


def test_interrupted_metadata_write_leaves_the_previous_complete_export(tmp_path, monkeypatch):
    path = tmp_path / "environment.json"
    previous = {"batches": [{"index": 0}], "plan_sha256": "committed-counts"}
    write_json_atomic(path, previous)
    original = Path.write_text

    def interrupt(self, text, *args, **kwargs):
        if self.name.endswith(".partial"):
            original(self, '{"batches":', *args, **kwargs)
            raise KeyboardInterrupt
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_json_atomic(path, {"batches": [{"index": 0}, {"index": 1}]})
    assert json.loads(path.read_text()) == previous
