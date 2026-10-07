import gzip
import json
from types import SimpleNamespace

import pytest

from neural_modal import batch_name, completed_indices, write_batch


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
