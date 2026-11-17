"""Exercise the actual dataset builder without touching existing training files."""
import json
from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_v6_dataset.py"


def build(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    output = tmp_path / "backend" / "data" / "training"
    return [[json.loads(line) for line in (output / name).read_text().splitlines()] for name in ("v6_train.jsonl", "v6_val.jsonl")]


def fingerprints(rows):
    return {json.dumps(row, sort_keys=True) for row in rows}


def test_v6_train_validation_have_zero_overlap(tmp_path, monkeypatch):
    train, val = build(tmp_path, monkeypatch)
    assert train and val
    assert fingerprints(train).isdisjoint(fingerprints(val))


def test_v6_rows_are_deduplicated(tmp_path, monkeypatch):
    train, val = build(tmp_path, monkeypatch)
    assert len(train) == len(fingerprints(train))
    assert len(val) == len(fingerprints(val))
    assert len(train) + len(val) == 7


def test_v6_sources_do_not_cross_splits(tmp_path, monkeypatch):
    train, val = build(tmp_path, monkeypatch)
    source = lambda row: (row["system"], row["instruction"])
    assert {source(row) for row in train}.isdisjoint({source(row) for row in val})


def test_v6_split_is_reproducible(tmp_path, monkeypatch):
    first = build(tmp_path, monkeypatch)
    assert build(tmp_path, monkeypatch) == first
