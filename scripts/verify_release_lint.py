"""Reject Ruff diagnostics beyond the reviewed immutable-source debt baseline."""

import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/release_assembly/ruff-baseline.json"


def fingerprints(diagnostics, root):
    records = Counter()
    for item in diagnostics:
        path = Path(item["filename"])
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        source = path.read_text(encoding="utf-8-sig").splitlines()
        row = item["location"]["row"] - 1
        records[(relative, item["code"], item["message"], source[row].strip() if row < len(source) else "")] += 1
    return records


def excess_diagnostics(actual, allowed):
    return actual - allowed


def main():
    version = subprocess.check_output(["ruff", "--version"], text=True).strip()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if version != baseline["ruff_version"]:
        raise RuntimeError("Ruff version changed; review baseline with the pinned tool")
    result = subprocess.run(
        ["ruff", "check", "--config", "backend/pyproject.toml", "backend/", "scripts/", "--output-format", "json"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    actual = fingerprints(json.loads(result.stdout), ROOT)
    allowed = Counter({tuple(row["fingerprint"]): row["count"] for row in baseline["diagnostics"]})
    added = excess_diagnostics(actual, allowed)
    for finding, count in sorted(added.items()):
        print(f"NEW ({count}) {finding[0]} {finding[1]} {finding[2]} :: {finding[3]}")
    print(json.dumps({"total_diagnostics": sum(actual.values()), "new_diagnostics": sum(added.values()), "baseline_sources": baseline["sources"]}))
    return int(bool(added))


if __name__ == "__main__":
    raise SystemExit(main())
