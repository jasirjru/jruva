"""The debt gate must reject new locations and added duplicate diagnostics."""

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_lint_gate_rejects_new_and_added_duplicate_findings():
    path = Path(__file__).resolve().parents[2] / "scripts/verify_release_lint.py"
    spec = spec_from_file_location("release_lint", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    old = ("backend/old.py", "F401", "unused", "import old")
    new = ("backend/new.py", "F821", "undefined", "missing()")
    allowed = Counter({old: 1})
    assert module.excess_diagnostics(Counter({old: 2, new: 1}), allowed) == Counter({old: 1, new: 1})
    assert not module.excess_diagnostics(Counter(), allowed)
