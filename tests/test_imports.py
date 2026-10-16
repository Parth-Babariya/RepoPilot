import importlib
import pkgutil

import repopilot


def test_every_module_imports():
    """Catches syntax and import errors in modules no other test touches (e.g. main.py)."""
    failures = {}
    for module in pkgutil.walk_packages(repopilot.__path__, prefix="repopilot."):
        try:
            importlib.import_module(module.name)
        except Exception as e:
            failures[module.name] = f"{type(e).__name__}: {e}"
    assert not failures, failures
