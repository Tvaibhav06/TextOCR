"""
python-backend/run_tests.py
────────────────────────────────────────────────────────────────────────────────
Test launcher that stubs the globally-installed langsmith package before
invoking pytest.

langsmith 0.4.x ships a pytest11 entrypoint that uses pydantic v1 ForwardRef,
which is incompatible with Python 3.12 and crashes pytest startup.
This loader pre-stubs the problematic modules in sys.modules before
pytest's pluginmanager calls load_setuptools_entrypoints().

Usage:
    python run_tests.py                          # run all tests
    python run_tests.py tests/test_ocr.py -v    # specific file
    python run_tests.py -m "not live" -v        # skip live-model tests
    python run_tests.py -m "not slow" -v        # skip slow tests
"""

import sys
import types

# ── Stub langsmith before anything else loads ─────────────────────────────────
_LANGSMITH_STUBS = [
    "langsmith",
    "langsmith.utils",
    "langsmith.schemas",
    "langsmith.pytest_plugin",
    "langsmith.run_helpers",
    "langsmith.client",
    "langsmith.wrappers",
]
for _name in _LANGSMITH_STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

# ── Now import and run pytest ─────────────────────────────────────────────────
import pytest

if __name__ == "__main__":
    args = sys.argv[1:] or ["tests/", "-v"]
    sys.exit(pytest.main(args))
