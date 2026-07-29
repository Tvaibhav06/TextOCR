"""
python-backend/tests/conftest.py
────────────────────────────────────────────────────────────────────────────────
Pytest configuration loaded before any plugin or test discovery.

Purpose: Stub out the globally-installed `langsmith` package, which ships a
pytest plugin (pytest11 entrypoint) that is incompatible with Python 3.12.
Specifically, langsmith uses pydantic v1 ForwardRef._evaluate() which requires
a `recursive_guard` keyword argument in Python 3.12, causing a TypeError crash
during pytest startup — before any ini file or CLI option is processed.

This conftest.py runs first (before the entrypoint plugin loader fires) and
replaces `langsmith` and its sub-modules with harmless empty stubs, preventing
the crash entirely without uninstalling or modifying the package.

This file must NOT contain any imports that trigger langsmith loading.
"""

import sys
import types


def _stub_langsmith() -> None:
    """Replace the langsmith package with empty stub modules."""
    stub_names = [
        "langsmith",
        "langsmith.utils",
        "langsmith.schemas",
        "langsmith.pytest_plugin",
        "langsmith.run_helpers",
    ]
    for name in stub_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


_stub_langsmith()
