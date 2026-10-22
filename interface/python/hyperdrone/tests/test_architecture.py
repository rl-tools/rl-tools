"""Architecture tests: the import DAG is enforced, not aspirational.

The environment is the only integration point of dynamics and rendering: dynamics never imports
it, and importing hyperdrone loads none of its own submodules and no native module.
Each check runs in a fresh interpreter so this test's own imports cannot contaminate it.
"""
import json
import subprocess
import sys

CHECK = """
import json, sys
import {package}
modules = sorted(name for name in sys.modules if name.startswith(("hyperdrone", "hyperlux")))
print(json.dumps(modules))
"""


def imported_modules(package):
    result = subprocess.run(
        [sys.executable, "-c", CHECK.format(package=package)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_dynamics_never_imports_the_environment():
    modules = imported_modules("hyperdrone.dynamics")
    assert not any(name.startswith(("hyperdrone._environment", "hyperdrone._rotorcraft", "hyperdrone.gym")) for name in modules)


def test_root_import_is_lazy():
    modules = imported_modules("hyperdrone")
    assert [name for name in modules if name.startswith("hyperdrone")] == ["hyperdrone"]
    assert not any(name.startswith(("hyperdrone_", "hyperlux_")) for name in modules)
