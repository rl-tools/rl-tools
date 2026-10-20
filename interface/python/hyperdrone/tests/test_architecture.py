"""Architecture tests: the import DAG is enforced, not aspirational.

render and dynamics are peers that never see each other; env is the only integration
point; jit and cuda are infrastructure and import nothing from the domain packages.
Each check runs in a fresh interpreter so this test's own imports cannot contaminate it.
"""
import json
import subprocess
import sys

CHECK = """
import json, sys
import {package}
modules = sorted(name for name in sys.modules if name.startswith("hyperdrone"))
print(json.dumps(modules))
"""


def imported_modules(package):
    result = subprocess.run(
        [sys.executable, "-c", CHECK.format(package=package)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_render_never_imports_dynamics_or_env():
    modules = imported_modules("hyperdrone.render")
    assert not any(name.startswith("hyperdrone.dynamics") for name in modules)
    assert not any(name.startswith("hyperdrone.env") for name in modules)


def test_dynamics_never_imports_render_or_env():
    modules = imported_modules("hyperdrone.dynamics")
    assert not any(name.startswith("hyperdrone.render") for name in modules)
    assert not any(name.startswith("hyperdrone.env") for name in modules)


def test_infrastructure_imports_no_domain_packages():
    for package in ("hyperdrone._jit", "hyperdrone._cuda", "hyperdrone.camera", "hyperdrone._output_buffers"):
        modules = imported_modules(package)
        for domain in ("hyperdrone.render", "hyperdrone.dynamics", "hyperdrone.env"):
            assert not any(name.startswith(domain) for name in modules), (package, domain)


def test_root_import_is_lazy():
    modules = imported_modules("hyperdrone")
    assert modules == ["hyperdrone"]


def test_manual_spy_works_without_ipython_or_its_adapter():
    subprocess.run([sys.executable, "-c", """
import sys
from types import SimpleNamespace

class BlockIPython:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("IPython") or fullname == "hyperdrone.render._spy_ipython":
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockIPython())
from hyperdrone.render import Renderer
calls = []
renderer = Renderer.__new__(Renderer)
renderer._renderer = SimpleNamespace(
    spy_open=lambda camera: calls.append(("open", camera)),
    spy_update=lambda: True,
    spy_close=lambda: calls.append("close"),
)
renderer._spy = None
with renderer.spy(camera=2) as spy:
    assert spy.update()
assert not spy.update()
assert calls == [("open", 2), "close"]
assert "hyperdrone.render._spy_ipython" not in sys.modules
assert not any(name.startswith("IPython") for name in sys.modules)
"""], check=True, capture_output=True, text=True, timeout=20)
