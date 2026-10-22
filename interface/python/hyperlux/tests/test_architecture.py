"""Architecture tests: the import DAG is enforced, not aspirational.

hyperlux never imports hyperdrone, importing it loads no native module, and the infrastructure
(jit, cuda staging, camera descriptors, output buffers) imports nothing from the renderer modules.
Each check runs in a fresh interpreter so this test's own imports cannot contaminate it.
"""
import json
import subprocess
import sys

CHECK = """
import json, sys
import {package}
modules = sorted(name for name in sys.modules if name.startswith(("hyperlux", "hyperdrone")))
print(json.dumps(modules))
"""


def imported_modules(package):
    result = subprocess.run(
        [sys.executable, "-c", CHECK.format(package=package)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def test_import_loads_no_native_module_and_nothing_from_hyperdrone():
    modules = imported_modules("hyperlux")
    assert not any(name.startswith("hyperdrone") for name in modules)
    assert not any(name.startswith("hyperlux_") for name in modules)


def test_infrastructure_import_builds_nothing_and_registers_nothing():
    for module in ("hyperlux._jit", "hyperlux._cuda", "hyperlux.camera", "hyperlux._output_buffers", "hyperlux._dlpack", "hyperlux._outputs"):
        result = subprocess.run([sys.executable, "-c", f"""
import json, sys
import {module}
from hyperlux import _cuda
print(json.dumps([sorted(name for name in sys.modules if name.startswith("hyperlux_")), _cuda._buffer_type is None]))
"""], capture_output=True, text=True, check=True)
        native, unregistered = json.loads(result.stdout)
        assert native == [], (module, native)
        assert unregistered, module


def test_manual_spy_works_without_ipython_or_its_adapter():
    subprocess.run([sys.executable, "-c", """
import sys
from types import SimpleNamespace

class BlockIPython:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("IPython") or fullname == "hyperlux._spy_ipython":
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockIPython())
from hyperlux import HyperLux
calls = []
renderer = HyperLux.__new__(HyperLux)
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
assert "hyperlux._spy_ipython" not in sys.modules
assert not any(name.startswith("IPython") for name in sys.modules)
"""], check=True, capture_output=True, text=True, timeout=20)
