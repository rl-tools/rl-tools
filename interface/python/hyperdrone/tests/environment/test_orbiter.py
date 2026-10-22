"""The user task wrapper contract: compiles the Orbiter notebook's header (the spec_header
rung beyond constants) and flies it, so the notebook tracks the World's verb surface."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hyperdrone import examples
from hyperdrone import Config, HyperDrone
from hyperlux._jit import cache_root, source_root

WRITEFILE = "%%writefile my_task.h\n"


@pytest.fixture(scope="module")
def orbiter_header():
    notebook = json.loads((Path(examples.__file__).parent / "Orbiter.ipynb").read_text())
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    (source,) = [source for source in sources if source.startswith(WRITEFILE)]
    # a stable path next to the builds: the header's location is part of the JIT key
    header = cache_root() / "orbiter" / "my_task.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    content = source[len(WRITEFILE):]
    assert content == (Path(__file__).parent / "user_task_header.h").read_text()
    if not header.exists() or header.read_text() != content:
        header.write_text(content)
    return header


def test_orbiter_task(orbiter_header):
    data = Path(os.environ.get("HYPERDRONE_TEST_SCENE_DIR", source_root() / "tests" / "data"))
    scene, asset = data / "ProcTHOR-Train-1.glb", data / "x500.glb"
    if not scene.exists() or not asset.exists():
        pytest.skip(f"no ProcTHOR scene or x500 asset in {data}")
    lazy, explicit = (HyperDrone([scene], config=Config(spec_header=str(orbiter_header)), seed=0, drone_asset=asset, assets={"orbiter": asset}) for _ in range(2))
    try:
        assert lazy.device == "cpu"
        reset_all = np.ones(lazy.total_instances, dtype=bool)
        hold = np.zeros((lazy.total_instances, lazy.action_dim), dtype=np.float32)
        for env in (lazy, explicit):
            env.reset(reset_all)
            first = env.observe()
            env.step(hold)
        assert np.isfinite(first).all() and first.std() > 0
        # the chase reward starts at the orbit radius: the object circles the spawn
        np.testing.assert_allclose(lazy.rewards(), -1.5, atol=0.2)
        assert not lazy.terminated().any()
        # a frame observed without an explicit render goes through the task's render, which
        # moves the object, like the explicit one
        np.testing.assert_array_equal(lazy.observe(), explicit.observe())
    finally:
        lazy.close()
        explicit.close()
