from dataclasses import replace

import pytest

from hyperdrone.dynamics import Sim
from hyperdrone.dynamics._config import SimConfig
from hyperdrone.env import EnvConfig


@pytest.mark.parametrize("control", ["velocity", "CTBR", None])
def test_invalid_control_fails_before_build(control, monkeypatch):
    def build(*args, **kwargs):
        pytest.fail("invalid control reached JIT compilation")
    monkeypatch.setattr("hyperdrone.dynamics._sim.ensure_sim_library", build)
    with pytest.raises(ValueError, match="control"):
        Sim(4, control=control)
    with pytest.raises(ValueError, match="control"):
        EnvConfig(control=control)


def test_control_changes_jit_configuration():
    for base in (SimConfig(num_drones=4), EnvConfig(device="cpu")):
        assert base.key() == replace(base, control="motor").key()
        assert len({replace(base, control=control).key() for control in ("motor", "ctbr", "position")}) == 3
        prefix = "HYPERDRONE_DYNAMICS" if isinstance(base, SimConfig) else "HYPERDRONE_ENV"
        for index, control in enumerate(("motor", "ctbr", "position")):
            assert replace(base, control=control).defines()[f"{prefix}_CONTROL"] == index


@pytest.mark.parametrize("control", ["ctbr", "position"])
def test_task_and_custom_spec_choose_their_own_controller(control):
    with pytest.raises(ValueError, match="own controller"):
        EnvConfig(task="visual_inertial_localization", preset="x500_fpv_imu", control=control)
    with pytest.raises(ValueError, match="spec_header supersedes.*control"):
        EnvConfig(spec_header=__file__, control=control)
