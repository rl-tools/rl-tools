from dataclasses import replace

import pytest

from hyperdrone.dynamics import Sim
from hyperdrone.dynamics._config import SimConfig
from hyperdrone.env import EnvConfig

VALID = (("motor", "rotors"), ("ctbr", "ideal"), ("position", "rotors"))


@pytest.mark.parametrize("control", ["velocity", "CTBR", None])
def test_invalid_control_fails_before_build(control, monkeypatch):
    def build(*args, **kwargs):
        pytest.fail("invalid control reached JIT compilation")
    monkeypatch.setattr("hyperdrone.dynamics._sim.ensure_sim_library", build)
    with pytest.raises(ValueError, match="control"):
        Sim(4, control=control)
    with pytest.raises(ValueError, match="control"):
        EnvConfig(control=control)


@pytest.mark.parametrize("control, actuation, message", [
    ("ctbr", "rotors", "actuation='ideal'"),
    ("motor", "ideal", "control='ctbr'"),
    ("position", "ideal", "control='ctbr'"),
    ("motor", "kinematic", "actuation"),
])
def test_signal_and_actuation_must_agree(control, actuation, message, monkeypatch):
    def build(*args, **kwargs):
        pytest.fail("invalid configuration reached JIT compilation")
    monkeypatch.setattr("hyperdrone.dynamics._sim.ensure_sim_library", build)
    with pytest.raises(ValueError, match=message):
        Sim(4, control=control, actuation=actuation)
    with pytest.raises(ValueError, match=message):
        EnvConfig(control=control, actuation=actuation)


def test_control_and_actuation_change_jit_configuration():
    for base in (SimConfig(num_drones=4), EnvConfig(device="cpu")):
        assert base.key() == replace(base, control="motor", actuation="rotors").key()
        assert len({replace(base, control=control, actuation=actuation).key() for control, actuation in VALID}) == 3
        prefix = "HYPERDRONE_DYNAMICS" if isinstance(base, SimConfig) else "HYPERDRONE_ENV"
        for control, actuation in VALID:
            defines = replace(base, control=control, actuation=actuation).defines()
            assert defines[f"{prefix}_CONTROL"] == ("motor", "ctbr", "position").index(control)
            assert defines[f"{prefix}_ACTUATION"] == ("rotors", "ideal").index(actuation)


@pytest.mark.parametrize("control, actuation", [("ctbr", "ideal"), ("position", "rotors")])
def test_task_and_custom_spec_choose_their_own_controller(control, actuation):
    with pytest.raises(ValueError, match="own controller"):
        EnvConfig(task="visual_inertial_localization", preset="x500_fpv_imu", control=control, actuation=actuation)
    with pytest.raises(ValueError, match="spec_header supersedes.*control"):
        EnvConfig(spec_header=__file__, control=control, actuation=actuation)
