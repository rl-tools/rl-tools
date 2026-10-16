from __future__ import annotations

import numpy as np

from hypersplat.l2f_visual_tflite import (
    L2FVisualObservationBuilder,
    MahonyFilter,
    l2f_action_to_rotorpy,
    parse_observation_slug,
    split_top_level,
    spec_from_actor_meta,
)


OBS = (
    "CameraRGBStackedWithTarget(1.11352, 50, 80, 10, 10), "
    "OrientationWorldZ.AngularVelocity.LinearAccelerationBodyFrameHistory(1).ActionHistory(64)"
)


def test_split_top_level_keeps_nested_commas():
    assert split_top_level(OBS) == [
        "CameraRGBStackedWithTarget(1.11352, 50, 80, 10, 10)",
        "OrientationWorldZ.AngularVelocity.LinearAccelerationBodyFrameHistory(1).ActionHistory(64)",
    ]


def test_parse_current_l2f_visual_observation():
    spec = parse_observation_slug(OBS)
    assert spec.camera.height == 50
    assert spec.camera.width == 80
    assert spec.camera.frames == 10
    assert spec.camera.stride == 10
    assert spec.camera.with_target
    assert spec.camera.logical_image_inputs == 11
    assert spec.camera.logical_channels == 33
    assert spec.state.orientation == "OrientationWorldZ"
    assert spec.state.linear_acceleration_body_history == 1
    assert spec.state.action_history == 64
    assert spec.state.dim == 265


def test_spec_from_actor_meta_sets_output():
    spec = spec_from_actor_meta({"environment": {"observation": OBS, "output": "Action"}})
    assert spec.output == "Action"


def _fill_frame(spec, value: float) -> np.ndarray:
    return np.full(
        (spec.camera.height, spec.camera.width, spec.camera.channels_per_frame),
        value,
        dtype=np.float32,
    )


def test_logical_images_no_vision_delay_uses_current_frame():
    spec = parse_observation_slug(OBS)
    builder = L2FVisualObservationBuilder(None, spec)
    builder.frames = [_fill_frame(spec, float(i)) for i in range(4)]
    builder.target_frame = _fill_frame(spec, 99.0)
    builder.step_index = 3

    images = builder._logical_images()

    np.testing.assert_allclose(images[0], 3.0)
    np.testing.assert_allclose(images[1], 0.0)
    np.testing.assert_allclose(images[-1], 99.0)


def test_logical_images_applies_vision_delay_before_stack_stride():
    spec = parse_observation_slug(OBS)
    builder = L2FVisualObservationBuilder(None, spec, vision_delay_steps=2)
    builder.frames = [_fill_frame(spec, float(i)) for i in range(14)]
    builder.target_frame = _fill_frame(spec, 99.0)
    builder.step_index = 13

    images = builder._logical_images()

    np.testing.assert_allclose(images[0], 11.0)
    np.testing.assert_allclose(images[1], 1.0)
    np.testing.assert_allclose(images[2], 0.0)
    np.testing.assert_allclose(images[-1], 99.0)


def test_logical_images_clamps_vision_delay_to_episode_start():
    spec = parse_observation_slug(OBS)
    builder = L2FVisualObservationBuilder(None, spec, vision_delay_steps=5)
    builder.frames = [_fill_frame(spec, float(i)) for i in range(3)]
    builder.step_index = 2

    images = builder._logical_images()

    np.testing.assert_allclose(images[0], 0.0)
    np.testing.assert_allclose(images[1], 0.0)


def test_state_observation_adds_gyro_bias():
    spec = parse_observation_slug(OBS)
    builder = L2FVisualObservationBuilder(None, spec, gyro_bias=[0.1, -0.2, 0.3])
    builder.state = {
        "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "w": np.array([1.0, 2.0, 3.0], dtype=np.float32),
    }
    obs = builder._state_observation()
    np.testing.assert_allclose(obs[3:6], [1.1, 1.8, 3.3])


def test_mahony_filter_static_accel_world_z():
    filt = MahonyFilter()
    filt.update(0.0, 0.0, 9.80665, 0.0, 0.0, 0.0, 0.01)
    np.testing.assert_allclose(filt.orientation_world_z(), [0.0, 0.0, 1.0], atol=1e-6)


def test_state_observation_can_use_mahony_world_z():
    spec = parse_observation_slug(OBS)
    builder = L2FVisualObservationBuilder(None, spec, world_z_source="mahony")
    builder.state = {
        "q": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "w": np.zeros(3, dtype=np.float32),
    }
    builder.mahony_world_z = np.array([0.1, 0.2, 0.97], dtype=np.float32)
    obs = builder._state_observation()
    np.testing.assert_allclose(obs[:3], builder.mahony_world_z)


def test_mahony_update_uses_biased_gyro():
    class DummyDynamics:
        dt = 0.01

    class DummyEnv:
        dynamics = DummyDynamics()

    class Recorder:
        def __init__(self):
            self.args = None

        def update(self, *args):
            self.args = args

        def orientation_world_z(self):
            return (0.0, 0.0, 1.0)

    spec = parse_observation_slug(OBS)
    builder = L2FVisualObservationBuilder(
        DummyEnv(),
        spec,
        gyro_bias=[0.1, -0.2, 0.3],
        world_z_source="mahony",
    )
    builder.state = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    recorder = Recorder()
    builder.mahony = recorder
    builder._update_mahony(np.array([0.0, 0.0, 9.80665], dtype=np.float32))
    assert recorder.args is not None
    np.testing.assert_allclose(recorder.args[3:6], [1.1, 1.8, 3.3])


def test_l2f_action_to_rotorpy_order_and_clip():
    # RLtools/RAPTOR order [FR, BR, BL, FL] -> RotorPY [FL, FR, BR, BL].
    action = np.array([0.1, 0.2, 0.3, 2.0], dtype=np.float32)
    np.testing.assert_allclose(l2f_action_to_rotorpy(action), [1.0, 0.1, 0.2, 0.3])
