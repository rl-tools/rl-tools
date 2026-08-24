import numpy as np
import pytest

from hyperdrone import dynamics, render


def make_room(half=4.0, height=3.0, pillar=None):
    """A closed box room (normals inward and outward via double-sided quads), optionally
    with one square pillar."""
    scene = render.Scene()
    room = render.Object(name="room")

    def quad(v0, v1, v2, v3, color=(0.7, 0.7, 0.7)):
        vertices = np.array([v0, v1, v2, v3], dtype=np.float32)
        indices = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int32)
        room.add_mesh(render.Mesh(vertices, indices, color=color))

    quad((-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0))
    quad((-half, -half, height), (half, -half, height), (half, half, height), (-half, half, height))
    quad((-half, -half, 0), (half, -half, 0), (half, -half, height), (-half, -half, height))
    quad((-half, half, 0), (half, half, 0), (half, half, height), (-half, half, height))
    quad((-half, -half, 0), (-half, half, 0), (-half, half, height), (-half, -half, height))
    quad((half, -half, 0), (half, half, 0), (half, half, height), (half, -half, height))
    if pillar is not None:
        x, y, r = pillar
        quad((x - r, y - r, 0), (x + r, y - r, 0), (x + r, y - r, height), (x - r, y - r, height), (0.9, 0.3, 0.3))
        quad((x - r, y + r, 0), (x + r, y + r, 0), (x + r, y + r, height), (x - r, y + r, height), (0.9, 0.3, 0.3))
        quad((x - r, y - r, 0), (x - r, y + r, 0), (x - r, y + r, height), (x - r, y - r, height), (0.9, 0.3, 0.3))
        quad((x + r, y - r, 0), (x + r, y + r, 0), (x + r, y + r, height), (x + r, y - r, height), (0.9, 0.3, 0.3))
    scene.add_object(room)
    scene.add_light(render.SceneLight.directional(direction=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0)))
    return scene


CLEARANCE = 0.5


@pytest.fixture(scope="module")
def sampler():
    return render.FreeSpaceSampler(make_room(), clearance=CLEARANCE, batch=64, probes=32)


def test_sampling_determinism(sampler):
    first = sampler.sample(32, seed=0)
    second = sampler.sample(32, seed=0)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, sampler.sample(32, seed=1))


def test_sampling_respects_clearance(sampler):
    positions = sampler.sample(64, seed=0)
    assert positions.shape == (64, 3)
    # independent verification: re-probe every accepted sample
    distances = sampler.clearance_distances(positions)
    assert (distances >= CLEARANCE).all()
    # geometric verification against the known room. The probe set is a finite direction
    # sample, so the perpendicular distance can undercut the probe-measured clearance by
    # the angular discretization — allow 10%.
    slack = CLEARANCE * 0.1
    half, height = 4.0, 3.0
    assert (np.abs(positions[:, :2]) <= half - CLEARANCE + slack).all()
    assert (positions[:, 2] >= CLEARANCE - slack).all()
    assert (positions[:, 2] <= height - CLEARANCE + slack).all()


def test_sampling_avoids_obstacles():
    pillar = (1.0, 1.0, 0.5)
    sampler = render.FreeSpaceSampler(make_room(pillar=pillar), clearance=CLEARANCE, batch=64, probes=64)
    positions = sampler.sample(64, seed=0)
    x, y, r = pillar
    # no sample inside the pillar footprint (plus most of the clearance; probe sets are
    # finite so grazing corners can shave a little off the exact euclidean clearance)
    inside = (np.abs(positions[:, 0] - x) < r + CLEARANCE * 0.5) & (np.abs(positions[:, 1] - y) < r + CLEARANCE * 0.5)
    assert not inside.any()


def compose(scene, sim, renderer):
    """The manual render + dynamics wiring (host camera path): spawn and step closures."""
    renderer.init(scene)

    def update_cameras():
        renderer.set_cameras(sim.camera_bases_numpy(aspect=renderer.aspect))

    def spawn(positions):
        sim.reset(seed=0, sample_states=False)
        sim.state["position"] = np.ascontiguousarray(positions, dtype=np.float32)
        update_cameras()

    def step(actions):
        sim.step(actions)
        update_cameras()
        renderer.render()
        return renderer

    return spawn, step


def test_manual_composition_end_to_end():
    num_drones = 4
    scene = make_room()
    sim = dynamics.Sim(num_drones=num_drones, model="crazyflie", device="cpu")
    renderer = render.Renderer(width=32, height=32, num_cameras=num_drones, output="rgbd", fidelity="low")
    spawn, step = compose(scene, sim, renderer)
    sampler = render.FreeSpaceSampler(scene, clearance=CLEARANCE, batch=64, probes=32)
    positions = sampler.sample(num_drones, seed=0)
    spawn(positions)

    actions = np.zeros((num_drones, sim.action_dim), dtype=np.float32)
    out = step(actions)
    depth = out.depth()
    assert depth.shape == (num_drones, 32, 32)
    # every camera sits inside a closed room with >= clearance to any surface: all depths
    # positive and beyond a sanity floor, but finite (the room is closed)
    assert (depth > 0.05).all()
    assert (depth < 20.0).all()

    frame = out.frame()
    assert frame.shape == (num_drones, 32, 32, 4)
    assert frame[..., :3].max() > 0


def test_manual_composition_camera_tracks_drone():
    num_drones = 2
    scene = make_room()
    sim = dynamics.Sim(num_drones=num_drones, model="crazyflie", device="cpu")
    renderer = render.Renderer(width=16, height=16, num_cameras=num_drones, output="depth", fidelity="low")
    spawn, step = compose(scene, sim, renderer)
    spawn(np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]], dtype=np.float32))
    actions = np.zeros((num_drones, sim.action_dim), dtype=np.float32)
    first = step(actions).depth().copy()
    # drones drift under mid-throttle gravity mismatch; the camera must move with them
    for _ in range(30):
        out = step(actions)
    assert not np.array_equal(first, out.depth())
