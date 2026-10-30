import numpy as np

from ._component import load_core
from ._renderer import HyperLux
from .camera import CameraSpec, Probes


class FreeSpace:
    """Score-ranked free-space poses of a scene from RLtools' annotation scan, the table the C++
    HyperDrone environment spawns from: Halton candidates inside the scene bounds are probed
    along a deterministic direction set and accepted by hit ratio, mean hit distance and
    min_clearance (see hyperlux.FreeSpaceParameters). Works on every backend. Loaded GLB scenes
    with HSSD region annotations also require an indoor room and exclude outdoor regions.
    Loaded GLB scenes cache their table under cache_directory by content hash, probe count and parameters
    (default: RLtools' ~/.cache/rl_tools/annotations, shared with the C++ tools; "" disables);
    procedural scenes are scanned every time. batch is the probe transport width and does not
    affect the result.
    """

    def __init__(self, scene, *, probes=64, batch=64, parameters=None, cache_directory=None):
        core = load_core()
        if parameters is None:
            parameters = core.FreeSpaceParameters()
        if cache_directory is None:
            cache_directory = core.default_annotation_cache_directory()
        renderer = HyperLux(
            width=1, height=1, cameras=(CameraSpec("probe", outputs=(Probes(),)),), batch_size=int(batch), num_probes=int(probes), fidelity="low",
        )
        renderer.init(scene)
        renderer.generate_probe_directions()
        table = renderer._renderer.annotate_free_space(parameters, str(cache_directory))
        if len(table) == 0:
            raise RuntimeError(
                "hyperlux: free-space annotation found no valid positions (min_clearance too large "
                "for this scene, or the scene is not enclosed?)"
            )
        self.parameters = parameters
        self.positions = np.ascontiguousarray(table[:, :3])
        self.yaw = np.ascontiguousarray(table[:, 3])
        self.score = np.ascontiguousarray(table[:, 4])

    def __len__(self):
        return len(self.positions)

    def sample(self, count, *, seed=0):
        """count positions drawn uniformly with replacement from the table, (count, 3) float32;
        deterministic given seed (np.random.default_rng), as the C++ environment samples."""
        rng = np.random.default_rng(seed)
        return self.positions[rng.integers(0, len(self), size=int(count))]
