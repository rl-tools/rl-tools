import numpy as np

from ._renderer import Renderer


class ClearanceSampler:
    """Deterministic rejection sampler for positions clear along the collision-probe rays.

    Rides the renderer's collision-probe machinery (works on every backend, including
    GENERIC on CPU-only machines): batches of candidate positions become probe origins
    casting a deterministic direction set; a candidate is free iff every probe either
    misses or hits farther than the clearance radius.
    """

    def __init__(self, scene, *, clearance=0.4, batch=256, probes=64, margin=0.05):
        if int(probes) < 1:
            raise ValueError("hyperdrone: ClearanceSampler needs at least one probe")
        self._clearance = float(clearance)
        self._batch = int(batch)
        self._margin = float(margin)
        self._renderer = Renderer(
            width=1, height=1, num_cameras=self._batch, num_probes=int(probes),
            output="depth", fidelity="low",
        )
        self._renderer.init(scene)
        self._renderer.generate_probe_directions()
        bounds = self._renderer.scene_bounds
        self._center = np.asarray(bounds["center"], dtype=np.float64)
        self._half_extent = np.asarray(bounds["half_extent"], dtype=np.float64)

    @property
    def bounds(self):
        return {"center": self._center.copy(), "half_extent": self._half_extent.copy()}

    def clearance_distances(self, positions):
        """Minimum probe hit distance per position (inf where all probes miss)."""
        positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        results = np.empty(len(positions))
        for start in range(0, len(positions), self._batch):
            chunk = positions[start:start + self._batch]
            cameras = np.zeros((self._batch, 12), dtype=np.float32)
            cameras[:len(chunk), 0:3] = chunk
            cameras[:, 3] = 1.0  # arbitrary forward; probe directions are pre-generated
            self._renderer.set_cameras(cameras)
            self._renderer.probe()
            distances, hits = self._renderer.collisions()
            minimum = distances.min(axis=1)
            results[start:start + len(chunk)] = minimum[:len(chunk)]
        return results

    def sample(self, count, *, seed=0):
        """count positions meeting the probe clearance, (count, 3) float32; deterministic given seed."""
        rng = np.random.default_rng(seed)
        accepted = []
        attempts = 0
        shrink = 1.0 - self._margin
        while sum(len(chunk) for chunk in accepted) < count:
            attempts += self._batch
            if attempts > max(100000, 1000 * count):
                raise RuntimeError(
                    f"hyperdrone: clearance sampling stalled (clearance {self._clearance} "
                    "too large for this scene?)"
                )
            candidates = self._center + rng.uniform(-shrink, shrink, size=(self._batch, 3)) * self._half_extent
            distances = self.clearance_distances(candidates)
            free = candidates[distances >= self._clearance]
            if len(free):
                accepted.append(free)
        return np.concatenate(accepted)[:count].astype(np.float32)
