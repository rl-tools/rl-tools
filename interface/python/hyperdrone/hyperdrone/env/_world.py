import numpy as np


class World:
    """Thin wiring of a Sim and a Renderer over one scene: spawn -> step -> frames.

    Sugar, not load-bearing — it connects sim.step -> camera_bases -> set_cameras ->
    render with the right streams and nothing else; every capability remains reachable by
    composing Sim and Renderer directly.
    """

    def __init__(self, scene, sim, renderer, mount=None, fov=None):
        if renderer.num_cameras != sim.num_drones:
            raise ValueError(
                f"hyperdrone: renderer has {renderer.num_cameras} cameras but the sim has "
                f"{sim.num_drones} drones — they must match"
            )
        self.scene = scene
        self.sim = sim
        self.renderer = renderer
        self.mount = mount
        self.fov = fov
        self._device_handoff = (
            renderer.backend == "optix" and sim.device == "cuda"
        )
        if getattr(renderer, "_scene", None) is None:
            renderer.init(scene)

    def spawn(self, positions, orientations=None, seed=0):
        """Place drones at the given positions (e.g. from FreeSpaceSampler.sample) with
        hover-initial state everywhere else."""
        positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(self.sim.num_drones, 3)
        self.sim.reset(seed=seed, sample_states=False)
        self.sim.state["position"] = positions
        if orientations is not None:
            orientations = np.ascontiguousarray(orientations, dtype=np.float32).reshape(self.sim.num_drones, 4)
            self.sim.state["orientation"] = orientations
        self._update_cameras()
        return self

    def _update_cameras(self):
        kwargs = {}
        if self.mount is not None:
            kwargs["mount"] = self.mount
        if self.fov is not None:
            kwargs["fov"] = self.fov
        kwargs["aspect"] = self.renderer.aspect
        if self._device_handoff:
            bases = self.sim.camera_bases(**kwargs)
            self.renderer.set_cameras(bases, stream=self.sim.stream)
        else:
            self.renderer.set_cameras(self.sim.camera_bases_numpy(**kwargs))

    def step(self, actions, render_target="all"):
        """Advance the dynamics one dt, retarget the cameras from the new poses, and
        render. Returns the renderer for output access (frames/depth/segmentation)."""
        self.sim.step(actions)
        self._update_cameras()
        self.renderer.render(render_target)
        return self.renderer
