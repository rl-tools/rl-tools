"""sdist self-containment: snapshot the required subset of the rl-tools repository into
hyperdrone/_vendor/rl-tools so `pip install hyperdrone` works without a checkout. The
manifest is explicit — never "copy everything". Wheels built directly from a checkout skip
vendoring (they are for local use where the checkout exists); release wheels are built
from the sdist, where the vendored tree is already on disk."""
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

VENDOR_MANIFEST = (
    "include/rl_tools",
    "include/conta",
    "src/rendering/raytracing",
    "CMakeLists.txt",
    "cmake",
    "LICENSE",
)


class VendorRlToolsHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "sdist":
            return
        repository = self._repository_root()
        if repository is None:
            return  # building from an sdist: the vendored tree is already in place
        force_include = build_data.setdefault("force_include", {})
        for entry in VENDOR_MANIFEST:
            source = repository / entry
            if source.exists():
                force_include[str(source)] = f"hyperdrone/_vendor/rl-tools/{entry}"

    def _repository_root(self):
        candidate = Path(self.root)
        for _ in range(8):
            if (candidate / "include" / "rl_tools").is_dir():
                return candidate
            candidate = candidate.parent
        return None
