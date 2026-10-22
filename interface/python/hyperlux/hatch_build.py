"""Distribution self-containment: snapshot the required subset of the rl-tools repository into
hyperlux/_vendor/rl-tools so an installed hyperlux compiles against its own sources, whether it
came from an sdist, a release wheel or a wheel pip built straight from a checkout. The manifest is
explicit — never "copy everything". Editable installs skip vendoring and use the checkout."""
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

VENDOR_MANIFEST = (
    "include/rl_tools",
    "include/conta",
    "include/metra",
    "src/hyperlux",
    "CMakeLists.txt",
    "cmake",
    "LICENSE",
)


class VendorRlToolsHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if version == "editable":
            return
        repository = self._repository_root()
        if repository is None:
            return  # building from an sdist: the vendored tree is already in place
        force_include = build_data.setdefault("force_include", {})
        for entry in VENDOR_MANIFEST:
            source = repository / entry
            sources = sorted(source.rglob("*")) if source.is_dir() else [source]
            for path in sources:
                target = f"hyperlux/_vendor/rl-tools/{path.relative_to(repository).as_posix()}"
                if path.is_file() and not self.build_config.path_is_excluded(target):
                    force_include[str(path)] = target

    def _repository_root(self):
        candidate = Path(self.root)
        for _ in range(8):
            if (candidate / "include" / "rl_tools").is_dir():
                return candidate
            candidate = candidate.parent
        return None
