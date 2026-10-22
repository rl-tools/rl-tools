"""Write one version into the three literals versions.py checks.

    python set_version.py 0.3.0        both packages and the pin
    python set_version.py 0.3.0.post1  hyperdrone only; hyperlux and the pin stay at 0.3.0
"""
import re
import sys

from versions import HYPERDRONE_INIT, HYPERDRONE_PYPROJECT, HYPERLUX_INIT, PIN, VERSION, base, check


def write(path, pattern, replacement):
    text = path.read_text()
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"{path}: version literal not found")
    path.write_text(updated)


def main(version):
    if not re.fullmatch(r"\d+\.\d+\.\d+(\.post\d+)?", version):
        raise SystemExit(f"{version!r} is not MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH.postN")
    write(HYPERDRONE_INIT, VERSION, f'__version__ = "{version}"')
    if ".post" not in version:
        write(HYPERLUX_INIT, VERSION, f'__version__ = "{version}"')
    write(HYPERDRONE_PYPROJECT, PIN, f'"hyperlux=={base(version)}"')
    check()
    print(version)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
