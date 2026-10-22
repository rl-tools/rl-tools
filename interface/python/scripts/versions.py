"""The one version number of the hyperlux and hyperdrone packages.

Three literals carry it: hyperlux/__init__.py, hyperdrone/__init__.py and the exact hyperlux pin
in hyperdrone's pyproject.toml. set_version.py writes them; check() refuses any disagreement.
A hyperdrone post-release (0.3.0.post1) keeps the pin and hyperlux at the base version.

    python versions.py          prints the version, exits 1 on disagreement
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYPERLUX_INIT = ROOT / "hyperlux/hyperlux/__init__.py"
HYPERDRONE_INIT = ROOT / "hyperdrone/hyperdrone/__init__.py"
HYPERDRONE_PYPROJECT = ROOT / "hyperdrone/pyproject.toml"
VERSION = re.compile(r'^__version__ = "([^"]+)"$', re.M)
PIN = re.compile(r'"hyperlux==([^"]+)"')


def read():
    hyperlux = VERSION.search(HYPERLUX_INIT.read_text()).group(1)
    hyperdrone = VERSION.search(HYPERDRONE_INIT.read_text()).group(1)
    pin = PIN.search(HYPERDRONE_PYPROJECT.read_text()).group(1)
    return hyperlux, hyperdrone, pin


def base(version):
    return version.split(".post")[0]


def check():
    hyperlux, hyperdrone, pin = read()
    assert hyperlux == pin == base(hyperdrone), f"hyperlux {hyperlux}, hyperdrone {hyperdrone}, pin hyperlux=={pin}: run scripts/set_version.py"
    return hyperlux, hyperdrone


if __name__ == "__main__":
    try:
        hyperlux, hyperdrone = check()
    except AssertionError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    print(hyperdrone)
