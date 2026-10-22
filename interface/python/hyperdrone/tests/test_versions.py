"""hyperdrone pins hyperlux to its own version: the two package versions and the pin in
pyproject.toml must agree, as interface/python/scripts/set_version.py writes them."""
import importlib.util
from pathlib import Path


def test_versions_agree():
    script = Path(__file__).resolve().parents[2] / "scripts" / "versions.py"
    specification = importlib.util.spec_from_file_location("versions", script)
    versions = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(versions)
    versions.check()
