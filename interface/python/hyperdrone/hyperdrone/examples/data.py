"""Shared data resolution and download helpers for runnable examples."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from hyperdrone import jit


PROCTHOR_FILENAME = "ProcTHOR-Train-1.glb"
PROCTHOR_REVISION = "db1af73cfd47318df311ed8821caa8e4e0ffeb00"
X500_FILENAME = "x500.glb"
X500_REVISION = PROCTHOR_REVISION


def _test_data_path(filename, revision, override_variable):
    override = os.environ.get(override_variable)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"hyperdrone: {override_variable} does not exist: {path}"
            )
        return path

    checkout_path = jit.source_root() / "tests" / "data" / filename
    if checkout_path.is_file():
        return checkout_path

    return Path(hf_hub_download(
        repo_id="rl-tools/test-data",
        repo_type="dataset",
        filename=filename,
        revision=revision,
        cache_dir=jit.cache_root() / "huggingface",
        local_files_only=bool(os.environ.get("HYPERDRONE_OFFLINE")),
        library_name="hyperdrone",
    ))


def procthor_scene_path():
    """Return the local ProcTHOR scene, downloading and caching it when necessary.

    Resolution order: HYPERDRONE_PROCTHOR_PATH, an rl-tools checkout's tests/data,
    then Hugging Face's versioned cache under <hyperdrone cache>/huggingface.
    """
    return _test_data_path(
        PROCTHOR_FILENAME,
        PROCTHOR_REVISION,
        "HYPERDRONE_PROCTHOR_PATH",
    )


def x500_model_path():
    """Return the local x500 GLB, downloading and caching it when necessary.

    Resolution order: HYPERDRONE_X500_PATH, an rl-tools checkout's tests/data,
    then Hugging Face's versioned cache under <hyperdrone cache>/huggingface.
    """
    return _test_data_path(
        X500_FILENAME,
        X500_REVISION,
        "HYPERDRONE_X500_PATH",
    )
