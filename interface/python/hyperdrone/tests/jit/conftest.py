import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_cache(tmp_path_factory):
    """The toy component builds in ~a second, so jit tests always run against a fresh
    cache — exercising the configure path every time and never polluting the user's."""
    cache = tmp_path_factory.mktemp("hyperdrone-cache")
    previous = os.environ.get("HYPERDRONE_CACHE_DIR")
    os.environ["HYPERDRONE_CACHE_DIR"] = str(cache)
    yield cache
    if previous is None:
        del os.environ["HYPERDRONE_CACHE_DIR"]
    else:
        os.environ["HYPERDRONE_CACHE_DIR"] = previous
