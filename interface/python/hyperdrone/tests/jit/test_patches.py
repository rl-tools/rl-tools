import pytest

from hyperdrone.jit import _patches, _workspace


@pytest.mark.parametrize("cached_sources", [False, True])
def test_assimp_reuses_archive_without_source_override(tmp_path, monkeypatch, cached_sources):
    monkeypatch.setattr(_workspace, "dependencies_root", lambda: tmp_path)
    donor = tmp_path / "render-generic"
    (donor / "assimp-download").mkdir(parents=True)
    (donor / "assimp-src").mkdir()
    (donor / "assimp-src/CMakeLists.txt").write_text("unverified source")
    archive_name = "assimp-v6.0.4.tar.gz"
    (donor / "assimp-download" / archive_name).write_bytes(b"archive for CMake to verify")
    target = tmp_path / "env-generic"
    if cached_sources:
        (target / "assimp-src").mkdir(parents=True)
    seeded = _patches.seed_dependencies(target.name)
    assert "ASSIMP" not in seeded
    assert (target / "assimp-download" / archive_name).read_bytes() == b"archive for CMake to verify"
    assert not (target / "assimp-src/CMakeLists.txt").exists()


def test_existing_archive_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(_workspace, "dependencies_root", lambda: tmp_path)
    archive_name = "assimp-v6.0.4.tar.gz"
    for component, contents in [("render-generic", b"donor"), ("env-generic", b"cached")]:
        directory = tmp_path / component / "assimp-download"
        directory.mkdir(parents=True)
        (directory / archive_name).write_bytes(contents)
    _patches.seed_dependencies("env-generic")
    assert (tmp_path / "env-generic/assimp-download" / archive_name).read_bytes() == b"cached"


def test_owl_commit_pin(tmp_path, monkeypatch):
    declaration = tmp_path / "cmake/optional/optix.cmake"
    declaration.parent.mkdir(parents=True)
    pin = "a" * 40
    declaration.write_text(f"FetchContent_Declare(OWL\n GIT_TAG {pin}\n)")
    monkeypatch.setattr(_patches, "source_root", lambda: tmp_path)
    assert _patches.declared_pin("OWL") == pin
    monkeypatch.setattr(_patches, "checkout_head", lambda tree: pin)
    assert _patches.matches_pin(tmp_path, pin)
    assert not _patches.matches_pin(tmp_path, "b" * 40)
