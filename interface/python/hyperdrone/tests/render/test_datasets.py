import hashlib

import pytest

from hyperdrone import render


def test_catalog_identity_without_asset_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTA_ROOT", str(tmp_path / "missing_store"))
    dataset = render.datasets.Train200()
    assert dataset.name == "procthor_train_200"
    assert dataset.source == "ai2thor-hab/ProcTHOR"
    assert dataset.split == "train"
    assert dataset.representation == "glb"
    assert len(dataset) == 200
    identity = "".join(f"{scene.id}\0{scene.reference}\n" for scene in dataset)
    assert hashlib.sha256(identity.encode()).hexdigest() == "9115ad15f9d9c9d27b52943fd8d8c273adbe1e1ceafc6f1c79d716dc03c04989"
    assert dataset.references == [scene.reference for scene in dataset]
    assert not (tmp_path / "missing_store").exists()


def test_scene_lookup_and_immutable_catalog():
    dataset = render.datasets.Train200()
    assert dataset["ProcTHOR-Train-1"].reference == "conta:07b9d8e44f3891573b6292f8cc47bf7725a43b89"
    assert dataset[-1].id == "ProcTHOR-Train-199"
    assert dataset[-200].id == "ProcTHOR-Train-0"
    for index in (200, -201):
        with pytest.raises(IndexError):
            dataset[index]
    with pytest.raises(KeyError):
        dataset["ProcTHOR-Test-1"]
    with pytest.raises(AttributeError):
        dataset[0].reference = "conta:other"
    references = dataset.references
    references.clear()
    assert len(dataset.references) == 200
