import hashlib

import pytest

from hyperdrone import render

CATALOGS = [
    ("procthor", "Train200", "procthor_train_200", "ai2thor-hab/ProcTHOR", "train", 200, "9115ad15f9d9c9d27b52943fd8d8c273adbe1e1ceafc6f1c79d716dc03c04989"),
    ("hssd", "Train125", "hssd_train_125", "hssd/hssd-hab", "train", 125, "7cd6dbc35e1b4dcad093b72b10a4078b4b658948e517166253e99da57f1c8159"),
    ("hssd", "Val43", "hssd_val_43", "hssd/hssd-hab", "val", 43, "75581ea7f0a030eafdab0cbb74206bbfd77543f0c88fee9ceb93fb610119ef0c"),
]


@pytest.mark.parametrize("family, catalog, name, source, split, size, identity_hash", CATALOGS)
def test_catalog_identity_without_asset_resolution(monkeypatch, tmp_path, family, catalog, name, source, split, size, identity_hash):
    monkeypatch.setenv("CONTA_ROOT", str(tmp_path / "missing_store"))
    dataset = getattr(getattr(render.datasets, family), catalog)()
    assert dataset.name == name
    assert dataset.source == source
    assert dataset.split == split
    assert dataset.representation == "glb"
    assert len(dataset) == size
    identity = "".join(f"{scene.id}\0{scene.reference}\n" for scene in dataset)
    assert hashlib.sha256(identity.encode()).hexdigest() == identity_hash
    assert dataset.references == [scene.reference for scene in dataset]
    assert not (tmp_path / "missing_store").exists()


def test_scene_lookup_and_immutable_catalog():
    dataset = render.datasets.procthor.Train200()
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


def test_hssd_splits_are_disjoint_and_keyed_by_scene_id():
    train = render.datasets.hssd.Train125()
    val = render.datasets.hssd.Val43()
    assert train["107734119_175999932"].reference == "conta:b8dd4207a0221dd224f733ebc90627fca2b18011"
    assert val[0].id == "107734158_175999998"
    assert not {scene.id for scene in train} & {scene.id for scene in val}
    assert not set(train.references) & set(val.references)
    with pytest.raises(KeyError):
        val["107734119_175999932"]
    with pytest.raises(IndexError):
        val[43]
