import numpy as np

from conformal_seg.data import (
    DefectSegDataset,
    discover_items,
    load_pair,
    split_items,
)


def test_discovery_finds_all_annotated(synth_root):
    items = discover_items(synth_root, "synth")
    assert len(items) == 12
    assert all(i.mask_path.exists() for i in items)


def test_split_is_deterministic_and_disjoint(synth_root):
    items = discover_items(synth_root, "synth")
    a = split_items(items, seed=17)
    b = split_items(items, seed=17)
    assert [i.image_path for i in a["train"]] == [i.image_path for i in b["train"]]
    paths = [i.image_path for split in a.values() for i in split]
    assert len(paths) == len(set(paths)) == 12
    assert split_items(items, seed=18)["train"] != a["train"]


def test_load_pair_shapes_and_ranges(synth_root):
    item = discover_items(synth_root, "synth")[0]
    rgb, mask = load_pair(item, size=64)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.float32
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0
    assert mask.shape == (64, 64) and set(np.unique(mask)) <= {0, 1}
    assert mask.sum() > 0  # the ellipse survived the resize


def test_dataset_tensors(synth_root):
    items = discover_items(synth_root, "synth")
    ds = DefectSegDataset(items, size=64, train=True)
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert y.shape == (1, 64, 64)
    assert y.max() <= 1.0
