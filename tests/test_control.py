"""The defect-free control split: what the guarantee costs on parts that are fine.

FNR is blind to clean parts by construction (no defect pixels, nothing to miss),
so a threshold chosen to bound FNR can flag every clean part on the line and the
conformal report would still look perfect. These tests pin the other half down.
"""

import numpy as np

from conformal_seg.conformal import control_curve, control_report
from conformal_seg.data import discover_good_items, discover_items, load_image
from conformal_seg.metrics import flagged_fraction, image_flagged


def test_flagged_fraction_and_image_flag():
    pred = np.zeros((10, 10), dtype=bool)
    assert flagged_fraction(pred) == 0.0
    assert not image_flagged(pred, min_area_frac=1e-3)

    pred[0, 0] = True  # 1 of 100 px = 0.01
    assert flagged_fraction(pred) == 0.01
    assert image_flagged(pred, min_area_frac=1e-3)      # above the trigger
    assert not image_flagged(pred, min_area_frac=0.5)   # below a stricter one

    assert flagged_fraction(np.ones((4, 4), dtype=bool)) == 1.0


def _clean_probs(n=20, size=16, seed=3):
    """Prob maps for defect-free parts: mostly low, with a little noise."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.0, 0.4, (size, size)) for _ in range(n)]


def test_control_report_shape_and_thresholds():
    probs = _clean_probs()
    rep = control_report(probs, {"naive": 0.5, "conformal": 0.1}, min_area_frac=1e-3)

    assert rep["n_control"] == len(probs)
    assert set(rep["at"]) == {"naive", "conformal"}

    naive, conf = rep["at"]["naive"], rep["at"]["conformal"]
    # Nothing reaches 0.5 here, so the naive threshold raises no alarm at all.
    assert naive["mean_flagged_pixel_fraction"] == 0.0
    assert naive["image_false_alarm_rate"] == 0.0
    # Dropping the threshold can only flag more. This is the trade the conformal
    # layer makes, and it is why the control split has to exist.
    assert conf["mean_flagged_pixel_fraction"] > naive["mean_flagged_pixel_fraction"]
    assert conf["image_false_alarm_rate"] >= naive["image_false_alarm_rate"]
    assert 0.0 <= conf["image_false_alarm_rate"] <= 1.0
    assert conf["max_flagged_pixel_fraction"] >= conf["mean_flagged_pixel_fraction"]


def test_control_report_is_monotone_in_threshold():
    probs = _clean_probs(seed=11)
    rep = control_report(probs, {f"t{i}": t for i, t in enumerate([0.0, 0.2, 0.4, 0.6])})
    flagged = [rep["at"][f"t{i}"]["mean_flagged_pixel_fraction"] for i in range(4)]
    assert flagged == sorted(flagged, reverse=True)  # higher threshold, fewer pixels
    assert flagged[0] == 1.0  # t=0 flags everything


def test_control_curve_monotone_and_gridded():
    curve = control_curve(_clean_probs(seed=7), grid=np.linspace(0, 1, 21))
    assert len(curve) == 21
    thresholds = [r[0] for r in curve]
    assert thresholds == sorted(thresholds)
    flagged = [r[1] for r in curve]
    assert flagged == sorted(flagged, reverse=True)
    assert all(0.0 <= r[1] <= 1.0 and 0.0 <= r[2] <= 1.0 for r in curve)


def test_empty_control_is_handled():
    """A category with no test/good/ must not crash calibration."""
    rep = control_report([], {"naive": 0.5})
    assert rep["n_control"] == 0 and rep["at"] == {}
    assert control_curve([]) == ()


def test_discovery_separates_good_from_defective(synth_root):
    good = discover_good_items(synth_root, "synth")
    defective = discover_items(synth_root, "synth")

    assert good, "synthetic fixtures should include defect-free images"
    assert all(p.parent.name == "good" for p in good)
    # discover_items must never pick up the unmasked good/ images.
    assert not any(item.image_path.parent.name == "good" for item in defective)

    img = load_image(good[0], size=32)
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_missing_good_dir_returns_empty(tmp_path):
    (tmp_path / "nocat" / "test").mkdir(parents=True)
    assert discover_good_items(tmp_path, "nocat") == []
