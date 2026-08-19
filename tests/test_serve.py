"""The serving layer, end to end, on a synthetic registry.

No dataset, no trained weights, no network: a random-weight model is exported at
64 px and paired with hand-written calibration reports, which is enough to
exercise every path the container takes. Same CI contract as the rest of the suite.

Two categories are registered off the same network. A random-weight model puts
every pixel above 0.5, so `synth` always flags the whole frame; `synth_strict`
carries a threshold no probability can reach and therefore always flags nothing.
That pins both decision branches without depending on what an untrained model
happens to output.
"""

import base64
import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from conformal_seg import registry
from conformal_seg.model import DefectSeg
from conformal_seg.onnx_export import export


def calibration(threshold: float) -> dict:
    return {
        "calibration": {
            "threshold": threshold,
            "alpha": 0.10,
            "n": 11,
            "risk_curve": [[0.0, 0.09], [0.5, 0.10], [1.0, 0.95]],
        },
        "held_out": {
            "alpha": 0.10,
            "threshold": threshold,
            "held_out_fnr_mean": 0.055,
            "held_out_fnr_max": 0.21,
            "mean_predicted_mask_fraction": 0.2,
            "n_test": 12,
        },
        "control": {
            "n_control": 20,
            "min_area_frac": 1e-3,
            "at": {
                "naive": {"threshold": 0.5, "mean_flagged_pixel_fraction": 0.0,
                          "max_flagged_pixel_fraction": 0.0,
                          "image_false_alarm_rate": 0.0},
                "conformal": {"threshold": threshold,
                              "mean_flagged_pixel_fraction": 0.0004,
                              "max_flagged_pixel_fraction": 0.003,
                              "image_false_alarm_rate": 0.05},
            },
        },
    }


@pytest.fixture(scope="module")
def monkeymodule():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeymodule):
    root = tmp_path_factory.mktemp("serving")
    models_root = root / "models"

    for name, threshold in (("synth", 0.5), ("synth_strict", 1.01)):
        run_dir = root / "runs" / name
        run_dir.mkdir(parents=True)
        ckpt = run_dir / "best.pt"
        torch.save({"state_dict": DefectSeg(pretrained=False).state_dict(),
                    "category": name, "size": 64, "seed": 0, "val_iou": 0.0}, ckpt)
        export(ckpt, run_dir / "model.onnx", size=64, check=False)
        (run_dir / "calibration.json").write_text(json.dumps(calibration(threshold)))
        registry.build(run_dir, models_root / name, name)

    from conformal_seg.serve import app as app_module

    monkeymodule.setattr(app_module, "MODEL_ROOT", models_root)
    app_module.bundle.cache_clear()
    yield TestClient(app_module.app)
    app_module.bundle.cache_clear()


def _png_bytes(value: int = 120, size: int = 96) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _image():
    return {"image": ("part.png", _png_bytes(), "image/png")}


def _model(client, category: str) -> dict:
    return next(m for m in client.get("/models").json() if m["category"] == category)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_registry_reads_size_from_the_graph(client):
    """input_size must come from the ONNX, not a flag: serving at a resolution the
    threshold was not calibrated for is the one silent failure that matters."""
    m = _model(client, "synth")
    assert m["input_size"] == 64
    assert m["source_run"] == "synth"
    assert m["threshold"] == 0.5 and m["alpha"] == 0.10


def test_models_surfaces_the_false_alarm_rate(client):
    """The number alpha says nothing about has to travel with alpha."""
    m = _model(client, "synth")
    assert m["false_alarm_rate"] == 0.05
    assert m["n_control"] == 20
    assert m["min_area_frac"] == pytest.approx(1e-3)


def test_predict_returns_a_decision_and_the_guarantee(client):
    r = client.post("/predict?category=synth", files=_image())
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] in {"pass", "escalate"}
    assert 0.0 <= d["flagged_fraction"] <= 1.0
    assert d["input_size"] == 64
    assert d["mask_png"] is None
    g = d["guarantee"]
    assert g["alpha"] == 0.10 and g["threshold"] == 0.5
    assert g["false_alarm_rate"] == 0.05          # reported on every response
    assert g["held_out_fnr"] <= g["alpha"]        # the guarantee actually held


def test_both_decision_branches_are_reachable(client):
    """A threshold every pixel clears escalates; one nothing clears passes."""
    flagged = client.post("/predict?category=synth", files=_image()).json()
    clean = client.post("/predict?category=synth_strict", files=_image()).json()

    assert flagged["flagged_fraction"] == 1.0 and flagged["decision"] == "escalate"
    assert clean["flagged_fraction"] == 0.0 and clean["decision"] == "pass"


def test_escalation_trigger_is_inclusive_and_does_not_move_the_mask(client):
    """min_area_frac is the decision rule, not a mask parameter."""
    default = client.post("/predict?category=synth", files=_image()).json()
    f = default["flagged_fraction"]

    at_zero = client.post("/predict?category=synth&min_area_frac=0.0", files=_image()).json()
    at_f = client.post(f"/predict?category=synth&min_area_frac={f}", files=_image()).json()

    assert at_zero["decision"] == "escalate"
    assert at_f["decision"] == "escalate"          # the comparison is >=, not >
    # Changing the trigger must not change the pixels it is applied to.
    assert at_zero["flagged_fraction"] == at_f["flagged_fraction"] == f

    # And on the strict model, a trigger above what it flags passes.
    strict = client.post("/predict?category=synth_strict&min_area_frac=0.5",
                         files=_image()).json()
    assert strict["decision"] == "pass"


def test_return_mask_is_a_decodable_png_at_model_resolution(client):
    from PIL import Image

    d = client.post("/predict?category=synth&return_mask=true", files=_image()).json()
    img = Image.open(io.BytesIO(base64.b64decode(d["mask_png"])))
    assert img.size == (64, 64)
    assert set(np.unique(np.asarray(img))) <= {0, 255}


def test_unknown_category_is_503(client):
    assert client.post("/predict?category=nope", files=_image()).status_code == 503


def test_undecodable_upload_is_400_not_500(client):
    r = client.post("/predict?category=synth",
                    files={"image": ("notes.txt", b"this is not an image", "text/plain")})
    assert r.status_code == 400


def test_empty_upload_is_400(client):
    r = client.post("/predict?category=synth",
                    files={"image": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_oversized_upload_is_413(client, monkeymodule):
    from conformal_seg.serve import app as app_module

    monkeymodule.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    try:
        r = client.post("/predict?category=synth", files=_image())
        assert r.status_code == 413
    finally:
        monkeymodule.setattr(app_module, "MAX_UPLOAD_BYTES", 12 * 1024 * 1024)


def test_empty_registry_is_503(tmp_path, monkeymodule):
    from conformal_seg.serve import app as app_module

    original = app_module.MODEL_ROOT
    monkeymodule.setattr(app_module, "MODEL_ROOT", tmp_path)
    app_module.bundle.cache_clear()
    try:
        assert TestClient(app_module.app).get("/models").status_code == 503
    finally:
        monkeymodule.setattr(app_module, "MODEL_ROOT", original)
        app_module.bundle.cache_clear()


def test_registry_refuses_an_incomplete_run(tmp_path):
    """A run without a calibration report has no threshold, so it cannot be served."""
    run = tmp_path / "runs" / "half"
    run.mkdir(parents=True)
    (run / "model.onnx").write_bytes(b"not really onnx")
    with pytest.raises(FileNotFoundError, match="calibration"):
        registry.build(run, tmp_path / "models" / "half", "half")


def test_models_does_not_load_any_network(client, monkeymodule):
    """Listing categories is a metadata request. Building Bundles here loaded one
    onnxruntime session per category, which blew the Lambda cold-start timeout
    once five were registered."""
    from conformal_seg.serve import app as app_module

    calls = []
    real = app_module.Bundle

    class Counting(real):
        def __init__(self, category):
            calls.append(category)
            super().__init__(category)

    monkeymodule.setattr(app_module, "Bundle", Counting)
    app_module.bundle.cache_clear()
    try:
        assert client.get("/models").status_code == 200
        assert calls == [], f"/models constructed Bundles: {calls}"
    finally:
        monkeymodule.setattr(app_module, "Bundle", real)
        app_module.bundle.cache_clear()
