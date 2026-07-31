import pytest

from conformal_seg.data import make_synthetic_dir


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mvtec_synth")
    make_synthetic_dir(root, category="synth", n=12, size=96, seed=5)
    return root
