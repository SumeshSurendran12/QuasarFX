import importlib.util
from pathlib import Path

import pytest


def test_unet_model_output_shape():
    module_path = Path(__file__).resolve().parents[1] / "U-Net.py"
    if not module_path.exists():
        pytest.skip(f"Legacy U-Net module is not present in this repository: {module_path.name}")

    pytest.importorskip("tensorflow", reason="TensorFlow is required for the legacy U-Net smoke test.")

    spec = importlib.util.spec_from_file_location("u_net", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.unet_model()
    assert model.output_shape == (None, 256, 256, 1)
