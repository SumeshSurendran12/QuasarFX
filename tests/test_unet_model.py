import importlib.util
from pathlib import Path


def test_unet_model_output_shape():
    module_path = Path(__file__).resolve().parents[1] / "U-Net.py"
    spec = importlib.util.spec_from_file_location("u_net", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.unet_model()
    assert model.output_shape == (None, 256, 256, 1)
