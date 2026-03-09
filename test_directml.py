import pytest

def test_directml():
    torch = pytest.importorskip(
        "torch",
        reason="PyTorch is not installed in this environment.",
    )
    torch_directml = pytest.importorskip(
        "torch_directml",
        reason="torch_directml is not installed in this environment.",
    )

    try:
        # Initialize DirectML
        dml = torch_directml.device()
        print(f"DirectML device: {dml}")
        
        # Create a test tensor
        x = torch.randn(1000, 1000).to(dml)
        y = torch.randn(1000, 1000).to(dml)
        
        # Perform a matrix multiplication
        z = torch.matmul(x, y)
        
        # Move result back to CPU and verify
        z_cpu = z.cpu()
        print("Matrix multiplication successful!")
        print(f"Result shape: {z_cpu.shape}")
        print(f"Device used: {z.device}")
        assert z_cpu.shape == (1000, 1000)
    except Exception as e:
        pytest.fail(f"Error testing DirectML: {str(e)}")

if __name__ == "__main__":
    test_directml() 
