"""
PyTorch baseline implementation using cuDNN.
"""
import torch
import torch.nn.functional as F
from typing import Tuple

from .common import ConvConfig, create_test_tensors, get_output_shape


class PyTorchConv2d:
    """PyTorch conv2d baseline using cuDNN."""
    
    def __init__(self, config: ConvConfig):
        self.config = config
        
    def __call__(self, input_tensor: torch.Tensor, filter_tensor: torch.Tensor) -> torch.Tensor:
        """Run conv2d using PyTorch's F.conv2d (cuDNN backend)."""
        return F.conv2d(
            input_tensor,
            filter_tensor,
            bias=None,
            stride=self.config.stride,
            padding=self.config.padding,
            dilation=self.config.dilation,
            groups=1
        )
    
    def get_reference_output(self, input_tensor: torch.Tensor, filter_tensor: torch.Tensor) -> torch.Tensor:
        """Get reference output for correctness checking."""
        return self(input_tensor, filter_tensor)


def run_pytorch_conv2d(config: ConvConfig, dtype: torch.dtype = torch.float16, device: str = "cuda") -> torch.Tensor:
    """Convenience function to run PyTorch conv2d with given config."""
    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False  # For FP16/BF16 precision
    
    input_tensor, filter_tensor = create_test_tensors(config, dtype, device)
    conv = PyTorchConv2d(config)
    
    return conv(input_tensor, filter_tensor)


def test_pytorch_conv2d():
    """Test PyTorch conv2d implementation."""
    from .common import BENCHMARK_SHAPES
    
    for i, config in enumerate(BENCHMARK_SHAPES):
        print(f"Testing shape {i+1}: {config}")
        
        input_tensor, filter_tensor = create_test_tensors(config)
        conv = PyTorchConv2d(config)
        
        output = conv(input_tensor, filter_tensor)
        expected_shape = get_output_shape(config)
        
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"  ✓ Passed\n")


if __name__ == "__main__":
    test_pytorch_conv2d()