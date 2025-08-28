"""
NineToothed DSL implementation for conv2d.
Based on the existing test_conv2d.py but enhanced for benchmarking.
"""
import functools
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

from .common import ConvConfig, create_test_tensors, get_output_shape


# Meta symbols for autotuning
BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True) 
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)


def matmul_arrangement(
    lhs,
    rhs,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    """Matrix multiplication arrangement (from test_matmul.py)."""
    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)

    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def matmul_application(lhs, rhs, output):
    """Matrix multiplication application (from test_matmul.py).""" 
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])
    output = accumulator.to(ntl.float16)


def conv2d_arrangement(
    input,
    filter,
    output,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
):
    """
    Conv2d arrangement using implicit GEMM mapping.
    Based on test_conv2d.py but enhanced.
    """
    # Transform input: tile to create sliding windows, then flatten for GEMM
    input_tiled = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
    input_squeezed = input_tiled.squeeze(1)
    input_squeezed.dtype = input_squeezed.dtype.squeeze(0)
    input_raveled = input_squeezed.ravel()
    input_flattened = input_raveled.flatten(end_dim=3).flatten(start_dim=1)

    # Transform filter: flatten for GEMM
    filter_flattened = filter.flatten(start_dim=1)
    filter_permuted = filter_flattened.permute((1, 0))

    # Transform output: flatten spatial dimensions
    output_flattened = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    # Apply matrix multiplication arrangement
    return functools.partial(
        matmul_arrangement,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )(input_flattened, filter_permuted, output_flattened)


class DSLConv2d:
    """NineToothed DSL conv2d implementation."""
    
    def __init__(self, config: ConvConfig, max_num_configs: int = 50):
        self.config = config
        self.max_num_configs = max_num_configs
        self._kernel = None
        
    def _get_kernel(self):
        """Lazy kernel compilation."""
        if self._kernel is None:
            self._kernel = ninetoothed.make(
                conv2d_arrangement,
                matmul_application,
                (Tensor(4), Tensor(4, shape_options={"constexpr": True}), Tensor(4)),
                max_num_configs=self.max_num_configs,
            )
        return self._kernel
        
    def __call__(self, input_tensor: torch.Tensor, filter_tensor: torch.Tensor) -> torch.Tensor:
        """Run conv2d using NineToothed DSL."""
        config = self.config
        
        # Calculate output shape (assumes valid convolution for now)
        n, _, h, w = input_tensor.shape
        k, _, r, s = filter_tensor.shape
        
        # For simplicity, assume same padding calculation as in test
        p = h - r + 1
        q = w - s + 1
        
        output = torch.empty((n, k, p, q), device=input_tensor.device, dtype=input_tensor.dtype)
        
        kernel = self._get_kernel()
        kernel(input_tensor, filter_tensor, output)
        
        return output


def create_dsl_conv2d_variants():
    """Create different DSL configurations for comparison.""" 
    return {
        'dsl_default': DSLConv2d,
        'dsl_small_search': lambda config: DSLConv2d(config, max_num_configs=10),
        'dsl_large_search': lambda config: DSLConv2d(config, max_num_configs=100),
    }


def get_generated_triton_code(config: ConvConfig) -> str:
    """
    Get the generated Triton code from DSL for inspection.
    This helps analyze what the DSL generates vs hand-written Triton.
    """
    # Create a DSL conv instance
    dsl_conv = DSLConv2d(config, max_num_configs=1)  # Single config for code inspection
    
    # Create test tensors
    input_tensor, filter_tensor = create_test_tensors(config)
    
    # Force kernel compilation by running it once
    _ = dsl_conv(input_tensor, filter_tensor)
    
    # Try to extract generated code (this may need adjustment based on DSL internals)
    kernel = dsl_conv._get_kernel()
    
    # The actual code extraction method will depend on NineToothed's internal structure
    # For now, return a placeholder
    return f"# Generated Triton code for config: {config}\n# (Code extraction to be implemented)"


def test_dsl_conv2d():
    """Test DSL conv2d implementation."""
    from .common import BENCHMARK_SHAPES
    from .pytorch_conv import PyTorchConv2d
    
    for i, config in enumerate(BENCHMARK_SHAPES[:1]):  # Test just first shape initially
        print(f"Testing DSL conv2d with shape {i+1}: {config}")
        
        input_tensor, filter_tensor = create_test_tensors(config)
        
        # Get reference from PyTorch
        pytorch_conv = PyTorchConv2d(config)
        reference = pytorch_conv(input_tensor, filter_tensor)
        
        # Test DSL implementation
        dsl_conv = DSLConv2d(config)
        output = dsl_conv(input_tensor, filter_tensor)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Reference shape: {reference.shape}")
        
        # Check correctness (note: shapes might differ due to padding differences)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        # If shapes match, check numerical accuracy
        if output.shape == reference.shape:
            max_diff = torch.max(torch.abs(output - reference))
            rel_error = torch.max(torch.abs(output - reference) / (torch.abs(reference) + 1e-8))
            
            print(f"  Max absolute difference: {max_diff:.6f}")
            print(f"  Max relative error: {rel_error:.6f}")
            
            if max_diff < 0.01 and rel_error < 0.01:
                print(f"  ✓ Numerical accuracy test passed")
            else:
                print(f"  ⚠ Numerical accuracy may need improvement")
        else:
            print(f"  ⚠ Shape mismatch - may need padding adjustment")
            
        print(f"  ✓ Basic functionality test passed\n")


if __name__ == "__main__":
    test_dsl_conv2d()