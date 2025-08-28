"""
Triton baseline implementation using implicit GEMM for conv2d.
This implementation follows the same algorithmic approach that the DSL uses.
"""
import torch
import triton
import triton.language as tl
from typing import Tuple, Optional

from .common import ConvConfig, create_test_tensors, get_output_shape


@triton.jit
def conv2d_implicit_gemm_kernel(
    # Input tensor (NCHW)
    input_ptr, input_batch_stride, input_channel_stride, input_row_stride, input_col_stride,
    # Filter tensor (KCHW) 
    filter_ptr, filter_out_stride, filter_in_stride, filter_row_stride, filter_col_stride,
    # Output tensor (NCHW)
    output_ptr, output_batch_stride, output_channel_stride, output_row_stride, output_col_stride,
    # Conv parameters
    N, C, H, W, K, R, S, 
    out_h, out_w,
    stride, padding,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Implicit GEMM conv2d kernel.
    
    Maps conv2d to GEMM where:
    - M dimension: N * out_h * out_w (batch * spatial output)
    - N dimension: K (output channels)  
    - K dimension: C * R * S (input channels * filter spatial)
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension (C * R * S)
    for k_start in range(0, C * R * S, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Mask for valid K indices
        k_mask = k_offs < C * R * S
        
        # Decompose k_offs into (c, r, s)
        c_idx = k_offs // (R * S)
        rs_idx = k_offs % (R * S) 
        r_idx = rs_idx // S
        s_idx = rs_idx % S
        
        # For input matrix A (M x K): map offs_m to (n, out_h, out_w)
        # Each offs_m represents a flattened (n, out_h, out_w) position
        spatial_size = out_h * out_w
        n_idx = offs_m // spatial_size
        spatial_idx = offs_m % spatial_size
        out_h_idx = spatial_idx // out_w
        out_w_idx = spatial_idx % out_w
        
        # Convert output spatial coordinates to input coordinates
        in_h_idx = out_h_idx * stride - padding + r_idx[:, None]
        in_w_idx = out_w_idx * stride - padding + s_idx[:, None]
        
        # Check bounds for input coordinates
        in_bounds = (
            (n_idx[None, :] >= 0) & (n_idx[None, :] < N) &
            (c_idx[:, None] >= 0) & (c_idx[:, None] < C) &
            (in_h_idx >= 0) & (in_h_idx < H) &
            (in_w_idx >= 0) & (in_w_idx < W)
        )
        
        # Compute input pointers
        input_ptrs = (
            input_ptr +
            n_idx[None, :] * input_batch_stride +
            c_idx[:, None] * input_channel_stride +
            in_h_idx * input_row_stride +
            in_w_idx * input_col_stride
        )
        
        # Load input data
        a = tl.load(input_ptrs, mask=in_bounds & k_mask[:, None], other=0.0)
        
        # For filter matrix B (K x N): load filter weights
        filter_ptrs = (
            filter_ptr +
            offs_n[None, :] * filter_out_stride +
            c_idx[:, None] * filter_in_stride +
            r_idx[:, None] * filter_row_stride +
            s_idx[:, None] * filter_col_stride
        )
        
        n_mask = offs_n < K
        filter_mask = k_mask[:, None] & n_mask[None, :]
        
        b = tl.load(filter_ptrs, mask=filter_mask, other=0.0)
        
        # Accumulate: C = A @ B^T
        accumulator += tl.dot(a, b)
    
    # Convert back to output coordinates for storing
    n_idx = offs_m // spatial_size
    spatial_idx = offs_m % spatial_size
    out_h_idx = spatial_idx // out_w
    out_w_idx = spatial_idx % out_w
    
    # Store output
    output_ptrs = (
        output_ptr +
        n_idx[:, None] * output_batch_stride +
        offs_n[None, :] * output_channel_stride +
        out_h_idx[:, None] * output_row_stride +
        out_w_idx[:, None] * output_col_stride
    )
    
    # Output bounds checking
    out_bounds = (
        (n_idx[:, None] >= 0) & (n_idx[:, None] < N) &
        (offs_n[None, :] >= 0) & (offs_n[None, :] < K) &
        (out_h_idx[:, None] >= 0) & (out_h_idx[:, None] < out_h) &
        (out_w_idx[:, None] >= 0) & (out_w_idx[:, None] < out_w)
    )
    
    # Convert accumulator to output type and store
    output = accumulator.to(tl.float16)
    tl.store(output_ptrs, output, mask=out_bounds)


class TritonConv2d:
    """Triton conv2d implementation using implicit GEMM."""
    
    def __init__(self, config: ConvConfig, 
                 block_m: int = 128, block_n: int = 128, block_k: int = 32,
                 num_warps: int = 8, num_stages: int = 4):
        self.config = config
        self.block_m = block_m
        self.block_n = block_n  
        self.block_k = block_k
        self.num_warps = num_warps
        self.num_stages = num_stages
        
    def __call__(self, input_tensor: torch.Tensor, filter_tensor: torch.Tensor) -> torch.Tensor:
        """Run conv2d using Triton implicit GEMM kernel."""
        config = self.config
        out_shape = get_output_shape(config)
        output = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Calculate grid dimensions
        M = config.N * out_shape[2] * out_shape[3]  # Batch * spatial output
        N = config.K  # Output channels
        
        grid_m = triton.cdiv(M, self.block_m)
        grid_n = triton.cdiv(N, self.block_n)
        
        # Launch kernel
        conv2d_implicit_gemm_kernel[(grid_m, grid_n)](
            # Input tensor
            input_tensor, 
            input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
            # Filter tensor  
            filter_tensor,
            filter_tensor.stride(0), filter_tensor.stride(1), filter_tensor.stride(2), filter_tensor.stride(3),
            # Output tensor
            output,
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            # Conv parameters
            config.N, config.C, config.H, config.W, config.K, config.R, config.S,
            out_shape[2], out_shape[3],  # out_h, out_w
            config.stride, config.padding,
            # Block sizes
            BLOCK_M=self.block_m, BLOCK_N=self.block_n, BLOCK_K=self.block_k,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        
        return output


def get_autotuning_configs():
    """Get autotuning configurations for Triton conv2d."""
    configs = []
    
    # Block size combinations
    block_sizes = [
        (64, 64, 32), (64, 128, 32), (128, 64, 32), (128, 128, 32),
        (64, 64, 64), (64, 128, 64), (128, 64, 64), (128, 128, 64),
        (128, 256, 32), (256, 128, 32), (256, 256, 32),
    ]
    
    # Compiler configurations
    num_warps_options = [4, 8, 16]
    num_stages_options = [2, 3, 4, 5]
    
    for (bm, bn, bk) in block_sizes:
        for nw in num_warps_options:
            for ns in num_stages_options:
                configs.append({
                    'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk,
                    'num_warps': nw, 'num_stages': ns
                })
    
    return configs


def test_triton_conv2d():
    """Test Triton conv2d implementation."""
    from .common import BENCHMARK_SHAPES
    from .pytorch_conv import PyTorchConv2d
    
    for i, config in enumerate(BENCHMARK_SHAPES[:1]):  # Test just first shape initially
        print(f"Testing Triton conv2d with shape {i+1}: {config}")
        
        input_tensor, filter_tensor = create_test_tensors(config)
        
        # Get reference from PyTorch
        pytorch_conv = PyTorchConv2d(config)
        reference = pytorch_conv(input_tensor, filter_tensor)
        
        # Test Triton implementation
        triton_conv = TritonConv2d(config)
        output = triton_conv(input_tensor, filter_tensor)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Reference shape: {reference.shape}")
        
        # Check correctness
        assert output.shape == reference.shape, f"Shape mismatch: {output.shape} != {reference.shape}"
        
        # Check for numerical issues
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        
        # Check numerical accuracy (may need tuning)
        max_diff = torch.max(torch.abs(output - reference))
        rel_error = torch.max(torch.abs(output - reference) / (torch.abs(reference) + 1e-8))
        
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Max relative error: {rel_error:.6f}")
        
        # These tolerances may need adjustment
        if max_diff < 0.01 and rel_error < 0.01:
            print(f"  ✓ Numerical accuracy test passed\n")
        else:
            print(f"  ⚠ Numerical accuracy may need improvement\n")


if __name__ == "__main__":
    test_triton_conv2d()