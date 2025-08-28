"""
Common utilities and configuration for conv2d benchmarking.
"""
import torch
from typing import List, Tuple, NamedTuple


class ConvConfig(NamedTuple):
    """Configuration for a convolution operation."""
    N: int  # Batch size
    C: int  # Input channels
    H: int  # Input height
    W: int  # Input width
    K: int  # Output channels
    R: int  # Filter height
    S: int  # Filter width
    stride: int = 1
    padding: int = 1
    dilation: int = 1


# Benchmark shapes as specified in the problem statement
BENCHMARK_SHAPES = [
    ConvConfig(N=4, C=512, H=14, W=14, K=512, R=3, S=3, stride=1, padding=1),
    ConvConfig(N=32, C=64, H=56, W=56, K=64, R=3, S=3, stride=1, padding=1),
    ConvConfig(N=32, C=128, H=28, W=28, K=128, R=3, S=3, stride=1, padding=1),
]

# GEMM parity check shapes  
GEMM_SHAPES = [
    (4096, 4096, 4096),
    (2048, 2048, 2048),  # For batched case with B=4
]


def get_output_shape(config: ConvConfig) -> Tuple[int, int, int, int]:
    """Calculate output shape for given conv config."""
    pad_h = 2 * config.padding
    pad_w = 2 * config.padding
    
    out_h = (config.H + pad_h - config.dilation * (config.R - 1) - 1) // config.stride + 1
    out_w = (config.W + pad_w - config.dilation * (config.S - 1) - 1) // config.stride + 1
    
    return (config.N, config.K, out_h, out_w)


def create_test_tensors(config: ConvConfig, dtype: torch.dtype = torch.float16, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Create test input and filter tensors for benchmarking."""
    torch.manual_seed(42)  # For reproducibility
    
    input_tensor = torch.randn(config.N, config.C, config.H, config.W, dtype=dtype, device=device)
    filter_tensor = torch.randn(config.K, config.C, config.R, config.S, dtype=dtype, device=device)
    
    return input_tensor, filter_tensor


def warmup_gpu(device: str = "cuda", iterations: int = 10):
    """Warmup GPU for consistent benchmarking."""
    x = torch.randn(1024, 1024, device=device)
    for _ in range(iterations):
        y = torch.matmul(x, x)
        torch.cuda.synchronize()


def get_tflops(flops: float, time_ms: float) -> float:
    """Calculate TFLOP/s from FLOPs and time in milliseconds."""
    return (flops / time_ms) / 1e9  # Convert to TFLOP/s


def conv2d_flops(config: ConvConfig) -> int:
    """Calculate FLOPs for conv2d operation."""
    out_h, out_w = get_output_shape(config)[2:]
    return config.N * config.K * out_h * out_w * config.C * config.R * config.S * 2  # 2 for multiply-add


class TimingResult(NamedTuple):
    """Result of timing a kernel."""
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    tflops: float


def time_kernel(kernel_func, *args, warmup_iters: int = 10, timing_iters: int = 100) -> TimingResult:
    """Time a kernel function with proper warmup and multiple iterations."""
    # Warmup
    for _ in range(warmup_iters):
        kernel_func(*args)
        torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(timing_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        kernel_func(*args)
        end_event.record()
        
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    times = torch.tensor(times)
    return TimingResult(
        median_time_ms=float(torch.median(times)),
        min_time_ms=float(torch.min(times)),
        max_time_ms=float(torch.max(times)),
        std_time_ms=float(torch.std(times)),
        tflops=0.0  # Will be calculated by caller
    )