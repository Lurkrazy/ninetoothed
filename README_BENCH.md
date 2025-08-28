# Conv2d Performance Benchmarking Infrastructure

This infrastructure provides comprehensive performance analysis for conv2d implementations, comparing:

1. **PyTorch (cuDNN)** - Production baseline
2. **Hand-written Triton** - Custom implicit GEMM implementation  
3. **NineToothed DSL** - High-level domain-specific language

## Quick Start

```bash
# Setup environment
make install
make sysinfo

# Run basic tests (CPU-only for validation)
make test  

# Run benchmarks (requires CUDA GPU)
make bench

# Run comprehensive benchmarks with autotuning
make bench-full

# Profile with Nsight Compute
make profile

# Analyze results
make analyze
```

## Requirements

- **GPU:** A100, RTX 4090, or RTX 6000 Ada (CUDA sm_80/sm_89)
- **Software:** CUDA 12.x, PyTorch 2.4+, Triton 3.0.0+
- **Tools:** Nsight Compute CLI (`ncu`) for profiling

## Architecture

### Directory Structure

```
├── kernels/              # Implementation modules
│   ├── common.py         # Shared utilities & configs
│   ├── pytorch_conv.py   # PyTorch baseline
│   ├── triton_conv.py    # Hand-written Triton
│   └── dsl_conv.py       # NineToothed DSL
├── bench/                # Benchmarking & analysis
│   ├── bench_conv.py     # Main benchmark harness
│   ├── profile_conv.sh   # Profiling automation
│   └── analyze.py        # Results analysis & visualization
├── results/              # Generated outputs
└── REPORT.md             # Comprehensive analysis report
```

### Benchmark Shapes

The infrastructure tests these conv2d configurations (from problem specification):

| Config | N | C | H×W | K | R×S | FLOPs (G) |
|--------|---|---|-----|---|-----|-----------|
| 1 | 4 | 512 | 14×14 | 512 | 3×3 | 3.70 |
| 2 | 32 | 64 | 56×56 | 64 | 3×3 | 7.40 |
| 3 | 32 | 128 | 28×28 | 128 | 3×3 | 7.40 |

## Implementation Details

### PyTorch Baseline
- Uses `torch.nn.functional.conv2d` with cuDNN
- Optimized with `cudnn.benchmark=True` and `allow_tf32=False`

### Hand-written Triton
- Implements implicit GEMM mapping: conv2d → matrix multiplication
- Configurable tiling: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
- Tunable compiler settings: `num_warps`, `num_stages`

### NineToothed DSL  
- Uses arrange-and-apply paradigm from existing test suite
- Automatic tensor transformations for conv→GEMM mapping
- Built-in autotuning across configuration space

## Profiling & Analysis

### Performance Metrics
- **Latency:** Median time over 100 iterations
- **Throughput:** TFLOP/s calculation
- **Speedup:** Relative performance vs PyTorch baseline

### Profiling Data (via Nsight Compute)
- Occupancy and resource utilization
- Memory throughput (DRAM, L2, shared memory)
- Tensor core utilization
- Instruction mix and stall analysis

### Analysis Outputs
- Performance comparison charts
- Speedup analysis plots
- Roofline analysis
- Code-level differences (PTX/SASS)

## Usage Examples

### Basic Benchmarking
```bash
# Test specific implementation
python -m kernels.pytorch_conv
python -m kernels.triton_conv  
python -m kernels.dsl_conv

# Run full benchmark suite
python -m bench.bench_conv --dtype float16

# Benchmark with autotuning (slower but comprehensive)
python -m bench.bench_conv --autotune-triton
```

### Profiling
```bash
# Profile all implementations for shape 0
bash bench/profile_conv.sh all 0

# Profile specific implementation
bash bench/profile_conv.sh triton 1
```

### Analysis
```bash
# Analyze latest results
python -m bench.analyze

# Analyze specific file
python -m bench.analyze --results-file results/conv2d_bench_1234567890.json
```

## Expected Outcomes

The analysis aims to determine:

1. **Performance Gaps:** Where DSL matches/exceeds hand-written Triton
2. **Root Causes:** Memory patterns, occupancy, autotuning quality
3. **Optimization Insights:** Specific improvements for Triton kernels
4. **Best Practices:** When to use each implementation approach

## Limitations

- **Hardware Dependency:** Requires modern NVIDIA GPU with tensor cores
- **Scope:** Focuses on 3×3 conv2d with stride=1, padding=1
- **Precision:** Primary testing with FP16 (BF16 optional)

## Extending the Analysis

To add new test cases:

1. **Shapes:** Modify `BENCHMARK_SHAPES` in `kernels/common.py`
2. **Implementations:** Add new modules following existing patterns
3. **Metrics:** Extend `analyze.py` for additional profiling data
4. **Configurations:** Update autotuning parameters in respective modules

## Troubleshooting

### Common Issues
- **CUDA not available:** Verify GPU drivers and CUDA installation
- **Triton compilation errors:** Check kernel parameters and block sizes
- **Memory errors:** Reduce batch size or use smaller shapes
- **Profiling failures:** Ensure `ncu` is in PATH and has proper permissions

### Debug Mode
```bash
# Verbose output
CUDA_LAUNCH_BLOCKING=1 python -m bench.bench_conv

# Triton debug
TRITON_INTERPRET=1 python -m kernels.triton_conv
```

---

For detailed results and analysis, see [REPORT.md](REPORT.md) after running benchmarks.