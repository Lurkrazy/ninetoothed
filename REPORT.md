# Conv2d Performance Analysis: Why NineToothed DSL Can Match Hand-Written Triton

## Executive Summary

This report analyzes the performance characteristics of three conv2d implementations:
1. **PyTorch (cuDNN)** - Production baseline using NVIDIA's optimized cuDNN library
2. **Hand-written Triton** - Custom implementation using implicit GEMM mapping
3. **NineToothed DSL** - High-level DSL that generates Triton code

**Key Findings:**
- [To be filled after running benchmarks]
- [Performance comparison results]
- [Root cause analysis of performance differences]

## Methodology

### Hardware Configuration
- **Primary Target:** A100 80GB PCIe (sm_80)
- **Secondary Targets:** RTX 4090 (sm_89), RTX 6000 Ada (sm_89)
- **Software:** CUDA 12.x, PyTorch 2.4+, Triton 3.0.0+

### Benchmark Shapes
Following the problem specification, we test these conv2d configurations:

| Shape | N | C | H×W | K | R×S | Stride | Padding | FLOPs |
|-------|---|---|-----|---|-----|--------|---------|-------|
| 1 | 4 | 512 | 14×14 | 512 | 3×3 | 1 | 1 | [calculated] |
| 2 | 32 | 64 | 56×56 | 64 | 3×3 | 1 | 1 | [calculated] |
| 3 | 32 | 128 | 28×28 | 128 | 3×3 | 1 | 1 | [calculated] |

### Implementation Details

#### PyTorch Baseline
- Uses `torch.nn.functional.conv2d` with cuDNN backend
- `torch.backends.cudnn.benchmark=True` for optimal kernel selection
- `allow_tf32=False` for FP16/BF16 precision consistency

#### Hand-written Triton
- Implements implicit GEMM mapping: conv2d → (N×H'×W', K, C×R×S) matrix multiplication
- Configurable block sizes: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
- Tunable parameters: `num_warps`, `num_stages`
- Memory optimization: vectorized loads, shared memory tiling

#### NineToothed DSL
- Uses the existing "arrange-and-apply" paradigm
- Automatically maps conv2d to matrix multiplication via tensor transformations
- Built-in autotuning across multiple configurations
- Generates optimized Triton code internally

### Profiling Protocol

For each implementation and shape:

1. **Performance Measurement:**
   - 10 warmup iterations
   - 100 timed iterations  
   - Report median latency and TFLOP/s
   - GPU synchronization for accurate timing

2. **Profiling with Nsight Compute:**
   ```bash
   ncu --metrics launch,occupancy,sm__throughput,smsp__pipe_tensor,dram__throughput \
       --csv --export profile.csv python benchmark.py
   ```

3. **Code Analysis:**
   - Extract PTX/SASS for instruction-level analysis
   - Compare register usage, shared memory consumption
   - Analyze tensor core utilization patterns

## Results

### Performance Comparison

[Table to be generated from benchmark results]

| Implementation | Shape 1 (ms) | Shape 2 (ms) | Shape 3 (ms) | Avg TFLOP/s |
|----------------|---------------|---------------|---------------|-------------|
| PyTorch        | [TBD] | [TBD] | [TBD] | [TBD] |
| Triton         | [TBD] | [TBD] | [TBD] | [TBD] |
| NineToothed    | [TBD] | [TBD] | [TBD] | [TBD] |

### Speedup Analysis

[Chart showing relative performance vs PyTorch baseline]

### Profiling Results

#### Occupancy and Resource Usage

| Implementation | Occupancy (%) | Registers/Thread | Shared Memory (KB) | Tensor Core Util (%) |
|----------------|---------------|------------------|--------------------|----------------------|
| PyTorch        | [TBD] | [TBD] | [TBD] | [TBD] |
| Triton         | [TBD] | [TBD] | [TBD] | [TBD] |
| NineToothed    | [TBD] | [TBD] | [TBD] | [TBD] |

#### Memory Throughput

| Implementation | DRAM BW (%) | L2 Hit Rate (%) | Shared Load BW (%) |
|----------------|-------------|-----------------|-------------------|
| PyTorch        | [TBD] | [TBD] | [TBD] |
| Triton         | [TBD] | [TBD] | [TBD] |
| NineToothed    | [TBD] | [TBD] | [TBD] |

## Root Cause Analysis

### Hypothesis Testing Results

#### 1. Algorithmic Mapping Parity ✓/✗
**Hypothesis:** DSL systematically uses implicit GEMM while community Triton often uses direct convolution.

**Findings:** [To be filled after analysis]
- [Comparison of algorithmic approaches]
- [Performance impact of mapping choice]

#### 2. Launch Geometry & Tiling Defaults ✓/✗
**Hypothesis:** DSL picks better default block sizes and configuration parameters.

**Findings:** [To be filled after analysis]
- [Analysis of autotuned configurations]
- [Comparison of optimal parameters]

#### 3. Memory Pipeline & Async Copies ✓/✗
**Hypothesis:** DSL codegen emits better memory access patterns.

**Findings:** [To be filled after analysis]
- [PTX/SASS analysis results]
- [Memory coalescing comparison]

#### 4. Register Pressure vs Occupancy ✓/✗
**Hypothesis:** DSL better manages register usage for higher occupancy.

**Findings:** [To be filled after analysis]
- [Register spill analysis]
- [Occupancy vs performance tradeoffs]

#### 5. Autotuning Quality ✓/✗
**Hypothesis:** DSL has superior autotuning search space and heuristics.

**Findings:** [To be filled after analysis]
- [Search space comparison]
- [Convergence analysis]

### Code Comparison

#### Generated DSL Code vs Hand-written Triton

**Memory Access Pattern Differences:**
```python
# Hand-written Triton (excerpt)
[Code snippet showing memory access]

# DSL Generated (excerpt)  
[Code snippet showing generated pattern]
```

**Key Differences:**
- [Difference 1: e.g., vectorization width]
- [Difference 2: e.g., shared memory usage]
- [Difference 3: e.g., loop unrolling]

## Recommendations

Based on the analysis, here are specific recommendations for practitioners:

### For Hand-written Triton Users
1. **Use implicit GEMM mapping** for conv2d when [conditions]
2. **Optimal configurations** for tested shapes:
   - Shape 1: `BLOCK_M=X, BLOCK_N=Y, num_warps=Z`
   - Shape 2: [configuration]
   - Shape 3: [configuration]

### For DSL Adoption
1. **When to use the DSL:**
   - [Scenarios where DSL excels]
   - [Trade-offs to consider]

2. **Performance optimization tips:**
   - [Configuration recommendations]
   - [Best practices]

### Implementation Improvements
1. **Triton kernel enhancements:**
   - [Specific code changes to match DSL performance]
   - [Memory access optimizations]

2. **DSL improvements:**
   - [Areas where DSL could be enhanced]

## Reproducibility

### Running the Benchmarks
```bash
# Basic benchmark
make bench

# Comprehensive benchmark with autotuning  
make bench-full

# Profiling
make profile

# Analysis
make analyze
```

### Environment Setup
```bash
# Install dependencies
make install

# Verify setup
make sysinfo
make test
```

### File Structure
```
├── kernels/
│   ├── common.py          # Shared utilities
│   ├── pytorch_conv.py    # PyTorch baseline
│   ├── triton_conv.py     # Hand-written Triton
│   └── dsl_conv.py        # NineToothed DSL
├── bench/
│   ├── bench_conv.py      # Main benchmark script
│   ├── profile_conv.sh    # Profiling automation
│   └── analyze.py         # Results analysis
├── results/               # Benchmark outputs
└── Makefile              # Automation
```

## Limitations and Future Work

### Current Limitations
1. [Limitation 1: e.g., limited shape coverage]
2. [Limitation 2: e.g., single precision testing]
3. [Limitation 3: e.g., hardware dependency]

### Future Investigations
1. **Extended shape coverage:** Include stride>1, dilation, grouped convolutions
2. **End-to-end analysis:** Multi-layer conv blocks with fusion
3. **Hardware portability:** Test on V100, H100, consumer GPUs
4. **Precision variants:** BF16, INT8, FP8 analysis

## Conclusion

[To be written after completing benchmarks and analysis]

### Key Takeaways
1. [Main finding about DSL vs Triton performance]
2. [Insights about optimization strategies]
3. [Recommendations for practitioners]

### Broader Implications
- [Impact on DSL design decisions]
- [Lessons for kernel optimization]
- [Future research directions]

---

**Generated:** [Date]  
**Hardware:** [GPU details]  
**Software:** PyTorch [version], Triton [version], NineToothed [version]