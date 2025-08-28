"""
Main benchmarking script for conv2d implementations.
Compares PyTorch (cuDNN), Triton, and NineToothed DSL.
"""
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any
import torch

from kernels.common import (
    ConvConfig, BENCHMARK_SHAPES, 
    create_test_tensors, conv2d_flops, time_kernel, get_tflops,
    warmup_gpu, TimingResult
)
from kernels.pytorch_conv import PyTorchConv2d
from kernels.triton_conv import TritonConv2d, get_autotuning_configs
from kernels.dsl_conv import DSLConv2d, create_dsl_conv2d_variants


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self):
        self.results = []
        
    def add_result(self, 
                   impl_name: str,
                   config: ConvConfig, 
                   timing: TimingResult,
                   flops: int,
                   config_params: Dict = None):
        """Add a benchmark result."""
        result = {
            'implementation': impl_name,
            'config': config._asdict(),
            'median_time_ms': timing.median_time_ms,
            'min_time_ms': timing.min_time_ms,
            'max_time_ms': timing.max_time_ms,
            'std_time_ms': timing.std_time_ms,
            'tflops': get_tflops(flops, timing.median_time_ms),
            'flops': flops,
            'config_params': config_params or {},
            'timestamp': time.time(),
        }
        self.results.append(result)
        
    def save_json(self, filepath: Path):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def save_csv(self, filepath: Path):
        """Save results to CSV file."""
        if not self.results:
            return
            
        fieldnames = [
            'implementation', 'N', 'C', 'H', 'W', 'K', 'R', 'S',
            'median_time_ms', 'min_time_ms', 'max_time_ms', 'std_time_ms',
            'tflops', 'flops', 'timestamp'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'implementation': result['implementation'],
                    **result['config'],
                    'median_time_ms': result['median_time_ms'],
                    'min_time_ms': result['min_time_ms'], 
                    'max_time_ms': result['max_time_ms'],
                    'std_time_ms': result['std_time_ms'],
                    'tflops': result['tflops'],
                    'flops': result['flops'],
                    'timestamp': result['timestamp'],
                }
                writer.writerow(row)


def benchmark_pytorch(config: ConvConfig, dtype: torch.dtype = torch.float16) -> TimingResult:
    """Benchmark PyTorch conv2d."""
    print(f"  Benchmarking PyTorch...")
    
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    
    input_tensor, filter_tensor = create_test_tensors(config, dtype)
    conv = PyTorchConv2d(config)
    
    # Warmup and timing
    def run_conv():
        return conv(input_tensor, filter_tensor)
    
    timing = time_kernel(run_conv)
    flops = conv2d_flops(config)
    
    return timing


def benchmark_triton(config: ConvConfig, 
                    dtype: torch.dtype = torch.float16,
                    autotune: bool = False) -> List[TimingResult]:
    """Benchmark Triton conv2d with different configurations.""" 
    print(f"  Benchmarking Triton...")
    
    input_tensor, filter_tensor = create_test_tensors(config, dtype)
    flops = conv2d_flops(config)
    results = []
    
    if autotune:
        configs = get_autotuning_configs()[:10]  # Limit for initial testing
    else:
        # Default configuration
        configs = [{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 4}]
    
    for i, cfg in enumerate(configs):
        try:
            conv = TritonConv2d(config, 
                              block_m=cfg['BLOCK_M'], 
                              block_n=cfg['BLOCK_N'],
                              block_k=cfg['BLOCK_K'],
                              num_warps=cfg['num_warps'],
                              num_stages=cfg['num_stages'])
            
            def run_conv():
                return conv(input_tensor, filter_tensor)
            
            timing = time_kernel(run_conv, warmup_iters=5, timing_iters=50)
            results.append((timing, cfg))
            
            if i == 0 or i % 5 == 0:
                print(f"    Config {i+1}/{len(configs)}: {timing.median_time_ms:.3f}ms")
                
        except Exception as e:
            print(f"    Config {i+1} failed: {e}")
            continue
    
    return results


def benchmark_dsl(config: ConvConfig, dtype: torch.dtype = torch.float16) -> List[TimingResult]:
    """Benchmark NineToothed DSL conv2d."""
    print(f"  Benchmarking NineToothed DSL...")
    
    input_tensor, filter_tensor = create_test_tensors(config, dtype)
    flops = conv2d_flops(config)
    results = []
    
    variants = create_dsl_conv2d_variants()
    
    for variant_name, variant_class in variants.items():
        try:
            conv = variant_class(config)
            
            def run_conv():
                return conv(input_tensor, filter_tensor)
            
            timing = time_kernel(run_conv, warmup_iters=5, timing_iters=30)
            results.append((timing, {'variant': variant_name}))
            
            print(f"    {variant_name}: {timing.median_time_ms:.3f}ms")
            
        except Exception as e:
            print(f"    {variant_name} failed: {e}")
            continue
    
    return results


def run_benchmarks(shapes: List[ConvConfig] = None, 
                  dtype: torch.dtype = torch.float16,
                  autotune_triton: bool = False) -> BenchmarkResult:
    """Run comprehensive benchmarks."""
    if shapes is None:
        shapes = BENCHMARK_SHAPES
    
    results = BenchmarkResult()
    
    print(f"Running conv2d benchmarks with dtype={dtype}")
    print(f"Shapes to test: {len(shapes)}")
    print(f"Triton autotuning: {autotune_triton}")
    print("="*60)
    
    # Warmup GPU
    print("Warming up GPU...")
    warmup_gpu()
    
    for i, config in enumerate(shapes):
        print(f"\nShape {i+1}/{len(shapes)}: {config}")
        flops = conv2d_flops(config)
        print(f"  FLOPs: {flops:,}")
        
        try:
            # Benchmark PyTorch
            pytorch_timing = benchmark_pytorch(config, dtype)
            results.add_result('pytorch', config, pytorch_timing, flops)
            print(f"    PyTorch: {pytorch_timing.median_time_ms:.3f}ms, {get_tflops(flops, pytorch_timing.median_time_ms):.2f} TFLOP/s")
            
        except Exception as e:
            print(f"    PyTorch failed: {e}")
        
        try:
            # Benchmark Triton
            triton_results = benchmark_triton(config, dtype, autotune_triton)
            if triton_results:
                # Use best result
                best_timing, best_config = min(triton_results, key=lambda x: x[0].median_time_ms)
                results.add_result('triton', config, best_timing, flops, best_config)
                print(f"    Triton best: {best_timing.median_time_ms:.3f}ms, {get_tflops(flops, best_timing.median_time_ms):.2f} TFLOP/s")
                
        except Exception as e:
            print(f"    Triton failed: {e}")
        
        try:
            # Benchmark DSL
            dsl_results = benchmark_dsl(config, dtype)
            if dsl_results:
                # Use best result  
                best_timing, best_config = min(dsl_results, key=lambda x: x[0].median_time_ms)
                results.add_result('dsl', config, best_timing, flops, best_config)
                print(f"    DSL best: {best_timing.median_time_ms:.3f}ms, {get_tflops(flops, best_timing.median_time_ms):.2f} TFLOP/s")
                
        except Exception as e:
            print(f"    DSL failed: {e}")
    
    return results


def main():
    """Main benchmark entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Conv2d Performance Benchmarking')
    parser.add_argument('--autotune-triton', action='store_true', 
                       help='Enable Triton autotuning (slower but more thorough)')
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                       help='Output directory for results')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16',
                       help='Data type for benchmarking')
    
    args = parser.parse_args()
    
    # Setup output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Convert dtype
    dtype = torch.float16 if args.dtype == 'float16' else torch.bfloat16
    
    # Run benchmarks
    results = run_benchmarks(
        shapes=BENCHMARK_SHAPES,
        dtype=dtype,
        autotune_triton=args.autotune_triton
    )
    
    # Save results
    timestamp = int(time.time())
    results.save_json(args.output_dir / f'conv2d_bench_{timestamp}.json')
    results.save_csv(args.output_dir / f'conv2d_bench_{timestamp}.csv')
    
    print(f"\nResults saved to {args.output_dir}")
    print("Benchmark complete!")


if __name__ == "__main__":
    main()