# Makefile for Conv2d Performance Analysis Project

.PHONY: help install bench profile analyze clean test

# Default target
help:
	@echo "Conv2d Performance Analysis"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run basic tests"
	@echo "  bench       - Run performance benchmarks"
	@echo "  bench-full  - Run comprehensive benchmarks with autotuning"
	@echo "  profile     - Run profiling with Nsight Compute"
	@echo "  analyze     - Analyze latest benchmark results"
	@echo "  clean       - Clean up generated files"
	@echo ""
	@echo "Environment variables:"
	@echo "  DTYPE       - Data type (float16, bfloat16) [default: float16]"
	@echo "  SHAPE_IDX   - Shape index for profiling [default: 0]"
	@echo "  IMPL        - Implementation for profiling (pytorch, triton, dsl, all) [default: all]"

# Install dependencies
install:
	pip install -e .
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision
	pip install matplotlib seaborn pandas tabulate

# Test basic functionality
test:
	@echo "Testing PyTorch implementation..."
	python -m kernels.pytorch_conv
	@echo ""
	@echo "Testing Triton implementation..."
	python -m kernels.triton_conv || echo "Triton test failed - check CUDA/Triton setup"
	@echo ""
	@echo "Testing DSL implementation..."
	python -m kernels.dsl_conv || echo "DSL test failed - check dependencies"

# Run basic benchmarks
bench:
	@echo "Running conv2d benchmarks..."
	python -m bench.bench_conv --dtype=${DTYPE:-float16}

# Run comprehensive benchmarks with autotuning
bench-full:
	@echo "Running comprehensive conv2d benchmarks with autotuning..."
	python -m bench.bench_conv --autotune-triton --dtype=${DTYPE:-float16}

# Run profiling
profile:
	@echo "Running profiling with Nsight Compute..."
	@if command -v ncu >/dev/null 2>&1; then \
		bash bench/profile_conv.sh ${IMPL:-all} ${SHAPE_IDX:-0}; \
	else \
		echo "Error: Nsight Compute (ncu) not found. Please install CUDA toolkit."; \
		exit 1; \
	fi

# Analyze results
analyze:
	@echo "Analyzing benchmark results..."
	python -m bench.analyze

# Clean up generated files
clean:
	rm -rf results/
	rm -rf __pycache__/
	rm -rf kernels/__pycache__/
	rm -rf bench/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Development targets
dev-setup: install
	pip install pytest pytest-cov black isort mypy

lint:
	black kernels/ bench/
	isort kernels/ bench/

# Quick development test
dev-test:
	@echo "Quick development test..."
	python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Compute capability: {torch.cuda.get_device_capability()}')
"
	@echo ""
	python -c "
try:
    import triton
    print(f'Triton version: {triton.__version__}')
except ImportError:
    print('Triton not available')
"
	@echo ""
	python -c "
try:
    import ninetoothed
    print('NineToothed imported successfully')
except ImportError as e:
    print(f'NineToothed import failed: {e}')
"

# Documentation generation (if needed)
docs:
	@echo "Generating documentation..."
	@echo "This would generate docs if sphinx or similar is set up"

# CI/CD friendly targets
ci-test: dev-setup test

ci-bench: install bench

# Show system info
sysinfo:
	@echo "System Information"
	@echo "=================="
	python -c "
import sys
import torch
import subprocess
import platform

print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory//1024**3}GB)')

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f'NVIDIA Driver: {result.stdout.strip()}')
except:
    pass

try:
    import triton
    print(f'Triton: {triton.__version__}')
except:
    print('Triton: Not available')

try:
    import ninetoothed
    print('NineToothed: Available')
except:
    print('NineToothed: Not available')
"