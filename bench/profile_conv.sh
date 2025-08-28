#!/bin/bash
# Profiling script for conv2d kernels using Nsight Compute
# Usage: ./profile_conv.sh <implementation> <shape_index> [output_dir]

set -e

IMPLEMENTATION=${1:-"all"}
SHAPE_INDEX=${2:-"0"} 
OUTPUT_DIR=${3:-"results/profiles"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Nsight Compute sections to profile
NCU_SECTIONS="launch,occupancy,sm__throughput,smsp__inst_executed,smsp__pipe_tensor,dram__throughput,l2tex__throughput,shared_load,shared_store,warp_stall"

# Base ncu command with stable settings
NCU_BASE="ncu --metrics $NCU_SECTIONS --csv --log-file /dev/null"

echo "Conv2d Profiling Script"
echo "======================"
echo "Implementation: $IMPLEMENTATION"
echo "Shape index: $SHAPE_INDEX"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to profile a specific implementation
profile_impl() {
    local impl=$1
    local output_file="$OUTPUT_DIR/profile_${impl}_shape${SHAPE_INDEX}.csv"
    
    echo "Profiling $impl implementation..."
    
    # Create temporary Python script for profiling
    cat > /tmp/profile_${impl}.py << EOF
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from kernels.common import BENCHMARK_SHAPES, create_test_tensors
from kernels.pytorch_conv import PyTorchConv2d
from kernels.triton_conv import TritonConv2d
from kernels.dsl_conv import DSLConv2d

def main():
    # Get the specified shape
    shape_idx = int(os.environ.get('SHAPE_INDEX', '0'))
    if shape_idx >= len(BENCHMARK_SHAPES):
        print(f"Shape index {shape_idx} out of range")
        return
        
    config = BENCHMARK_SHAPES[shape_idx]
    print(f"Profiling shape: {config}")
    
    # Create test tensors
    input_tensor, filter_tensor = create_test_tensors(config, torch.float16)
    
    # Warmup
    for _ in range(5):
        torch.cuda.synchronize()
    
    # Select implementation
    impl = "$impl"
    if impl == "pytorch":
        conv = PyTorchConv2d(config)
        output = conv(input_tensor, filter_tensor)
    elif impl == "triton":
        conv = TritonConv2d(config)
        output = conv(input_tensor, filter_tensor)  
    elif impl == "dsl":
        conv = DSLConv2d(config)
        output = conv(input_tensor, filter_tensor)
    else:
        print(f"Unknown implementation: {impl}")
        return
    
    # Ensure computation is complete
    torch.cuda.synchronize()
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
EOF

    # Run profiling with ncu
    SHAPE_INDEX=$SHAPE_INDEX $NCU_BASE --export "$output_file" python /tmp/profile_${impl}.py
    
    echo "  Profile saved to: $output_file"
    
    # Cleanup
    rm -f /tmp/profile_${impl}.py
}

# Function to extract PTX/SASS
extract_code() {
    local impl=$1
    local code_dir="$OUTPUT_DIR/code_${impl}_shape${SHAPE_INDEX}"
    
    echo "Extracting PTX/SASS for $impl..."
    mkdir -p "$code_dir"
    
    # Create script to compile and extract code
    cat > /tmp/extract_${impl}.py << EOF
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import subprocess
import tempfile
from pathlib import Path
from kernels.common import BENCHMARK_SHAPES, create_test_tensors
from kernels.triton_conv import TritonConv2d
from kernels.dsl_conv import DSLConv2d

def extract_triton_code(config, output_dir):
    """Extract generated Triton code and compile artifacts."""
    conv = TritonConv2d(config)
    
    # Compile kernel by running it once
    input_tensor, filter_tensor = create_test_tensors(config, torch.float16)
    _ = conv(input_tensor, filter_tensor)
    
    # Try to extract kernel information
    # This would need access to Triton's internals for full code extraction
    with open(f"{output_dir}/triton_kernel_info.txt", "w") as f:
        f.write(f"Triton Conv2d Configuration:\\n")
        f.write(f"Config: {config}\\n")
        f.write(f"Block sizes: M={conv.block_m}, N={conv.block_n}, K={conv.block_k}\\n")
        f.write(f"Warps: {conv.num_warps}, Stages: {conv.num_stages}\\n")

def extract_dsl_code(config, output_dir):
    """Extract DSL generated code."""
    from kernels.dsl_conv import get_generated_triton_code
    
    code = get_generated_triton_code(config)
    with open(f"{output_dir}/dsl_generated.py", "w") as f:
        f.write(code)

def main():
    shape_idx = int(os.environ.get('SHAPE_INDEX', '0'))
    config = BENCHMARK_SHAPES[shape_idx]
    output_dir = "$code_dir"
    impl = "$impl"
    
    if impl == "triton":
        extract_triton_code(config, output_dir)
    elif impl == "dsl":
        extract_dsl_code(config, output_dir)
    
    print(f"Code extraction complete for {impl}")

if __name__ == "__main__":
    main()
EOF

    SHAPE_INDEX=$SHAPE_INDEX python /tmp/extract_${impl}.py
    
    echo "  Code extracted to: $code_dir"
    rm -f /tmp/extract_${impl}.py
}

# Main profiling logic
if [ "$IMPLEMENTATION" = "all" ]; then
    for impl in pytorch triton dsl; do
        echo ""
        profile_impl $impl
        if [ "$impl" != "pytorch" ]; then
            extract_code $impl
        fi
    done
else
    profile_impl $IMPLEMENTATION
    if [ "$IMPLEMENTATION" != "pytorch" ]; then
        extract_code $IMPLEMENTATION
    fi
fi

echo ""
echo "Profiling complete!"
echo "Results in: $OUTPUT_DIR"

# Generate summary report
cat > "$OUTPUT_DIR/profile_summary.md" << EOF
# Profiling Summary

Generated: $(date)

## Configuration
- Implementation: $IMPLEMENTATION  
- Shape index: $SHAPE_INDEX
- Output directory: $OUTPUT_DIR

## Files Generated
- Profile data: profile_*_shape${SHAPE_INDEX}.csv
- Code extraction: code_*_shape${SHAPE_INDEX}/
- This summary: profile_summary.md

## Analysis Commands
\`\`\`bash
# View profile data
cat "$OUTPUT_DIR/profile_*_shape${SHAPE_INDEX}.csv"

# Compare implementations
python -c "
import pandas as pd
import glob
files = glob.glob('$OUTPUT_DIR/profile_*_shape${SHAPE_INDEX}.csv')
for f in files:
    print(f'\\n=== {f} ===')
    df = pd.read_csv(f)
    print(df.head())
"
\`\`\`
EOF

echo "Summary report: $OUTPUT_DIR/profile_summary.md"