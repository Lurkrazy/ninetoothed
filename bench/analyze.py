"""
Analysis script for conv2d benchmark results.
Generates plots, tables, and insights for the performance comparison.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BenchmarkAnalyzer:
    """Analyzer for conv2d benchmark results."""
    
    def __init__(self, results_file: Path):
        """Load benchmark results from JSON file."""
        with open(results_file, 'r') as f:
            self.raw_results = json.load(f)
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.raw_results)
        
        # Add derived columns
        self._add_derived_columns()
        
    def _add_derived_columns(self):
        """Add derived columns for analysis."""
        # Shape identifier
        self.df['shape_id'] = self.df.apply(
            lambda row: f"N{row['config']['N']}_C{row['config']['C']}_H{row['config']['H']}_K{row['config']['K']}", 
            axis=1
        )
        
        # Relative performance vs PyTorch
        pytorch_times = {}
        for _, row in self.df[self.df['implementation'] == 'pytorch'].iterrows():
            pytorch_times[row['shape_id']] = row['median_time_ms']
        
        self.df['pytorch_baseline_ms'] = self.df['shape_id'].map(pytorch_times)
        self.df['speedup_vs_pytorch'] = self.df['pytorch_baseline_ms'] / self.df['median_time_ms']
        self.df['relative_performance'] = (self.df['speedup_vs_pytorch'] - 1) * 100  # Percentage improvement
        
    def get_summary_table(self) -> pd.DataFrame:
        """Generate summary table of results."""
        summary = self.df.groupby(['implementation', 'shape_id']).agg({
            'median_time_ms': 'mean',
            'tflops': 'mean',
            'speedup_vs_pytorch': 'mean',
            'relative_performance': 'mean'
        }).round(3)
        
        return summary.reset_index()
    
    def plot_performance_comparison(self, save_path: Optional[Path] = None):
        """Plot performance comparison across implementations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Latency comparison
        pivot_time = self.df.pivot(index='shape_id', columns='implementation', values='median_time_ms')
        pivot_time.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Latency Comparison (Lower is Better)')
        ax1.set_ylabel('Time (ms)')
        ax1.set_xlabel('Shape Configuration')
        ax1.legend(title='Implementation')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: TFLOP/s comparison
        pivot_tflops = self.df.pivot(index='shape_id', columns='implementation', values='tflops')
        pivot_tflops.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Throughput Comparison (Higher is Better)')
        ax2.set_ylabel('TFLOP/s')
        ax2.set_xlabel('Shape Configuration')
        ax2.legend(title='Implementation')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_speedup_analysis(self, save_path: Optional[Path] = None):
        """Plot speedup analysis relative to PyTorch."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Filter out PyTorch (baseline)
        non_pytorch = self.df[self.df['implementation'] != 'pytorch']
        
        # Create speedup plot
        pivot_speedup = non_pytorch.pivot(index='shape_id', columns='implementation', values='speedup_vs_pytorch')
        pivot_speedup.plot(kind='bar', ax=ax, width=0.8)
        
        # Add horizontal line at 1.0 (parity)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='PyTorch Baseline')
        
        ax.set_title('Speedup vs PyTorch (cuDNN)')
        ax.set_ylabel('Speedup (Higher is Better)')
        ax.set_xlabel('Shape Configuration')
        ax.legend(title='Implementation')
        ax.grid(True, alpha=0.3)
        
        # Add text annotations for significant speedups
        for impl in pivot_speedup.columns:
            for i, shape in enumerate(pivot_speedup.index):
                speedup = pivot_speedup.loc[shape, impl]
                if speedup > 1.05:  # More than 5% improvement
                    ax.annotate(f'+{speedup-1:.1%}', 
                              xy=(i, speedup), 
                              xytext=(0, 5), 
                              textcoords='offset points',
                              ha='center', va='bottom',
                              fontsize=8, color='green', weight='bold')
                elif speedup < 0.95:  # More than 5% regression
                    ax.annotate(f'{speedup-1:.1%}', 
                              xy=(i, speedup), 
                              xytext=(0, -15), 
                              textcoords='offset points',
                              ha='center', va='top',
                              fontsize=8, color='red', weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roofline_analysis(self, save_path: Optional[Path] = None):
        """Create a roofline-style analysis plot."""
        # Hardware theoretical peaks (A100 80GB estimates)
        PEAK_TENSOR_TFLOPS = 312  # FP16 tensor cores
        PEAK_HBM_BW_GBS = 2039    # GB/s
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Calculate arithmetic intensity for each configuration
        # This is a simplified calculation - real analysis would need memory access patterns
        for impl in self.df['implementation'].unique():
            impl_data = self.df[self.df['implementation'] == impl]
            
            # Simple arithmetic intensity estimate (FLOP/byte)
            # Assumes input + filter + output memory access
            arithmetic_intensities = []
            tflops_achieved = []
            
            for _, row in impl_data.iterrows():
                config = row['config']
                
                # Memory footprint estimate (FP16 = 2 bytes)
                input_size = config['N'] * config['C'] * config['H'] * config['W'] * 2
                filter_size = config['K'] * config['C'] * config['R'] * config['S'] * 2
                output_size = config['N'] * config['K']  # Simplified, need actual output H/W
                total_memory = input_size + filter_size + output_size
                
                arithmetic_intensity = row['flops'] / total_memory
                arithmetic_intensities.append(arithmetic_intensity)
                tflops_achieved.append(row['tflops'])
            
            ax.scatter(arithmetic_intensities, tflops_achieved, 
                      label=impl, s=100, alpha=0.7)
        
        # Add roofline
        ai_range = np.logspace(-2, 2, 100)
        memory_bound = ai_range * PEAK_HBM_BW_GBS / 1000  # Convert to TFLOP/s 
        compute_bound = np.full_like(ai_range, PEAK_TENSOR_TFLOPS)
        roofline = np.minimum(memory_bound, compute_bound)
        
        ax.plot(ai_range, roofline, 'k--', linewidth=2, label='Theoretical Roofline')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
        ax.set_ylabel('Performance (TFLOP/s)')
        ax.set_title('Roofline Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_insights(self) -> List[str]:
        """Generate textual insights from the benchmark results."""
        insights = []
        
        # Overall winner analysis
        summary = self.get_summary_table()
        
        for shape in summary['shape_id'].unique():
            shape_data = summary[summary['shape_id'] == shape]
            best_impl = shape_data.loc[shape_data['tflops'].idxmax(), 'implementation']
            best_tflops = shape_data['tflops'].max()
            
            pytorch_tflops = shape_data[shape_data['implementation'] == 'pytorch']['tflops'].iloc[0]
            improvement = (best_tflops - pytorch_tflops) / pytorch_tflops * 100
            
            if best_impl != 'pytorch':
                insights.append(
                    f"Shape {shape}: {best_impl} achieves {best_tflops:.2f} TFLOP/s "
                    f"({improvement:+.1f}% vs PyTorch's {pytorch_tflops:.2f} TFLOP/s)"
                )
            else:
                insights.append(
                    f"Shape {shape}: PyTorch (cuDNN) is fastest at {pytorch_tflops:.2f} TFLOP/s"
                )
        
        # DSL vs Triton comparison
        dsl_data = self.df[self.df['implementation'] == 'dsl']
        triton_data = self.df[self.df['implementation'] == 'triton']
        
        if not dsl_data.empty and not triton_data.empty:
            dsl_avg_speedup = dsl_data['speedup_vs_pytorch'].mean()
            triton_avg_speedup = triton_data['speedup_vs_pytorch'].mean()
            
            if dsl_avg_speedup > triton_avg_speedup:
                diff = (dsl_avg_speedup - triton_avg_speedup) / triton_avg_speedup * 100
                insights.append(
                    f"DSL achieves {diff:.1f}% better average performance than hand-written Triton"
                )
            else:
                diff = (triton_avg_speedup - dsl_avg_speedup) / dsl_avg_speedup * 100
                insights.append(
                    f"Hand-written Triton achieves {diff:.1f}% better average performance than DSL"
                )
        
        return insights
    
    def export_analysis_report(self, output_dir: Path):
        """Export comprehensive analysis report."""
        output_dir.mkdir(exist_ok=True)
        
        # Generate plots
        self.plot_performance_comparison(output_dir / 'performance_comparison.png')
        self.plot_speedup_analysis(output_dir / 'speedup_analysis.png') 
        self.plot_roofline_analysis(output_dir / 'roofline_analysis.png')
        
        # Generate summary table
        summary_table = self.get_summary_table()
        summary_table.to_csv(output_dir / 'summary_table.csv', index=False)
        
        # Generate insights
        insights = self.generate_performance_insights()
        
        # Create markdown report
        report_md = f"""# Conv2d Performance Analysis Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

{chr(10).join(f"- {insight}" for insight in insights)}

## Detailed Results

### Summary Table

{summary_table.to_markdown(index=False)}

### Performance Metrics by Implementation

{self.df.groupby('implementation').agg({
    'median_time_ms': ['mean', 'std'],
    'tflops': ['mean', 'std'], 
    'speedup_vs_pytorch': ['mean', 'std']
}).round(3).to_markdown()}

## Visualizations

- Performance Comparison: `performance_comparison.png`
- Speedup Analysis: `speedup_analysis.png`  
- Roofline Analysis: `roofline_analysis.png`

## Raw Data

- Summary Table: `summary_table.csv`
- Full Results: Available in original benchmark output

## Methodology

Benchmarks were run using:
- 100 timing iterations after 10 warmup iterations
- FP16 precision with tensor cores enabled
- cuDNN benchmark mode enabled for PyTorch
- Multiple autotuning configurations tested for Triton and DSL

## Hardware Configuration

Target hardware: A100 80GB PCIe (sm_80)
- Theoretical peak: 312 TFLOP/s (FP16 tensor cores)
- Memory bandwidth: 2,039 GB/s
"""
        
        with open(output_dir / 'analysis_report.md', 'w') as f:
            f.write(report_md)
        
        print(f"Analysis report exported to {output_dir}")


def analyze_latest_results(results_dir: Path = Path('results')):
    """Analyze the latest benchmark results."""
    # Find latest results file
    json_files = list(results_dir.glob('conv2d_bench_*.json'))
    if not json_files:
        print(f"No benchmark results found in {results_dir}")
        return
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing results from: {latest_file}")
    
    analyzer = BenchmarkAnalyzer(latest_file)
    
    # Generate analysis
    analysis_dir = results_dir / 'analysis'
    analyzer.export_analysis_report(analysis_dir)
    
    return analyzer


def main():
    """Main analysis entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze conv2d benchmark results')
    parser.add_argument('--results-file', type=Path, 
                       help='Specific results file to analyze')
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                       help='Results directory (will use latest file)')
    
    args = parser.parse_args()
    
    if args.results_file:
        analyzer = BenchmarkAnalyzer(args.results_file)
        output_dir = args.results_file.parent / 'analysis'
        analyzer.export_analysis_report(output_dir)
    else:
        analyzer = analyze_latest_results(args.results_dir)
    
    if analyzer:
        print("\nKey Insights:")
        for insight in analyzer.generate_performance_insights():
            print(f"  • {insight}")


if __name__ == "__main__":
    main()