#!/usr/bin/env python3
"""
Example: How to use the Statistical Analysis Features

This script demonstrates how to run benchmarks with comprehensive statistical analysis.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from benchmark_framework import BenchmarkFramework


def example_statistical_benchmark():
    """Example of running benchmark with statistical analysis."""
    
    print("📊 Statistical Analysis Example")
    print("=" * 50)
    print("This example shows how to run benchmarks with:")
    print("• Bootstrap confidence intervals (95% CI)")
    print("• Paired bootstrap significance tests")  
    print("• Holm-Bonferroni multiple comparison correction")
    print("• Per-dataset statistical reporting")
    print()
    
    # Initialize framework with statistical analysis enabled
    framework = BenchmarkFramework(
        use_lambda_simulation=False,
        enable_statistical_analysis=True
    )
    
    # Example with limited methods and samples for quick demonstration
    datasets = ['cnn_dailymail']  # Single dataset for demo
    methods = ['textrank', 'tfidfrank']  # Simple methods that don't require model downloads
    max_samples = 10  # Small number for quick demo
    
    print(f"Running benchmark with:")
    print(f"• Datasets: {datasets}")
    print(f"• Methods: {methods}")
    print(f"• Max samples: {max_samples}")
    print(f"• Statistical analysis: Enabled")
    print()
    
    try:
        # Run comprehensive benchmark with statistics
        benchmark_results, statistical_results = framework.run_comprehensive_benchmark_with_statistics(
            datasets=datasets,
            methods=methods,
            max_samples=max_samples,
            max_sentences=3
        )
        
        print("\n" + "="*60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The statistical analysis provides:")
        print("• Bootstrap confidence intervals for all metrics")
        print("• Identification of the best method (highest Combined_ROUGE)")
        print("• Paired bootstrap tests: best method vs all others")
        print("• Holm-Bonferroni corrected p-values")
        print("• Effect sizes (Cohen's d)")
        print()
        print("Files generated:")
        print("• statistical_analysis.json - Full statistical results")
        print("• detailed_results_*.csv - Per-method detailed results")
        print("• benchmark_results.csv - Summary results")
        
    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("This might happen if:")
        print("• Required dependencies are not installed")
        print("• Dataset files are not available")
        print("• Network issues (for downloading datasets)")


def show_command_line_usage():
    """Show how to use statistical analysis from command line."""
    
    print("\n" + "="*60)
    print("COMMAND LINE USAGE")
    print("="*60)
    print("To enable statistical analysis from the command line:")
    print()
    print("python main.py --enable-statistics \\")
    print("               --datasets cnn_dailymail arxiv \\")
    print("               --methods textrank bart Pegasus-X \\")
    print("               --max-samples 100")
    print()
    print("This will:")
    print("1. Run standard benchmark")
    print("2. Perform statistical analysis on results")
    print("3. Generate reports with:")
    print("   • Combined_ROUGE (mean ± 95% bootstrap CI)")
    print("   • BERTScore (mean ± 95% bootstrap CI)")
    print("   • Latency (mean ± 95% bootstrap CI)")
    print("   • Cost (mean ± 95% bootstrap CI)")
    print("4. Run significance tests (best vs all others)")
    print("5. Apply Holm-Bonferroni correction")
    print("6. Save detailed statistical results")


def show_interpretation_guide():
    """Show how to interpret statistical results."""
    
    print("\n" + "="*60)
    print("INTERPRETING STATISTICAL RESULTS")
    print("="*60)
    print()
    print("BOOTSTRAP CONFIDENCE INTERVALS:")
    print("• Combined_ROUGE: 0.7543 ± (0.7321, 0.7765)")
    print("  → Mean performance is 0.7543")
    print("  → 95% confident true mean is between 0.7321 and 0.7765")
    print("  → Narrower intervals = more precise estimates")
    print()
    print("SIGNIFICANCE TESTS:")
    print("• Best vs Other:")
    print("  - Effect size (Cohen's d): 0.8245")
    print("    → Small: 0.2, Medium: 0.5, Large: 0.8+")
    print("  - Corrected p-value: 0.0023 ***")
    print("    → *** p < 0.001 (highly significant)")
    print("  - Difference CI: (0.0234, 0.0876)")
    print("    → Best method performs 0.0234-0.0876 points higher")
    print()
    print("SIGNIFICANCE LEVELS:")
    print("• * p < 0.05   (significant)")
    print("• ** p < 0.01  (highly significant)")
    print("• *** p < 0.001 (very highly significant)")
    print()
    print("MULTIPLE COMPARISONS:")
    print("• Holm-Bonferroni correction controls family-wise error rate")
    print("• Prevents false discoveries when testing multiple methods")
    print("• More conservative than uncorrected tests")


if __name__ == '__main__':
    print("🧪 Text Summarization Statistical Analysis")
    print("=" * 60)
    
    # Show usage examples
    show_command_line_usage()
    show_interpretation_guide()
    
    print("\n" + "="*60)
    print("LIVE EXAMPLE (Optional)")
    print("="*60)
    print("Would you like to run a live example? (requires datasets)")
    print("This will download datasets and run a small benchmark...")
    
    # Uncomment the next line to run the live example
    # example_statistical_benchmark()