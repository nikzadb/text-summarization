#!/usr/bin/env python3
"""
Text Summarization Benchmarking Tool

This script benchmarks various text summarization techniques including:
1. Traditional methods (TextRank, TFIDFRank)
2. Open source LLM models (BART, T5, DistilBART)
3. Gemini API

Datasets: CNN/DailyMail, arXiv scientific papers, WikiHow
Metrics: ROUGE scores, BERTScore, BLEURT
"""

import argparse
import os
import sys
from dotenv import load_dotenv

import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.benchmark_framework import BenchmarkFramework


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Text Summarization Benchmarking Tool')
    parser.add_argument('--datasets', nargs='+', default=['cnn_dailymail'], 
                       choices=['cnn_dailymail', 'arxiv', 'wikihow', 'govreport', 'mediasum'], 
                       help='Datasets to benchmark on')
    parser.add_argument('--methods', nargs='+', 
                       default=['textrank', 'tfidfrank', 'distilbart', 'bart', 'gemini', 'GPT-5-mini'],
                       choices=['textrank', 'tfidfrank', 'distilbart', 'bart', 'gemini', 'GPT-5-mini'],
                       help='Summarization methods to benchmark')
    parser.add_argument('--max-samples', type=int, default=0,
                       help='Maximum number of samples per dataset')
    parser.add_argument('--max-sentences', type=int, default=3,
                       help='Maximum sentences in summary')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='Output file for results (csv or json)')
    
    args = parser.parse_args()
        
    print("🔬 Text Summarization Benchmarking Tool")
    print("=" * 50)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Max sentences per summary: {args.max_sentences}")
    print(f"Output file: {args.output}")
    print()
    
    # Initialize benchmark framework
    framework = BenchmarkFramework()
    
    # Run comprehensive benchmark
    try:
        results = framework.run_comprehensive_benchmark(
            datasets=args.datasets,
            methods=args.methods,
            max_samples=args.max_samples,
            max_sentences=args.max_sentences
        )

        # Save summarisations
        pd.DataFrame(results).to_csv('summarisatuion-results.csv', index=False)        
        
        # Display results
        framework.print_summary()
        
        # Save results
        framework.save_results(args.output)
        
        print(f"\n✅ Benchmarking completed! Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n❌ Benchmarking interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()