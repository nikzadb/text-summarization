#!/usr/bin/env python3
"""
Text Summarization Benchmarking Tool

This script benchmarks various text summarization techniques including:
1. Traditional methods (TextRank, TFIDFRank)
2. Open source LLM models (BART, T5, DistilBART)
3. Gemini API

Datasets: CNN/DailyMail, arXiv scientific papers
Metrics: ROUGE scores, BERTScore
Computing: AWS Lambda simulation (optional)
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
                       choices=['cnn_dailymail', 'arxiv'],
                       help='Datasets to benchmark on')
    parser.add_argument('--methods', nargs='+', 
                       default=['textrank', 'tfidfrank', 't5', 'distilbart', 'bart', 'gemini', 'hybrid_textrank_gemini'],
                       choices=['textrank', 'tfidfrank', 't5', 'distilbart', 'bart', 'gemini', 'hybrid_textrank_gemini'],
                       help='Summarization methods to benchmark')
    parser.add_argument('--max-samples', type=int, default=0,
                       help='Maximum number of samples per dataset')
    parser.add_argument('--max-sentences', type=int, default=3,
                       help='Maximum sentences in summary')
    parser.add_argument('--use-lambda', action='store_true', default=False,
                       help='Use AWS Lambda simulation')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='Output file for results (csv or json)')
    parser.add_argument('--gemini-api-key', type=str,
                       help='Gemini API key (or set GEMINI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Set Gemini API key if provided
    if args.gemini_api_key:
        os.environ['GEMINI_API_KEY'] = args.gemini_api_key
    
    print("🔬 Text Summarization Benchmarking Tool")
    print("=" * 50)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Max sentences per summary: {args.max_sentences}")
    print(f"AWS Lambda simulation: {'Enabled' if args.use_lambda else 'Disabled'}")
    print(f"Output file: {args.output}")
    print()
    
    # Initialize benchmark framework
    framework = BenchmarkFramework(use_lambda_simulation=args.use_lambda)
    
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