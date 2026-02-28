#!/usr/bin/env python3
"""
Text Summarization Benchmarking Tool

This script benchmarks various text summarization techniques including:
1. Traditional methods (TextRank, TFIDFRank)
2. Open source LLM models (BART, T5, DistilBART)
3. Gemini API

Datasets: CNN/DailyMail, arXiv scientific papers, WikiHow
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
                       choices=['cnn_dailymail', 'arxiv', 'wikihow', 'govreport', 'mediasum'], #, 'samsum', 'qmsum', 'callsum'],
                       help='Datasets to benchmark on')
    parser.add_argument('--methods', nargs='+', 
                       default=['textrank', 'tfidfrank', 
                                 'distilbart', 'bart',             
                                 'LongformerEncoderDecoder', 'Retrieval-Augmented-Summarizer',
                                 'gemini', 'GPT-5-mini'
                                 ],
                       choices=['textrank', 'tfidfrank', 
                                't5', 'distilbart', 'bart', 
                                'gemini', 'GPT-5-mini', 
                                'hybrid_textrank_gemini', 'hybrid_tfidfrank_gemini',
                                'LongformerEncoderDecoder', 'Retrieval-Augmented-Summarizer',
                                'Pegasus-X'],
                       help='Summarization methods to benchmark')
    parser.add_argument('--max-samples', type=int, default=0,
                       help='Maximum number of samples per dataset')
    parser.add_argument('--max-sentences', type=int, default=3,
                       help='Maximum sentences in summary')
    parser.add_argument('--use-lambda', action='store_true', default=False,
                       help='Use AWS Lambda simulation')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='Output file for results (csv or json)')
    parser.add_argument('--enable-statistics', action='store_true', default=True,
                       help='Enable comprehensive statistical analysis with bootstrap CI and significance tests')
    
    args = parser.parse_args()
    
    
    print("🔬 Text Summarization Benchmarking Tool")
    print("=" * 50)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Max sentences per summary: {args.max_sentences}")
    print(f"AWS Lambda simulation: {'Enabled' if args.use_lambda else 'Disabled'}")
    print(f"Statistical analysis: {'Enabled' if args.enable_statistics else 'Disabled'}")
    print(f"Output file: {args.output}")
    print()
    
    # Initialize benchmark framework with only requested methods
    framework = BenchmarkFramework(
        use_lambda_simulation=args.use_lambda,
        enable_statistical_analysis=args.enable_statistics,
        methods=args.methods
    )
    
    # Run comprehensive benchmark
    try:
        if args.enable_statistics:
            # Run benchmark with statistical analysis
            results, statistical_results = framework.run_comprehensive_benchmark_with_statistics(
                datasets=args.datasets,
                methods=args.methods,
                max_samples=args.max_samples,
                max_sentences=args.max_sentences
            )
            
            # Display results with statistics
            framework.print_summary_with_statistics(statistical_results)
        else:
            # Run standard benchmark
            results = framework.run_comprehensive_benchmark(
                datasets=args.datasets,
                methods=args.methods,
                max_samples=args.max_samples,
                max_sentences=args.max_sentences
            )
            
            # Display standard results
            framework.print_summary()

        # Save summarisations
        pd.DataFrame(results).to_csv('summarisatuion-results.csv', index=False)
        
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