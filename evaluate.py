#!/usr/bin/env python3
"""
Step 2: score the summaries from step 1 and compare methods.

Reads <input-dir>/<dataset>/summaries_<dataset>_<method>.csv from one or more
input directories (e.g. the CPU and GPU instances' runs), computes ROUGE-1/2/L,
BERTScore and BLEURT per document, then the cross-method Z-scores, composite
quality and bootstrap statistics over all methods of each dataset together.
Run on a GPU: BERTScore (RoBERTa-large) and BLEURT are model-based.

Writes to <output-dir>/<dataset>/:
  detailed_results_<dataset>_<method>.csv            per-document scores (cached)
  benchmark_results_with_CI_<dataset>.csv            per-method means, CIs, Z-scores
  comprehensive_statistical_analysis_<dataset>.json  full statistics
and <output-dir>/benchmark_results.csv for all datasets.

Example:
  python evaluate.py --inputs runs/cpu runs/gpu \
      --datasets cnn_dailymail arxiv wikihow mediasum --output-dir results/main
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluation_pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description='Step 2: evaluate generated summaries')
    parser.add_argument('--inputs', nargs='+', required=True,
                        help='Step-1 output directories (each containing <dataset>/summaries_*.csv)')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to include (default: every method found). All methods of a '
                             'dataset are Z-scored together, so this set defines the comparison.')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Bootstrap iterations for confidence intervals (manuscript: 10,000)')
    parser.add_argument('--force', action='store_true',
                        help='Re-score every method even if cached per-document scores exist')
    args = parser.parse_args()

    pipeline = EvaluationPipeline(output_dir=args.output_dir, n_bootstrap=args.n_bootstrap, force=args.force)
    summary = pipeline.run(args.inputs, args.datasets, args.methods)
    if summary.empty:
        sys.exit("No results: no summaries found for the given inputs and datasets.")

    print("\n=== RESULTS (sorted by composite quality within each dataset) ===")
    cols = ['Dataset', 'Method', 'Combined Zscore', 'ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1',
            'BERT F1', 'BLEURT Score', 'Avg Time (s)', 'Total Cost ($)']
    print(summary[cols].to_string(index=False, float_format='{:.4f}'.format))
    print(f"\n✅ Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
