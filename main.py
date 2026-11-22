#!/usr/bin/env python3
"""
Step 1: generate summaries and measure latency and cost.

Runs each method on the fixed document set of each dataset and writes
<output-dir>/<dataset>/summaries_<dataset>_<method>.csv (plus a .meta.json
recording the hardware and timing). No evaluation happens here; score the
summaries afterwards with evaluate.py (step 2, on a GPU).

Methods:
1. Extractive (TextRank, TF-IDFRank)                         - CPU
2. Local transformers (DistilBART, BART, PEGASUS, LongT5)    - GPU
3. API LLMs (Gemini, GPT-5.4-mini)                           - CPU client

Datasets: CNN/DailyMail, arXiv, WikiHow, MediaSum (GovReport: no manifest)
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.benchmark_framework import BenchmarkFramework, SUMMARIZERS

# Output length per dataset, matched to the average reference length (as in the original runs)
DEFAULT_MAX_SENTENCES = {'cnn_dailymail': 4, 'arxiv': 10, 'wikihow': 5, 'mediasum': 2, 'govreport': 3}


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description='Step 1: generate summaries (latency and cost, no evaluation)')
    parser.add_argument('--datasets', nargs='+', default=['cnn_dailymail'],
                       choices=list(DEFAULT_MAX_SENTENCES),
                       help='Datasets to benchmark on')
    parser.add_argument('--methods', nargs='+', required=True, choices=list(SUMMARIZERS),
                       help='Summarization methods to run')
    parser.add_argument('--max-samples', type=int, default=0,
                       help='Use only the first N documents of each dataset (0 = all)')
    parser.add_argument('--max-sentences', type=int, default=None,
                       help='Summary length for every dataset; by default each dataset uses its own '
                            f'value {DEFAULT_MAX_SENTENCES}')
    parser.add_argument('--fixed-manifest-dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manifests'),
                       help='Directory of <dataset>_manifest.csv files. A dataset with a manifest '
                            'is loaded from it (the same documents as the original runs; '
                            '--max-samples takes the first N); one without falls back to seeded '
                            'sampling')
    parser.add_argument('--seed', type=int, default=42,
                       help='Sampling seed for datasets without a manifest')
    parser.add_argument('--output-dir', type=str, default='runs',
                       help='Summaries are written to <output-dir>/<dataset>/')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip (dataset, method) pairs whose summaries file is already complete, '
                            'e.g. to resume an interrupted run')

    args = parser.parse_args()
    max_sentences = args.max_sentences or {ds: DEFAULT_MAX_SENTENCES[ds] for ds in args.datasets}

    print("🔬 Text Summarization Benchmarking Tool - step 1: generation")
    print("=" * 50)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Max samples per dataset: {args.max_samples or 'all'}")
    print(f"Max sentences per summary: {max_sentences}")
    print(f"Fixed manifest dir: {args.fixed_manifest_dir}")
    print(f"Sampling seed (datasets without a manifest): {args.seed}")
    print(f"Output dir: {args.output_dir}")
    print()

    framework = BenchmarkFramework(output_dir=args.output_dir)

    try:
        framework.run_comprehensive_benchmark(
            datasets=args.datasets,
            methods=args.methods,
            max_samples=args.max_samples,
            max_sentences=max_sentences,
            manifest_dir=args.fixed_manifest_dir,
            seed=args.seed,
            skip_existing=args.skip_existing
        )
        framework.print_summary()
        print(f"\n✅ Generation completed. Summaries are in {args.output_dir}/; score them with evaluate.py")

    except KeyboardInterrupt:
        print("\n❌ Benchmarking interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
