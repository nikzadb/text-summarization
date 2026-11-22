#!/usr/bin/env python3
"""
Batched local-inference throughput benchmark (reviewer-requested realistic
batching configuration for locally deployed transformer models).

Measures docs/sec and per-document latency at batch sizes {1, 4, 8, 16} for
BART, DistilBART, PEGASUS, and LongT5 on a fixed document manifest (see
scripts/export_document_manifest.py), so throughput is measured on the same
documents used elsewhere in the paper. GPU-bound: intended to run on the same
AWS ml.g4dn.xlarge tier as the rest of the local-model experiments, not on
this CPU-only development machine.

Usage:
    python scripts/benchmark_batching.py \
        --dataset cnn_dailymail --method bart \
        --manifest manifests/cnn_dailymail_manifest.csv \
        --batch-sizes 1 4 8 16 --dtype fp32 \
        --out batching_results_cnn_dailymail_bart.csv
"""
import argparse
import csv
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset_loader import DatasetLoader
from src.summarizers.llm import BARTSummarizer, DistilBARTSummarizer
from src.summarizers.long_context import PegasusSummarizer, LongT5Summarizer


DTYPES = {'fp32': None, 'fp16': torch.float16}


def build_summarizer(method: str, dtype: str):
    torch_dtype = DTYPES[dtype]
    if method == 'bart':
        return BARTSummarizer('facebook/bart-large-cnn', torch_dtype=torch_dtype)
    elif method == 'distilbart':
        return DistilBARTSummarizer(torch_dtype=torch_dtype)
    elif method == 'pegasus':
        return PegasusSummarizer(torch_dtype=torch_dtype)
    elif method == 'longt5':
        return LongT5Summarizer(torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unsupported method for batching benchmark: {method}")


def run_batched(summarizer, texts, batch_size: int, max_sentences: int):
    """Runs the summarizer's HF pipeline with an explicit batch_size. A CUDA
    out-of-memory at a given batch size is recorded (oom=True) rather than
    aborting the sweep, since it is itself the feasibility result for that
    configuration on a 16 GB T4."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    oom = False
    try:
        summarizer.summarize_batch(texts, max_sentences, batch_size=batch_size)
    except torch.cuda.OutOfMemoryError:
        oom = True
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float('nan')
    if oom:
        elapsed = float('nan')
    return {
        'n_docs': len(texts),
        'batch_size': batch_size,
        'oom': oom,
        'peak_gpu_mem_gb': peak_mem_gb,
        'total_time_s': elapsed,
        'docs_per_sec': len(texts) / elapsed if elapsed > 0 else float('nan'),
        'avg_latency_per_doc_s': elapsed / len(texts) if texts else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser(description='Batched local-inference throughput benchmark.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True, choices=['bart', 'distilbart', 'pegasus', 'longt5'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--max-sentences', type=int, default=None,
                        help="Defaults to the dataset's standard value (as in main.py)")
    parser.add_argument('--max-docs', type=int, default=200,
                         help='Cap on documents used per batch-size trial (throughput stabilizes '
                              'well before 1,000 docs; keeps the sweep tractable across 4 batch '
                              'sizes x 4 models x 4 datasets).')
    parser.add_argument('--dtype', choices=list(DTYPES), default='fp32',
                        help='fp16 is the reduced-precision variant (not supported for LongT5)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.max_sentences is None:
        from main import DEFAULT_MAX_SENTENCES
        args.max_sentences = DEFAULT_MAX_SENTENCES[args.dataset]

    loader = DatasetLoader(args.dataset)
    samples = loader.load_fixed_manifest(args.dataset, args.manifest)
    if args.max_docs and len(samples) > args.max_docs:
        samples = samples[:args.max_docs]
    texts = [s['article'] for s in samples]

    print(f"Loading {args.method} for batching benchmark on {args.dataset} ({len(texts)} docs)...")
    summarizer = build_summarizer(args.method, args.dtype)

    # Warm-up so the first batch size isn't charged for CUDA init / kernel selection
    summarizer.summarize_batch(texts[:2], args.max_sentences, batch_size=1)

    rows = []
    for bs in args.batch_sizes:
        print(f"  batch_size={bs} ...")
        result = run_batched(summarizer, texts, bs, args.max_sentences)
        result.update({'dataset': args.dataset, 'method': args.method, 'dtype': args.dtype})
        rows.append(result)
        print(f"    {result['docs_per_sec']:.3f} docs/sec, "
              f"{result['avg_latency_per_doc_s']:.4f} s/doc, "
              f"peak {result['peak_gpu_mem_gb']:.2f} GB, oom={result['oom']}")

    summarizer.cleanup()

    fieldnames = ['dataset', 'method', 'dtype', 'batch_size', 'n_docs', 'oom', 'peak_gpu_mem_gb',
                  'total_time_s', 'docs_per_sec', 'avg_latency_per_doc_s']
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved batching results to {args.out}")


if __name__ == '__main__':
    main()
