#!/usr/bin/env python3
"""
Quantify the impact of each local transformer's input limit (1,024 tokens
for BART, DistilBART and PEGASUS; 4,096 for LongT5): for each dataset, what fraction of each document's tokens are
retained before truncation kicks in, and on what fraction of documents does
truncation occur at all. Pure tokenization — no model inference required.

Usage:
    python scripts/quantify_truncation.py --manifest-dir manifests --out truncation_report.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import AutoTokenizer
from src.dataset_loader import DatasetLoader
from src.summarizers.long_context import PegasusSummarizer, LongT5Summarizer

MODEL_TOKEN_LIMIT = 1024
# name -> (checkpoint, revision, input token limit used in the experiments)
CHECKPOINTS = {
    'BART': ('facebook/bart-large-cnn', None, 1024),
    'DistilBART': ('sshleifer/distilbart-cnn-12-6', None, 1024),
    'PEGASUS': (PegasusSummarizer.MODEL_NAME, PegasusSummarizer.REVISION, 1024),
    'LongT5': (LongT5Summarizer.MODEL_NAME, LongT5Summarizer.REVISION, LongT5Summarizer.MAX_INPUT_TOKENS),
}


def quantify(dataset_name: str, manifest_path: str, tokenizer, token_limit: int = MODEL_TOKEN_LIMIT):
    loader = DatasetLoader(dataset_name)
    samples = loader.load_fixed_manifest(dataset_name, manifest_path)

    retained_pct = []
    truncated_flags = []
    doc_token_counts = []

    for s in samples:
        ids = tokenizer.encode(s['article'], add_special_tokens=False)
        n = len(ids)
        doc_token_counts.append(n)
        retained = min(n, token_limit)
        retained_pct.append(100.0 * retained / n if n > 0 else 100.0)
        truncated_flags.append(n > token_limit)

    return {
        'n_docs': len(samples),
        'mean_doc_tokens': float(np.mean(doc_token_counts)),
        'median_doc_tokens': float(np.median(doc_token_counts)),
        'pct_docs_truncated': 100.0 * float(np.mean(truncated_flags)),
        'mean_pct_retained': float(np.mean(retained_pct)),
        'median_pct_retained': float(np.median(retained_pct)),
        'p10_pct_retained': float(np.percentile(retained_pct, 10)),
    }


def main():
    parser = argparse.ArgumentParser(description='Quantify local-transformer truncation impact per dataset.')
    parser.add_argument('--manifest-dir', default='manifests')
    parser.add_argument('--datasets', nargs='+', default=['cnn_dailymail', 'arxiv', 'wikihow', 'mediasum'])
    parser.add_argument('--out', default='truncation_report.csv')
    args = parser.parse_args()

    tokenizers = {name: (AutoTokenizer.from_pretrained(ckpt, revision=rev), limit)
                  for name, (ckpt, rev, limit) in CHECKPOINTS.items()}

    rows = []
    for dataset_name in args.datasets:
        manifest_path = os.path.join(args.manifest_dir, f'{dataset_name}_manifest.csv')
        if not os.path.exists(manifest_path):
            print(f"Skipping {dataset_name}: no manifest at {manifest_path}")
            continue
        for model_name, (tok, limit) in tokenizers.items():
            print(f"Quantifying truncation: {dataset_name} / {model_name} (limit {limit}) ...")
            stats = quantify(dataset_name, manifest_path, tok, token_limit=limit)
            stats.update({'dataset': dataset_name, 'model': model_name, 'token_limit': limit})
            rows.append(stats)
            print(f"  mean_pct_retained={stats['mean_pct_retained']:.1f}%  "
                  f"pct_docs_truncated={stats['pct_docs_truncated']:.1f}%  "
                  f"median_doc_tokens={stats['median_doc_tokens']:.0f}")

    fieldnames = ['dataset', 'model', 'token_limit', 'n_docs', 'mean_doc_tokens', 'median_doc_tokens',
                  'pct_docs_truncated', 'mean_pct_retained', 'median_pct_retained', 'p10_pct_retained']
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved truncation report to {args.out}")


if __name__ == '__main__':
    main()
