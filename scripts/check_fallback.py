#!/usr/bin/env python3
"""
Detect summaries that are the lead-sentence fallback rather than model output.

Every summarizer falls back to the first N sentences of the article when a
call fails (API error, quota, CUDA error), and the run carries on. A failing
method therefore produces a complete-looking results file. This script counts,
per detailed_results file, how many generated summaries are exactly that
fallback, and exits non-zero if any method exceeds the threshold.

Usage:
    python scripts/check_fallback.py runs/cpu/cnn_dailymail/summaries_cnn_dailymail_*.csv \
        --dataset cnn_dailymail --max-sentences 4
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.dataset_loader import DatasetLoader

MANIFEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'manifests')
# Extractive methods can legitimately reproduce the lead sentences
EXTRACTIVE = ('textrank', 'tfidfrank')


def _hash(text) -> str:
    return hashlib.sha256((text or '').strip().encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Count lead-sentence fallback outputs per method.')
    parser.add_argument('files', nargs='+',
                        help='summaries_<dataset>_<method>.csv (or detailed_results_*) files')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--max-sentences', type=int, default=None,
                        help='Defaults to the dataset\'s standard value (as in main.py)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Maximum tolerated fallback share per method (default 1%%)')
    args = parser.parse_args()

    if args.max_sentences is None:
        from main import DEFAULT_MAX_SENTENCES
        args.max_sentences = DEFAULT_MAX_SENTENCES[args.dataset]

    loader = DatasetLoader(args.dataset)
    docs = loader.load_fixed_manifest(args.dataset, os.path.join(MANIFEST_DIR, f'{args.dataset}_manifest.csv'))
    articles = {_hash(d['reference_summary']): d['article'] for d in docs}

    failed = False
    for path in args.files:
        name = os.path.basename(path)[:-4]
        method = name.replace(f'summaries_{args.dataset}_', '').replace(f'detailed_results_{args.dataset}_', '')
        df = pd.read_csv(path)
        if not {'reference_summary', 'generated_summary'} <= set(df.columns):
            continue  # e.g. the per-method BLEURT summary file
        total = fallback = 0
        for ref, summary in zip(df['reference_summary'], df['generated_summary']):
            article = articles.get(_hash(ref))
            if article is None:
                continue
            total += 1
            lead = '. '.join(article.split('.')[:args.max_sentences]) + '.'
            fallback += str(summary).strip() == lead.strip()
        share = fallback / total if total else 0.0
        flag = ''
        if share > args.threshold and method.lower() not in EXTRACTIVE:
            flag = '  <-- FAILED: outputs are the lead-sentence fallback'
            failed = True
        print(f"{method:22} fallback {fallback}/{total} ({share:.1%}){flag}")

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
