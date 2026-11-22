#!/usr/bin/env python3
"""
Freeze each dataset's evaluation set (the manifest documents) to a CSV, so
experiments read the exact records from disk instead of downloading and
re-sampling the source datasets.

Writes eval_data/<dataset>.csv with columns
    sample_id, text, reference_summary, reference_summary_sha256
in manifest order. Files above --zip-above-mb are written as
eval_data/<dataset>.csv.zip instead (pandas reads it directly, no unzipping).
DatasetLoader.load_evaluation_set() picks these files up automatically.

Usage:
    python scripts/export_eval_data.py                 # all datasets with a manifest
    python scripts/export_eval_data.py --datasets arxiv
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.dataset_loader import DatasetLoader, MANIFEST_DIR, EVAL_DATA_DIR, reference_hash


def main():
    available = sorted(os.path.basename(p)[:-len('_manifest.csv')]
                       for p in glob.glob(os.path.join(MANIFEST_DIR, '*_manifest.csv')))
    parser = argparse.ArgumentParser(description='Freeze evaluation sets to eval_data/.')
    parser.add_argument('--datasets', nargs='+', default=available, choices=available)
    parser.add_argument('--out-dir', default=EVAL_DATA_DIR)
    parser.add_argument('--zip-above-mb', type=float, default=25.0,
                        help='Write a zipped CSV when the plain CSV would exceed this size')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for dataset in args.datasets:
        loader = DatasetLoader(dataset)
        # Load through the manifest (not an existing frozen file), so a re-export
        # always reflects the source data
        docs = loader.load_fixed_manifest(dataset, os.path.join(MANIFEST_DIR, f'{dataset}_manifest.csv'))
        df = pd.DataFrame({
            'sample_id': range(len(docs)),
            'text': [d['article'] for d in docs],
            'reference_summary': [d['reference_summary'] for d in docs],
            'reference_summary_sha256': [reference_hash(d['reference_summary']) for d in docs],
        })

        plain = os.path.join(args.out_dir, f'{dataset}.csv')
        zipped = plain + '.zip'
        for stale in (plain, zipped):
            if os.path.exists(stale):
                os.remove(stale)
        df.to_csv(plain, index=False)
        size_mb = os.path.getsize(plain) / 1e6
        if size_mb > args.zip_above_mb:
            df.to_csv(zipped, index=False,
                      compression={'method': 'zip', 'archive_name': f'{dataset}.csv'})
            os.remove(plain)
            print(f"{dataset}: {len(df)} documents -> {zipped} "
                  f"({size_mb:.1f} MB CSV, {os.path.getsize(zipped) / 1e6:.1f} MB zipped)")
        else:
            print(f"{dataset}: {len(df)} documents -> {plain} ({size_mb:.1f} MB)")

        # Round-trip check: the file must load back to exactly the same records
        reloaded = DatasetLoader(dataset).load_evaluation_set(dataset, eval_data_dir=args.out_dir)
        assert [(d['article'], d['reference_summary']) for d in reloaded] == \
               [(d['article'], d['reference_summary']) for d in docs], f"{dataset}: round-trip mismatch"


if __name__ == '__main__':
    main()
