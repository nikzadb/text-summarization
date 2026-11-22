#!/usr/bin/env python3
"""
Export a document manifest (SHA-256 hash of each reference summary, in
original sample order) from an already-collected detailed_results CSV.

New architectures (e.g. PEGASUS, LongT5) must be benchmarked against the
exact same 1,000 documents per dataset as the six originally evaluated
architectures for the paired bootstrap comparison to remain valid. Use this
script once per dataset against any existing detailed_results file, then pass
the manifest to `DatasetLoader.load_fixed_manifest()` (wired through
`main.py --fixed-manifest-dir`) when running new architectures.

Usage:
    python scripts/export_document_manifest.py \
        --detailed-csv experiments/cnn_dailymail/detailed_results_cnn_dailymail_bart.csv \
        --out manifests/cnn_dailymail_manifest.csv
"""
import argparse
import csv
import hashlib
import os


def _hash(text: str) -> str:
    return hashlib.sha256((text or '').strip().encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Export a document manifest for fixed cross-run sampling.')
    parser.add_argument('--detailed-csv', required=True, help='Path to an existing detailed_results_<dataset>_<method>.csv')
    parser.add_argument('--out', required=True, help='Output manifest CSV path')
    args = parser.parse_args()

    with open(args.detailed_csv, 'r', newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit(f'No rows found in {args.detailed_csv}')

    # Reference summaries occasionally collide (e.g. CNN/DailyMail contains a
    # small number of duplicate articles sharing identical highlights). A
    # colliding document cannot be told apart from its duplicate using only
    # the fields persisted in detailed_results, so both copies of any
    # colliding group are dropped from the manifest rather than guessed at.
    hash_counts = {}
    for row in rows:
        h = _hash(row['reference_summary'])
        hash_counts[h] = hash_counts.get(h, 0) + 1

    manifest_rows = []
    dropped = 0
    for row in rows:
        h = _hash(row['reference_summary'])
        if hash_counts[h] > 1:
            dropped += 1
            continue
        manifest_rows.append({
            'sample_id': row['sample_id'],
            'dataset': row['dataset'],
            'reference_summary_sha256': h,
        })

    if dropped:
        n_groups = sum(1 for c in hash_counts.values() if c > 1)
        print(
            f"Warning: dropped {dropped} document(s) across {n_groups} colliding "
            f"reference-summary group(s) ({dropped/len(rows):.2%} of {len(rows)}) — "
            f"these cannot be uniquely re-identified from detailed_results alone. "
            f"Document this reduced N in the manuscript if it is used for a new "
            f"architecture's evaluation set."
        )

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'dataset', 'reference_summary_sha256'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f'Wrote manifest for {len(manifest_rows)} documents to {args.out}')


if __name__ == '__main__':
    main()
