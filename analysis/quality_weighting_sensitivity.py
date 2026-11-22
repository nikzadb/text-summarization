#!/usr/bin/env python3
"""
Composite-quality weighting sensitivity analysis (reviewer request: justify
equal weighting / test rank stability). Data transcribed directly from the
manuscript's four per-dataset result tables (ROUGE-1/2/L F1, BERTScore F1,
BLEURT).

Finding surfaced by inspecting src/evaluation_metrics.py
(compute_cross_method_zscore): the manuscript's "equal-weight average of
ROUGE, BERTScore, and BLEURT" is implemented as five separately Z-scored
sub-metrics (rouge1_f1, rouge2_f1, rougeL_f1, bert_f1, bleurt_score) averaged
with equal weight per sub-metric — not equal weight per metric *family*. Since
three of the five components are ROUGE variants, ROUGE as a family receives
3/5 = 60% of the composite's effective weight, BERTScore and BLEURT 1/5 = 20%
each. This script (a) reproduces that scheme to confirm it matches the
manuscript's reported Composite Quality column, and (b) recomputes the
composite under two alternative, explicitly-justified weighting schemes to
test rank/Pareto-membership stability, as the reviewer requested.
"""
import numpy as np

# (model, rouge1_f1, rouge2_f1, rougeL_f1, bert_f1, bleurt)
DATA = {
    'CNN/DailyMail': [
        ('BART',              0.439, 0.212, 0.309, 0.880, -0.434),
        ('DistilBART',        0.441, 0.212, 0.305, 0.883, -0.453),
        ('GPT-5-mini',        0.386, 0.172, 0.243, 0.869, -0.580),
        ('Gemini-2.5 Flash',  0.385, 0.172, 0.243, 0.868, -0.582),
        ('TextRank',          0.337, 0.137, 0.221, 0.861, -0.608),
        ('TF-IDFRank',        0.328, 0.132, 0.217, 0.860, -0.613),
    ],
    'arXiv': [
        ('BART',              0.363, 0.111, 0.202, 0.836, -0.467),
        ('DistilBART',        0.355, 0.104, 0.196, 0.836, -0.483),
        ('TextRank',          0.345, 0.126, 0.186, 0.824, -0.675),
        ('GPT-5-mini',        0.351, 0.099, 0.176, 0.833, -0.591),
        ('Gemini-2.5 Flash',  0.345, 0.097, 0.173, 0.832, -0.609),
        ('TF-IDFRank',        0.302, 0.107, 0.166, 0.815, -0.769),
    ],
    'WikiHow': [
        ('TextRank',          0.259, 0.069, 0.164, 0.840, -0.739),
        ('TF-IDFRank',        0.258, 0.069, 0.160, 0.840, -0.737),
        ('BART',              0.256, 0.058, 0.165, 0.840, -0.6661),
        ('DistilBART',        0.258, 0.059, 0.164, 0.835, -0.665),
        ('GPT-5-mini',        0.244, 0.056, 0.151, 0.838, -0.747),
        ('Gemini-2.5 Flash',  0.241, 0.055, 0.149, 0.837, -0.7542),
    ],
    'MediaSum': [
        ('BART',               0.177, 0.057, 0.144, 0.848, -0.961),
        ('DistilBART',         0.168, 0.054, 0.135, 0.839, -0.999),
        ('Gemini-2.5 Flash',   0.134, 0.042, 0.110, 0.818, -1.186),
        ('GPT-5-mini',         0.134, 0.042, 0.110, 0.818, -1.185),
        ('TF-IDFRank',         0.100, 0.025, 0.077, 0.814, -1.249),
        ('TextRank',           0.104, 0.026, 0.079, 0.814, -1.275),
    ],
}

REPORTED_COMPOSITE = {
    'CNN/DailyMail': {'BART': 1.193, 'DistilBART': 1.188, 'GPT-5-mini': -0.193,
                       'Gemini-2.5 Flash': -0.204, 'TextRank': -0.927, 'TF-IDFRank': -1.058},
    'arXiv': {'BART': 0.906, 'DistilBART': 0.599, 'TextRank': 0.138,
              'GPT-5-mini': -0.092, 'Gemini-2.5 Flash': -0.266, 'TF-IDFRank': -1.285},
    'WikiHow': {'TextRank': 0.592, 'TF-IDFRank': 0.589, 'BART': 0.548,
                'DistilBART': 0.126, 'GPT-5-mini': -0.816, 'Gemini-2.5 Flash': -1.039},
    'MediaSum': {'BART': 1.338, 'DistilBART': 0.987, 'Gemini-2.5 Flash': -0.152,
                 'GPT-5-mini': -0.157, 'TF-IDFRank': -1.002, 'TextRank': -1.015},
}


def zscore(values):
    values = np.array(values, dtype=float)
    std = values.std(ddof=1)
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def composite_5way(models):
    """Reproduces evaluation_metrics.py's compute_cross_method_zscore exactly:
    5 sub-metrics (R1,R2,RL,BERT,BLEURT), each Z-scored across methods, equal-averaged."""
    names = [m[0] for m in models]
    cols = list(zip(*[m[1:] for m in models]))  # 5 columns, one per metric
    z_cols = [zscore(c) for c in cols]
    combined = np.mean(z_cols, axis=0)
    return dict(zip(names, combined))


def composite_3way_family(models):
    """Alternative: equal weight per metric *family* (ROUGE averaged first, then
    Z-scored alongside BERTScore and BLEURT), rather than per sub-metric."""
    names = [m[0] for m in models]
    rouge_avg = [np.mean(m[1:4]) for m in models]
    bert = [m[4] for m in models]
    bleurt = [m[5] for m in models]
    z_rouge = zscore(rouge_avg)
    z_bert = zscore(bert)
    z_bleurt = zscore(bleurt)
    combined = np.mean([z_rouge, z_bert, z_bleurt], axis=0)
    return dict(zip(names, combined))


def composite_leave_one_out(models, drop: str):
    """drop in {'rouge', 'bert', 'bleurt'}: equal-average the remaining sub-metrics
    under the original 5-way (per-sub-metric) scheme."""
    names = [m[0] for m in models]
    idx_by_name = {'rouge1': 1, 'rouge2': 2, 'rougeL': 3, 'bert': 4, 'bleurt': 5}
    if drop == 'rouge':
        keep = ['bert', 'bleurt']
    elif drop == 'bert':
        keep = ['rouge1', 'rouge2', 'rougeL', 'bleurt']
    elif drop == 'bleurt':
        keep = ['rouge1', 'rouge2', 'rougeL', 'bert']
    else:
        raise ValueError(drop)
    cols = [[m[idx_by_name[k]] for m in models] for k in keep]
    z_cols = [zscore(c) for c in cols]
    combined = np.mean(z_cols, axis=0)
    return dict(zip(names, combined))


def rank_order(scores: dict):
    return [name for name, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def top1(scores: dict):
    return rank_order(scores)[0]


def main():
    for dataset, models in DATA.items():
        print(f"\n=== {dataset} ===")
        c5 = composite_5way(models)
        c3 = composite_3way_family(models)
        c_no_rouge = composite_leave_one_out(models, 'rouge')
        c_no_bert = composite_leave_one_out(models, 'bert')
        c_no_bleurt = composite_leave_one_out(models, 'bleurt')

        reported = REPORTED_COMPOSITE[dataset]
        max_abs_diff = max(abs(c5[name] - reported[name]) for name in reported)
        print(f"Reproduction check vs. manuscript's reported composite: max |diff| = {max_abs_diff:.4f}")

        print(f"{'Scheme':<38} {'Rank order (best -> worst)'}")
        print(f"{'5-way per-sub-metric (as published)':<38} {rank_order(c5)}")
        print(f"{'3-way per-metric-family':<38} {rank_order(c3)}")
        print(f"{'Leave-out: ROUGE':<38} {rank_order(c_no_rouge)}")
        print(f"{'Leave-out: BERTScore':<38} {rank_order(c_no_bert)}")
        print(f"{'Leave-out: BLEURT':<38} {rank_order(c_no_bleurt)}")

        top1_stable = len({top1(c5), top1(c3), top1(c_no_rouge), top1(c_no_bert), top1(c_no_bleurt)}) == 1
        print(f"Top-ranked model stable across all schemes: {top1_stable} "
              f"(top model = {top1(c5)})" if top1_stable else
              f"Top-ranked model NOT stable across schemes: "
              f"5-way={top1(c5)}, 3-way={top1(c3)}, no-ROUGE={top1(c_no_rouge)}, "
              f"no-BERT={top1(c_no_bert)}, no-BLEURT={top1(c_no_bleurt)}")


if __name__ == '__main__':
    main()
