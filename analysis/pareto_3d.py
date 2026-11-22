#!/usr/bin/env python3
"""
Programmatic Pareto-frontier computation (reviewer-requested: three-objective
quality-cost-latency analysis, reported alongside the existing two-objective
quality-cost frontier, explicitly identifying which models become dominated
once latency is included).

Data below is transcribed directly from the six per-architecture result
tables already in the manuscript (Tables tab:cnn-result, tab:arxiv-result,
tab:wikihow-result, tab:mediasum-result — composite quality, avg latency (s),
total cost for 1,000 docs). This script computes Pareto dominance
programmatically rather than by hand, as a cross-check on the manuscript's
existing (manually curated) 2D frontier, and adds the 3D analysis.

Dominance rule (matches manuscript's definition, Section 3.3): model a
dominates model b if Q_a >= Q_b, C_a <= C_b, (and L_a <= L_b for the 3D case),
with at least one strict inequality.
"""
from itertools import combinations

# (model, quality, cost_per_1000docs, avg_latency_s)
DATA = {
    'CNN/DailyMail': [
        ('BART',              1.193, 0.282,   1.068),
        ('DistilBART',        1.188, 0.195,   0.741),
        ('GPT-5-mini',       -0.193, 0.501,   5.031),
        ('Gemini-2.5 Flash', -0.204, 0.577,   0.372),
        ('TextRank',         -0.927, 0.00053, 0.008),
        ('TF-IDFRank',       -1.058, 0.00033, 0.005),
    ],
    'arXiv': [
        ('BART',              0.906, 0.406,  1.538),
        ('DistilBART',        0.599, 0.272,  1.030),
        ('TextRank',          0.138, 0.004,  0.067),
        ('GPT-5-mini',       -0.092, 2.702,  7.933),
        ('Gemini-2.5 Flash', -0.266, 3.149,  0.454),
        ('TF-IDFRank',       -1.285, 0.001,  0.016),
    ],
    'WikiHow': [
        ('TextRank',           0.592, 0.0004, 0.0059),
        ('TF-IDFRank',         0.589, 0.0002, 0.0037),
        ('BART',               0.548, 0.22,   0.834),
        ('DistilBART',         0.126, 0.142,  0.538),
        ('GPT-5-mini',        -0.816, 0.371,  5.492),
        ('Gemini-2.5 Flash',  -1.039, 0.417,  0.402),
    ],
    'MediaSum': [
        ('BART',               1.338, 0.185,  0.699),
        ('DistilBART',         0.987, 0.14,   0.531),
        ('Gemini-2.5 Flash',  -0.152, 0.724,  0.379),
        ('GPT-5-mini',        -0.157, 0.635,  5.553),
        ('TF-IDFRank',        -1.002, 0.0005, 0.007),
        ('TextRank',          -1.015, 0.001,  0.015),
    ],
}


def dominates(a, b, use_latency: bool):
    """a = (name, Q, C, L); returns True if a dominates b."""
    _, qa, ca, la = a
    _, qb, cb, lb = b
    if use_latency:
        at_least_as_good = (qa >= qb) and (ca <= cb) and (la <= lb)
        strictly_better = (qa > qb) or (ca < cb) or (la < lb)
    else:
        at_least_as_good = (qa >= qb) and (ca <= cb)
        strictly_better = (qa > qb) or (ca < cb)
    return at_least_as_good and strictly_better


def pareto_frontier(models, use_latency: bool):
    frontier = []
    for m in models:
        dominated = any(dominates(other, m, use_latency) for other in models if other[0] != m[0])
        if not dominated:
            frontier.append(m[0])
    return set(frontier)


def main():
    print(f"{'Dataset':<15} {'2D frontier (Q,C)':<45} {'3D frontier (Q,C,L)':<45} {'Newly dominated by latency'}")
    print('-' * 150)
    for dataset, models in DATA.items():
        f2 = pareto_frontier(models, use_latency=False)
        f3 = pareto_frontier(models, use_latency=True)
        newly_dominated = f2 - f3
        print(f"{dataset:<15} {', '.join(sorted(f2)):<45} {', '.join(sorted(f3)):<45} "
              f"{', '.join(sorted(newly_dominated)) if newly_dominated else '(none)'}")

    print()
    print("Detail: for each dataset, which model(s) dominate a newly-excluded model in 3D:")
    for dataset, models in DATA.items():
        f2 = pareto_frontier(models, use_latency=False)
        f3 = pareto_frontier(models, use_latency=True)
        newly_dominated = f2 - f3
        for name in sorted(newly_dominated):
            m = next(x for x in models if x[0] == name)
            dominators = [other[0] for other in models if other[0] != name and dominates(other, m, use_latency=True)]
            print(f"  {dataset} / {name} (Q={m[1]}, C={m[2]}, L={m[3]}) dominated by: {dominators}")


if __name__ == '__main__':
    main()
