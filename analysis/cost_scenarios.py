#!/usr/bin/env python3
"""
Reviewer-requested cost analysis under multiple deployment scenarios, plus a
break-even document-volume analysis (local GPU deployment vs. each API
model).

Break-even framing: the manuscript's existing cost model treats local GPU
cost as purely usage-proportional (Cost_infra = latency/3600 * hourly_rate),
which is the right model for *synchronous, on-demand* comparison but has no
fixed cost term, so "break-even volume" is undefined under it (marginal
per-doc cost is already constant regardless of volume). A break-even volume
is only a meaningful question under a *reserved/dedicated* GPU deployment,
where the hourly rate is paid whether or not the instance is idle. This
script therefore computes the standard reserved-infrastructure break-even:

    N* (documents/hour) = hourly_GPU_rate / per_document_API_cost

i.e. the number of documents per hour at which a continuously-running
reserved GPU instance costs the same as paying per-document for the API.
Below N*, pay-per-use API is cheaper (you'd otherwise pay for idle GPU time);
above N*, the reserved GPU is cheaper. This does NOT require the batched
local-inference throughput numbers (those establish *whether* N* is
achievable within an hour on one instance — a feasibility ceiling, not part
of the break-even value itself), so it is computed here from data already in
the manuscript's result tables.

Scenario 3 (asynchronous batch API, ~50% of real-time token pricing, already
cited in the manuscript's Cost Estimation section) is applied as a discount
to the API-side per-document cost, which mechanically doubles N* for every
API model — i.e. batch API pricing requires roughly 2x the hourly document
volume before reserved local GPU deployment becomes preferable.
"""

GPU_HOURLY_RATE = 0.958  # ml.g4dn.xlarge, Table tab:computing-spec

# (dataset, api_model, total_cost_per_1000_docs_usd)
API_COSTS = [
    ('CNN/DailyMail', 'GPT-5-mini',       0.501),
    ('CNN/DailyMail', 'Gemini-2.5 Flash', 0.577),
    ('arXiv',         'GPT-5-mini',       2.702),
    ('arXiv',         'Gemini-2.5 Flash', 3.149),
    ('WikiHow',       'GPT-5-mini',       0.371),
    ('WikiHow',       'Gemini-2.5 Flash', 0.417),
    ('MediaSum',      'GPT-5-mini',       0.635),
    ('MediaSum',      'Gemini-2.5 Flash', 0.724),
]

ASYNC_BATCH_DISCOUNT = 0.5  # ~50% of real-time token pricing, per manuscript Section 4.7


def main():
    print(f"{'Dataset':<15} {'API model':<20} {'$/doc (sync)':<14} "
          f"{'N* docs/hr (sync)':<20} {'N* docs/hr (async batch)':<26}")
    print('-' * 100)
    for dataset, api_model, cost_per_1000 in API_COSTS:
        per_doc_sync = cost_per_1000 / 1000.0
        per_doc_async = per_doc_sync * ASYNC_BATCH_DISCOUNT

        n_star_sync = GPU_HOURLY_RATE / per_doc_sync
        n_star_async = GPU_HOURLY_RATE / per_doc_async

        print(f"{dataset:<15} {api_model:<20} ${per_doc_sync:<13.5f} "
              f"{n_star_sync:<20.1f} {n_star_async:<26.1f}")

    print()
    print("Reading: below N* documents/hour, pay-per-use API is cheaper than paying for a")
    print("continuously reserved GPU instance; above N*, the reserved GPU is cheaper.")
    print("N* values assume the GPU can actually sustain that throughput within the hour —")
    print("this feasibility ceiling requires the batched-inference throughput measurements")
    print("(scripts/benchmark_batching.py), pending the AWS run, and is reported separately.")


if __name__ == '__main__':
    main()
