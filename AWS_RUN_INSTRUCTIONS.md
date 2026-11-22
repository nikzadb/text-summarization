# Full experiment run (revision)

All experiments for the revised manuscript: the eight architectures on four datasets, the chunk-and-aggregate comparison, batched throughput, and truncation statistics.

## How the pipeline is split

| Step | Script | What it does | Runs on |
|---|---|---|---|
| 1. Generation | `main.py` | Summarizes each document; records latency, cost, tokens, timestamp. Writes `summaries_<dataset>_<method>.csv` | The instance each method is priced at (below) |
| 2. Evaluation | `evaluate.py` | Scores all summaries (ROUGE-1/2/L, BERTScore, BLEURT); computes Z-scores, composite quality and bootstrap CIs over all methods of a dataset together | GPU (BERTScore and BLEURT are model-based) |

Step 1 loads no evaluation models, so the CPU instance never needs them. Step 2 reads the summary files from both instances, so the eight-architecture statistics are computed in one place.

Instances for step 1, matching the manuscript's cost model (Table 1):

| Instance | Spec | Rate | Methods |
|---|---|---|---|
| **CPU:** ml.c5.xlarge | 4 vCPU, 8 GB RAM | $0.238/h | TextRank, TF-IDFRank (local CPU); Gemini-3.8 Flash, GPT-5.4-mini (inference on the provider's servers; the instance only sends requests) |
| **GPU:** ml.g4dn.xlarge | 4 vCPU, 16 GB RAM, 1× T4 16 GB | $0.958/h | DistilBART, BART, PEGASUS, LongT5 |

Latency is measured on the hardware the method is priced at, and cost is derived from that latency (plus token pricing for the API models).

For Gemini and GPT, latency runs from sending the request until the summary is received. Waits for the rate limit, failed attempts (rate limits, overload, timeouts) and the backoff between retries are not counted. API calls are capped at 2.5 per second per method (`max_calls_per_second` in `gemini.py` and `openai_gpt.py`).

## Order of work

| Part | Instance | Steps |
|---|---|---|
| A | CPU | A0 smoke test → A1 generate CPU methods → copy `runs/cpu/` to the GPU instance |
| B | GPU | B0 smoke test → B1 generate GPU methods → B2 chunk-and-aggregate → B3 batching → B4 truncation |
| C | GPU | C1 evaluate the eight architectures → C2 evaluate chunk-and-aggregate |

Parts A and B run at the same time on separate machines. Part C starts once A1 and B1–B2 have finished. The CPU instance can be stopped as soon as its summaries are copied.

## Same documents in every run

Every run loads its documents from `manifests/<dataset>_manifest.csv`. These are the exact documents the original experiments used (CNN/DailyMail 998, the others 1,000), always in the same order. Both instances, all methods and all re-runs are scored on identical records; `evaluate.py` checks this and stops if two methods' documents differ.

- This is the default; no flag is needed.
- `--max-samples N` takes the **first N** manifest documents, so a test run is always a subset of the full set.
- A run stops with an error if any manifest document can't be found in the source data.
- Only datasets without a manifest (e.g. GovReport) fall back to seeded sampling (`--seed`, default 42), with a printed warning.

## Settings

Summary length per dataset is built in and matches the original runs. Don't pass `--max-sentences`.

| Dataset | Sentences |
|---|---|
| cnn_dailymail | 4 |
| arxiv | 10 |
| wikihow | 5 |
| mediasum | 2 |

Models:

| Model | Checkpoint | Revision | Input limit |
|---|---|---|---|
| BART | `facebook/bart-large-cnn` | latest | 1,024 |
| DistilBART | `sshleifer/distilbart-cnn-12-6` | latest | 1,024 |
| PEGASUS | `google/pegasus-cnn_dailymail` | `40d588fdab0cc077b80d950b300bf66ad3c75b92` | 1,024 |
| LongT5 | `pszemraj/long-t5-tglobal-base-16384-book-summary` | `a40cddfd840d398653f37d98c384bfe56d677125` | 4,096 |
| Gemini-3.8 Flash | `gemini-3.8-flash` (API) | provider alias | full document |
| GPT-5.4-mini | `gpt-5.4-mini` (API), reasoning off | provider alias | full document |

All local transformers decode with 4 beams in fp32 (B3 adds fp16 as a separate measurement). Evaluation uses ROUGE-1/2/L, BERTScore (RoBERTa-large) and BLEURT (`Elron/bleurt-base-512`, the same checkpoint as the original results), with 10,000 bootstrap iterations.

**Before Part A,** fix two open issues in `src/summarizers/gemini.py`:

- **Prices:** it still carries Gemini 2.5 Flash prices ($0.30 / $2.50 per M tokens; see the TODO there). Set the Gemini-3.8 Flash prices first, otherwise every Gemini cost is wrong.
- **Truncated summaries:** Gemini's thinking counts against `max_output_tokens=200` and uses up most of it, so summaries come back cut off (in testing, Gemini-3.8 Flash returned only `a single sentence summary:**`). Either turn thinking off (needs the newer `google-genai` SDK) or raise the limit. Until this is fixed, Gemini summaries are unusable.

## Setup (both instances)

```bash
git clone <repo-url> text-summarization   # or: git pull
cd text-summarization
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`data/wikihowAll.csv` isn't in git; copy it into `data/` on both instances.

Check the hardware: on the GPU instance, `nvidia-smi` must show the T4. On the CPU instance, `nproc` must print 4 and `python -c "import torch; print(torch.cuda.is_available())"` must print `False`.

Run everything from the repo root, inside `tmux` so a dropped SSH session doesn't kill the run:

```bash
tmux new -s runs
cd ~/text-summarization && source .venv/bin/activate
DATASETS="cnn_dailymail arxiv wikihow mediasum"
mkdir -p runs results
```

If a step-1 run is interrupted, re-run the same command with `--skip-existing`. Completed (dataset, method) pairs are skipped.

---

# Part A: CPU instance (ml.c5.xlarge)

Create `.env` in the repo root (see `.env.example`). Only this instance needs it:

```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

**The Gemini key must be on a paid (billing-enabled) tier.** In the original runs, 98–100% of Gemini calls failed with "You exceeded your current quota", and every OpenAI call failed with a response-parsing error that has since been fixed. Each failed call silently produced the article's lead sentences instead of a summary, so the run looked complete.

That fallback has been removed for the API models:

- **Temporary errors are retried:** rate limits (429), overload (503), timeouts and server errors get up to 8 attempts, with waits of 2 s, 4 s, 8 s … up to 60 s between them. Each retry logs a `⚠ API error (attempt n/8)` line.
- **Persistent errors stop the method:** if all 8 attempts fail, or the error is permanent (bad key, quota exhausted, bad request), that method stops on that dataset and is reported with `✗`. Nothing is saved for that pair, because summaries are only written once all documents finish.

The fallback checks below are still a useful safety net; don't skip them. Set a spending limit on both API accounts.

## A0. Smoke test (about 5 minutes)

```bash
python main.py --datasets $DATASETS --methods textrank tfidfrank gemini GPT-5.4-mini \
  --max-samples 5 --output-dir runs/smoke/cpu 2>&1 | tee runs_smoke_cpu.log
grep -E "✗|not found" runs_smoke_cpu.log                    # should print nothing
grep -c "API error" runs_smoke_cpu.log                      # retries; a few are fine
for ds in $DATASETS; do
  python scripts/check_fallback.py runs/smoke/cpu/$ds/summaries_${ds}_*.csv --dataset $ds
done
```

Each dataset must have four `summaries_*.csv` files, and `check_fallback.py` must report 0 fallback outputs for Gemini and GPT. Open one Gemini and one GPT summary to confirm they read as complete summaries, not cut off mid-sentence. Don't start A1 until all of this holds.

If almost every call needs retries, the rate limit is too high for the account tier. Lower `max_calls_per_second` before A1, rather than letting retries absorb it.

## A1. Generate: CPU methods

```bash
python main.py --datasets $DATASETS --methods textrank tfidfrank gemini GPT-5.4-mini \
  --output-dir runs/cpu 2>&1 | tee runs/cpu_generation.log
for ds in $DATASETS; do
  python scripts/check_fallback.py runs/cpu/$ds/summaries_${ds}_*.csv --dataset $ds
done
pip freeze > runs/cpu/pip_freeze.txt; lscpu > runs/cpu/lscpu.txt
```

An API method that still fails after its retries is reported with `✗` and skipped; the run continues with the rest. Once the cause (quota, key, an outage) is fixed, re-run the same command with `--skip-existing` to generate only the missing pairs. A method above 1% fallback on a dataset must also be regenerated: delete its `summaries_*` file and re-run with that method and dataset only.

## A2. Copy the summaries to the GPU instance

Through S3:

```bash
aws s3 sync runs/cpu s3://<bucket>/text-summarization/runs/cpu        # on the CPU instance
aws s3 sync s3://<bucket>/text-summarization/runs/cpu runs/cpu        # on the GPU instance
```

or directly: `scp -r runs/cpu <gpu-host>:~/text-summarization/runs/`.

---

# Part B: GPU instance (ml.g4dn.xlarge)

No API keys are needed here.

## B0. Smoke test (about 20 minutes, mostly model downloads)

```bash
python main.py --datasets $DATASETS --methods distilbart bart pegasus longt5 \
  --max-samples 5 --output-dir runs/smoke/gpu 2>&1 | tee runs_smoke_gpu.log
grep -E "✗|generation failed|not found" runs_smoke_gpu.log  # should print nothing
grep "Using device" runs_smoke_gpu.log | sort -u            # must say cuda
for ds in $DATASETS; do
  python scripts/check_fallback.py runs/smoke/gpu/$ds/summaries_${ds}_*.csv --dataset $ds
done
python evaluate.py --inputs runs/smoke/gpu --datasets $DATASETS \
  --output-dir results/smoke --n-bootstrap 200               # checks step 2 runs on this machine
```

## B1. Generate: local transformers

```bash
python main.py --datasets $DATASETS --methods distilbart bart pegasus longt5 \
  --output-dir runs/gpu 2>&1 | tee runs/gpu_generation.log
for ds in $DATASETS; do
  python scripts/check_fallback.py runs/gpu/$ds/summaries_${ds}_*.csv --dataset $ds
done
```

A local model falls back to lead sentences when generation fails, e.g. on a CUDA out-of-memory error (logged as `generation failed`). Any fallback above 1% means that method has to be regenerated on that dataset.

## B2. Generate: chunk-and-aggregate (arXiv and MediaSum)

Written to its own directory so it stays out of the eight-architecture comparison:

```bash
python main.py --datasets arxiv mediasum --methods bart_chunked distilbart_chunked \
  --output-dir runs/gpu_chunked 2>&1 | tee runs/gpu_chunked_generation.log
```

Each chunk is 1,000 tokens and is summarized to 2 sentences; the concatenated chunk summaries are then summarized to the dataset's sentence count. Documents that fit in one chunk give exactly the leading-truncation output.

## B3. Batched throughput (200 documents per configuration)

```bash
mkdir -p runs/gpu_batching
for ds in $DATASETS; do
  for m in bart distilbart pegasus longt5; do
    for dt in fp32 fp16; do
      [ "$m" = longt5 ] && [ "$dt" = fp16 ] && continue   # T5 models produce NaNs in fp16
      python scripts/benchmark_batching.py --dataset $ds --method $m --dtype $dt \
        --manifest manifests/${ds}_manifest.csv --batch-sizes 1 4 8 16 \
        --out runs/gpu_batching/${ds}_${m}_${dt}.csv 2>&1 | tee -a runs/gpu_batching/batching.log
    done
  done
done
```

Each configuration uses the first 200 manifest documents. An out-of-memory error at a batch size is recorded as `oom=True` in the CSV (it is itself the result: that configuration doesn't fit on a T4) and the sweep continues.

## B4. Truncation statistics

Tokenizer only, a few minutes:

```bash
python scripts/quantify_truncation.py --out runs/truncation_report.csv
pip freeze > runs/gpu/pip_freeze.txt; nvidia-smi > runs/gpu/nvidia_smi.txt
```

Per dataset: the share of documents truncated and the mean share of tokens kept, for BART, DistilBART, PEGASUS (1,024 tokens) and LongT5 (4,096 tokens). Covers manuscript Table 6.

Run B1–B4 one at a time: two GPU jobs at once would inflate each other's latencies.

---

# Part C: evaluation (GPU instance)

Start once `runs/cpu/` (from A2), `runs/gpu/` and `runs/gpu_chunked/` are all on the GPU instance, and no generation job is running.

## C1. The eight architectures

```bash
python evaluate.py --inputs runs/cpu runs/gpu --datasets $DATASETS \
  --methods textrank tfidfrank gemini GPT-5.4-mini distilbart bart pegasus longt5 \
  --output-dir results/main 2>&1 | tee results_main.log
```

Listing `--methods` explicitly makes the run stop if any of the eight is missing for a dataset, rather than silently Z-scoring a smaller set.

## C2. Chunk-and-aggregate vs. leading truncation

```bash
python evaluate.py --inputs runs/gpu runs/gpu_chunked --datasets arxiv mediasum \
  --methods bart bart_chunked distilbart distilbart_chunked \
  --output-dir results/chunked 2>&1 | tee results_chunked.log
```

Compare `bart` with `bart_chunked` and `distilbart` with `distilbart_chunked` on the raw metrics. The Z-scores in this output are relative to these four methods only, so they aren't comparable with C1's.

Per-document scores are cached in `detailed_results_*.csv`; re-running `evaluate.py` only scores methods whose summaries changed (use `--force` to re-score everything).

---

## Rough runtime

| Part | Steps | Estimate |
|---|---|---|
| A (CPU) | A1 | 5–12 h; API latency is unknown, since the original API timings were mostly failed calls |
| B (GPU) | B1 | 6–10 h; LongT5 on arXiv is the slowest |
| B (GPU) | B2 | 4–8 h; arXiv averages ~7 chunks per document |
| B (GPU) | B3 | 3–6 h |
| B (GPU) | B4 | < 15 min |
| C (GPU) | C1 + C2 | 1–3 h |

Part B is the critical path; Part A finishes alongside it. In total, about 14–27 h.

## Approximate API spend

Rough, from input sizes (the original runs' recorded API costs are estimates for calls that mostly failed): about $10–15 for GPT-5.4-mini ($0.75 / $4.50 per M tokens) across all four datasets, about 60% of it on arXiv. Gemini-3.8 Flash depends on its prices (see above). The smoke test is negligible.

## What to bring back

From the GPU instance, which by then holds everything:

- `runs/cpu/`, `runs/gpu/`, `runs/gpu_chunked/`: `summaries_*.csv` and `summaries_*.meta.json` (hardware, timing, git commit, API call dates), plus `pip_freeze.txt`, `lscpu.txt` and `nvidia_smi.txt`.
- `runs/gpu_batching/` and `runs/truncation_report.csv`.
- `results/main/` and `results/chunked/`: `detailed_results_*`, `benchmark_results_with_CI_*`, `comprehensive_statistical_analysis_*.json`, `benchmark_results.csv`.
- All `*.log` files.
