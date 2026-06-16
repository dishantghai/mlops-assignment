# Actual Run Log — HW3 LLM Inference Tuning

Operational log. Short entries. Saw → Hypothesized → Changed → Result format.
Educational detail lives in `mlops-hw3-runlog.md`.

---

## Config History

| Iter | Key Flags | TTFT P95 | E2E P95 | KV Cache % | Queue Peak | Notes |
|------|-----------|----------|---------|------------|------------|-------|
| 0 | no flags (BF16 defaults) | — | — | — | — | CRASH — KV OOM at startup |
| 1 | `--max-model-len 8192 --no-prefix-cache` (chunked prefill forced on) | **59ms** | **1.059s** avg / **1.895s** P95 wall | ~14% | 0 | 10 RPS × 120s, 1200/1200 ok |

---

## Active Config

**File:** `scripts/start_vllm.sh`
**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`

```bash
# ITER 1 — true baseline (BF16, capped context, all optimizations off)
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill
```

**Why these two flags are needed for a clean baseline:**
vLLM 0.10.x auto-enables `prefix_caching=True` and `chunked_prefill=True` for Qwen3MoE models even with no explicit flags. Without explicitly disabling them, Iter 1 is already a partially-optimized config. Adding them back one at a time in later iterations lets us measure the isolated impact of each.

---

## SLO Target

| Metric | Target |
|--------|--------|
| E2E P95 latency | < 5.0s |
| Sustained RPS | 10 |
| Load test duration | 300s |

---

## Iteration 0 — No-Flag Baseline (CRASH)

**Date:** 2026-06-16
**Config:** No flags. Pure vLLM defaults.
**Outcome:** Crashed at startup. Never served a request.

**Saw:**
- Model loaded in BF16 → weights took **56.93 GiB**
- Only **8.68 GiB** HBM left for KV cache
- vLLM defaulted `max_model_len=262144` (model's native 256K window)
- Minimum KV cache for even one sequence at 262144 tokens = **24 GiB** → impossible

**Error:**
```
ValueError: To serve at least one request with max seq len (262144),
24.00 GiB KV cache is needed, larger than available KV cache memory (8.68 GiB).
Estimated maximum model length is 94784.
```

**Diagnosis:** BF16 weights leave too little HBM for any meaningful KV cache when context window is uncapped. This is a hard startup failure, not a performance issue.

**Action taken:** Added `--max-model-len 8192` to `scripts/start_vllm.sh`. This caps per-sequence KV allocation so vLLM can at least start and serve within the 8.68 GiB KV budget. Our workload only needs ~1,500 tokens max, so 8192 gives plenty of headroom.

---

## Iteration 1 — BF16 + max-model-len 8192

**Date:** 2026-06-16
**Config:** `--max-model-len 8192` only. Still BF16.
**Goal:** Get vLLM running. Establish true first-call latency baseline in BF16. Expect poor concurrency but want to measure it.

### Smoke Test

```bash
bash scripts/start_vllm.sh

# After "Application startup complete":
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "Write a SQL query to list all superhero names. Write only SQL, no explanation."}
    ],
    "max_tokens": 200
  }' | jq '{content: .choices[0].message.content, tokens: .usage}'

# Check KV headroom immediately
curl -s localhost:8000/metrics | grep -v "^#" | grep -E "kv_cache|num_requests"
```

### Startup Confirmed

```
Weights:                56.93 GiB (BF16)
KV cache budget:        8.68 GiB → 94,784 total tokens
Max concurrency:        11.57x at 8192 tokens/seq (worst case)
                        ~79x at actual ~1,200 tokens/seq (real workload)
Prefix caching:         OFF (confirmed via non-default args)
Chunked prefill:        ON  (vLLM force-enables for Qwen3MoE, flag ignored)
Default sampling:       temperature=0.7, top_k=20, top_p=0.8 (from HF config)
                        → must override per request with temperature=0.0
MoE kernel:             WARNING — default config, sub-optimal expert dispatch
```

### Smoke Test Results — formula_1 DB (worst-case prompt)

```
Prompt tokens (actual):   1,206
Completion tokens:         56
Thinking triggered:        No
Wall clock (Python urllib): 2.50s   ← includes 2.1s first-call HTTP overhead

vLLM internal metrics (from /metrics):
  TTFT:      0.062s  (62ms  — prefill of 1,206 tokens, H100 is fast)
  ITL avg:   0.006s  (6.1ms/token — 0.3369s / 55 inter-token gaps)
  E2E:       0.399s  (399ms — vLLM's total compute time)

KV cache %: 0% (idle after request completed)
```

**SQL output (correct):**
```sql
SELECT r.fastestLapTime, d.forename, d.surname
FROM results r
JOIN drivers d ON r.driverId = d.driverId
WHERE r.fastestLapTime IS NOT NULL
ORDER BY r.fastestLapTime
LIMIT 1;
```

### Key Insight

The model compute is fast: **399ms E2E** for worst-case prompt (1,206 tokens in, 56 out). The 2.50s wall clock is dominated by Python urllib cold-start HTTP overhead, not the model. The real question is what happens under concurrency.

SLO math revised with real numbers:
- Single-request happy path (2 LLM calls × 400ms): **~800ms** — easily under 5s
- Under 10 RPS load with ~11–46 concurrent sequences: TTFT will grow as queue builds
- The binding constraint at load will be **concurrency + KV cache saturation**, not single-request speed

### Diagnosis

**Saw:** TTFT=62ms, ITL=6.1ms/token, vLLM E2E=399ms at zero concurrency. Model is fast in isolation.

**Hypothesized:** Under 10 RPS concurrent load, queue depth and KV cache will be the failure mode, not raw model speed. Need to measure under actual concurrency.

**Next change:** Run the smoke_test.py in parallel (5–10 concurrent requests) to observe TTFT degradation under load before committing to a full load test. Also need to test thinking mode impact — not yet tested with a complex multi-table question.

---

### Thinking Mode Test

**Script:** `scripts/test_thinking.py`
**Prompt:** formula_1 schema + complex multi-table question (1,218 tokens)

| Temperature | Wall time | Completion tokens | Thinking triggered |
|---|---|---|---|
| 0.7 (model default) | 0.62s | 88 | **No** |
| 0.0 | 0.54s | 83 | **No** |

**Thinking is NOT triggering via vLLM's OpenAI-compatible API.**

Why: vLLM startup showed `reasoning_backend=''` — the chat template think-enable token is not inserted by default through the API. The model never enters thinking mode through our requests.

Cumulative metrics after 3 total requests:
```
Avg TTFT:    54ms   (0.163s / 3)
Avg ITL:     6.1ms/token  (1.368s / 224 gaps — consistent)
Avg E2E:     510ms  (1.531s / 3)
KV cache:    0% throughout
Warm wall clock: 0.54–0.62s (vs 2.50s cold first-call)
```

**Implication:** No `/no_think` flag needed in prompts. Output lengths will stay short (80–100 tokens for SQL). This is good for latency — the decode phase stays bounded.

**Next:** Mini concurrent load test — 10 parallel requests to measure TTFT degradation under queue pressure.

---

### Concurrent Load Test — 10 Parallel Requests

**Script:** `scripts/concurrent_test.py`
**Prompt:** formula_1 schema × 10 different questions (~1,218 tokens each)
**Concurrency:** 10 simultaneous threads

**Per-request wall clock:**
```
[06] 1.09s  12 tokens  (shortest output → fastest)
[08] 1.50s  42 tokens
[03] 1.54s  47 tokens
[04] 1.63s  54 tokens
[00] 1.67s  57 tokens
[02] 1.70s  61 tokens
[07] 1.76s  68 tokens
[09] 1.81s  73 tokens
[05] 1.82s  75 tokens
[01] 1.85s  80 tokens  (most output tokens → slowest)
```

**Summary:**
```
P50 wall:    1.70s
P95 wall:    1.85s
Max wall:    1.85s
Min wall:    1.09s
Success:     10/10
Total wall:  1.85s  (all requests completed together — chunked prefill batching)
```

**vLLM metrics (incremental — 10 concurrent requests only):**
```
Avg TTFT:  679ms   (6.792s / 10) — vs 54ms single-request → 12.6× degradation from queue
Avg E2E:   1,439ms (14.387s / 10) — vs 510ms single-request → 2.8× degradation
```

**Why TTFT degraded 12.6×:** All 10 requests arrived simultaneously. vLLM's chunked prefill batches them, but each request must wait in queue for the others' prefill chunks before its own prefill runs. The last request in queue waited ~650ms before its first token.

**Output-token/latency correlation:** Visible in the results — longer SQL output = longer wall clock. Request [06] (12 tokens) finished in 1.09s; request [01] (80 tokens) took 1.85s. This confirms decode time (6.1ms × tokens) is the main differentiator when TTFT is shared across the batch.

**SLO implications:**
```
Happy path (2 LLM calls × 1.85s P95): 3.70s  → UNDER SLO ✓
With revise  (3 LLM calls × 1.85s P95): 5.55s → OVER SLO  ✗  (+0.55s)
```
At 10 concurrent — already borderline on revise path. Real load at 10 RPS will have
~35 concurrent LLM requests (Little's Law: 10 RPS × 3.5s avg agent latency = 35),
which is 3.5× today's test. TTFT will degrade further.

**Diagnosis:**
**Saw:** P95=1.85s at 10 concurrent. TTFT 12.6× worse than single-request. Revise path already at 5.55s.
**Hypothesized:** At true 10 RPS (~35 concurrent LLM requests), TTFT will blow up further and revise path will comfortably exceed 5s SLO. FP8 quantization is the primary lever — it frees ~30GB HBM for more KV cache AND doubles compute throughput, directly reducing TTFT and ITL.
**Next change:** Run the 10 RPS load test to verify. (Mini concurrent tested burst only — need sustained load.)

---

### Sustained 10 RPS Load Test — 120 seconds

**Script:** `scripts/vllm_load_test.py`
**Target:** vLLM directly (port 8000), bypassing agent (agent Phase 3 not implemented yet)
**Load:** 10 RPS × 120s = 1,200 total requests
**Prompts:** Real questions from `evals/eval_set.jsonl` with real schemas (376–1,338 tokens)

**Per-request wall clock summary:**
```
P50:   0.971s
P95:   1.895s  ← SLO boundary = 5.0s
P99:   2.285s
Max:   2.394s
```

**Success rate:** 1200/1200  |  0 timeouts  |  0 http_errors  |  achieved 9.93 RPS

**vLLM Prometheus metrics (post-test cumulative, 1213 total requests):**
```
TTFT avg:        52.2ms  (vs 62ms single-req  → similar, queue barely delayed prefill)
TTFT P95:        59.4ms  (histogram interpolation: 97.7% ≤ 60ms, 99.2% ≤ 80ms)
ITL  avg:        18.0ms/token  (vs 6.1ms single-req → 3.0× degradation from batch decode)
E2E  avg:        1,059ms  (vs 399ms single-req → 2.7× degradation)
Avg output:      55.8 tokens/request
KV cache peak:   ~14% est.  (10.59 concurrent × 1,256 tokens / 94,784 budget)
```

**Why TTFT stayed low but ITL tripled:**
- Chunked prefill allows vLLM to batch prefills in chunks — TTFT doesn't spike because prefill 
  chunks interleave with decode, keeping the queue from completely blocking.
- ITL (decode throughput) goes from 6.1ms → 18ms because at ~10 concurrent sequences in decode, 
  each forward pass generates 1 token for ALL sequences but takes ~180ms (vs 6ms × 1 = 6ms 
  for single sequence). Each sequence waits for ALL others in the batch → 3× per-token latency.
- KV cache is NOT the bottleneck at 10 RPS. Only 14% utilization — plenty of headroom.

**SLO analysis for the agent (once Phase 3 is built):**

| Agent path | LLM calls | Projected P95 | SLO (< 5.0s) |
|---|---|---|---|
| Happy path | 2 | 2 × 1.895 = 3.79s | **PASS** |
| Revise once | 3 | 3 × 1.895 = 5.69s | **FAIL** (+0.69s) |

But this projection is optimistic — it assumes 10 vLLM RPS. The agent serving 10 user RPS with
~2.5 avg LLM calls per user request = **25 effective vLLM RPS**. At 2.5× load, ITL will worsen
further, pushing both paths higher.

**Diagnosis:**
**Saw:** At 10 vLLM RPS, P95=1.895s. Revise path (3 calls) = 5.69s → SLO bust. KV cache NOT
the bottleneck (14% peak). Bottleneck is compute throughput (ITL 3× degraded from batching).

**Hypothesized:** FP8 quantization addresses both failure modes:
1. Halves weight memory (57GB → ~31GB), nearly doubles available KV budget
2. Increases compute throughput → lower ITL under batch decode
These together should cut ITL degradation under load and bring the revise path under 5s.

**Next change:** Enable `--quantization fp8` and re-run the 10 RPS load test to measure improvement.

*Note: FP8 is prescribed BECAUSE we observed a specific failure — revise path exceeds SLO under
load due to compute-bound ITL degradation. Not added speculatively.*

---

## Quick Reference

```bash
# Check metrics live
watch -n 2 'curl -s localhost:8000/metrics | grep -v "^#" | grep "vllm:" | grep -E "kv_cache|num_requests|e2e"'

# Tail vLLM logs (if running in background)
tail -f /tmp/vllm.log

# Re-run load test at different RPS
uv run python load_test/driver.py --rps 5 --duration 120 --out results/scratch.json

# Parse load test output
cat results/iter0_baseline.json | python3 -c "import json,sys; s=json.load(sys.stdin)['summary']; [print(f'{k}: {v}') for k,v in s.items()]"
```
