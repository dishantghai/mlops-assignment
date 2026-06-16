# Actual Run Log — HW3 LLM Inference Tuning

Operational log. Short entries. Saw → Hypothesized → Changed → Result format.
Educational detail lives in `mlops-hw3-runlog.md`.

---

## Config History

| Iter | Key Flags | TTFT P95 | E2E P95 | KV Cache % | Queue Peak | Notes |
|------|-----------|----------|---------|------------|------------|-------|
| 0 | no flags (BF16 defaults) | — | — | — | — | CRASH — KV OOM at startup |
| 1 | `--max-model-len 8192 --no-prefix-cache` (chunked prefill forced on) | 62ms | 399ms (vLLM) / 2.50s (wall) | 0% | 0 | single-request only, no load yet |

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
