# Actual Run Log — HW3 LLM Inference Tuning

Operational log. Short entries. Saw → Hypothesized → Changed → Result format.
Educational detail lives in `mlops-hw3-runlog.md`.

---

## Config History

| Iter | Key Flags | TTFT P95 | E2E P95 | KV Cache % | Queue Peak | Notes |
|------|-----------|----------|---------|------------|------------|-------|
| 0 | no flags (BF16 defaults) | — | — | — | — | CRASH — KV OOM at startup |
| 1 | `--max-model-len 8192` | — | — | — | — | pending |

---

## Active Config

**File:** `scripts/start_vllm.sh`
**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`

```bash
# ITER 1 — min viable baseline (BF16, capped context)
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

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

### Results

```
Weights loaded:     56.93 GiB (BF16 — confirmed)
KV cache budget:    ___ GiB
Max seqs possible:  ___

Smoke test latency: ___s (wall clock)
Thinking triggered: [ ] Yes  [ ] No
Output tokens:      ___

TTFT P50:   ___s
TTFT P95:   ___s
KV cache %: ___% (after 1 query)
```

### Diagnosis

**Saw:**

**Hypothesized:**

**Next change:**

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
