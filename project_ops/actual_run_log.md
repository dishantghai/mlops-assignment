# Actual Run Log — HW3 LLM Inference Tuning

Operational log. Short entries. Saw → Hypothesized → Changed → Result format.
Educational detail lives in `mlops-hw3-runlog.md`.

---

## Config History

| Iter | Key Flags | TTFT P95 | E2E P95 | KV Cache % | Queue Peak | Notes |
|------|-----------|----------|---------|------------|------------|-------|
| 0 (baseline) | no flags | — | — | — | — | pending run |

---

## Active Config

**File:** `scripts/start_vllm.sh`
**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`

```bash
# ITER 0 — BASELINE (no optimization flags)
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000
```

---

## SLO Target

| Metric | Target |
|--------|--------|
| E2E P95 latency | < 5.0s |
| Sustained RPS | 10 |
| Load test duration | 300s |

---

## Iteration 0 — Baseline

**Date:** 2025-06-16
**Config:** No optimization flags. vLLM defaults.
**Goal:** Establish raw baseline before any tuning.

### Manual Smoke Test (3–5 queries before load test)

Commands to run on VM:
```bash
# 1. Start vLLM
bash scripts/start_vllm.sh

# 2. Wait for "Application startup complete" in logs, then:
curl -s localhost:8000/metrics | grep -E "^vllm:(e2e|time_to_first|inter_token|kv_cache|num_requests)" | grep -v "^#"

# 3. Fire one manual query
curl -s http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "List down Ajax superpowers.", "db": "superhero"}' | jq .

# 4. Read metrics again after query
curl -s localhost:8000/metrics | grep -v "^#" | grep "vllm:" | sort
```

### Smoke Test Results

| Query | DB | Latency | SQL Correct? | Iterations |
|-------|----|---------|--------------|------------|
| | | | | |
| | | | | |
| | | | | |

Thinking mode triggered? [ ] Yes [ ] No
(Check: did response include `<think>` tokens or very long output?)

### vLLM Metrics After Smoke Test

```
TTFT P50:   ___s
TTFT P95:   ___s
ITL P50:    ___ms
KV cache %: ___%
```

### Load Test

```bash
# Run from project root on VM
uv run python load_test/driver.py --rps 10 --duration 300 --out results/iter0_baseline.json
```

### Load Test Results

```
P50 E2E:        ___s
P95 E2E:        ___s    [SLO: 5.0s — MISS / HIT]
P99 E2E:        ___s
Achieved RPS:   ___
Timeouts:       ___
KV cache peak:  ___%
Queue peak:     ___
Tokens/sec:     ___
```

### Grafana Observations

- [ ] KV cache utilization chart captured (screenshot: `screenshots/iter0_kv_cache.png`)
- [ ] E2E P95 chart captured (screenshot: `screenshots/iter0_e2e_latency.png`)
- [ ] Queue depth chart captured

Notable patterns:
```
___
```

### Diagnosis

**Saw:**

**Hypothesized:**

**Next change:**

---

## Iteration 1 — (TBD after baseline)

*(Fill after Iter 0 results are in.)*

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
