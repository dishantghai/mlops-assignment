# Session State — HW3 Inference Tuning
# Last updated: 2026-06-16

## Where We Are

Completing Phase 1 (vLLM config baseline). Phase 3 (agent) not yet implemented.
Current task: run 10 RPS load test against vLLM DIRECTLY (bypassing agent)
to observe where Iter 1 baseline breaks before adding any optimizations.

---

## File Map

| File | Purpose |
|------|---------|
| `project_ops/actual_run_log.md` | OPERATIONAL log — short entries, iteration results, saw/hypothesized/changed/result |
| `project_ops/mlops-hw3-runlog.md` | EDUCATIONAL log — verbose, architecture analysis, issue explanations |
| `project_ops/LEARNING_GUIDE.md` | Assignment guide — what to build and measure |
| `project_ops/SOLUTIONS_REFERENCE.md` | Reference answers — check after attempting |
| `scripts/start_vllm.sh` | vLLM launch script — currently Iter 1 config |
| `scripts/smoke_test.py` | Single request test — formula_1 schema, measures wall+usage |
| `scripts/test_thinking.py` | Thinking mode test — 2 requests at temp 0.7 and 0.0 |
| `scripts/concurrent_test.py` | 10 parallel requests — measures P50/P95 wall clock |
| `agent/graph.py` | LangGraph agent — generate_sql_node done, verify/revise NOT IMPLEMENTED |
| `agent/server.py` | FastAPI /answer endpoint — ready but unusable until Phase 3 done |
| `agent/schema.py` | render_schema(db_id) — reads SQLite, returns CREATE TABLE text |
| `agent/prompts.py` | Prompt templates — ALL EMPTY, need Phase 3 implementation |
| `load_test/driver.py` | Load test driver — hits /answer endpoint at configurable RPS |
| `evals/eval_set.jsonl` | 30 eval questions across 9 BIRD databases |

---

## Current vLLM Config — Iter 1 (scripts/start_vllm.sh)

```bash
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill
```

Notes:
- `--no-enable-prefix-caching` works (confirmed in logs)
- `--no-enable-chunked-prefill` is IGNORED by vLLM 0.10.x for Qwen3MoE — chunked prefill stays ON
- BF16, no quantization
- vLLM auto-default: gpu_memory_utilization=0.9

---

## Measured Baseline Numbers (Iter 1)

### Single request (formula_1 schema, 1,206 tokens in, 56 out)
- Wall clock: 2.50s (first call, cold HTTP) / 0.54–0.62s (warm)
- TTFT: 62ms
- ITL: 6.1ms/token
- vLLM E2E: 399ms
- KV cache: 0%

### 10 concurrent requests (formula_1 schema, ~1,218 tokens each)
- P50 wall: 1.70s | P95: 1.85s | Max: 1.85s
- Avg TTFT: 679ms (12.6× degradation from queue pressure)
- Avg vLLM E2E: 1,439ms
- KV cache: 0% (never stressed at this concurrency)

### Thinking mode test
- NOT triggered at temperature=0.7 OR 0.0
- vLLM startup shows `reasoning_backend=''` — think token not inserted by chat template
- No `/no_think` needed in prompts
- Output stays 80–90 tokens (pure SQL)

### vLLM startup facts
- Weights: 56.93 GiB (BF16)
- KV cache budget: 8.68 GiB → 94,784 total tokens
- Max concurrency (worst case, 8192 tokens): 11.57x
- Max concurrency (real workload, ~1,200 tokens): ~79x
- Default sampling from HF config: temperature=0.7, top_k=20, top_p=0.8 (override per request)
- MoE kernel WARNING: config file missing for H100_80GB_HBM3 — using default, sub-optimal

---

## Setup Issues Resolved

1. **git identity** — `git config --global user.email/name` on VM
2. **transformers 5.9.0 incompatible with vLLM 0.10.2** — fixed by adding
   `"transformers>=4.45.0,<5.0.0"` to pyproject.toml, then `uv lock --upgrade-package transformers && uv sync`
3. **Iter 0 crash** — BF16 weights (57GB) + default max_model_len=262144 requires 24GB KV → only 8.68GB available. Fixed with `--max-model-len 8192`.

---

## Immediate Next Task

Write a **direct vLLM load test** (NOT using the agent /answer endpoint — Phase 3 not ready).

Script to create: `scripts/vllm_load_test.py`

What it should do:
- Use `asyncio` + `aiohttp` (like load_test/driver.py)
- Fire requests directly at `http://localhost:8000/v1/chat/completions`
- Sample questions from `evals/eval_set.jsonl` with corresponding schemas
- Target: 10 RPS for 120 seconds (shorter than the 300s agent load test)
- Record per-request latency
- At the end: print P50/P95/P99/max and pull /metrics for TTFT, ITL, KV cache
- Write output to `results/iter1_vllm_load.json`

Why direct vLLM not agent: agent needs Phase 3 (verify/revise) before it can serve.
This gives us the vLLM-level SLO picture which is the real bottleneck anyway.

---

## Iteration Plan (what comes after the load test)

| Iter | Change | What to measure | Trigger condition |
|------|--------|-----------------|-------------------|
| 1 | Baseline (BF16, no opts) | P95 under load, TTFT, KV% | ← RUNNING NOW |
| 2 | `--quantization fp8` | Same metrics | After observing Iter 1 failure |
| 3 | Add `--enable-prefix-caching` | Cache hit rate, TTFT improvement | After fp8 baseline |
| 4 | Tune `--max-num-seqs` | Queue depth, KV% | After seeing saturation |
| 5 | Full recommended config | Final SLO check | After individual flags measured |

---

## SLO Target
- P95 E2E < 5.0s at 10 RPS sustained

## Key SLO Math (from Iter 1 data)
- Single call: 399ms vLLM E2E
- At 10 concurrent: 1,439ms avg E2E (3.6× slowdown)
- Happy path (2 calls × 1.85s P95): 3.70s → under SLO
- Revise path (3 calls × 1.85s P95): 5.55s → over SLO
- Real 10 RPS steady state: ~35 concurrent LLM requests (Little's Law) → worse than 10-concurrent test

## tmux Sessions
- `vllm` — vLLM server (currently running Iter 1)
- `cmd` — command runner

## Git State
- Branch: main
- Last commit: 6ddb828 — concurrent test results
- All changes pushed to origin
