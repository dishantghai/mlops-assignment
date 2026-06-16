# Actual Run Log — HW3 LLM Inference Tuning

Operational log. Short entries. Saw → Hypothesized → Changed → Result format.
Educational detail lives in `mlops-hw3-runlog.md`.

---

## Config History

| Iter | Key Flags | TTFT P95 | E2E avg | Wall P95 | KV% | Notes |
|------|-----------|----------|---------|----------|-----|-------|
| 0 | no flags (BF16 defaults) | — | — | — | — | CRASH — KV OOM at startup |
| 1 | `--max-model-len 8192 --no-prefix-cache` BF16 | 59ms | 1,059ms | **1.895s** | ~14% | 1200/1200 ok — revise path FAILS SLO (5.685s) |
| 2 | + `--quantization fp8` | 59ms | 799ms | **1.498s** | ~2.5% | 1200/1200 ok — **revise path PASSES SLO (4.494s)** ✓ |

---

## Active Config

**File:** `scripts/start_vllm.sh`
**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`
**Current iteration:** Iter 2 (FP8)

```bash
# ITER 2 — FP8, capped context, prefix cache off
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --quantization fp8 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill
```

**Flag rationale:**
- `--max-model-len 8192`: mandatory — BF16 default (262144) requires 24 GiB KV, more than the 8.68 GiB available. 8192 gives 5× headroom over our workload max (~1640 tokens).
- `--quantization fp8`: halves weight footprint (57→29 GiB), frees 28 GiB for KV, reduces ITL under batch decode. Prescribed after Iter 1 confirmed revise path fails SLO by 0.69s.
- `--no-enable-prefix-caching`: off for clean measurement; Iter 3 will enable this.
- `--no-enable-chunked-prefill`: silently ignored for Qwen3MoE in vLLM 0.10.x (engine force-enables it regardless).

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

## Iteration 2 — FP8 Quantization

**Date:** 2026-06-16
**Change from Iter 1:** Added `--quantization fp8` to `scripts/start_vllm.sh`
**Hypothesis:** FP8 halves weight memory bandwidth → reduces ITL degradation under batch decode → revise path (3 calls) drops from 5.685s to under 5.0s SLO.

### Startup Facts

```
Weights:          29.08 GiB  (was 56.93 GiB BF16 → 51% smaller)
KV cache budget:  36.53 GiB → 399,040 tokens  (was 8.68 GiB → 94,784 → 4.2× more KV)
Load time:        ~14s  (vs ~273s BF16 — pre-quantized FP8 weights on HuggingFace)
Prefix caching:   OFF (confirmed)
Chunked prefill:  ON  (still force-enabled for Qwen3MoE, cannot disable)
MoE kernel:       WARNING — default config still (same as BF16, not FP8-specific)
```

### Single-Request Smoke Test (FP8, formula_1, 1206 tokens in, 57 out)

```
TTFT:   47.3ms  (was 62ms BF16 → 1.31× faster)
ITL:    6.1ms/token  (same as BF16 — at zero concurrency, compute is not bandwidth-limited)
E2E:    390ms  (was 399ms → essentially same at zero concurrency)
```

At zero concurrency, FP8 only improves TTFT (faster prefill from lower memory bandwidth for weights). ITL improvement requires concurrent load to be meaningful.

### Sustained 10 RPS Load Test — 120 seconds

**Script:** `scripts/vllm_load_test.py`
**Config:** Same as Iter 1 except `--quantization fp8`

**Per-request wall clock:**
```
P50:   0.712s  (was 0.971s → 1.37× faster)
P95:   1.498s  (was 1.895s → 1.27× faster)
P99:   1.811s  (was 2.285s → 1.26× faster)
Max:   2.261s
```

**Success rate:** 1200/1200  |  0 timeouts  |  0 http_errors  |  achieved 9.95 RPS

**vLLM Prometheus metrics (1201 cumulative, ≈1200 from load test):**
```
TTFT avg:    45.9ms  (was 52.2ms → 1.14× faster)
TTFT P95:    59.1ms  (was 59.4ms → same — already bottomed out by chunked prefill)
ITL  avg:    13.6ms/token  (was 18.0ms → 1.33× faster — the key improvement)
E2E  avg:    799ms  (was 1,059ms → 1.33× faster)
Avg output:  55.4 tokens/request
KV cache:    ~2.5%  (8.0 concurrent × 1,256 tokens / 399,040 budget)
```

**SLO verdict:**

| Agent path | LLM calls | Iter 1 BF16 | Iter 2 FP8 | SLO (< 5.0s) |
|---|---|---|---|---|
| Happy path | 2 | 3.790s | **2.996s** | **PASS** ✓ |
| Revise once | 3 | 5.685s ✗ | **4.494s** | **PASS** ✓ |

**The revise path flipped from FAIL to PASS. FP8 delivered 1.191s improvement on the revise path.**

**Why FP8 reduced ITL by 1.33×:**
The decode forward pass reads all weight matrices for every batch step. In BF16, weight reads = 57 GB × per-batch scan. In FP8, weight reads = 29 GB — half the memory bandwidth pressure. Each forward pass completes faster → each sequence in the decode batch gets its next token sooner → ITL decreases. This is pure memory-bandwidth relief, not FLOP reduction (the arithmetic is still full-precision for key computations, only weight storage is FP8).

**Concurrency change:**
- Iter 1 BF16: avg E2E 1.059s → Little's Law = 10 × 1.059 = 10.6 concurrent sequences
- Iter 2 FP8:  avg E2E 0.799s → Little's Law = 10 × 0.799 = 8.0 concurrent sequences
- Fewer concurrent sequences in decode → smaller batch → lower ITL → self-reinforcing improvement

**KV cache now trivially utilized (2.5% vs 14%).** FP8 unlocked 4.2× more KV budget, but we're not KV-constrained. The extra KV headroom is reserve capacity for future load increases or longer prompts.

**Diagnosis:**
**Saw:** ITL dropped 1.33×, wall P95 dropped 1.27×, revise path now under SLO at 10 vLLM RPS.
**Hypothesized:** The real agent will drive ~25 effective vLLM RPS (10 user RPS × 2.5 LLM calls). At 2.5× load, ITL will degrade further — but starting from 13.6ms (vs 18ms in BF16) gives more margin before hitting SLO. Worth building the agent and testing end-to-end before adding more flags.
**Next change:** Build Phase 3 agent (verify_node, revise_node), then run the full agent load test to see if the SLO holds at real 10 user RPS. If not, add `--enable-prefix-caching` as Iter 3 lever.

---

## Phase 2 — Observability

**Date:** 2026-06-16
**Objective:** Prometheus metrics inventory → Grafana dashboard (12 panels)

### Step 1: Metrics Inventory

```bash
curl localhost:8000/metrics | grep -v "^#" | sort
```

Captured post-Iter-2 load test (1201 cumulative requests, FP8 idle). Key findings:

| Category | Key Metrics | Idle values |
|---|---|---|
| Gauges | `kv_cache_usage_perc`, `num_requests_running`, `num_requests_waiting` | 0.0, 0.0, 0.0 |
| Counters | `prompt_tokens_total`, `generation_tokens_total`, `num_preemptions_total` | 945K, 67K, 0 |
| Histograms | `e2e_request_latency_seconds`, `time_to_first_token_seconds`, `time_per_output_token_seconds`, `request_queue_time_seconds` | avgs: 799ms, 46ms, 13.6ms, 3.7µs |

Zero preemptions across all tests — KV cache never stressed.  
`request_queue_time_seconds` avg = **3.7 microseconds** at 10 RPS FP8 — no backpressure.

Full metric catalogue with PromQL in `mlops-hw3-runlog.md`.

### Step 2: Grafana Dashboard Built

**File:** `infra/grafana/provisioning/dashboards/serving.json` (version: 2)
**uid:** `vllm-serving` | refresh: 5s | window: last 30m

12 panels across 5 rows:

| Row | Panels | Answers |
|---|---|---|
| SLO Health | E2E P50/P95/P99 (w=24, SLO threshold line at 5.0s red) | Is it slow? |
| Latency Decomp | TTFT P50/P95 (w=12), ITL P50/P95 (w=12) | Where is the slowness? |
| Queue & Memory | Requests Running/Waiting (w=8), Queue Wait P95 with thresholds (w=8), KV Cache % gauge 0-100 (w=8) | Do I have headroom? |
| Throughput | Request Rate stop/length (w=8), Gen+Prompt tokens/sec (w=8), Preemptions/sec bar (w=8) | GPU utilization |
| Quality Signals | Avg Output Tokens (thinking detector, threshold 300) (w=8), Prefix Cache Hit Rate (w=8), Avg Prompt Tokens (w=8) | Prompt drift & thinking mode |

**Histogram PromQL pattern:** `histogram_quantile(0.95, sum(rate(vllm:METRIC_bucket[2m])) by (le))`  
**KV cache panel:** `vllm:kv_cache_usage_perc * 100` (gauge, thresholds at 60/80/95%)

### Step 3: Dashboard Reload

```bash
curl -s -X POST http://admin:admin@localhost:3000/api/admin/provisioning/dashboards/reload
```

### Step 4: Live Verification — 60s × 10 RPS

Ran `uv run python scripts/vllm_load_test.py --rps 10 --duration 60` while watching all 12 Grafana panels.

```
All 12 panels populated with live data.

E2E P95 (Grafana):   1.614s  ← matches 1.498s wall P95 + ~120ms HTTP overhead
TTFT P95 (Grafana):  ~59ms   ← chunked prefill keeping this bounded
ITL avg (Grafana):   ~14ms   ← consistent with Iter 2 batch decode
KV Cache (Grafana):  ~2.5%   ← massive headroom confirmed
Queue depth:         0        ← no backpressure at 10 RPS FP8
Preemptions:         0        ← confirmed across all load tests
```

**Readable-cold test: PASSED.** A cold reader can identify slow/healthy, prefill vs decode bottleneck, and capacity headroom without any explanation.

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

---

## Phase 3 — LangGraph Agent: verify + revise Nodes

**Date:** 2026-06-16
**Files changed:** `agent/graph.py`, `agent/prompts.py`

### What was implemented

| Item | Description |
|------|-------------|
| `GENERATE_SQL_SYSTEM/USER` | SQL expert prompt; output raw SQL only; `{schema}` + `{question}` placeholders |
| `VERIFY_SYSTEM/USER` | Judge prompt; output only `{"ok": bool, "issue": str}`; lists specific failure cases |
| `REVISE_SYSTEM/USER` | Fix prompt; takes failing SQL + execution result + issue; output raw corrected SQL |
| `_parse_verify()` | Defensive JSON extractor — handles fences, string booleans, prose fallback |
| `verify_node` | Calls LLM with VERIFY prompts; feeds `execution.render()`; returns `{verify_ok, verify_issue}` |
| `revise_node` | Calls LLM with REVISE prompts (SQL + result + issue + schema); bumps `iteration`; appends to `history` |
| `route_after_verify` | `"end"` if `verify_ok` or `iteration >= MAX_ITERATIONS`, else `"revise"` |

All prompts prefix `/no_think` in the system message to suppress Qwen3 reasoning mode.

### Level 2 Test — Python Direct Invocation

**Question:** "What is the average number of students enrolled in schools in Los Angeles?"
**DB:** `california_schools`

```
Step 1  generate_sql  → SELECT AVG(s.Enrollment) ...  [s.Enrollment does not exist]
Step 2  verify        → ok=False, issue: "The column 'Enrollment' does not exist in
                        the schools table; the query should reference the correct column
                        name for student enrollment."
Step 3  revise        → SELECT AVG(f."Enrollment (K-12)") ... JOIN frpm f ...
Step 4  verify        → ok=True

iterations: 2  |  verify_ok: True  |  answer: 691.66 avg students
```

Loop triggered on a wrong column reference, produced a precise issue string, and revise fixed it on the first attempt.

### Level 4 Test — HTTP via /answer Endpoint

```bash
# Terminal 1
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

# Terminal 2
curl -s -X POST http://localhost:8001/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the average number of students enrolled in schools in Los Angeles?", "db": "california_schools"}'
```

Response confirmed: `iterations: 2`, `ok: true`, `rows: [[691.66]]`. HTTP endpoint working end-to-end.

### SLO Projection for Agent

Each agent run makes 2–3 sequential LLM calls. From Iter 2 FP8 per-call wall P95 = 1.498s (at 10 vLLM RPS):

| Path | Calls | Projected agent P95 | SLO (< 5.0s) |
|------|-------|---------------------|--------------|
| Happy path (no revise) | 2 | 2 × 1.498 = **2.996s** | PASS ✓ |
| Revise once | 3 | 3 × 1.498 = **4.494s** | PASS ✓ |

These are projections at 10 vLLM RPS. The real agent drives ~25 effective vLLM RPS (10 user RPS × ~2.5 LLM calls/run). Phase 6 end-to-end load test will measure the actual number.
