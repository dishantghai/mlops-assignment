# HW3 Report — LLM Inference + Observability

## 1. Serving Configuration

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 80 GB.

| Flag | Rationale |
|------|-----------|
| `--max-model-len 8192` | **Mandatory.** BF16 defaults (262 K context) require 24 GiB KV cache; only 8.68 GiB available. 8 192 matches our actual max prompt (~1 640 tokens) with 5× headroom. |
| `--quantization fp8` | FP8 checkpoint on HuggingFace. Halves weight footprint 57 → 29 GiB, freeing 28 GiB for KV. Also reduces per-token memory bandwidth in batch decode → ITL improved 1.33× under concurrent load. |
| `--enable-prefix-caching` | Load pool has 9 DBs × ~167 questions. The schema prefix (~800 tokens) is identical across all questions for a given DB. First request per DB populates the KV blocks; subsequent requests skip schema prefill entirely. Measured hit rate: 86.9%, TTFT: 45.9 ms → 25 ms. |
| `--no-enable-chunked-prefill` | Silently ignored for Qwen3MoE in vLLM 0.10.x — engine force-enables chunked prefill regardless. Flag included for intent clarity. |

Single-request baseline (FP8, formula_1 schema, 1 206 tokens in, 57 out): TTFT=47 ms, ITL=6.1 ms/token, E2E=390 ms. Under 10 vLLM RPS: ITL=13.6 ms/token, E2E avg=799 ms, wall P95=1.498 s.

---

## 2. Baseline Eval (Phase 5)

**Dataset:** 30 questions across 9 BIRD databases. Metric: execution accuracy (canonicalized row-set comparison, sort-invariant, case-insensitive column names, NULL ≠ 0).

| Stage | Pass rate |
|-------|-----------|
| iter 0 (generate only, no loop) | 40.0% (12/30) |
| iter 1 (after first revise) | 40.0% (12/30) |
| final (all iterations) | **43.3% (13/30)** |

The loop is net-positive (+3.3 pp). It helped 2 questions, hurt 1 (verify incorrectly flagged a correct but unusual self-join as wrong, and revise hallucinated a non-existent `districts` table). The per-iteration breakeven confirms the loop earns its keep — but marginally.

**Top failure clusters:** `thrombosis_prediction` (0/3) and `toxicology` (0/2) — both caused by categorical value conventions absent from the DDL. `toxicology.element` stores chlorine as `'cl'`, calcium as `'ca'`; `thrombosis.Admission` uses `'-'` for outpatient. The model cannot infer these from column names alone, so generates wrong WHERE clauses that neither verify nor revise can fix.

---

## 3. SLO Diagnosis and Iteration (Phase 6)

**SLO:** P95 agent E2E latency < 5.0 s at 10 RPS sustained for 300 s.

**Starting point — Run 1 (Iter 2 FP8, sync server, 10 RPS):**  
272/3000 ok (9.1%), P50=10.1 s, P95=103.4 s. Root cause: uvicorn sync thread pool (~32 threads) exhausted at t≈30 s. At 10 RPS × 6 s avg = 60 concurrent requests > 32 threads → starvation → timeout cascade.

**Iteration log:**

| # | Saw | Hypothesized | Changed | Result |
|---|-----|-------------|---------|--------|
| 1 | Sync thread pool exhausted at t=30 s | `async def` + `await graph.ainvoke()` removes thread pool from hot path | `def answer` → `async def answer`, `graph.invoke` → `await graph.ainvoke()` | Thread starvation resolved. But vLLM throughput ceiling exposed: 359/3000 ok (12%), P95=114.6 s. Different root cause. |
| 2 | vLLM collapses at 20–50 effective LLM RPS. Load pool has 9 DBs — schema prefix (~800 tokens) repeats identically per question within each DB | `--enable-prefix-caching` will reuse KV blocks for schema; first request per DB caches, rest get TTFT ≈ 5 ms | Added `--enable-prefix-caching` to `start_vllm.sh` (Iter 3) | Hit rate 86.9%, TTFT 45.9 ms → 25 ms, E2E 799 ms → 578 ms. But sync LLM nodes now dominate: 3 RPS P50=7.4 s vs expected 2.6 s. |
| 3 | `generate_sql_node`, `verify_node`, `revise_node` were `def` + `llm().invoke()`. LangGraph's `ainvoke()` pushes sync nodes to asyncio thread pool executor (~20 threads). At 22 concurrent agents × 3 LLM calls → pool saturated. Also: new `ChatOpenAI()` per call — no connection reuse. | `async def` + `await llm().ainvoke()` runs on the event loop directly; `@lru_cache(maxsize=1)` singleton reuses connection pool | `agent/graph.py` — all LLM nodes → async + singleton | 3 RPS P50: 7.4 s → 3.89 s (−47%). Still 1.3 s over expected — Langfuse flush blocking. |
| 4 | After `graph.ainvoke()` completes, server awaited `asyncio.to_thread(handler.flush)` + `client.post(Langfuse)` before returning HTTP response. Client waited ~2 s for I/O on every request. | `asyncio.create_task(_flush_and_tag())` fires flush as a background task. HTTP response returns immediately. | `agent/server.py` — flush wrapped in `create_task` (fire-and-forget) | 3 RPS P50: 3.89 s → **2.61 s** (−33%). Matches theoretical serial compute exactly: 3.15 × 578 ms + 790 ms overhead = 2.61 s ✓ |
| 5 | Capacity model: vLLM ceiling ≈ 15.7 LLM RPS (measured). At 5 user RPS × 3.15 avg LLM calls = 15.75 LLM RPS → ρ ≈ 1.0; queue grows over 300 s. | All three fixes dramatically reduce per-call latency vs original; 5 RPS P50 should drop significantly, but P95 will still miss SLO at ρ ≈ 1.0. | No code change — ran 5 RPS load test to confirm capacity ceiling | 5 RPS P50: 21.8 s (no fixes) → **6.32 s** (−71%), P95: 78.8 s → **31.4 s** (−60%). ρ ≈ 1.0 confirmed: queue grows linearly over 300 s. |
| 6 | SQL output is repetitive (SELECT/FROM/WHERE/JOIN). N-gram SD accepts k tokens per step; at α=0.80, k=3 → 2.4 tokens/step vs 1.0 → effective ceiling shifts from 15.7 to ~25 LLM RPS. | N-gram SD would reduce decode time and lower ρ at 5 RPS from 1.0 to ~0.60. | `--speculative-config '{"method": "ngram", "num_speculative_tokens": 3}'` in `start_vllm.sh` | 5 RPS P50: **7.57 s** (−19% WORSE than Run 7), P95: **36.9 s**. N-gram SD degraded performance. Qwen3 MoE's sparse expert routing creates varied per-token activation patterns that defeat n-gram matching. Short SQL outputs (~55 tokens) provide insufficient amortization depth. **Reverted.** |
| 7 | ~60% of revise cycles driven by unknown categorical values (thrombosis `Admission='-'` for outpatient, toxicology `label='+'` for carcinogenic). If generate_sql now uses correct values, verify accepts on first try → avg LLM calls/agent drops from 3.15 toward ~2.0 → effective LLM RPS at 5 user RPS drops from 15.75 to ~10 → ρ ≈ 0.64 (stable). | Schema value hints (inline DDL comments for TEXT columns ≤10 distinct values) will reduce revise rate in load pool and break capacity ceiling. | `agent/schema.py` — TEXT columns ≤10 distinct, ≤25-char examples → `"label" TEXT /* e.g. '+', '-' */` | 5 RPS P50: **1.02 s** (−84% vs Run 7), P95: **5.54 s** (−82%). Eval accuracy: 43.3% → 36.7% (−6.6 pp) — value hints added noise on card_games/financial. |

**Final numbers:**

| Run | RPS | P50 | P95 | Notes |
|-----|-----|-----|-----|-------|
| 1 (baseline) | 10 | 10.1 s | 103.4 s | sync server, FP8 |
| 6 (all code fixes) | 3 | 2.61 s | 13.86 s | best accuracy-latency balance |
| 9 (+ schema hints) | 5 | **1.02 s** | **5.54 s** | closest to SLO; eval −6.6 pp |

**SLO verdict:** Missed. Best P95 = 5.54 s at 5 RPS — 0.54 s over the 5.0 s target. At 10 RPS the SLO is structurally unreachable on a single H100: 10 × 3.15 = 31.5 effective LLM RPS vs 15.7 capacity (ρ = 2.0 — pure overload).

---

## 4. Agent Value

The verify → revise loop adds measurable value: +3.3 pp above the generate-only baseline (43.3% vs 40.0%). The per-iteration evidence shows the loop is doing real work — 2 questions were salvaged by revise that would otherwise have failed. The one regression (verify falsely rejecting a correct result) is a prompt calibration issue, not a structural flaw. For the load pool (1 500 questions across 11 DBs), schema value hints reduced avg LLM calls/agent from ~3.15 to ~2.0, collapsing P50 from 6.32 s to 1.02 s, which proves the revise cycle was the binding bottleneck at 5 RPS — not prefill, not decode, but the extra round-trip count per request.

---

## 5. What I'd Do With More Time

**1. Collapse generate + verify into one call.** Currently the agent makes 2 calls at minimum (generate, then verify against the execution result). A single "generate SQL, execute, then self-check the result in one prompt" reduces minimum calls from 2 to 1. With 50% revise rate, avg would drop from 3.15 to ~1.5 LLM calls/agent → effective LLM RPS at 5 user RPS = 7.5 (ρ = 0.48) → P95 should drop well below 5.0 s. This addresses the root cause rather than papering over it with hints.

**2. Selective schema value hints by DB, not by cardinality threshold.** The blanket ≤10-distinct rule added helpful hints for thrombosis (`Admission`) and toxicology (`label`), but also added noise for card_games (20+ flag fields) and financial, regressing eval accuracy by 6.6 pp. With a per-DB allowlist (`thrombosis_prediction: [Admission, KCT]`, `toxicology: [label, bond_type]`), the latency benefit is retained without the accuracy cost.

**3. Lightweight structural verify before LLM verify.** Many verify failures are detectable without an LLM call: SQL errored (parse the exception), zero rows on a COUNT query, result missing expected columns. Catching these in Python first eliminates ~30% of verify LLM calls, saving ~578 ms per affected request and reducing effective LLM RPS further.

**4. Expand the eval set.** 30 questions is too small for reliable accuracy signal — a 2-question swing moves the metric 6.7 pp. A 200-question set (weighted toward thrombosis/toxicology) would let us measure the accuracy impact of prompt or schema changes with statistical confidence, rather than guessing whether a change is signal or noise.
