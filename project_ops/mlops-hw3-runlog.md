## Phase 0 Setup Log

```
Date/Time: 16 June 15:58 IST
VM hostname: dishant-ghai-instance

Port forwards established: [x] 3000  [x] 9090  [x] 3001  [x] 8000  [x] 8001
Method used: [x] VSCode Remote-SSH  [ ] Plain SSH

Services verified:
[x] Prometheus accessible at localhost:9090
[x] Grafana accessible at localhost:3000
[x] Langfuse accessible at localhost:3001, project created, keys saved to .env
[x] BIRD data present at data/bird/ (9 .sqlite files for eval DBs)
[x] .env created from .env.example

Issues encountered: None
Resolution: None
```

---

## Phase 1 Pre-Work: Understanding the Workload Before Touching a Flag

*Measured from actual eval_set.jsonl (30 questions across 9 databases) and agent/schema.py output.*

---

### Q1: What is the prompt length distribution — and what does it mean for prefill time and KV cache pressure?

**Measured schema sizes across all 9 eval databases:**

| Database | Tables | Schema chars | Est. tokens |
|---|---|---|---|
| toxicology | 4 | 708 | ~200 |
| thrombosis_prediction | 3 | 1,323 | ~380 |
| superhero | 10 | 1,709 | ~490 |
| student_club | 8 | 1,832 | ~525 |
| financial | 8 | 2,238 | ~640 |
| california_schools | 3 | 2,367 | ~680 |
| codebase_community | 8 | 2,616 | ~750 |
| card_games | 6 | 3,093 | ~885 |
| formula_1 | 13 | 4,066 | ~1,165 |

Token estimates use ~3.5 chars/token for mixed SQL+English content (SQL identifiers tokenize denser than prose).

**Full prompt breakdown (schema + question + system prompt overhead ~150 tokens):**

- Min observed: ~376 tokens (toxicology DB + short question)
- Max observed: ~1,338 tokens (formula_1 DB + long question)
- Average: ~846 tokens
- P90: ~1,333 tokens

The assignment's stated 1,500–3,000 token range reflects the broader BIRD dataset, which includes databases with 10–20+ tables and verbose column naming. Our specific 9 eval DBs sit at the lower end of that range. formula_1 (13 tables) is the closest to the upper bound. The verify and revise calls will be slightly longer because they also include the prior SQL, execution result rows, and error text — roughly adding 100–400 tokens on top.

**What this means for prefill time:**

Prefill is compute-bound: the GPU processes all input tokens in a single forward pass before any output token is generated. With Qwen3-30B-A3B's MoE architecture (~3B active parameters per forward pass), prefill is faster than a dense 30B model — but it still scales linearly with token count. A 1,200-token prompt takes roughly 3× the prefill compute of a 400-token prompt.

Concrete example: if a single generate_sql call for the formula_1 DB (1,165 token schema + question) takes 800ms in prefill, the same call for toxicology (200 token schema) might take only 200ms. At 10 RPS with many formula_1 questions, prefill is the dominant component of TTFT.

**What this means for KV cache pressure:**

The KV cache stores key-value attention tensors for every token in every active sequence. Crucially, KV cache size depends on the attention layer count and the sequence length — NOT on the FFN layer count. This means the MoE sparsity of Qwen3-30B-A3B gives no KV cache benefit: we still have full attention over all input tokens. A 1,200-token sequence occupies exactly 3× the KV cache of a 400-token sequence.

At 10 agent RPS with avg ~850-token prompts plus ~80-token outputs = ~930 tokens/sequence, and ~2-3 LLM calls per agent run, each active sequence holds roughly 1,000 tokens in KV. At 50 concurrent agent runs (see Q3), we need space for 50 sequences × ~1,000 tokens each in KV cache. This is why `--gpu-memory-utilization` and `--max-model-len` are the first levers to reason about: they directly control how many sequences can coexist in HBM.

---

### Q2: What is the output length distribution — and how does short output affect prefill vs decode ratio?

**Measured from gold SQL in eval_set.jsonl:**

| Percentile | Output tokens (est.) |
|---|---|
| Min | ~29 tokens |
| P50 | ~55 tokens |
| P90 | ~97 tokens |
| Max | ~180 tokens |

The shortest gold SQL: `SELECT COUNT(Id) FROM badges WHERE Name = 'Commentator' AND STRFTIME('%Y', Date) = '2014'` — about 29 tokens.

The longest gold SQL is a complex CASE/INSTR expression for lapTimes ordering — about 180 tokens. Even this extreme case is short in absolute terms.

**The prefill-to-decode ratio:**

End-to-end latency = TTFT (prefill) + N_output_tokens × ITL (decode per token)

With avg prompt ~850 tokens (prefill) and avg output ~62 tokens at ITL ~30ms/token (typical for a MoE model on H100):
- TTFT ≈ prefill_time (dominant)
- Decode contribution ≈ 62 × 30ms = ~1,860ms

With P90 output of 97 tokens: decode ≈ 97 × 30ms = ~2,910ms

So for a 2s total target per LLM call (derived from the SLO in Q4):
- Prefill budget: ~200–500ms (leaving room for decode)
- Decode budget: ~1,500–1,800ms (covers P90 output)

**The key insight:** TTFT is the dominant lever for tail latency. When queue depth rises and new requests wait before their prefill starts, that wait time adds directly to TTFT. Decode is relatively predictable since output lengths are short and bounded. This makes `--enable-chunked-prefill` worth investigating: long prompts (formula_1 at 1,165 tokens) running their full prefill can block decode steps for concurrent requests.

Short outputs also mean speculative decoding is unlikely to help much. The benefit of spec decoding is amortized across many output tokens; at 50–100 tokens, the overhead of draft model + verification likely washes out any gain.

---

### Q3: What is the actual vLLM request rate when the agent SLO requires 10 RPS?

**The layered request math:**

The SLO is 10 agent-level RPS. Each agent run calls vLLM 2–3 times sequentially:
1. `generate_sql` → always called (1 vLLM call)
2. `execute` → no vLLM call (SQLite only)
3. `verify` → always called (1 vLLM call)
4. `revise` → called only when verify fails (1 vLLM call, loops back to execute → verify)

With MAX_ITERATIONS = 3 and a realistic revision rate of ~30% of questions:
- Avg LLM calls per agent run ≈ 2 (generate + verify) + 0.3 × 1 (revise) ≈ 2.3 calls/run

**vLLM-level RPS:**
```
10 agent RPS × 2.3 avg LLM calls = ~23 vLLM requests/second
```

In the worst case (all runs hit the 3-iteration cap):
```
10 agent RPS × 3 LLM calls × 2 execute loops = ~50 vLLM requests/second
```

But there's a more important calculation: **concurrency**, not just RPS.

Using Little's Law: `N_concurrent = λ × W` where λ = arrival rate and W = average time per request.

If avg LLM call latency is 2s:
- Agent run time: 2.3 calls × 2s = ~4.6s average
- Concurrent agent runs: 10 RPS × 4.6s = **46 concurrent agent runs**
- Concurrent vLLM sequences: since calls are sequential within each run, ~46 sequences in flight at any moment

This 46-sequence concurrency number is what sets your `--max-num-seqs` floor and KV cache headroom requirement. Setting `--max-num-seqs` below 46 creates a hard ceiling that causes queueing even if the GPU is not saturated. Setting it too high without enough KV cache causes preemption.

**Per-database prefix reuse opportunity:**

Our eval set has 9 databases and 30 questions. Distribution:
- codebase_community: 5 questions
- formula_1: 4 questions
- student_club, california_schools, financial, card_games, thrombosis_prediction, superhero: 3–4 each
- toxicology: 2 questions

Multiple requests sharing the same DB target share an identical system prompt + schema prefix. With prefix caching enabled (`--enable-prefix-caching`), the second through Nth request for formula_1 would reuse the KV blocks for the 1,165-token schema — turning that prefill cost into a cache lookup. This is one of the strongest optimization opportunities for this specific workload.

---

### Q4: What is the SLO structure — and what ceiling does it put on per-LLM-call latency?

**The SLO: P95 end-to-end < 5 seconds.**

"End-to-end" means from the moment the HTTP POST hits `/answer` to when the response comes back. It includes: schema rendering, all LLM calls, all SQLite executions, and HTTP overhead. SQLite execution is sub-millisecond (local reads), and schema rendering is cached via `@lru_cache`. The budget is entirely dominated by LLM calls.

**Latency budget arithmetic:**

Happy path (no revise, 2 LLM calls):
```
budget_per_call = 5.0s / 2 calls = 2.5s max per call at P95
```

With one revision (3 LLM calls):
```
budget_per_call = 5.0s / 3 calls = 1.67s max per call at P95
```

At 10 RPS sustained over 5 minutes (the load test duration from driver.py), the P95 is measured over 3,000 total agent requests (10 × 300s). The 5th-percentile tail must stay under 5s — meaning 2,850 of 3,000 requests complete under 5s, and 150 can be slower.

**The ceiling this puts on configuration choices:**

If you see TTFT averaging 1.5s under load (plausible for 800-token prefills with moderate queue depth), and ITL is 30ms × 60 output tokens = 1.8s, a single LLM call takes 3.3s — already over the 2.5s budget for a 2-call happy path.

This means you need either:
- Sub-1s TTFT under concurrent load (requires keeping queue depth near zero), OR
- Sub-20ms ITL (requires very fast decode, achievable with FP8 on H100), OR
- Both, with some margin

**The math that tells you the SLO is genuinely challenging:**

At 10 RPS with avg 4.6s run time, you have 46 concurrent agent runs = 46 concurrent vLLM sequences. Each sequence holds ~950 tokens in KV. On Qwen3-30B-A3B in BF16, the KV cache per token is `2 × n_layers × d_head × n_heads × 2 bytes`. For Qwen3 30B MoE: 64 attention layers, 8 KV heads, 128 head dim → 64 × 8 × 128 × 2 × 2 = 262,144 bytes = ~256KB per token. At 50 sequences × 950 tokens × 256KB = 12GB just for KV cache. With model weights at ~60GB (BF16), the H100's 80GB is extremely tight. FP8 quantization is not just nice-to-have — it may be necessary to make the concurrency math work at all.

**Practical implication for Phase 6:**

When P95 misses the SLO, the first diagnostic questions are:
1. Is TTFT high? → Requests are waiting in queue or prefill is slow (look at queue depth and KV cache %)
2. Is ITL high? → Decode is saturated (look at tokens/sec generation rate)
3. Are multiple paths hitting the 3-call revise? → Reduce MAX_ITERATIONS or improve generate/verify prompts

A P95 at 7s when avg is 4s tells you the tail is being driven by the revise cases — those extra 2s are a third LLM call. The fastest path to hitting the SLO may be improving the generate_sql prompt quality to reduce revision rate, rather than tuning the serving infrastructure further.

