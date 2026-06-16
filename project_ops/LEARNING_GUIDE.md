# HW3 Masterclass: LLM Inference + Observability
## A Learning Journey, Not a Recipe

> **Before Phase 0**: If you have not set up your Nebius Cloud VM yet, start with [NEBIUS_SETUP.md](./NEBIUS_SETUP.md). It covers everything from creating a Nebius account and provisioning an H100 VM through SSH keys, port forwarding, Docker stack startup, and cost management. Both this guide and SOLUTIONS_REFERENCE.md assume you already have a running VM with SSH access and all five ports forwarded.

---

## How to Use This Guide

This document will not give you configs to copy or code to paste. It will give you the mental models, the right questions to ask, and pointers to where the answers live. That approach has a purpose: graders are explicitly evaluating your reasoning process, not your ability to find a working config. A missed SLO with a grounded diagnosis earns more points than a hit SLO you cannot explain.

**The discipline this guide asks of you:**

1. Attempt each phase before looking at any solution references.
2. Keep a running experiment log. At minimum: what you tried, what you expected, what actually happened.
3. When something surprises you, stop and figure out why before moving on. Surprise is where learning lives.
4. When you miss a target, write it up honestly. "I expected X, got Y, I think because Z" is the format graders want to see.

The H100 is expensive. But the most expensive waste is running the H100 without watching what it's doing. Grafana open, experiment log ready, one change at a time.

---

## The Mental Model Before You Start

Before you write a single line of code, build this picture in your head:

```
┌─────────────────────────────────────────────────────────┐
│                    Your Laptop Browser                   │
│         (Grafana at :3000, Langfuse at :3001)           │
└───────────────────────┬─────────────────────────────────┘
                        │  SSH port forward (5 ports)
┌───────────────────────┴─────────────────────────────────┐
│                     Cloud VM (H100)                      │
│                                                          │
│  ┌─────────────────┐    ┌────────────────────────────┐  │
│  │   Agent Server  │───▶│   vLLM (port 8000)         │  │
│  │   (port 8001)   │    │   Qwen3-30B-A3B-Instruct   │  │
│  │   LangGraph     │    │   OpenAI-compatible API     │  │
│  └────────┬────────┘    └────────────┬───────────────┘  │
│           │                          │                   │
│           ▼                          ▼                   │
│  ┌─────────────────┐    ┌────────────────────────────┐  │
│  │    Langfuse     │    │      Prometheus (9090)      │  │
│  │    (port 3001)  │    │      scrapes /metrics       │  │
│  │    traces       │    └────────────┬───────────────┘  │
│  └─────────────────┘                 │                   │
│                          ┌───────────┴───────────────┐  │
│                          │       Grafana (3000)       │  │
│                          │       dashboards           │  │
│                          └───────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**What each layer can observe and tune:**

| Layer | What you observe | What you tune |
|-------|-----------------|---------------|
| vLLM | /metrics: TTFT, TPOT, KV cache, queue depth, token rates | flags: memory, batching, quantization, prefill |
| Agent (LangGraph) | per-node latency, revision rate, iteration count | prompts, routing logic, iteration cap |
| Langfuse | per-span waterfall, token counts, agent metadata | metadata tags for filtering |
| Grafana | time-series aggregation, percentiles, alerts | panel queries, alert thresholds |

Every phase of this assignment exercises one or more of these layers. Phase 6 (SLO diagnosis) requires reading all of them together. That is why it is worth 25% of the grade.

**The four-layer production inference stack** (from Module 3 Week 5): API Gateway → Inference Router → Inference Engine → Data/Observability. In this assignment, you are building a toy version of this entire stack. The agent server is your gateway + router. vLLM is your inference engine. Prometheus + Grafana is your observability layer. Understanding this picture tells you which layer to look at when something is wrong.

---

## Phase 0: Environment Setup

### Understanding What You're Building First

Before running any commands, answer these questions:
- Why do you need five ports forwarded, not just one?
- What is Prometheus doing that Grafana cannot do directly?
- Why does Langfuse need to be local rather than using a cloud API?

If you cannot answer these, the setup steps will feel like incantations. When something breaks (it will break), you will not know where to look.

### Setup Steps

**Port Forwarding**: The VM runs all services on localhost. Your browser is on your laptop. SSH port forwarding creates a tunnel so that hitting `localhost:3000` on your laptop reaches port 3000 on the VM. Each `-L local:remote:port` line in the SSH command creates one tunnel. If any service is unreachable, check whether its port is in your forward list before debugging anything else.

**VSCode/Cursor Remote-SSH** is strongly recommended over plain SSH for this assignment. You get one-click port forwarding and a filesystem explorer that makes editing prompts and configs far less painful.

**Clone and sync**: `uv sync` creates an isolated Python environment from the lockfile. This matters because vLLM has sharp dependency version requirements. Do not use system pip.

**BIRD data**: `python scripts/load_data.py` downloads approximately 500 MB of SQLite databases and JSON files. Verify that `data/bird/` exists and contains both `.sqlite` files and question JSONs before proceeding. Later phases depend on this.

**Docker compose**: The `docker compose up -d` command starts Prometheus, Grafana, and Langfuse as Docker containers. These services are not Python processes — they are independent services that run in the background.

### Sanity Checks (Do All of These Before Phase 1)

Work through each check systematically:

1. **Prometheus** at `localhost:9090`: Click "Status" → "Targets". You should see vLLM listed. Before vLLM is running, it will show as DOWN. That is expected. Check that Prometheus itself is healthy.

2. **Grafana** at `localhost:3000`: Default credentials are admin/admin. You should see the starter dashboard under Dashboards. It has two pre-built panels. They will show no data until vLLM is running and Prometheus is scraping.

3. **Langfuse** at `localhost:3001`: Sign up with any email/password — this is a local instance, not a cloud service. No real data goes anywhere. Create a project and save the API keys; you need them in Phase 4.

4. **Port forward verification**: If a URL times out, the most likely cause is a missing or broken port forward. The service might be running fine on the VM but unreachable from your laptop.

### Experiment Log Template for Phase 0

```
Phase 0 Setup Log
Date/Time: 16 June 15:58 IST
VM hostname: dishant-ghai-instance

Port forwards established: [x] 3000  [x] 9090  [x] 3001  [x] 8000  [x] 8001
Method used: [x] VSCode Remote-SSH  [ ] Plain SSH

Services verified:
[x] Prometheus accessible at localhost:9090
[x] Grafana accessible at localhost:3000
[x] Langfuse accessible at localhost:3001, project created, keys saved to .env
[x] BIRD data present at data/bird/ (count of .sqlite files: ___)
[x] .env created from .env.example

Issues encountered: None
Resolution: None
```

---

## Phase 1: vLLM Configuration — The Art of Inference Tuning

### Before You Touch a Flag: Understand Your Workload

Every competent inference engineer answers these questions before touching a configuration knob. Do not skip this step:

1. **What is the prompt length distribution?** The assignment tells you: 1,500–3,000 tokens (English question + BIRD schema). What does that mean for prefill time? For KV cache pressure?

2. **What is the output length distribution?** SQL queries are short — typically 50–300 tokens. How does short output length affect the ratio of time spent in prefill vs decode?

3. **What is the concurrency target?** The SLO says 10 RPS with P95 < 5s. But each "request" is actually 2–3 sequential LLM calls (generate → verify → maybe revise). What does that mean for the actual request rate hitting vLLM?

4. **What is the SLO structure?** P95 < 5s end-to-end. That is not per-LLM-call. It is the total time for the agent to return an answer. If each vLLM call takes 2s, what ceiling does that put on the agent latency?

Write down your answers before reading the rest of this section. These numbers will tell you which levers matter most.

### Understanding Qwen3-30B-A3B as a MoE Model

The model name encodes critical architecture information. Before serving it, understand what you are serving:

**What does "30B-A3B" mean?** The model has 30 billion parameters total, but only approximately 3 billion are "active" (A3B = Active 3 Billion) per forward pass. This is the MoE (Mixture of Experts) architecture: the model routes each token through a sparse subset of expert FFN layers, not all of them.

**Questions to answer through reading:**
- Why does having 30B total parameters but only 3B active parameters change the GPU memory picture relative to a dense model?
- In a dense model, every token goes through every weight. In a MoE model, tokens go through different expert subsets. What does this mean for KV cache size? (Hint: KV cache depends on attention weights, not FFN weights. How does the attention architecture of Qwen3 MoE compare to a dense model?)
- What is "expert parallelism" (EP) and when would you use it vs tensor parallelism (TP)?
- On a single H100 80GB, does Qwen3-30B-A3B fit in BF16? In FP8? What are the weight sizes?

**Key insight from the knowledge base (Module 4 Week 1):** An H100 has 80GB HBM. Qwen3-30B-A3B in BF16 uses approximately 60–70GB for weights alone. That leaves very little room for KV cache, which is why your memory management flags matter enormously.

**Thinking mode vs non-thinking mode (critical for text-to-SQL):**
Qwen3 models support two modes: thinking mode (default, where the model generates a `<think>...</think>` block before answering) and non-thinking mode. For structured SQL generation, thinking mode can significantly increase output length and therefore latency. Research how to control this via the `/no_think` token in the system prompt or the `enable_thinking` parameter. Consider what mode makes sense for each of your three agent calls.

### The vLLM Configuration Space

Think of vLLM flags as falling into four categories. For each category, I will tell you what question it answers and what metric it affects — but not the specific values. Finding the right value for your workload is the exercise.

**Category 1: Memory Management**
- Question it answers: How much of the GPU do I give to the KV cache?
- Key flag: `--gpu-memory-utilization`
- What it affects: How many concurrent sequences can fit in HBM simultaneously
- Metric to watch: `vllm:kv_cache_usage_perc` in Grafana
- The tension: Higher utilization → more KV cache → more concurrency → but less headroom before OOM
- Research question: What is the default value? What happens at the extremes?

**Category 2: Sequence and Batch Limits**
- Question it answers: How many requests do I let run simultaneously?
- Key flags: `--max-num-seqs`, `--max-model-len`
- What they affect: Queue behavior, latency tail, memory consumption
- Metric to watch: `vllm:num_requests_running`, `vllm:num_requests_waiting`
- The tension: More concurrent seqs → higher throughput → but higher per-request latency
- Research question: What does `--max-model-len` cap, and why would you set it lower than the model's native context window?

**Category 3: Prefill/Decode Scheduling**
- Question it answers: Do I let long prompts monopolize the GPU?
- Key flag: `--enable-chunked-prefill`
- What it affects: TTFT for new requests vs TPOT for running requests (from Module 4 Week 2)
- Metric to watch: `vllm:time_to_first_token_seconds` vs `vllm:inter_token_latency_seconds`
- The tension: Chunked prefill smooths TPOT but increases TTFT for long prompts
- Research question: For 1.5–3K token prompts with short outputs, is TTFT or TPOT the dominant contributor to end-to-end latency?

**Category 4: Quantization**
- Question it answers: Should I trade some precision for speed and memory?
- Key consideration: `--quantization fp8`
- What it affects: Weight size in memory, compute throughput, generation quality
- The knowledge base fact (Module 4 Week 3): H100 has native FP8 Tensor Core support. The H100 at Nebius that was switched from A100 40GB to H100 80GB with FP8 showed 2.5x more users per GPU. FP8 on H100 is roughly 2x the FLOP rate of BF16.
- Research question: Does Qwen3-30B-A3B-Instruct-2507 come with pre-quantized FP8 weights, or do you need to quantize at serving time?

**Category 5: Prefix Caching**
- Question it answers: Can I reuse KV cache across requests that share a prefix?
- Key flag: `--enable-prefix-caching`
- The text-to-SQL opportunity: Every request for the same database includes the same schema in the prompt. If 30 eval questions hit 5 databases, many requests share a system prompt + schema prefix. Prefix caching would reuse those KV blocks.
- Research question: What is the cache hit rate metric in vLLM /metrics? When does prefix caching help vs hurt?

**Speculative Decoding (Optional but Instructive):**
From Module 4 Week 3: speculative decoding works by having a fast draft model propose tokens that the target model verifies in parallel. The speedup formula is roughly `(1 + α*γ)/(1 + cost_draft/cost_target)`. The key question for MoE models: does the sparse computation pattern of Qwen3-30B-A3B change when speculative decoding helps vs hurts? At batch size > 1, speculative decoding often hurts throughput because verification adds compute. For short outputs (SQL), is this likely to help?

### The Iterative Tuning Process

Here is the method you must follow, or Phase 6 will be very painful:

```
1. Start with a minimal configuration (as few flags as possible)
2. Verify the model loads and responds to a manual query
3. Fire 3-5 queries from eval_set.jsonl manually
4. Watch /metrics in your terminal: curl localhost:8000/metrics | grep -v "^#" | sort | grep vllm
5. Note the baseline numbers (TTFT, generation tokens/sec, KV cache %)
6. Form ONE hypothesis about what to change and why
7. Change ONE flag
8. Measure again
9. Note whether your hypothesis was confirmed or refuted
10. Repeat
```

**The /metrics endpoint is your friend before Grafana is ready.** You do not need a dashboard to see what vLLM is reporting. A simple curl to `localhost:8000/metrics` gives you every Prometheus metric as plain text. Spend 10 minutes reading through these before building any panels. The metric names tell you what they measure.

### What Good and Bad Look Like Upfront

**Signs your config is undersized:**
- KV cache usage immediately goes to 80%+ with even a few requests
- Requests pile up in the waiting queue (`vllm:num_requests_waiting` > 0 under light load)
- P95 latency is already near your SLO with no concurrency

**Signs your config has memory problems:**
- OOM errors during model load (reduce `--gpu-memory-utilization`)
- vLLM crashes on longer prompts (reduce `--max-model-len`)

**Signs your config has throughput problems:**
- TPOT is high but KV cache is not full (consider whether you are compute-bound or decode-bound)
- Queue depth grows linearly with load (concurrency ceiling hit)

### Experiment Log Template for Phase 1

Keep this table and add a row for each config you try:

```
| Config Change | Hypothesis | TTFT P50 | TTFT P95 | TPOT | KV Cache % | Queue | Learning |
|--------------|-----------|----------|----------|------|-----------|-------|---------|
| Initial      | baseline  |          |          |      |           |       |         |
| Change 1: __ | __        |          |          |      |           |       |         |
| Change 2: __ | __        |          |          |      |           |       |         |
```

### Questions to Answer Before Moving On

Before calling Phase 1 done, write one sentence for each flag you chose explaining:
- What problem does this flag address for this specific workload?
- What metric confirms it is working as expected?

If you cannot write that sentence, you have not finished Phase 1.

---

## Phase 2: Observability — Building a Dashboard That Answers Questions

### What Questions Should a Serving Dashboard Answer?

Start here, not with "what metrics does vLLM expose." Start with the questions:

**The 3am test:** Imagine you are woken up at 3am because users are complaining the system is slow. You open Grafana on your phone. What three questions do you need answered in the next 30 seconds?

1. **Is it slow?** (Is P95 latency above the SLO threshold right now?)
2. **Where in the request lifecycle is the slowness?** (Is it waiting to start? Slow prefill? Slow generation? Queue backup?)
3. **Do I have headroom, or am I at capacity?** (KV cache saturation? Queue depth?)

Design every panel to answer one of these questions. A panel that does not answer a question is visual noise.

### vLLM's /metrics Endpoint — The Treasure Map

Before building any panels, spend 15 minutes exploring `/metrics` directly:

```bash
curl localhost:8000/metrics | grep -v "^#" | sort
```

The metrics use the prefix `vllm:`. Read the names carefully — vLLM uses a consistent naming convention. Group them mentally:

- Metrics with `request` in the name → request lifecycle and counts
- Metrics with `token` in the name → throughput measurements  
- Metrics with `latency` or `time` in the name → latency histograms
- Metrics with `cache` in the name → KV cache state
- Metrics with `num_` prefix → current counts (gauges)

**For each metric you plan to use, answer:**
- Is it a Counter, Gauge, or Histogram? (Look at the `# TYPE` line)
- What is the unit? (seconds? tokens? fraction?)
- Does it reset on restart?

Histograms require special PromQL treatment: you use `histogram_quantile(0.95, rate(metric_bucket[5m]))` — not just the raw metric. Understanding this is essential for the latency panels.

### Latency Panel Design

**Panel 1: End-to-End Request Latency Percentiles**

Your SLO is a P95 end-to-end latency target. Which vLLM metric measures end-to-end request latency? (Search for metrics containing `e2e` or `latency` or look at the vLLM metrics documentation at https://docs.vllm.ai/en/latest/design/metrics/)

PromQL pattern to research: `histogram_quantile(0.95, sum(rate(METRIC_NAME_bucket[5m])) by (le))`

Build panels for P50, P95, and P99 on the same chart. Ask yourself: if P50 is fine but P95 is bad, what does that tell you? If both P50 and P95 are bad, what does that tell you?

**Panel 2: Time to First Token (TTFT)**

TTFT measures the prefill phase. From Module 4 Week 1: prefill is compute-bound (processing many input tokens at once). What metric name captures TTFT? Should you plot P50 or P95 or both? What is "too slow" for TTFT in a system where users expect answers within 5 seconds?

**Panel 3: Inter-Token Latency (ITL) / Generation Time**

ITL measures the decode phase. From Module 4 Week 1: decode is memory-bound (loading model weights for each generated token). What metric captures this? For short SQL outputs (50–150 tokens), does ITL or TTFT dominate end-to-end latency? (Work out the math: TTFT + N_tokens × ITL = E2E.)

**The diagnostic value:** When P95 E2E is high, you want to know whether TTFT is high (prefill problem) or ITL is high (decode problem). Different root causes, different fixes. Your dashboard must allow this diagnosis.

### Throughput Panel Design

**Panel 4: Request Rate**

How many requests per second are being served? What metric captures completed requests? (Look for counters with `request` and `total` in the name.) Use `rate()` over a 1-minute window. Plot both "requests running" and "requests waiting" — the ratio tells you about saturation.

**Panel 5: Token Throughput**

Tokens per second is the raw measure of GPU utilization for generation work. What are the two separate token metrics? (Prompt tokens vs generation tokens.) Why do you want both separately? (A system can have high prompt tokens/sec but low generation tokens/sec, which points to a specific bottleneck.)

**Panel 6: Queue Depth**

The queue depth tells you whether requests are waiting before they even start running. If queue depth is growing, throughput is insufficient for incoming load. What metric captures currently waiting requests?

### KV Cache Panel Design

**Panel 7: KV Cache Utilization**

Recall from Module 4 Week 2 (PagedAttention): the KV cache holds key-value tensors for all active sequences. When the KV cache fills, vLLM must either preempt (pause) a running sequence or reject new requests. 

Look for a metric that gives you KV cache utilization as a fraction (0–1). Plot it as a percentage gauge.

**Critical thinking questions for this panel:**
- At what utilization level should you start worrying? (Think about what happens as you approach 100%. From the knowledge base: preemption is expensive — it involves swapping KV blocks to CPU memory or recomputing them.)
- Under your target load of 10 RPS with 3 LLM calls per request, what KV cache utilization would you expect at steady state?
- If KV cache hits 100% and spikes your latency, which flag from Phase 1 would you reach for first?

### The "Readable Cold" Test

Before declaring your dashboard done, apply this test: show it to someone who has not seen it (or imagine them). Without explanation:
- Can they tell whether the system is currently healthy or degraded?
- Can they identify the slowest part of the request lifecycle?
- Can they see whether you are at capacity or have headroom?

If the answer to any of these is "not without explanation," the dashboard needs work.

**Anti-patterns to avoid:**
- Raw metric counts without time-windowing (meaningless without context)
- All panels in the same time scale when they have different natural scales
- Panels without units labeled (is that seconds? milliseconds? tokens?)
- Missing axis labels

### Experiment Log Template for Phase 2

```
Dashboard Build Log

Panel Name | Metric Used | PromQL Query | What it shows | Why I chose this
----------|-------------|--------------|---------------|------------------
E2E P95   |             |              |               |
TTFT P50  |             |              |               |
TTFT P95  |             |              |               |
ITL P50   |             |              |               |
Req/sec   |             |              |               |
Queue     |             |              |               |
KV Cache  |             |              |               |

Readable Cold Test: [ ] Passed  [ ] Failed
Issues: ___________________________
```

---

## Phase 3: The LangGraph Agent — Where Architecture Adds Value

### Understanding the Graph Before Implementing It

Draw this before writing any code. Literally draw it on paper or in a text editor:

```
State machine nodes:
- generate_sql: takes (question, schema) → produces (sql)
- execute: takes (sql) → produces (rows OR error) [PROVIDED]
- verify: takes (question, sql, rows, error) → produces (ok: bool, issue: str)
- revise: takes (question, sql, issue) → produces (new_sql)

Routing:
- After verify: if ok=True → END, if ok=False → revise (up to max_iterations)
- After revise: → execute (try the new SQL)

State: What data needs to flow through ALL these nodes?
```

**Questions to answer before implementing:**

1. What is the minimum state the graph needs to carry? List every field. (Hint: think about what `revise` needs that `generate_sql` produced but `verify` also used.)

2. Why is `verify` a separate LLM call rather than part of `generate_sql`? (What can verify see that generate_sql cannot?)

3. What does the `execute` node's output look like when SQL is syntactically valid but returns zero rows? When SQL is a syntax error? Your `verify` node must handle both cases differently.

4. Why is there an iteration cap? What goes wrong without it?

### The generate_sql Node — Study the Scaffolded Code

The assignment provides `generate_sql_node` as a worked example. Before writing `verify` or `revise`, read `generate_sql_node` carefully and answer:

- What is the return type of a LangGraph node?
- How does the node update state? (Does it return the new state, or return a diff, or something else?)
- How does the node call vLLM? (What client? What parameters?)
- What does the schema look like in the prompt? (Understanding the schema format is critical for writing good prompts.)

The mechanics are given to you. The cognitive work is understanding why each design choice was made.

### Designing the verify Node

The purpose of verify is to answer: "Given the question the user asked and the SQL result we got, does this result actually answer the question?"

**What failure cases must verify catch?**

Case 1: SQL syntax error → execution returned an error string, zero rows
- Example: question asks for "top 5 employees by salary," SQL has a syntax error, execution returns an error
- How should verify detect this? (Hint: it can see whether `error` in state is non-empty)

Case 2: SQL ran but returned zero rows when the question implies there should be rows
- Example: question asks "which departments have more than 10 employees?" and result is empty
- How should verify distinguish "correctly empty" (maybe no departments qualify) from "incorrectly empty" (SQL had a semantic error)?
- This is the hardest case. The LLM must reason about whether zero rows makes sense given the question.

Case 3: SQL ran but the columns or values clearly do not answer the question
- Example: question asks for "total sales by region" but result has columns `[name, email]`
- How does verify detect column mismatch?

**The output contract:**

```python
{"ok": bool, "issue": str}
```

The `issue` string is not just for your log — it feeds directly into the `revise` node's prompt. A vague issue like "the result seems wrong" produces a vague revision. A specific issue like "the query returned zero rows but the question asks for top-5 results, suggesting a HAVING or ORDER BY clause is missing" produces a targeted revision.

**Prompt engineering for verify:**
Research how Qwen3's structured output works in vLLM. You want the model to reliably return `{"ok": true}` or `{"ok": false, "issue": "..."}` — not markdown, not prose, not JSON with extra text. How do you enforce JSON output? Look at vLLM's guided decoding support and Qwen3's JSON mode.

### Designing the revise Node

The revise node has one job: take a failed SQL and fix it using the specific issue that verify identified.

**Questions to answer before writing the prompt:**

1. What information does revise need that was not in the original generate_sql prompt?
   - The original SQL (so it knows what to fix, not just redo from scratch)
   - The error or issue string from verify
   - The execution output (rows or error message)
   - The original question (in case it needs to re-read it)

2. Should revise always regenerate the SQL from scratch, or should it try to surgically fix the identified issue? What are the tradeoffs?

3. If verify says "the query returned the wrong columns," what information in the schema helps revise figure out the right columns?

**A critical design question:** Should revise look at ALL previous iterations or just the most recent? If the agent has tried twice and failed both times, does showing revise the history of attempts help or hurt? (Consider: context window length vs. the risk of the model fixating on a bad approach.)

### Wiring the Conditional Edge

In LangGraph, a conditional edge is a routing function that takes the current state and returns the name of the next node to visit.

Your `route_after_verify` function needs to:
1. Check `state["verify_result"]["ok"]`
2. Check whether the iteration count has reached the cap
3. Return `"END"` (or `"__end__"`) if verify passed or iterations exhausted
4. Return `"revise"` if verify failed and iterations remain

**Questions to research in LangGraph docs:**
- How do you add a conditional edge in LangGraph? (What function and what arguments?)
- What values can the routing function return?
- How do you map return values to destination node names?
- How do you add the iteration counter to state, and where do you increment it?

### Prompt Engineering for Text-to-SQL

**The three prompts you need to write:**

Prompt 1 (generate_sql): Given a question and a database schema, generate SQL.
- What schema format does BIRD use? (Look at the eval_set.jsonl entries to understand the schema structure before writing the prompt.)
- Qwen3 has a specific system prompt format and thinking/no-thinking control. For a deterministic SQL generation task, should you use thinking mode? Consider the latency implications.
- What constraints should you impose on the output format? (Just SQL? SQL in a code block? JSON with a SQL field?)

Prompt 2 (verify): Given question, SQL, and result, evaluate whether the result is correct.
- The model must output structured JSON. How do you ensure this?
- What context should you include to help the model make a good judgment?

Prompt 3 (revise): Given question, original SQL, issue description, and result, generate improved SQL.
- The issue string from verify is the key. How do you use it?
- Should you include the original SQL to help the model see what to change?

**Temperature note:** For SQL generation (deterministic, structured task), lower temperature (0.1–0.3) is usually better. For verification (reasoning task), slightly higher temperature may help. Research Qwen3's recommended parameters for thinking vs non-thinking mode.

### Testing Strategy

You do not need the H100 running to build and test the graph structure. Use the `.env` file to point at a small local model or the CPU vLLM instance for graph wiring and prompt iteration.

When testing with the real Qwen3-30B-A3B endpoint:
1. Test each node in isolation before testing the full graph
2. Deliberately trigger a revise: use a question that is likely to produce wrong SQL on the first try (ambiguous phrasing, multi-table join, aggregation with filter)
3. Watch the Langfuse trace (once Phase 4 is done) to see exactly what each node received and returned

### Experiment Log Template for Phase 3

```
Agent Development Log

Node: verify_node
Issue caught: _______ (syntax error / zero rows / wrong columns)
How detected: _______
Issue string produced: _______
Revise result: _______

Node: revise_node
Input issue: _______
SQL before revise: _______
SQL after revise: _______
Did it fix the problem? _______

Iteration cap tested: [ ] Yes  [ ] No
At cap behavior: _______

Prompt iterations:
v1 prompt for verify: [note key design decision]
Problem with v1: _______
v2 change: _______
Result: _______
```

---

## Phase 4: Langfuse Tracing — Making the Invisible Visible

### Why Tracing Matters for Multi-Step Agents

Without traces, you have a black box: a request goes in, an answer (or timeout) comes out. Grafana tells you aggregate statistics. But it cannot tell you:
- For this specific slow request, was it slow because of generate_sql or verify?
- For this specific failed request, did it iterate 3 times or 0 times?
- What was the exact prompt that caused the model to produce bad SQL?

Tracing gives you per-request, per-step visibility. It is the microscope to Grafana's thermometer.

**The waterfall you are trying to see:**

```
trace: agent_run (total: ~4.2s)
  ├── span: generate_sql (1.8s)
  │     prompt: "..."   tokens_in: 1847   tokens_out: 89
  ├── span: execute (0.05s)
  │     sql: "SELECT..."  result: [{...}]
  ├── span: verify (1.2s)  
  │     ok: false   issue: "zero rows returned for..."
  ├── span: revise (1.1s)
  │     prompt: "..."   tokens_in: 2203   tokens_out: 112
  └── span: execute (0.05s)
        sql: "SELECT..."  result: [{...}, {...}, {...}]
```

Without this, when Phase 6 asks you "is the agent slow because of verify or generate_sql?", you cannot answer.

### Setup Steps

1. Sign up at `localhost:3001` (local instance, no internet)
2. Create a project, copy the public key and secret key
3. Add both to `.env`: `LANGFUSE_PUBLIC_KEY=...` and `LANGFUSE_SECRET_KEY=...`
4. The callback handler picks these up automatically from environment variables

**The integration is one line in your graph invocation:**
```python
from langfuse.callback import CallbackHandler
handler = CallbackHandler()
result = graph.invoke(state, config={"callbacks": [handler]})
```

LangGraph will automatically create spans for each node. The handler captures timing, inputs, and outputs without any additional instrumentation.

### What Good Trace Metadata Looks Like

The assignment asks you to tag traces with metadata you will filter on in Phase 6. Think ahead: in Phase 6, you will want to filter by:

- Database name (is one database harder than others?)
- Number of iterations taken (did the loop help?)
- Whether the final SQL was valid (did the agent succeed?)
- Whether a revise was triggered at all

Useful metadata fields to add:
```python
handler.update_current_trace(
    metadata={
        "db_name": state["db"],
        "num_iterations": state["iteration"],
        "final_ok": state["verify_result"]["ok"],
        "revise_triggered": state["iteration"] > 0,
        "question_id": question_id,  # for cross-referencing with eval results
    }
)
```

The `question_id` is particularly useful: when Phase 5 identifies specific questions that failed, you can look them up directly in Langfuse to see the full agent trace.

### Experiment Log Template for Phase 4

```
Langfuse Setup Log

Project created: [ ] Yes
API keys in .env: [ ] Yes
Callback handler added: [ ] Yes

Trace verification:
- Fired 10 questions: [ ] Yes
- Found a trace showing generate_sql → execute → verify → revise: [ ] Yes
- Waterfall screenshot taken: [ ] Yes

Metadata tags added:
- db_name: [ ] Yes
- num_iterations: [ ] Yes
- final_ok: [ ] Yes
- revise_triggered: [ ] Yes
- question_id: [ ] Yes
- Other: _______

One interesting thing observed in the trace waterfall:
_______________________________________
```

---

## Phase 5: Evaluation — Separating Luck from Capability

### Execution Accuracy: The Right Metric for Text-to-SQL

**Why not string match?** Two SQL queries can produce the same result set while looking completely different syntactically. `SELECT a, b FROM t ORDER BY a` and `SELECT a, b FROM t ORDER BY 1` are equivalent. String match would call them different; execution accuracy calls them the same.

**Why not BLEU?** BLEU measures n-gram overlap. SQL syntax is not natural language — a single keyword difference can produce a completely different result.

**Execution accuracy** runs both your generated SQL and the gold SQL against the actual database and compares the result sets. If the result sets match (same rows, same values), the query is correct regardless of how it was phrased.

**The canonicalization problem:** What does "same result set" mean? You need to decide:
- Do column names matter? (BIRD says: ignore column-name case when comparing)
- Does row order matter? (Sort both result sets before comparing)
- Does whitespace in string values matter? (Usually no)
- What if one returns `None` and the other returns `0`? (Edge case to handle)

Think through your canonicalization logic before implementing. Subtle bugs here make your eval signal unreliable.

### Building the Eval Runner

The eval loop has this structure:

```python
for question in eval_set:
    # 1. Call agent HTTP endpoint
    response = requests.post("http://localhost:8001/answer", 
                            json={"question": q["question"], "db": q["db"]})
    agent_sql = response.json()["sql"]
    
    # 2. Run gold SQL against target DB
    gold_rows = run_sql(q["db_path"], q["gold_sql"])
    
    # 3. Run agent SQL against target DB
    agent_rows = run_sql(q["db_path"], agent_sql)
    
    # 4. Compare canonicalized row sets
    correct = canonicalize(gold_rows) == canonicalize(agent_rows)
    
    # 5. Record result
    results.append({
        "question_id": q["id"],
        "correct": correct,
        "num_iterations": response.json()["iterations"],  # need agent to return this
        "db": q["db"],
        ...
    })
```

**Per-iteration pass rate:** The agent returns which iteration produced the final SQL. You can compute:
- `iter0_pass_rate`: % correct where `num_iterations == 0` (first try)
- `iter1_pass_rate`: % correct where `num_iterations <= 1`
- `final_pass_rate`: % correct overall

If `iter0_pass_rate == final_pass_rate`, the verify→revise loop is doing nothing. The loop only adds value if `final_pass_rate > iter0_pass_rate`.

**Questions to answer in your eval:**
- What does the agent return if it hits the iteration cap without verify passing? (You need to handle this in the eval runner.)
- What does the eval do with a runtime exception? (Agent crashes, network timeout, etc.)
- Should you run questions sequentially or in parallel? (Parallel is faster but harder to correlate with Grafana.)

### Reading the Results

Before you run the eval, form hypotheses:

**Hypothesis A: The loop genuinely helps**
Expected: `final_pass_rate` is meaningfully higher than `iter0_pass_rate` (say, 5–15pp higher)
Meaning: your verify prompt is catching real errors and your revise prompt is fixing them

**Hypothesis B: The loop is neutral**
Expected: `iter0_pass_rate ≈ final_pass_rate`
Meaning: verify is either never triggering (too permissive) or triggering but revise makes no improvement
Diagnosis path: check traces to see how often revise is triggered. If rarely → verify prompt is too permissive. If often but no improvement → revise prompt is not using the issue string effectively.

**Hypothesis C: The loop actually hurts (regression)**
Expected: `final_pass_rate < iter0_pass_rate`
Meaning: revise is sometimes taking a correct SQL and producing an incorrect one
This is a real possibility if your verify prompt has false positives (calling correct results wrong)
Diagnosis: look at cases where `num_iterations > 0` and `correct == False` despite iter0 being correct

### The Grafana Connection

Run the eval while watching Grafana. What you expect to see:
- 30 questions × ~2.5 calls each ≈ 75 vLLM requests
- KV cache utilization rising and falling as batches process
- TTFT and ITL visible as a time series

What to note: does your configuration from Phase 1 hold up under the eval load? Is KV cache utilization staying healthy? Is P95 latency within the SLO? This gives you your baseline before Phase 6.

### Experiment Log Template for Phase 5

```
Eval Results Log

Run date: ___________
vLLM config version: (tag from Phase 1 iteration log)
Total questions: 30

Overall pass rate: ___/30 = ___%
iter0 pass rate: ___/30 = ___%
iter1 pass rate: ___/30 = ___%
iter2 pass rate: ___/30 = ___%
iter3 (final) pass rate: ___/30 = ___%

Is the loop earning its keep? [ ] Yes (+__pp)  [ ] No (no improvement)  [ ] Regression (-__pp)

Top failure patterns observed:
1. _______
2. _______
3. _______

Grafana observations during eval run:
- Peak KV cache utilization: ___%
- P95 E2E latency during eval: ___s
- Any queue buildup? ___
```

---

## Phase 6: SLO Diagnosis — The Most Important Phase

This phase is worth 25% of your grade. Read this section carefully.

### The SLO: P95 < 5s @ 10 RPS

First, make sure you understand what this SLO actually requires:

**"10 RPS"** means 10 full agent runs per second sustained over a 5-minute window. Each agent run makes 2–3 LLM calls. So vLLM is seeing 20–30 requests per second. Does your H100 support that? Work out the math:

- If each LLM call takes 2s average and there are 2.5 calls per agent run
- Agent serial latency = 5s per request (all calls sequential)
- To achieve 10 agent RPS, you need 10 × 5s = 50 "request-seconds" of capacity per second
- That means at least 50 concurrent LLM requests being processed simultaneously
- Is that feasible for a single H100 with a MoE model?

**This math might tell you the SLO is extremely tight or impossible under naive config.** Understanding the ceiling is the first step of diagnosis.

**"P95 < 5s"** means 95% of all agent runs complete within 5 seconds. The remaining 5% can be slower. When you see P95 at 8s, it does not mean most requests are slow — it means the tail is bad. Where does your tail latency come from?

### The Diagnostic Loop

This is the heart of Phase 6, and it is a method, not a process you run once:

```
Step 1: OBSERVE — look at the dashboard, identify the metric that is failing your SLO
Step 2: HYPOTHESIZE — form a specific, falsifiable hypothesis about the cause
         Good: "KV cache is at 95% utilization, causing preemption, which spikes TTFT for queued requests"
         Bad: "The system is slow"
Step 3: CHANGE ONE THING — the one flag or config change that addresses your hypothesis
Step 4: MEASURE — run the load test again, check whether the metric you targeted actually moved
Step 5: INTERPRET — did end-to-end latency improve? Did it not? Why?
Step 6: REPEAT — if you are iteration 7 and still guessing, stop and re-read the dashboard
```

**Why "one thing at a time" is not just a formality:** If you change three flags simultaneously and latency improves by 30%, you have no idea which flag caused the improvement. You might remove a flag later thinking it was useless, when it was actually the key one. One change = one data point you can trust.

### Reading Grafana for Diagnosis

**The diagnostic sequence when P95 is bad:**

```
1. Is TTFT high? → prefill is the bottleneck
   → Look at: request queue depth (are requests waiting before even starting?)
   → Look at: KV cache utilization (is there room for new requests?)
   → Consider: chunked prefill, max-model-len reduction, max-num-seqs reduction

2. Is generation time high but TTFT is fine? → decode is the bottleneck
   → Look at: tokens/sec generation rate
   → Look at: batch size (are you generating too many tokens in parallel, saturating memory bandwidth?)
   → Consider: max-num-seqs, quantization to FP8 (faster memory bandwidth)

3. Is the queue growing? → concurrency ceiling hit
   → Look at: num_requests_running vs max-num-seqs config
   → Look at: KV cache utilization (are you hitting memory limits?)
   → Consider: whether concurrency limit or memory limit is binding

4. Is KV cache > 90%? → preemption risk
   → From Module 4 Week 2: when KV cache is full, vLLM either preempts (expensive) or rejects requests
   → Preemption adds to TTFT for affected requests because KV blocks must be swapped back in
   → Consider: gpu-memory-utilization increase, max-model-len reduction, max-num-seqs reduction
```

### The Configuration Levers (What to Consider, Not What to Set)

For each lever, understand the problem it solves before reaching for it:

**Lever 1: `--max-model-len`**
- Problem it solves: Limits the KV cache allocation per sequence, freeing memory for more concurrent sequences
- When to use: When you have long prompts eating too much KV cache, pushing KV utilization high
- What metric tells you it's the problem: KV cache % high even with few concurrent requests
- What you sacrifice: Cannot handle prompts longer than the cap (eval set prompts must be shorter)
- For this workload: BIRD prompts are 1.5–3K tokens. What cap leaves room for output?

**Lever 2: `--max-num-seqs`**
- Problem it solves: Limits concurrent sequences → limits peak KV cache usage → prevents OOM-induced preemption
- When to use: When KV cache spikes to 100% causing latency spikes
- What you sacrifice: Throughput — you artificially cap concurrency
- The tension: Too high → KV cache pressure; too low → requests queue, hurting P95

**Lever 3: `--enable-chunked-prefill`**
- Problem it solves: Long prefills monopolize the GPU, causing decode pauses for running requests
- When to use: When TTFT is being hit by long-running prefills interleaving with decode
- For this workload: 2K token prompts are medium-length. Is chunked prefill warranted?
- What metric tells you it helps: ITL smoothness under load (is it consistent or spiky?)

**Lever 4: `--quantization fp8`**
- Problem it solves: FP8 reduces weight memory footprint by 2x vs BF16, doubles compute throughput on H100
- Memory impact: More HBM available for KV cache after weights loaded
- Compute impact: Faster prefill and decode (H100 FP8 Tensor Cores are 2x faster than BF16)
- What to verify: Does Qwen3-30B-A3B quality hold under FP8? Run the eval after changing.
- Note: This is a significant change. Run the eval set again after enabling it.

**Lever 5: `--enable-prefix-caching`**
- Problem it solves: Multiple requests sharing a schema prefix reuse cached KV blocks
- For the eval: If multiple questions target the same database, they share the system prompt + schema
- What metric to check: Cache hit rate in /metrics (look for `vllm:gpu_prefix_cache_hit_rate`)
- When it does not help: If every request has a fully unique prefix, there is nothing to cache

**Lever 6: `--kv-cache-dtype`**
- Problem it solves: Stores KV cache in lower precision (fp8 vs bfloat16), reducing HBM usage per sequence
- Benefit: More sequences fit in same HBM → higher effective concurrency
- Risk: Accuracy loss in long-sequence generation (KV cache quantization errors accumulate)
- For this workload: SQL generation is short. KV cache quantization risk is lower than for long generations.

### What the Iteration Log Should Look Like

This is the format graders want:

**Bad example (do not write this):**
> "I changed max_tokens and performance improved."

**Good example (write this):**
> "Saw: P95 E2E latency at 8.4s during 10 RPS load test. KV cache utilization at 93% in Grafana, queue depth spiking to 15–20 during load. Hypothesized: KV cache near saturation is causing preemption, which forces TTFT spikes for queued requests. Changed: --gpu-memory-utilization from 0.85 to 0.92, giving more HBM to KV cache. Result: KV cache utilization dropped to 74% at steady state, queue depth stayed under 5, P95 E2E fell to 6.2s. SLO still missed. The improvement confirmed the hypothesis — more KV cache capacity reduced preemption — but the SLO gap suggests a second bottleneck."

Notice the structure: **saw X → hypothesized Y → changed Z → result was W**. The result includes both what the targeted metric did AND whether the SLO improved. These can diverge (metric improves but SLO does not move), which is itself a learning.

### What if You Miss the SLO?

Read the grading rubric again:

> "A missed SLO with a metric-grounded diagnosis is better than a hit SLO you can't explain."

If you miss the SLO, write up the gap quantitatively:
- Baseline: P95 = X seconds
- After iteration 1: P95 = Y seconds (delta: Z)
- After iteration 2: P95 = W seconds (delta: V)
- Final: P95 = A seconds, SLO requires B seconds, gap = C seconds

Then explain why you believe you cannot close this gap further on this hardware: "The theoretical lower bound for this workload is approximately D seconds per agent run (each of 2.5 LLM calls at minimum E seconds each, serialized). Achieving P95 < 5s would require sub-2s per LLM call, which at 10 RPS concurrency requires the H100 to process N simultaneous requests. At that concurrency, KV cache pressure exceeds available HBM even at maximum utilization."

This is a professional miss. It shows you understand the system.

**If you hit the SLO on the first try**, the graders want you to push past it anyway: "Find what breaks." Reduce `--gpu-memory-utilization`, increase `--max-num-seqs`, crank up the load test RPS until something fails. Then document what failed and why. This demonstrates you understand the system's limits, not just that you got lucky with a good initial config.

### Experiment Log Template for Phase 6

```
SLO Diagnosis Log

Baseline (Phase 1 config, 5-min load test @ 10 RPS):
- P50 E2E latency: ___s
- P95 E2E latency: ___s  [SLO target: 5.0s]
- KV cache utilization: ___%
- Queue depth peak: ___
- Tokens/sec: ___

Iteration 1:
Saw: __________________________________
Hypothesized: __________________________
Changed: _______________________________
Metric moved: _________ (from ___ to ___)
P95 E2E after: ___s
SLO status: [ ] Hit  [ ] Still missing by ___s
Learning: _____________________________

Iteration 2:
Saw: __________________________________
Hypothesized: __________________________
Changed: _______________________________
Metric moved: _________ (from ___ to ___)
P95 E2E after: ___s
SLO status: [ ] Hit  [ ] Still missing by ___s
Learning: _____________________________

[Add rows as needed]

Final config:
- P95 E2E: ___s
- SLO verdict: [ ] Hit  [ ] Missed by ___s
- Eval pass rate maintained? [ ] Yes  [ ] Regressed (new rate: ___)
```

---

## Phase 7: The Report

### What Makes a Strong REPORT.md

The report is a professional document, not a homework submission. It should read like something a tech lead would write after completing a project. Three characteristics distinguish strong reports:

**Honest**: If the SLO was missed, say so and quantify the gap. If quality regressed after tuning, say so. If a phase did not work the way you expected, explain what happened.

**Specific**: Numbers, not adjectives. "P95 latency improved from 8.4s to 6.2s after increasing gpu-memory-utilization" not "performance improved significantly."

**Evidence-linked**: Every claim backed by a measurement. "The verify→revise loop improved pass rate by 8pp (from 62% to 70%)" not "the agent loop was helpful."

### The "What I'd Do With More Time" Section

This section is explicitly called out in the grading rubric as requiring specificity. "Add Kubernetes" does not count.

Good examples:
- "Profile the prefill kernel with Nsight Systems to find whether the attention computation or the expert routing is the dominant contributor to TTFT on long prompts"
- "Implement batch execution in the eval runner to run 5 questions concurrently — each takes ~5s serially, so 30 questions takes 150s; at 5x concurrency that drops to 30s, making rapid prompt iteration much cheaper"
- "Add P/D disaggregation: deploy a separate prefill instance for the 2K token prompts and a decode instance for the short SQL outputs, since the two phases have fundamentally different optimal configurations"
- "Train an EAGLE-style draft head on Qwen3-30B-A3B using SQL generation traces, since the predictable structure of SQL output should yield high acceptance rates (α > 0.8) and potentially 2x speedup"
- "Implement per-database schema caching in the agent to avoid re-encoding the full schema on every request — with prefix caching enabled, this would dramatically improve KV cache hit rates for the BIRD workload"

Notice these are specific, grounded in what you learned in the assignment, and point to a specific expected improvement with a reason.

### Structure Template

```markdown
# REPORT.md

## Serving Configuration

| Flag | Value | Rationale |
|------|-------|-----------|
| --model | ... | ... |
| --max-model-len | ... | ... |
| --gpu-memory-utilization | ... | ... |
| ... | ... | ... |

## Eval Results

| Metric | Baseline | After Tuning |
|--------|----------|-------------|
| Overall pass rate | X% | Y% |
| iter0 pass rate | X% | Y% |
| iter3 pass rate | X% | Y% |
| Loop improvement | — | +Zpp |

[Agent value paragraph here]

## SLO Diagnosis

Baseline: P95 = Xs @ 10 RPS

| Iteration | Saw | Hypothesized | Changed | Result |
|-----------|-----|-------------|---------|--------|
| 1 | ... | ... | ... | P95: Xs → Ys |
| 2 | ... | ... | ... | P95: Ys → Zs |
| 3 | ... | ... | ... | P95: Zs → Ws |

Final: P95 = Ws. SLO [hit / missed by Xs].

## What I'd Do With More Time

1. [Specific, grounded item with expected improvement]
2. [Specific, grounded item with expected improvement]
3. [Specific, grounded item with expected improvement]
```

---

## Final Checklist

Before submitting, verify:

**Files required:**
- [ ] REPORT.md (2–3 pages, no more)
- [ ] infra/grafana/provisioning/dashboards/serving.json (exported from Grafana UI)
- [ ] agent/graph.py (verify, revise nodes + conditional edge implemented)
- [ ] agent/prompts.py (three prompts: generate, verify, revise)
- [ ] evals/run_eval.py (with canonicalization and per-iteration pass rate)
- [ ] results/eval_baseline.json
- [ ] results/eval_after_tuning.json
- [ ] All 6 screenshots in the screenshots/ directory

**Quality checks:**
- [ ] Dashboard passes the "readable cold" test
- [ ] REPORT.md iteration log has the saw → hypothesized → changed → result format
- [ ] "What I'd do with more time" is specific (not "add Kubernetes")
- [ ] Phase 6 discussion includes both what the metric did AND whether P95 moved with it
- [ ] Eval results distinguish iter0 pass rate from final pass rate

**The diagnostic capability check:**
Read your REPORT.md out loud. For each config change you made:
- Can you explain what metric told you to make that change?
- Can you explain what the theory was behind the change?
- Can you explain what actually happened after the change?

If yes for all three: you have passed Phase 6 in spirit, regardless of whether you hit the SLO number.
