# Solutions Reference — HW3: LLM Inference + Observability

---

## Nebius Cloud Quick Reference

> For full setup instructions (account creation, VM provisioning, SSH keys, port forwarding), see [NEBIUS_SETUP.md](./NEBIUS_SETUP.md). This section is a fast reference for students who have already provisioned their VM.

### Instance Type for H100 80GB

| Field | Value |
|-------|-------|
| Platform | `gpu-h100-sxm` |
| Preset (1 GPU) | `1gpu-16vcpu-200gb` |
| GPU memory | 80 GB HBM3 |
| vCPUs | 16 |
| RAM | 200 GiB |
| Recommended OS image | Ubuntu 22.04 with CUDA pre-installed |
| Recommended boot disk size | 200 GiB SSD |

### Estimated H100 Hours Needed

| Phase | H100 Required? | Estimated Hours |
|-------|---------------|-----------------|
| 0 (Setup, data load, Docker stack) | No | 0 |
| 1 (vLLM config + first model load) | Yes | 1–3 |
| 2 (Grafana dashboard) | CPU vLLM OK | 0 (H100) |
| 3 (Agent implementation) | No | 0 |
| 4 (Langfuse tracing) | No | 0 |
| 5 (Eval run with 30B model) | Yes | 1–2 |
| 6 (SLO diagnosis + multiple load tests) | Yes | 3–6 |
| **Total focused run** | | **~6–12 H100 hours** |

At ~$3.85/hr on-demand, budget $25–50 for a focused run. Stop the VM (not delete) between sessions.

### Key URLs Once Set Up

With port forwarding active, all services are accessible from your laptop browser:

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| vLLM API | http://localhost:8000 | none (api_key="token-abc") |
| vLLM metrics | http://localhost:8000/metrics | none |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | none |
| Langfuse | http://localhost:3001 | set at local sign-up |
| Agent server | http://localhost:8001 | none |

### SSH Quick Reference

```bash
# Connect with all 5 ports forwarded
ssh -i ~/.ssh/id_ed25519 \
    -L 3000:localhost:3000 \
    -L 9090:localhost:9090 \
    -L 3001:localhost:3001 \
    -L 8000:localhost:8000 \
    -L 8001:localhost:8001 \
    -o ServerAliveInterval=60 \
    ubuntu@<YOUR_VM_IP>

# Verify GPU on the VM
nvidia-smi

# Check Docker stack is running
docker compose ps
```

---

## Read This First

This document contains complete, working solutions for every phase. It is intended as a reference to check your work and understand why particular choices were made — not as a substitute for attempting each phase yourself.

**The learning only happens if you have tried first.** Looking at a solution before forming your own mental model shortcuts the experience that makes the knowledge stick. The graders are specifically evaluating your reasoning process, not your ability to reproduce these solutions.

**Recommended usage:**
- Attempt each phase using the LEARNING_GUIDE.md
- Get stuck (expected and fine)
- Check this document for the specific piece you are missing
- Understand *why* the solution works, not just *what* it does
- Return to your implementation with that understanding

---

## Phase 1: vLLM Configuration

### The Model Size Reality

Qwen3-30B-A3B-Instruct-2507 is a MoE model: 30B total parameters, ~3B active per forward pass. Weight sizes:
- BF16: ~60–70 GB (tight on a single H100 80GB, leaves little KV cache headroom)
- FP8: ~30–35 GB (leaves ~40–45 GB for KV cache — much more room for concurrent sequences)

On a single H100 80GB, **FP8 quantization is not just an optimization, it is practically required** to get reasonable concurrency. Without it, your KV cache budget is so small that you cannot sustain meaningful batch sizes.

### Recommended Starting Configuration

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --quantization fp8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tensor-parallel-size 1
```

### Flag-by-Flag Justification

| Flag | Value | Rationale |
|------|-------|-----------|
| `--model` | Qwen/Qwen3-30B-A3B-Instruct-2507 | The assignment-specified model |
| `--dtype bfloat16` | bfloat16 | Compute dtype — BF16 is the standard for H100 (avoids FP16 overflow issues, matches H100 Tensor Core preference). Note: model weights stored in FP8 due to `--quantization fp8`. |
| `--quantization fp8` | fp8 | **Critical for single H100**: reduces weights from ~70GB (BF16) to ~35GB (FP8), freeing ~35GB of HBM for KV cache. H100 has native FP8 Tensor Core support — 2x compute throughput vs BF16. Qwen3-30B-A3B quality is well-maintained at FP8 (consistent with the H100→FP8 case study in Module 4 Week 1: "2.5x more users per GPU, 10% faster generation"). |
| `--max-model-len` | 8192 | Limits KV cache allocation per sequence. The workload uses 1.5–3K token prompts + short SQL outputs (<300 tokens). Setting this to 8192 gives headroom for the full prompt + output while capping KV cache waste from allocating for the model's full 131K context window. With 40GB of KV cache budget and 8192 max len, you can fit many more concurrent sequences than at 131K. |
| `--gpu-memory-utilization` | 0.90 | Gives vLLM 90% of H100 HBM for weights + KV cache combined. After FP8 weights (~35GB), approximately 37GB remains for KV cache. The 10% headroom prevents OOM from GPU memory fragmentation and CUDA reserved memory. |
| `--max-num-seqs` | 32 | Maximum concurrent sequences. At 10 agent RPS with ~2.5 LLM calls each = ~25 concurrent LLM requests. Setting 32 provides buffer above the steady-state concurrency target. Tune this based on KV cache utilization — if cache hits 95%+, lower this. |
| `--enable-chunked-prefill` | (flag only) | Prevents long prefills (2K+ token prompts) from monopolizing the GPU and spiking TPOT for concurrently decoding sequences. For this workload, prompt prefill takes meaningful time — chunked prefill interleaves it with decode steps, smoothing inter-token latency for all active requests. |
| `--enable-prefix-caching` | (flag only) | Multiple BIRD questions hit the same database, sharing the system prompt + full schema prefix (potentially 1,000–1,500 tokens). Prefix caching reuses those KV blocks via PagedAttention's block table. Reduces effective TTFT for the 2nd–Nth request against the same database. |
| `--tensor-parallel-size` | 1 | Single H100 — no tensor parallelism needed. TP would only be needed if the model did not fit on one GPU (it does, with FP8). EP (`--enable-expert-parallel`) is unnecessary at TP=1. |

### Why Not Expert Parallelism?

Expert parallelism (`--enable-expert-parallel`) splits MoE expert layers across multiple GPUs. With only one GPU, this flag is irrelevant. It becomes relevant when you have 2+ GPUs and want to scale a MoE model. At TP=1, vLLM handles MoE routing within the single GPU.

### Why Not Speculative Decoding?

Speculative decoding helps when:
1. You have a compatible, smaller draft model in the same model family
2. Batch size is small (output is short enough to be memory-bound)
3. Acceptance rate is high (structured, predictable outputs like SQL)

SQL output acceptance rates can be high (α = 0.75–0.85 for structured content per Module 4 Week 3). However:
- There is no off-the-shelf EAGLE head for Qwen3-30B-A3B-Instruct-2507
- At 10 RPS with 32 concurrent sequences, speculative decoding adds compute overhead that may reduce throughput
- N-gram speculative decoding could theoretically help for SQL (repetitive patterns), but requires careful measurement

**Verdict for this assignment**: Do not use speculative decoding in your initial config. If you hit the SLO early and want to experiment, n-gram SD (`--speculative_model ngram --num_speculative_tokens 3`) is the lowest-risk option.

### Qwen3 Thinking Mode: A Critical Config Decision

Qwen3-30B-A3B has a "thinking" mode (default) that generates `<think>...</think>` blocks before answering. For SQL generation, this can add 500–2,000 extra output tokens per request, dramatically increasing latency.

**For this workload, disable thinking mode in your prompts:**

Option 1: Add `/no_think` at the end of the system prompt:
```
You are a SQL expert. Generate only valid SQL. /no_think
```

Option 2: Use vLLM's chat template kwargs (check vLLM docs for the current flag).

**Why this matters:** A thinking-mode response for a SQL query might generate 1,000 tokens of reasoning + 100 tokens of SQL = 1,100 output tokens. Non-thinking generates just the 100 SQL tokens. That is a 10x difference in generation latency. At 10 RPS, this is often the single biggest latency contributor.

**After disabling thinking**: Use temperature=0.1 for SQL generation (deterministic, structured task), temperature=0.3 for verify (needs some reasoning flexibility).

### Verifying the Config Works

```bash
# Quick sanity check — should return SQL within ~3s
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [
      {"role": "system", "content": "You are a SQL expert. Answer with only SQL. /no_think"},
      {"role": "user", "content": "List all tables in the database."}
    ],
    "max_tokens": 200,
    "temperature": 0.1
  }'

# Check metrics are exposed
curl localhost:8000/metrics | grep vllm | head -30
```

---

## Phase 2: Grafana Dashboard

### Complete PromQL Queries

All vLLM metrics use the `vllm:` prefix (verified from vLLM docs). These are Prometheus histograms and gauges.

**Important note on histogram queries**: vLLM metrics are histograms. Use `histogram_quantile(p, rate(metric_bucket[window]))` not the raw metric.

#### Latency Panels

**Panel: E2E Request Latency (P50 and P95)**
```promql
# P95
histogram_quantile(0.95,
  sum(rate(vllm:e2e_request_latency_seconds_bucket[2m])) by (le)
)

# P50
histogram_quantile(0.50,
  sum(rate(vllm:e2e_request_latency_seconds_bucket[2m])) by (le)
)
```

**Panel: Time to First Token — TTFT (P50 and P95)**
```promql
# P95 TTFT
histogram_quantile(0.95,
  sum(rate(vllm:time_to_first_token_seconds_bucket[2m])) by (le)
)

# P50 TTFT
histogram_quantile(0.50,
  sum(rate(vllm:time_to_first_token_seconds_bucket[2m])) by (le)
)
```

**Panel: Inter-Token Latency / Generation Time (P50 and P95)**
```promql
# P95 ITL (represents decode speed)
histogram_quantile(0.95,
  sum(rate(vllm:inter_token_latency_seconds_bucket[2m])) by (le)
)

# P50 ITL
histogram_quantile(0.50,
  sum(rate(vllm:inter_token_latency_seconds_bucket[2m])) by (le)
)
```

**Why all three latency panels?** The diagnostic value: if E2E P95 is bad but TTFT P95 is fine, the problem is in decode (generation). If TTFT P95 is also bad, the problem is in prefill (or queue). If both are fine but E2E is bad, check whether the agent overhead (HTTP round-trip, Python processing) is the issue.

#### Throughput Panels

**Panel: Request Rate (running vs waiting)**
```promql
# Currently running
vllm:num_requests_running

# Currently waiting in queue
vllm:num_requests_waiting
```

Plot these on the same graph with different colors. When `waiting > 0` under your target load, the system is saturated — either compute, KV cache, or max-num-seqs is the binding constraint.

**Panel: Token Throughput**
```promql
# Generation tokens per second (the GPU doing useful work)
rate(vllm:generation_tokens_total[1m])

# Prompt tokens processed per second (prefill work)
rate(vllm:prompt_tokens_total[1m])
```

**Panel: Request Completion Rate**
```promql
# Requests completed per second
rate(vllm:request_success_total[1m])
```

#### KV Cache Panel

**Panel: KV Cache Utilization (%)**
```promql
# This metric is already a fraction 0–1, multiply by 100 for percentage
vllm:gpu_cache_usage_perc * 100
```

Configure this panel with:
- Thresholds: Green (0–70%), Yellow (70–90%), Red (90–100%)
- Y-axis: 0–100 with "%" unit
- Stat panel or gauge visualization works well for this metric

**Why does 90%+ matter?** From Module 4 Week 2 (PagedAttention): when KV cache blocks are exhausted, vLLM must preempt (swap blocks to CPU RAM, expensive!) or reject requests. Preemption adds seconds to TTFT for the next request that gets those blocks back. Keeping KV cache below 85% under load is a practical target.

**Panel: Prefix Cache Hit Rate (if enabled)**
```promql
# Hit rate 0–1, multiply by 100 for percentage
vllm:gpu_prefix_cache_hit_rate * 100
```

This tells you whether `--enable-prefix-caching` is actually helping. If hit rate is 0%, every request has a unique prefix and caching is not helping.

### Dashboard JSON Structure

The starter dashboard already has two panels. Extend it by adding panels in the Grafana UI (click "Add panel"), then export the JSON via Dashboard Settings → JSON Model.

Panel organization recommendation:
- Row 1: Latency (TTFT P50/P95, E2E P50/P95, ITL P50/P95) — side by side
- Row 2: Throughput (requests running/waiting, token throughput, request completion rate)
- Row 3: KV Cache (utilization %, prefix cache hit rate if enabled)

**Panel units:**
- Latency panels: Unit = "seconds" in Grafana field configuration
- Token/request rates: Unit = "/ second"
- KV Cache %: Unit = "Percent (0-100)"

---

## Phase 3: Agent Implementation

### State Definition

```python
# agent/graph.py
from typing import TypedDict, Optional, List, Any

class AgentState(TypedDict):
    # Input
    question: str
    db: str
    schema: str  # Rendered schema string
    
    # Populated by generate_sql / revise
    current_sql: str
    
    # Populated by execute (provided node)
    execution_result: Optional[List[Any]]  # rows or None
    execution_error: Optional[str]         # error message or None
    
    # Populated by verify
    verify_ok: bool
    verify_issue: str
    
    # Loop control
    iteration: int
    max_iterations: int
    
    # Final output
    final_sql: Optional[str]
    final_result: Optional[List[Any]]
```

### verify_node Implementation

```python
# agent/graph.py
import json
from langchain_openai import ChatOpenAI
from agent.prompts import VERIFY_PROMPT

def verify_node(state: AgentState) -> dict:
    """
    Evaluates whether the executed SQL result actually answers the question.
    
    Returns state update with:
      - verify_ok: bool
      - verify_issue: str (empty string if ok=True, specific diagnosis if ok=False)
    """
    llm = ChatOpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        api_key=os.getenv("OPENAI_API_KEY", "token-abc"),
        temperature=0.3,
        max_tokens=256,
    )
    
    # Build context for verify
    result_str = "EXECUTION ERROR: " + state["execution_error"] if state["execution_error"] \
                 else json.dumps(state["execution_result"][:10], indent=2)  # cap to 10 rows
    
    prompt = VERIFY_PROMPT.format(
        question=state["question"],
        schema=state["schema"],
        sql=state["current_sql"],
        result=result_str,
    )
    
    response = llm.invoke(prompt)
    
    # Parse JSON response
    try:
        # Strip any markdown code fences if present
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text.strip())
        ok = bool(parsed.get("ok", False))
        issue = str(parsed.get("issue", "")) if not ok else ""
    except (json.JSONDecodeError, KeyError):
        # If model fails to output JSON, treat as not ok
        ok = False
        issue = f"Could not parse verify output: {response.content[:200]}"
    
    return {
        "verify_ok": ok,
        "verify_issue": issue,
    }
```

### revise_node Implementation

```python
# agent/graph.py
from agent.prompts import REVISE_PROMPT

def revise_node(state: AgentState) -> dict:
    """
    Generates a revised SQL query based on the verification failure.
    
    Receives the original question, the failed SQL, and the specific issue
    identified by verify_node. Returns improved SQL.
    """
    llm = ChatOpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        api_key=os.getenv("OPENAI_API_KEY", "token-abc"),
        temperature=0.1,
        max_tokens=512,
    )
    
    result_str = "EXECUTION ERROR: " + state["execution_error"] if state["execution_error"] \
                 else json.dumps(state["execution_result"][:5], indent=2)
    
    prompt = REVISE_PROMPT.format(
        question=state["question"],
        schema=state["schema"],
        original_sql=state["current_sql"],
        issue=state["verify_issue"],
        result=result_str,
    )
    
    response = llm.invoke(prompt)
    
    # Extract SQL (strip markdown fences if present)
    sql = response.content.strip()
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        sql = sql.split("```")[1].split("```")[0].strip()
    
    return {
        "current_sql": sql,
        "iteration": state["iteration"] + 1,
        # Reset verify state for next execute
        "verify_ok": False,
        "verify_issue": "",
        "execution_result": None,
        "execution_error": None,
    }
```

### route_after_verify Implementation

```python
# agent/graph.py
from typing import Literal

def route_after_verify(state: AgentState) -> Literal["revise", "end"]:
    """
    Routes to 'end' if SQL is verified correct OR iteration cap is reached.
    Routes to 'revise' if verify failed and iterations remain.
    """
    if state["verify_ok"]:
        return "end"
    
    if state["iteration"] >= state.get("max_iterations", 3):
        # Hit iteration cap — return whatever we have
        return "end"
    
    return "revise"
```

### Graph Wiring

```python
# agent/graph.py
from langgraph.graph import StateGraph, END

def build_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("generate_sql", generate_sql_node)   # provided
    graph.add_node("execute", execute_node)               # provided
    graph.add_node("verify", verify_node)
    graph.add_node("revise", revise_node)
    
    # Linear edges
    graph.add_edge("generate_sql", "execute")
    graph.add_edge("execute", "verify")
    graph.add_edge("revise", "execute")      # revise → re-execute (not back to verify directly)
    
    # Conditional edge after verify
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "revise": "revise",
            "end": END,
        }
    )
    
    # Entry point
    graph.set_entry_point("generate_sql")
    
    return graph.compile()
```

### Prompts

```python
# agent/prompts.py

# ============================================================
# GENERATE SQL PROMPT
# ============================================================
GENERATE_SQL_PROMPT = """\
You are a SQL expert. Given a database schema and a question, write a SQL query that answers the question.

Database Schema:
{schema}

Question: {question}

Rules:
- Output ONLY the SQL query, nothing else.
- Do NOT include markdown code fences or explanation.
- Use only tables and columns from the schema above.
- If the question asks for a count, use COUNT(*) or COUNT(column).
- If the question asks for top N results, use ORDER BY and LIMIT.

SQL: /no_think"""

# ============================================================
# VERIFY PROMPT
# ============================================================
VERIFY_PROMPT = """\
You are evaluating whether a SQL result correctly answers a question.

Database Schema:
{schema}

Question: {question}

SQL Query:
{sql}

Execution Result:
{result}

Evaluate whether the execution result correctly and completely answers the question.

Consider these failure modes:
1. EXECUTION ERROR: The SQL had a syntax error (result is an error message)
2. ZERO ROWS: The result is empty when the question implies there should be results
3. WRONG COLUMNS: The result columns do not match what the question asks for
4. WRONG VALUES: The result values are clearly wrong (e.g., wrong table, wrong filter)
5. INCOMPLETE: Missing GROUP BY, missing joins, only partial data returned

Output ONLY this JSON, nothing else:
{{"ok": true}} if the result correctly answers the question
{{"ok": false, "issue": "<specific one-sentence description of what is wrong and what SQL change would fix it>"}} if not

JSON: /no_think"""

# ============================================================
# REVISE PROMPT
# ============================================================
REVISE_PROMPT = """\
You are a SQL expert. A previous SQL query failed verification. Your job is to fix it.

Database Schema:
{schema}

Question: {question}

Original SQL (which failed):
{original_sql}

Execution Result of Original SQL:
{result}

Verification Issue:
{issue}

Write a corrected SQL query that fixes the identified issue.

Rules:
- Output ONLY the corrected SQL query, nothing else.
- Do NOT include markdown code fences or explanation.
- Address the specific issue identified above.
- Use only tables and columns from the schema above.

Corrected SQL: /no_think"""
```

### Why the `/no_think` Token Matters

Appending `/no_think` to each prompt instructs Qwen3 to skip the thinking/reasoning block and output the answer directly. For SQL generation (a structured, deterministic task), thinking mode:
- Adds 500–2,000 extra output tokens per call
- Does not significantly improve SQL quality for well-structured prompts
- Multiplies latency by 5–15x depending on question complexity

At 2.5 LLM calls per agent run and 10 RPS, disabling thinking is often the difference between meeting and missing the SLO.

---

## Phase 4: Langfuse Integration

### Complete Integration Code

```python
# In your agent server entrypoint (where graph.invoke is called)
import os
from langfuse.callback import CallbackHandler

def run_agent(question: str, db: str, question_id: str = None):
    """Run the LangGraph agent with Langfuse tracing."""
    
    # Initialize callback handler (picks up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY from env)
    handler = CallbackHandler()
    
    # Build initial state
    schema = get_schema(db)  # your schema rendering function
    initial_state = {
        "question": question,
        "db": db,
        "schema": schema,
        "current_sql": "",
        "execution_result": None,
        "execution_error": None,
        "verify_ok": False,
        "verify_issue": "",
        "iteration": 0,
        "max_iterations": 3,
        "final_sql": None,
        "final_result": None,
    }
    
    # Invoke graph with Langfuse callback
    result = graph.invoke(
        initial_state,
        config={"callbacks": [handler]}
    )
    
    # Add metadata to the trace AFTER invocation
    # This is the metadata you will filter on in Phase 6
    handler.langfuse.trace(
        id=handler.get_trace_id(),
        metadata={
            "db_name": db,
            "num_iterations": result["iteration"],
            "final_ok": result["verify_ok"],
            "revise_triggered": result["iteration"] > 0,
            "question_id": question_id or "",
        },
        tags=[
            f"db:{db}",
            f"iterations:{result['iteration']}",
            "revise_triggered" if result["iteration"] > 0 else "no_revise",
        ]
    )
    
    return result
```

### Useful Metadata for Phase 6 Diagnosis

| Metadata Field | Why It's Useful in Phase 6 |
|---------------|--------------------------|
| `db_name` | Filter: is one database consistently harder/slower? |
| `num_iterations` | Filter: do 2-iteration runs have higher latency? Isolate agent overhead from model latency. |
| `final_ok` | Filter: what does a successful trace look like vs failed? Is there a latency difference? |
| `revise_triggered` | Quick filter for traces that went through the revise path |
| `question_id` | Cross-reference: find the Langfuse trace for a specific eval question that failed |

### What the Trace Waterfall Should Show

After running 10 questions, open Langfuse at `localhost:3001` and find a trace that triggered a revise. You should see:

```
trace: [question text]  (total: ~5-8s)
├── RunnableSequence  (LangGraph wrapper)
│   ├── generate_sql (1.5-2.5s)
│   │     tokens_in: ~1,847  tokens_out: ~95
│   ├── execute (0.02-0.1s)
│   ├── verify (1.0-1.8s)
│   │     tokens_in: ~2,100  tokens_out: ~40
│   ├── revise (1.2-2.0s)
│   │     tokens_in: ~2,350  tokens_out: ~120
│   └── execute (0.02-0.1s)
```

If you see only a flat trace without nested spans, the callback handler may not be propagating correctly. Check that you are passing `config={"callbacks": [handler]}` to `graph.invoke`, not to the graph compilation.

---

## Phase 5: Eval Runner

### Complete evals/run_eval.py

```python
#!/usr/bin/env python3
"""
Eval runner for HW3 text-to-SQL agent.

Reads eval_set.jsonl, calls the agent for each question,
compares results using execution accuracy, and writes
per-iteration pass rates to results/eval_baseline.json.
"""
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ============================================================
# Configuration
# ============================================================
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8001/answer")
BIRD_DATA_DIR = Path(os.getenv("BIRD_DATA_DIR", "data/bird"))
EVAL_SET_PATH = Path("evals/eval_set.jsonl")
RESULTS_DIR = Path("results")


# ============================================================
# SQL Execution
# ============================================================
def run_sql(db_path: str, sql: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Execute SQL against a SQLite database.
    
    Returns (rows, None) on success.
    Returns (None, error_str) on failure.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        # Convert Row objects to plain lists for comparison
        return [list(row) for row in rows], None
    except Exception as e:
        return None, str(e)


def canonicalize(rows: Optional[List[Any]]) -> Optional[frozenset]:
    """
    Canonicalize a result set for comparison:
    - Sort rows
    - Normalize string whitespace and case for string values
    - Convert to frozenset of tuples for order-independent comparison
    
    Returns None if rows is None (error case).
    """
    if rows is None:
        return None
    
    normalized = []
    for row in rows:
        normalized_row = []
        for val in row:
            if val is None:
                normalized_row.append(None)
            elif isinstance(val, str):
                normalized_row.append(val.strip().lower())
            elif isinstance(val, float):
                # Round floats to avoid floating-point comparison issues
                normalized_row.append(round(val, 4))
            else:
                normalized_row.append(val)
        normalized.append(tuple(normalized_row))
    
    # Use frozenset for order-independent comparison
    return frozenset(normalized)


def results_match(gold_rows: List[Any], agent_rows: List[Any]) -> bool:
    """Compare two result sets after canonicalization."""
    gold_canon = canonicalize(gold_rows)
    agent_canon = canonicalize(agent_rows)
    
    # If either is None (error), they don't match
    if gold_canon is None or agent_canon is None:
        return False
    
    return gold_canon == agent_canon


# ============================================================
# Agent Call
# ============================================================
def call_agent(question: str, db: str, question_id: str) -> Dict:
    """
    Call the agent HTTP endpoint.
    
    Returns dict with:
      - sql: final SQL produced
      - iterations: number of revise iterations (0 = first try succeeded)
      - error: error string if agent call failed
    """
    try:
        response = requests.post(
            AGENT_URL,
            json={"question": question, "db": db, "question_id": question_id},
            timeout=60,  # Agent can take up to 30s for 3 iterations
        )
        response.raise_for_status()
        data = response.json()
        return {
            "sql": data.get("sql", ""),
            "iterations": data.get("iterations", 0),
            "error": None,
        }
    except requests.exceptions.Timeout:
        return {"sql": "", "iterations": -1, "error": "timeout"}
    except Exception as e:
        return {"sql": "", "iterations": -1, "error": str(e)}


# ============================================================
# Main Eval Loop
# ============================================================
def run_eval(output_file: str = "results/eval_baseline.json"):
    """Run the full evaluation suite."""
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Load eval set
    eval_questions = []
    with open(EVAL_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_questions.append(json.loads(line))
    
    print(f"Loaded {len(eval_questions)} questions from {EVAL_SET_PATH}")
    
    # Results storage
    results = []
    
    for idx, question in enumerate(eval_questions):
        q_id = question.get("id", str(idx))
        q_text = question["question"]
        db_name = question["db"]
        gold_sql = question["gold_sql"]
        db_path = str(BIRD_DATA_DIR / db_name / f"{db_name}.sqlite")
        
        print(f"\n[{idx+1}/{len(eval_questions)}] ID: {q_id}")
        print(f"  DB: {db_name}")
        print(f"  Q: {q_text[:80]}...")
        
        # Time the full agent call
        start_time = time.time()
        agent_result = call_agent(q_text, db_name, q_id)
        elapsed = time.time() - start_time
        
        if agent_result["error"]:
            print(f"  ERROR: {agent_result['error']}")
            results.append({
                "id": q_id,
                "db": db_name,
                "question": q_text,
                "gold_sql": gold_sql,
                "agent_sql": "",
                "iterations": -1,
                "correct": False,
                "error": agent_result["error"],
                "elapsed_s": elapsed,
            })
            continue
        
        agent_sql = agent_result["sql"]
        iterations = agent_result["iterations"]
        
        # Run both SQLs
        gold_rows, gold_err = run_sql(db_path, gold_sql)
        agent_rows, agent_err = run_sql(db_path, agent_sql)
        
        # Compare
        correct = False
        if gold_err:
            print(f"  GOLD SQL ERROR: {gold_err}")
        elif agent_err:
            print(f"  AGENT SQL ERROR: {agent_err}")
        else:
            correct = results_match(gold_rows, agent_rows)
        
        status = "CORRECT" if correct else "WRONG"
        print(f"  {status} | iterations={iterations} | elapsed={elapsed:.1f}s")
        if not correct and agent_sql:
            print(f"  Agent SQL: {agent_sql[:100]}...")
        
        results.append({
            "id": q_id,
            "db": db_name,
            "question": q_text,
            "gold_sql": gold_sql,
            "agent_sql": agent_sql,
            "iterations": iterations,
            "correct": correct,
            "gold_error": gold_err,
            "agent_error": agent_err,
            "elapsed_s": elapsed,
        })
    
    # ============================================================
    # Compute Aggregated Metrics
    # ============================================================
    total = len(results)
    
    # Overall pass rate
    overall_correct = sum(1 for r in results if r["correct"])
    overall_pass_rate = overall_correct / total if total > 0 else 0.0
    
    # Per-iteration pass rates
    # "iter0": would have passed if we stopped after first generate (iterations=0)
    # "iter1": would have passed if we stopped after first revise (iterations<=1)
    # etc.
    iter_pass_rates = {}
    for max_iter in range(4):
        # Questions that passed AND used at most max_iter iterations
        eligible = [r for r in results if r["iterations"] >= 0]
        passed_at_iter = sum(
            1 for r in eligible
            if r["correct"] and r["iterations"] <= max_iter
        )
        iter_pass_rates[f"iter{max_iter}_pass_rate"] = passed_at_iter / total if total > 0 else 0.0
    
    # Pass rate breakdown by iteration depth
    iter_distribution = {}
    for n_iter in range(-1, 5):
        count = sum(1 for r in results if r["iterations"] == n_iter)
        iter_distribution[f"iterations_{n_iter}"] = count
    
    # Detailed results structure
    output = {
        "summary": {
            "total": total,
            "correct": overall_correct,
            "overall_pass_rate": round(overall_pass_rate, 4),
            **{k: round(v, 4) for k, v in iter_pass_rates.items()},
            "iteration_distribution": iter_distribution,
            "loop_improvement_pp": round(
                (overall_pass_rate - iter_pass_rates.get("iter0_pass_rate", 0)) * 100, 2
            ),
        },
        "per_question": results,
    }
    
    # Write results
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVAL RESULTS SUMMARY")
    print("="*60)
    print(f"Total questions: {total}")
    print(f"Overall pass rate: {overall_correct}/{total} = {overall_pass_rate:.1%}")
    print()
    print("Per-iteration pass rates (if we had stopped at iteration N):")
    for max_iter in range(4):
        key = f"iter{max_iter}_pass_rate"
        rate = iter_pass_rates[key]
        print(f"  After iter {max_iter}: {rate:.1%}")
    print()
    print(f"Loop improvement: +{output['summary']['loop_improvement_pp']:.1f}pp")
    print(f"\nResults written to: {output_file}")
    
    return output


if __name__ == "__main__":
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else "results/eval_baseline.json"
    run_eval(output_file)
```

### Interpreting the Results

| Pattern | Likely Cause | Investigation |
|---------|-------------|---------------|
| `iter0_pass_rate == final_pass_rate` | Loop not helping | Check traces: is verify ever triggering? If yes, is revise producing different SQL? |
| `final_pass_rate` > `iter0_pass_rate` by 5–15pp | Loop is working as intended | Good. Characterize which question types benefit. |
| `final_pass_rate` < `iter0_pass_rate` | False positives in verify | Verify is calling correct results wrong, then revise is breaking them |
| Many `iterations=-1` | Timeouts or errors | Agent server crashing under load? Check logs. |

---

## Phase 6: SLO Diagnosis — Common Patterns and Fixes

### The Load Test

```bash
uv run python load_test/driver.py --rps 10 --duration 300
```

Run this while Grafana is open. You need both the load test running and the dashboard visible simultaneously.

### Typical Diagnostic Sequence for This Workload

**Most common Phase 1 (what you will likely see with initial config):**

At 10 RPS with BF16 and default settings, the most common failure mode is memory pressure. The sequence:

1. **P95 latency begins high** (~10–15s) under load
2. **Check KV cache panel**: likely at 80–95% immediately under load
3. **Check queue**: `num_requests_waiting` growing, sometimes reaching 20–30
4. **Root cause**: At BF16 (~70GB weights), only ~8GB remains for KV cache. At 8192 max_model_len in BF16, each sequence can use up to 1.4GB of KV cache. That means only 5–6 concurrent sequences can fit — but 10 RPS requires 25+ concurrent sequences. Queue depth explodes.

**Fix sequence 1: Add FP8 quantization (biggest single improvement)**

```
Saw: P95 at 12.3s, KV cache at 95%, queue depth averaging 18
Hypothesized: BF16 weights leave insufficient KV cache for the target concurrency (10 RPS × 2.5 calls × ~2s per call = 25 concurrent requests, but only 5-6 fit in remaining HBM)
Changed: Added --quantization fp8
Result: KV cache peak dropped to 58%, queue depth under 5, P95 fell to 5.8s
```

**Fix sequence 2: Tune GPU memory utilization**

After FP8:
```
Saw: P95 at 5.8s, KV cache stabilizing at 55–65% under load, SLO at 6.0s just missed
Hypothesized: KV cache headroom is adequate but we may benefit from a few percent more
Changed: --gpu-memory-utilization from 0.90 to 0.92
Result: P95 at 5.4s. Modest improvement. Try disabling thinking mode next.
```

**Fix sequence 3: Disable Qwen3 thinking mode (often the biggest win)**

```
Saw: P95 at 5.4s, examining Langfuse traces — generate_sql calls generating 800-1200 output tokens including think blocks, verify similarly verbose
Hypothesized: Thinking mode is generating 700-900 extra tokens per call (think blocks), adding 2-4s of generation latency per agent run
Changed: Added /no_think to all three prompts
Result: generate_sql output tokens dropped from ~950 to ~90, verify from ~800 to ~35. Per-call latency dropped from 2.8s to 0.9s average. P95 E2E fell to 2.1s.
```

This third fix — disabling thinking mode — is often the most impactful and the one most students discover late. Langfuse traces are what reveal it: if you see generate_sql taking 3s and producing 900 tokens when your SQL should be 80 tokens, the thinking block is eating your latency budget.

### Working Iteration Log Template (Fill In Your Numbers)

```
ITERATION LOG

Baseline (initial config, 5-min @ 10 RPS):
P95 E2E: ___.s | KV cache: ___% | queue depth avg: ___ | tokens/sec: ___

Iteration 1 (Fri 14:23):
Saw: P95 at [___]s, KV cache at [___]%, queue depth spiking to [___]
Hypothesized: [FP8 / thinking mode / max-num-seqs / etc.] is causing [specific problem]
Changed: [specific flag + value]
P95 after: [___]s | KV cache after: [___]%
Learning: [what did the metric do? did P95 follow?]

Iteration 2 (Fri 14:51):
[same format]

Iteration 3 (Fri 15:18):
[same format]

Final state:
P95 E2E: ___s | [SLO HIT / MISSED by ___s]
Quality check: eval after tuning → ___/30 = ___% [vs baseline ___/30]
```

### When Quality Regresses After Tuning

If you enable FP8 and your eval pass rate drops:
- Run 5 questions manually and compare outputs between BF16 and FP8 serving
- FP8 quality degradation on Qwen3-30B-A3B is typically minimal (<2pp) because the model has sufficient parameters to absorb quantization noise
- If you see significant degradation, check: are you using the model's native FP8 checksum (if available) vs dynamic quantization at serving time?
- Document the regression honestly: "FP8 improved P95 by Xs but reduced eval pass rate by Ypp. The tradeoff is [acceptable / unacceptable] because [reason]."

---

## Phase 7: REPORT.md Template

```markdown
# HW3 Report: LLM Inference + Observability

## Serving Configuration

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 80GB

| Flag | Value | Rationale |
|------|-------|-----------|
| `--dtype` | bfloat16 | Compute dtype standard for H100; avoids FP16 dynamic range issues |
| `--quantization` | fp8 | H100 native FP8 Tensor Cores (2× compute vs BF16); reduces weight footprint from ~70GB to ~35GB, freeing HBM for KV cache |
| `--max-model-len` | 8192 | BIRD prompts are 1.5–3K tokens + short SQL output; native 131K context would waste KV cache budget |
| `--gpu-memory-utilization` | 0.92 | 92% of HBM for weights + KV cache; 8% headroom for CUDA overhead |
| `--max-num-seqs` | 32 | Supports target concurrency (10 RPS × 2.5 calls × ~1s avg = ~25 concurrent) with headroom |
| `--enable-chunked-prefill` | — | Smooths TPOT by interleaving 2K token prefills with decode steps |
| `--enable-prefix-caching` | — | BIRD questions hit same DB repeatedly; schema prefix (1K+ tokens) cached after first request |

Thinking mode disabled via `/no_think` suffix in all prompts. Without this, generate_sql produces 800–1,200 output tokens including think blocks; with it, ~80–120 tokens. Largest single latency improvement.

## Baseline Eval Results (Phase 5)

| Metric | Value |
|--------|-------|
| Total questions | 30 |
| iter0 pass rate | __/30 = _._% |
| iter1 pass rate | __/30 = _._% |
| iter2 pass rate | __/30 = _._% |
| Final pass rate | __/30 = _._% |
| Loop improvement | +_._pp |

**Agent Value**: The verify→revise loop improved execution accuracy by [X]pp (from [Y]% at iter0 to [Z]% at final). [X out of Y] questions that failed on the first SQL attempt were fixed by revision. The revise loop was most effective at fixing [SQL syntax errors / missing joins / wrong aggregations — choose what you observed]. In [N] cases, verify triggered a revise but the revised SQL was still incorrect, suggesting the verify prompt has false negatives for [pattern].

## SLO Diagnosis (Phase 6)

**Target**: P95 E2E agent latency < 5.0s at 10 RPS, sustained 5 minutes

**Baseline** (initial config):
- P95 E2E: ___s | KV cache: ___% | queue depth: ___

| # | Saw | Hypothesized | Changed | Result |
|---|-----|-------------|---------|--------|
| 1 | P95 ___s, KV cache ___% | [your hypothesis] | [your change] | P95 ___s → ___s |
| 2 | [next observation] | [hypothesis] | [change] | P95 ___s → ___s |
| 3 | [next observation] | [hypothesis] | [change] | P95 ___s → ___s |

**Final**: P95 = ___s. **[SLO HIT / MISSED by ___s].**

Post-tuning eval: __/30 = ___% [improvement / regression of ___pp vs baseline].

[If missed]: The remaining gap of ___s reflects a fundamental constraint: at 10 RPS with 2.5 sequential LLM calls per agent run, the minimum E2E latency floor is approximately ___s (___s per LLM call × 2.5 calls). Closing this gap would require either disaggregated prefill/decode to reduce per-call TTFT, or parallelizing some agent steps.

## What I'd Do With More Time

1. **[Specific item]**: [expected improvement and why]. Example: "Profile the attention kernel with Nsight Systems to determine whether the bottleneck is expert routing dispatch overhead or KV cache memory bandwidth — the diagnosis changes which optimization is worth pursuing."

2. **[Specific item]**: [expected improvement and why].

3. **[Specific item]**: [expected improvement and why].
```

---

## Quick Reference: Key vLLM Metrics

| Metric Name | Type | What it measures |
|------------|------|-----------------|
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request latency from arrival to final token |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT — measures prefill phase |
| `vllm:inter_token_latency_seconds` | Histogram | Per-token generation latency — measures decode phase |
| `vllm:gpu_cache_usage_perc` | Gauge | KV cache utilization (0–1) |
| `vllm:gpu_prefix_cache_hit_rate` | Gauge | Fraction of KV cache served from prefix cache |
| `vllm:num_requests_running` | Gauge | Currently executing requests |
| `vllm:num_requests_waiting` | Gauge | Requests in queue waiting to start |
| `vllm:generation_tokens_total` | Counter | Total tokens generated (use `rate()` for tokens/sec) |
| `vllm:prompt_tokens_total` | Counter | Total prompt tokens processed (use `rate()` for tokens/sec) |
| `vllm:request_success_total` | Counter | Completed requests (use `rate()` for RPS) |

**PromQL pattern for all histogram percentile panels:**
```promql
histogram_quantile(0.95,
  sum(rate(METRIC_NAME_bucket[2m])) by (le)
)
```

Replace `METRIC_NAME` with any histogram metric above, add `_bucket` suffix.
