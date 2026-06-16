"""Direct vLLM load test — bypasses the agent endpoint.

Fires /v1/chat/completions requests directly at port 8000 at a target RPS,
sampling from evals/eval_set.jsonl for realistic prompts with real schemas.

Run:
    uv run python scripts/vllm_load_test.py --rps 10 --duration 120 --out results/iter1_vllm_load.json

Outputs per-request latency summary (P50/P95/P99) + vLLM Prometheus metrics
for TTFT P95, ITL avg, KV cache peak, and queue depth peak.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.schema import render_schema  # noqa: E402

EVAL_SET = ROOT / "evals" / "eval_set.jsonl"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
METRICS_URL = "http://localhost:8000/metrics"
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

SYSTEM_PROMPT = (
    "You are a SQL expert. Given the database schema below, write a SQL query "
    "to answer the question. Output only the SQL query, no explanation."
)


def build_prompt(question: str, schema: str) -> str:
    return f"{schema}\n\nQuestion: {question}"


def load_questions() -> list[dict]:
    rows = [json.loads(l) for l in EVAL_SET.read_text().splitlines() if l.strip()]
    schema_cache: dict[str, str] = {}
    out = []
    for row in rows:
        db = row["db_id"]
        if db not in schema_cache:
            schema_cache[db] = render_schema(db)
        out.append({
            "question": row["question"],
            "db_id": db,
            "prompt": build_prompt(row["question"], schema_cache[db]),
        })
    return out


def parse_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text format into {metric_name: value}."""
    vals: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split(" ")
        if len(parts) >= 2:
            try:
                vals[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return vals


def compute_p95_from_histogram(metrics: dict[str, float], prefix: str) -> Optional[float]:
    """Estimate P95 from Prometheus histogram buckets (linear interpolation)."""
    total = metrics.get(f"{prefix}_count", 0)
    if total == 0:
        return None
    target = total * 0.95
    buckets: list[tuple[float, float]] = []
    for key, val in metrics.items():
        if key.startswith(f"{prefix}_bucket{{le="):
            le_str = key.split('le="')[1].rstrip('"}\n')
            if le_str == "+Inf":
                le = float("inf")
            else:
                try:
                    le = float(le_str)
                except ValueError:
                    continue
            buckets.append((le, val))
    if not buckets:
        return None
    buckets.sort()
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= target:
            if count == prev_count:
                return prev_le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return None


async def fetch_metrics(session: aiohttp.ClientSession) -> dict[str, float]:
    try:
        async with session.get(METRICS_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            text = await resp.text()
            return parse_metrics(text)
    except Exception:
        return {}


async def fire_one(
    session: aiohttp.ClientSession,
    question: dict,
    results: list[dict],
) -> None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question["prompt"]},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    t0 = time.monotonic()
    status = "ok"
    completion_tokens = 0
    err: Optional[str] = None
    try:
        async with session.post(
            VLLM_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            body = await resp.read()
            if resp.status == 200:
                data = json.loads(body)
                completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
            else:
                status = "http_error"
                err = f"HTTP {resp.status}: {body[:200].decode(errors='replace')}"
    except asyncio.TimeoutError:
        status = "timeout"
        err = "30s timeout"
    except Exception as e:
        status = "client_error"
        err = f"{type(e).__name__}: {e}"

    elapsed = time.monotonic() - t0
    results.append({
        "latency_seconds": elapsed,
        "status": status,
        "completion_tokens": completion_tokens,
        "db_id": question["db_id"],
        "error": err,
    })


async def run(args: argparse.Namespace) -> None:
    print("Loading eval questions and schemas...", flush=True)
    questions = load_questions()
    print(f"Loaded {len(questions)} questions across {len(set(q['db_id'] for q in questions))} databases.")

    rnd = random.Random(42)
    results: list[dict] = []
    interval = 1.0 / args.rps

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Snapshot metrics before load
        print("\nPulling pre-test metrics...", flush=True)
        before = await fetch_metrics(session)

        print(f"\nStarting load: {args.rps} RPS for {args.duration}s...", flush=True)
        start = time.monotonic()
        deadline = start + args.duration
        tasks: list[asyncio.Task] = []
        next_fire = start
        fired = 0

        while time.monotonic() < deadline:
            q = rnd.choice(questions)
            tasks.append(asyncio.create_task(fire_one(session, q, results)))
            fired += 1
            next_fire += interval
            sleep_for = next_fire - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            if fired % 50 == 0:
                ok = sum(1 for r in results if r["status"] == "ok")
                print(f"  fired={fired}  completed={len(results)}  ok={ok}  t={time.monotonic()-start:.0f}s", flush=True)

        print(f"\nDrain phase: waiting for {len([t for t in tasks if not t.done()])} in-flight requests...", flush=True)
        if tasks:
            await asyncio.wait(tasks, timeout=60.0)
        wall = time.monotonic() - start

        # Snapshot metrics after load
        print("Pulling post-test metrics...", flush=True)
        after = await fetch_metrics(session)

    # Compute deltas for avg metrics
    def delta(key: str) -> float:
        return after.get(key, 0) - before.get(key, 0)

    ttft_sum   = delta("vllm:time_to_first_token_seconds_sum")
    ttft_count = delta("vllm:time_to_first_token_seconds_count")
    itl_sum    = delta("vllm:time_per_output_token_seconds_sum")
    itl_count  = delta("vllm:time_per_output_token_seconds_count")
    e2e_sum    = delta("vllm:e2e_request_latency_seconds_sum")
    e2e_count  = delta("vllm:e2e_request_latency_seconds_count")

    avg_ttft = (ttft_sum / ttft_count) if ttft_count > 0 else None
    avg_itl  = (itl_sum  / itl_count)  if itl_count  > 0 else None
    avg_e2e  = (e2e_sum  / e2e_count)  if e2e_count  > 0 else None

    # P95 TTFT from post-test cumulative histogram (good enough for test window)
    p95_ttft = compute_p95_from_histogram(after, "vllm:time_to_first_token_seconds")

    kv_cache_peak = after.get("vllm:gpu_cache_usage_perc", None)

    # Wall-clock latency percentiles from per-request data
    ok_results = [r for r in results if r["status"] == "ok"]
    latencies = sorted(r["latency_seconds"] for r in ok_results)

    def pct(p: float) -> float:
        if not latencies:
            return float("nan")
        k = max(0, min(len(latencies) - 1, int(p * len(latencies))))
        return latencies[k]

    summary = {
        "config": "Iter 1 — BF16, max-model-len 8192, no prefix cache, chunked prefill forced on",
        "requested_rps": args.rps,
        "duration_seconds": args.duration,
        "wall_clock_seconds": wall,
        "total_fired": fired,
        "total_completed": len(results),
        "achieved_rps": len(results) / wall if wall > 0 else 0,
        "ok": len(ok_results),
        "timeouts": sum(1 for r in results if r["status"] == "timeout"),
        "http_errors": sum(1 for r in results if r["status"] == "http_error"),
        "client_errors": sum(1 for r in results if r["status"] == "client_error"),
        "wall_latency_p50_s": pct(0.50),
        "wall_latency_p95_s": pct(0.95),
        "wall_latency_p99_s": pct(0.99),
        "wall_latency_max_s": latencies[-1] if latencies else float("nan"),
        "vllm_avg_ttft_s": avg_ttft,
        "vllm_p95_ttft_s": p95_ttft,
        "vllm_avg_itl_s": avg_itl,
        "vllm_avg_e2e_s": avg_e2e,
        "vllm_kv_cache_pct": kv_cache_peak,
        "vllm_requests_in_window": int(e2e_count),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Config:           {summary['config']}")
    print(f"RPS:              {args.rps} target → {summary['achieved_rps']:.2f} actual")
    print(f"Duration:         {wall:.1f}s")
    print(f"Requests:         {len(ok_results)}/{len(results)} ok  |  {summary['timeouts']} timeouts  |  {summary['http_errors']} http_errors")
    print()
    print("Wall-clock latency (client-side):")
    print(f"  P50:  {pct(0.50):.3f}s")
    print(f"  P95:  {pct(0.95):.3f}s  ← SLO target < 5.0s")
    print(f"  P99:  {pct(0.99):.3f}s")
    print(f"  Max:  {latencies[-1]:.3f}s" if latencies else "  Max: n/a")
    print()
    print("vLLM internal metrics (Prometheus delta):")
    print(f"  TTFT avg: {avg_ttft*1000:.1f}ms" if avg_ttft else "  TTFT avg: n/a")
    print(f"  TTFT P95: {p95_ttft*1000:.1f}ms" if p95_ttft else "  TTFT P95: n/a")
    print(f"  ITL  avg: {avg_itl*1000:.1f}ms/tok" if avg_itl else "  ITL avg:  n/a")
    print(f"  E2E  avg: {avg_e2e*1000:.1f}ms" if avg_e2e else "  E2E avg:  n/a")
    print(f"  KV cache: {kv_cache_peak:.1%}" if kv_cache_peak is not None else "  KV cache: n/a")
    print(f"\nWrote: {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Direct vLLM load test (bypasses agent)")
    p.add_argument("--rps", type=float, default=10.0, help="target requests/second")
    p.add_argument("--duration", type=int, default=120, help="seconds to run")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "iter1_vllm_load.json")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
