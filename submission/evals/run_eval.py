"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"

# Must match MAX_ITERATIONS in agent/graph.py
MAX_ITER_SLOTS = 3


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implementation ---------------------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    iter_correct is a list of MAX_ITER_SLOTS booleans with carry-forward:
    if the agent stopped at iteration j < k, slot k inherits slot j's result
    (the agent was done; whatever it had is what would have been served).
    """
    db_id = question["db_id"]
    q_text = question["question"]
    gold_sql = question["gold_sql"]
    # Stable ID for cross-referencing with Langfuse traces.
    question_id = question.get("id", f"{db_id}__{abs(hash(q_text)) % 100000:05d}")

    # 1. Run gold SQL — if this fails it's a data quality issue, not agent issue.
    gold_ok, gold_rows, gold_error = run_sql(db_id, gold_sql)
    if not gold_ok:
        print(f"  WARNING: gold SQL failed for {question_id}: {gold_error}", flush=True)

    # 2. Call the agent.
    agent_sql = ""
    agent_rows = None
    iterations = 0
    agent_ok = False
    agent_error = None
    history: list[dict] = []
    try:
        resp = httpx.post(
            agent_url,
            json={"question": q_text, "db": db_id, "question_id": question_id},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        agent_sql = data.get("sql", "")
        iterations = data.get("iterations", 0)
        agent_ok = data.get("ok", False)
        agent_error = data.get("error")
        history = data.get("history", [])
        # Re-execute agent SQL against the DB to get canonical rows.
        # (The rows in the response are correct too, but re-running avoids
        # float repr differences and gives us a single canonicalization path.)
        if agent_ok and agent_sql:
            _, agent_rows, _ = run_sql(db_id, agent_sql)
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"

    # 3. Final correctness.
    correct = matches(gold_rows, agent_rows)

    # 4. Per-iteration correctness with carry-forward.
    # Extract each SQL the agent attempted, in order (generate_sql, then revise nodes).
    iter_sqls: list[str] = [
        h["sql"] for h in history
        if h.get("node") in ("generate_sql", "revise") and h.get("sql")
    ]

    iter_correct: list[bool] = []
    for i, sql in enumerate(iter_sqls):
        if i == len(iter_sqls) - 1:
            # Final attempt — result already computed above.
            iter_correct.append(correct)
        else:
            # Earlier attempt — run it against gold to see if it was already correct.
            _, rows, _ = run_sql(db_id, sql)
            iter_correct.append(matches(gold_rows, rows))

    # Carry-forward: pad to MAX_ITER_SLOTS so summarize() can index uniformly.
    if iter_correct:
        while len(iter_correct) < MAX_ITER_SLOTS:
            iter_correct.append(iter_correct[-1])
    else:
        # Agent returned no history (HTTP error path) — all slots false.
        iter_correct = [False] * MAX_ITER_SLOTS

    return {
        "question_id": question_id,
        "db_id": db_id,
        "question": q_text,
        "gold_sql": gold_sql,
        "agent_sql": agent_sql,
        "correct": correct,
        "agent_ok": agent_ok,
        "iterations": iterations,
        "agent_error": agent_error,
        "gold_error": gold_error,
        "iter_correct": iter_correct,  # [iter0, iter1, iter2], carry-forward applied
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results into pass rates and loop diagnostics.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    if total == 0:
        return {}

    # Exclude questions where gold SQL itself failed — they cannot be scored.
    scoreable = [r for r in results if r["gold_error"] is None]
    n = len(scoreable)
    skipped = total - n

    overall = sum(r["correct"] for r in scoreable)
    iter0 = sum(r["iter_correct"][0] for r in scoreable)
    iter1 = sum(r["iter_correct"][1] for r in scoreable)
    iter2 = sum(r["iter_correct"][2] for r in scoreable)

    def pct(num: int, den: int) -> float:
        return round(num / den * 100, 1) if den else 0.0

    loop_improvement_pp = round(pct(overall, n) - pct(iter0, n), 1)

    # Per-DB breakdown.
    db_counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in scoreable:
        db_counts[r["db_id"]]["total"] += 1
        db_counts[r["db_id"]]["correct"] += int(r["correct"])
    per_db = {
        db: {
            "total": v["total"],
            "correct": v["correct"],
            "pass_rate_pct": pct(v["correct"], v["total"]),
        }
        for db, v in sorted(db_counts.items())
    }

    # Revision loop diagnostics.
    revised = sum(1 for r in scoreable if r["iterations"] > 1)
    revision_helped = sum(
        1 for r in scoreable
        if r["iterations"] > 1 and r["correct"] and not r["iter_correct"][0]
    )
    revision_hurt = sum(
        1 for r in scoreable
        if r["iterations"] > 1 and not r["correct"] and r["iter_correct"][0]
    )

    return {
        "total_questions": total,
        "scoreable": n,
        "skipped_gold_error": skipped,
        "overall_correct": overall,
        "overall_pass_rate_pct": pct(overall, n),
        "iter0_correct": iter0,
        "iter0_pass_rate_pct": pct(iter0, n),
        "iter1_correct": iter1,
        "iter1_pass_rate_pct": pct(iter1, n),
        "iter2_correct": iter2,
        "iter2_pass_rate_pct": pct(iter2, n),
        "loop_improvement_pp": loop_improvement_pp,
        "questions_revised": revised,
        "revision_helped": revision_helped,   # revise turned a wrong into a right
        "revision_hurt": revision_hurt,        # revise turned a right into a wrong
    } | {"per_db": per_db}


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
