"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, question_id?, tags?} and returns
the agent's final SQL, the result rows, and per-iteration history.

Phase 4: Langfuse tracing is enabled when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are present in the environment. A fresh CallbackHandler
is created per request so that traces do not cross-contaminate. After
graph.ainvoke() completes, the trace is updated with agent-level metadata
tags that Phase 6 uses for filtering in Langfuse.

Phase 6: endpoint is async so the event loop handles concurrent requests
without exhausting a thread pool. graph.ainvoke() is used instead of
graph.invoke() — LangGraph runs sync nodes in a thread executor internally.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent.graph import AgentState, graph  # noqa: E402

# True when both Langfuse keys are present in the environment.
_LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)


app = FastAPI()


class AnswerRequest(BaseModel):
    question: str
    db: str
    # Optional stable identifier for the question; used as a Langfuse
    # metadata tag so eval results can be cross-referenced with traces.
    question_id: str | None = None
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    # Fresh handler per request — prevents trace state leaking between calls.
    handler: Any = None
    if _LANGFUSE_ENABLED:
        from langfuse.langchain import CallbackHandler
        handler = CallbackHandler()

    state = AgentState(question=req.question, db_id=req.db)
    config: dict[str, Any] = {
        "callbacks": [handler] if handler is not None else [],
        "metadata": req.tags,
    }
    try:
        # ainvoke lets the event loop multiplex hundreds of concurrent agent
        # runs without blocking — sync nodes inside the graph are offloaded
        # to a thread executor by LangGraph automatically.
        final = await graph.ainvoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    # ── Phase 4: post-run trace metadata ──────────────────────────────────────
    # iteration==1 → only generate_sql ran (no revise)
    # iteration>=2 → at least one revise cycle fired
    #
    # langfuse 4.x dropped Langfuse.trace(); use the REST upsert endpoint
    # instead. POST /api/public/traces with the existing trace ID merges our
    # fields into the trace that the LangChain callback already created.
    if handler is not None:
        try:
            handler._langfuse_client.flush()
            lf_host = os.environ.get("LANGFUSE_HOST", "http://localhost:3001")
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{lf_host}/api/public/traces",
                    auth=(
                        os.environ["LANGFUSE_PUBLIC_KEY"],
                        os.environ["LANGFUSE_SECRET_KEY"],
                    ),
                    json={
                        "id": handler.last_trace_id,
                        "name": "agent_run",
                        "metadata": {
                            "db_name": req.db,
                            "num_iterations": iteration,
                            "final_ok": bool(final.get("verify_ok", False)),
                            "revise_triggered": iteration > 1,
                            "question_id": req.question_id,
                        },
                        "tags": [req.db],
                    },
                    timeout=5.0,
                )
        except Exception:
            pass  # tracing must never break the answer endpoint

    if execution is None:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
