import urllib.request, json, time
import sys
sys.path.insert(0, ".")
from agent.schema import render_schema

schema = render_schema("formula_1")
question = "Which constructor won the most championships between 2010 and 2020, and how many did they win?"

prompt = f"""You are a SQL expert. Given the database schema below, write a SQL query to answer the question.
Output only the SQL query, no explanation.

{schema}

Question: {question}"""

def test(label, temperature, extra_body=None):
    payload = {
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": temperature,
    }
    if extra_body:
        payload.update(extra_body)

    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    elapsed = time.time() - t0

    usage = resp["usage"]
    content = resp["choices"][0]["message"]["content"]
    thinking = "<think>" in content

    print(f"\n=== {label} ===")
    print(f"Temperature:       {temperature}")
    print(f"Wall time:         {elapsed:.2f}s")
    print(f"Prompt tokens:     {usage['prompt_tokens']}")
    print(f"Completion tokens: {usage['completion_tokens']}")
    print(f"Thinking triggered: {thinking}")
    if thinking:
        think_end = content.find("</think>")
        think_tokens_est = len(content[:think_end]) // 4
        print(f"Think block est tokens: ~{think_tokens_est}")
    print(f"Output (first 300 chars):\n{content[:300]}")

# Test 1: Default temperature from model (0.7) — thinking likely to fire
test("Default temp (0.7) — no override", temperature=0.7)

# Test 2: temperature=0.0 — thinking less likely
test("temperature=0.0", temperature=0.0)
