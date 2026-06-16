import urllib.request, json, time, sys
sys.path.insert(0, ".")
from agent.schema import render_schema

# Use formula_1 — our largest schema (1,165 tokens), worst-case prompt
db = "formula_1"
schema = render_schema(db)
question = "What is the fastest lap time ever recorded and which driver achieved it?"

prompt = f"""You are a SQL expert. Given the database schema below, write a SQL query to answer the question.
Output only the SQL query, no explanation.

{schema}

Question: {question}"""

payload = {
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 300,
    "temperature": 0.0,
}

print(f"Prompt tokens (est): ~{len(prompt)//4}")
print(f"Schema chars: {len(schema)}")
print("Sending request...")

req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

t0 = time.time()
resp = json.loads(urllib.request.urlopen(req).read())
elapsed = time.time() - t0

usage = resp["usage"]
content = resp["choices"][0]["message"]["content"]

print(f"\n--- Results ---")
print(f"Wall time:         {elapsed:.2f}s")
print(f"Prompt tokens:     {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"Thinking triggered: {'<think>' in content}")
print(f"\nSQL output:\n{content}")
