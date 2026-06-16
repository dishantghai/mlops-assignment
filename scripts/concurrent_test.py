import urllib.request, json, time, threading
import sys
sys.path.insert(0, ".")
from agent.schema import render_schema

schema = render_schema("formula_1")
CONCURRENCY = 10

questions = [
    "What is the fastest lap time ever recorded and which driver achieved it?",
    "Which constructor won the most championships between 2010 and 2020?",
    "List the top 5 drivers by total race wins of all time.",
    "Which circuit has hosted the most Formula 1 races?",
    "What was the average pit stop duration in the 2019 season?",
    "Which driver had the most pole positions in 2018?",
    "List all circuits in the United Kingdom.",
    "Who finished second in the 2021 constructors championship?",
    "What is the total number of laps in the Monaco Grand Prix circuit?",
    "Which driver scored the most points in a single season?",
]

prompt_template = """You are a SQL expert. Write a SQL query to answer the question. Output only SQL.

{schema}

Question: {question}"""

results = []
lock = threading.Lock()

def fire(idx, question):
    payload = {
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages": [{"role": "user", "content": prompt_template.format(schema=schema, question=question)}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        elapsed = time.time() - t0
        tokens = resp["usage"]["completion_tokens"]
        with lock:
            results.append({"idx": idx, "latency": elapsed, "tokens": tokens, "ok": True})
            print(f"  [{idx:02d}] {elapsed:.2f}s  {tokens} tokens")
    except Exception as e:
        elapsed = time.time() - t0
        with lock:
            results.append({"idx": idx, "latency": elapsed, "ok": False, "error": str(e)})
            print(f"  [{idx:02d}] FAILED after {elapsed:.2f}s: {e}")

print(f"Firing {CONCURRENCY} parallel requests...")
t_start = time.time()
threads = [threading.Thread(target=fire, args=(i, questions[i])) for i in range(CONCURRENCY)]
for t in threads: t.start()
for t in threads: t.join()
total = time.time() - t_start

ok = [r for r in results if r["ok"]]
latencies = sorted(r["latency"] for r in ok)

print(f"\n--- Summary ({CONCURRENCY} concurrent requests) ---")
print(f"Wall clock (all done): {total:.2f}s")
print(f"Successful:  {len(ok)}/{CONCURRENCY}")
if latencies:
    print(f"Latency P50: {latencies[len(latencies)//2]:.2f}s")
    print(f"Latency P95: {latencies[int(len(latencies)*0.95)]:.2f}s")
    print(f"Latency max: {latencies[-1]:.2f}s")
    print(f"Latency min: {latencies[0]:.2f}s")

print("\nNow pull metrics:")
print("  curl -s localhost:8000/metrics | grep -v '^#' | grep -E 'ttft|time_to_first_token_seconds_(sum|count)|e2e_request_latency_seconds_(sum|count)'")
