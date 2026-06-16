"""Take Grafana dashboard screenshots for Phase 2 submission.

Starts a background load test (10 RPS), waits for panels to fill, then
captures 6 screenshots to screenshots/.

Run:
    uv run python scripts/take_screenshots.py
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS = ROOT / "screenshots"
GRAFANA = "http://localhost:3000"
PROMETHEUS = "http://localhost:9090"
DASHBOARD_UID = "vllm-serving"
DASHBOARD_SLUG = "vllm-serving-e28094-hw3"  # slug from Grafana API

PANEL_IDS = {
    "e2e_latency": 1,
    "ttft": 2,
    "itl": 3,
    "queue_memory_running": 4,
    "queue_memory_wait": 5,
    "queue_memory_kv": 6,
    "throughput_req": 7,
    "throughput_tok": 8,
    "throughput_preempt": 9,
}

# Time window: 30m captures both current load test + prior run history
TIME_PARAMS = "from=now-30m&to=now&refresh=5s"


def dashboard_url(extra: str = "") -> str:
    base = f"{GRAFANA}/d/{DASHBOARD_UID}/{DASHBOARD_SLUG}?orgId=1&{TIME_PARAMS}"
    return base + extra


def panel_url(panel_id: int) -> str:
    return dashboard_url(f"&viewPanel={panel_id}")


async def grafana_login(page) -> None:
    """Log into Grafana, handling the change-password prompt."""
    await page.goto(f"{GRAFANA}/login", wait_until="networkidle", timeout=20000)
    await page.fill('input[name="user"]', "admin")
    await page.fill('input[name="password"]', "admin")
    await page.click('button[type="submit"]')
    await asyncio.sleep(3)

    # Grafana 11 shows "Update your password" on first login — click Skip
    skip = page.locator("a:has-text('Skip'), button:has-text('Skip')")
    if await skip.count() > 0:
        await skip.first.click()
        await asyncio.sleep(2)

    # Confirm we're logged in by checking we're no longer on /login
    if "/login" in page.url:
        raise RuntimeError(f"Login failed, still at: {page.url}")
    await asyncio.sleep(1)


async def wait_for_panels(page, timeout_ms: int = 20000) -> None:
    """Wait until Grafana panels finish rendering their charts."""
    # 1. Wait for any loading spinners to disappear
    try:
        await page.wait_for_selector('[data-testid="panel-loading"]', state="detached", timeout=timeout_ms)
    except Exception:
        pass
    try:
        await page.wait_for_selector('.panel-loading', state="detached", timeout=5000)
    except Exception:
        pass
    # 2. Wait for canvas or SVG elements (chart render)
    try:
        await page.wait_for_selector("canvas, svg.graph-svg", state="attached", timeout=timeout_ms)
    except Exception:
        pass
    # 3. Fixed buffer for JS chart paint (Grafana renders async)
    await asyncio.sleep(5)


async def take_screenshots(load_test_proc: subprocess.Popen) -> None:
    SCREENSHOTS.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        ctx = await browser.new_context(
            viewport={"width": 1600, "height": 4000},  # tall enough for all 5 rows
            device_scale_factor=1.5,
        )
        page = await ctx.new_page()

        # Login to Grafana first (session cookie persists for this context)
        print("  Logging into Grafana...", flush=True)
        await grafana_login(page)

        # ── Screenshot 1: Full dashboard overview ─────────────────────────────
        # Use tall viewport (4000px) so all panels render in the initial DOM —
        # full_page=True unmounts off-screen Grafana panels, so avoid it.
        print("  [1/6] Full dashboard overview...", flush=True)
        await page.goto(dashboard_url(), wait_until="networkidle", timeout=30000)
        await wait_for_panels(page)
        await page.screenshot(path=str(SCREENSHOTS / "01_dashboard_overview.png"))

        # ── Screenshot 2: E2E latency panel (expanded, SLO line visible) ──────
        print("  [2/6] E2E latency panel (expanded)...", flush=True)
        await page.goto(panel_url(PANEL_IDS["e2e_latency"]), wait_until="networkidle", timeout=30000)
        await wait_for_panels(page)
        await page.screenshot(path=str(SCREENSHOTS / "02_e2e_latency_slo.png"))

        # ── Screenshot 3: TTFT panel ───────────────────────────────────────────
        print("  [3a/6] TTFT panel...", flush=True)
        await page.goto(panel_url(PANEL_IDS["ttft"]), wait_until="networkidle", timeout=30000)
        await wait_for_panels(page)
        await page.screenshot(path=str(SCREENSHOTS / "03a_ttft.png"))

        # ── Screenshot 3b: ITL panel ───────────────────────────────────────────
        print("  [3b/6] ITL panel...", flush=True)
        await page.goto(panel_url(PANEL_IDS["itl"]), wait_until="networkidle", timeout=30000)
        await wait_for_panels(page)
        await page.screenshot(path=str(SCREENSHOTS / "03b_itl.png"))

        # ── Screenshot 4: Queue & Memory row ──────────────────────────────────
        print("  [4/6] Queue & Memory row...", flush=True)
        await page.goto(dashboard_url(), wait_until="networkidle", timeout=30000)
        await wait_for_panels(page)
        row_el = page.locator("text=Queue & Memory")
        if await row_el.count() > 0:
            await row_el.first.scroll_into_view_if_needed()
            await asyncio.sleep(2)
        await page.screenshot(path=str(SCREENSHOTS / "04_queue_memory_row.png"))

        # ── Screenshot 5: Throughput row ───────────────────────────────────────
        print("  [5/6] Throughput row...", flush=True)
        await page.goto(dashboard_url(), wait_until="networkidle", timeout=30000)
        await wait_for_panels(page)
        row_el = page.locator("text=Throughput")
        if await row_el.count() > 0:
            await row_el.first.scroll_into_view_if_needed()
            await asyncio.sleep(2)
        await page.screenshot(path=str(SCREENSHOTS / "05_throughput_row.png"))

        # ── Screenshot 6: Prometheus targets page ─────────────────────────────
        print("  [6/6] Prometheus targets page...", flush=True)
        await page.goto(f"{PROMETHEUS}/targets", wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)
        await page.screenshot(path=str(SCREENSHOTS / "06_prometheus_targets.png"))

        await browser.close()

    print("\nAll screenshots saved to screenshots/:")
    for f in sorted(SCREENSHOTS.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")


async def main() -> None:
    print("Starting load test (10 RPS × 240s) in background...", flush=True)
    load_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "--help"  # placeholder — replaced below
        ]
    )
    load_proc.kill()

    # Actually start the real load test via uv run
    load_proc = subprocess.Popen(
        [
            "uv", "run", "python", "scripts/vllm_load_test.py",
            "--rps", "10",
            "--duration", "240",
            "--out", "results/phase2_screenshots.json",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print("Waiting 50s for metrics to accumulate in Grafana...", flush=True)
    for i in range(50, 0, -5):
        print(f"  {i}s remaining...", flush=True)
        await asyncio.sleep(5)

    print("\nCapturing screenshots...", flush=True)
    try:
        await take_screenshots(load_proc)
    finally:
        if load_proc.poll() is None:
            load_proc.terminate()
            load_proc.wait(timeout=5)

    print("\nDone. Load test terminated.")


if __name__ == "__main__":
    asyncio.run(main())
