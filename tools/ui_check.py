"""Drive the review gate in a headless browser: screenshot + interaction test.

The screenshots are how UI changes get eyeballed without a human clicking
around; the assertions are the regression net. Run it after any change to
tools/review_server.py.

Usage:
  .venv/bin/python tools/ui_check.py [--workdir work_jeff] [--out DIR]

Writes: 01_loaded.png, 02_shot_selected.png, 03_annotated.png into --out,
and verifies the full annotate flow ends up in review_notes.json.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="work_jeff")
    ap.add_argument("--out", default="/tmp/review_ui_check")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)
    notes_path = workdir / "review_notes.json"
    stash = None
    if notes_path.exists():  # never clobber real annotations
        stash = notes_path.read_text()

    port = free_port()
    srv = subprocess.Popen(
        [sys.executable, "tools/review_server.py",
         "--workdir", args.workdir, "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    failures = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(f"http://127.0.0.1:{port}/")
            page.wait_for_selector(".shot")
            page.screenshot(path=out / "01_loaded.png")

            n_shots = page.locator(".shot").count()
            if n_shots < 2:
                failures.append(f"expected shots in strip, got {n_shots}")

            # click shot 3 -> selected, header updates, video seeks
            page.locator(".shot").nth(3).click()
            page.wait_for_timeout(300)
            header = page.locator("#sel").inner_text()
            if not header.startswith("shot 3"):
                failures.append(f"selection header wrong: {header!r}")
            if "current" not in page.locator("#shot3").get_attribute("class"):
                failures.append("clicked shot not highlighted")
            page.screenshot(path=out / "02_shot_selected.png")

            # tag it + write a note -> autosaves to review_notes.json
            page.locator("#chips button[data-tag=reroll]").click()
            page.locator("#note").fill("ui_check: remove the laugh here")
            page.wait_for_timeout(1200)  # debounce + POST
            saved = json.loads(notes_path.read_text())
            entry = saved.get("ep01", {}).get("3", {})
            if "reroll" not in entry.get("tags", []):
                failures.append(f"reroll tag not saved: {entry}")
            if "remove the laugh" not in entry.get("note", ""):
                failures.append(f"note not saved: {entry}")
            if "noted" not in page.locator("#shot3").get_attribute("class"):
                failures.append("annotated shot not marked in strip")
            page.screenshot(path=out / "03_annotated.png")

            # keyboard flow: -> advances selection, S tags slomo
            page.locator("body").click(position={"x": 5, "y": 5})
            page.keyboard.press("ArrowRight")
            page.wait_for_timeout(200)
            if not page.locator("#sel").inner_text().startswith("shot 4"):
                failures.append("ArrowRight didn't advance selection")
            page.keyboard.press("s")
            page.wait_for_timeout(1200)
            saved = json.loads(notes_path.read_text())
            if "slomo" not in saved.get("ep01", {}).get("4", {}).get("tags", []):
                failures.append("S key didn't tag slomo")

            # apply button present and status endpoint sane (never click
            # it here - it would really re-synthesize)
            if not page.locator("#apply").is_visible():
                failures.append("Apply & re-render button missing")
            status = page.evaluate(
                "fetch('/apply/status').then(r => r.json())")
            if status.get("running") is not False:
                failures.append(f"apply status endpoint odd: {status}")

            # QA -> notes integration: a finding written by qa_takes must
            # surface in the GUI as a pre-tagged re-roll in the queue
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from qa_takes import write_notes
            write_notes([(str(workdir), 9, ["laughter (synthetic ui_check)"])])
            page.reload()
            page.wait_for_selector(".shot")
            if "noted" not in page.locator("#shot9").get_attribute("class"):
                failures.append("QA finding not marked on shot in strip")
            queue = page.locator("#qlist").inner_text()
            if "QA: laughter" not in queue or "shot 9" not in queue:
                failures.append(f"QA finding missing from queue: {queue!r}")
            page.locator(".shot").nth(9).click()
            page.wait_for_timeout(200)
            if "on" not in page.locator(
                    "#chips button[data-tag=reroll]").get_attribute("class"):
                failures.append("QA re-roll tag not active on flagged shot")
            page.screenshot(path=out / "04_qa_prefill.png")

            # video element must be playable (Range support working)
            ready = page.evaluate("document.querySelector('video').readyState")
            if ready < 1:
                failures.append(f"video not loading (readyState={ready})")
            browser.close()
    finally:
        srv.terminate()
        if stash is not None:
            notes_path.write_text(stash)
        elif notes_path.exists():
            notes_path.unlink()

    print(f"screenshots -> {out}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("all UI checks passed")


if __name__ == "__main__":
    main()
