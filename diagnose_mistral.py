"""
Standalone diagnostic for the Mistral 429 issue in the Ambient Lighting Engine.

Run this DIRECTLY on the machine that runs webapp.py (not through the app):

    python diagnose_mistral.py

It does NOT import engine.py or ai_lighting_engine.py, and does NOT touch
your running app or its throttle state at all — it's a completely separate,
minimal script that talks to Mistral directly so we can see exactly what
Mistral's servers actually say, instead of inferring it from behavior.

What it checks, in order:
  1. Whether more than one webapp.py process is currently running (would
     mean two independent engine loops, each pacing itself correctly on its
     own, but together landing calls on Mistral much closer together than
     either one believes it is).
  2. The full, real response (status + ALL headers + body) from GET
     /v1/models with your actual key — this tells us your account's real
     model list, and Mistral's rate-limit headers (if any) tell us what
     tier Mistral itself thinks this key is on.
  3. Which of the app's 4 hardcoded candidate model names actually exist in
     that list (checks the exact typo-prone strings from ai_lighting_engine.py).
  4. A single lightweight POST to /v1/chat/completions against whichever
     model gets resolved, printing the FULL status/headers/body — including
     a 429 if one happens, with every header Mistral sends back (Retry-After,
     and any x-ratelimit-* headers), not just the status code.
  5. A deliberate back-to-back burst (2 calls ~0.2s apart) to see what a
     genuine, unambiguous 429 looks like from this specific account, so we
     can compare it against what the app logs during normal use.

Paste the FULL output back — especially the headers blocks — that's the
part that tells us whether this is a pacing issue or an account/tier issue.
"""
import json
import os
import subprocess
import sys
import time

try:
    import requests
except ImportError:
    print("Installing 'requests'...")
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    import requests

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

VISION_MODEL_CANDIDATES = [
    "ministral-3-3b-2512",
    "ministral-3-8b-2512",
    "pixtral-12b-2409",
    "mistral-small-2603",
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def dump_response(resp):
    print(f"HTTP {resp.status_code} {resp.reason}")
    print("-- headers --")
    for k, v in resp.headers.items():
        print(f"  {k}: {v}")
    print("-- body --")
    try:
        print(json.dumps(resp.json(), indent=2)[:4000])
    except ValueError:
        print(resp.text[:4000])


def main():
    section("1. Checking for multiple webapp.py processes")
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["wmic", "process", "where", "name like '%python%'", "get", "ProcessId,CommandLine"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            if not out.strip():
                # wmic may be missing/deprecated on newer Windows builds — fall back to tasklist
                out = subprocess.run(["tasklist", "/v", "/fi", "imagename eq python*"],
                                      capture_output=True, text=True, timeout=15).stdout
        else:
            out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=15).stdout
        matches = [line for line in out.splitlines() if "webapp.py" in line]
        print(f"Found {len(matches)} process(es) referencing webapp.py:")
        for m in matches:
            print("  " + m.strip())
        if len(matches) > 1:
            print("\n  *** MORE THAN ONE webapp.py PROCESS IS RUNNING ***")
            print("  This alone could explain persistent 429s: each process runs its own")
            print("  engine loop with its own independent pacing/throttle state, so two")
            print("  processes together can land requests on Mistral much closer together")
            print("  than either one intends, even though each individually paces itself")
            print("  correctly by its own bookkeeping.")
    except Exception as e:
        print(f"  (couldn't enumerate processes: {e})")

    if not os.path.exists(CONFIG_PATH):
        print(f"\nCouldn't find config.json next to this script at {CONFIG_PATH}")
        print("Copy this script into the project folder (next to webapp.py) and run it from there.")
        return

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    api_key = cfg.get("mistral_api_key", "")
    if not api_key:
        print("\nNo mistral_api_key found in config.json — nothing to test.")
        return
    print(f"\nUsing mistral_api_key from config.json (starts with {api_key[:6]}...)")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    section("2. GET /v1/models — full response")
    try:
        resp = requests.get("https://api.mistral.ai/v1/models", headers=headers, timeout=15)
        dump_response(resp)
    except requests.RequestException as e:
        print(f"Request failed entirely: {e}")
        return

    models = []
    try:
        models = resp.json().get("data", [])
    except ValueError:
        pass
    by_id = {m["id"]: m for m in models if m.get("id")}

    section("3. Candidate model name check")
    print(f"Account's live model list has {len(by_id)} model(s):")
    for mid, m in by_id.items():
        caps = m.get("capabilities", {})
        print(f"  {mid}  (vision={caps.get('vision')}, chat={caps.get('completion_chat')})")
    print("\nApp's hardcoded candidates (ai_lighting_engine.py):")
    resolved = None
    for c in VISION_MODEL_CANDIDATES:
        hit = c in by_id
        print(f"  {c}: {'FOUND in account model list' if hit else 'NOT FOUND'}")
        if hit and resolved is None:
            resolved = c
    if resolved is None:
        # mimic the app's own fallback chain
        for m in models:
            caps = m.get("capabilities", {})
            if caps.get("completion_chat") and caps.get("vision"):
                resolved = m["id"]
                break
        if resolved is None:
            for m in models:
                if m.get("capabilities", {}).get("completion_chat"):
                    resolved = m["id"]
                    break
    if resolved is None:
        resolved = VISION_MODEL_CANDIDATES[0]
    print(f"\n=> The app would currently resolve to: {resolved}")
    if resolved not in by_id:
        print("   *** This model is NOT in your account's own model list at all — the app")
        print("   would be calling a name Mistral doesn't recognize for this key, which")
        print("   typically 400s/404s rather than 429s, but worth confirming below. ***")

    section("4. Single chat/completions call against the resolved model — full response")
    payload = {
        "model": resolved,
        "messages": [{"role": "user", "content": "Reply with just the word OK."}],
        "temperature": 0.7,
    }
    try:
        resp2 = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=20)
        dump_response(resp2)
    except requests.RequestException as e:
        print(f"Request failed entirely: {e}")
        resp2 = None

    section("5. Deliberate back-to-back burst (2 calls, ~0.2s apart) to force a clean 429")
    for i in range(2):
        try:
            r = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=20)
            print(f"\n--- burst call {i+1} ---")
            dump_response(r)
        except requests.RequestException as e:
            print(f"Request failed entirely: {e}")
        time.sleep(0.2)

    section("Done")
    print("Paste this entire output back. In particular:")
    print(" - Did step 1 find more than one webapp.py process?")
    print(" - In steps 4/5, what HTTP status came back, and what did EVERY header say")
    print("   (especially anything starting with 'x-ratelimit' or 'retry-after')?")
    print(" - Did the 403/29/other unexpected status appear anywhere, which would point")
    print("   to a plan/activation problem rather than a pure rate-limit problem?")


if __name__ == "__main__":
    main()
