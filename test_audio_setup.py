"""
Measurements for audio_setup.py's isolation monitor.

The monitor makes a claim that a user will trust — "nothing but Spotify has
reached the strip" — so the thing that matters most is that it CANNOT say
that when it isn't true, and that it doesn't cry wolf on ordinary pauses.
Both are tested here against scripted timelines rather than asserted.
"""
import sys

from audio_setup import (
    IsolationMonitor, PAUSE_GRACE_S, MIN_EVIDENCE_S, looks_like_cable,
    setup_steps,
)

FAILURES = []
DT = 1.0 / 25.0   # the engine's frame rate


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


def section(t):
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


def play(timeline, seconds, spotify, audio):
    """Append `seconds` of (spotify_playing, audio_live) at 25fps."""
    for _ in range(int(seconds / DT)):
        timeline.append((spotify, audio))


def run(timeline, t0=1_000_000.0):
    m = IsolationMonitor()
    t = t0
    for spotify, audio in timeline:
        m.observe(t, spotify, audio)
        t += DT
    return m, m.verdict(t)


# ---- 1. an ordinary listening session --------------------------------------

section("1. Normal use: music, pauses, silence during pauses")
tl = []
for _ in range(6):
    play(tl, 120.0, True, True)    # two minutes of music
    play(tl, 20.0, False, False)   # paused, silent
m, v = run(tl)
print(f"  12 min of music, 6 pauses of 20s each, nothing else playing")
print(f"  verdict: {v.state}")
print(f"  {v.detail}")
print(f"  paused evidence: {v.paused_seconds:.0f}s   contaminated: "
      f"{v.contaminated_seconds:.1f}s")
check("reports isolated", v.state == "isolated", v.state)
check("no contamination episodes", v.events == 0, str(v.events))


# ---- 2. the pause tail must not be flagged ---------------------------------

section("2. Spotify's own fade-out after a pause")
tl = []
for _ in range(6):
    play(tl, 60.0, True, True)
    # Spotify reports paused, but audio keeps arriving for a moment: the
    # 0.25s poll lag, the player's fade, and the analyser's 1.2s gate hold.
    play(tl, 2.0, False, True)
    play(tl, 25.0, False, False)
m, v = run(tl)
print(f"  every pause has a 2.0s audible tail (grace period is "
      f"{PAUSE_GRACE_S}s)")
print(f"  verdict: {v.state}   episodes flagged: {v.events}")
check("the tail is not mistaken for contamination", v.state == "isolated",
      f"{v.state}, {v.events} episodes")


# ---- 3. real contamination is caught ---------------------------------------

section("3. A YouTube tab playing while Spotify is paused")
tl = []
play(tl, 120.0, True, True)
play(tl, 60.0, False, False)    # clean pause, builds evidence
play(tl, 90.0, False, True)     # something else is playing
play(tl, 30.0, False, False)
m, v = run(tl)
print(f"  90s of audio while Spotify reported paused")
print(f"  verdict: {v.state}")
print(f"  {v.detail}")
print(f"  contaminated: {v.contaminated_seconds:.0f}s in {v.events} episode(s)")
check("reports contaminated", v.state == "contaminated", v.state)
check("attributes roughly the right duration",
      80.0 < v.contaminated_seconds < 95.0, f"{v.contaminated_seconds:.0f}s")


# ---- 4. a short notification is still caught --------------------------------

section("4. A single Discord notification during a pause")
tl = []
play(tl, 120.0, True, True)
play(tl, 40.0, False, False)
play(tl, 1.2, False, True)      # one ping
play(tl, 60.0, False, False)
m, v = run(tl)
print(f"  one 1.2s notification during an otherwise silent pause")
print(f"  verdict: {v.state}   episodes: {v.events}   "
      f"contaminated: {v.contaminated_seconds:.1f}s")
check("a single ping is enough to fail the claim",
      v.state == "contaminated", v.state)


# ---- 5. it must never claim isolation without evidence ---------------------

section("5. Can it be made to claim isolation it hasn't earned?")
# Almost no paused time at all — the case where a user plays music
# continuously and never gives the monitor anything to observe.
tl = []
play(tl, 600.0, True, True)
play(tl, 3.0, False, False)
m, v = run(tl)
print(f"  10 min of continuous music, one 3s pause")
print(f"  paused evidence: {v.paused_seconds:.1f}s "
      f"(needs {MIN_EVIDENCE_S:.0f}s)")
print(f"  verdict: {v.state}")
check("says unknown rather than isolated", v.state == "unknown", v.state)

# And the degenerate case: nothing observed at all.
m2 = IsolationMonitor()
v2 = m2.verdict(1_000_000.0)
print(f"  with no observations at all: {v2.state}")
check("an unused monitor does not claim isolation", v2.state == "unknown",
      v2.state)

# Audio that only ever appears while Spotify is playing proves nothing about
# isolation, and must not be counted as evidence FOR it.
tl = []
play(tl, 300.0, True, True)
m3, v3 = run(tl)
print(f"  5 min of music with no pause at all: {v3.state}, "
      f"{v3.paused_seconds:.0f}s evidence")
check("playing time is not counted as evidence", v3.paused_seconds == 0.0,
      f"{v3.paused_seconds:.1f}s")


# ---- 6. a setup that rots later ---------------------------------------------

section("6. Routing silently resets after a reboot")
tl = []
# A long clean history first...
for _ in range(10):
    play(tl, 120.0, True, True)
    play(tl, 30.0, False, False)
m, v = run(tl)
print(f"  after 25 min of clean listening: {v.state}")
check("clean history reads as isolated", v.state == "isolated", v.state)
# ...then the per-app routing resets and the whole mix comes back.
t = 1_000_000.0 + len(tl) * DT
for spotify, audio in [(False, True)] * int(30.0 / DT):
    m.observe(t, spotify, audio)
    t += DT
v = m.verdict(t)
print(f"  then 30s of contamination: {v.state}")
print(f"  {v.detail}")
check("a long clean history does not outvote fresh contamination",
      v.state == "contaminated", v.state)


# ---- 7. cable name matching -------------------------------------------------

section("7. Recognising a virtual cable by device name")
cases = [
    ("CABLE Input (VB-Audio Virtual Cable)", True),
    ("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", True),
    ("Line 1 (Virtual Audio Cable)", True),
    ("Speakers (Realtek(R) Audio)", False),
    ("Headphones (2- USB Audio Device)", False),
    ("NVIDIA High Definition Audio", False),
    ("", False),
]
for name, expected in cases:
    got = looks_like_cable(name)
    print(f"  {'cable ' if got else 'normal'}  {name or '(empty)'}")
    check(f"classifies {name or '(empty)'!r} correctly", got == expected)


# ---- 8. the checklist tells the truth about the current state --------------

section("8. Checklist state when no cable is installed")
steps = setup_steps({"audio_capture_device": ""}, "Speakers (Realtek)", None)
for s in steps:
    print(f"  [{s['state']:7s}] {s['title']}")
    print(f"            {s['detail'][:88]}")
by_key = {s["key"]: s for s in steps}
if "cable" in by_key:
    check("the verified step is not shown as merely incomplete",
          by_key.get("verified", {}).get("state") in ("blocked", "todo"),
          by_key.get("verified", {}).get("state"))
    check("it says plainly that the whole mix is being captured",
          "whole system mix" in by_key.get("verified", {}).get("detail", ""),
          by_key.get("verified", {}).get("detail", "")[:40])
else:
    print("  (PyAudioWPatch absent here, so the checklist stops at deps —")
    print("   which is itself the correct behaviour on this machine)")
    check("stops at the dependency step when libraries are missing",
          by_key["deps"]["state"] == "todo", by_key["deps"]["detail"][:40])




# ---- 9. a cable that is already installed ----------------------------------

section("9. What if a virtual cable is ALREADY installed?")
from audio_setup import (
    DeviceObservation as DO, classify_device, choose_capture_device,
)

cases = [
    ("already carrying Spotify, nothing else",
     DO("CABLE Input", paused_seconds=8.0, paused_audible_seconds=0.0,
        playing_seconds=8.0, playing_audible_seconds=7.6),
     "carrying-spotify"),
    ("in use by Discord/OBS — audible while Spotify is paused",
     DO("CABLE Input", paused_seconds=8.0, paused_audible_seconds=6.0,
        playing_seconds=8.0, playing_audible_seconds=8.0),
     "busy"),
    ("installed but nothing routed to it",
     DO("CABLE Input", paused_seconds=8.0, paused_audible_seconds=0.0,
        playing_seconds=8.0, playing_audible_seconds=0.0),
     "idle"),
    ("one stray buffer while paused — still idle, not busy",
     DO("CABLE Input", paused_seconds=8.0, paused_audible_seconds=0.3,
        playing_seconds=8.0, playing_audible_seconds=0.1),
     "idle"),
    ("probe too short to say",
     DO("CABLE Input", paused_seconds=1.0, paused_audible_seconds=0.0,
        playing_seconds=1.0, playing_audible_seconds=1.0),
     "unknown"),
]
for label, obs, expected in cases:
    role, detail = classify_device(obs)
    print(f"  {label:52s} -> {role}")
    check(f"classifies: {label}", role == expected, f"got {role}")

section("10. Picking a device when several cables exist")
scenarios = [
    ("Spotify already routed — setup is already done",
     [DO("CABLE Input", 8, 6, 8, 8),            # busy
      DO("CABLE-B Input", 8, 0, 8, 7.5)],       # carrying spotify
     "already-set-up", "CABLE-B Input"),
    ("one busy, one free — use the free one",
     [DO("VoiceMeeter Input", 8, 7, 8, 8),
      DO("CABLE Input", 8, 0, 8, 0)],
     "use-idle", "CABLE Input"),
    ("every cable is in use — do NOT hijack one",
     [DO("CABLE Input", 8, 5, 8, 8),
      DO("VoiceMeeter Input", 8, 6, 8, 8)],
     "all-busy", None),
    ("no cable installed at all",
     [], "none", None),
]
for label, obs, expected_outcome, expected_device in scenarios:
    device, outcome, detail = choose_capture_device(obs)
    print(f"  {label}")
    print(f"      -> {outcome}" + (f" ({device})" if device else ""))
    print(f"      {detail[:92]}")
    check(f"outcome: {label}", outcome == expected_outcome, outcome)
    check(f"device:  {label}", device == expected_device, str(device))


# ---- 11. selected device with nothing routed to it -------------------------

section("11. Cable selected, but Spotify never routed to it")
# The failure this catches: no audio arrives, so no contamination is ever
# observed, so a naive monitor certifies the setup as isolated while the
# strip reacts to absolutely nothing.
tl = []
for _ in range(8):
    play(tl, 60.0, True, False)    # Spotify "playing" but capture is silent
    play(tl, 25.0, False, False)
m, v = run(tl)
print(f"  8 min of playback with no audio reaching the capture device")
print(f"  verdict: {v.state}")
print(f"  {v.detail}")
print(f"  playing evidence {v.playing_seconds:.0f}s, of which audible "
      f"{v.audible_playing_seconds:.0f}s")
check("reports silent, NOT isolated", v.state == "silent", v.state)

# And the genuine article still passes.
tl = []
for _ in range(8):
    play(tl, 60.0, True, True)
    play(tl, 25.0, False, False)
m, v = run(tl)
print(f"  same timeline with audio actually arriving: {v.state}")
check("a real isolated capture is still reported isolated",
      v.state == "isolated", v.state)


print()
print("=" * 74)
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
