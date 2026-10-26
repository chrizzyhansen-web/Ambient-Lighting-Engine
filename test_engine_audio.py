"""
Integration test for the audio path THROUGH engine.py's real loop.

This deliberately does not reimplement the policy and check the
reimplementation — that would only prove the copy agrees with itself. It
stubs the hardware and the AI, injects a scripted listener, runs the actual
_loop(), and records what actually reached effects.render().

The clock is faked so a three-minute track runs in under a second and the
result is deterministic.
"""
import sys
import threading
import types

import engine
import effects

FAILURES = []
FPS = 25.0
DT = 1.0 / FPS


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


def section(t):
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


# ---- fakes -----------------------------------------------------------------

class LoopStalled(Exception):
    """Raised by the fake clock's watchdog. A real clock turns a loop that
    never reaches the render path into a slow test; a fake one turns it into
    a hang, so the watchdog is not optional."""


class FakeClock:
    """engine.py reads wall clock in several places and sleeps between
    frames. Replacing both makes the loop advance exactly one frame per
    iteration with no real waiting."""

    def __init__(self, max_sleeps=20_000):
        self.now = 1_700_000_000.0
        self.sleeps = 0
        self.max_sleeps = max_sleeps

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.sleeps += 1
        if self.sleeps > self.max_sleeps:
            raise LoopStalled(
                f"loop ran {self.sleeps} iterations without finishing — it is "
                f"probably not reaching the render path at all"
            )


class FakeGovee:
    segment_count = 15
    lan_available = True
    capability_warning = None
    light_scenes: dict = {}
    diy_scenes: dict = {}
    max_brightness_pct = 20

    def __init__(self, *a, **k):
        self.frames = []

    def cloud_slots_available(self):
        return 10

    def apply_raw_lan_gradient(self, flat):
        self.frames.append(list(flat))

    def set_color(self, *a, **k):
        pass

    def set_brightness(self, *a, **k):
        pass

    def release_lan_stream(self):
        pass


class FakeKeyframe:
    """Stands in for ai_lighting_engine.HueKeyframe."""

    def __init__(self, brightness_pct):
        self.hue_deg = 210.0
        self.hue_spread_deg = 60.0
        self.saturation_pct = 80
        self.brightness_pct = brightness_pct
        self.hardware_scene_name = None
        self.at_pct = 0.0


class FakeTrackSource:
    """Always playing, unless a script says otherwise.

    `playing_script` takes elapsed seconds and returns whether Spotify
    reports itself playing, so a test can put the loop through real pauses.
    """

    def __init__(self, clock=None, playing_script=None):
        self.clock = clock
        self.playing_script = playing_script
        self.t0 = clock.now if clock else 0.0

    def get_now_playing(self):
        playing = True
        if self.playing_script is not None and self.clock is not None:
            playing = self.playing_script(self.clock.now - self.t0)
        return {
            "track_id": "t1", "title": "T", "artist": "A",
            "album_art_url": None, "is_playing": playing,
            "progress_pct": 10.0, "duration_ms": 200_000,
        }


class ScriptedListener:
    """A listener whose output is a function of frame number, so a test can
    state exactly what the audio was doing and check what the strip did."""

    def __init__(self, script):
        self.script = script
        self.frame = 0
        self.device_name = "Scripted"
        # Must mirror LoopbackListener's real interface — engine.py reads
        # this to report whether the strip is reacting to one device or to
        # the whole system mix.
        self.device_name_contains = None
        self.error = None
        self._onsets = []

    @property
    def available(self):
        return self.script(self.frame)[0]

    def get_state(self):
        live, energy, intensity, surge = self.script(self.frame)
        return engine.audio_listener.AudioState(
            energy=energy, intensity=intensity, onset=0.0, live=live,
            surge_db=surge, loudness_db=-20.0, floor_db=-44.0, ceiling_db=-20.0,
        )

    def pop_onsets(self):
        out = self._onsets
        self._onsets = []
        return out

    def stop(self):
        pass


def run_loop(script, frames, plan_energy=40.0, playing_script=None):
    """Run engine._loop with everything stubbed, for `frames` frames.
    Returns (energy_pct per render call, events list per render call)."""
    clock = FakeClock()
    govee = FakeGovee()
    listener = ScriptedListener(script)

    rendered = []

    def fake_render(self, t, **kw):
        rendered.append((kw["energy_pct"], list(kw.get("events") or [])))
        listener.frame = len(rendered)
        if len(rendered) >= frames:
            engine._stop_event.set()
        return [0x112233] * 15

    saved = {
        "time": engine.time,
        "GoveeClient": engine.GoveeClient,
        "render": effects.EffectEngine.render,
        "sample_keyframes": engine.sample_keyframes,
        "SpicetifyBridgeSource": engine.SpicetifyBridgeSource,
        "Groq": engine.GroqLightingEngine,
    }
    fake_time = types.SimpleNamespace(time=clock.time, sleep=clock.sleep)
    engine.time = fake_time
    engine.GoveeClient = lambda *a, **k: govee
    engine.SpicetifyBridgeSource = lambda: FakeTrackSource(clock, playing_script)
    engine.sample_keyframes = lambda kfs, pct: FakeKeyframe(plan_energy)
    effects.EffectEngine.render = fake_render

    class FakeTimeline:
        steps: list = []

    class FakeDecision:
        timeline = FakeTimeline()
        keyframes = [FakeKeyframe(plan_energy)]
        anchor_hues = [210, 260, 300]

        def model_dump(self):
            return {"fake": True}

    class FakeLighting:
        def __init__(self, *a, **k):
            pass

        def decide(self, **kw):
            return FakeDecision()

        def decision_from_cached(self, cached):
            return FakeDecision()

        def cacheable_payload(self, decision):
            return {}

    engine.GroqLightingEngine = FakeLighting

    # A live plan, so the LAN render path is taken from the first frame.
    engine.local_keyframes = [FakeKeyframe(plan_energy)]
    engine.local_timeline_buffer = []

    # Inject the scripted listener in place of a real capture device. When
    # audio_listener is absent entirely there is nothing to inject, which is
    # the case section 5 exercises.
    real_loopback = None
    if engine.audio_listener is not None:
        real_loopback = engine.audio_listener.LoopbackListener
        engine.audio_listener.LoopbackListener = lambda *a, **k: listener
        listener.start = lambda: True

    engine._stop_event.clear()
    try:
        engine._loop({
            "track_source": "spicetify", "ai_provider": "groq",
            "groq_api_key": "x", "govee_api_key": "x", "govee_device_id": "x",
            "govee_device_model": "H619B", "enable_lan_control": True,
            "remember_songs": False, "glitch_effect": True,
            "enable_audio_listener": True,
        })
    finally:
        engine.time = saved["time"]
        engine.GoveeClient = saved["GoveeClient"]
        effects.EffectEngine.render = saved["render"]
        engine.sample_keyframes = saved["sample_keyframes"]
        engine.SpicetifyBridgeSource = saved["SpicetifyBridgeSource"]
        engine.GroqLightingEngine = saved["Groq"]
        if real_loopback is not None:
            engine.audio_listener.LoopbackListener = real_loopback
        engine._stop_event.set()
    return rendered


def run_loop_until(script, seconds, playing_script, plan_energy=40.0):
    """Like run_loop but stops on simulated elapsed time rather than on a
    frame count, because a paused loop renders no frames at all."""
    holder = {}

    class Stopper:
        def __init__(self, clock):
            self.clock = clock
            self.t0 = clock.now

        def __call__(self):
            if self.clock.now - self.t0 >= seconds:
                engine._stop_event.set()

    real_sleep = FakeClock.sleep

    def patched_sleep(self, s):
        real_sleep(self, s)
        stopper = holder.get("stopper")
        if stopper:
            stopper()

    FakeClock.sleep = patched_sleep
    orig_init = FakeClock.__init__

    def patched_init(self, *a, **k):
        orig_init(self, *a, **k)
        holder["stopper"] = Stopper(self)

    FakeClock.__init__ = patched_init
    try:
        return run_loop(script, frames=10**9, plan_energy=plan_energy,
                        playing_script=playing_script)
    finally:
        FakeClock.sleep = real_sleep
        FakeClock.__init__ = orig_init


# ---- 1. measured energy replaces the plan's guess ---------------------------

section("1. Does measured energy actually reach effects.render?")
rendered = run_loop(lambda f: (True, 80.0, 50.0, 0.0), frames=400, plan_energy=20.0)
energies = [e for e, _ in rendered]
print(f"  plan says 20, audio says 80. frames rendered: {len(energies)}")
print(f"  first frame {energies[0]:.1f}  frame 25 {energies[25]:.1f}  "
      f"frame 100 {energies[100]:.1f}  last {energies[-1]:.1f}")
check("starts from the plan's value", abs(energies[0] - 20.0) < 3.0,
      f"{energies[0]:.1f}")
check("ends at the measured value", abs(energies[-1] - 80.0) < 3.0,
      f"{energies[-1]:.1f}")
# The handover must be a ramp. A single-frame step is what would jolt the
# wave speed, palette width and resting lightness all at once.
steps = [abs(energies[i + 1] - energies[i]) for i in range(len(energies) - 1)]
print(f"  largest single-frame change during handover: {max(steps):.2f} points")
check("handover is a ramp, not a step", max(steps) < 3.0, f"{max(steps):.2f}")
half = next(i for i, e in enumerate(energies) if e >= 50.0)
print(f"  halfway to the measured value after {half * DT:.2f}s")
check("handover completes in about a second and a half",
      0.5 < half * DT < 2.5, f"{half * DT:.2f}s")


# ---- 2. falls back when the listener dies -----------------------------------

section("2. Does it fall back cleanly when capture stops?")


def dies_at_200(f):
    return (f < 200, 80.0, 50.0, 0.0)


rendered = run_loop(dies_at_200, frames=500, plan_energy=20.0)
energies = [e for e, _ in rendered]
print(f"  audio present for 200 frames (8.0s), then gone")
print(f"  frame 190 {energies[190]:.1f}  frame 260 {energies[260]:.1f}  "
      f"last {energies[-1]:.1f}")
check("returns to the plan's value", abs(energies[-1] - 20.0) < 4.0,
      f"{energies[-1]:.1f}")
steps = [abs(energies[i + 1] - energies[i]) for i in range(len(energies) - 1)]
check("the fallback is also a ramp", max(steps) < 3.0, f"{max(steps):.2f}")


# ---- 3. a surge fires exactly one sweep -------------------------------------

section("3. Does a surge fire one sweep, and only one?")


def sustained_surge(f):
    # A loud section arrives at frame 100 and stays loud. surge_db stays
    # high for a while, which is exactly the case that a naive trigger
    # turns into repeated sweeps.
    surge = 4.0 if 100 <= f < 300 else 0.0
    return (True, 70.0, 60.0, surge)


rendered = run_loop(sustained_surge, frames=500, plan_energy=40.0)
sweeps = []
seen = set()
for i, (_, events) in enumerate(rendered):
    for kind, started, strength in events:
        if kind == "sweep" and started not in seen:
            seen.add(started)
            sweeps.append((i, strength))
print(f"  surge held above threshold from frame 100 to 300 (8 seconds)")
print(f"  sweeps fired: {[(f'frame {i}', round(s, 2)) for i, s in sweeps]}")
check("exactly one sweep for one sustained surge", len(sweeps) == 1,
      f"{len(sweeps)}")
if sweeps:
    check("it fires as the surge arrives", 95 <= sweeps[0][0] <= 115,
          f"frame {sweeps[0][0]}")


# ---- 4. repeated surges respect the refractory ------------------------------

section("4. Can the strip be made to strobe?")


def rapid_surges(f):
    # Surge on and off every 25 frames (1s) for the whole run — far faster
    # than any real arrangement, to check the hard limit rather than the
    # detector's taste.
    surge = 5.0 if (f // 25) % 2 == 0 else 0.0
    return (True, 70.0, 60.0, surge)


rendered = run_loop(rapid_surges, frames=750, plan_energy=40.0)
starts = sorted({st for _, evs in rendered for k, st, _ in evs if k == "sweep"})
gaps = [b - a for a, b in zip(starts, starts[1:])]
print(f"  30 seconds of surge toggling once a second")
print(f"  sweeps: {len(starts)}   gaps: {[round(g, 2) for g in gaps]}")
check("no more than one sweep per refractory period",
      not gaps or min(gaps) >= engine._AUDIO_SURGE_REFRACTORY - 0.05,
      f"smallest gap {min(gaps):.2f}s" if gaps else "one sweep")
rate = len(starts) / (750 * DT / 60.0)
print(f"  worst-case sweep rate: {rate:.1f}/min")
check("worst-case rate stays under 11/min", rate < 11.0, f"{rate:.1f}/min")


# ---- 5. audio off entirely = the old behaviour ------------------------------

section("5. With no listener, is the old path untouched?")
saved_al = engine.audio_listener
engine.audio_listener = None
try:
    rendered = run_loop(lambda f: (False, 0.0, 0.0, 0.0), frames=200,
                        plan_energy=55.0)
finally:
    engine.audio_listener = saved_al
energies = [e for e, _ in rendered]
print(f"  audio_listener import unavailable; plan energy 55")
print(f"  rendered energy: min {min(energies):.1f} max {max(energies):.1f}")
check("energy is exactly the plan's, every frame",
      max(abs(e - 55.0) for e in energies) < 1e-9,
      f"max deviation {max(abs(e - 55.0) for e in energies):.2e}")
check("status reports the plan as the source",
      engine.status["energy_source"] == "plan",
      engine.status["energy_source"])


# ---- 6. isolation monitoring through the real loop -------------------------

section("6. Does the isolation monitor actually get fed during pauses?")
# This is the wiring that is easy to get wrong: the paused branch of _loop
# `continue`s before the render path, so a monitor fed only from the render
# path would never observe a single paused frame — and paused frames are the
# ONLY ones that prove anything about isolation.
engine.status["audio_isolation"] = None

# Spotify plays for 10s, pauses for 90s, plays again.
def paused_in_the_middle(elapsed):
    return not (10.0 <= elapsed < 100.0)

# ...and the captured audio keeps arriving throughout, including during the
# pause. That can only mean something other than Spotify is being captured.
run_loop_until(lambda f: (True, 60.0, 55.0, 0.0), seconds=130.0,
               playing_script=paused_in_the_middle)
v = engine.status["audio_isolation"]
print(f"  Spotify paused 10s-100s; audio present the whole time")
print(f"  verdict: {v}")
print(f"  {engine.status['audio_isolation_detail']}")
check("contamination is detected through the real loop", v == "contaminated", str(v))

# And the clean case: audio stops when Spotify does.
engine.status["audio_isolation"] = None
frame_clock = {"t": 0}

def only_when_playing(f):
    # The listener reports live only while Spotify is playing. Frame counter
    # is not advanced during pauses (no renders happen), so key off the
    # engine's own view instead.
    return (engine.status.get("_test_playing", True), 60.0, 55.0, 0.0)

class PlayAwareListener(ScriptedListener):
    pass

def clean_run():
    # Rebuild the script so "audio live" follows the play state exactly.
    state = {"playing": True}

    def playing_script(elapsed):
        state["playing"] = not (10.0 <= elapsed < 100.0)
        return state["playing"]

    return run_loop_until(lambda f: (state["playing"], 60.0, 55.0, 0.0),
                          seconds=130.0, playing_script=playing_script)

engine.status["audio_isolation"] = None
clean_run()
v = engine.status["audio_isolation"]
print(f"  same timeline, but audio stops when Spotify does")
print(f"  verdict: {v}")
print(f"  {engine.status['audio_isolation_detail']}")
check("a properly isolated capture is reported as isolated",
      v == "isolated", str(v))


print()
print("=" * 74)
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
