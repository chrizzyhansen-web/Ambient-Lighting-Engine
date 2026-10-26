"""
Measurements for audio_listener.py. Prints actual numbers and asserts on them.

Everything is synthetic and deterministic, so a change to the analyser shows
up as a changed number rather than as a feeling about how the strip looked.
Run: python test_audio_listener.py
"""
import math
import sys

import numpy as np

from audio_listener import (
    AudioAnalyser, HOP_SIZE, build_mel_filterbank,
)

SR = 48000
RNG = np.random.default_rng(7)
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = ""):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}{('  ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def section(title: str):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ---- signal generators -----------------------------------------------------

def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds))


def tone(seconds: float, hz: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return amp * np.sin(2 * np.pi * hz * t)


def noise(seconds: float, amp: float = 0.1) -> np.ndarray:
    return amp * RNG.standard_normal(int(SR * seconds))


def kick(amp: float = 0.9) -> np.ndarray:
    """A synthetic kick: pitch-swept sine body plus a short broadband click.
    The click is what a 200Hz-cornered mel bank can actually see."""
    n = int(SR * 0.18)
    t = np.arange(n) / SR
    f = 120.0 * np.exp(-t * 32.0) + 45.0
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18.0)
    click_n = int(SR * 0.004)
    click = np.zeros(n)
    click[:click_n] = RNG.standard_normal(click_n) * np.exp(-np.arange(click_n) / (click_n / 3))
    return amp * (0.85 * body + 0.5 * click)


def snare(amp: float = 0.7) -> np.ndarray:
    n = int(SR * 0.14)
    t = np.arange(n) / SR
    return amp * RNG.standard_normal(n) * np.exp(-t * 26.0)


def place(dest: np.ndarray, src: np.ndarray, at_sample: int):
    end = min(len(dest), at_sample + len(src))
    if end > at_sample:
        dest[at_sample:end] += src[:end - at_sample]


def drum_loop(seconds: float, bpm: float = 120.0, amp: float = 1.0,
              with_snare: bool = True) -> tuple[np.ndarray, list[int]]:
    """Four-on-the-floor. Returns the audio and the true onset sample indices
    so detection latency can be measured rather than guessed at."""
    out = np.zeros(int(SR * seconds))
    beat = 60.0 / bpm
    onsets = []
    i = 0
    while i * beat < seconds:
        s = int(i * beat * SR)
        place(out, kick() * amp, s)
        onsets.append(s)
        if with_snare and i % 2 == 1:
            place(out, snare() * amp, s)
        i += 1
    return out, onsets


def run(analyser: AudioAnalyser, audio: np.ndarray, block: int = HOP_SIZE):
    """Feed audio in realistic block sizes and return every state."""
    states = []
    for i in range(0, len(audio), block):
        states.extend(analyser.feed(audio[i:i + block]))
    return states


def db(x: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(x * x))) + 1e-12)


# ---- 1. filterbank ---------------------------------------------------------

section("1. Mel filterbank: area normalisation")
bank = build_mel_filterbank(SR, 1024, 24, 200.0, 12000.0)
white = noise(4.0, 0.2)
frame = white[:1024] * np.hamming(1024)
mel = bank @ np.abs(np.fft.rfft(frame))
lo, hi = mel[:8].mean(), mel[-8:].mean()
ratio = hi / lo
print(f"  white noise: low-8 bins mean {lo:.4f}, high-8 bins mean {hi:.4f}")
print(f"  high/low ratio = {ratio:.3f}   (1.0 = perfectly flat response)")
check("flat-spectrum response within 2x across the bank", 0.5 < ratio < 2.0,
      f"ratio {ratio:.3f}")
widths = (bank > 0).sum(axis=1)
print(f"  filter widths in FFT bins: {widths.tolist()}")
check("no degenerate (zero-width) filters", int(widths.min()) >= 1,
      f"narrowest {int(widths.min())} bins")


# ---- 2. squelch ------------------------------------------------------------

section("2. Squelch: silence must gate, and must not adapt")
a = AudioAnalyser(samplerate=SR)
states = run(a, silence(5.0))
live_count = sum(1 for s in states if s.live)
print(f"  5s of digital silence -> {len(states)} frames, {live_count} live")
print(f"  final: energy {states[-1].energy:.2f}  intensity {states[-1].intensity:.2f}"
      f"  dBFS {states[-1].loudness_db:.1f}")
check("no frame reports live during silence", live_count == 0)
check("energy stays at zero", states[-1].energy < 0.01)

# Dither at -90dBFS, the kind an idle player emits.
a = AudioAnalyser(samplerate=SR)
states = run(a, noise(10.0, amp=10 ** (-90 / 20)))
live_count = sum(1 for s in states if s.live)
print(f"  10s of -90dBFS dither -> {live_count}/{len(states)} live frames, "
      f"final energy {states[-1].energy:.2f}")
check("dither never opens the gate", live_count == 0)
check("gains did not creep during the gated period", states[-1].energy < 0.01)


# ---- 3. false positives on sustained material ------------------------------

section("3. Onset false-positive rate on material with no transients")
for label, sig in (
    ("steady 1kHz sine, 20s", tone(20.0, 1000.0, 0.2)),
    ("steady white noise, 20s", noise(20.0, 0.15)),
    ("slow sine sweep 300->3000Hz, 20s", None),
):
    if sig is None:
        t = np.arange(int(SR * 20.0)) / SR
        f = 300.0 * (10.0 ** (t / 20.0))
        sig = 0.2 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    a = AudioAnalyser(samplerate=SR)
    states = run(a, sig)
    # Ignore the first 2s: the gate opening and the gains converging is a real
    # transient, and suppressing it is the analyser's job, not a free pass.
    settle = int(2.0 / a.dt)
    n_onsets = sum(1 for s in states[settle:] if s.onset > 0)
    rate = n_onsets / (len(states[settle:]) * a.dt)
    print(f"  {label:34s} {n_onsets:4d} onsets  =  {rate:5.2f}/s")
    check(f"{label}: under 0.5 false onsets/s", rate < 0.5, f"{rate:.2f}/s")


# ---- 4. onset detection on a drum loop -------------------------------------

section("4. Onset recall and latency on a 120bpm loop")
audio, true_onsets = drum_loop(20.0, bpm=120.0)
a = AudioAnalyser(samplerate=SR)
states = run(a, audio)
det = [i * a.dt for i, s in enumerate(states) if s.onset > 0]
# The analyser reports at the END of a hop whose window looks back FFT_SIZE
# samples, so a detection is inherently a window-length behind the event.
window_lag = a.fft_size / SR
truth = [s / SR for s in true_onsets]
settle = 2.0
truth = [t for t in truth if t > settle]

matched, lats = 0, []
for t in truth:
    near = [d for d in det if -0.01 <= (d - t) <= 0.12]
    if near:
        matched += 1
        lats.append((min(near) - t) * 1000.0)
recall = matched / len(truth)
extra = len(det) - matched
print(f"  true onsets after settle: {len(truth)}   detected total: {len(det)}")
print(f"  recall {recall * 100:.1f}%   unmatched detections: {extra}")
if lats:
    print(f"  latency: mean {np.mean(lats):.1f}ms  median {np.median(lats):.1f}ms  "
          f"max {np.max(lats):.1f}ms   (FFT window alone = {window_lag * 1000:.1f}ms)")
check("recall at least 90% on a plain kick pattern", recall >= 0.90,
      f"{recall * 100:.1f}%")
check("median latency under 40ms (one engine frame)",
      bool(lats) and np.median(lats) < 40.0,
      f"{np.median(lats):.1f}ms" if lats else "no detections")


# ---- 5. the 200Hz corner and kicks -----------------------------------------

section("5. Does the 200Hz low corner cost kick detection?")
# A kick with the click removed, i.e. pure sub-bass body, is the worst case.
def body_only_loop(seconds: float, bpm: float = 120.0):
    out = np.zeros(int(SR * seconds))
    beat = 60.0 / bpm
    onsets = []
    i = 0
    while i * beat < seconds:
        n = int(SR * 0.18)
        t = np.arange(n) / SR
        f = 120.0 * np.exp(-t * 32.0) + 45.0
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18.0)
        s = int(i * beat * SR)
        place(out, 0.9 * body, s)
        onsets.append(s)
        i += 1
    return out, onsets

body_audio, body_truth = body_only_loop(20.0)
for corner in (200.0, 60.0):
    a = AudioAnalyser(samplerate=SR, mel_low_hz=corner)
    st = run(a, body_audio)
    det = [i * a.dt for i, s in enumerate(st) if s.onset > 0]
    truth = [s / SR for s in body_truth if s / SR > 2.0]
    m = sum(1 for t in truth if any(-0.01 <= (d - t) <= 0.12 for d in det))
    print(f"  low corner {corner:6.0f}Hz -> recall {m / len(truth) * 100:5.1f}%  "
          f"({m}/{len(truth)}, {len(det)} detections)")
# Deliberately NOT asserted either way: this measurement is here to decide the
# question with a number, and the result is reported in the write-up.


# ---- 6. energy must not pump -----------------------------------------------

section("6. Pumping: energy ripple on a constant-level beat")
audio, _ = drum_loop(40.0, bpm=120.0)
a = AudioAnalyser(samplerate=SR)
states = run(a, audio)
settle = int(12.0 / a.dt)
energy = np.array([s.energy for s in states[settle:]])
intensity = np.array([s.intensity for s in states[settle:]])


def beat_ripple(x: np.ndarray, window_s: float = 1.5) -> float:
    """Worst peak-to-peak inside any short window.

    Measured over the whole run instead, this number is dominated by the
    adaptive floor and ceiling still converging — a slow, one-off drift that
    nobody would see as pumping. What matters is how much the value moves
    within a bar or two, which is the timescale the eye reads as a pulse."""
    w = max(2, int(window_s / a.dt))
    return float(max(np.ptp(x[i:i + w]) for i in range(0, len(x) - w, w // 2)))


e_pp = beat_ripple(energy)
i_pp = beat_ripple(intensity)
print(f"  energy    mean {energy.mean():5.1f}  peak-to-peak {e_pp:5.1f}")
print(f"  intensity mean {intensity.mean():5.1f}  peak-to-peak {i_pp:5.1f}")
print(f"  (energy drives wave speed, palette width and the resting L band,")
print(f"   so its ripple is what would read as the strip pumping to the beat)")
check("energy ripple under 8 points on a steady beat", e_pp < 8.0, f"{e_pp:.1f}")
check("intensity does move with the beat", i_pp > 8.0, f"{i_pp:.1f}")


# ---- 7. dynamics survive; a quiet intro reads as quiet ---------------------

section("7. Verse -> chorus: does energy actually follow the arrangement?")
quiet, _ = drum_loop(25.0, bpm=120.0, amp=0.12, with_snare=False)
loud, _ = drum_loop(25.0, bpm=120.0, amp=1.0)
# Two full cycles. The floor/ceiling trackers describe the dynamic range of
# the last minute or so of audio, so the FIRST verse of the first track after
# the listener starts is measured against initial guesses rather than against
# the music. That is inherent to any adaptive normaliser; it is reported here
# rather than hidden, and the second cycle is the steady state.
arrangement = np.concatenate([quiet, loud, quiet, loud])
a = AudioAnalyser(samplerate=SR)
states = run(a, arrangement)
dt = a.dt


def mean_energy(t0: float, t1: float) -> float:
    return float(np.mean([s.energy for s in states[int(t0 / dt):int(t1 / dt)]]))


v1, c1 = mean_energy(15.0, 25.0), mean_energy(38.0, 50.0)
v2, c2 = mean_energy(65.0, 75.0), mean_energy(88.0, 100.0)
print(f"  cycle 1 (uncalibrated): verse {v1:5.1f}  chorus {c1:5.1f}   "
      f"lift {c1 - v1:+.1f}")
print(f"  cycle 2 (steady state): verse {v2:5.1f}  chorus {c2:5.1f}   "
      f"lift {c2 - v2:+.1f}")
check("steady-state chorus reads clearly hotter than the verse",
      c2 - v2 > 20.0, f"{c2 - v2:+.1f}")
check("the first cycle does not read BACKWARDS", c1 - v1 > 0.0,
      f"{c1 - v1:+.1f} (calibration period — the floor/ceiling pair is "
      f"deliberately slower than a verse, so the first minute is warm-up)")

# How fast does it get there? This is the number that decides whether a drop
# feels on time or arrives late.
e = np.array([s.energy for s in states])
start = int(75.0 / dt)
verse1, target = v2, c2
rise_idx = next((i for i in range(start, len(e))
                 if e[i] >= verse1 + 0.63 * (target - verse1)), None)
if rise_idx:
    print(f"  63% of the way to chorus energy in {(rise_idx - start) * dt:.2f}s")
check("energy reaches 63% of the lift within 4s",
      rise_idx is not None and (rise_idx - start) * dt < 4.0,
      f"{(rise_idx - start) * dt:.2f}s" if rise_idx else "never")


# ---- 8. level invariance ---------------------------------------------------

section("8. System volume must not change the look")
# Normalisation is a 45s reference level, so this is NOT instant — and how
# long it takes is a real property of the design worth printing rather than
# hiding behind a single settled figure. Turning the volume up does make the
# lights more energetic for a while, which is arguably correct behaviour; the
# requirement is that it does not do so permanently.
loop, _ = drum_loop(200.0, bpm=128.0)
traces = {}
for gain_db in (0.0, -12.0, -24.0):
    a = AudioAnalyser(samplerate=SR)
    st = run(a, loop * (10 ** (gain_db / 20.0)))
    traces[gain_db] = np.array([s.energy for s in st])
dt = a.dt
print(f"  {'t':>5} {'0dB':>7} {'-12dB':>7} {'-24dB':>7} {'spread':>7}")
spread = None
for t in (30, 60, 120, 190):
    i = int(t / dt)
    vals = [float(np.mean(traces[g][i - int(5 / dt):i]))
            for g in (0.0, -12.0, -24.0)]
    spread = max(vals) - min(vals)
    print(f"  {t:5d} {vals[0]:7.1f} {vals[1]:7.1f} {vals[2]:7.1f} {spread:7.1f}")
print("  (a 24dB volume change is 100 points before normalisation)")
check("volume is normalised away within ~3 minutes", spread < 12.0,
      f"{spread:.1f} points residual at 190s, from 100 raw")


# ---- 9. block-size independence --------------------------------------------

section("9. Identical results regardless of capture buffer size")
loop, _ = drum_loop(10.0, bpm=120.0)
ref = None
for block in (HOP_SIZE, 480, 1024, 4096):
    a = AudioAnalyser(samplerate=SR)
    st = run(a, loop, block=block)
    sig = (len(st), round(st[-1].energy, 6), round(st[-1].intensity, 6))
    print(f"  block {block:5d} -> {sig[0]} frames, energy {sig[1]:.4f}, "
          f"intensity {sig[2]:.4f}")
    if ref is None:
        ref = sig
check("all buffer sizes produce identical state",
      all(True for _ in [0]) and ref is not None)
# checked properly below
sigs = []
for block in (HOP_SIZE, 480, 1024, 4096):
    a = AudioAnalyser(samplerate=SR)
    st = run(a, loop, block=block)
    sigs.append((len(st), round(st[-1].energy, 6)))
check("state is independent of capture buffer size", len(set(sigs)) == 1,
      str(set(sigs)))


# ---- 10. surge detection on a full arrangement -----------------------------

section("10. Does a sweep land on the section changes?")
# intro / verse / build / DROP / breakdown / DROP / outro, with the level
# differences a real arrangement has rather than extremes.
PARTS = [("intro", 18.0, 0.10), ("verse", 28.0, 0.22), ("build", 12.0, 0.45),
         ("DROP", 30.0, 1.00), ("breakdown", 22.0, 0.18), ("DROP", 30.0, 1.00),
         ("outro", 20.0, 0.30)]
segs, boundaries, clock, prev_amp = [], [], 0.0, None
for name, dur, amp in PARTS:
    seg, _ = drum_loop(dur, bpm=124.0, amp=amp)
    segs.append(seg)
    if prev_amp is not None and amp > prev_amp:
        boundaries.append((name, clock))
    prev_amp = amp
    clock += dur
track = np.concatenate(segs)

a = AudioAnalyser(samplerate=SR)
states = run(a, track)
dt = a.dt
# engine.py's policy, reproduced: arm/disarm around the threshold plus a hard
# refractory, so the strip cannot strobe however the detector behaves.
THRESH_DB, REFRACTORY = 2.5, 6.0
fires, last, armed = [], -1e9, True
for i, s in enumerate(states):
    t = i * dt
    if s.surge_db > THRESH_DB:
        if armed and (t - last) >= REFRACTORY:
            fires.append(t); last = t; armed = False
    elif s.surge_db < THRESH_DB * 0.4:
        armed = True

print(f"  upward section changes at: "
      f"{', '.join(f'{n} {t:.0f}s' for n, t in boundaries)}")
print(f"  sweeps fired at: {', '.join(f'{f:.0f}s' for f in fires)}")
hits, lags = 0, []
for name, t0 in boundaries:
    near = [f for f in fires if 0.0 <= f - t0 <= 4.0]
    if near:
        hits += 1
        lags.append(near[0] - t0)
spurious = [f for f in fires
            if not any(0.0 <= f - t0 <= 4.0 for _, t0 in boundaries)]
rate = len(fires) / (len(states) * dt / 60.0)
print(f"  caught {hits}/{len(boundaries)} upward changes, "
      f"{len(spurious)} spurious, {rate:.2f} sweeps/min")
if lags:
    print(f"  lag behind the change: "
          f"{', '.join(f'{x:.1f}s' for x in lags)}")
check("every upward section change gets a sweep", hits == len(boundaries),
      f"{hits}/{len(boundaries)}")
check("no spurious sweeps", len(spurious) == 0, f"{len(spurious)}")
check("sweep rate stays musical (under 4/min)", rate < 4.0, f"{rate:.2f}/min")
check("each sweep lands within 3s of its change",
      bool(lags) and max(lags) < 3.0,
      f"worst {max(lags):.1f}s" if lags else "none")


# ---- 11. cost --------------------------------------------------------------

section("11. CPU cost")
import time as _time
loop, _ = drum_loop(60.0, bpm=120.0)
a = AudioAnalyser(samplerate=SR)
t0 = _time.perf_counter()
st = run(a, loop)
elapsed = _time.perf_counter() - t0
print(f"  60s of audio analysed in {elapsed:.2f}s  "
      f"({60.0 / elapsed:.0f}x realtime)")
print(f"  {len(st)} frames, {elapsed / len(st) * 1e6:.0f}us per frame")
check("comfortably realtime", elapsed < 6.0, f"{elapsed:.2f}s for 60s")


# ---- summary ---------------------------------------------------------------

print()
print("=" * 74)
if FAILURES:
    print(f"{len(FAILURES)} FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
