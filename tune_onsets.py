"""Trade-off curve for the onset detector's threshold. Not part of the test
suite — this is the working that chose the constant."""
import numpy as np
from audio_listener import AudioAnalyser, HOP_SIZE

SR = 48000
RNG = np.random.default_rng(7)


def kick(amp=0.9):
    n = int(SR * 0.18); t = np.arange(n) / SR
    f = 120.0 * np.exp(-t * 32.0) + 45.0
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 18.0)
    cn = int(SR * 0.004); click = np.zeros(n)
    click[:cn] = RNG.standard_normal(cn) * np.exp(-np.arange(cn) / (cn / 3))
    return amp * (0.85 * body + 0.5 * click)


def snare(amp=0.7):
    n = int(SR * 0.14); t = np.arange(n) / SR
    return amp * RNG.standard_normal(n) * np.exp(-t * 26.0)


def loop(sec, bpm=120.0, amp=1.0, with_snare=True):
    out = np.zeros(int(SR * sec)); beat = 60.0 / bpm; i = 0; on = []
    while i * beat < sec:
        s = int(i * beat * SR)
        srcs = [kick() * amp] + ([snare() * amp] if (with_snare and i % 2) else [])
        for src in srcs:
            e = min(len(out), s + len(src)); out[s:e] += src[:e - s]
        on.append(s); i += 1
    return out, on


def run(a, audio):
    st = []
    for i in range(0, len(audio), HOP_SIZE):
        st.extend(a.feed(audio[i:i + HOP_SIZE]))
    return st


drums, truth = loop(25.0)
truth = [s / SR for s in truth if s / SR > 2.0]
noise20 = 0.15 * RNG.standard_normal(int(SR * 20))
t = np.arange(int(SR * 20)) / SR
sweep = 0.2 * np.sin(2 * np.pi * np.cumsum(300.0 * (10.0 ** (t / 20.0))) / SR)
pad = 0.2 * sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330, 440)) / 4

print(f"{'thr':>5} {'melrise':>8} | {'recall':>7} {'extra':>6} | "
      f"{'noise/s':>8} {'sweep/s':>8} {'pad/s':>7}")
for thr in (2.6, 3.5, 4.5, 6.0, 8.0):
    for melrise in (0.025, 0.05):
        kw = dict(onset_threshold=thr, mel_rise_s=melrise)
        a = AudioAnalyser(samplerate=SR, **kw)
        det = [i * a.dt for i, s in enumerate(run(a, drums)) if s.onset > 0]
        m = sum(1 for x in truth if any(-0.01 <= (d - x) <= 0.12 for d in det))
        rates = []
        for sig in (noise20, sweep, pad):
            a2 = AudioAnalyser(samplerate=SR, **kw)
            st2 = run(a2, sig); settle = int(2.0 / a2.dt)
            n = sum(1 for s in st2[settle:] if s.onset > 0)
            rates.append(n / (len(st2[settle:]) * a2.dt))
        print(f"{thr:5.1f} {melrise:8.3f} | {m / len(truth) * 100:6.1f}% "
              f"{len(det) - m:6d} | {rates[0]:8.2f} {rates[1]:8.2f} {rates[2]:7.2f}")
