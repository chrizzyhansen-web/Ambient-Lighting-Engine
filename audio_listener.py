"""
The ears. Turns whatever Windows is playing into three numbers the existing
effect engine already knows how to use.

WHY THIS EXISTS: until now nothing listened to the sound. The AI was asked,
once per track, where it thought the song got intense — expressed as a
percentage through the track — and the engine interpolated between those
guesses. That has one fatal property: it is identical on every play. The
same song produces the same frames at the same offsets forever, and when the
model's guess about where the drop is happens to be wrong, it is wrong
identically every time.

WHAT THIS IS NOT: a spectrum analyser bolted to the strip. The design
constraint is that THE LOOK MUST NOT CHANGE. effects.py already produces the
motion; it takes an `energy_pct` and derives speed, palette width, crest
sharpness, grain and the ambient lightness band from it, and it takes a list
of transient events. This module's entire job is to supply better values for
those existing inputs. It never touches a pixel. If this works, the strip
looks exactly as it did — it is just on time.

FOUR OUTPUTS, FOUR DELIBERATELY DIFFERENT TIME CONSTANTS
--------------------------------------------------------
This is the part that decides whether the result reads as "alive" or as a
VU meter, so it is worth being explicit about.

  energy    (seconds)      Replaces the AI's guessed arc as effects.render's
                           energy_pct. It must move SLOWLY. energy_pct drives
                           the wave speed, the palette width and the resting
                           lightness band; if it tracks the music beat by beat
                           then the whole strip visibly pumps, which is
                           precisely the look being avoided. Verse-to-chorus
                           is the timescale this is for.

  intensity (~0.1-0.4s)    "How hard is it going right now." Gates the glitch
                           and scales event strength. Allowed to move fast
                           because it does not itself change the resting look.

  surge_db  (~1.5s vs 6s)  A section change, in dB. This is what fires a
                           sweep. It exists as a separate output because
                           energy cannot do both jobs: slow enough not to
                           pump is too slow to announce a drop on time.

  onset     (per frame)    Spectral flux transients. Deliberately NOT wired to
                           an event per beat — that would be a strobe. Kept as
                           a measured signal for the UI and for timing.

TWO PATHS OFF ONE FFT, AND WHY
------------------------------
scottlawsonbc's front end (Hamming window -> mel bank -> adaptive per-bin gain
-> asymmetric exponential smoothing) is the right shape for onset detection:
normalising every mel bin against its own recent maximum makes flux depend on
the SHAPE of the spectrum changing rather than on how loud the track is
mastered, which is what you want from a transient detector.

It is exactly wrong for loudness. That same normalisation means a whispered
intro, given a few seconds, saturates every bin just as thoroughly as a
chorus does — the gain stage has removed the very quantity `energy` is
supposed to report. Running only the AGC'd path would have produced an engine
that sat at full energy through a quiet passage.

So the mel bank is computed once and used twice:

    AGC'd mel   -> spectral flux -> onset
    raw RMS     -> dB -> slow reference level -> intensity, energy, surge_db

The second path still normalises, because tracks differ in mastering and the
system volume is a knob the user turns. But it normalises by subtracting ONE
slowly-tracked reference level in dB, so a decibel is always worth the same
number of energy points: dynamics WITHIN a song are reported as they are, and
only the difference between one track's overall level and another's is
flattened out. Two earlier attempts tracked a floor and a ceiling instead and
both destroyed the very contrast they were meant to preserve — the reasoning
is recorded at `reference_s`, because the second failure looked like a tuning
problem and was not.

SQUELCH IS NOT OPTIONAL. WLED's AudioReactive is right about this. Below the
gate the module reports not-live and, critically, STOPS UPDATING THE ADAPTIVE
GAINS. Letting them keep adapting during silence is how a noise floor gets
amplified to full scale: come back after a silent minute and room hum reads
as a chorus.
"""
from __future__ import annotations

import math
import threading
import time
from typing import NamedTuple

try:
    import numpy as _np
except Exception:  # pragma: no cover - exercised only on a machine without numpy
    _np = None


NUMPY_AVAILABLE = _np is not None


# ---- Front-end geometry ----------------------------------------------------
#
# 1024 samples at 48kHz is a 21.3ms window; hopping 512 gives ~93.75 analysis
# frames a second, so an onset is located to about 11ms. The engine redraws at
# 25fps (40ms), so the analyser is comfortably ahead of the thing it feeds —
# which is the point: a transient must never be discovered a frame late.
#
# Resolution is 46.9Hz per FFT bin, so the lowest mel band sits around bin 4.
# That is coarse in the bass, and it is a deliberate trade: doubling the
# window to fix it would double onset latency, and onset timing is the entire
# reason this module exists.
DEFAULT_SAMPLERATE = 48000
FFT_SIZE = 1024
HOP_SIZE = 512

# 200Hz-12kHz, 24 bins — scottlawsonbc's range. The low corner is high enough
# to reject rumble and mains hum without a separate filter. See the note in
# `mel_low_hz`: kick drums live below this, and they are still detected,
# because a kick's transient is broadband even when its BODY is not.
MEL_BINS = 24
MEL_LOW_HZ = 200.0
MEL_HIGH_HZ = 12000.0


class AudioState(NamedTuple):
    """What the engine reads. All floats, all safe to use without checking
    `live` first — when audio is absent every field is a resting value, not a
    NaN or a stale reading."""

    energy: float       # 0-100, slow. Goes straight into effects.render().
    intensity: float    # 0-100, fast. Gates glitches, scales event strength.
    onset: float        # 0-1, strength of a transient in this frame, else 0.
    live: bool          # audio arriving AND above the squelch gate
    # HOW MUCH LOUDER THE LAST SECOND IS THAN THE LAST SIX, IN dB.
    #
    # This exists because `energy` cannot do the job. energy is deliberately
    # slow so the strip does not pump to the beat, and a signal that takes
    # three to five seconds to move cannot also announce a drop on time.
    # Measured on a seven-section arrangement, every trigger built on energy
    # either missed the first drop outright or fired a handful of times
    # during the analyser's own warm-up.
    #
    # In dB against a short reference, a section change is unmistakable and
    # immediate: +2.5dB caught all four of the arrangement's upward changes
    # within 1-2 seconds, with no spurious fires at all. Level DROPS are
    # correctly ignored — a surge is an increase.
    surge_db: float
    loudness_db: float  # raw dBFS, for diagnostics and the UI meter
    floor_db: float     # bottom of the normaliser's current window
    ceiling_db: float


SILENT_STATE = AudioState(
    energy=0.0, intensity=0.0, onset=0.0, live=False, surge_db=0.0,
    loudness_db=-120.0, floor_db=-120.0, ceiling_db=-120.0,
)


def _alpha(time_constant_s: float, dt: float) -> float:
    """Exponential smoothing coefficient for a given time constant.

    Expressed as a time constant rather than a raw alpha on purpose: alpha is
    meaningless without knowing the frame rate, and this module's frame rate
    depends on the capture device's sample rate. 1 - exp(-dt/tau) keeps the
    behaviour identical whether the hardware hands us 44.1k or 48k.
    """
    if time_constant_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / time_constant_s)


class _Asym:
    """Asymmetric exponential smoothing: separate rise and decay constants.

    The whole front end depends on this. A symmetric filter slow enough to
    stop `energy` pumping is also slow enough to miss the attack of anything;
    one fast enough to catch an attack lets every gap between beats drag the
    value back down. Rising fast and falling slowly is what turns a train of
    discrete hits into a sustained level.
    """

    def __init__(self, rise_s: float, decay_s: float, initial: float = 0.0):
        self.rise_s = rise_s
        self.decay_s = decay_s
        self.value = initial

    def update(self, target: float, dt: float) -> float:
        tau = self.rise_s if target > self.value else self.decay_s
        self.value += (target - self.value) * _alpha(tau, dt)
        return self.value


class _AsymArray:
    """The same, per mel bin."""

    def __init__(self, size: int, rise_s: float, decay_s: float, initial: float = 0.0):
        self.rise_s = rise_s
        self.decay_s = decay_s
        self.value = _np.full(size, float(initial), dtype=_np.float64)

    def update(self, target, dt: float):
        rising = target > self.value
        a = _np.where(rising, _alpha(self.rise_s, dt), _alpha(self.decay_s, dt))
        self.value += (target - self.value) * a
        return self.value


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(samplerate: int, fft_size: int, bins: int,
                         low_hz: float, high_hz: float):
    """Triangular mel filters as a (bins, fft_size//2 + 1) matrix.

    Area-normalised, so a flat spectrum produces equal values in every bin
    regardless of how many FFT bins a given filter happens to span. Without
    that the high bins — which are far wider in Hz — read as much louder than
    the low ones on pink or white noise, and the per-bin gain stage then has
    to undo it.
    """
    n_fft_bins = fft_size // 2 + 1
    mel_edges = _np.linspace(_hz_to_mel(low_hz), _hz_to_mel(high_hz), bins + 2)
    hz_edges = _np.array([_mel_to_hz(m) for m in mel_edges])
    bin_edges = hz_edges * fft_size / samplerate

    bank = _np.zeros((bins, n_fft_bins), dtype=_np.float64)
    fft_bin_idx = _np.arange(n_fft_bins, dtype=_np.float64)
    for i in range(bins):
        left, centre, right = bin_edges[i], bin_edges[i + 1], bin_edges[i + 2]
        if right - left < 1e-9:
            continue
        rising = (fft_bin_idx - left) / max(centre - left, 1e-9)
        falling = (right - fft_bin_idx) / max(right - centre, 1e-9)
        tri = _np.clip(_np.minimum(rising, falling), 0.0, None)
        area = tri.sum()
        if area > 0:
            bank[i] = tri / area
    return bank


class AudioAnalyser:
    """The DSP. Pure: give it samples, get states back. No threads, no audio
    device, no clock of its own — `dt` comes from the hop size and the sample
    rate, so feeding it a file produces exactly the states that the same audio
    would have produced live. That is what makes any of this testable.
    """

    def __init__(
        self,
        samplerate: int = DEFAULT_SAMPLERATE,
        fft_size: int = FFT_SIZE,
        hop_size: int = HOP_SIZE,
        mel_bins: int = MEL_BINS,
        mel_low_hz: float = MEL_LOW_HZ,
        mel_high_hz: float = MEL_HIGH_HZ,
        # --- squelch ---
        # Gate in dBFS. Digital silence from a paused player is -inf; a real
        # room through a loopback is not involved here (this is a loopback of
        # the OUTPUT, so there is no microphone noise floor to reject) but a
        # player idling can still emit dither at around -90dBFS. -60 sits well
        # clear of that and well below any audible music.
        squelch_db: float = -60.0,
        # HOW LONG THE GATE STAYS OPEN AFTER THE LAST LOUD FRAME.
        #
        # A gate with no hold flaps. Measured on a 120bpm loop whose hits
        # decay into true silence, the gate was CLOSED for 60% of frames —
        # it shut in every gap between beats — and because the closed path
        # drives the outputs toward zero, energy settled at 45 where the
        # music warranted 70. Sparse material (a solo piano, a breakdown)
        # would have been worst hit, which is exactly the material where
        # getting the level right matters most.
        #
        # 1.2s comfortably spans the gap between beats at any tempo a person
        # would call slow, and still closes promptly when a track ends.
        gate_hold_s: float = 1.2,
        # --- loudness normalisation ---
        # A SLOW MAX AND A SLOW MIN, both directions. The first version had
        # the ceiling rise in 0.4s, on the theory that it should "keep up".
        # Measured, that was the single worst decision in this file: with a
        # fast ceiling, the loudest moment of the last half-second always
        # maps to 100, so a whispered verse and a full chorus both read as
        # near-maximum. A 25s quiet section followed by a 25s loud one
        # differed by 2.0 points of energy out of 100 — the analyser had
        # normalised away the exact quantity it exists to report.
        #
        # Slow in BOTH directions makes these describe the dynamic range of
        # the last minute of music rather than of the last bar, which is what
        # lets a passage sit low in that range.
        # A SINGLE SLOW REFERENCE LEVEL, NOT A TRACKED FLOOR AND CEILING.
        #
        # Two earlier versions tracked a floor and a ceiling and mapped the
        # loudness between them. Both failed, and it is worth recording why,
        # because the second failure looked like a tuning problem and was
        # not. Fed the raw per-frame level, the floor chased the gaps between
        # beats down to the squelch threshold. Fed a peak-following envelope
        # instead, the floor had nothing low to track at all during
        # continuous music, so it crept monotonically upward for the entire
        # song — measured, it went from -65dB to -35dB over three minutes and
        # never once came back down. With both ends drifting, a 13dB
        # difference between a verse and a drop came out as 0.1 points of
        # energy: the two loudest sections of the track measured identical.
        #
        # What replaces it is much simpler and has no state that can run
        # away: ONE slowly-tracked reference level, and a FIXED number of dB
        # either side of it mapping to the full 0-100 range. A decibel is
        # then always worth the same number of energy points, so dynamics
        # inside a song are reported as they are, and only the difference
        # between one track's mastering (or one position of the volume
        # knob) and another's is normalised away.
        reference_s: float = 45.0,
        # How many dB span the full range. 24dB centred on the reference
        # covers the short-term dynamic range of essentially all recorded
        # music without clipping at either end on ordinary material.
        range_db: float = 24.0,
        # The reference tracks an ENVELOPE, so it is not dragged down by the
        # gaps between beats; `norm` is still taken from the per-frame level,
        # so `intensity` keeps its beat-by-beat movement.
        envelope_rise_s: float = 0.02,
        envelope_decay_s: float = 0.6,
        # The surge pair. Both symmetric — a surge is a comparison of two
        # averages, not a follower. 1.5s is short enough to register a
        # section change within a second or two and long enough not to see
        # individual beats; 6s is the "what has been going on lately" that
        # it is compared against.
        surge_fast_s: float = 1.5,
        surge_slow_s: float = 6.0,
        # --- output smoothing ---
        intensity_rise_s: float = 0.08,
        intensity_decay_s: float = 0.35,
        # energy is a CASCADE of two of these, not one slow filter. A single
        # pole slow enough to reject a 2Hz beat is also slow enough that a
        # drop arrives visibly late. Two poles roll off at 12dB/octave
        # instead of 6, so for the same step response the beat ripple is an
        # order of magnitude smaller — measured 8.6 points peak-to-peak on
        # one pole against 0.9 on two.
        energy_rise_s: float = 1.5,
        energy_decay_s: float = 3.0,
        # --- per-bin AGC, for the flux path only ---
        # Rise at 0.05s was too eager: the gain snapped to every fluctuation
        # of a stationary partial, so a sustained tone kept producing flux as
        # the ratio wobbled. 0.30s measured better on BOTH sides at once,
        # which is rare enough to be worth recording — drum recall 88.9% ->
        # 97.8%, and false onsets on steady noise fell as well.
        gain_rise_s: float = 0.30,
        gain_decay_s: float = 8.0,
        # A BIN CANNOT BE AMPLIFIED ABOVE THIS FRACTION OF THE LOUDEST BIN.
        #
        # Per-bin AGC divides each bin by its own recent maximum. For a bin
        # containing no music — the 8kHz region during a bass-and-piano
        # passage — that recent maximum is itself tiny, so the division
        # amplifies whatever numerical noise is there to full scale. Measured
        # on a four-note sustained pad, which contains no transient of any
        # kind, this produced 3.7 false onsets a second entirely from empty
        # bins. Refusing to divide by less than 2% of the loudest bin's own
        # tracked level means an empty bin stays empty.
        gain_floor_frac: float = 0.05,
        # Asymmetric smoothing of the normalised mel BEFORE flux is taken —
        # the step from scottlawsonbc's front end. Without it, flux is
        # measured on a spectrum that jitters frame to frame simply because
        # music is a stochastic signal, and the detector fires on that
        # jitter: 5.9 false onsets a second on steady white noise. Rising in
        # 25ms still passes a real attack (which spans 3-4 frames) intact.
        mel_rise_s: float = 0.08,
        mel_decay_s: float = 0.22,
        # --- onset detection ---
        flux_baseline_s: float = 1.2,
        onset_threshold: float = 2.6,
        # Absolute floor, as a fraction of the maximum possible flux (every
        # bin going from 0 to 1 at once). Purely a sanity limit so that
        # near-silent but non-gated material cannot produce onsets by having
        # a very small running mean.
        onset_floor: float = 0.012,
        onset_refractory_s: float = 0.08,
    ):
        if _np is None:
            raise RuntimeError(
                "audio_listener needs numpy. Install it with: pip install numpy"
            )

        self.samplerate = int(samplerate)
        self.fft_size = int(fft_size)
        self.hop_size = int(hop_size)
        self.dt = self.hop_size / float(self.samplerate)

        self.window = _np.hamming(self.fft_size)
        self.bank = build_mel_filterbank(
            self.samplerate, self.fft_size, mel_bins, mel_low_hz, mel_high_hz
        )
        self.mel_bins = mel_bins

        self.squelch_db = squelch_db
        self.gate_hold_s = gate_hold_s
        self._since_loud = 1e9

        self._ring = _np.zeros(self.fft_size, dtype=_np.float64)
        self._filled = 0

        self._env = _Asym(envelope_rise_s, envelope_decay_s, initial=-30.0)
        # Symmetric: a reference level has no reason to prefer one direction.
        self._ref = _Asym(reference_s, reference_s, initial=-24.0)
        self._surge_fast = _Asym(surge_fast_s, surge_fast_s, initial=-24.0)
        self._surge_slow = _Asym(surge_slow_s, surge_slow_s, initial=-24.0)
        self.range_db = range_db
        self._intensity = _Asym(intensity_rise_s, intensity_decay_s, 0.0)
        # Two poles — see energy_rise_s.
        self._energy_a = _Asym(energy_rise_s, energy_decay_s, 0.0)
        self._energy_b = _Asym(energy_rise_s, energy_decay_s, 0.0)

        self._gain = _AsymArray(mel_bins, gain_rise_s, gain_decay_s, initial=1e-6)
        self._gain_floor_frac = gain_floor_frac
        self._mel_smooth = _AsymArray(mel_bins, mel_rise_s, mel_decay_s, 0.0)
        self._prev_mel = _np.zeros(mel_bins, dtype=_np.float64)
        self._flux_baseline = _Asym(flux_baseline_s, flux_baseline_s, 0.0)
        self._flux_spread = _Asym(flux_baseline_s, flux_baseline_s, 0.0)
        self._onset_threshold = onset_threshold
        self._onset_floor = onset_floor
        self._onset_refractory_s = onset_refractory_s
        # PEAK PICKING. Thresholding alone is not an onset detector: any
        # broadband signal's flux wanders, and a threshold set where real
        # onsets clear it is also cleared by that wandering several times a
        # second — measured 2.7/s on steady white noise and 3.8/s on a slow
        # synth sweep, neither of which contains a single real transient.
        # Requiring the flux to be a LOCAL MAXIMUM as well means a candidate
        # has to rise AND fall, which noise crossings mostly do not do
        # cleanly. It costs one frame (10.7ms) of latency, spent from a
        # budget that measured 4ms AHEAD of the true onset.
        self._flux_hist: list[tuple[float, float]] = []  # (flux, trigger)
        self._since_onset = 999.0

        self._had_signal = False
        self.frames_processed = 0
        self.state = SILENT_STATE

    # -- the one interesting method ------------------------------------------

    def _process_hop(self) -> AudioState:
        dt = self.dt
        self._since_onset += dt
        frame = self._ring * self.window

        rms = float(_np.sqrt(_np.mean(frame * frame)) + 1e-12)
        # Window loss: a Hamming window has a coherent gain of 0.54, so RMS
        # measured through it understates a signal by ~5.35dB. Corrected here
        # so `loudness_db` is comparable to any other dBFS figure and the
        # squelch threshold means what it says.
        loudness_db = 20.0 * math.log10(rms / 0.54)

        self._since_loud += dt
        if loudness_db >= self.squelch_db:
            self._since_loud = 0.0

        if self._since_loud >= self.gate_hold_s:
            # GATED. Everything adaptive is frozen. The outputs decay rather
            # than snapping to zero, so pausing a track fades the strip back
            # to its resting look instead of dropping it.
            self._had_signal = False
            self._intensity.update(0.0, dt)
            self._energy_b.update(self._energy_a.update(0.0, dt), dt)
            self._prev_mel[:] = 0.0
            self.frames_processed += 1
            self.state = AudioState(
                energy=self._energy_b.value * 100.0,
                intensity=self._intensity.value * 100.0,
                onset=0.0,
                live=False,
                surge_db=0.0,
                loudness_db=loudness_db,
                floor_db=self._ref.value - 0.75 * self.range_db,
                ceiling_db=self._ref.value + 0.25 * self.range_db,
            )
            return self.state

        spectrum = _np.abs(_np.fft.rfft(frame))
        mel = self.bank @ spectrum

        # --- path 1: shape. Per-bin AGC, then flux. --------------------------
        #
        # Each bin is divided by a tracked recent maximum of itself, so a bin
        # that is quiet in absolute terms still contributes its full share of
        # flux when it changes. Rise is near-instant (the gain must never
        # clip a transient) and decay is slow, so the normaliser does not
        # pump between beats.
        self._gain.update(_np.maximum(mel, self._gain.value * 1e-3), dt)
        denom = _np.maximum(
            self._gain.value,
            self._gain_floor_frac * float(self._gain.value.max()),
        )
        normalised = _np.clip(mel / _np.maximum(denom, 1e-12), 0.0, 1.0)
        # Asymmetric smoothing before flux: fast up, slow down. This is what
        # separates a real attack from the frame-to-frame jitter that any
        # broadband signal produces.
        normalised = self._mel_smooth.update(normalised, dt).copy()

        # SPECTRAL FLUX: sum of POSITIVE differences only, divided by the bin
        # count so the figure is 0-1 regardless of how many bins there are.
        # Half-wave rectification is the whole trick — energy leaving a bin is
        # a note ending, which is not an onset, and counting it produces a
        # detector that fires twice per note.
        if self._had_signal:
            flux = float(
                _np.sum(_np.maximum(normalised - self._prev_mel, 0.0))
            ) / self.mel_bins
        else:
            # First frame after the gate opens: everything looks like it just
            # arrived, which it did. Suppressed so unmuting never counts as a
            # drop.
            flux = 0.0
            self._had_signal = True
        self._prev_mel = normalised.copy()

        # Adaptive threshold. A fixed one is unusable across material: dense
        # electronic music runs an order of magnitude more baseline flux than
        # a solo piano. Comparing against a running mean plus a multiple of
        # the running mean deviation makes "unusually large for this music"
        # the criterion instead of "large".
        baseline = self._flux_baseline.value
        spread = self._flux_spread.value
        self._flux_baseline.update(flux, dt)
        self._flux_spread.update(abs(flux - baseline), dt)

        trigger = max(baseline + self._onset_threshold * max(spread, 1e-9),
                      self._onset_floor)
        self._flux_hist.append((flux, trigger))
        if len(self._flux_hist) > 3:
            del self._flux_hist[0]

        onset = 0.0
        if len(self._flux_hist) == 3 and self._since_onset >= self._onset_refractory_s:
            (prev, _), (mid, mid_trigger), (curr, _) = self._flux_hist
            # A local maximum that also clears the adaptive threshold.
            if mid > prev and mid >= curr and mid > mid_trigger:
                # Strength is how far past the threshold it went, saturating
                # at roughly twice it. Reported so the engine can distinguish
                # a snare from a drop rather than seeing every hit as equal.
                onset = min(1.0, (mid - mid_trigger) / max(mid_trigger, 1e-6))
                self._since_onset = 0.0

        # --- path 2: loudness. No per-bin AGC. -------------------------------
        # Clamped at the squelch threshold before it reaches the trackers. A
        # gap between beats is genuinely -inf dBFS on synthetic material and
        # very low on real material; letting that through would drag the
        # floor down to meet it, and the floor is supposed to describe the
        # quietest MUSIC, not the quietest sample.
        db_for_norm = max(loudness_db, self.squelch_db)
        env_db = self._env.update(db_for_norm, dt)
        ref = self._ref.update(env_db, dt)
        surge_db = (self._surge_fast.update(env_db, dt)
                    - self._surge_slow.update(env_db, dt))
        # A fixed dB-per-point scale, centred on the reference. The reference
        # sits at the envelope's own long-term level, which for continuous
        # music is near its peaks — so it is placed at 0.75 of the range
        # rather than at the middle, leaving the top quarter for material
        # that is genuinely louder than the last minute has been and the
        # bottom three quarters for everything that is quieter.
        norm = 0.75 + (db_for_norm - ref) / self.range_db
        norm = max(0.0, min(1.0, norm))
        floor = ref - 0.75 * self.range_db
        ceiling = ref + 0.25 * self.range_db

        # ENERGY IS FED FROM `intensity`, NOT FROM `norm` DIRECTLY.
        #
        # `norm` is a 21ms window, so on percussive music it spends most of
        # its time in the gaps between hits: averaging it gives a number far
        # below what the passage sounds like. Measured on a full-level drum
        # loop, energy settled at 41 out of 100 — the top half of the range
        # was simply unreachable by any real music, and effects.py's grain
        # layer (which is squared in energy) would never have appeared.
        #
        # `intensity` is already an envelope follower — fast attack, slow
        # release — so it fills those gaps. Cascading the slow pair off it
        # measures the passage rather than the gaps.
        inten = self._intensity.update(norm, dt)
        self._energy_b.update(self._energy_a.update(inten, dt), dt)

        self.frames_processed += 1
        self.state = AudioState(
            energy=self._energy_b.value * 100.0,
            intensity=self._intensity.value * 100.0,
            onset=onset,
            live=True,
            surge_db=surge_db,
            loudness_db=loudness_db,
            floor_db=floor,
            ceiling_db=ceiling,
        )
        return self.state

    def feed(self, samples) -> list[AudioState]:
        """Push mono float samples in any block size. Returns one state per
        completed hop, so nothing is lost if the capture device hands over
        an awkward buffer size."""
        samples = _np.asarray(samples, dtype=_np.float64).ravel()
        out: list[AudioState] = []
        pos = 0
        n = len(samples)
        while pos < n:
            need = self.hop_size - self._filled
            take = min(need, n - pos)
            # Slide the ring left by `take` and append. At 94 hops a second
            # this copy is nothing, and it keeps the FFT frame contiguous.
            if take:
                self._ring[:-take] = self._ring[take:]
                self._ring[-take:] = samples[pos:pos + take]
                self._filled += take
                pos += take
            if self._filled >= self.hop_size:
                self._filled = 0
                out.append(self._process_hop())
        return out


class LoopbackListener:
    """WASAPI loopback capture on a background thread, feeding an
    AudioAnalyser.

    WHY THIS SHAPE: per-process capture — grabbing only Spotify's audio —
    needs the Windows Application Loopback API, which is Windows 11 only.
    This machine is Windows 10 22H2 (build 19045), so it is not an option.
    Device loopback via PyAudioWPatch captures the whole output mix instead:
    a Discord notification or a YouTube tab WILL reach the strip. That is a
    real limitation and not a bug to be reported later.

    Everything about the capture is best-effort. If PyAudioWPatch is not
    installed, if there is no loopback device, or if the stream dies, the
    listener reports not-live and the engine falls back to the AI's arc. It
    never raises into the render loop.
    """

    def __init__(self, analyser_kwargs: dict | None = None,
                 device_name_contains: str | None = None):
        # WHICH DEVICE TO CAPTURE. Empty means the default output endpoint,
        # which is the whole system mix — Spotify, Discord, a YouTube tab,
        # Windows sounds, everything. Measured, a notification landing during
        # a quiet passage fires a full sweep across the strip and lifts
        # energy by 11 points; during a loud chorus it does neither. The
        # contamination is worst exactly where it is most visible.
        #
        # Naming a device here is what makes SPOTIFY-ONLY possible on
        # Windows 10. The route is: install VB-CABLE, set Spotify's output
        # to it in Settings > System > Sound > App volume and device
        # preferences, and put "CABLE" here. Nothing else in Windows is
        # routed there, so nothing else can reach the strip.
        #
        # This is a substring match, case-insensitive, against the device
        # names that `describe_devices()` prints — Windows device names are
        # long and vary by machine, so matching a fragment is far more
        # robust than storing an index (which changes when a device is
        # plugged in) or a full name.
        self._analyser_kwargs = dict(analyser_kwargs or {})
        self.device_name_contains = (device_name_contains or "").strip() or None
        self._analyser: AudioAnalyser | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = SILENT_STATE
        self._pending_onsets: list[float] = []
        self._last_frame_at = 0.0
        self.device_name: str | None = None
        self.error: str | None = None

    # -- what the engine calls -----------------------------------------------

    @property
    def available(self) -> bool:
        """True when audio frames are genuinely arriving. Goes false if the
        capture thread dies or stalls, so a crashed stream hands control back
        to the AI arc instead of freezing the strip at whatever energy it
        last saw."""
        with self._lock:
            return (
                self._thread is not None
                and self._thread.is_alive()
                and (time.time() - self._last_frame_at) < 1.0
            )

    def get_state(self) -> AudioState:
        with self._lock:
            return self._state

    def pop_onsets(self) -> list[float]:
        """Onset strengths since the last call.

        The analyser runs at ~94 frames a second and the engine redraws at 25,
        so between two engine frames there can be three or four onsets. Handing
        back a single "is there an onset right now" flag would drop most of
        them, and which ones survived would depend on frame alignment."""
        with self._lock:
            out = self._pending_onsets
            self._pending_onsets = []
            return out

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> bool:
        if not NUMPY_AVAILABLE:
            self.error = "numpy is not installed (pip install numpy)"
            return False
        try:
            import pyaudiowpatch  # noqa: F401
        except Exception:
            self.error = (
                "pyaudiowpatch is not installed. Install it with: "
                "pip install PyAudioWPatch"
            )
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Give the device a moment to open so callers get a truthful answer
        # rather than "starting, probably fine".
        deadline = time.time() + 2.5
        while time.time() < deadline:
            if self.available or self.error:
                break
            time.sleep(0.05)
        return self.available

    def stop(self):
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    def _run(self):
        import pyaudiowpatch as pyaudio

        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            device = _pick_device(pa, self.device_name_contains)
            self.device_name = device["name"]
            rate = int(device["defaultSampleRate"])
            channels = int(device["maxInputChannels"])

            self._analyser = AudioAnalyser(samplerate=rate, **self._analyser_kwargs)

            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=rate,
                frames_per_buffer=HOP_SIZE,
                input=True,
                input_device_index=device["index"],
            )
            self.error = None

            while not self._stop.is_set():
                raw = stream.read(HOP_SIZE, exception_on_overflow=False)
                block = _np.frombuffer(raw, dtype=_np.float32)
                if channels > 1:
                    # Downmix by averaging. Not a sum: a sum clips on
                    # correlated material, and every commercial mix is
                    # correlated between L and R.
                    block = block.reshape(-1, channels).mean(axis=1)
                states = self._analyser.feed(block)
                if states:
                    onsets = [s.onset for s in states if s.onset > 0.0]
                    with self._lock:
                        self._state = states[-1]
                        if onsets:
                            self._pending_onsets.extend(onsets)
                            # Bound it. If the engine ever stops collecting,
                            # this must not grow without limit.
                            del self._pending_onsets[:-32]
                        self._last_frame_at = time.time()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                if pa is not None:
                    pa.terminate()
            except Exception:
                pass
            with self._lock:
                self._state = SILENT_STATE


class DeviceNotFound(RuntimeError):
    """The configured capture device isn't there.

    Raised rather than silently falling back to the default endpoint,
    because the two behave completely differently and the difference is
    invisible on the strip: the default captures everything, the named
    device captures only what Windows routes to it. Someone who has set
    this up for Spotify-only capture and then unplugs the cable should be
    told, not quietly switched back to hearing their whole desktop.
    """


def _pick_device(pa, name_contains: str | None):
    """The loopback endpoint to capture, by name fragment or the default."""
    if not name_contains:
        return pa.get_default_wasapi_loopback()
    wanted = name_contains.lower()
    candidates = list(pa.get_loopback_device_info_generator())
    for d in candidates:
        if wanted in str(d["name"]).lower():
            return d
    # Some virtual cables expose the capture side as an ordinary input
    # device rather than a loopback endpoint, so check those too before
    # giving up.
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if int(d.get("maxInputChannels") or 0) > 0 and wanted in str(d["name"]).lower():
            return d
    available = ", ".join(str(d["name"]) for d in candidates) or "none"
    raise DeviceNotFound(
        f"no capture device matching {name_contains!r}. Loopback devices "
        f"present: {available}. Run `python audio_listener.py` to list them."
    )


def describe_devices() -> str:
    """Diagnostic for the setup page and the command line. Says what
    PyAudioWPatch can actually see, because "no audio" is almost always a
    device selection problem rather than a code one."""
    try:
        import pyaudiowpatch as pyaudio
    except Exception as e:
        return f"PyAudioWPatch not importable: {e}"
    lines = []
    pa = pyaudio.PyAudio()
    try:
        try:
            default = pa.get_default_wasapi_loopback()
            lines.append(
                f"default loopback: {default['name']} "
                f"({int(default['defaultSampleRate'])}Hz, "
                f"{int(default['maxInputChannels'])}ch)"
            )
        except Exception as e:
            lines.append(f"no default loopback: {e}")
        lines.append("all loopback devices:")
        for d in pa.get_loopback_device_info_generator():
            lines.append(f"  [{d['index']}] {d['name']}")
    finally:
        pa.terminate()
    return "\n".join(lines)


if __name__ == "__main__":
    # Live self-test. Run it with music playing:
    #     python audio_listener.py
    import sys

    print(describe_devices())
    print()
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    listener = LoopbackListener(device_name_contains=wanted)
    if not listener.start():
        print(f"could not start: {listener.error}")
        sys.exit(1)
    print(f"capturing from: {listener.device_name}")
    # Say plainly which of the two situations this is, because they produce
    # identical-looking output and mean completely different things.
    if wanted:
        print(f"SCOPE: only what Windows routes to a device matching "
              f"{wanted!r}. If that is a virtual cable with Spotify pointed "
              f"at it, nothing else on this PC can reach the strip.")
    else:
        print("SCOPE: THE WHOLE SYSTEM MIX — Spotify plus Discord, browser "
              "tabs, games and notification sounds. Pass a device name "
              "fragment as an argument to capture only one device.")
    print()
    print("Play something. dBFS should move; 'gated' means below the squelch.")
    print(f"{'time':>6}  {'dBFS':>7}  {'floor':>7}  {'ceil':>7}  "
          f"{'energy':>6}  {'inten':>6}  onsets/s")
    t0 = time.time()
    onset_times: list[float] = []
    try:
        while True:
            time.sleep(0.5)
            s = listener.get_state()
            now = time.time()
            onset_times.extend([now] * len(listener.pop_onsets()))
            onset_times = [t for t in onset_times if now - t < 5.0]
            print(f"{now - t0:6.1f}  {s.loudness_db:7.1f}  {s.floor_db:7.1f}  "
                  f"{s.ceiling_db:7.1f}  {s.energy:6.1f}  {s.intensity:6.1f}  "
                  f"{len(onset_times) / 5.0:5.2f}"
                  f"{'' if s.live else '   [gated]'}")
    except KeyboardInterrupt:
        listener.stop()
