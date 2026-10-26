"""
Setup state and self-verification for the audio listener.

THE PROBLEM THIS SOLVES. Capturing the whole system mix and capturing only
Spotify produce output that is indistinguishable from the inside: the same
numbers, in the same ranges, at the same rate. Nothing about a stream of
audio frames says where they came from. So a setup that "looks fine" and a
setup that is quietly feeding your Discord notifications to the light strip
are, to every other part of this app, identical.

That is what makes audio setup feel fragile, and it is not fixed by making
the install easier. Windows 10 cannot do per-process audio capture at all
(the Application Loopback API needs build 20348+; this machine is 19045), so
Spotify-only means routing Spotify to a virtual cable and capturing that —
a driver install, a UAC prompt, a reboot and a per-app routing setting. Each
of those can be silently undone later. The per-app routing in particular is
known to reset after reboots and app updates, and when it does, nothing
visibly breaks. The lights just quietly start reacting to your browser.

SO THIS MODULE DOES NOT TRY TO PROVE THE SETUP IS RIGHT. It proves it
EMPIRICALLY, continuously, from evidence, using something most
audio-reactive projects do not have: a second, independent source of truth
about what Spotify is doing.

The Spicetify bridge already tells this app exactly when Spotify is playing.
So:

    Spotify is paused, AND audio is arriving  ->  that audio is not Spotify.

That single inference is enough. It needs no API, no driver introspection
and no cooperation from Windows. It cannot produce a false "isolated"
verdict, because the only way to be marked isolated is to go a long time
with no audio during pauses. And it runs for free, forever, in the
background — so the answer is not "it was set up correctly once", it is
"there has been no contamination in the last 40 minutes of listening".
"""
from __future__ import annotations

import time
from typing import NamedTuple

# How long after Spotify reports a pause before silence is expected.
#
# Three separate lags stack up here and all of them are real: the track
# source is polled every 0.25s so `is_playing` can be that stale; Spotify
# fades out rather than cutting; and the analyser's own squelch holds its
# gate open for 1.2s so that gaps between beats don't close it. Counting
# contamination inside that window would flag every single pause.
PAUSE_GRACE_S = 2.5

# Contamination has to persist this long to count. A single frame above the
# gate during a pause is far more likely to be the tail of the track than
# another application starting up.
MIN_CONTAMINATION_S = 0.4

# How much paused-time evidence is needed before claiming isolation. Below
# this the honest answer is "don't know yet" — which is why the verdict is a
# three-way and not a boolean.
MIN_EVIDENCE_S = 20.0

# Evidence older than this stops counting, so a setting that silently reset
# an hour ago shows up as a fresh problem rather than being outvoted by a
# long clean history.
EVIDENCE_WINDOW_S = 3600.0


class IsolationVerdict(NamedTuple):
    # "isolated"     — proven: audio during play, none during pauses
    # "contaminated" — audio arrived while Spotify was paused
    # "silent"       — nothing arrives even while Spotify is PLAYING
    # "unknown"      — not enough evidence yet
    #
    # "silent" exists because leaving it out made the monitor actively
    # misleading. If a capture device is selected but Spotify was never
    # routed to it, no audio ever arrives — so no contamination is ever
    # observed either, and the monitor would happily certify the setup as
    # isolated while the strip was reacting to nothing at all. A claim that
    # is technically true and practically the opposite of useful. Silence
    # during playback is its own failure and gets its own verdict.
    state: str
    detail: str         # one line, for the dashboard
    paused_seconds: float       # how much paused-time evidence there is
    contaminated_seconds: float
    last_contamination_at: float | None   # wall clock, or None
    events: int         # distinct contamination episodes in the window
    playing_seconds: float = 0.0
    audible_playing_seconds: float = 0.0


class IsolationMonitor:
    """Watches for audio that arrives while Spotify says it is paused.

    Fed one observation per engine frame. Holds no reference to the engine,
    the listener or the track source — just the two facts it needs — so it
    can be tested by handing it a scripted sequence.
    """

    def __init__(self, now: float | None = None):
        self._paused_since: float | None = None
        self._contam_started: float | None = None
        # (timestamp, seconds) per contamination episode.
        self._episodes: list[tuple[float, float]] = []
        self._paused_seconds = 0.0
        # (timestamp, seconds) so paused evidence also ages out of the window.
        self._paused_chunks: list[tuple[float, float]] = []
        self._last_observation: float | None = None
        # Playing-time evidence, for the "silent" verdict. Without this the
        # monitor cannot tell a correctly isolated capture from a capture
        # that is receiving nothing whatsoever, because both look like
        # "no contamination during pauses".
        self._playing_chunks: list[tuple[float, float]] = []
        self._audible_playing_chunks: list[tuple[float, float]] = []

    def observe(self, now: float, spotify_playing: bool, audio_live: bool):
        """One engine frame's worth of evidence."""
        last = self._last_observation
        self._last_observation = now
        dt = 0.0 if last is None else max(0.0, min(1.0, now - last))

        if spotify_playing:
            # Playing: no inference about CONTAMINATION is available here.
            # Audio arriving now could be Spotify, or Spotify plus something
            # else, and this module deliberately does not guess between them.
            #
            # It is still worth recording, for a different question: is any
            # audio reaching us at all? A capture device with nothing routed
            # to it is silent during playback too, and that has to be
            # distinguishable from a clean one.
            self._paused_since = None
            self._close_episode(now)
            if dt > 0.0:
                self._playing_chunks.append((now, dt))
                if audio_live:
                    self._audible_playing_chunks.append((now, dt))
                self._prune(now)
            return

        if self._paused_since is None:
            self._paused_since = now
            return
        if now - self._paused_since < PAUSE_GRACE_S:
            return

        # Past the grace period, with Spotify paused. Anything audible now is
        # something else.
        self._paused_seconds += dt
        self._paused_chunks.append((now, dt))

        if audio_live:
            if self._contam_started is None:
                self._contam_started = now
        else:
            self._close_episode(now)

        self._prune(now)

    def _close_episode(self, now: float):
        if self._contam_started is None:
            return
        length = now - self._contam_started
        self._contam_started = None
        if length >= MIN_CONTAMINATION_S:
            self._episodes.append((now, length))

    def _prune(self, now: float):
        cutoff = now - EVIDENCE_WINDOW_S
        self._episodes = [e for e in self._episodes if e[0] >= cutoff]
        dropped = [c for c in self._paused_chunks if c[0] < cutoff]
        if dropped:
            self._paused_seconds -= sum(c[1] for c in dropped)
            self._paused_seconds = max(0.0, self._paused_seconds)
            self._paused_chunks = [c for c in self._paused_chunks if c[0] >= cutoff]
        self._playing_chunks = [c for c in self._playing_chunks if c[0] >= cutoff]
        self._audible_playing_chunks = [
            c for c in self._audible_playing_chunks if c[0] >= cutoff
        ]

    def verdict(self, now: float | None = None) -> IsolationVerdict:
        now = time.time() if now is None else now
        self._prune(now)
        contaminated = sum(e[1] for e in self._episodes)
        # An episode still in progress counts immediately once it is long
        # enough — a problem should be reported while it is happening, not
        # only once it stops.
        if self._contam_started is not None:
            live = now - self._contam_started
            if live >= MIN_CONTAMINATION_S:
                contaminated += live

        last_at = self._episodes[-1][0] if self._episodes else None
        if self._contam_started is not None and now - self._contam_started >= MIN_CONTAMINATION_S:
            last_at = now

        events = len(self._episodes) + (
            1 if self._contam_started is not None
            and now - self._contam_started >= MIN_CONTAMINATION_S else 0
        )

        playing = sum(c[1] for c in self._playing_chunks)
        audible = sum(c[1] for c in self._audible_playing_chunks)

        if contaminated > 0.0:
            mins = (now - last_at) / 60.0 if last_at else 0.0
            when = "right now" if mins < 0.2 else f"{mins:.0f} min ago"
            return IsolationVerdict(
                "contaminated",
                f"Audio from something other than Spotify reached the strip "
                f"({events} time{'s' if events != 1 else ''}, most recently "
                f"{when}).",
                self._paused_seconds, contaminated, last_at, events,
                playing, audible,
            )

        # NOTHING ARRIVING AT ALL, checked BEFORE isolation is granted.
        # Order matters here: a device with nothing routed to it produces no
        # contamination either, so if isolation were checked first it would
        # certify a capture that is receiving silence.
        if playing >= MIN_EVIDENCE_S and audible < 0.05 * playing:
            return IsolationVerdict(
                "silent",
                f"No audio at all across {playing / 60.0:.0f} min of "
                f"playback — the capture device is selected but nothing is "
                f"routed to it, so the lights are reacting to nothing.",
                self._paused_seconds, 0.0, None, 0, playing, audible,
            )

        if self._paused_seconds < MIN_EVIDENCE_S:
            return IsolationVerdict(
                "unknown",
                f"Not enough evidence yet — needs about "
                f"{MIN_EVIDENCE_S - self._paused_seconds:.0f} more seconds "
                f"with Spotify paused.",
                self._paused_seconds, 0.0, None, 0, playing, audible,
            )
        return IsolationVerdict(
            "isolated",
            f"Verified: nothing but Spotify has reached the strip across "
            f"{self._paused_seconds / 60.0:.0f} min of paused time.",
            self._paused_seconds, 0.0, None, 0, playing, audible,
        )


# ---- Setup state ------------------------------------------------------------
#
# What the UI needs to know to tell someone the ONE next thing to do. The
# steps deliberately return what is true rather than what was configured:
# "config names a cable" and "we are actually capturing only Spotify" are
# different claims, and conflating them is how a setup silently rots.

# Substrings that identify a virtual cable endpoint. Matching on the vendor
# name rather than an exact device name because Windows device names carry
# machine-specific suffixes.
CABLE_HINTS = ("cable", "vb-audio", "voicemeeter", "virtual audio")


def looks_like_cable(device_name: str) -> bool:
    n = (device_name or "").lower()
    return any(h in n for h in CABLE_HINTS)


class DeviceObservation(NamedTuple):
    """What was heard on one candidate device during a guided two-step probe:
    a few seconds with Spotify paused, then a few with it playing."""
    device: str
    paused_seconds: float
    paused_audible_seconds: float
    playing_seconds: float
    playing_audible_seconds: float


# A cable is "already carrying Spotify" or "idle" only if it is properly
# silent when it should be. Left at zero this would trip on a single frame
# of noise; 10% of the probe window is loose enough to tolerate a stray
# buffer and tight enough that real audio cannot hide under it.
QUIET_FRACTION = 0.10
# ...and audible means audible for a decent share of the window, not once.
AUDIBLE_FRACTION = 0.40
# Minimum probe length per state before any claim is made.
MIN_PROBE_S = 3.0


def classify_device(obs: DeviceObservation) -> tuple[str, str]:
    """What role is this device already playing? Returns (role, detail).

    THIS IS THE QUESTION "IS A CABLE INSTALLED?" CANNOT ANSWER, and getting
    it wrong is worse than not asking. Virtual cables are common — people
    install them for Discord, OBS, mic routing, VoiceMeeter setups. A cable
    that already exists may already be carrying something, and routing
    Spotify into it would do two bad things at once: contaminate the capture
    we are trying to isolate, and silently inject Spotify into whatever the
    user built that cable for. Detecting presence and assuming availability
    is how a setup helper breaks someone's working audio.

    Roles:
      "carrying-spotify" — silent while paused, audible while playing. This
                           is the finished state: Spotify is already routed
                           here. Nothing to install, nothing to change.
      "busy"             — audible while Spotify is PAUSED, so something
                           else is using it. Do not touch it.
      "idle"             — silent in both states. Nothing is routed here;
                           free to use.
      "unknown"          — the probe was too short to say.
    """
    if obs.paused_seconds < MIN_PROBE_S or obs.playing_seconds < MIN_PROBE_S:
        return "unknown", "Probe too short to classify this device."

    paused_audible = obs.paused_audible_seconds / obs.paused_seconds
    playing_audible = obs.playing_audible_seconds / obs.playing_seconds

    if paused_audible > QUIET_FRACTION:
        return "busy", (
            f"Something else is already using {obs.device} — it carried audio "
            f"while Spotify was paused. Routing Spotify here would mix the two "
            f"and break whatever this cable was set up for."
        )
    if playing_audible >= AUDIBLE_FRACTION:
        return "carrying-spotify", (
            f"Spotify is already routed to {obs.device}, and nothing else is. "
            f"No install needed."
        )
    return "idle", (
        f"{obs.device} is installed and nothing is routed to it — free to use."
    )


def choose_capture_device(observations: list[DeviceObservation]
                          ) -> tuple[str | None, str, str]:
    """Pick the device to capture. Returns (device or None, outcome, detail).

    Outcomes: "already-set-up", "use-idle", "all-busy", "none", "unknown".
    """
    if not observations:
        return None, "none", (
            "No virtual audio cable found. One needs installing — it is the "
            "only step that requires a reboot."
        )

    classified = [(d, *classify_device(d)) for d in observations]

    for obs, role, detail in classified:
        if role == "carrying-spotify":
            return obs.device, "already-set-up", detail
    for obs, role, detail in classified:
        if role == "idle":
            return obs.device, "use-idle", detail

    if all(role == "busy" for _, role, _ in classified):
        names = ", ".join(o.device for o, _, _ in classified)
        return None, "all-busy", (
            f"Every cable on this PC is already in use ({names}). Rather than "
            f"take one over, install a second one — VB-CABLE ships A and B "
            f"variants for exactly this, and VoiceMeeter Banana has three "
            f"virtual inputs."
        )
    return None, "unknown", "Not enough evidence yet — probe for longer."


def find_cable_devices() -> list[str]:
    """Virtual cable endpoints currently present, by name. Empty if none —
    which is the signal that a driver install is the next step."""
    try:
        import pyaudiowpatch as pyaudio
    except Exception:
        return []
    out = []
    pa = pyaudio.PyAudio()
    try:
        for d in pa.get_loopback_device_info_generator():
            if looks_like_cable(str(d["name"])):
                out.append(str(d["name"]))
    except Exception:
        pass
    finally:
        pa.terminate()
    return out


def setup_steps(cfg: dict, listener_device: str | None,
                verdict: IsolationVerdict | None) -> list[dict]:
    """The setup checklist, each step reporting its own real state.

    Returns dicts with: key, title, state ("done"|"todo"|"blocked"|"unknown"),
    detail, and optionally `action` — a hint for what the UI's button does.
    """
    steps: list[dict] = []

    try:
        import numpy  # noqa: F401
        import pyaudiowpatch  # noqa: F401
        deps_ok = True
        deps_detail = "numpy and PyAudioWPatch are installed."
    except Exception as e:
        deps_ok = False
        deps_detail = f"Missing: {e}. Run: pip install numpy PyAudioWPatch"
    steps.append({
        "key": "deps", "title": "Audio libraries",
        "state": "done" if deps_ok else "todo", "detail": deps_detail,
        "action": None if deps_ok else "install_deps",
    })
    if not deps_ok:
        return steps

    cables = find_cable_devices()
    steps.append({
        "key": "cable", "title": "Virtual audio cable",
        "state": "done" if cables else "todo",
        "detail": (f"Found: {cables[0]}" if cables else
                   "Not installed. This is the only step that needs a reboot — "
                   "everything after it is instant."),
        "action": None if cables else "open_vbcable_download",
    })
    if not cables:
        # Without a cable, the honest thing to report is that the app is
        # capturing everything — not to leave the remaining steps looking
        # merely incomplete.
        steps.append({
            "key": "route", "title": "Send Spotify to the cable",
            "state": "blocked", "detail": "Needs the cable first.",
            "action": None,
        })
        steps.append({
            "key": "verified", "title": "Spotify-only, verified",
            "state": "blocked",
            "detail": "Currently capturing the whole system mix, so "
                      "notifications and other apps reach the strip.",
            "action": None,
        })
        return steps

    configured = (cfg.get("audio_capture_device") or "").strip()
    capturing_cable = bool(listener_device and looks_like_cable(listener_device))
    steps.append({
        "key": "route", "title": "Send Spotify to the cable",
        "state": "done" if capturing_cable else "todo",
        "detail": (
            f"Capturing {listener_device}." if capturing_cable else
            "In Windows: Settings > System > Sound > App volume and device "
            "preferences, set Spotify's Output to the cable. Spotify has to "
            "be playing for it to appear in that list."
            + ("" if configured else
               f" Then set audio_capture_device to \"{cables[0].split(' (')[0]}\".")
        ),
        "action": None if capturing_cable else "open_sound_settings",
    })

    if verdict is None or verdict.state == "unknown":
        steps.append({
            "key": "verified", "title": "Spotify-only, verified",
            "state": "unknown",
            "detail": (verdict.detail if verdict else
                       "Waiting for the engine to gather evidence."),
            "action": "run_verification",
        })
    else:
        steps.append({
            "key": "verified", "title": "Spotify-only, verified",
            "state": "done" if verdict.state == "isolated" else "todo",
            "detail": verdict.detail,
            "action": None if verdict.state == "isolated" else "run_verification",
        })
    return steps


VBCABLE_URL = "https://vb-audio.com/Cable/"
