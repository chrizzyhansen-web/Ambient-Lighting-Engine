"""
The actual polling loop, refactored to run as a background thread that the
web dashboard can start/stop and observe via a shared status dict.

Lighting decisions run on a separate cadence from the Govee-updating loop:
a background worker computes the next decision (via the AI pipeline, which
can take several seconds) while the main loop keeps the currently-active
 decision displayed while background plan generation runs rather than blocking
 on every AI call.
"""
import concurrent.futures
import colorsys
import random
import threading
import time
import traceback

from spotify_client import SpotifyClient
from spicetify_bridge import SpicetifyBridgeSource
import spicetify_bridge
from govee_client import GoveeClient, MAX_BRIGHTNESS_PCT
from ai_lighting_engine import GroqLightingEngine
import effects

status = {
    "running": False,
    "last_track": None,
    "last_decision": None,
    "last_error": None,
    "sleep_timer_ends_at": None,  # epoch seconds, or None if not active
    "loading": False,
    "loading_track_id": None,  # which track_id we're currently holding paused, if any
    "govee_transport": None,
    "govee_lan_available": False,
    "govee_independent_control": False,
    "govee_segment_count": 0,
}

_runtime_settings_lock = threading.RLock()
_runtime_settings = {"max_brightness_pct": MAX_BRIGHTNESS_PCT, "flicker_enabled": True}
_active_govee_client = None


def update_runtime_settings(max_brightness_pct: int, flicker_enabled: bool):
    """Apply the dashboard's brightness and flicker preferences immediately."""
    global _active_govee_client
    brightness = max(1, min(100, int(max_brightness_pct)))
    with _runtime_settings_lock:
        _runtime_settings["max_brightness_pct"] = brightness
        _runtime_settings["flicker_enabled"] = bool(flicker_enabled)
        active = _active_govee_client
    if active is not None:
        active.set_max_brightness_pct(brightness)


def _get_runtime_settings() -> dict:
    with _runtime_settings_lock:
        return dict(_runtime_settings)

_thread: threading.Thread | None = None
_stop_event = threading.Event()
# 2 workers, not 1: if a single Groq call ever hangs, a second
# worker means the NEXT track's decision isn't permanently stuck queued
# behind it. Recreated fresh on every start() so a stuck call from a
# previous session can't contaminate a new one either.
_pipeline_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# How long to wait between checks when there's nothing to react to yet.
# Reading Spicetify's bridge state is a free in-memory lookup (no network
# call), so there's no reason to wait a full poll_interval (which exists to
# rate-limit the *paid Spotify Web API* path) — doing so was the reason
# pause-on-new-track landed inconsistently: the loop could be mid-sleep for
# up to 4 seconds when a song actually started.
_SPICETIFY_IDLE_INTERVAL = 0.3


def _make_lighting_engine(cfg: dict, segment_count: int):
    return GroqLightingEngine(cfg["groq_api_key"], segment_count=segment_count)


def _needs_palette_transition(previous: tuple | None, current: tuple) -> bool:
    """Detect song changes and large RGB jumps that deserve a white handoff."""
    if previous is None or previous[0] != current[0]:
        return True
    old_colors = [previous[1], *previous[2]]
    new_colors = [current[1], *current[2]]
    if len(old_colors) != len(new_colors):
        return True
    return any(
        max(abs(((old >> shift) & 0xFF) - ((new >> shift) & 0xFF)) for shift in (16, 8, 0)) >= 72
        for old, new in zip(old_colors, new_colors)
    )


def _effect_profile(decision, anchor_hues: list[float] | None = None, hue_spread_deg: float = 52.0) -> dict:
    """Adapt the current RGB plan into the preserved hue-based effect inputs."""
    hue, saturation, _ = colorsys.rgb_to_hsv(
        (decision.rgb >> 16 & 0xFF) / 255.0,
        (decision.rgb >> 8 & 0xFF) / 255.0,
        (decision.rgb & 0xFF) / 255.0,
    )
    anchors = list(anchor_hues or [])
    for segment in sorted(decision.segments, key=lambda item: min(item.index, default=0)) if not anchors else []:
        red, green, blue = (
            (segment.rgb >> 16 & 0xFF) / 255.0,
            (segment.rgb >> 8 & 0xFF) / 255.0,
            (segment.rgb & 0xFF) / 255.0,
        )
        anchor_hue, anchor_saturation, _ = colorsys.rgb_to_hsv(red, green, blue)
        if anchor_saturation >= 0.12:
            degrees = anchor_hue * 360.0
            if not anchors or all(
                min(abs(degrees - old) % 360, 360 - abs(degrees - old) % 360) >= 12
                for old in anchors
            ):
                anchors.append(degrees)
    return {
        "hue_deg": hue * 360.0,
        "palette_width_deg": hue_spread_deg,
        "saturation_pct": max(62.0, saturation * 100.0),
        "energy_pct": float(decision.brightness_pct),
        "anchor_hues": anchors[:4] or None,
    }


def _cloud_palette_groups(frame: list[int], max_groups: int = 3) -> list[dict]:
    """Compress a rendered frame into a few contiguous cloud API updates."""
    if not frame:
        return []
    group_count = min(max_groups, len(frame))
    groups = []
    for group in range(group_count):
        start = group * len(frame) // group_count
        end = (group + 1) * len(frame) // group_count
        colors = frame[start:end]
        channels = [
            round(sum((rgb >> shift) & 0xFF for rgb in colors) / len(colors))
            for shift in (16, 8, 0)
        ]
        groups.append({
            "index": list(range(start, end)),
            "rgb": channels[0] << 16 | channels[1] << 8 | channels[2],
        })
    return groups


def _loop(cfg: dict):
    global _active_govee_client
    if cfg.get("track_source") == "spicetify":
        track_source = SpicetifyBridgeSource()
    else:
        track_source = SpotifyClient(
            cfg["spotify_client_id"], cfg["spotify_client_secret"], cfg["spotify_refresh_token"]
        )
    initial_brightness = max(1, min(100, int(cfg.get("max_brightness_pct", MAX_BRIGHTNESS_PCT))))
    flicker_default = bool(cfg.get("flicker_enabled", True))
    with _runtime_settings_lock:
        _runtime_settings.update({
            "max_brightness_pct": initial_brightness,
            "flicker_enabled": flicker_default,
        })
    govee = GoveeClient(
        cfg["govee_api_key"], cfg["govee_device_id"], cfg["govee_device_model"],
        max_brightness_pct=initial_brightness,
    )
    with _runtime_settings_lock:
        _active_govee_client = govee
    lighting = _make_lighting_engine(cfg, govee.segment_count)

    def _sync_govee_status():
        status["govee_transport"] = govee.transport
        status["govee_lan_available"] = govee.lan_available
        status["govee_independent_control"] = govee.supports_independent_control
        status["govee_segment_count"] = govee.segment_count

    sleep_minutes = float(cfg.get("sleep_timer_minutes") or 0)
    timer_ends_at = time.time() + sleep_minutes * 60 if sleep_minutes > 0 else None
    status["sleep_timer_ends_at"] = timer_ends_at

    poll_interval = float(cfg.get("poll_interval_seconds") or 4)

    govee.set_power(True)
    govee.set_brightness(govee.max_brightness_pct)
    _sync_govee_status()
    effect_segment_count = govee.segment_count if govee.supports_independent_control else 1
    effect_renderer = effects.EffectEngine(effect_segment_count)
    animation_started_at = time.monotonic()
    next_track_poll = 0.0
    cached_track = None
    last_target_signature = None
    last_cloud_signature = None
    current_profile = None
    previous_profile = None
    shift_started_at = None
    effect_events: list[tuple[str, float, float]] = []
    smoothed_frame = None

    current_track_id = None
    current_plan = None
    current_decision = None
    current_frame = None
    pending_future: concurrent.futures.Future | None = None
    pending_future_track_id: str | None = None
    pending_started_at = 0.0
    prefetched_track_ids: set[str] = set()
    next_cloud_update_at = 0.0
    last_cloud_flicker_start = 0.0
    cloud_flicker_restore_at = 0.0
    cloud_flicker_segment: int | None = None
    cloud_flicker_restore_rgb: int | None = None
    flicker_events: list[tuple[str, float, float]] = []
    next_flicker_at = random.uniform(3.0, 7.0)
    next_lan_frame_at = animation_started_at
    pause_water_active = False
    pause_water_started_at = 0.0
    next_pause_cloud_update_at = 0.0
    # track_id we ourselves auto-paused while its first decision was loading —
    # so we know to resume it (and not mistake our own pause for the user's).
    we_paused_track_id: str | None = None
    is_spicetify = cfg.get("track_source") == "spicetify"

    def _apply_decision(decision):
        govee.set_brightness(govee.max_brightness_pct)
        if decision.segments:
            govee.set_segment_colors([s.model_dump() for s in decision.segments])
        else:
            govee.set_color(decision.rgb)
        _sync_govee_status()

    def _show_amber(reason: str):
        """Use a steady amber state for pipeline failures."""
        nonlocal last_cloud_signature
        govee.set_brightness(govee.max_brightness_pct)
        govee.set_color(0xFF9A00)
        last_cloud_signature = None
        _sync_govee_status()
        status["last_decision"] = {
            "rgb": 0xFF9A00,
            "brightness_pct": govee.max_brightness_pct,
            "loop_delay_seconds": poll_interval,
            "segments": [],
            "reasoning": reason,
        }

    while not _stop_event.is_set():
        try:
            remaining_min = None
            if timer_ends_at is not None:
                remaining_min = (timer_ends_at - time.time()) / 60
                if remaining_min <= 0:
                    govee.set_power(False)
                    _sync_govee_status()
                    status["running"] = False
                    status["last_decision"] = {"note": "Sleep timer expired — lights turned off."}
                    with _runtime_settings_lock:
                        if _active_govee_client is govee:
                            _active_govee_client = None
                    try:
                        govee.close()
                    except Exception:
                        pass
                    return
                if remaining_min > 15:
                    remaining_min = None  # only kicks in for the final 15 minutes

            now = time.monotonic()
            if now >= next_track_poll:
                cached_track = track_source.get_now_playing()
                poll_period = _SPICETIFY_IDLE_INTERVAL if is_spicetify else poll_interval
                next_track_poll = now + poll_period
                status["last_track"] = cached_track
            track = cached_track

            if not track:
                if is_spicetify:
                    spicetify_bridge.set_pending_command(None)
                time.sleep(_SPICETIFY_IDLE_INTERVAL if is_spicetify else poll_interval)
                continue

            # The Spicetify bridge supplies the first item in the Spotify
            # queue. Generate its song-long keyframe plan while this track is
            # still playing so track changes normally need no wait.
            next_track = track.get("next_track") or {}
            next_id = next_track.get("track_id")
            if (next_id and next_id != track.get("track_id")
                    and next_id not in prefetched_track_ids
                    and next_track.get("title")):
                _pipeline_executor.submit(
                    lighting.get_plan,
                    title=next_track.get("title", ""),
                    artist=next_track.get("artist", ""),
                    album_art_url=next_track.get("album_art_url"),
                    track_id=next_id,
                )
                prefetched_track_ids.add(next_id)
                if len(prefetched_track_ids) > 2000:
                    prefetched_track_ids.clear()

            # Normally "not playing" just means skip this tick — except when
            # WE are the one who paused it (holding it still while its first
            # decision loads). In that case keep processing so the pipeline
            # can finish and we can resume it ourselves.
            is_our_paused_track = we_paused_track_id is not None and track["track_id"] == we_paused_track_id
            if not track["is_playing"] and not is_our_paused_track:
                if is_spicetify:
                    spicetify_bridge.set_pending_command(None)
                if not pause_water_active:
                    pause_water_active = True
                    pause_water_started_at = time.monotonic()
                    next_pause_cloud_update_at = 0.0
                    last_cloud_signature = None
                    flicker_events.clear()

                # A slow blue palette continues to drift while paused. It has
                # no transient events, so the neon flicker never runs here.
                pause_profile = {
                    "hue_deg": 207.0,
                    "palette_width_deg": 42.0,
                    "saturation_pct": 82.0,
                    "energy_pct": 22.0,
                    "anchor_hues": [190.0, 207.0, 224.0],
                }
                pause_time = time.monotonic() - pause_water_started_at
                paused_frame = effect_renderer.render(pause_time, **pause_profile, events=[])
                smoothed_frame = effects.smooth_towards(smoothed_frame, paused_frame, alpha=0.28)
                govee.set_brightness(govee.max_brightness_pct)
                if govee.lan_available:
                    if govee.supports_independent_control:
                        govee.set_animated_frame(smoothed_frame, 0x164BFF)
                    else:
                        govee.set_color(smoothed_frame[0])
                    next_lan_frame_at += 1.0 / 30.0
                    frame_now = time.monotonic()
                    if next_lan_frame_at < frame_now:
                        next_lan_frame_at = frame_now
                    pause_wait = max(0.0, next_lan_frame_at - frame_now)
                else:
                    # The cloud API cannot stream frames; gently change the
                    # whole light every few seconds without flicker or bursts.
                    frame_now = time.monotonic()
                    if frame_now >= next_pause_cloud_update_at:
                        govee.set_color(smoothed_frame[0])
                        next_pause_cloud_update_at = time.monotonic() + 2.5
                    pause_wait = min(
                        0.5,
                        max(0.05, next_track_poll - frame_now),
                        max(0.05, next_pause_cloud_update_at - frame_now),
                    )
                _sync_govee_status()
                status["loading"] = False
                status["loading_track_id"] = None
                status["last_decision"] = {
                    "rgb": smoothed_frame[0],
                    "brightness_pct": govee.max_brightness_pct,
                    "loop_delay_seconds": pause_wait,
                    "segments": [],
                    "reasoning": "Playback paused — slow blue water motion; flicker is off.",
                }
                _stop_event.wait(pause_wait)
                continue
            pause_water_active = False

            track_changed = track["track_id"] != current_track_id

            if track_changed:
                current_track_id = track["track_id"]
                current_plan = None
                current_decision = None  # hold the existing lights while the first plan resolves
                status["loading_track_id"] = track["track_id"]
                # Pause playback while we figure out the first decision, so
                # the song doesn't run ahead of the lights during the wait —
                # only if it was genuinely playing when we first saw it (not
                # e.g. a scrub-to-new-track while already paused by the user).
                if track["is_playing"] and is_spicetify and not lighting.has_plan(track["track_id"]):
                    we_paused_track_id = track["track_id"]
                    spicetify_bridge.set_pending_command("pause")
            if track_changed:
                stale_future = pending_future is not None and pending_future_track_id != track["track_id"]
                if pending_future is None or pending_future.done() or (track_changed and stale_future):
                    pending_future = _pipeline_executor.submit(
                        lighting.get_plan,
                        title=track["title"],
                        artist=track["artist"],
                        album_art_url=track["album_art_url"],
                        track_id=track["track_id"],
                    )
                    pending_future_track_id = track["track_id"]
                    pending_started_at = time.time()
            # Sample the old continuous keyframe curve at the current song
            # position; this does not call Groq and follows each section smoothly.
            if current_plan is not None:
                current_frame = lighting.sample_keyframe(current_plan, track["progress_pct"])
                current_decision = lighting.decision_at(current_plan, track["progress_pct"], remaining_min)
                status["last_decision"] = current_decision.model_dump()
                status["last_decision"]["brightness_pct"] = govee.max_brightness_pct

            # Check on any in-flight decision computation.
            if pending_future is not None and pending_future.done():
                # Only adopt the result if it's actually for the track we're
                # currently on — a slow-to-finish call for a track we've
                # since moved past shouldn't overwrite what's playing now.
                result_is_current = pending_future_track_id == track["track_id"]
                try:
                    new_plan = pending_future.result()
                    if result_is_current:
                        current_plan = new_plan
                        current_frame = lighting.sample_keyframe(new_plan, track["progress_pct"])
                        current_decision = lighting.decision_at(new_plan, track["progress_pct"], remaining_min)
                        status["last_decision"] = current_decision.model_dump()
                        status["last_decision"]["brightness_pct"] = govee.max_brightness_pct
                        status["last_error"] = None
                except Exception as e:
                    if result_is_current:
                        status["last_error"] = f"{e}\n{traceback.format_exc(limit=2)}"
                        if current_decision is None:
                            current_decision = _fallback_decision(remaining_min)
                            current_frame = None
                            status["last_decision"] = current_decision.model_dump()
                            status["last_decision"]["brightness_pct"] = govee.max_brightness_pct
                if result_is_current:
                    pending_future = None
                    pending_future_track_id = None

                # We now have a decision (real or fallback) for whichever
                # track we'd paused for — safe to let it play again.
                if is_our_paused_track and current_decision is not None:
                    spicetify_bridge.set_pending_command("resume")
                    we_paused_track_id = None

            if current_decision is None:
                # Keep the previous light state while the next plan resolves;
                # do not flash a loading pattern across the user's strip.
                status["loading"] = True
                now = time.time()
                # Give up waiting after the AI call's time budget plus slack.
                if time.time() - pending_started_at > 25:
                    current_decision = _fallback_decision(remaining_min)
                    current_frame = None
                    status["last_decision"] = current_decision.model_dump()
                    status["last_decision"]["brightness_pct"] = govee.max_brightness_pct
                    status["last_error"] = "Pipeline never returned — using fallback colors."
                    if is_our_paused_track:
                        spicetify_bridge.set_pending_command("resume")
                        we_paused_track_id = None
                time.sleep(0.25)
                if current_decision is not None:
                    status["loading"] = False
                    status["loading_track_id"] = None
                continue
            else:
                status["loading"] = False
                status["loading_track_id"] = None
                if is_spicetify and not is_our_paused_track:
                    spicetify_bridge.set_pending_command(None)
                target_signature = (
                    track["track_id"], current_decision.rgb,
                    tuple(segment.rgb for segment in current_decision.segments),
                    current_decision.brightness_pct,
                    current_frame.hue_spread_deg if current_frame else 52.0,
                    current_frame.saturation_pct if current_frame else 80,
                )
                palette_transition_requested = False
                if target_signature != last_target_signature:
                    palette_transition_requested = _needs_palette_transition(
                        last_target_signature, target_signature
                    )
                    next_profile = _effect_profile(
                        current_decision,
                        current_plan.anchor_hues if current_plan else None,
                        current_frame.hue_spread_deg if current_frame else 52.0,
                    )
                    if palette_transition_requested:
                        previous_profile = current_profile
                        shift_started_at = time.monotonic() - animation_started_at
                        effect_events.append(("white_sweep", shift_started_at, 1.0))
                    current_profile = next_profile
                    last_target_signature = target_signature

                effect_time = time.monotonic() - animation_started_at
                effect_events = [
                    event for event in effect_events
                    if event[0] != "flicker" and effect_time - event[1] <= effects.SWEEP_DURATION
                ]
                energy = float(current_profile["energy_pct"]) / 100.0
                flicker_enabled = _get_runtime_settings()["flicker_enabled"]
                if flicker_enabled and effect_time >= next_flicker_at:
                    flicker_events.append(("flicker", effect_time, random.uniform(0.8, 2.4)))
                    # Shorter gaps as the music gets more energetic, while
                    # keeping the little neon failures occasional at rest.
                    interval_low = max(4.0, 8.0 - 4.0 * energy)
                    interval_high = max(interval_low + 1.5, 14.0 - 7.0 * energy)
                    next_flicker_at = effect_time + random.uniform(interval_low, interval_high)
                elif not flicker_enabled:
                    flicker_events.clear()
                    next_flicker_at = effect_time + random.uniform(8.0, 14.0)
                effect_events.extend(flicker_events)
                flicker_events = [event for event in flicker_events if effect_time - event[1] <= event[2]]
                if govee.lan_available:
                    rendered_frame = effect_renderer.render(
                        effect_time,
                        **current_profile,
                        events=effect_events,
                        previous_hue_deg=(previous_profile["hue_deg"] if previous_profile else None),
                        previous_anchor_hues=(previous_profile["anchor_hues"] if previous_profile else None),
                        shift_started_at=shift_started_at,
                    )
                    smoothed_frame = effects.smooth_towards(smoothed_frame, rendered_frame)
                    govee.set_brightness(govee.max_brightness_pct)
                    if govee.supports_independent_control:
                        govee.set_animated_frame(smoothed_frame, current_decision.rgb)
                    else:
                        govee.set_color(smoothed_frame[0])
                    _sync_govee_status()
                    # Pace LAN frames at 30 FPS. The LAN sender has the same
                    # 33 ms floor, but relying on that alone makes unchanged
                    # frames spin; the old extra 50 ms sleep capped motion
                    # below the transport's frame rate.
                    next_lan_frame_at += 1.0 / 30.0
                    now = time.monotonic()
                    if next_lan_frame_at < now:
                        next_lan_frame_at = now
                    sleep_for = max(0.0, next_lan_frame_at - now)
                else:
                    # Cloud control is deliberately event-driven to respect
                    # Govee's API limits. Preserve advertised independent zones
                    # at each AI-selected keyframe; otherwise use solid color.
                    palette_update_due = (
                        target_signature != last_cloud_signature
                        and (time.monotonic() >= next_cloud_update_at or palette_transition_requested)
                    )
                    if palette_update_due:
                        # A fresh base frame replaces any temporary cloud flicker.
                        cloud_flicker_segment = None
                        cloud_flicker_restore_rgb = None
                    if palette_update_due:
                        if palette_transition_requested:
                            # Cloud cannot stream a spatial frame. A single
                            # white cue followed by the target palette is its
                            # rate-limit-safe transition fallback.
                            govee.set_color(0xFFFFFF)
                        if govee.supports_independent_control:
                            govee.set_brightness(govee.max_brightness_pct)
                            rendered_frame = effect_renderer.render(
                                effect_time,
                                **current_profile,
                                events=[],
                            )
                            govee.set_segment_colors(_cloud_palette_groups(rendered_frame))
                        else:
                            govee.set_brightness(govee.max_brightness_pct)
                            govee.set_color(current_decision.rgb)
                        last_cloud_signature = target_signature
                        next_cloud_update_at = time.monotonic() + 3.0

                    if (cloud_flicker_restore_rgb is not None and
                            effect_time >= cloud_flicker_restore_at and not palette_update_due):
                        if cloud_flicker_segment is not None:
                            govee.set_segment_colors([{
                                "index": [cloud_flicker_segment],
                                "rgb": cloud_flicker_restore_rgb,
                            }])
                        else:
                            govee.set_color(cloud_flicker_restore_rgb)
                        cloud_flicker_segment = None
                        cloud_flicker_restore_rgb = None

                    fresh_flicker = next((
                        event for event in reversed(flicker_events)
                        if event[1] > last_cloud_flicker_start
                    ), None)
                    if fresh_flicker is not None and not palette_update_due:
                        base_frame = effect_renderer.render(
                            effect_time, **current_profile, events=[],
                        )
                        flash_frame = effect_renderer.render(
                            effect_time, **current_profile, events=[fresh_flicker],
                        )
                        changed_segments = [
                            index for index, (base, flash) in enumerate(zip(base_frame, flash_frame))
                            if base != flash
                        ]
                        selected = changed_segments[0] if changed_segments else random.randrange(len(base_frame))
                        base_rgb = base_frame[selected]
                        flash_rgb = flash_frame[selected]
                        if base_rgb == flash_rgb:
                            channels = [((base_rgb >> shift) & 0xFF) for shift in (16, 8, 0)]
                            channels = [round(channel + (255 - channel) * 0.08) for channel in channels]
                            flash_rgb = (channels[0] << 16) | (channels[1] << 8) | channels[2]
                        if govee.supports_independent_control:
                            govee.set_segment_colors([{"index": [selected], "rgb": flash_rgb}])
                            cloud_flicker_segment = selected
                        else:
                            govee.set_color(flash_rgb)
                            cloud_flicker_segment = None
                        cloud_flicker_restore_rgb = base_rgb
                        cloud_flicker_restore_at = fresh_flicker[1] + fresh_flicker[2]
                        last_cloud_flicker_start = fresh_flicker[1]
                    _sync_govee_status()
                    sleep_for = max(0.05, min(next_track_poll - time.monotonic(), 1.0))
        except Exception as e:  # keep the loop alive on transient errors
            status["last_error"] = f"{e}\n{traceback.format_exc(limit=2)}"
            try:
                _show_amber("A playback or lighting error occurred — showing steady amber.")
            except Exception:
                pass
            sleep_for = poll_interval

        _stop_event.wait(sleep_for)

    status["running"] = False
    try:
        govee.close()
    except Exception:
        pass
    with _runtime_settings_lock:
        if _active_govee_client is govee:
            _active_govee_client = None


def _fallback_decision(sleep_timer_minutes_remaining):
    """A safe, non-white default used only if the AI pipeline can't produce
    a real answer in time — better than lights sitting off or stuck loading
    forever."""
    from ai_lighting_engine import LightingDecision
    if sleep_timer_minutes_remaining is not None:
        return LightingDecision(
            rgb=0xFF9A00, brightness_pct=20, loop_delay_seconds=30.0, segments=[],
            reasoning="Fallback: pipeline unavailable during sleep timer; showing dim amber.",
        )
    return LightingDecision(
        rgb=0xFF9A00, brightness_pct=35, loop_delay_seconds=4.0, segments=[],
        reasoning="Fallback: AI pipeline timed out or errored; showing amber.",
    )


def start(cfg: dict):
    global _thread, _pipeline_executor
    if status["running"]:
        return
    _stop_event.clear()
    status["running"] = True
    status["last_error"] = None
    status["loading"] = False
    # Fresh executor each run: if the previous session left a genuinely
    # stuck call behind (see the timeout note above), it stays isolated in
    # the old executor rather than occupying a worker in this one.
    _pipeline_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    _thread = threading.Thread(target=_loop, args=(cfg,), daemon=True)
    _thread.start()


def stop():
    _stop_event.set()
    status["running"] = False


def get_status():
    out = dict(status)
    if out["sleep_timer_ends_at"]:
        out["sleep_timer_minutes_remaining"] = max(
            0, round((out["sleep_timer_ends_at"] - time.time()) / 60, 1)
        )
    else:
        out["sleep_timer_minutes_remaining"] = None
    return out
