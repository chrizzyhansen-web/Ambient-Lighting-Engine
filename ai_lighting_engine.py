"""Groq-based album art and track analysis for Spotify-synced lighting.

GroqLightingEngine creates one structured, cached lighting plan per Spotify song.
"""
import base64
import colorsys
import gzip
from io import BytesIO
import json
import math
import os
import random
import threading
import time
from collections import deque
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError


PLAN_CACHE_PATH = Path(__file__).with_name("song_lighting_cache.json.gz")
RECOVERED_PLAN_CACHE_PATH = Path(__file__).with_name("track_plans.json.gz")
UNCOMPRESSED_OLD_PLAN_CACHE_PATH = Path(__file__).with_name("track_plans.json")
_PLAN_CACHE_LOCK = threading.RLock()
_GROQ_REQUEST_LOCK = threading.RLock()
_PLAN_CACHE: dict[str, dict] | None = None
_REFERENCE_PLANS: dict[str, dict] | None = None
_GROQ_REQUEST_TIMES: deque[float] = deque()
_LOCAL_REQUESTS_PER_MINUTE = 2


class SegmentColor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: list[int] = Field(description="Zero-based segment indices this color applies to")
    rgb: int = Field(ge=0, le=0xFFFFFF, description="24-bit RGB integer, e.g. 16711680 for pure red")


class LightingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rgb: int = Field(ge=0, le=0xFFFFFF, description="Primary 24-bit RGB integer for the whole strip")
    brightness_pct: int = Field(ge=0, le=100, description="0-100 brightness")
    loop_delay_seconds: float = Field(
        ge=0.5,
        le=60,
        description="How long to wait before the next lighting update, in seconds"
    )
    segments: list[SegmentColor] = Field(
        description="Per-segment colors for RGBIC strips. Return an empty list [] "
        "if not using segments — this field is required, never omit it.",
    )
    reasoning: str = Field(description="One short sentence on why these values were chosen")


class LightingKeyframe(BaseModel):
    """One point on the original continuous per-song lighting curve."""

    model_config = ConfigDict(extra="forbid")

    at_pct: float = Field(ge=0, le=100)
    hue_deg: float = Field(ge=0, le=360)
    hue_spread_deg: float = Field(ge=0, le=180)
    saturation_pct: int = Field(ge=0, le=100)
    brightness_pct: int = Field(ge=0, le=100)


class TrackLightingPlan(BaseModel):
    """The original full-song format: timed keyframes plus palette anchors."""

    model_config = ConfigDict(extra="forbid")

    # Three-point plans from the old cache remain valid; newly generated plans
    # are asked for six keyframes to preserve the richer original structure.
    keyframes: list[LightingKeyframe] = Field(min_length=3, max_length=8)
    anchor_hues: list[float] = Field(max_length=4)
    reasoning: str


def _load_plan_cache() -> dict[str, dict]:
    global _PLAN_CACHE
    with _PLAN_CACHE_LOCK:
        if _PLAN_CACHE is None:
            raw = None
            if PLAN_CACHE_PATH.exists():
                try:
                    raw = json.loads(gzip.decompress(PLAN_CACHE_PATH.read_bytes()).decode("utf-8"))
                except (OSError, ValueError, EOFError, gzip.BadGzipFile):
                    raw = None
            if not isinstance(raw, dict) or raw.get("v") != 2:
                # Reference examples are deliberately separate from personal
                # saved songs and must never be imported into this cache.
                raw = {"v": 2, "tracks": {}}
            tracks = raw.get("tracks", {}) if isinstance(raw, dict) else {}
            _PLAN_CACHE = tracks if isinstance(tracks, dict) else {}
            _write_plan_cache_unlocked()
        return _PLAN_CACHE


def _old_verbose_entry(entry: dict) -> dict:
    keyframes = entry.get("keyframes", [])
    return {
        "k": [[frame.get("at_pct", 0), frame.get("hue_deg", 0),
               frame.get("hue_spread_deg", 30), frame.get("saturation_pct", 75),
               frame.get("brightness_pct", 35)] for frame in keyframes if isinstance(frame, dict)],
        "a": entry.get("anchor_hues", [])[:4],
        "r": entry.get("reasoning", ""),
        "t": entry.get("title", ""),
        "n": entry.get("artist", ""),
        "s": int(entry.get("saved_at", 0) or 0),
    }


def _cache_entry_plan(entry: dict) -> dict:
    """Expand the compact old-format cache entry into a validated plan."""
    frames = []
    for frame in entry.get("k", []):
        if isinstance(frame, dict):
            frames.append(frame)
        elif isinstance(frame, (list, tuple)) and len(frame) >= 5:
            frames.append({
                "at_pct": frame[0], "hue_deg": frame[1],
                "hue_spread_deg": frame[2], "saturation_pct": frame[3],
                "brightness_pct": frame[4],
            })
    return {
        "keyframes": frames,
        "anchor_hues": entry.get("a", []),
        "reasoning": entry.get("r", ""),
    }


def _load_reference_plans() -> dict[str, dict]:
    """Load the preserved 84-plan archive, separate from user-saved plans."""
    global _REFERENCE_PLANS
    if _REFERENCE_PLANS is None:
        try:
            payload = json.loads(gzip.decompress(RECOVERED_PLAN_CACHE_PATH.read_bytes()).decode("utf-8"))
            _REFERENCE_PLANS = payload.get("tracks", {}) if payload.get("v") == 2 else {}
        except (OSError, ValueError, EOFError, gzip.BadGzipFile):
            _REFERENCE_PLANS = {}
    return _REFERENCE_PLANS


def _get_reference_plan(track_id: str) -> TrackLightingPlan | None:
    """Find a preserved plan by Spotify URI or plain Spotify track ID."""
    if not track_id:
        return None
    references = _load_reference_plans()
    candidates = [track_id]
    if track_id.startswith("spotify:track:"):
        candidates.append(track_id.removeprefix("spotify:track:"))
    else:
        candidates.append(f"spotify:track:{track_id}")
    for candidate in candidates:
        entry = references.get(candidate)
        if isinstance(entry, dict):
            try:
                return TrackLightingPlan.model_validate(_cache_entry_plan(entry))
            except ValidationError:
                continue
    return None


def _reference_examples(title: str, artist: str, limit: int = 3) -> list[dict]:
    """Return a few similar old plans as compact examples for prompt guidance."""
    references = _load_reference_plans()

    def words(value: str) -> set[str]:
        return {word.lower() for word in value.split() if len(word) > 2}

    query = words(f"{title} {artist}")
    ranked = []
    for entry in references.values():
        if not isinstance(entry, dict):
            continue
        expanded = _cache_entry_plan(entry)
        try:
            plan = TrackLightingPlan.model_validate(expanded)
        except ValidationError:
            continue
        if not plan.keyframes or not plan.reasoning.strip():
            continue
        text = " ".join((entry.get("t", ""), entry.get("n", ""), plan.reasoning))
        reference_words = words(text)
        score = len(query & reference_words) / max(1, len(query | reference_words))
        # Prefer populated, varied examples if metadata provides no match.
        score += min(len(plan.anchor_hues), 4) * 0.002
        compact = {
            "mood_and_palette": plan.reasoning[:200],
            "anchor_hues": [round(hue, 1) for hue in plan.anchor_hues[:4]],
            "keyframes": [[round(frame.at_pct), round(frame.hue_deg),
                           round(frame.hue_spread_deg), frame.saturation_pct,
                           frame.brightness_pct] for frame in plan.keyframes[:6]],
        }
        ranked.append((score, compact))
    ranked.sort(key=lambda item: item[0], reverse=True)

    examples = []
    used_hues = []
    for _, example in ranked:
        hue = example["keyframes"][0][1] if example["keyframes"] else None
        if hue is not None and any(min((hue - old) % 360, (old - hue) % 360) < 28 for old in used_hues):
            continue
        examples.append(example)
        if hue is not None:
            used_hues.append(hue)
        if len(examples) >= limit:
            break
    return examples


def _write_plan_cache_unlocked() -> None:
    # gzip keeps the local history small; an atomic replace avoids a partial
    # cache if the app exits during a write.
    payload = {"v": 2, "tracks": _PLAN_CACHE or {}}
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    temporary = PLAN_CACHE_PATH.with_suffix(".tmp")
    temporary.write_bytes(gzip.compress(encoded, compresslevel=9, mtime=0))
    os.replace(temporary, PLAN_CACHE_PATH)


def _store_plan(track_id: str, plan: TrackLightingPlan, title: str = "", artist: str = "") -> None:
    cache = _load_plan_cache()
    with _PLAN_CACHE_LOCK:
        cache[track_id] = {
            "k": [[frame.at_pct, frame.hue_deg, frame.hue_spread_deg,
                   frame.saturation_pct, frame.brightness_pct] for frame in plan.keyframes],
            "a": plan.anchor_hues[:4],
            "r": plan.reasoning[:280],
            "t": title[:160],
            "n": artist[:160],
            "s": int(time.time()),
        }
        # Keep old songs available while bounding local storage.
        while len(cache) > 2000:
            oldest = min(cache, key=lambda key: cache[key].get("s", 0))
            cache.pop(oldest, None)
        _write_plan_cache_unlocked()


def list_saved_songs() -> list[dict]:
    cache = _load_plan_cache()
    with _PLAN_CACHE_LOCK:
        songs = []
        for track_id, meta in cache.items():
            songs.append({
                "track_id": track_id,
                "title": meta.get("t") or "Unknown title",
                "artist": meta.get("n") or meta.get("r", "")[:120] or "Spotify track ID available",
                "saved_at": meta.get("s", 0),
            })
        return sorted(songs, key=lambda item: item["saved_at"], reverse=True)


def delete_saved_song(track_id: str) -> bool:
    cache = _load_plan_cache()
    with _PLAN_CACHE_LOCK:
        if track_id not in cache:
            return False
        cache.pop(track_id, None)
        _write_plan_cache_unlocked()
        return True


def _remember_song_title(track_id: str, title: str, artist: str) -> None:
    cache = _load_plan_cache()
    with _PLAN_CACHE_LOCK:
        meta = cache.get(track_id)
        if meta is None:
            return
        changed = False
        if title and meta.get("t") != title:
            meta["t"] = title[:160]
            changed = True
        if artist and meta.get("n") != artist:
            meta["n"] = artist[:160]
            changed = True
        if changed:
            meta.setdefault("s", int(time.time()))
            _write_plan_cache_unlocked()


def _reserve_groq_slot() -> None:
    """Locally cap calls at two image requests per rolling minute."""
    while True:
        now = time.monotonic()
        while _GROQ_REQUEST_TIMES and now - _GROQ_REQUEST_TIMES[0] >= 60:
            _GROQ_REQUEST_TIMES.popleft()
        if len(_GROQ_REQUEST_TIMES) < _LOCAL_REQUESTS_PER_MINUTE:
            _GROQ_REQUEST_TIMES.append(now)
            return
        time.sleep(max(0.05, 60 - (now - _GROQ_REQUEST_TIMES[0])))


# Groq JSON mode guarantees valid JSON, not this application's specific shape,
# so the required fields are also spelled out in the prompt.
JSON_SHAPE_INSTRUCTIONS = """
Respond with ONLY a JSON object (no markdown fences, no commentary) with
exactly these keys:
{
  "keyframes": [{"at_pct": <0-100>, "hue_deg": <0-360>,
    "hue_spread_deg": <0-180>, "saturation_pct": <0-100>,
    "brightness_pct": <0-100>}, ...],
  "anchor_hues": [<up to four hue angles>],
  "reasoning": "<one short sentence>"
}
"""


class _BaseLightingEngine:
    """Shared album-art downsampling/caching and prompt-building."""

    ART_MAX_DIMENSION = 128
    ART_JPEG_QUALITY = 45

    def __init__(self, segment_count: int = 0):
        self.segment_count = segment_count
        self._art_cache_track_id = None
        self._art_cache_bytes = None

    def _get_album_art_bytes(self, track_id: str, album_art_url: str) -> bytes | None:
        if track_id == self._art_cache_track_id and self._art_cache_bytes:
            return self._art_cache_bytes
        if not album_art_url:
            return None
        try:
            resp = requests.get(album_art_url, timeout=10)
            resp.raise_for_status()
            # Dominant palette and broad visual mood do not need Spotify's
            # full-resolution cover. Downsample and recompress before sending
            # any image bytes to the AI provider.
            with Image.open(BytesIO(resp.content)) as source:
                image = source.convert("RGB")
                image.thumbnail((self.ART_MAX_DIMENSION, self.ART_MAX_DIMENSION), Image.Resampling.LANCZOS)
                compressed = BytesIO()
                image.save(compressed, format="JPEG", quality=self.ART_JPEG_QUALITY, optimize=True)
            art_bytes = compressed.getvalue()
            self._art_cache_track_id = track_id
            self._art_cache_bytes = art_bytes
            return art_bytes
        except (requests.RequestException, OSError, ValueError, UnidentifiedImageError):
            return None

    def _clamp(self, decision: LightingDecision, sleep_timer_minutes_remaining: float | None) -> LightingDecision:
        decision.brightness_pct = max(0, min(100, decision.brightness_pct))
        decision.loop_delay_seconds = max(0.5, min(60.0, decision.loop_delay_seconds))
        if sleep_timer_minutes_remaining is not None:
            fraction = max(0.0, min(1.0, sleep_timer_minutes_remaining / 15.0))
            decision.loop_delay_seconds = max(30.0, decision.loop_delay_seconds)
            decision.brightness_pct = min(decision.brightness_pct, round(35 * fraction))
            decision.rgb = 0x0A1A4D
            decision.segments = []
        # Keep system transition white distinct from AI-selected song colors.
        decision.rgb = _avoid_white(decision.rgb)
        for seg in decision.segments:
            seg.rgb = _avoid_white(seg.rgb)
        return decision


def _avoid_white(rgb_int: int) -> int:
    """If rgb is white or near-white (all channels bright and close together),
    nudge it into a clearly-colored, still-bright hue instead. Near-white is
    defined as all three channels above 235 — a normal warm/cool bright color
    won't trip this, only true white or things a viewer would perceive as
    white."""
    r, g, b = (rgb_int >> 16) & 0xFF, (rgb_int >> 8) & 0xFF, rgb_int & 0xFF
    if r > 235 and g > 235 and b > 235:
        return 0xFFD24D  # warm bright gold — reads as "bright" without being white
    return rgb_int


# Oklab color math used by the preserved procedural effect renderer. These
# helpers were part of the earlier lighting brain but were lost when the AI
# planner was replaced, leaving effects.py present but impossible to import.
_MIN_PERCEIVED_L = 0.08
_MAX_PERCEIVED_L = 1.0
_SATURATION_FLOOR_PCT = 62.0


def _srgb_to_linear(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1.0 / 2.4) - 0.055


def _linear_rgb_to_oklab(red: float, green: float, blue: float) -> tuple[float, float, float]:
    l = math.cbrt(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue)
    m = math.cbrt(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue)
    s = math.cbrt(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def _linear_rgb_from_oklab(lightness: float, a: float, b: float) -> tuple[float, float, float]:
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )


def _oklab_hue_angle(hue_deg: float) -> float:
    red, green, blue = colorsys.hsv_to_rgb((hue_deg % 360.0) / 360.0, 1.0, 0.60)
    _, a, b = _linear_rgb_to_oklab(
        _srgb_to_linear(red), _srgb_to_linear(green), _srgb_to_linear(blue)
    )
    return math.degrees(math.atan2(b, a)) % 360.0


def _hue_lerp(old: float, new: float, amount: float) -> float:
    delta = ((new - old + 540.0) % 360.0) - 180.0
    return (old + delta * max(0.0, min(1.0, amount))) % 360.0


def _max_in_gamut_chroma(lightness: float, hue_deg: float) -> float:
    radians = math.radians(hue_deg)
    low, high = 0.0, 0.5
    for _ in range(24):
        chroma = (low + high) * 0.5
        channels = _linear_rgb_from_oklab(
            lightness, chroma * math.cos(radians), chroma * math.sin(radians)
        )
        if all(0.0 <= channel <= 1.0 for channel in channels):
            low = chroma
        else:
            high = chroma
    return low


def _oklab_to_rgb_int(lightness: float, a: float, b: float) -> int:
    channels = _linear_rgb_from_oklab(lightness, a, b)
    encoded = [round(_linear_to_srgb(channel) * 255) for channel in channels]
    return encoded[0] << 16 | encoded[1] << 8 | encoded[2]


class GroqLightingEngine(_BaseLightingEngine):
    """Generates and caches one continuous keyframe plan per Spotify song."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "qwen/qwen3.8-27b"

    def __init__(self, api_key: str, segment_count: int = 0):
        super().__init__(segment_count)
        self.api_key = api_key

    def _build_plan_prompt(self, title: str, artist: str) -> str:
        examples = _reference_examples(title, artist)
        return f"""You are the lighting planner for a Spotify-synced Govee LED visualizer.
Create one cohesive, continuous lighting plan for the entire song from its title,
artist, and attached album artwork. Do not claim to hear audio: infer likely mood
and energy from metadata and cover art. Treat the old user-approved plans below as
examples of the intended style and structure. Reuse their thoughtful, coherent
palette development, but choose colors that fit this song's own artwork and mood.

Track title: {json.dumps(title)}
Artist: {json.dumps(artist)}
Controllable color zones: {self.segment_count} (0 means solid-color only).

Return the OLD full-song keyframe structure, not intro/main/outro phases:
- Give 6 keyframes at about 0, 18, 38, 60, 82, and 100 percent.
- Keyframe hues, spread, saturation, and brightness should evolve smoothly;
  let important sections change naturally without abrupt random jumps.
- Use 2-4 anchor_hues that describe the cover-art/song palette.
- Make the palette visibly varied across the strip, not a single solid wash:
  choose at least 3 distinct, related anchor_hues when the artwork supports it,
  and order them to create a gentle color progression across the segments.
  Keep neighboring anchors different enough to be visibly distinct, while
  staying within one cohesive song palette; avoid repeating nearly identical
  anchors or collapsing the whole strip to one hue. If only one zone is
  controllable, use a moderate hue_spread_deg so its color still drifts over
  the song rather than remaining static.
- brightness_pct describes musical energy and movement; the device brightness
  cap is controlled separately by the dashboard.
- Never use pure white as a song color. Keep reasoning to one sentence.

Reference examples selected from the 84 saved old-format plans (JSON data only;
adapt the style, do not copy unrelated colors):
{json.dumps(examples, ensure_ascii=False, separators=(',', ':'))}"""

    def _call_groq(self, prompt: str, image_bytes: bytes | None) -> dict:
        text_only_content = [{"type": "text", "text": prompt}]
        content = list(text_only_content)
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )

        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": content}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "track_lighting_plan",
                    "strict": True,
                    "schema": TrackLightingPlan.model_json_schema(),
                },
            },
            "max_completion_tokens": 1024,
            "reasoning_effort": "none",
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
        }

        # Two image-bearing plans per rolling minute leaves headroom under
        # Qwen 3.8's base 8K TPM limit. The shared lock prevents worker bursts.
        with _GROQ_REQUEST_LOCK:
            for attempt in range(3):
                _reserve_groq_slot()
                try:
                    # Album art is sent only on the first attempt for this
                    # uncached track. If the request needs a retry, use the
                    # metadata-only prompt so the image is never uploaded twice.
                    attempt_payload = payload if attempt == 0 else {
                        **payload,
                        "messages": [{"role": "user", "content": text_only_content}],
                    }
                    resp = requests.post(
                        self.API_URL,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=attempt_payload,
                        timeout=20,
                    )
                except (requests.Timeout, requests.ConnectionError):
                    if attempt >= 2:
                        raise
                    time.sleep(2 ** attempt + random.random())
                    continue

                if resp.status_code == 429:
                    if attempt >= 2:
                        resp.raise_for_status()
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after is not None else 5.0
                    except ValueError:
                        delay = 5.0
                    if delay > 25:
                        resp.raise_for_status()
                    time.sleep(delay + random.uniform(0.05, 0.25))
                    continue

                if 500 <= resp.status_code < 600:
                    if attempt >= 2:
                        resp.raise_for_status()
                    time.sleep(2 ** attempt + random.random())
                    continue

                if resp.status_code == 401:
                    raise RuntimeError(
                        "Groq rejected the API key (401 Unauthorized). Check that the Groq key "
                        "in Setup is current, copied without extra spaces, and has not been revoked."
                    )

                resp.raise_for_status()
                raw_text = resp.json()["choices"][0]["message"]["content"]
                return json.loads(raw_text)

        raise RuntimeError("Groq did not return a lighting plan.")

    def get_plan(
        self,
        title: str,
        artist: str,
        album_art_url: str,
        track_id: str,
    ) -> TrackLightingPlan:
        """Load a personal plan, then an archived plan, generating only if neither exists."""
        cache = _load_plan_cache()
        if track_id and track_id in cache:
            try:
                _remember_song_title(track_id, title, artist)
                return TrackLightingPlan.model_validate(_cache_entry_plan(cache[track_id]))
            except ValidationError:
                with _PLAN_CACHE_LOCK:
                    cache.pop(track_id, None)
                    _write_plan_cache_unlocked()

        reference_plan = _get_reference_plan(track_id)
        if reference_plan is not None:
            return reference_plan

        # Recheck after acquiring the shared request lock so concurrent workers
        # reuse the first generated plan instead of issuing duplicate requests.
        with _GROQ_REQUEST_LOCK:
            cache = _load_plan_cache()
            if track_id and track_id in cache:
                try:
                    _remember_song_title(track_id, title, artist)
                    return TrackLightingPlan.model_validate(_cache_entry_plan(cache[track_id]))
                except ValidationError:
                    with _PLAN_CACHE_LOCK:
                        cache.pop(track_id, None)
                        _write_plan_cache_unlocked()

            reference_plan = _get_reference_plan(track_id)
            if reference_plan is not None:
                return reference_plan

            image_bytes = self._get_album_art_bytes(track_id, album_art_url)
            parsed = self._call_groq(self._build_plan_prompt(title, artist), image_bytes)
            plan = TrackLightingPlan.model_validate(parsed)
            if track_id:
                _store_plan(track_id, plan, title, artist)
            return plan

    def has_plan(self, track_id: str) -> bool:
        cache = _load_plan_cache()
        if not track_id:
            return False
        with _PLAN_CACHE_LOCK:
            entry = cache.get(track_id)
            if entry is not None:
                try:
                    TrackLightingPlan.model_validate(_cache_entry_plan(entry))
                    return True
                except ValidationError:
                    pass
        return _get_reference_plan(track_id) is not None

    def sample_keyframe(self, plan: TrackLightingPlan, progress_pct: float) -> LightingKeyframe:
        frames = sorted(plan.keyframes, key=lambda frame: frame.at_pct)
        pct = max(0.0, min(100.0, progress_pct))
        left, right = frames[0], frames[-1]
        for first, second in zip(frames, frames[1:]):
            if first.at_pct <= pct <= second.at_pct:
                left, right = first, second
                break
        amount = 0.0 if right.at_pct == left.at_pct else (pct - left.at_pct) / (right.at_pct - left.at_pct)
        hue_delta = (right.hue_deg - left.hue_deg + 180.0) % 360.0 - 180.0
        hue = (left.hue_deg + hue_delta * amount) % 360.0
        spread = left.hue_spread_deg + (right.hue_spread_deg - left.hue_spread_deg) * amount
        saturation = round(left.saturation_pct + (right.saturation_pct - left.saturation_pct) * amount)
        brightness = round(left.brightness_pct + (right.brightness_pct - left.brightness_pct) * amount)
        return LightingKeyframe(
            at_pct=pct, hue_deg=hue, hue_spread_deg=spread,
            saturation_pct=saturation, brightness_pct=brightness,
        )

    def decision_at(
        self,
        plan: TrackLightingPlan,
        progress_pct: float,
        sleep_timer_minutes_remaining: float | None = None,
    ) -> LightingDecision:
        frame = self.sample_keyframe(plan, progress_pct)
        hue, spread = frame.hue_deg, frame.hue_spread_deg
        saturation, brightness = frame.saturation_pct, frame.brightness_pct
        r, g, b = (round(channel * 255) for channel in colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, 1.0))
        segments = []
        anchors = plan.anchor_hues[:min(len(plan.anchor_hues), self.segment_count)] if self.segment_count > 0 else []
        if anchors:
            for slot, anchor in enumerate(anchors):
                start = slot * self.segment_count // len(anchors)
                end = (slot + 1) * self.segment_count // len(anchors)
                if end <= start:
                    continue
                ar, ag, ab = (round(channel * 255) for channel in colorsys.hsv_to_rgb(anchor / 360.0, saturation / 100.0, 1.0))
                segments.append(SegmentColor(index=list(range(start, end)), rgb=(ar << 16) | (ag << 8) | ab))
        decision = LightingDecision(
            rgb=(r << 16) | (g << 8) | b,
            brightness_pct=brightness,
            loop_delay_seconds=4.0,
            segments=segments,
            reasoning=plan.reasoning,
        )
        return self._clamp(decision, sleep_timer_minutes_remaining)

    def decide(
        self,
        title: str,
        artist: str,
        album_art_url: str,
        track_id: str,
        progress_pct: float,
        govee=None,
        sleep_timer_minutes_remaining: float | None = None,
    ) -> LightingDecision:
        plan = self.get_plan(title, artist, album_art_url, track_id)
        return self.decision_at(plan, progress_pct, sleep_timer_minutes_remaining)
