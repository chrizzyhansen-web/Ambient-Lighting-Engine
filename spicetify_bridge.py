"""
In-memory "now playing" store fed by the Spicetify browser extension
(see spicetify-extension/ambientLightingBridge.js), instead of polling
Spotify's paid Developer API.

The extension runs inside the Spotify desktop client itself and POSTs
the current track's metadata + progress to /api/spicetify/now-playing
every few seconds. This works with a completely free Spotify account,
since it never touches Spotify's Web API at all — it reads state that's
already available locally inside the desktop app.
"""
import threading
import time

_lock = threading.Lock()
_state = {
    "track_id": None,
    "title": None,
    "artist": None,
    "album_art_url": None,
    "progress_ms": 0,
    "duration_ms": 1,
    "is_playing": False,
    "next_track": None,
    "last_seen": 0.0,
    "last_unauthorized_at": 0.0,
}

# A one-shot command for the extension to act on next time it pushes —
# "pause"/"resume" while the AI pipeline is computing a track's first
# lighting decision. Delivered in the response to the extension's own
# POST, then cleared so it's only acted on once.
_pending_command: str | None = None
_command_lock = threading.Lock()

# If we haven't heard from the extension in this long, treat it as
# disconnected (e.g. Spotify closed, extension disabled) rather than
# trusting stale progress/track data.
STALE_AFTER_SECONDS = 15


def ingest(payload: dict):
    """Called by the Flask route whenever the extension posts an update
    with a valid token."""
    with _lock:
        _state.update(
            {
                "track_id": payload.get("track_id"),
                "title": payload.get("title"),
                "artist": payload.get("artist"),
                "album_art_url": payload.get("album_art_url"),
                "progress_ms": payload.get("progress_ms", 0),
                "duration_ms": payload.get("duration_ms") or 1,
                "is_playing": bool(payload.get("is_playing", False)),
                "next_track": payload.get("next_track") if isinstance(payload.get("next_track"), dict) else None,
                "last_seen": time.time(),
            }
        )


def set_pending_command(cmd: str | None):
    """Called by engine.py to queue a one-shot instruction ('pause' or
    'resume') for the Spicetify extension to act on next time it pushes."""
    global _pending_command
    with _command_lock:
        _pending_command = cmd


def consume_pending_command() -> str | None:
    """Called by the Flask route when responding to the extension's push —
    reads and clears the pending command so it's delivered exactly once."""
    global _pending_command
    with _command_lock:
        cmd = _pending_command
        _pending_command = None
        return cmd


def record_unauthorized():
    """Called when a request hits the endpoint with a missing/wrong token.
    Lets the Setup page distinguish 'nothing's reaching us' from 'something's
    reaching us but the token is stale/wrong' (usually a mismatched manual
    install vs. a re-generated token)."""
    with _lock:
        _state["last_unauthorized_at"] = time.time()


def unauthorized_recently() -> bool:
    with _lock:
        return (
            _state["last_unauthorized_at"] > 0
            and (time.time() - _state["last_unauthorized_at"]) < STALE_AFTER_SECONDS
        )


def _is_connected_unlocked() -> bool:
    """Assumes the caller already holds _lock. Never call this directly
    from outside this module."""
    return (time.time() - _state["last_seen"]) < STALE_AFTER_SECONDS and _state["last_seen"] > 0


def is_connected() -> bool:
    with _lock:
        return _is_connected_unlocked()


def get_now_playing() -> dict | None:
    """Same return shape as SpotifyClient.get_now_playing(), so engine.py
    can use either source interchangeably."""
    with _lock:
        if not _is_connected_unlocked() or not _state["title"]:
            return None
        duration_ms = _state["duration_ms"] or 1
        progress_ms = _state["progress_ms"] or 0
        return {
            "track_id": _state["track_id"],
            "title": _state["title"],
            "artist": _state["artist"],
            "album_art_url": _state["album_art_url"],
            "progress_ms": progress_ms,
            "duration_ms": duration_ms,
            "progress_pct": round(100 * progress_ms / duration_ms, 2),
            "is_playing": _state["is_playing"],
            "next_track": _state["next_track"],
        }


class SpicetifyBridgeSource:
    """Thin class wrapper so engine.py can do
    `source = SpicetifyBridgeSource()` symmetrically with `SpotifyClient(...)`."""

    def get_now_playing(self) -> dict | None:
        return get_now_playing()
