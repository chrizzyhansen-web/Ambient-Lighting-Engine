"""
Local config storage. Replaces manual .env editing with a JSON file
that the web setup wizard reads and writes.

File lives at config.json in the project root and is gitignored —
never commit it, since it holds your live API keys/tokens.
"""
import json
import os
import secrets
import threading

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
_lock = threading.Lock()

DEFAULTS = {
    "track_source": "spicetify",  # "spotify_api" or "spicetify"
    "spotify_client_id": "",
    "spotify_client_secret": "",
    "spotify_refresh_token": "",
    "groq_api_key": "",
    "ai_provider": "groq",
    "govee_api_key": "",
    "govee_device_id": "",
    "govee_device_model": "",
    "poll_interval_seconds": 4,
    "sleep_timer_minutes": 0,
    "max_brightness_pct": 5,
    "flicker_enabled": True,
    "bridge_token": "",  # shared secret the Spicetify extension must send
}


def load_config() -> dict:
    with _lock:
        if not os.path.exists(CONFIG_PATH):
            data = dict(DEFAULTS)
            data["bridge_token"] = secrets.token_hex(16)
            with open(CONFIG_PATH, "w") as f:
                json.dump(data, f, indent=2)
            return data
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        merged = dict(DEFAULTS)
        merged.update(data)
        # This app now has a single AI backend. Ignore any provider left over
        # from an older config so setup changes cannot silently restore it.
        merged["ai_provider"] = "groq"
        if not merged.get("bridge_token"):
            merged["bridge_token"] = secrets.token_hex(16)
            with open(CONFIG_PATH, "w") as f:
                json.dump(merged, f, indent=2)
        return merged


def save_config(updates: dict) -> dict:
    with _lock:
        current = dict(DEFAULTS)
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                current.update(json.load(f))
        current.update(updates)
        current["ai_provider"] = "groq"
        current.pop("gemini_api_key", None)
        with open(CONFIG_PATH, "w") as f:
            json.dump(current, f, indent=2)
        return current


def is_fully_configured(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    required = ["govee_api_key", "govee_device_id", "govee_device_model"]
    required.append("groq_api_key")
    if cfg.get("track_source") == "spicetify":
        return all(cfg.get(k) for k in required)
    required += ["spotify_client_id", "spotify_client_secret", "spotify_refresh_token"]
    return all(cfg.get(k) for k in required)
