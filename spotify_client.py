"""
Thin wrapper around Spotify's Web API for reading live playback state.
Uses the Authorization Code refresh-token flow so it can run unattended.
"""
import base64
import time
import requests


TOKEN_URL = "https://accounts.spotify.com/api/token"
CURRENTLY_PLAYING_URL = "https://api.spotify.com/v1/me/player/currently-playing"
QUEUE_URL = "https://api.spotify.com/v1/me/player/queue"


class SpotifyClient:
    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self._access_token = None
        self._access_token_expires_at = 0
        self._next_track = None
        self._queue_checked_at = 0.0

    def _refresh_access_token(self):
        basic = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["access_token"]
        # Spotify tokens last 3600s; refresh a little early to be safe.
        self._access_token_expires_at = time.time() + data.get("expires_in", 3600) - 60
        # Spotify sometimes rotates the refresh token itself.
        if "refresh_token" in data:
            self.refresh_token = data["refresh_token"]

    def _get_access_token(self) -> str:
        if not self._access_token or time.time() >= self._access_token_expires_at:
            self._refresh_access_token()
        return self._access_token

    def get_now_playing(self) -> dict | None:
        """
        Returns a dict:
        {
            "track_id": str,
            "title": str,
            "artist": str,
            "album_art_url": str,
            "progress_ms": int,
            "duration_ms": int,
            "progress_pct": float,   # 0-100
            "is_playing": bool,
        }
        or None if nothing is currently playing (paused, no active device, ads, etc.)
        """
        token = self._get_access_token()
        resp = requests.get(
            CURRENTLY_PLAYING_URL,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )

        if resp.status_code == 204 or not resp.content:
            return None
        if resp.status_code == 401:
            # Token might have just been invalidated; force a refresh next call.
            self._access_token = None
            return None
        resp.raise_for_status()

        data = resp.json()
        item = data.get("item")
        if not item or data.get("currently_playing_type") != "track":
            return None  # podcast episode, ad, or nothing usable

        duration_ms = item.get("duration_ms") or 1
        progress_ms = data.get("progress_ms") or 0
        images = item.get("album", {}).get("images", [])
        album_art_url = images[0]["url"] if images else None

        now = time.monotonic()
        if now - self._queue_checked_at >= 20.0:
            self._queue_checked_at = now
            try:
                queue_resp = requests.get(
                    QUEUE_URL,
                    headers={"Authorization": f"Bearer {self._get_access_token()}"},
                    timeout=8,
                )
                if queue_resp.ok:
                    queued = (queue_resp.json().get("queue") or [])
                    next_item = next((item for item in queued if item.get("type") == "track"), None)
                    if next_item:
                        images = next_item.get("album", {}).get("images", [])
                        self._next_track = {
                            "track_id": next_item.get("uri") or next_item.get("id"),
                            "title": next_item.get("name"),
                            "artist": ", ".join(a.get("name", "") for a in next_item.get("artists", []) if a.get("name")),
                            "album_art_url": images[0].get("url") if images else None,
                        }
                    else:
                        self._next_track = None
            except requests.RequestException:
                pass  # keep the last known queue item; playback polling still works

        return {
            "track_id": item.get("id"),
            "title": item.get("name"),
            "artist": ", ".join(a["name"] for a in item.get("artists", [])),
            "album_art_url": album_art_url,
            "progress_ms": progress_ms,
            "duration_ms": duration_ms,
            "progress_pct": round(100 * progress_ms / duration_ms, 2),
            "is_playing": data.get("is_playing", False),
            "next_track": self._next_track,
        }
