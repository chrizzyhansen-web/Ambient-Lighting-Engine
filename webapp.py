"""
Local web app for the ambient lighting engine.

Run:
    python webapp.py
Then open http://127.0.0.1:5000 in your browser.

No .env editing required — paste your keys into the Setup page and click
"Connect Spotify" to handle OAuth automatically.
"""
import base64
import os
import subprocess
import threading
import time
import urllib.parse
import webbrowser

import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, session

import config_store
import engine
import spicetify_bridge
import ai_lighting_engine

app = Flask(__name__)
app.secret_key = "dev-only-local-secret"  # fine for a localhost-only tool

SPOTIFY_REDIRECT_URI = "http://127.0.0.1:5000/spotify/callback"
SPOTIFY_SCOPES = "user-read-currently-playing user-read-playback-state"


@app.route("/")
def index():
    cfg = config_store.load_config()
    if not config_store.is_fully_configured(cfg):
        return redirect(url_for("setup"))
    return render_template("dashboard.html", cfg=cfg, active_page="dashboard")


@app.route("/setup", methods=["GET", "POST"])
def setup():
    if request.method == "POST":
        config_store.save_config(
            {
                "track_source": request.form.get("track_source", "spicetify"),
                "spotify_client_id": request.form.get("spotify_client_id", "").strip(),
                "spotify_client_secret": request.form.get("spotify_client_secret", "").strip(),
                "ai_provider": "groq",
                "groq_api_key": request.form.get("groq_api_key", "").strip(),
                "govee_api_key": request.form.get("govee_api_key", "").strip(),
                "govee_device_id": request.form.get("govee_device_id", "").strip(),
                "govee_device_model": request.form.get("govee_device_model", "").strip(),
            }
        )
        return redirect(url_for("setup"))

    cfg = config_store.load_config()
    return render_template("setup.html", cfg=cfg, active_page="setup")


@app.after_request
def add_cors_headers(response):
    # The Spicetify extension runs inside Spotify's Chromium-based client,
    # a different origin from this Flask server, so it's subject to normal
    # browser CORS rules. Our custom X-Bridge-Token header makes every
    # request "non-simple," which triggers a preflight OPTIONS check first —
    # without these headers, the browser blocks the real POST from ever
    # being sent, even though nothing is wrong with the POST itself.
    if request.path.startswith("/api/spicetify/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Bridge-Token"
        response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


@app.route("/api/spicetify/now-playing", methods=["POST", "OPTIONS"])
def spicetify_now_playing():
    if request.method == "OPTIONS":
        return "", 204
    cfg = config_store.load_config()
    token = request.headers.get("X-Bridge-Token", "")
    if not cfg.get("bridge_token") or token != cfg["bridge_token"]:
        spicetify_bridge.record_unauthorized()
        return jsonify({"ok": False, "error": "invalid or missing bridge token"}), 401
    spicetify_bridge.ingest(request.get_json(force=True, silent=True) or {})
    return jsonify({"ok": True, "command": spicetify_bridge.consume_pending_command()})


@app.route("/api/spicetify/status")
def spicetify_status():
    return jsonify(
        {
            "connected": spicetify_bridge.is_connected(),
            "unauthorized_recently": spicetify_bridge.unauthorized_recently(),
        }
    )


@app.route("/api/spicetify/install", methods=["POST"])
def spicetify_install():
    cfg = config_store.load_config()
    token = cfg["bridge_token"]
    log = []  # step-by-step trace, returned on failure so real errors are visible

    def run(cmd, timeout):
        log.append("$ " + " ".join(cmd))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            log.append("  -> command not found on PATH")
            return None
        except subprocess.TimeoutExpired:
            log.append(f"  -> timed out after {timeout}s")
            return None
        if r.stdout.strip():
            log.append("  stdout: " + r.stdout.strip())
        if r.stderr.strip():
            log.append("  stderr: " + r.stderr.strip())
        log.append(f"  -> exit code {r.returncode}")
        return r

    # `spicetify config-dir` OPENS a file manager window — it does not print
    # a path. `spicetify -c` prints the path to config-xpui.ini instead,
    # whose parent folder is what we actually need.
    result = run(["spicetify", "-c"], timeout=10)
    if result is None:
        return jsonify(
            {
                "ok": False,
                "error": "Couldn't run the 'spicetify' command at all — it's either not "
                "installed, or not on your PATH yet (closing and reopening this app after "
                "installing Spicetify sometimes fixes that). Install it from spicetify.app "
                "first, then try again.",
                "log": log,
            }
        )
    if result.returncode != 0 or not result.stdout.strip():
        return jsonify(
            {
                "ok": False,
                "error": "spicetify -c did not return a config path. See the log below for "
                "exactly what it said.",
                "log": log,
            }
        )

    config_file_path = result.stdout.strip()
    config_dir = os.path.dirname(config_file_path)
    ext_dir = os.path.join(config_dir, "Extensions")
    try:
        os.makedirs(ext_dir, exist_ok=True)
    except OSError as e:
        return jsonify({"ok": False, "error": f"Couldn't create {ext_dir}: {e}", "log": log})

    template_path = os.path.join(
        os.path.dirname(__file__), "spicetify-extension", "ambientLightingBridge.js"
    )
    try:
        with open(template_path, "r") as f:
            content = f.read()
    except OSError as e:
        return jsonify({"ok": False, "error": f"Couldn't read the bundled extension file: {e}", "log": log})
    content = content.replace("__BRIDGE_TOKEN__", token)

    dest_path = os.path.join(ext_dir, "ambientLightingBridge.js")
    try:
        with open(dest_path, "w") as f:
            f.write(content)
        log.append(f"Wrote extension to {dest_path}")
    except OSError as e:
        return jsonify({"ok": False, "error": f"Couldn't write {dest_path}: {e}", "log": log})

    result = run(["spicetify", "config", "extensions", "ambientLightingBridge.js"], timeout=15)
    if result is None or result.returncode != 0:
        return jsonify(
            {
                "ok": False,
                "error": "spicetify config extensions failed. See the log below for details.",
                "log": log,
            }
        )

    result = run(["spicetify", "apply"], timeout=45)
    if result is None or result.returncode != 0:
        apply_output = "\n".join(
            part for part in (
                result.stdout if result is not None else "",
                result.stderr if result is not None else "",
            ) if part
        ).casefold()
        if "preprocessed spotify data is outdated" in apply_output:
            log.append(
                "Detected stale preprocessed Spotify data; running Spicetify's "
                "documented restore/backup/apply recovery."
            )
            result = run(["spicetify", "restore", "backup", "apply"], timeout=90)
            if result is not None and result.returncode == 0:
                return jsonify(
                    {
                        "ok": True,
                        "message": "Spicetify refreshed its Spotify backup and applied the bridge. "
                        "Open Spotify desktop and play a song to confirm.",
                        "log": log,
                    }
                )
            return jsonify(
                {
                    "ok": False,
                    "error": "Spicetify's recovery command also failed. Fully close Spotify, "
                    "then click Install extension automatically again. See the log below.",
                    "log": log,
                }
            )
        return jsonify(
            {
                "ok": False,
                "error": "spicetify apply failed. See the log below — a common cause is "
                "Spotify desktop needing to be fully closed first.",
                "log": log,
            }
        )

    return jsonify(
        {
            "ok": True,
            "message": "Installed and applied. Open Spotify desktop and play a song to confirm.",
            "log": log,
        }
    )


@app.route("/help")
def help_page():
    return render_template("help.html", active_page="help")


@app.route("/spotify/connect")
def spotify_connect():
    cfg = config_store.load_config()
    if not cfg["spotify_client_id"]:
        return redirect(url_for("setup"))
    params = {
        "client_id": cfg["spotify_client_id"],
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SPOTIFY_SCOPES,
    }
    return redirect("https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params))


@app.route("/spotify/callback")
def spotify_callback():
    code = request.args.get("code")
    error = request.args.get("error")
    if error or not code:
        return render_template("setup.html", cfg=config_store.load_config(), active_page="setup",
                                spotify_error=error or "No authorization code returned.")

    cfg = config_store.load_config()
    basic = base64.b64encode(
        f"{cfg['spotify_client_id']}:{cfg['spotify_client_secret']}".encode()
    ).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
        },
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    if not resp.ok:
        return render_template("setup.html", cfg=cfg, active_page="setup", spotify_error=resp.text)

    tokens = resp.json()
    config_store.save_config({"spotify_refresh_token": tokens["refresh_token"]})
    return redirect(url_for("setup"))


@app.route("/engine/start", methods=["POST"])
def engine_start():
    cfg = config_store.load_config()
    if not config_store.is_fully_configured(cfg):
        return redirect(url_for("setup"))
    sleep_minutes = request.form.get("sleep_timer_minutes", "0")
    poll_interval = request.form.get("poll_interval_seconds", "4")
    config_store.save_config(
        {
            "sleep_timer_minutes": float(sleep_minutes or 0),
            "poll_interval_seconds": float(poll_interval or 4),
        }
    )
    engine.start(config_store.load_config())
    return redirect(url_for("index"))


@app.route("/engine/stop", methods=["POST"])
def engine_stop():
    engine.stop()
    return redirect(url_for("index"))


@app.route("/api/status")
def api_status():
    return jsonify(engine.get_status())


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True, silent=True) or {}
    try:
        brightness = max(1, min(100, int(data.get("max_brightness_pct", 5))))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Brightness must be between 1 and 100."}), 400
    flicker_enabled = bool(data.get("flicker_enabled", True))
    config_store.save_config({
        "max_brightness_pct": brightness,
        "flicker_enabled": flicker_enabled,
    })
    engine.update_runtime_settings(brightness, flicker_enabled)
    return jsonify({"ok": True, "max_brightness_pct": brightness, "flicker_enabled": flicker_enabled})


@app.route("/api/saved-songs")
def api_saved_songs():
    return jsonify({"songs": ai_lighting_engine.list_saved_songs()})


@app.route("/api/saved-songs/delete", methods=["POST"])
def api_delete_saved_song():
    data = request.get_json(force=True, silent=True) or {}
    track_id = str(data.get("track_id", ""))
    if not track_id or not ai_lighting_engine.delete_saved_song(track_id):
        return jsonify({"ok": False, "error": "That song plan was not found."}), 404
    return jsonify({"ok": True})


def _open_browser_when_ready():
    # Small delay so Flask has time to bind the port before the tab loads.
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    # threaded=True lets Flask handle multiple concurrent requests (the
    # Dashboard polling /api/status every 2s, the Spicetify extension
    # posting every 3s, etc.) instead of queuing them one at a time, which
    # otherwise makes later requests hang indefinitely under real usage.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
