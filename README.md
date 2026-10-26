# Ambient Lighting Engine

> [!WARNING]
> Epilepsy warning: This app drives real LED lights and may produce subtle flicker,
> flashing, or brightness changes. The effects are intentionally gentle and the
> neon flicker can be disabled in the dashboard, but people with photosensitive
> epilepsy or sensitivity to flashing lights should avoid it or keep flicker off.

quick setup summary for non-nerds, all you really need to do is download spicetify,
(https://spicetify.app/#install) get a groq api, (https://console.groq.com/keys) apply
for a govee api on your phone (Profile > Settings > Apply for Api) then let the auto setup
thing handle the rest of it.

Turns a Govee RGBIC LED strip into a Spotify-synced visualizer. A background
loop polls your currently playing track every few seconds. On the first play
of a song, it sends the album art and track metadata to Groq once and saves a
continuous keyframe plan locally. Replays reuse that plan and interpolate its
colors locally without another AI request. The app
pushes the resulting color/brightness back to your lights — building up during intros,
peaking during choruses, fading during outros, and winding down to warm
amber/blue tones if you set a sleep timer.

If you already have Spicetify installed and your API keys ready, setup is
pretty straightforward: just run the app, follow the built-in wizard, and fill
in the Spotify, Groq, and Govee details. No `.env` file to hand-edit — a local
web setup wizard handles all your keys.

## Reading "now playing" — two options

**Spicetify (default, recommended)** — a free extension that runs inside your
existing Spotify desktop app and pushes now-playing data to this app locally.
Works with a completely free Spotify account, no OAuth, no Premium. See the
in-app Help page (or `spicetify-extension/ambientLightingBridge.js`) for the
one-time install.

**Spotify Web API** — the original approach, using OAuth and Spotify's cloud
API. As of Spotify's Feb 2026 Developer Mode policy change, this now requires
the app-owning account to have Spotify Premium. Pick this in Setup only if
you have Premium and prefer the cloud API over Spicetify.

## Install & Run (Windows, no command line)

1. Make sure [Python](https://www.python.org/downloads/) is installed —
   during setup, check **"Add python.exe to PATH"**.
2. Double-click **`Setup (run once).bat`**. This installs everything the app
   needs. You only do this once.
3. Double-click **`Start App.bat`** any time you want to run it. It opens
your browser to the app automatically — no typed commands, no address to
remember.

### Want a real standalone `.exe` instead?

Once steps 1–2 above are done, you can turn this into a single `.exe` file
that runs without Python installed at all:

```
py -3 -m pip install pyinstaller
py -3 -m pyinstaller --onefile --add-data "templates;templates" --add-data "static;static" --add-data "templates;templates" --name "AmbientLightingEngine" webapp.py
```

This must be run once on a Windows machine (can't be cross-built from
Mac/Linux). The finished `.exe` will be in the `dist` folder — copy it
anywhere and double-click it like any other app. First run will still ask
Windows Defender/SmartScreen to confirm you trust it, since it's unsigned.

## Install & Run (Mac/Linux)

```bash
pip install -r requirements.txt --break-system-packages
python webapp.py
```

Open **http://127.0.0.1:5000** — it'll route you to the setup wizard on first
run.

## Setup wizard

1. **Spotify** — choose Spicetify for the free desktop bridge, or paste your
   Client ID/Secret from
   [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard).
   Add `http://127.0.0.1:5000/spotify/callback` as a Redirect URI in your
   Spotify app settings, save your keys, then click **Connect Spotify** —
   it handles the OAuth login and stores your refresh token automatically.
2. **Groq** — paste an API key from
   [console.groq.com/keys](https://console.groq.com/keys). Groq analyzes track
   metadata and album art and returns a full-song keyframe plan. It does not receive or
   listen to the song audio. Plans are saved in compressed `song_lighting_cache.json.gz` and
   reused when the same Spotify track plays again. Before its first AI request,
   the cover is reduced to at most 128×128 pixels and recompressed as a low-quality
   JPEG (quality 45) to preserve dominant colors with a much smaller upload.
3. **Govee** — paste your API key (kept for cloud-control fallback), plus
   your device's MAC-style ID and model/SKU (e.g. `H619B`). Enable **LAN
   Control** for the device in the Govee Home app and keep it on the same
   network as this computer.

Everything is saved to `config.json` in the project folder (gitignored — never
commit this file, it holds live credentials).

## Dashboard

Once configured, the dashboard shows a live color preview synced to what
your lights are currently doing, the track playing, and Start/Stop controls.
You can set:
- **Maximum brightness** — a saved, adjustable device-brightness ceiling,
  defaulting to 5%.
- **Neon segment flicker** — irregular bright/dim sputters on a small local
  group of segments, becoming more frequent with song energy; bursts stay under
  three seconds and can be disabled in the dashboard.
- **Saved song plans** — review cached plans and delete one to make Groq
  generate it again the next time that track plays.
- **Poll interval** — how often the loop checks Spotify. This does not cause
  extra Groq calls; each track's keyframe plan is generated once and sampled locally.
- **Sleep timer** — minutes until lights force ultra-low-intensity warm
  tones and eventually shut off entirely, overriding song energy.

## Govee connection behavior

The app discovers the configured light on your local network and sends
control commands directly over UDP when LAN Control is enabled. If discovery
fails or a LAN send raises an error, control falls back to Govee's cloud API.
The app also makes a startup cloud read to check device capabilities and learn
the RGBIC segment count. Govee LAN packets do not receive acknowledgements,
so a light that silently ignores a packet cannot be detected automatically.

When the device reports independently addressable zones and LAN control is
available, the engine renders a gentle water-like palette drift at up to 30 FPS and
sweeps a white transition wave across the zones on song changes or large color
shifts. Brief flicker bursts stay under three seconds. The app preloads the next
queued track's plan while the current song plays when Spotify exposes that
queue item. Other lights use whole-device color; cloud-only control stays
event-driven and applies a grouped palette when independent zones are available.

## Credits and references

- The layered, ocean-like motion is an original adaptation inspired by
  [Pacifica by Mark Kriegsman and Mary Corey March](https://github.com/FastLED/FastLED/blob/master/examples/Pacifica/Pacifica.ino)
  in the FastLED project. This app uses its own Oklab renderer and song-specific
  palettes rather than copying the sketch.
- The LAN frame timing and Razer/DreamView protocol were cross-checked against
  [LedFx](https://github.com/LedFx/LedFx) and the MIT-licensed
  [Govee Razer LED Controller](https://github.com/bochelork/govee_razer_led).
  No source code from those projects was copied for the frame-pacing change.

## Project structure

```
webapp.py            Flask app: setup wizard, OAuth callback, dashboard, API
engine.py             Background polling loop (start/stop/status)
spotify_client.py     Spotify Web API wrapper
ai_lighting_engine.py Builds the multimodal prompt, calls Groq, validates output
effects.py            Procedural Oklab palette motion and transition renderer
govee_client.py       LAN-first Govee control with cloud API fallback
config_store.py       Reads/writes config.json
templates/            Setup wizard + dashboard HTML
```

`audio_listener.py` and `audio_setup.py` are legacy audio-analysis modules and
are not connected to the active playback/lighting loop. The current AI infers
mood from Spotify's title, artist, and album art; it does not analyze the
recording's sound, beat, or lyrics.
