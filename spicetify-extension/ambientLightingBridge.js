// Ambient Lighting Engine — Spicetify bridge extension
//
// Runs inside your Spotify desktop client (via Spicetify) and pushes the
// current track's title, artist, album art, and playback progress to the
// local Ambient Lighting Engine app every few seconds. This needs a
// completely free Spotify account — it never calls Spotify's paid
// Developer API, it just reads state that's already sitting inside the
// desktop app.
//
// Install:
//   Easiest: use the "Install extension automatically" button on the app's
//   Setup page — it finds your Spicetify folder, writes this file with your
//   token already filled in, and runs the two Spicetify commands for you.
//
//   Manual alternative:
//   1. Copy this file into your Spicetify Extensions folder.
//      Find that folder's path by running: spicetify config-dir
//      (it's Config/spicetify/Extensions inside whatever that prints)
//   2. Replace __BRIDGE_TOKEN__ below with the token shown on the Setup page.
//   3. Run: spicetify config extensions ambientLightingBridge.js
//   4. Run: spicetify apply
//   5. Leave Spotify desktop open and logged in. That's it.

(function AmbientLightingBridge() {
  const ENDPOINT = "http://127.0.0.1:5000/api/spicetify/now-playing";
  const BRIDGE_TOKEN = "__BRIDGE_TOKEN__"; // auto-filled by the installer, or paste it in yourself
  const PUSH_INTERVAL_MS = 1000; // snappy enough that an auto-pause command lands quickly

  function init() {
    if (!Spicetify.Player || !Spicetify.Platform) {
      setTimeout(init, 200);
      return;
    }

    function toHttpImageUrl(spicetifyImageUri) {
      if (!spicetifyImageUri) return null;
      // Spicetify's metadata.image_*_url fields are internal URIs like
      // "spotify:image:ab67616d0000b273..." — convert to a real https URL.
      if (spicetifyImageUri.startsWith("spotify:image:")) {
        return "https://i.scdn.co/image/" + spicetifyImageUri.replace("spotify:image:", "");
      }
      return spicetifyImageUri; // already a full URL in some client versions
    }

    function queuedTrack(item) {
      if (!item) return null;
      const meta = item.metadata || {};
      const artists = item.artists || [];
      const albumImages = (item.album && item.album.images) || item.images || [];
      const artist = meta.artist_name || item.artist_name ||
        artists.map((a) => a.name).filter(Boolean).join(", ");
      const art = meta.image_xlarge_url || meta.image_large_url || meta.image_url ||
        (albumImages[0] && albumImages[0].url) || null;
      return {
        track_id: item.uri || item.id || null,
        title: meta.title || item.name || null,
        artist: artist || null,
        album_art_url: toHttpImageUrl(art),
      };
    }

    function currentPayload() {
      const data = Spicetify.Player.data;
      const item = data && (data.item || data.track);
      if (!item) {
        return { title: null, artist: null, album_art_url: null,
                 track_id: null, progress_ms: 0, duration_ms: 1, is_playing: false };
      }
      const meta = item.metadata || {};
      const albumArtRaw = meta.image_xlarge_url || meta.image_large_url || meta.image_url || null;
      const nextItem = data.nextItems && data.nextItems[0];

      return {
        track_id: item.uri || null,
        title: meta.title || null,
        artist: meta.artist_name || null,
        album_art_url: toHttpImageUrl(albumArtRaw),
        progress_ms: Spicetify.Player.getProgress ? Spicetify.Player.getProgress() : 0,
        duration_ms: Spicetify.Player.getDuration ? Spicetify.Player.getDuration() : 1,
        is_playing: Spicetify.Player.isPlaying ? Spicetify.Player.isPlaying() : false,
        next_track: queuedTrack(nextItem),
      };
    }

    // Tracks whether WE paused playback (vs. the person doing it themselves),
    // so we only auto-resume something we auto-paused — never fight a
    // deliberate manual pause.
    let weAutoPaused = false;

    function handleCommand(command) {
      if (command === "pause") {
        if (!weAutoPaused && Spicetify.Player.isPlaying && Spicetify.Player.isPlaying()) {
          Spicetify.Player.pause();
          weAutoPaused = true;
        }
      } else if (command === "resume") {
        if (weAutoPaused) {
          Spicetify.Player.play();
          weAutoPaused = false;
        }
      }
    }

    function push() {
      const payload = currentPayload();
      fetch(ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Bridge-Token": BRIDGE_TOKEN,
        },
        body: JSON.stringify(payload),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (data && data.command) handleCommand(data.command);
        })
        .catch(() => {
          // Local app probably isn't running right now — that's fine,
          // just try again on the next tick.
        });
    }

    // Push on every progress tick (throttled) and immediately on song change,
    // so the lighting engine doesn't wait a full interval after a track switch.
    let lastPush = 0;
    Spicetify.Player.addEventListener("onprogress", () => {
      const now = Date.now();
      if (now - lastPush >= PUSH_INTERVAL_MS) {
        lastPush = now;
        push();
      }
    });
    Spicetify.Player.addEventListener("songchange", () => {
      lastPush = Date.now();
      push();
    });
    Spicetify.Player.addEventListener("onplaypause", push);

    // Also push once right away so the dashboard shows "connected"
    // immediately after Spicetify (re)applies, even before a songchange fires.
    push();

    console.log("[AmbientLightingBridge] Spicetify extension active.");
  }

  init();
})();
