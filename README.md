# Ambient Lighting Engine

> [!WARNING]
> Epilepsy warning: This app drives real LED lights and may produce subtle flicker,
> flashing, or brightness changes. The effect is intentionally gentle and the neon
> flicker can be disabled in the dashboard, but people with photosensitive epilepsy
> or sensitivity to flashing lights should avoid using it or keep flicker turned off.

Turns a Govee RGBIC LED strip into a Spotify-synced visualizer. A background
loop polls your currently playing track every few seconds. On the first play
of a song, it sends the album art and track metadata to Groq once and saves a
continuous keyframe plan locally. Replays reuse that plan and interpolate its
colors locally without another AI request. The app
pushes the resulting color/brightness back to your lights — building up during intros,
peaking during choruses, fading during outros, and winding down to warm
amber/blue tones if you set a sleep timer.
