"""
The motion layer: what actually makes the strip look alive.

WHY THIS EXISTS: the AI plan alone produces a nearly static strip. It sets a
hue and a spread once per song and the code interpolates between keyframes, so
over a three-minute track the colour drifts about 0.4 degrees per second — and
if the model picks a small spread, which it usually does for anything calm,
every segment gets nearly the same colour and the result is a solid block that
sits there. Correct, smooth, and boring.

Motion has to be generated, not requested. So the AI's job shrinks to taste —
which colours suit this song, and how intense each part of it is — and
everything that MOVES is computed here, every frame.

THE APPROACH, adapted from Mark Kriegsman's Pacifica (the FastLED ocean
animation): several independent sine waves travel along the strip at speeds
that are deliberately not simple multiples of each other. Summed, they never
quite line up the same way twice, so the pattern reads as organic drift rather
than a loop. Where the layers constructively interfere you get a crest —
Pacifica calls these whitecaps and brightens them — which is what stops the
result looking like a uniform wash.

Three things are adapted rather than copied:

  * Pacifica renders in RGB. Everything here goes through the Oklab renderer
    in ai_lighting_engine, so crests brighten and colours shift without the
    perceptual lightness swings that made earlier gradients look blotchy.

  * Device brightness is hard-capped at 5%, so a crest cannot be expressed by turning
    the lamp up. Crests are rendered as CHROMA and perceptual-lightness
    changes within the palette instead, which cost nothing to send.

  * Energy drives everything. At low energy the layers crawl, the palette
    stays narrow and crests are soft — a cosy, barely-moving glow. As energy
    rises the layers speed up, the palette opens out, crests sharpen, and a
    fast ripple layer fades in to add grain. Same machinery throughout: no
    mode switch, so a song can move between the two without a visible jump.
"""
import math

from ai_lighting_engine import (
    _hue_lerp,
    _linear_to_srgb,
    _srgb_to_linear,
    _MAX_PERCEIVED_L,
    _MIN_PERCEIVED_L,
    _SATURATION_FLOOR_PCT,
    _avoid_white,
    _max_in_gamut_chroma,
    _oklab_hue_angle,
    _oklab_to_rgb_int,
)

# Wave layers: (spatial cycles along the strip, travel speed, weight).
#
# The speeds are intentionally awkward numbers. Harmonically related speeds
# resynchronise on a short cycle and the eye picks that up immediately as
# "it's looping"; these don't line up again for a very long time, which is the
# whole reason the result feels organic. The spatial frequencies are low on
# purpose — a 15-segment strip cannot show more than a couple of cycles before
# it turns into noise rather than a gradient.
#
# The fourth number is how fast that layer's own WEIGHT breathes, in Hz, so
# the proportions of the mix change slowly instead of being fixed.
#
# Worth being straight about what this does and doesn't do: it was added to
# fix "it gets stuck in one animation" and, measured on its own, it does NOT
# fix that — long-run variety moved in the fourth decimal, i.e. noise. The
# palette drift below is what fixes it. Breathing weights are kept because
# they do measurably help the OTHER failure mode: the worst-case colour
# spread across the strip rose from 0.084 to 0.094 at mid energy and from
# 0.022 to 0.029 on a two-stop palette, i.e. the flattest moments got less
# flat. It costs one sine per layer per frame.
_LAYERS = (
    (0.7, 0.037, 1.00, 0.0041),
    (1.1, 0.023, 0.62, 0.0073),
    (1.6, 0.061, 0.38, 0.0029),
    (0.4, 0.013, 0.80, 0.0053),
)

# How deeply those weights breathe. 0 restores fixed proportions; too high and
# layers vanish entirely, which shows up as the strip flattening out. At 0.45
# each layer ranges over roughly a third to its full weight and the summed
# field keeps a steady amplitude.
_WEIGHT_BREATH = 0.45

# A faster, finer layer that only appears as energy climbs. This is the
# "rougher when the song gets more intense" part: at rest it contributes
# nothing, and by full energy it adds visible grain and movement.
_GRAIN_LAYER = (2.3, 0.170, 0.55)

# Speed multiplier from calm to intense. The rest value is deliberately well
# under 1: a cosy passage should be something you notice has changed rather
# than something you watch moving. At 0.45 the slowest layer takes about a
# minute to complete a cycle.
_SPEED_AT_REST = 0.45
_SPEED_AT_PEAK = 2.4

# AMBIENT LIGHTNESS — the strip's resting color level. Device brightness is
# independently hard-capped to 5% in GoveeClient for every transport/state.
#
# The color lightness stays high enough to preserve 8-bit hue/chroma detail;
# the device-level cap controls actual output. Transient sweeps share that
# same 5% ceiling, while their higher color lightness keeps the motion visible.
#
# The device cap limits physical output for both the normal animation and
# transient sweeps; the lightness range keeps the palette legible at that cap.
_AMBIENT_L_MIN = 0.20
_AMBIENT_L_MAX = 0.36
_SWEEP_PEAK_L = 0.95
_MIN_VISIBLE_RGB_CHANNEL = 128

# THE UNDERLYING RAMP. The strip's own length always spans the whole palette
# — `travel` starts at `position` — and the waves displace that. This is what
# stops the strip going near-SOLID whenever the wave layers happen to line up;
# without it, calm passages measured as few as 6 distinct colours out of 15.
# (There was a _BASE_RAMP weight here when the two were BLENDED rather than
# added. Adding the field as a displacement is what fixed the first anchor
# never appearing, and it left that constant doing nothing, so it is gone.)

# How far the wave field shifts a segment's position along the palette. This
# is the movement you actually see in the colours; the strip's own length
# supplies the gradient underneath it.
_FIELD_TRAVEL = 0.28

# A slow drift of the WHOLE strip along the palette, on top of the waves.
# THIS is what stops the strip looking like it is stuck in one animation.
#
# The waves only ever push a segment back and forth around the place its
# position puts it, so the colour at any one point on the strip oscillates
# around a fixed value forever. Measured over ten minutes at rest: how
# different the strip looked from itself stopped growing at about a 20-second
# separation and never grew again — frames five minutes apart were no more
# different than frames twenty seconds apart. A whole song therefore showed
# no more variety than the first twenty seconds of it did. With the drift,
# that figure roughly doubles at one to five minutes of separation (0.047 to
# 0.097 at two minutes, low energy; 0.082 to 0.116 at mid energy).
#
# The palette is a ring (see _angle_at), so this wraps with no seam and
# without costing coverage.
#
# Two components, deliberately at an irrational-ish ratio. One alone is
# exactly periodic: at the first value tried, a full rotation took almost
# exactly two minutes, and the strip at t and t+120s measured as similar as
# frames twenty seconds apart — it had simply come back round. Two rates that
# don't divide each other take far longer to return to the same arrangement.
_PALETTE_DRIFT = 0.0037
_PALETTE_DRIFT_2 = 0.0011

# ---- Transient events -------------------------------------------------------
#
# The continuous wave field gives the strip life, but it has no PUNCTUATION:
# nothing ever arrives. Events are one-off transients laid over the top, so a
# moment in the music can actually land — a wave sweeping the length of the
# strip, or the palette changing outright.
#
# Each event is a pure function of how long ago it started, so the engine stays
# stateless per frame and a frame is still reproducible from (time, plan,
# events).

SWEEP_DURATION = 1.15      # seconds for a wave to cross the whole strip
SWEEP_WIDTH = 0.30         # how much of the strip the crest covers at once
SHIFT_DURATION = 0.85      # seconds to cross-fade from an old palette to a new


# ---- Accent: "the song changed, even though it looks the same" -------------
#
# The track-change wave works by REVEALING the new palette behind it. When the
# next song's colours are nearly the last song's, there is nothing for it to
# reveal and the wave reads as ordinary motion — you can miss the change
# entirely.
#
# Measured, at the resting lightness the strip actually sits at: a palette
# rotated by 25 degrees renders about 0.049 Oklab away from the original, and
# below roughly that figure the difference is easy to miss under a bright
# crest that is crossing at the same moment. So below SUBTLE_CHANGE_DEG the
# change gets an accent as well.
#
# The accent is a colour event rather than a brightness one: the crest itself
# takes a contrasting hue and carries it along the strip, leaving the new
# palette behind. Brightness is deliberately NOT the lever — a brighter flash
# is what the ordinary wave already does, so repeating it louder would read as
# the same event twice rather than a different one.
ACCENT_DURATION = 1.45     # a touch slower than a sweep, so it reads as deliberate
ACCENT_WIDTH = 0.34
ACCENT_HUE_SHIFT = 150.0   # degrees away from whatever the palette is doing
ACCENT_LIFT = 0.30         # how much lightness it adds, vs 1.0 for a sweep
SUBTLE_CHANGE_DEG = 25.0


# ---- Glitch --------------------------------------------------------------
#
# A deliberate stutter, for the cyberpunk look — as distinct from the flicker
# that was a bug. The difference is entirely in the constraints:
#
#   * It is LOCAL. A short run of segments stutters while the rest of the
#     strip carries on normally. Whole-strip flashing is what made the
#     accidental version unpleasant, and it is also the form that carries real
#     photosensitivity risk.
#   * It is SHORT, and that is enforced rather than intended: GLITCH_MAX_DURATION
#     is a hard clamp, so no caller can leave the strip stuttering.
#   * It stays IN PALETTE. The glitch jumps to a different point in the SAME
#     song palette rather than injecting an arbitrary colour, so it reads as
#     the light breaking up rather than as a wrong colour appearing.
#   * The blips are UNEVEN but deterministic — a fixed integer hash, not
#     randomness — so the same moment always renders the same way and it never
#     settles into a regular strobe.
#   * Neighbouring segments inside the affected run fire on DIFFERENT slots.
#     A first version had them all blink together, which measured as a 2/sec
#     blink with a 0.66 duty cycle and one case in which the whole run simply
#     stayed lit for the full 0.8s — a patch changing colour, not a stutter.
#   * No segment stays lit for more than two slots in a row. That is enforced
#     by looking at the two previous slots rather than by chance, so the worst
#     case is bounded instead of merely unlikely.

GLITCH_DURATION = 0.85
GLITCH_MAX_DURATION = 2.0   # hard limit; nothing may exceed this
GLITCH_WIDTH = 0.22         # fraction of the strip affected
GLITCH_RATE = 13.0          # slots per second; a blip is 1-2 slots
GLITCH_PALETTE_JUMP = 0.42  # how far along the palette a blip jumps
_GLITCH_BANDS = 32          # spatial resolution of the per-segment stagger
_GLITCH_THRESHOLD = 0.45    # below this, the slot stays dark


def _hash01(*parts: int) -> float:
    """A deterministic 0..1 value. Keeps the render reproducible — a frame is
    still a pure function of its inputs, which randomness would destroy."""
    h = 2166136261
    for p in parts:
        h = ((h ^ (p & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h / 4294967295.0


def _glitch_roll(seed: int, band: int, slot: int) -> float:
    """0 if this band is dark in this slot, else how hard it is lit."""
    if slot < 0:
        return 0.0
    r = _hash01(seed, band, slot, 31)
    return 0.0 if r < _GLITCH_THRESHOLD else r


def _glitch_amount(age: float, position: float, strength: float, seed: int) -> float:
    """How hard this segment is glitching right now, 0..1."""
    duration = min(GLITCH_DURATION, GLITCH_MAX_DURATION)
    if age < 0.0 or age > duration:
        return 0.0

    # A short run of segments, positioned from the seed so successive glitches
    # land in different places along the strip.
    centre = 0.12 + 0.76 * _hash01(seed, 7)
    if abs(position - centre) > GLITCH_WIDTH * 0.5:
        return 0.0

    # Each segment has its own band, so the run breaks up along its length
    # instead of blinking as one block.
    band = int(position * _GLITCH_BANDS)
    slot = int(age * GLITCH_RATE)
    roll = _glitch_roll(seed, band, slot)
    if roll <= 0.0:
        return 0.0
    # Hard cap on how long one segment can stay lit: if the two slots before
    # this one both fired, this one doesn't.
    if _glitch_roll(seed, band, slot - 1) > 0.0 and _glitch_roll(seed, band, slot - 2) > 0.0:
        return 0.0

    # Softer at the edges of the affected run, and fading over its lifetime.
    edge = 1.0 - (abs(position - centre) / (GLITCH_WIDTH * 0.5))
    fade = 1.0 - (age / duration) ** 1.5
    return strength * roll * edge * fade


def _sweep_centre(age: float) -> float:
    """Where the crest is along the strip, 0..1, for a sweep of this age.
    Runs slightly past both ends so the crest enters and leaves cleanly."""
    progress = (max(0.0, age) / SWEEP_DURATION) ** 0.85
    return -SWEEP_WIDTH + progress * (1.0 + 2.0 * SWEEP_WIDTH)


# HOW MUCH THE LIVE WAVE FIELD BENDS AN EVENT'S FRONT.
#
# Everything continuous on this strip flows — the layers are warped, the
# palette drifts, nothing travels in a straight line. Events did: a crest and
# a wipe boundary were both functions of `position` alone, so they crossed the
# strip as a perfectly straight front, at a constant shape, while the thing
# underneath them was rippling. That reads as a band sliding OVER the
# animation rather than as part of it.
#
# Feeding the same field that drives the ambient motion into the event's own
# position makes its front bulge where the flow is ahead and lag where it is
# behind. The crest is still one connected front — this is a displacement of
# a smooth field, not noise — it just isn't a ruler-straight one.
_EVENT_FLOW = 0.16


def _sweep_contribution(age: float, position: float, strength: float,
                        flow: float = 0.0) -> float:
    """A bright crest travelling from one end of the strip to the other.

    Returns an additive 0..1 boost for this segment. The crest is a gaussian
    so it has soft shoulders rather than a hard edge, and it accelerates
    slightly (age**0.85) so it reads as something arriving with momentum
    rather than a band sliding at constant speed. `flow` is the local wave
    field, which bends the front so it moves like liquid rather than a bar."""
    if age < 0.0 or age > SWEEP_DURATION:
        return 0.0
    centre = _sweep_centre(age)
    offset = (position + _EVENT_FLOW * flow - centre) / SWEEP_WIDTH
    crest = math.exp(-4.0 * offset * offset)
    # Fade the whole thing out over its lifetime so the tail doesn't just stop.
    fade = 1.0 - (age / SWEEP_DURATION) ** 2
    return strength * crest * fade


def _accent_contribution(age: float, position: float, strength: float,
                         flow: float = 0.0) -> float:
    """The accent crest — same travelling-gaussian shape as a sweep, on its
    own slightly slower clock and a little wider, so the two don't look like
    the same event when they land together."""
    if age < 0.0 or age > ACCENT_DURATION:
        return 0.0
    progress = (age / ACCENT_DURATION) ** 0.9
    centre = -ACCENT_WIDTH + progress * (1.0 + 2.0 * ACCENT_WIDTH)
    offset = (position + _EVENT_FLOW * flow - centre) / ACCENT_WIDTH
    crest = math.exp(-4.0 * offset * offset)
    fade = 1.0 - (age / ACCENT_DURATION) ** 2
    return strength * crest * fade


def palette_difference(
    old_hue_deg: float,
    old_width_deg: float,
    old_anchors: list | None,
    new_hue_deg: float,
    new_width_deg: float,
    new_anchors: list | None,
) -> float:
    """How far apart two song palettes are, in degrees of Oklab hue.

    Sampled at matched positions along both palettes rather than compared
    anchor-to-anchor, so palettes with different numbers of anchors still
    compare sensibly. Used to decide whether a track change needs an accent:
    the wave announces a change by revealing the new colours, which only
    works when there are new colours to reveal."""
    a = _palette_angles(old_hue_deg, old_width_deg, old_anchors)
    b = _palette_angles(new_hue_deg, new_width_deg, new_anchors)
    total = 0.0
    samples = 12
    for i in range(samples):
        travel = i / samples
        d = abs(_angle_at(travel, a) - _angle_at(travel, b)) % 360.0
        total += min(d, 360.0 - d)
    return total / samples


# Domain warping (Inigo Quilez's technique): instead of evaluating the wave
# field at a point, evaluate it at a point that has itself been DISPLACED by
# another slow field — f(p + h(p)) rather than f(p). The waves then bend,
# stretch and swirl instead of sliding along rigidly, which is what separates
# "flowing" from "scrolling". One level of warping is plenty here; IQ stacks
# several for 2D textures, but a 15-segment strip has nowhere near the spatial
# resolution to show a second level.
_WARP_CYCLES = 0.53
_WARP_SPEED = 0.011
_WARP_AT_REST = 0.20
_WARP_AT_PEAK = 0.34


def _warp(position: float, t: float, energy: float) -> float:
    """How far to displace the sampling point, in strip-lengths."""
    amount = _WARP_AT_REST + (_WARP_AT_PEAK - _WARP_AT_REST) * energy
    # Two slow, unrelated components so the distortion itself drifts rather
    # than being a fixed bend that merely slides.
    a = math.sin(2.0 * math.pi * (_WARP_CYCLES * position + _WARP_SPEED * t))
    b = math.sin(2.0 * math.pi * (0.31 * position - 0.007 * t + 0.37))
    return amount * (0.62 * a + 0.38 * b)


def _wave_field(position: float, t: float, energy: float) -> float:
    """Summed wave layers at a point along the strip, normalised to -1..1.

    `position` is 0.0 at the first segment, 1.0 at the last.
    """
    speed = _SPEED_AT_REST + (_SPEED_AT_PEAK - _SPEED_AT_REST) * energy
    # The warp is what makes this read as liquid rather than as sliding bands.
    position = position + _warp(position, t, energy)

    total = 0.0
    weight_sum = 0.0
    for cycles, layer_speed, weight, breath in _LAYERS:
        # Each layer's weight breathes on its own slow cycle, so the SHAPE of
        # the summed field reconfigures over minutes instead of one fixed
        # shape sliding along. Normalising by the live weights keeps the
        # amplitude steady while the mix changes.
        w = weight * (1.0 - _WEIGHT_BREATH + _WEIGHT_BREATH *
                      (0.5 + 0.5 * math.sin(2.0 * math.pi * breath * t)))
        total += w * math.sin(2.0 * math.pi * (cycles * position + layer_speed * speed * t))
        weight_sum += w

    cycles, layer_speed, weight = _GRAIN_LAYER
    # Squared so grain stays genuinely absent through the calm range rather
    # than merely quiet — a cosy passage should have no grain at all.
    grain = weight * (energy ** 2)
    if grain > 0.001:
        total += grain * math.sin(2.0 * math.pi * (cycles * position + layer_speed * speed * t))
        weight_sum += grain

    return total / weight_sum if weight_sum else 0.0


def _ring_travel(position: float, field: float, drift: float, stops: int) -> float:
    """Where on the palette ring this segment sits.

    The strip covers only PART of the ring — `stops - 1` hops out of `stops`,
    exactly the span it covered when the palette was an open list — and the
    drift rotates that window around the ring. Keeping the window the same
    width matters: letting the strip span the whole ring put 50% more hue
    across the same 15 segments, and the per-frame colour steps went up with
    it (measured 7 raw / 5 after smoothing, against a limit of 3), which is
    the flicker this code already fought once. Rotating a same-width window
    gives the drift for free, at the old hue velocity."""
    if stops <= 1:
        return 0.0
    window = (stops - 1) / stops
    return (window * (position + _FIELD_TRAVEL * field) + drift) % 1.0


def _sharpen(value: float, energy: float) -> float:
    """Push a -1..1 field toward its extremes as energy rises.

    At rest this is near enough a no-op and the field stays sinusoidal, so
    colour changes gradually across the strip. At high energy the curve
    steepens, compressing the mid-tones so crests read as distinct bands
    instead of a smooth wash — the difference between breathing and pulsing.

    USES tanh, NOT A POWER CURVE. The obvious version, sign(v) * |v| ** k,
    has an infinite derivative at zero: every time the wave field crossed
    zero the output lurched, which measured as frame-to-frame jumps eight
    times the median at high energy — visible as a stutter in what should be
    smooth fast motion. tanh is smooth everywhere and sharpens just as well.
    """
    if energy < 0.02:
        return value
    gain = 1.0 + 1.4 * energy
    return math.tanh(gain * value) / math.tanh(gain)


def _smooth_ceiling(L: float, angle_deg: float) -> float:
    """The gamut ceiling, averaged over a small window of hue.

    The true sRGB boundary has corners — around blue especially — and riding
    exactly against it means chroma changes as fast as the boundary does,
    which reads as flicker when the colour drifts across one. Averaging over
    a few degrees rounds those corners off. It gives up a sliver of
    achievable chroma near a corner in exchange for a colour that changes
    smoothly, which is the right trade for something animated."""
    total = 0.0
    for offset in (-8.0, -4.0, 0.0, 4.0, 8.0):
        total += _max_in_gamut_chroma(L, (angle_deg + offset) % 360.0)
    return total / 5.0


def _keep_visible(rgb: int) -> int:
    """Keep rendered frames visible after the device's low brightness cap.

    At a 5% device cap, near-black encoded RGB values can round down to what
    looks like an outage. Scale dim colors proportionally so their hue remains
    intact while their strongest channel stays above a visible floor.
    """
    channels = [(rgb >> shift) & 0xFF for shift in (16, 8, 0)]
    strongest = max(channels)
    if strongest == 0:
        return 0x204C8A
    if strongest < _MIN_VISIBLE_RGB_CHANNEL:
        scale = _MIN_VISIBLE_RGB_CHANNEL / strongest
        channels = [min(255, round(channel * scale)) for channel in channels]
    return channels[0] << 16 | channels[1] << 8 | channels[2]


def _palette_angles(hue_deg: float, width_deg: float, anchor_hues: list | None) -> list[float]:
    """The palette as a list of Oklab hue angles to travel between.

    Uses the song's ACTUAL anchor colours when they're available — the hues
    the model picked for this track, seeded by the album art's measured
    dominants. Those were being collected and then ignored: the strip was
    rendering a single hue plus a width, which is at most two related colours.
    Travelling across three real anchors is what puts genuinely different
    colours on the strip at once.

    Falls back to hue ± width when there are no anchors, so a cached plan from
    before anchors existed still renders.
    """
    if anchor_hues:
        usable = [h for h in anchor_hues if isinstance(h, (int, float))][:4]
        if len(usable) >= 2:
            return _at_least_three([_oklab_hue_angle(float(h)) for h in usable])
        if len(usable) == 1:
            hue_deg = float(usable[0])

    return _at_least_three([_oklab_hue_angle(hue_deg),
                            _oklab_hue_angle(hue_deg + width_deg)])


def _at_least_three(angles: list[float]) -> list[float]:
    """Split a two-stop palette at its midpoint.

    Purely so the palette RING isn't degenerate. Two stops make a ring that
    just retraces itself, and the drifting window then mirrors around an end
    instead of sliding — measured worst-case colour spread across the strip
    fell from 0.037 to 0.022 in that case, while three or more stops held
    steady. The midpoint is the colour the renderer was already interpolating
    through, so nothing looks different; it only gives the ring somewhere to
    go."""
    if len(angles) != 2:
        return angles
    a, b = angles
    delta = ((b - a + 540.0) % 360.0) - 180.0
    return [a, (a + delta * 0.5) % 360.0, b]


def _angle_at(travel: float, angles: list[float]) -> float:
    """Position `travel` along a multi-stop palette, each hop taken the short
    way round the wheel so no transition sweeps through unrelated hues.

    The palette is a RING: after the last stop it returns to the first, and
    `travel` wraps rather than clamping. That is what lets the whole strip
    drift along the palette without a seam — a clamped palette either stops
    at the ends (so the drift does nothing) or snaps when it wraps. It also
    means every stop is on the strip at every instant, so coverage no longer
    depends on where the drift happens to be: measured worst-case colour
    spread across the strip actually IMPROVED, from 0.099 to 0.111 at low
    energy, while a reflecting fold (tried first) halved it to 0.050."""
    if len(angles) == 1:
        return angles[0]
    k = len(angles)
    span = (travel % 1.0) * k
    i = int(span)
    frac = span - i
    a, b = angles[i % k], angles[(i + 1) % k]
    delta = ((b - a + 540.0) % 360.0) - 180.0
    return (a + delta * frac) % 360.0


def smooth_towards(previous: list[int] | None, target: list[int], alpha: float = 0.45) -> list[int]:
    """Ease a frame toward its target, blending in LINEAR light.

    Anti-flicker of last resort, and the one that works regardless of cause.
    The renderer is continuous in principle, but the sRGB gamut boundary has
    corners — pronounced around blue — and when a fast-moving hue crosses one,
    the achievable chroma changes abruptly through no fault of the maths.
    The visible-color floor is applied after this blend too, so crossfading
    between palettes cannot briefly dip into near-black.

    Blending happens in linear light rather than on the sRGB bytes, because
    sRGB is gamma-encoded: averaging the bytes darkens the result, and at this
    dim resting level that error is proportionally large.

    alpha 0.45 at 25fps is roughly a 50ms time constant — short enough that a
    sweep still arrives sharply (it reaches 95% within five frames) and long
    enough that no single-frame step survives.
    """
    if previous is None or len(previous) != len(target):
        return list(target)

    out = []
    for old, new in zip(previous, target):
        channels = []
        for shift in (16, 8, 0):
            o = _srgb_to_linear(((old >> shift) & 0xFF) / 255.0)
            n = _srgb_to_linear(((new >> shift) & 0xFF) / 255.0)
            blended = o + (n - o) * alpha
            channels.append(round(_linear_to_srgb(blended) * 255))
        out.append(_keep_visible((channels[0] << 16) | (channels[1] << 8) | channels[2]))
    return out


class EffectEngine:
    """Renders one frame of the animation.

    Holds no per-frame state: a frame is a pure function of (time, palette,
    energy). That means the same moment always looks the same, frames can be
    tested directly, and nothing can drift out of sync or accumulate error
    over a long session.
    """

    def __init__(self, segment_count: int):
        self.segment_count = max(1, int(segment_count))

    def render(
        self,
        t: float,
        hue_deg: float,
        palette_width_deg: float,
        saturation_pct: float,
        energy_pct: float,
        anchor_hues: list | None = None,
        events: list | None = None,
        previous_hue_deg: float | None = None,
        previous_anchor_hues: list | None = None,
        shift_started_at: float | None = None,
    ) -> list[int]:
        """One rgb int per segment.

        hue_deg            the song's primary colour
        palette_width_deg  how far the palette reaches from it
        saturation_pct     vividness, floored so colours never read as grey
        energy_pct         0-100: cosy drift at 0, rough and fast at 100
        anchor_hues        this song's real palette, travelled across in order
        events             transient (kind, start_time, strength) tuples
        previous_hue_deg   the palette being moved away from, during a shift
        shift_started_at   when that shift began
        """
        n = self.segment_count
        energy = max(0.0, min(1.0, energy_pct / 100.0))
        saturation = max(_SATURATION_FLOOR_PCT, min(100.0, saturation_pct)) / 100.0

        # The palette opens up with energy: a calm passage stays close to one
        # colour, an intense one ranges further. Clamped so even at full
        # energy the strip never becomes an indiscriminate rainbow.
        width = max(-160.0, min(160.0, palette_width_deg)) * (0.70 + 0.30 * energy)

        # A palette change is WIPED ACROSS THE STRIP BY THE CREST, not
        # cross-faded globally. Ahead of the wave the old colours are still
        # there; behind it, the new ones. The wave is the boundary between
        # them, which is what makes it read as the change being carried along
        # the strip rather than everything fading at once.
        old_palette = None
        wipe_centre = None
        if shift_started_at is not None and (
            previous_anchor_hues or previous_hue_deg is not None
        ):
            shift_age = t - shift_started_at
            if 0.0 <= shift_age <= SWEEP_DURATION:
                old_palette = _palette_angles(
                    previous_hue_deg if previous_hue_deg is not None else hue_deg,
                    width,
                    previous_anchor_hues,
                )
                wipe_centre = _sweep_centre(shift_age)

        # Working in Oklab hue angle rather than HSV, for the same reason the
        # gradient renderer does: equal steps there are equal to the eye.
        palette = _palette_angles(hue_deg, width, anchor_hues)

        # A slow global swell, independent of position — the whole strip
        # breathing gently under the travelling waves. Deliberately slower
        # than any layer so it never syncs up with them.
        swell = math.sin(2.0 * math.pi * 0.019 * t)
        # Energy spans nearly the whole ambient band, so a quiet passage sits
        # at the bottom of it and a loud one near the top. The band is narrow
        # on purpose — big brightness changes belong to sweeps, not to the
        # resting state.
        centre_L = _AMBIENT_L_MIN + (_AMBIENT_L_MAX - _AMBIENT_L_MIN) * (
            0.08 + 0.88 * energy
        )
        # How far crests reach above and troughs below the centre. Bigger at
        # high energy, which is where the visible "punch" comes from given
        # that actual brightness is capped.
        L_swing = (0.055 + 0.185 * energy) * (_MAX_PERCEIVED_L - _MIN_PERCEIVED_L)

        # The whole strip's slow walk along the palette — the same offset for
        # every segment, so the gradient keeps its shape while its colours
        # change. Tied to the energy speed, so an intense song moves through
        # its palette faster than a calm one.
        drift_speed = _SPEED_AT_REST + (_SPEED_AT_PEAK - _SPEED_AT_REST) * energy
        drift = (_PALETTE_DRIFT * drift_speed * t
                 + _PALETTE_DRIFT_2 * t)

        out: list[int] = []

        for i in range(n):
            position = 0.0 if n == 1 else i / (n - 1)
            field = _sharpen(_wave_field(position, t, energy), energy)

            # Position along the palette: the strip's own length spans the
            # WHOLE palette, and the wave field pushes that position back and
            # forth around it.
            #
            # The earlier form (a weighted blend of position and field) never
            # reached either end — measured travel ran about 0.19 to 0.81 —
            # so with three anchor colours the first one simply never appeared
            # on the strip. Adding the field as a displacement instead keeps
            # full coverage at every instant AND still moves.
            travel = _ring_travel(position, field, drift, len(palette))
            angle = _angle_at(travel, palette)

            if old_palette is not None:
                # 1 well behind the crest, 0 well ahead of it, easing across
                # the crest's own width so there is no hard colour boundary.
                # Bent by the same field as the crest, so the new palette
                # seeps in behind a flowing front rather than a straight edge.
                edge = (wipe_centre - position - _EVENT_FLOW * field) / (SWEEP_WIDTH * 0.9)
                wipe = max(0.0, min(1.0, edge * 0.5 + 0.5))
                wipe = wipe * wipe * (3.0 - 2.0 * wipe)
                if wipe < 0.999:
                    old_travel = _ring_travel(position, field, drift, len(old_palette))
                    angle = _hue_lerp(_angle_at(old_travel, old_palette), angle, wipe)

            # ...and the same field drives perceptual lightness, so crests
            # brighten and troughs deepen. Pacifica's whitecaps, in Oklab.
            L = centre_L + L_swing * field + 0.035 * swell

            # Transient sweeps ride on top: a crest of extra lightness and
            # chroma crossing the strip. This is what makes a moment ARRIVE
            # rather than merely drift.
            boost = 0.0
            glitch = 0.0
            accent = 0.0
            white_mix = 0.0
            flicker_scale = 1.0
            for kind, started_at, strength in (events or []):
                if kind == "sweep":
                    boost += _sweep_contribution(t - started_at, position, strength, field)
                elif kind == "white_sweep":
                    white_mix = max(
                        white_mix,
                        _sweep_contribution(t - started_at, position, strength, field),
                    )
                elif kind == "accent":
                    accent = max(accent,
                                 _accent_contribution(t - started_at, position, strength, field))
                elif kind == "glitch":
                    glitch = max(
                        glitch,
                        _glitch_amount(t - started_at, position, strength, int(started_at * 1000)),
                    )
                elif kind == "flicker":
                    age = t - started_at
                    duration = min(3.0, max(0.0, strength))
                    seed = int(started_at * 1_000_003) & 0xFFFFFFFF
                    start_segment = int(_hash01(seed, 91) * n) % n
                    flicker_width = min(n, 1 + int(_hash01(seed, 92) * min(3, n)))
                    local_offset = (i - start_segment) % n
                    if 0.0 <= age <= duration and local_offset < flicker_width:
                        # Each burst affects only a short, contiguous patch.
                        # Per-segment random holds create the uneven voltage
                        # stutter of a tired neon sign instead of a soft pulse.
                        rate = 5.0 + 10.0 * _hash01(seed, i, 93)
                        slot = int(age * rate)
                        state = _hash01(seed, i, slot, 94)
                        if state < 0.40:
                            target_scale = 0.68  # visible dip without approaching black
                        elif state < 0.67:
                            target_scale = 1.85  # uneven bright sputter
                        else:
                            target_scale = 1.0
                        envelope = math.sin(math.pi * age / max(duration, 0.01)) ** 0.7
                        scale = 1.0 + (target_scale - 1.0) * envelope
                        if abs(scale - 1.0) > abs(flicker_scale - 1.0):
                            flicker_scale = scale

            if accent > 0.0:
                # A contrasting hue travelling the strip. Applied BEFORE the
                # sweep's lightness boost so a sweep landing at the same
                # moment still brightens the accent's own colour rather than
                # washing it out, and left out of the glitch's palette jump
                # above so the two never fight over the same segment.
                mix = min(1.0, accent * 1.25)
                angle = _hue_lerp(angle, (angle + ACCENT_HUE_SHIFT) % 360.0, mix)
                L += (_SWEEP_PEAK_L - L) * (ACCENT_LIFT * mix)

            if glitch > 0.0:
                # Jump elsewhere in the SONG'S palette rather than to an
                # arbitrary colour, and lift the lightness — the light
                # breaking up, not a wrong colour appearing.
                angle = _angle_at((travel + GLITCH_PALETTE_JUMP) % 1.0, palette)
                L += (_SWEEP_PEAK_L - L) * (0.55 * glitch)
            if boost:
                # Pushed toward white-hot rather than simply added, so a strong
                # sweep reads as a surge of light instead of a paler colour.
                L += (_SWEEP_PEAK_L - L) * min(1.0, boost)

            L = max(_AMBIENT_L_MIN, min(0.99, L * flicker_scale))

            # Crests carry slightly more chroma too — a crest that only got
            # lighter would read as washed out rather than vivid.
            # Chroma is taken as a FRACTION of what this segment's own
            # lightness and hue can hold, not from a single ceiling computed
            # for the frame. The gamut ceiling varies strongly with lightness,
            # so one shared ceiling left segments at more permissive lightness
            # visibly under-saturated — measured at 0.47 of what they could
            # have shown. Working in ratios keeps vividness even across the
            # strip, which is what the eye actually reads as consistent.
            ceiling = _smooth_ceiling(L, angle)
            chroma = ceiling * saturation * (0.92 + 0.08 * (field * 0.5 + 0.5))

            radians = math.radians(angle)
            rgb = _oklab_to_rgb_int(L, chroma * math.cos(radians), chroma * math.sin(radians))
            if white_mix > 0.0:
                # White is normally excluded from AI-selected palettes. This
                # explicit transient is the transition crest the user asked
                # for, travelling through the strip rather than filling it.
                amount = min(1.0, white_mix)
                channels = [
                    round(_linear_to_srgb(
                        _srgb_to_linear((rgb >> shift & 0xFF) / 255.0) * (1.0 - amount) + amount
                    ) * 255)
                    for shift in (16, 8, 0)
                ]
                rgb = channels[0] << 16 | channels[1] << 8 | channels[2]
            else:
                rgb = _avoid_white(rgb)
            rgb = _keep_visible(rgb)
            out.append(rgb)
        return out
