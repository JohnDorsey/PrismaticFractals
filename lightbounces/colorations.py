"""Colorations: how the stored simulation data is turned into a coloured image. A coloration is three colour channels (each a Query, Buddha or Constant
source) in an RGB or HSV colour model; this module has the live editable ColorationConfig, its frozen copies (ColorationSnapshot, ColorationSettings),
ChannelSettings (one channel as a keyframe stores it), resolve_colorations (a coloration's values for a video frame) and the colour-model maths."""
import numpy as np
from .kernels import POINT_END, POINT_START, ST_BOUNCES, ST_HDIST, ST_MANHATTAN, ST_VDIST
from .metrics import (
    BUDDHA_COMPRESS_ARCTAN, CHAN_SRC_QUERY, DEFAULT_BUDDHA_BRIGHTNESS, DEFAULT_BUDDHA_POINTS,
    KIND_OTHER, KIND_POINT, NONE_ZERO
)
from .settings import resolve_value, strict_record


# Defaults per channel. Each channel remembers a point-comparison metric (plus
# the two points it compares) and an "other" metric, so switching its Type
# back and forth restores what was chosen.
DEFAULT_KINDS = (KIND_POINT, KIND_POINT, KIND_OTHER)
DEFAULT_POINT_METRICS = (ST_HDIST, ST_VDIST, ST_MANHATTAN)
DEFAULT_OTHER_METRICS = (ST_BOUNCES, ST_BOUNCES, ST_BOUNCES)
DEFAULT_POINT1S = (POINT_START, POINT_START, POINT_START)  # point 1 per channel
DEFAULT_POINT2S = (POINT_END, POINT_END, POINT_END)        # point 2 per channel
DEFAULT_CHAN_SRCS = (CHAN_SRC_QUERY, CHAN_SRC_QUERY, CHAN_SRC_QUERY)   # per channel
DEFAULT_CONSTANT_CHOICES = (NONE_ZERO, NONE_ZERO, NONE_ZERO)          # per channel
DEFAULT_BUDDHA_POINTS_PER_CHAN = (DEFAULT_BUDDHA_POINTS,) * 3
DEFAULT_BUDDHA_COMPRESSIONS = (BUDDHA_COMPRESS_ARCTAN,) * 3
DEFAULT_BUDDHA_BRIGHTNESSES = (DEFAULT_BUDDHA_BRIGHTNESS,) * 3
DEFAULT_MODE = "RGB"

# Color models: channel names (and label colors) for the three table rows
MODES = {
    "RGB": [("Red", "#c62828"), ("Green", "#2e7d32"), ("Blue", "#1565c0")],
    "HSV": [("Hue", "#6a1b9a"), ("Saturation", "#00838f"), ("Value", "#616161")],
}
MODE_NAMES = list(MODES)


# Colorations: showing more than 3 channels' worth of information at once, by
# rendering the same simulation multiple times with different colorations
# and placing the results side by side in an exported image or video. Each
# coloration is exactly what the Colorations tab already controls for a single
# image - a color model plus up to 3 channels - bundled as a ColorationConfig.
#
# self.colorations is one uniform list of ColorationConfig objects. The top-left preview
# shows self.live_coloration() - whichever coloration is currently selected
# (self.selected_coloration_id) in the Colorations tab; selecting a coloration for
# editing is what makes it the preview too, there's no separate step. Every
# coloration uses the same dynamic "coloration:<id>:..." keyframe keys, mirroring
# Shape/self.obstacles and "obst:<id>:<field>" exactly, since the two
# problems are the same shape: a dynamic list of same-structure things,
# each with its own independently keyframeable fields, where the SET of
# things is fixed for a given render but every field of every one of them
# can still be held or animated. Deleting the selected (live) coloration falls
# back to self.colorations[0] - no promotion step and no keyframe-key renaming is
# ever needed, because no coloration's keyframe data is tied to its list position.
class ColorationConfig:
    """One coloration's live state: a color model plus up to 3 channels - exactly
    what the Colorations tab controls. Position in self.colorations has no special
    meaning; which one the top-left preview shows is whichever is selected
    (see live_coloration/select_coloration), not necessarily self.colorations[0]."""
    __slots__ = ("id", "mode", "chan_src", "kinds", "point_metric", "other_metric",
                "point1_choice", "point2_choice", "buddha_points", "buddha_compression",
                "buddha_brightness", "constant_choice")

    def __init__(self, id_):
        self.id = id_
        self.mode = DEFAULT_MODE
        self.chan_src = list(DEFAULT_CHAN_SRCS)
        self.kinds = list(DEFAULT_KINDS)
        self.point_metric = list(DEFAULT_POINT_METRICS)
        self.other_metric = list(DEFAULT_OTHER_METRICS)
        self.point1_choice = list(DEFAULT_POINT1S)
        self.point2_choice = list(DEFAULT_POINT2S)
        self.buddha_points = list(DEFAULT_BUDDHA_POINTS_PER_CHAN)
        self.buddha_compression = list(DEFAULT_BUDDHA_COMPRESSIONS)
        self.buddha_brightness = list(DEFAULT_BUDDHA_BRIGHTNESSES)
        self.constant_choice = list(DEFAULT_CONSTANT_CHOICES)

    def set_channel(self, c, channel):
        """Make channel `c` what the ChannelSettings record `channel` says (a loaded keyframe
        or project)."""
        self.chan_src[c] = channel.chan_src
        self.kinds[c] = channel.kind
        if channel.kind == KIND_POINT:
            self.point_metric[c] = channel.stat
        else:
            self.other_metric[c] = channel.stat
        self.point1_choice[c] = channel.point1
        self.point2_choice[c] = channel.point2
        self.buddha_points[c] = channel.buddha_points
        self.buddha_compression[c] = channel.buddha_compression
        self.buddha_brightness[c] = channel.buddha_brightness
        self.constant_choice[c] = channel.constant

    def settings(self):
        """This coloration as the ColorationSettings the colouring code takes."""
        return ColorationSettings(mode=self.mode, kinds=self.kinds, point_metric=self.point_metric,
                            other_metric=self.other_metric, point1_choice=self.point1_choice,
                            point2_choice=self.point2_choice, chan_src=self.chan_src,
                            buddha_points=self.buddha_points,
                            buddha_compression=self.buddha_compression,
                            buddha_brightness=self.buddha_brightness,
                            constant_choice=self.constant_choice)


@strict_record
class ColorationSnapshot:
    """A frozen copy of one ColorationConfig's fields (see snapshot_coloration). Its fields
    mirror ColorationConfig.__slots__ (a test keeps them in step); the per-channel lists
    are stored as tuples."""
    id: int
    mode: str
    chan_src: tuple
    kinds: tuple
    point_metric: tuple
    other_metric: tuple
    point1_choice: tuple
    point2_choice: tuple
    buddha_points: tuple
    buddha_compression: tuple
    buddha_brightness: tuple
    constant_choice: tuple


def snapshot_coloration(v):
    """A plain, immutable copy of one ColorationConfig's fields - safe to read from
    a background render thread while the GUI thread might still be editing
    the live ColorationConfig object (mirrors Shape.snapshot). ColorationConfig's
    per-channel fields are mutable lists (the UI mutates them in place, e.g.
    self.chan_src[c] = ...), so each one is copied here rather than shared -
    a shared reference would let a background thread see a live edit
    mid-render, exactly what a "snapshot" is meant to prevent (the record stores
    the per-channel lists as tuples)."""
    return ColorationSnapshot(**{f: getattr(v, f) for f in ColorationConfig.__slots__})


@strict_record
class ChannelSettings:
    """One color channel of a coloration, as a keyframe stores it (the value of a
    "coloration:<id>:chan<c>" key). `constant` is a field of its own even though it only
    means something when chan_src is CHAN_SRC_CONSTANT - it is never packed into
    `stat`, whose meaning already depends on `kind`, so reading it never requires
    guessing which of several interpretations currently applies."""
    chan_src: int
    kind: int
    stat: int                  # the point metric if kind is KIND_POINT, else the other metric
    point1: int
    point2: int
    buddha_points: tuple
    buddha_compression: int
    buddha_brightness: int
    constant: int


def coloration_channel_value(v, c):
    """Channel c's current setting for a ColorationConfig or ColorationSnapshot `v`, as
    stored in a keyframe (a ChannelSettings)."""
    stat = v.point_metric[c] if v.kinds[c] == KIND_POINT else v.other_metric[c]
    return ChannelSettings(
        chan_src=v.chan_src[c], kind=v.kinds[c], stat=stat, point1=v.point1_choice[c],
        point2=v.point2_choice[c], buddha_points=v.buddha_points[c],
        buddha_compression=v.buddha_compression[c], buddha_brightness=v.buddha_brightness[c],
        constant=v.constant_choice[c])


# Every field a coloration can define in a keyframe: (field, label, is_float) -
# mirrors the shapes' KEYFRAME_FIELDS. None of these are float (a color model or a whole
# channel's configuration is held between keyframes, never interpolated).
COLORATION_FIELD_INFO = [("mode", "Color model", False), ("chan0", "Channel 1", False),
                   ("chan1", "Channel 2", False), ("chan2", "Channel 3", False)]


def coloration_field_value(v, field):
    """The current value of one of a coloration's COLORATION_FIELD_INFO fields, in the
    same shape a keyframe stores it in."""
    if field == "mode":
        return v.mode
    return coloration_channel_value(v, int(field[-1]))


@strict_record
class ColorationSettings:
    """One coloration, resolved, as apply_coloration reads it (per-channel values
    are 3-tuples)."""
    mode: str
    kinds: tuple
    point_metric: tuple
    other_metric: tuple
    point1_choice: tuple
    point2_choice: tuple
    chan_src: tuple
    buddha_points: tuple
    buddha_compression: tuple
    buddha_brightness: tuple
    constant_choice: tuple


def decompose_chans(mode, chans):
    """A color model plus 3 channels' ChannelSettings (as stored in a keyframe, or
    built by coloration_channel_value) -> a ColorationSettings, decomposed into the separate
    per-channel fields render_full_image's coloring step actually reads. Used by
    resolve_colorations() for every coloration alike."""
    return ColorationSettings(
        mode=mode,
        kinds=tuple(c.kind for c in chans),
        point_metric=tuple(c.stat if c.kind == KIND_POINT else DEFAULT_POINT_METRICS[i]
                           for i, c in enumerate(chans)),
        other_metric=tuple(c.stat if c.kind == KIND_OTHER else DEFAULT_OTHER_METRICS[i]
                           for i, c in enumerate(chans)),
        point1_choice=tuple(c.point1 for c in chans),
        point2_choice=tuple(c.point2 for c in chans),
        chan_src=tuple(c.chan_src for c in chans),
        buddha_points=tuple(c.buddha_points for c in chans),
        buddha_compression=tuple(c.buddha_compression for c in chans),
        buddha_brightness=tuple(c.buddha_brightness for c in chans),
        constant_choice=tuple(c.constant for c in chans))


def resolve_colorations(coloration_snapshots, keyframes_sorted, frame_number):
    """tuple of ColorationSettings, one per coloration, in the same order as
    coloration_snapshots - list/export order, unrelated to which coloration the
    top-left preview happens to be showing (see live_coloration/select_coloration):
    the SET of colorations is fixed (whichever existed when the render started),
    but each one's color model and channels are independently resolved
    (held, never smoothly interpolated) from the keyframes - falling back
    to that coloration's own value (at render-start time) wherever no keyframe
    defines it. Mirrors resolve_obstacle_arrays."""
    result = []
    for snap in coloration_snapshots:
        mode = resolve_value(f"coloration:{snap.id}:mode", keyframes_sorted, frame_number,
                             snap.mode, False)
        chans = [resolve_value(f"coloration:{snap.id}:chan{c}", keyframes_sorted, frame_number,
                               coloration_channel_value(snap, c), False)
                for c in range(3)]
        result.append(decompose_chans(mode, chans))
    return tuple(result)


def hsv_to_rgb_np(h, s, v):
    """Vectorised HSV -> RGB. Inputs/outputs are arrays in 0..1."""
    h6 = (h % 1.0) * 6.0
    fl = np.floor(h6)
    f = h6 - fl
    i = fl.astype(np.int64) % 6
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return r, g, b


def channels_to_rgb(mode, v0, v1, v2):
    """
    Convert three arrays of channel values (0..1) in the given color model
    to red, green, blue arrays on a 0..255 scale.
      RGB: the values are red, green, blue.
      HSV: the values are hue, saturation, value.
    """
    if mode == "HSV":
        r, g, b = hsv_to_rgb_np(v0, v1, v2)
        return r * 255.0, g * 255.0, b * 255.0
    return v0 * 255.0, v1 * 255.0, v2 * 255.0
