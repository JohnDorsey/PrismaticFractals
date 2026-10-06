"""What a view can ask of the stored simulation data and how Buddha coloration scatters it: the per-ray metrics (each scaled to 0..1 so it can
drive a colour component) with their labels, the sample points that can be picked out of a ray's path, the Buddha point catalog,
compression and brightness options, the Buddha blur (the functions that spread scattered points over their neighbours), and make_mapping,
which says what each colour channel is currently using. The numbers the kernels fill in (ST_*, POINT_*, ...) live in kernels.py."""
from math import ceil, pi
import numpy as np
from .constants import (
    BUDDHA_BLUR_BOKEH, BUDDHA_BLUR_CATADIOPTRIC_INNER_RATIO, BUDDHA_BLUR_GAUSSIAN,
    BUDDHA_BLUR_GAUSSIAN_TRUNCATE, BUDDHA_BLUR_TRIANGULAR
)
from .kernels import (
    N_LEGACY_POINTS, N_LEGACY_STATS, POINT_ALL_MEAN, POINT_BOUNDARY_MEAN, POINT_END, POINT_SELF_MEAN,
    POINT_SLOT, STAT_SLOT, ST_EUCLIDEAN, ST_HDIST, ST_MANHATTAN, ST_VDIST, TERM_CONFINED, TERM_DIST,
    TERM_PERPENDICULAR
)


def world_radius_to_pixels(radius_control, out_w, out_h):
    """The blur radius control's value -> a radius in pixels for an out_w x
    out_h output. The control is a multiple of the world view's smallest
    dimension (always 2.0 units on both axes here), and pixels-per-unit is
    min(out_w, out_h) / 2.0 for a same-scale (never-distorted) circular
    blur footprint - the 2.0s cancel, leaving radius_px = radius_control *
    min(out_w, out_h)."""
    return radius_control * min(out_w, out_h)


def build_buddha_blur_kernel(radius_px, blur_type):
    """The (row_offset, col_offset, weight) neighborhood for scattering one
    point's weight of 1 across nearby pixels instead of just its own, as a
    float64 (K, 3) array with weights normalized to sum to 1 - or exactly
    [[0, 0, 1.0]] when radius_px is negligible (no blur at all, the
    original single-pixel behavior). Only Gaussian extends past radius_px
    itself (truncated at BUDDHA_BLUR_GAUSSIAN_TRUNCATE standard
    deviations); the other three kernels are exactly zero beyond it."""
    if radius_px < 0.5:
        return np.array([[0.0, 0.0, 1.0]])
    extent_mult = BUDDHA_BLUR_GAUSSIAN_TRUNCATE if blur_type == BUDDHA_BLUR_GAUSSIAN else 1.0
    extent = int(ceil(extent_mult * radius_px))
    rows, cols = [], []
    for dr in range(-extent, extent + 1):
        for dc in range(-extent, extent + 1):
            rows.append(dr)
            cols.append(dc)
    dr_arr = np.array(rows, dtype=np.float64)
    dc_arr = np.array(cols, dtype=np.float64)
    d = np.hypot(dr_arr, dc_arr)
    if blur_type == BUDDHA_BLUR_GAUSSIAN:
        w = np.exp(-(d * d) / (2.0 * radius_px * radius_px))
        w[d > BUDDHA_BLUR_GAUSSIAN_TRUNCATE * radius_px] = 0.0
    elif blur_type == BUDDHA_BLUR_TRIANGULAR:
        w = np.clip(1.0 - d / radius_px, 0.0, None)
    elif blur_type == BUDDHA_BLUR_BOKEH:
        w = (d <= radius_px).astype(np.float64)
    else:                                              # BUDDHA_BLUR_CATADIOPTRIC
        inner = BUDDHA_BLUR_CATADIOPTRIC_INNER_RATIO * radius_px
        w = ((d >= inner) & (d <= radius_px)).astype(np.float64)
    keep = w > 0.0
    total = w[keep].sum()
    if total <= 0.0:
        return np.array([[0.0, 0.0, 1.0]])      # degenerate (e.g. a sub-pixel ring) ->
                                                # fall back to no blur rather than
                                                # silently dropping the point entirely
    return np.stack([dr_arr[keep], dc_arr[keep], w[keep] / total], axis=1)


def scatter_points_raw(counts, rows, cols, valid):
    """Scatters a plain +1 at each of the given integer pixel positions
    (rows[valid], cols[valid]) into `counts` (in place) - cheap, no blur
    kernel involved at all. Blur (if any) is applied once, afterwards, to
    the whole accumulated histogram - see blur_counts_image - rather than
    per point, since summing many individually-blurred points is
    mathematically identical to blurring the sum once (convolution
    distributes over sums), and the latter costs nothing extra per point
    scattered."""
    np.add.at(counts, (rows[valid], cols[valid]), 1)


def blur_counts_image(counts, kernel):
    """Blurs the whole `counts` array once with `kernel` (the same (K, 3)
    offset+weight table build_buddha_blur_kernel returns), via FFT - so the
    cost depends on the image size, not on the kernel size (a large blur
    radius is no more expensive than a small one), unlike blurring each
    scattered point individually. Returns a new array; never mutates
    `counts`. The no-blur kernel ([[0, 0, 1.0]]) is a fast-path skip."""
    if kernel.shape[0] == 1 and kernel[0, 0] == 0.0 and kernel[0, 1] == 0.0:
        return counts
    h, w = counts.shape
    max_offset = int(np.max(np.abs(kernel[:, :2])))
    ksize = 2 * max_offset + 1
    dense = np.zeros((ksize, ksize))
    for dr, dc, wt in kernel:
        dense[int(dr) + max_offset, int(dc) + max_offset] = wt
    fft_h = h + ksize - 1
    fft_w = w + ksize - 1
    freq_counts = np.fft.rfft2(counts, s=(fft_h, fft_w))
    freq_kernel = np.fft.rfft2(dense, s=(fft_h, fft_w))
    full = np.fft.irfft2(freq_counts * freq_kernel, s=(fft_h, fft_w))
    return full[max_offset:max_offset + h, max_offset:max_offset + w]


# What a view can choose to colour a channel by: N_STAT_CHOICES metrics (the ST_* numbers). A ray's stored metrics (N_STATS in
# kernels.py) hold these first, in the same slots, followed by reductions of the engine's named sequences nothing offers yet.
N_STAT_CHOICES = N_LEGACY_STATS

# Metrics that compare two sample points (the others ignore the point choices).
# They are evaluated at coloring time from the stored points; the copies stored
# in the metrics array are for the end of the ray and feed the preview list.
POINT_STATS = (ST_HDIST, ST_VDIST, ST_MANHATTAN, ST_EUCLIDEAN)
OTHER_STATS = tuple(s for s in range(N_STAT_CHOICES) if s not in POINT_STATS)

STAT_LABELS = [     # shown in the color-mapping table
    "Horizontal distance",
    "Vertical distance",
    "Manhattan distance",
    "Euclidean distance",
    "Bounces / iteration limit",
    "Ray-bounces / bounces",
    "Boundary-bounces / bounces",
    "Loop area / same-perimeter circle",
    "Mean bounce angle / 180\u00b0",
    "Total ray length (arctan)",
    "Last segment direction",
    "Termination reason",
    "Last segment / uncut length",
]
STAT_SHORT = [      # compact names for the ray-preview panel (end-point values)
    "H dist", "V dist", "Manhattan", "Euclidean", "Bounces/limit", "Ray hits/bounces",
    "Boundary hits/bounces", "Loop area/circle", "Angle sum/(180\u00b7n)",
    "Length (arctan)", "Last seg direction", "Termination reason",
    "Last seg / uncut",
]
STAT_DESCRIPTIONS = [
    "horizontal distance between point 1 and point 2 (|x1 - x2|), divided by "
    "2 and clipped to 0..1.",
    "vertical distance between point 1 and point 2 (|y1 - y2|), divided by 2 "
    "and clipped to 0..1.",
    "Manhattan (taxicab) distance between point 1 and point 2 "
    "(|x1-x2| + |y1-y2|), divided by 2 and clipped to 0..1.",
    "straight-line (Euclidean) distance between point 1 and point 2, divided "
    "by 2 and clipped to 0..1.",
    "number of bounces divided by the iteration limit.",
    "bounces off the ray's own earlier path, divided by total bounces "
    "(0 if there were none).",
    "bounces off the edge of a shape (a reflecting edge), divided by total bounces "
    "(0 if there were none).",
    "area of the loop formed when the ray first strikes its own path, "
    "divided by the area of a circle with the same perimeter as the loop "
    "(1 = a perfect circle; smaller = more elongated or jagged; 0 if it "
    "never touches itself).",
    "the average deflection angle over all bounces (the angle in degrees "
    "between incoming and outgoing directions), divided by 180 "
    "(i.e. angle sum / (180 \u00d7 bounces)).",
    "total path length, squashed into 0..1 with (2/\u03c0)\u00b7arctan(length/5).",
    "direction of the final segment, as an angle from -180\u00b0 to 180\u00b0 "
    "(counterclockwise from pointing right) mapped to 0..1: 0.5 = right, "
    "0.75 = up, 0.25 = down, 0 or 1 = left.",
    "why the ray stopped: 0 = iteration limit, 0.2 = confined to less than "
    "a pixel, 0.4 = distance limit, 0.6 = reached an edge set to "
    "\"terminate\", 0.8 = hit a reflective surface perfectly perpendicularly, "
    "1 = escaped (nothing left to hit).",
    "length of the last segment divided by the length it would have had "
    "without the distance limit (1 unless the distance limit cut it short).",
]

# Constant "metrics" a channel can be given instead of a real one (listed among
# the "Other" metrics)
NONE_ZERO = -1
NONE_ONE = -2
CHAN_BUDDHA = -3          # self.mapping[c] sentinel: channel c uses Buddha
                         # coloration, not a Query stat at all

# Dropdown contents for the coloration table
KIND_POINT = 0         # the channel uses a point-comparison metric (+ two points)
KIND_OTHER = 1         # the channel uses any other metric
KIND_LABELS = ["Point comparison", "Other"]
POINT_METRIC_LABELS = [STAT_LABELS[s] for s in POINT_STATS]
OTHER_LABELS = [STAT_LABELS[s] for s in OTHER_STATS]

# Each channel's coloration source: "Query" (the system above - a metric is
# evaluated once per ray, for the pixel it was emitted from), "Buddha" (named
# after the Buddhabrot: every ray's characteristic points are scattered into a
# visit-count histogram, and a pixel's brightness comes from how many times
# ITS OWN location was landed on by any ray's points - see
# BUDDHA_POINT_CATALOG below), or "Constant" (always the same value, taking no
# part in the simulation at all).
CHAN_SRC_QUERY = 0
CHAN_SRC_BUDDHA = 1
CHAN_SRC_CONSTANT = 2
CHAN_SRC_LABELS = ["Query", "Buddha", "Constant"]

# "Constant" source: a channel's value never depends on the simulation at
# all. Stored in the same slot other_metric[c] already occupies (as
# NONE_ZERO/NONE_ONE) rather than a separate field, since only one of the
# two ever applies to a given channel at once.
CHAN_CONSTANT_LABELS = ["Always 0", "Always 1"]


def constant_label_to_stat(label):
    return NONE_ZERO if label == CHAN_CONSTANT_LABELS[0] else NONE_ONE


def constant_stat_to_label(stat):
    return CHAN_CONSTANT_LABELS[0] if stat == NONE_ZERO else CHAN_CONSTANT_LABELS[1]


def other_label_to_stat(label):
    """'Other' dropdown text -> metric index."""
    return OTHER_STATS[OTHER_LABELS.index(label)]


def other_stat_to_label(stat):
    return STAT_LABELS[stat]


def point_comparison_metric(stat, p1x, p1y, p2x, p2y):
    """Value (0..1) of a point-comparison metric between sample points 1 and
    2. Works on scalars or arrays."""
    dx = p1x - p2x
    dy = p1y - p2y
    if stat == ST_HDIST:
        v = np.abs(dx) * 0.5
    elif stat == ST_VDIST:
        v = np.abs(dy) * 0.5
    elif stat == ST_MANHATTAN:
        v = (np.abs(dx) + np.abs(dy)) * 0.5
    else:                                           # ST_EUCLIDEAN
        v = np.hypot(dx, dy) * 0.5
    return np.clip(v, 0.0, 1.0)


# Two fixed reference points, unrelated to any ray - always the same regardless of
# where the ray went. They are CHOICES only, with no slot of their own: the ids below are
# not indexes into a ray's stored points (whose slots 0..N_LEGACY_POINTS - 1 are the other
# choices' ids, and whose later slots, from N_LEGACY_POINTS on, belong to no choice yet), so any
# code reading a chosen point's coordinates must check CONSTANT_POINT_COORDS first.
POINT_ORIGIN = N_LEGACY_POINTS            # the fixed point (0, 0)
POINT_TOP_LEFT_CORNER = N_LEGACY_POINTS + 1   # the fixed point (-1, 1): the top-left corner of the
                              # [-1, 1] square the simulation runs in
N_POINT_CHOICES = N_LEGACY_POINTS + 2     # total selectable points (13 stored + 2 fixed)
CONSTANT_POINT_COORDS = {POINT_ORIGIN: (0.0, 0.0), POINT_TOP_LEFT_CORNER: (-1.0, 1.0)}

POINT_LABELS = [
    "Start point",
    "End point",
    "All bounces mean",
    "Boundary bounces mean",
    "Self bounces mean",
    "Uppermost bounce",
    "Lowermost bounce",
    "Rightmost bounce",
    "Leftmost bounce",
    "Uppermost self bounce",
    "Lowermost self bounce",
    "Rightmost self bounce",
    "Leftmost self bounce",
    "Origin",
    "Top left corner",
]
POINT_DESCRIPTIONS = [
    "the ray's starting point (the pixel it was emitted from).",
    "the last point of the ray (if the distance limit ended it part-way "
    "along a segment, the cut point).",
    "the average of every point where the ray bounced, of any kind.",
    "the average of the points where the ray bounced off the edge of a "
    "shape (a reflecting edge).",
    "the average of the points where the ray bounced off its own earlier "
    "path.",
    "the bounce point with the largest vertical coordinate.",
    "the bounce point with the smallest vertical coordinate.",
    "the bounce point with the largest horizontal coordinate.",
    "the bounce point with the smallest horizontal coordinate.",
    "the bounce point with the largest vertical coordinate, among only "
    "the bounces off the ray's own earlier path.",
    "the bounce point with the smallest vertical coordinate, among only "
    "the bounces off the ray's own earlier path.",
    "the bounce point with the largest horizontal coordinate, among only "
    "the bounces off the ray's own earlier path.",
    "the bounce point with the smallest horizontal coordinate, among only "
    "the bounces off the ray's own earlier path.",
    "the fixed point (0, 0) - unrelated to the ray, always the same.",
    "the fixed point (-1, 1), the top-left corner of the simulation's "
    "[-1, 1] square - unrelated to the ray, always the same.",
]
POINT_NOTE = (" (Start and end always exist, and Origin/Top left corner don't depend "
             "on the ray at all; the rest fall back to the end point if the ray "
             "never bounced that way.)")
POINT_SHORT = ["Start", "End", "AllMean", "BoundMean", "SelfMean",
              "Top", "Bottom", "Right", "Left",
              "SelfTop", "SelfBottom", "SelfRight", "SelfLeft",
              "Origin", "TopLeft"]

# The order the two point dropdowns list the choices in: the two fixed points
# (which don't depend on the ray at all) first, then the points picked out of
# the ray's path. Only the display order - a point's id (its index in
# POINT_LABELS, and into the per-ray sample array for the real ones) never
# changes, and choosing by label still maps back through POINT_LABELS.index().
POINT_MENU_ORDER = [POINT_ORIGIN, POINT_TOP_LEFT_CORNER] + list(range(N_LEGACY_POINTS))
POINT_MENU_LABELS = [POINT_LABELS[i] for i in POINT_MENU_ORDER]

# The ray-bouncing engine's named sequences of points (SEQ_* in kernels.py) and what is stored about each: every
# reduction below of every sequence, per ray. The metrics and points above are aliases of some of these (POINT_ALIASES /
# STAT_ALIASES in kernels.py); the rest are stored but not yet offered to a view. Another engine declares its own
# sequences the same way; every engine should offer SEQ_PATH ("Path": all its points, in order), so that a channel
# set to it means something whichever engine made the figure.
SEQUENCE_LABELS = ["Path", "All bounces", "Boundary bounces", "Self bounces"]
SEQUENCE_DESCRIPTIONS = [
    "every point of the ray's path, in order: its start, each bounce and its end point.",
    "the points where the ray bounced, of any kind (not its start, nor an end point that is not a bounce).",
    "the points where the ray bounced off the edge of a shape (a reflecting edge).",
    "the points where the ray bounced off its own earlier path.",
]
POINT_REDUCTION_LABELS = ["First point", "Last point", "Mean point", "Uppermost point", "Lowermost point",
                          "Rightmost point", "Leftmost point"]
POINT_REDUCTION_DESCRIPTIONS = [
    "the first point of the sequence.", "the last point of the sequence.",
    "the average of the sequence's points.",
    "the sequence's point with the largest vertical coordinate.",
    "the sequence's point with the smallest vertical coordinate.",
    "the sequence's point with the largest horizontal coordinate.",
    "the sequence's point with the smallest horizontal coordinate.",
]
STAT_REDUCTION_LABELS = ["Number of points / iteration limit", "Length (arctan)", "Last segment direction",
                         "Share of the bounces"]
STAT_REDUCTION_DESCRIPTIONS = [
    "how many points the sequence has, divided by the iteration limit (at most 1).",
    "the length of the line through the sequence's points in order, squashed into 0..1 with (2/π)·arctan(length/5).",
    "the direction of the last segment of the line through the sequence's points, from -180° to 180° "
    "mapped to 0..1 (0.5 if it has fewer than two points).",
    "how many points the sequence has, divided by how many bounces the path has (0 if it has none; at most 1).",
]


def point_slot(seq, reduction):
    """The index into a ray's stored points of one reduction (PRED_*) of one named sequence (SEQ_*)."""
    return int(POINT_SLOT[seq, reduction])


def stat_slot(seq, reduction):
    """The index into a ray's stored metrics of one scalar reduction (SRED_*) of one named sequence (SEQ_*)."""
    return int(STAT_SLOT[seq, reduction])


def extract_point_xy(points_arr, point_idx, computed, n):
    """(x, y) arrays of length n holding the chosen point's coordinates for
    every computed pixel: read from `points_arr` (an (H, W, N_POINTS, 2)
    array of the stored points) for a real simulation point, or a constant-filled array for
    POINT_ORIGIN/POINT_TOP_LEFT_CORNER, which was never written into
    `points_arr` at all."""
    if point_idx in CONSTANT_POINT_COORDS:
        cx, cy = CONSTANT_POINT_COORDS[point_idx]
        return np.full(n, cx), np.full(n, cy)
    xy = points_arr[..., point_idx, :][computed].astype(np.float64)
    return xy[:, 0], xy[:, 1]


def resolve_point_xy(sp, point_idx):
    """(x, y) for a single ray's chosen point: read from `sp` (that ray's
    own (N_POINTS, 2) stored points) for a real simulation point, or one of
    the two fixed constant points."""
    if point_idx in CONSTANT_POINT_COORDS:
        return CONSTANT_POINT_COORDS[point_idx]
    return float(sp[point_idx, 0]), float(sp[point_idx, 1])


# Buddha coloration: the points a channel can choose to scatter into its
# visit-count histogram (any number of them, at once). Most are "simple" -
# one point per ray, read straight from the already-computed sample points,
# optionally only for rays that ended with a specific termination reason -
# but the last four are "multi": every one of several points per ray (found
# by re-tracing, since the per-pixel summary doesn't keep the whole path).
BUDDHA_SIMPLE = 0
BUDDHA_MULTI = 1
BUDDHA_MULTI_ALL_SELF = "all_self"           # every bounce off the ray's own path
BUDDHA_MULTI_ALL_MID = "all_mid"             # every segment's midpoint
BUDDHA_MULTI_PROG_BOUNCE = "prog_bounce"     # progressive means of bounce points
BUDDHA_MULTI_PROG_MID = "prog_mid"           # progressive means of segment midpoints

# (label, style, point index (BUDDHA_SIMPLE) or multi-kind (BUDDHA_MULTI),
#  required termination reason (BUDDHA_SIMPLE only) or None for no filter)
BUDDHA_POINT_CATALOG = [
    ("All bounces mean", BUDDHA_SIMPLE, POINT_ALL_MEAN, None),
    ("Self bounces mean", BUDDHA_SIMPLE, POINT_SELF_MEAN, None),
    ("Boundary bounces mean", BUDDHA_SIMPLE, POINT_BOUNDARY_MEAN, None),
    ("Termination point", BUDDHA_SIMPLE, POINT_END, None),
    ("Termination point (distance limit)", BUDDHA_SIMPLE, POINT_END, TERM_DIST),
    ("Termination point (perpendicular collision)", BUDDHA_SIMPLE, POINT_END,
    TERM_PERPENDICULAR),
    ("Termination point (confined)", BUDDHA_SIMPLE, POINT_END, TERM_CONFINED),
    ("All self bounces", BUDDHA_MULTI, BUDDHA_MULTI_ALL_SELF, None),
    ("All segment midpoints", BUDDHA_MULTI, BUDDHA_MULTI_ALL_MID, None),
    ("Progressive means of bounce points", BUDDHA_MULTI, BUDDHA_MULTI_PROG_BOUNCE, None),
    ("Progressive means of segment midpoints", BUDDHA_MULTI, BUDDHA_MULTI_PROG_MID, None),
]
N_BUDDHA_POINTS = len(BUDDHA_POINT_CATALOG)
BUDDHA_POINT_LABELS = [c[0] for c in BUDDHA_POINT_CATALOG]
BUDDHA_POINT_SHORT = ["AllMean", "SelfMean", "BoundMean", "Term", "Term(dist)", "Term(perp)",
                      "Term(confined)", "AllSelf", "AllMid", "ProgBounce", "ProgMid"]
DEFAULT_BUDDHA_POINTS = (True,) + (False,) * (N_BUDDHA_POINTS - 1)   # "All bounces mean" only

# Buddha coloration: how a pixel's (unbounded, non-negative integer) visit
# count maps down to the 0..255 a color channel needs.
BUDDHA_COMPRESS_ARCTAN = 0
BUDDHA_COMPRESS_MODULUS = 1
BUDDHA_COMPRESSION_LABELS = ["Arctangent", "Modulus"]
BUDDHA_ARCTAN_SCALE = 8.0   # count=8 maps to the arctan curve's midpoint (0.5);
                           # a reasonable default giving visible gradation
                           # across small-to-moderate visit counts

# Buddha coloration: an optional final brightness boost, applied (and
# clamped back to 0..1) after compression - useful since compressed values
# often cluster near the low end for sparse histograms.
BUDDHA_BRIGHTNESS_OPTIONS = ["1x", "2x", "4x", "8x", "16x"]
BUDDHA_BRIGHTNESS_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_BUDDHA_BRIGHTNESS = 0    # index into the lists above: "1x", no change


def buddha_compress(count, method, brightness_mult=1.0):
    """A visit count (a non-negative int, or an array of them) -> 0..1."""
    if method == BUDDHA_COMPRESS_ARCTAN:
        v = (2.0 / pi) * np.arctan(count / BUDDHA_ARCTAN_SCALE)
    else:
        v = (np.asarray(count) % 256) / 255.0          # BUDDHA_COMPRESS_MODULUS
    if brightness_mult != 1.0:
        v = np.clip(v * brightness_mult, 0.0, 1.0)
    return v


def make_mapping(kinds, point_metrics, other_metrics, chan_src=None, constant_choices=None):
    """The metric (or NONE_*/CHAN_BUDDHA) each channel is currently using."""
    def one(c):
        if chan_src is not None and chan_src[c] == CHAN_SRC_BUDDHA:
            return CHAN_BUDDHA
        if chan_src is not None and chan_src[c] == CHAN_SRC_CONSTANT:
            if constant_choices is None:
                raise ValueError(
                    f"channel {c} is CHAN_SRC_CONSTANT but make_mapping was called "
                    "without constant_choices")
            return constant_choices[c]
        return point_metrics[c] if kinds[c] == KIND_POINT else other_metrics[c]
    return tuple(one(c) for c in range(3))
