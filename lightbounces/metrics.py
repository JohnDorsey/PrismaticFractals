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
    N_POINTS, N_PRED, N_SEQS, N_SRED, N_STATS, POINT_ALL_MEAN, POINT_END, POINT_SLOT, STAT_SLOT, ST_ANGLES, ST_EUCLIDEAN, ST_HDIST, ST_LAST_RATIO, ST_LOOP_AREA, ST_MANHATTAN,
    ST_TERMINATION, ST_VDIST, TERM_CONFINED, TERM_DIST, TERM_PERPENDICULAR
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


# --------------------------------------------------------------------------- #
# What a view can choose: the points and metrics, generated from the named sequences
# --------------------------------------------------------------------------- #
# The ray-bouncing engine's named sequences of points (SEQ_* in kernels.py) and what is stored about each: every
# reduction below of every sequence, per ray. A view chooses among ALL of them, so each choice below is one
# (sequence, reduction) and its label, short name and description are made from the sequence's and the reduction's;
# only the measurements of a ray's own run, which are no reduction of a sequence, are written out by hand. Another
# engine declares its own sequences the same way; every engine should offer SEQ_PATH ("Path": all its points, in
# order), so that a channel set to it means something whichever engine made the figure.
SEQUENCE_LABELS = ["Path", "All bounces", "Boundary bounces", "Self bounces"]
SEQUENCE_SHORT = ["Path", "Bounces", "Boundary", "Self"]
SEQUENCE_DESCRIPTIONS = [
    "every point of the ray's path, in order: its start, each bounce and its end point.",
    "the points where the ray bounced, of any kind (not its start, nor an end point that is not a bounce).",
    "the points where the ray bounced off the edge of a shape (a reflecting edge).",
    "the points where the ray bounced off its own earlier path.",
]
POINT_REDUCTION_LABELS = ["First point", "Last point", "Mean point", "Uppermost point", "Lowermost point",
                          "Rightmost point", "Leftmost point"]
POINT_REDUCTION_SHORT = ["first", "last", "mean", "top", "bottom", "right", "left"]
POINT_REDUCTION_DESCRIPTIONS = [
    "the first point of the sequence.", "the last point of the sequence.",
    "the average of the sequence's points.",
    "the sequence's point with the largest vertical coordinate.",
    "the sequence's point with the smallest vertical coordinate.",
    "the sequence's point with the largest horizontal coordinate.",
    "the sequence's point with the smallest horizontal coordinate.",
]
STAT_REDUCTION_LABELS = ["Point count / limit", "Length (arctan)", "Last segment direction", "Share of the bounces"]
STAT_REDUCTION_SHORT = ["count", "length", "last dir", "share"]
STAT_REDUCTION_DESCRIPTIONS = [
    "how many points the sequence has, divided by the iteration limit (at most 1).",
    "the length of the line through the sequence's points in order, squashed into 0..1 with (2/\u03c0)\u00b7arctan(length/5).",
    "the direction of the last segment of the line through the sequence's points, from -180\u00b0 to 180\u00b0 "
    "mapped to 0..1 (0.5 if it has fewer than two points).",
    "how many points the sequence has, divided by how many bounces the path has (0 if it has none; at most 1).",
]


def point_slot(seq, reduction):
    """The index into a ray's stored points of one reduction (PRED_*) of one named sequence (SEQ_*)."""
    return int(POINT_SLOT[seq, reduction])


def stat_slot(seq, reduction):
    """The index into a ray's stored metrics of one scalar reduction (SRED_*) of one named sequence (SEQ_*)."""
    return int(STAT_SLOT[seq, reduction])


def _slot_owners(slots):
    """{slot: (sequence, reduction)} for a POINT_SLOT / STAT_SLOT table."""
    return {int(slots[seq, red]): (seq, red) for seq in range(slots.shape[0]) for red in range(slots.shape[1])}


POINT_OF_SLOT = _slot_owners(POINT_SLOT)
STAT_OF_SLOT = _slot_owners(STAT_SLOT)

# The measurements of a ray's own run (compute_stats writes them; they are no reduction of a sequence):
# slot -> (label, short name for the ray-preview panel, description)
RUN_STAT_INFO = {
    ST_HDIST: ("Horizontal distance", "H dist",
               "horizontal distance between point 1 and point 2 (|x1 - x2|), divided by 2 and clipped to 0..1."),
    ST_VDIST: ("Vertical distance", "V dist",
               "vertical distance between point 1 and point 2 (|y1 - y2|), divided by 2 and clipped to 0..1."),
    ST_MANHATTAN: ("Manhattan distance", "Manhattan",
                   "Manhattan (taxicab) distance between point 1 and point 2 (|x1-x2| + |y1-y2|), divided by 2 "
                   "and clipped to 0..1."),
    ST_EUCLIDEAN: ("Euclidean distance", "Euclidean",
                   "straight-line (Euclidean) distance between point 1 and point 2, divided by 2 and clipped to 0..1."),
    ST_LOOP_AREA: ("Loop area / same-perimeter circle", "Loop area/circle",
                   "area of the loop formed when the ray first strikes its own path, divided by the area of a circle "
                   "with the same perimeter as the loop (1 = a perfect circle; smaller = more elongated or jagged; 0 "
                   "if it never touches itself)."),
    ST_ANGLES: ("Mean bounce angle / 180\u00b0", "Angle sum/(180\u00b7n)",
                "the average deflection angle over all bounces (the angle in degrees between incoming and outgoing "
                "directions), divided by 180 (i.e. angle sum / (180 \u00d7 bounces))."),
    ST_TERMINATION: ("Termination reason", "Termination reason",
                     "why the ray stopped: 0 = iteration limit, 0.2 = confined to less than a pixel, 0.4 = distance "
                     "limit, 0.6 = reached an edge set to \"terminate\", 0.8 = hit a reflective surface perfectly "
                     "perpendicularly, 1 = escaped (nothing left to hit)."),
    ST_LAST_RATIO: ("Last segment / uncut length", "Last seg / uncut",
                    "length of the last segment divided by the length it would have had without the distance limit "
                    "(1 unless the distance limit cut it short)."),
}


def _sequence_note(seq):
    return f" The sequence is {SEQUENCE_LABELS[seq].lower()}: {SEQUENCE_DESCRIPTIONS[seq]}"


STAT_LABELS = []             # shown in the color-mapping table
STAT_SHORT = []              # compact names for the ray-preview panel (end-point values)
STAT_DESCRIPTIONS = []
for _slot in range(N_STATS):
    if _slot in RUN_STAT_INFO:
        _label, _short, _description = RUN_STAT_INFO[_slot]
    else:
        _seq, _red = STAT_OF_SLOT[_slot]
        _label = f"{SEQUENCE_LABELS[_seq]}: {STAT_REDUCTION_LABELS[_red]}"
        _short = f"{SEQUENCE_SHORT[_seq]} {STAT_REDUCTION_SHORT[_red]}"
        _description = STAT_REDUCTION_DESCRIPTIONS[_red] + _sequence_note(_seq)
    STAT_LABELS.append(_label)
    STAT_SHORT.append(_short)
    STAT_DESCRIPTIONS.append(_description)

# Metrics that compare two sample points (the others ignore the point choices).
# They are evaluated at coloring time from the stored points; the copies stored
# in the metrics array are for the end of the ray and feed the preview list.
POINT_STATS = (ST_HDIST, ST_VDIST, ST_MANHATTAN, ST_EUCLIDEAN)
# every other metric, in the order the "Other" dropdown lists them: the run's own measurements, then each sequence's
OTHER_STATS = (tuple(slot for slot in RUN_STAT_INFO if slot not in POINT_STATS)
               + tuple(stat_slot(seq, red) for seq in range(N_SEQS) for red in range(N_SRED)))

# Two fixed reference points, unrelated to any ray - always the same regardless of
# where the ray went. They are CHOICES only, with no slot of their own (the ids come after the
# stored points' slots), so any code reading a chosen point's coordinates must check
# CONSTANT_POINT_COORDS first.
POINT_ORIGIN = N_POINTS                       # the fixed point (0, 0)
POINT_TOP_LEFT_CORNER = N_POINTS + 1          # the fixed point (-1, 1): the top-left corner of the
                                              # [-1, 1] square the simulation runs in
N_POINT_CHOICES = N_POINTS + 2                # total selectable points (every stored one + 2 fixed)
CONSTANT_POINT_COORDS = {POINT_ORIGIN: (0.0, 0.0), POINT_TOP_LEFT_CORNER: (-1.0, 1.0)}

POINT_LABELS = []
POINT_DESCRIPTIONS = []
POINT_SHORT = []
for _slot in range(N_POINTS):
    _seq, _red = POINT_OF_SLOT[_slot]
    POINT_LABELS.append(f"{SEQUENCE_LABELS[_seq]}: {POINT_REDUCTION_LABELS[_red]}")
    POINT_DESCRIPTIONS.append(POINT_REDUCTION_DESCRIPTIONS[_red] + _sequence_note(_seq))
    POINT_SHORT.append(f"{SEQUENCE_SHORT[_seq]}.{POINT_REDUCTION_SHORT[_red]}")
POINT_LABELS += ["Origin", "Top left corner"]
POINT_DESCRIPTIONS += ["the fixed point (0, 0) - unrelated to the ray, always the same.",
                       "the fixed point (-1, 1), the top-left corner of the simulation's [-1, 1] square - unrelated "
                       "to the ray, always the same."]
POINT_SHORT += ["Origin", "TopLeft"]
POINT_NOTE = (" (Origin and Top left corner don't depend on the ray at all; a sequence the ray has no points in "
              "falls back to the ray's end point.)")

# The order the two point dropdowns list the choices in: the two fixed points
# (which don't depend on the ray at all) first, then each sequence's points. Only the display
# order - a point's id (its index in POINT_LABELS, and into the per-ray stored points for
# every one but the fixed two) does not depend on it, and choosing by label still maps back
# through POINT_LABELS.index().
POINT_MENU_ORDER = [POINT_ORIGIN, POINT_TOP_LEFT_CORNER] + [
    point_slot(seq, red) for seq in range(N_SEQS) for red in range(N_PRED)]
POINT_MENU_LABELS = [POINT_LABELS[i] for i in POINT_MENU_ORDER]

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
#  required termination reason (BUDDHA_SIMPLE only) or None for no filter).
# The simple ones are every point stored for a ray - each reduction of each named sequence, in the order of
# the stored points - followed by the path's last point for the rays a given limit ended; then the multi ones.
BUDDHA_POINT_CATALOG = (
    [(POINT_LABELS[slot], BUDDHA_SIMPLE, slot, None) for slot in range(N_POINTS)]
    + [("Path: Last point (distance limit)", BUDDHA_SIMPLE, POINT_END, TERM_DIST),
       ("Path: Last point (perpendicular collision)", BUDDHA_SIMPLE, POINT_END, TERM_PERPENDICULAR),
       ("Path: Last point (confined)", BUDDHA_SIMPLE, POINT_END, TERM_CONFINED),
       ("All self bounces", BUDDHA_MULTI, BUDDHA_MULTI_ALL_SELF, None),
       ("All segment midpoints", BUDDHA_MULTI, BUDDHA_MULTI_ALL_MID, None),
       ("Progressive means of bounce points", BUDDHA_MULTI, BUDDHA_MULTI_PROG_BOUNCE, None),
       ("Progressive means of segment midpoints", BUDDHA_MULTI, BUDDHA_MULTI_PROG_MID, None)])
N_BUDDHA_POINTS = len(BUDDHA_POINT_CATALOG)
BUDDHA_POINT_LABELS = [c[0] for c in BUDDHA_POINT_CATALOG]
BUDDHA_POINT_SHORT = POINT_SHORT[:N_POINTS] + ["Path.last(dist)", "Path.last(perp)", "Path.last(confined)",
                                               "AllSelf", "AllMid", "ProgBounce", "ProgMid"]
# "Bounces: Mean point" only - the default has always been the mean of all the bounces
DEFAULT_BUDDHA_POINTS = tuple(i == POINT_ALL_MEAN for i in range(N_BUDDHA_POINTS))

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
