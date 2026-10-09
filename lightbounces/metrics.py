"""What a coloration can ask of the stored simulation data and how Buddha coloration scatters it: the per-ray metrics (each scaled to 0..1 so it can
drive a colour component) with their labels, the sample points that can be picked out of a ray's path, the Buddha point catalog,
compression and brightness options, the Buddha blur (the functions that spread scattered points over their neighbours), and make_mapping,
which says what each colour channel is currently using. The numbers the kernels fill in (ST_*, POINT_*, ...) live in kernels.py."""
import numpy as np
from .kernels import (
    N_POINTS, N_PRED, N_SEQS, N_SRED, N_STATS, POINT_ALL_MEAN, POINT_END, POINT_SLOT, STAT_SLOT,
    ST_ANGLES, ST_EUCLIDEAN, ST_HDIST, ST_LAST_RATIO, ST_LOOP_AREA, ST_MANHATTAN,
    ST_TERMINATION, ST_VDIST, TERM_CONFINED, TERM_DIST, TERM_ESCAPED, TERM_ITER, TERM_OBSTACLE, TERM_PERPENDICULAR,
    DISTANCE_METRIC_DIVISOR, LENGTH_METRIC_DIVISOR, N_TERM
)


# --------------------------------------------------------------------------- #
# What a coloration can choose: the points and metrics, generated from the named sequences
# --------------------------------------------------------------------------- #
# The ray-bouncing engine's named sequences of points (SEQ_* in kernels.py) and what is stored about each: every
# reduction below of every sequence, per ray. A coloration chooses among ALL of them, so each choice below is one
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
    f"the length of the line through the sequence's points in order, squashed into 0..1 with "
    f"(2/\u03c0)\u00b7arctan(length/{LENGTH_METRIC_DIVISOR:g}).",
    "the direction of the last segment of the line through the sequence's points, from -180\u00b0 to 180\u00b0 "
    "mapped to 0..1 (0.5 if it has fewer than two points).",
    "how many points the sequence has, divided by how many bounces the path has (0 if it has none; at most 1).",
]


def point_slot(seq, reduction):
    """The index into a ray's characteristic points of one reduction (PRED_*) of one named sequence (SEQ_*)."""
    return int(POINT_SLOT[seq, reduction])


def stat_slot(seq, reduction):
    """The index into a ray's stored metrics of one scalar reduction (SRED_*) of one named sequence (SEQ_*)."""
    return int(STAT_SLOT[seq, reduction])


def _slot_owners(slots):
    """{slot: (sequence, reduction)} for a POINT_SLOT / STAT_SLOT table."""
    return {int(slots[seq, red]): (seq, red) for seq in range(slots.shape[0]) for red in range(slots.shape[1])}


POINT_OF_SLOT = _slot_owners(POINT_SLOT)
STAT_OF_SLOT = _slot_owners(STAT_SLOT)

# The measurements of a ray's own run (compute_run_measurements writes them; they are no reduction of a sequence):
# slot -> (label, short name for the ray-preview panel, description)
RUN_METRIC_INFO = {
    ST_HDIST: ("Horizontal distance", "H dist",
               f"horizontal distance between point 1 and point 2 (|x1 - x2|), divided by {DISTANCE_METRIC_DIVISOR:g} and "
               "clipped to 0..1."),
    ST_VDIST: ("Vertical distance", "V dist",
               f"vertical distance between point 1 and point 2 (|y1 - y2|), divided by {DISTANCE_METRIC_DIVISOR:g} and "
               "clipped to 0..1."),
    ST_MANHATTAN: ("Manhattan distance", "Manhattan",
                   f"Manhattan (taxicab) distance between point 1 and point 2 (|x1-x2| + |y1-y2|), divided by "
                   f"{DISTANCE_METRIC_DIVISOR:g} and clipped to 0..1."),
    ST_EUCLIDEAN: ("Euclidean distance", "Euclidean",
                   f"straight-line (Euclidean) distance between point 1 and point 2, divided by {DISTANCE_METRIC_DIVISOR:g} "
                   "and clipped to 0..1."),
    ST_LOOP_AREA: ("Loop area / same-perimeter circle", "Loop area/circle",
                   "area of the loop formed when the ray first strikes its own path, divided by the area of a circle "
                   "with the same perimeter as the loop (1 = a perfect circle; smaller = more elongated or jagged; 0 "
                   "if it never touches itself)."),
    ST_ANGLES: ("Mean bounce angle / 180\u00b0", "Angle sum/(180\u00b7n)",
                "the average deflection angle over all bounces (the angle in degrees between incoming and outgoing "
                "directions), divided by 180 (i.e. angle sum / (180 \u00d7 bounces))."),
    ST_TERMINATION: ("Termination reason", "Termination reason",
                     "why the ray terminated: 0 = iteration limit, 0.2 = confined to less than a pixel, 0.4 = distance "
                     "limit, 0.6 = reached an edge set to \"terminate\", 0.8 = hit a reflective surface perfectly "
                     "perpendicularly, 1 = escaped (nothing left to hit)."),
    ST_LAST_RATIO: ("Last segment / uncut length", "Last seg / uncut",
                    "length of the last segment divided by the length it would have had without the distance limit "
                    "(1 unless the distance limit cut it short)."),
}


def _sequence_note(seq):
    return f" The sequence is {SEQUENCE_LABELS[seq].lower()}: {SEQUENCE_DESCRIPTIONS[seq]}"


METRIC_LABELS = []             # shown in the color-mapping table
METRIC_SHORT = []              # compact names for the ray-preview panel (end-point values)
METRIC_DESCRIPTIONS = []
for _slot in range(N_STATS):
    if _slot in RUN_METRIC_INFO:
        _label, _short, _description = RUN_METRIC_INFO[_slot]
    else:
        _seq, _red = STAT_OF_SLOT[_slot]
        _label = f"{SEQUENCE_LABELS[_seq]}: {STAT_REDUCTION_LABELS[_red]}"
        _short = f"{SEQUENCE_SHORT[_seq]} {STAT_REDUCTION_SHORT[_red]}"
        _description = STAT_REDUCTION_DESCRIPTIONS[_red] + _sequence_note(_seq)
    METRIC_LABELS.append(_label)
    METRIC_SHORT.append(_short)
    METRIC_DESCRIPTIONS.append(_description)

# Metrics that compare two sample points (the others ignore the point choices).
# They are evaluated at coloring time from the characteristic points; the copies stored
# in the metrics array are for the end of the ray and feed the preview list.
POINT_METRICS = (ST_HDIST, ST_VDIST, ST_MANHATTAN, ST_EUCLIDEAN)
# every other metric, in the order the "Other" dropdown lists them: the run's own measurements, then each sequence's
OTHER_METRICS = (tuple(slot for slot in RUN_METRIC_INFO if slot not in POINT_METRICS)
               + tuple(stat_slot(seq, red) for seq in range(N_SEQS) for red in range(N_SRED)))

# Three fixed reference points, unrelated to any ray - always the same regardless of
# where the ray went. They are CHOICES only, with no slot of their own (the ids come after the
# characteristic points' slots), so any code reading a chosen point's coordinates must check
# CONSTANT_POINT_COORDS first.
CHOICE_ORIGIN = N_POINTS                       # the fixed point (0, 0)
CHOICE_POINT_NEG1_POS1 = N_POINTS + 1          # the fixed world point (-1, 1): the top-left corner of the
                                              # world the program first showed (not of any framing)
CHOICE_TOP_LEFT_POINT = N_POINTS + 2           # the fixed world point (-2, 2): the top-left corner of the default
                                              # framing (not of the selected one; if what "top left" means ever changes,
                                              # it is a new choice with a new key, as this one is to (-1,1))
N_POINT_CHOICES = N_POINTS + 3                # total selectable points (every stored one + 3 fixed)
CONSTANT_POINT_COORDS = {CHOICE_ORIGIN: (0.0, 0.0), CHOICE_POINT_NEG1_POS1: (-1.0, 1.0),
                         CHOICE_TOP_LEFT_POINT: (-2.0, 2.0)}

POINT_LABELS = []
POINT_DESCRIPTIONS = []
POINT_SHORT = []
for _slot in range(N_POINTS):
    _seq, _red = POINT_OF_SLOT[_slot]
    POINT_LABELS.append(f"{SEQUENCE_LABELS[_seq]}: {POINT_REDUCTION_LABELS[_red]}")
    POINT_DESCRIPTIONS.append(POINT_REDUCTION_DESCRIPTIONS[_red] + _sequence_note(_seq))
    POINT_SHORT.append(f"{SEQUENCE_SHORT[_seq]}.{POINT_REDUCTION_SHORT[_red]}")
POINT_LABELS += ["Origin", "(-1,1)", "Top left point"]
POINT_DESCRIPTIONS += ["the fixed point (0, 0) - unrelated to the ray, always the same.",
                       "the fixed world point (-1, 1), whichever framing is selected - unrelated to the ray, "
                       "always the same.",
                       "the fixed world point (-2, 2), the top-left corner of the default framing (whichever framing "
                       "is selected) - unrelated to the ray, always the same."]
POINT_SHORT += ["Origin", "(-1,1)", "TopLeft"]
POINT_NOTE = (" (Origin, (-1,1) and Top left point don't depend on the ray at all; a sequence the ray has no points in "
              "falls back to the ray's end point.)")

# The order the two point dropdowns list the choices in: the three fixed points
# (which don't depend on the ray at all) first, then each sequence's points. Only the display
# order - a point's id (its index in POINT_LABELS, and into the per-ray characteristic points for
# every one but the fixed two) does not depend on it, and choosing by label still maps back
# through POINT_LABELS.index().
POINT_MENU_ORDER = [CHOICE_ORIGIN, CHOICE_TOP_LEFT_POINT, CHOICE_POINT_NEG1_POS1] + [
    point_slot(seq, red) for seq in range(N_SEQS) for red in range(N_PRED)]
POINT_MENU_LABELS = [POINT_LABELS[i] for i in POINT_MENU_ORDER]

# Constant "metrics" a channel can be given instead of a real one (listed among
# the "Other" metrics)
NONE_ZERO = -1
NONE_ONE = -2
CHAN_BUDDHA = -3          # self.mapping[c] sentinel: channel c uses Buddha
                         # coloration, not a Query metric at all

# Dropdown contents for the coloration table
METRIC_TYPE_POINT_PAIR = 0         # the channel uses a point-comparison metric (+ two points)
METRIC_TYPE_OTHER = 1         # the channel uses any other metric
METRIC_TYPE_LABELS = ["Point comparison", "Other"]
POINT_METRIC_LABELS = [METRIC_LABELS[s] for s in POINT_METRICS]
OTHER_LABELS = [METRIC_LABELS[s] for s in OTHER_METRICS]

# Each channel's coloration source: "Query" (the system above - a metric is
# evaluated once per ray, for the pixel it was fired from), "Buddha" (named
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
# all. Stored in the same slot the channel's metric choice already occupies (as
# NONE_ZERO/NONE_ONE; `ColorationSettings.constant_choice` holds the choice) rather than a
# separate field, since only one of the two ever applies to a given channel at once.
CHAN_CONSTANT_LABELS = ["Always 0", "Always 1"]


def constant_label_to_metric(label):
    return NONE_ZERO if label == CHAN_CONSTANT_LABELS[0] else NONE_ONE


def constant_metric_to_label(metric_slot):
    return CHAN_CONSTANT_LABELS[0] if metric_slot == NONE_ZERO else CHAN_CONSTANT_LABELS[1]


def other_label_to_metric(label):
    """'Other' dropdown text -> metric index."""
    return OTHER_METRICS[OTHER_LABELS.index(label)]


def other_metric_to_label(metric_slot):
    return METRIC_LABELS[metric_slot]


def point_comparison_metric(metric_slot, p1x, p1y, p2x, p2y):
    """Value (0..1) of a point-comparison metric between sample points 1 and
    2. Works on scalars or arrays."""
    dx = p1x - p2x
    dy = p1y - p2y
    if metric_slot == ST_HDIST:
        v = np.abs(dx) / DISTANCE_METRIC_DIVISOR
    elif metric_slot == ST_VDIST:
        v = np.abs(dy) / DISTANCE_METRIC_DIVISOR
    elif metric_slot == ST_MANHATTAN:
        v = (np.abs(dx) + np.abs(dy)) / DISTANCE_METRIC_DIVISOR
    else:                                           # ST_EUCLIDEAN
        v = np.hypot(dx, dy) / DISTANCE_METRIC_DIVISOR
    return np.clip(v, 0.0, 1.0)


def extract_point_xy(characteristic_points, point_idx, computed, n, layout=None):
    """(x, y) arrays of length n holding the chosen point's coordinates for every computed pixel: read from
    `characteristic_points` (an (H, W, K, 2) array of the characteristic points the image keeps: all N_POINTS of them,
    or - when `layout`, a stored_layout.StoredLayout, says which are kept - only those) for a real simulation point, or
    a constant-filled array for CHOICE_ORIGIN/CHOICE_POINT_NEG1_POS1, which was never written into
    `characteristic_points` at all."""
    if point_idx in CONSTANT_POINT_COORDS:
        cx, cy = CONSTANT_POINT_COORDS[point_idx]
        return np.full(n, cx), np.full(n, cy)
    position = point_idx if layout is None else layout.point_position(point_idx)
    xy = characteristic_points[..., position, :][computed].astype(np.float64)
    return xy[:, 0], xy[:, 1]


def resolve_point_xy(ray_characteristic_points, point_idx):
    """(x, y) for a single ray's chosen point: read from `ray_characteristic_points` (that ray's
    own (N_POINTS, 2) characteristic points) for a real simulation point, or one of
    the two fixed constant points."""
    if point_idx in CONSTANT_POINT_COORDS:
        return CONSTANT_POINT_COORDS[point_idx]
    return float(ray_characteristic_points[point_idx, 0]), float(ray_characteristic_points[point_idx, 1])


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
BUDDHA_MULTI_ALL_PATH = "all_path"           # every point of the path, the starting point included

# The three kinds of path a Buddha channel can be limited to (Engine_Refactor_Design.md D16): a path that reached the
# escape radius, one that was terminated by an obstacle, and every other path that ended (the iteration limit, the
# distance limit, confinement - and, for a ray, a perpendicular collision). A channel ticks the kinds it scatters.
PATH_KIND_ESCAPING = 0
PATH_KIND_NON_ESCAPING = 1
PATH_KIND_TERMINATED_BY_OBSTACLE = 2
PATH_KIND_LABELS = ["Escaping", "Non-escaping", "Terminated by obstacle"]
PATH_KIND_SHORT = ["esc", "non-esc", "obst"]
PATH_KIND_OF_TERM = {TERM_ESCAPED: PATH_KIND_ESCAPING, TERM_OBSTACLE: PATH_KIND_TERMINATED_BY_OBSTACLE,
                     TERM_ITER: PATH_KIND_NON_ESCAPING, TERM_DIST: PATH_KIND_NON_ESCAPING,
                     TERM_CONFINED: PATH_KIND_NON_ESCAPING, TERM_PERPENDICULAR: PATH_KIND_NON_ESCAPING}
ALL_PATH_KINDS = (True, True, True)             # every kind ticked: no filter at all (a channel's default)
DEFAULT_BUDDHA_PATH_KINDS = ALL_PATH_KINDS


def term_allowed_flags(kinds):
    """The bool array (one flag per TERM_* code, N_TERM) that says which reasons for a path's ending the path kinds
    `kinds` (a tuple of three flags, in PATH_KIND_* order) allow: what the Buddha kernels take as `term_allowed`."""
    return np.array([bool(kinds[PATH_KIND_OF_TERM[term]]) for term in range(N_TERM)], dtype=np.bool_)


# (label, style, point index (BUDDHA_SIMPLE) or multi-kind (BUDDHA_MULTI),
#  required termination reason (BUDDHA_SIMPLE only) or None for no filter).
# The simple ones are every point stored for a ray - each reduction of each named sequence, in the order of
# the characteristic points - followed by the path's last point for the rays a given limit ended; then the multi ones.
BUDDHA_POINT_CATALOG = (
    [(POINT_LABELS[slot], BUDDHA_SIMPLE, slot, None) for slot in range(N_POINTS)]
    + [("Path: Last point (distance limit)", BUDDHA_SIMPLE, POINT_END, TERM_DIST),
       ("Path: Last point (perpendicular collision)", BUDDHA_SIMPLE, POINT_END, TERM_PERPENDICULAR),
       ("Path: Last point (confined)", BUDDHA_SIMPLE, POINT_END, TERM_CONFINED),
       ("All self bounces", BUDDHA_MULTI, BUDDHA_MULTI_ALL_SELF, None),
       ("All segment midpoints", BUDDHA_MULTI, BUDDHA_MULTI_ALL_MID, None),
       ("Progressive means of bounce points", BUDDHA_MULTI, BUDDHA_MULTI_PROG_BOUNCE, None),
       ("Progressive means of segment midpoints", BUDDHA_MULTI, BUDDHA_MULTI_PROG_MID, None),
       ("All path points", BUDDHA_MULTI, BUDDHA_MULTI_ALL_PATH, None)])
N_BUDDHA_POINTS = len(BUDDHA_POINT_CATALOG)
BUDDHA_POINT_LABELS = [c[0] for c in BUDDHA_POINT_CATALOG]
BUDDHA_POINT_SHORT = POINT_SHORT[:N_POINTS] + ["Path.last(dist)", "Path.last(perp)", "Path.last(confined)",
                                               "AllSelf", "AllMid", "ProgBounce", "ProgMid", "AllPath"]
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


def make_mapping(metric_types, point_metrics, other_metrics, chan_src=None, constant_choices=None):
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
        return point_metrics[c] if metric_types[c] == METRIC_TYPE_POINT_PAIR else other_metrics[c]
    return tuple(one(c) for c in range(3))
