"""The simulation kernels: everything numba compiles, and every constant and type they and the code that calls them agree on.

THE RULE (verified: python dev/refactor/numba_cache_check.py): when numba reuses a cached compile it only checks the
modification time of the file the kernel is defined in. So a kernel keeps running STALE code if a constant it reads, or
another @njit function it calls, is changed in a different file. Therefore this one file defines
  * every @njit function (they all call each other),
  * every constant a kernel reads - and the rest of each numbered family (TERM_*, SHAPE_*, ST_*, POINT_*, SEQ_*, PRED_*,
    SRED_*, RAW_*, KIND_*, OBSTACLE_EFFECT_*, ...) so that a family is never split between files,
  * the kernel argument types (FuzzRule, ConfineRule, EmissionRule, ObstacleArrays, WorldRect and the bundles
    NaturalTerminationRule and RayArgs),
and imports nothing from the rest of the package except numba's njit/prange (decorators, not values a compiled function
freezes in). test_the_numba_kernels_read_only_names_defined_in_their_own_file enforces this. Every other module imports
these names from here (never the other way round)."""
from collections import namedtuple
from math import acos, atan, atan2, cos, floor, pi, sin, sqrt
import numpy as np
from .external import njit, prange


# --------------------------------------------------------------------------- #
# Constants: the kernel ABI (read by the kernels below, used by everything else)
# --------------------------------------------------------------------------- #
PREVIEW_BOX = 512                 # the longest side, in pixels, of the image on screen
DEFAULT_PIXEL_WORLD_WIDTH = 2.0 / PREVIEW_BOX         # width of one pixel in world units for the default framing (it spans [-1, 1])

# Why a ray terminated (returned by trace_path as `term`)
TERM_CONFINED = 0   # confined to less than one pixel
TERM_ITER = 1       # iteration limit reached
TERM_DIST = 2       # distance limit reached
TERM_OBSTACLE = 3   # hit an obstacle whose ray effect is "terminate"
TERM_PERPENDICULAR = 4   # hit a reflective surface at exactly perpendicular
                         # incidence (which side it would retrace along is
                         # undefined - see trace_path)
TERM_ESCAPED = 5         # nothing left ahead to hit: the ray leaves for good (its
                         # last point is where it reaches the escape radius)

# Shapes
SHAPE_ELLIPSE = 0
SHAPE_POLYGON = 1
SHAPE_MOSS_EGG = 2
SHAPE_SECTOR = 3
SHAPE_SEGMENT = 4
SHAPE_LENS = 5
SHAPE_REULEAUX = 6
SHAPE_CAPSULE = 7
SHAPE_TREFOIL = 8           # from here to SHAPE_ARBELOS: arc loops, which can be concave (see inside_arcs_parity)
SHAPE_QUATREFOIL = 9
SHAPE_LUNE = 10
SHAPE_ARBELOS = 11
SHAPE_FIRST_ARC_LOOP = SHAPE_TREFOIL
SHAPE_SUPERELLIPSE = 12     # the two squircles and the spiral ribbon: found by root search, not arcs; they are
SHAPE_FG_SQUIRCLE = 13      # tested in the dispatchers before the arc loops' range check, which would take them
SHAPE_SPIRAL = 14
SHAPE_SNAIL = 15            # the snail of Pythagoras: a union of triangles
SHAPE_EGG = 16              # the Hugelschaffer egg: a cubic
SHAPE_FLOWER = 17           # the sine flower: rho < R (1 + A sin(k phi))

# Ray emission direction (a figure-defining setting, not part of fuzzing): each
# ray's base direction is either straight up (90 degrees) or from its start
# pixel towards a target point; the angle offset, positive meaning more
# counterclockwise, is added to that. Fuzzing's emission jitter is added on top.
# The defaults (Up, offset -45) give the up-and-to-the-right (45 degree)
# direction the program has always used.
EMIT_UP = 0
EMIT_TOWARDS = 1

# Confinement: a ray terminates (TERM_CONFINED) once its recent path shows it has
# stopped getting anywhere. Two things are chosen: how many of the path's most
# recent points to look at (the lookback), and the method that judges them:
#   Box          the lookback's bounding box is smaller than the threshold in BOTH
#                directions
#   Path length  the lookback's segments (one fewer than its points) add up to
#                less than the threshold
# The threshold is in world units (the image spans [-1, 1]), not pixels, so a
# ray behaves the same at every export resolution. The defaults are what was
# always hard-wired: a 4-point lookback, the Box method, and one pixel of the
# 512-pixel preview.
CONFINE_BOX = 0
CONFINE_PATH = 1

EPS = 1e-9
SELF_TIE_EPS = 1e-7        # a self-segment collision must beat a boundary/obstacle
                          # collision by more than this to override it (avoids
                          # floating-point noise letting a self-intersection
                          # "win" a near-tie against a solid wall - see trace_path)
RETRO_EPS = 1e-9          # how close to EXACTLY perpendicular (a ray reflecting
                          # straight back the way it came) counts as "perfectly
                          # perpendicular" - see trace_path
INV_SQRT2 = 1.0 / sqrt(2.0)   # direction (1, 1) / sqrt(2): up and to the right (45 deg)

TAG_START = 0.0           # a path point's tag: which sequences it is in (in_sequence), and the segment graphic's colour
TAG_BOUNDARY = 1.0        # a bounce off a shape's edge
TAG_SELF = 2.0            # a bounce off the ray's own earlier path
TAG_DISTCUT = 3.0
TAG_OBSTACLE_TERM = 4.0   # the point where a "terminate" obstacle ended the ray
TAG_ESCAPED = 5.0         # the point where an escaping ray reaches the escape radius
TAG_PERPENDICULAR = 6.0   # the point where a ray hit a surface at exactly perpendicular incidence and ended (no bounce)

# What the renderer knows about a pixel (the `pixel_state` array of an image, one uint8 each)
PIXEL_PENDING = 0          # not reached yet (the state of every pixel before its row is rendered)
PIXEL_TRACED = 1           # a ray was fired here: its stats and characteristic points are stored
PIXEL_NO_RAY = 2           # reached, but no ray fires here (the pixel is outside every firing shape)

# What happens to a ray as it crosses an obstacle's edge - chosen separately for
# a ray coming in from outside ("incoming") and one going out from inside
# ("outgoing"). Passing through in both directions is as if the obstacle
# weren't there at all.
OBSTACLE_EFFECT_REFLECT = 0
OBSTACLE_EFFECT_TERMINATE = 1
OBSTACLE_EFFECT_PASS = 2

# Every obstacle's data as the parallel arrays the simulation kernels take, bundled in one
# argument (numba reads the fields by name). Build it only with build_obstacle_arrays.
ObstacleArrays = namedtuple("ObstacleArrays", [
    "shapes", "bounds", "cx", "cy", "effects", "fire_inside"])

# The rules the simulation kernels take, as namedtuples (numba reads their fields by
# name - a misspelled one is a compile error - and keeps each field's own type). Build
# them only with fuzz_rule / confine_rule / emission_rule below, and read them in the
# kernels only by field name.
FuzzRule = namedtuple("FuzzRule", [
    "enabled",                                       # bool
    "emit_jitter",                                   # ray emission (added to its direction)
    "bnd_offset", "bnd_jitter", "bnd_mult",          # bounces off a shape's edge
    "self_offset", "self_jitter", "self_mult"])      # bounces off the ray itself
ConfineRule = namedtuple("ConfineRule", [
    "lookback",                                      # how many of the most recent path points are judged (int)
    "method",                                        # CONFINE_BOX / CONFINE_PATH (int)
    "threshold"])                                    # world units
EmissionRule = namedtuple("EmissionRule", [
    "mode",                                          # EMIT_UP / EMIT_TOWARDS (int)
    "target_x", "target_y",                          # the point rays are fired towards
    "offset"])                                       # degrees, positive = counterclockwise

# A rectangle of the world: the rectangle of world coordinates an image covers - or that a grid of seed points is laid
# over. An image's top row is at y_stop. The rectangle (-1, 1, -1, 1), WORLD_RECT_DEFAULT, is what every image covered before
# framings existed (the default framing's). All floats. Build it from a framing with framing_world_rect.
WorldRect = namedtuple("WorldRect", [
    "x_start", "x_stop",                             # left and right edge (world units)
    "y_start", "y_stop"])                            # bottom and top edge
WORLD_RECT_DEFAULT = WorldRect(-1.0, 1.0, -1.0, 1.0)

# What brings ANY sequence of points to its natural end - not something in the sequence's own rules (a terminating
# obstacle, a hit at exactly perpendicular incidence) but that it has gone on long enough, gone far enough, got far
# enough from the origin or stopped getting anywhere: the limits every engine that produces sequences of points
# shares (the ray-bouncing engine, and any other, such as an escape-time iteration). The four limits are exactly
# the termination reasons TERM_ITER, TERM_DIST, TERM_ESCAPED and TERM_CONFINED. Build it only with
# natural_termination_rule / settings_natural_termination; read it in the kernels only by field name.
NaturalTerminationRule = namedtuple("NaturalTerminationRule", [
    "max_iter",                                      # the most points after the first (int)
    "max_dist",                                      # the longest path, in world units (inf: no limit)
    "escape_radius",                                 # how far from the origin a sequence ends, in world units
    "confine"])                                      # a ConfineRule: stopped getting anywhere

# Everything the ray-bouncing engine takes besides the natural termination rule: the part of its arguments that another
# engine would replace with its own. Build it only with ray_args.
RayArgs = namedtuple("RayArgs", [
    "seg_mult",                                      # how much of each bounce's segment is travelled (0..1)
    "fuzz",                                          # a FuzzRule
    "emission",                                      # an EmissionRule
    "obstacles"])                                    # an ObstacleArrays

ST_HDIST = 0        # horizontal distance |x1-x2| between two points  [comparison]
ST_VDIST = 1        # vertical distance |y1-y2| between two points    [comparison]
ST_MANHATTAN = 2    # |x1-x2| + |y1-y2| between two points            [comparison]
ST_EUCLIDEAN = 3    # straight-line distance between two points       [comparison]
ST_BOUNCE_COUNT = 4      # bounces / iteration limit               (linear)
ST_SELF_SHARE = 5     # bounces against ray / bounces           (linear)
ST_BOUNDARY_SHARE = 6  # bounces against the shape / bounces   (linear)
ST_LOOP_AREA = 7    # first self-loop area / same-perimeter circle (linear)
ST_ANGLES = 8       # sum of bounce angles / (180 * bounces)  (linear)
ST_LENGTH = 9       # total ray length                        (arctangent)
ST_LAST_DIR = 10    # direction of the last line segment      (linear angle)
ST_TERMINATION = 11  # why the ray terminated (see TERM_STAT_VALUE)
ST_LAST_RATIO = 12  # last segment length / its uncut length
# (ST_* are the metrics a coloration can choose - see N_STATS below for how many slots a ray's stored metrics really have)

# Sample points that can be picked out of a ray's path
POINT_START = 0          # the ray's starting point (the pixel it was fired from)
POINT_END = 1            # the last point of the ray (a cut point if the distance
                         # limit ended it part-way along a segment)
POINT_ALL_MEAN = 2       # mean of every bounce point, of any kind
POINT_BOUNDARY_MEAN = 3  # mean of bounce points off the boundary (or a reflecting obstacle)
POINT_SELF_MEAN = 4      # mean of bounce points off the ray's own earlier path
POINT_TOP = 5            # bounce point with the largest y (uppermost), any kind
POINT_BOTTOM = 6         # bounce point with the smallest y (lowermost), any kind
POINT_RIGHT = 7          # bounce point with the largest x (rightmost), any kind
POINT_LEFT = 8           # bounce point with the smallest x (leftmost), any kind
POINT_SELF_TOP = 9       # uppermost bounce off the ray's own earlier path
POINT_SELF_BOTTOM = 10   # lowermost bounce off the ray's own earlier path
POINT_SELF_RIGHT = 11    # rightmost bounce off the ray's own earlier path
POINT_SELF_LEFT = 12     # leftmost bounce off the ray's own earlier path
# (POINT_* are the sample points a coloration can choose; N_POINTS below is how many a ray's characteristic points really have)

# --------------------------------------------------------------------------- #
# Named sequences of points: what the ray-bouncing engine offers, and what is stored about each
# --------------------------------------------------------------------------- #
# A traced ray is ONE path of tagged points (trace_path: path_points and path_point_tags). The engine declares named sequences, each a
# filtered view of that path - a membership rule, in_sequence below - and every sequence is reduced to the same fixed
# set of per-pixel numbers (points, and scalars) when the ray is traced, so recolouring stays instant whichever
# sequence a channel is given. Nothing about a reduction is specific to the ray engine: another engine that produces a
# path of points declares its own sequences the same way, and the reductions below work on any of them.
SEQ_PATH = 0        # every point of the path: the start, each bounce and the end point (every engine has this one)
SEQ_BOUNCES = 1     # the bounce points of any kind (tagged TAG_BOUNDARY or TAG_SELF: path_points[1] .. path_points[bounces]; not the start,
                    # nor an end point of another tag - cut, terminating, escaping or perpendicular)
SEQ_BOUNDARY = 2    # the bounce points off a shape's edge (TAG_BOUNDARY)
SEQ_SELF = 3        # the bounce points off the ray's own earlier path (TAG_SELF)
N_SEQS = 4

# What is stored, per sequence, about its points: one (x, y) each. An empty sequence reports the path's end point.
PRED_FIRST = 0      # the first point of the sequence
PRED_LAST = 1       # the last point
PRED_MEAN = 2       # the mean of its points
PRED_TOP = 3        # the point with the largest y (the first such, on a tie)
PRED_BOTTOM = 4     # the point with the smallest y
PRED_RIGHT = 5      # the point with the largest x
PRED_LEFT = 6       # the point with the smallest x
N_PRED = 7

# ... and one number each (0..1), about the polyline through its points taken in order:
SRED_COUNT = 0      # how many points it has / the iteration limit (at most 1)
SRED_LENGTH = 1     # the polyline's length, squashed by (2/pi) arctan(length / 5)
SRED_LAST_DIR = 2   # the direction of its last segment, -180..180 degrees mapped to 0..1 (0.5 if it has under two points)
SRED_SHARE = 3      # how many points it has / how many bounces the path has (0 if none; at most 1)
N_SRED = 4

# The metrics and sample points a coloration could choose before sequences existed are ALIASES: each is one reduction of one
# sequence (a metric the table has no row for - the distances between two points, the fractions, the loop area, the
# bounce angles, the termination reason, the last segment's ratio - is a measurement of the ray's own run, not of a
# sequence, and is still written by compute_run_measurements alone). Their numbers have not changed, and they are the first slots
# of the stored arrays; the rest of the grid follows, in (sequence, reduction) order.
POINT_ALIASES = (                                    # POINT_* number -> (sequence, reduction)
    (SEQ_PATH, PRED_FIRST), (SEQ_PATH, PRED_LAST),
    (SEQ_BOUNCES, PRED_MEAN), (SEQ_BOUNDARY, PRED_MEAN), (SEQ_SELF, PRED_MEAN),
    (SEQ_BOUNCES, PRED_TOP), (SEQ_BOUNCES, PRED_BOTTOM), (SEQ_BOUNCES, PRED_RIGHT), (SEQ_BOUNCES, PRED_LEFT),
    (SEQ_SELF, PRED_TOP), (SEQ_SELF, PRED_BOTTOM), (SEQ_SELF, PRED_RIGHT), (SEQ_SELF, PRED_LEFT))
STAT_ALIASES = {                                     # ST_* number -> (sequence, reduction), where there is one
    ST_BOUNCE_COUNT: (SEQ_BOUNCES, SRED_COUNT), ST_LENGTH: (SEQ_PATH, SRED_LENGTH),
    ST_LAST_DIR: (SEQ_PATH, SRED_LAST_DIR),
    ST_SELF_SHARE: (SEQ_SELF, SRED_SHARE), ST_BOUNDARY_SHARE: (SEQ_BOUNDARY, SRED_SHARE)}
N_LEGACY_POINTS = len(POINT_ALIASES)
N_LEGACY_STATS = 13


def _number_slots(n_seqs, n_reds, legacy_slots, first_free):
    """slot[sequence, reduction] -> the index into a ray's stored array: the aliases keep their old numbers, and
    every other (sequence, reduction) pair takes the next free one in order."""
    slots = np.full((n_seqs, n_reds), -1, dtype=np.int64)
    for slot, (seq, red) in legacy_slots:
        slots[seq, red] = slot
    nxt = first_free
    for seq in range(n_seqs):
        for red in range(n_reds):
            if slots[seq, red] < 0:
                slots[seq, red] = nxt
                nxt += 1
    return slots


POINT_SLOT = _number_slots(N_SEQS, N_PRED, list(enumerate(POINT_ALIASES)), N_LEGACY_POINTS)
STAT_SLOT = _number_slots(N_SEQS, N_SRED, list(STAT_ALIASES.items()), N_LEGACY_STATS)
N_POINTS = N_SEQS * N_PRED            # the points stored for each ray: every reduction of every sequence
N_STATS = N_LEGACY_STATS + N_SEQS * N_SRED - len(STAT_ALIASES)   # the metrics stored: the 13 a coloration can choose (three of
                                                                # them reductions of a sequence), then the other reductions

# Raw measurements filled in by trace_path / compute_run_measurements
RAW_SELF_K = 0      # index of the segment hit at the first self-touch (-1: never)
RAW_SELF_M = 1      # index of the hit point at the first self-touch
RAW_DEFLECT = 2     # sum of deflection angles, degrees
RAW_LENGTH = 3      # total path length
RAW_LAST = 4        # length of the last segment (as drawn)
RAW_AREA = 5        # area of the first self-loop
RAW_LAST_DIR = 6    # direction of the last segment, degrees (-180..180)
RAW_BOUNCES = 7     # number of bounces
RAW_TERM = 8        # why the ray terminated (TERM_*)
RAW_LAST_FULL = 9   # uncut length of a distance-cut last segment (0 if not cut)
RAW_DISP = 10       # straight-line distance from start point to end point
N_RAW = 11

OBST_OBJ_BASE = -100        # best_obj <= this encodes "obstacle index -100 - best_obj"


# --------------------------------------------------------------------------- #
# Simulation kernels
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _circle_roots(px, py, dx, dy, cx, cy, r):
    """Both t roots of the ray hitting circle (cx, cy, r) (near, far); a
    sentinel (1e30, -1e30) if the ray's line misses the circle entirely."""
    lx = px - cx
    ly = py - cy
    qb = lx * dx + ly * dy
    qc = lx * lx + ly * ly - r * r
    disc = qb * qb - qc
    if disc < 0.0:
        return 1e30, -1e30
    sq = sqrt(disc)
    return -qb - sq, -qb + sq


@njit(cache=True)
def _angle_in_range(ang, ang_lo, span):
    d = ang - ang_lo
    while d < 0.0:
        d += 2.0 * pi
    while d >= 2.0 * pi:
        d -= 2.0 * pi
    return d <= span


@njit(cache=True)
def arc_exit_hit(px, py, dx, dy, cx, cy, r, ang_lo, span):
    """Smallest t > EPS where the ray crosses the bounded arc (cx, cy, r,
    ang_lo, span); 1e30 if it never does."""
    t_near, t_far = _circle_roots(px, py, dx, dy, cx, cy, r)
    best = 1e30
    if t_near > EPS:
        hx = px + t_near * dx
        hy = py + t_near * dy
        if _angle_in_range(atan2(hy - cy, hx - cx), ang_lo, span):
            best = t_near
    if t_far > EPS and t_far < best:
        hx = px + t_far * dx
        hy = py + t_far * dy
        if _angle_in_range(atan2(hy - cy, hx - cx), ang_lo, span):
            best = t_far
    return best


@njit(cache=True)
def edge_exit_hit(px, py, dx, dy, ax, ay, bx, by):
    """Smallest t > EPS where the ray crosses the bounded segment a->b; 1e30
    if it never does."""
    ex = bx - ax
    ey = by - ay
    denom = dx * ey - dy * ex
    if -1e-14 < denom < 1e-14:
        return 1e30
    wx = ax - px
    wy = ay - py
    t = (wx * ey - wy * ex) / denom
    if t <= EPS:
        return 1e30
    u = (wx * dy - wy * dx) / denom
    if u < 0.0 or u > 1.0:
        return 1e30
    return t


@njit(cache=True)
def arcs_edges_hit(px, py, dx, dy, bound):
    """The nearest point (t > EPS) where a ray touches the actual boundary of
    an "arcs + edges" shape (Moss's egg / circular sector / circular segment /
    capsule, and the arc loops) - works identically whether the ray starts inside
    (this is then its exit) or outside (this is then its entry), since the
    nearest crossing of the boundary is always the way out of, or into, the shape
    (for the concave arc loops too)."""
    n_a = int(bound[1])
    n_e = int(bound[2])
    best_t = 1e30
    best_nx = 0.0
    best_ny = 0.0
    idx = 3
    for _i in range(n_a):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        ang_lo = bound[idx + 3]
        span = bound[idx + 4]
        idx += 5
        t = arc_exit_hit(px, py, dx, dy, cx, cy, r, ang_lo, span)
        if t < best_t:
            best_t = t
            best_nx = (px + t * dx) - cx
            best_ny = (py + t * dy) - cy
    for _i in range(n_e):
        ax = bound[idx]
        ay = bound[idx + 1]
        bx = bound[idx + 2]
        by = bound[idx + 3]
        idx += 4
        t = edge_exit_hit(px, py, dx, dy, ax, ay, bx, by)
        if t < best_t:
            best_t = t
            best_nx = by - ay
            best_ny = -(bx - ax)
    return best_t, best_nx, best_ny


@njit(cache=True)
def inside_arcs_edges_and(x, y, bound):
    """Is (x, y) inside an "arcs + edges" shape that IS exactly the
    intersection of its arcs' full circles and its edges' half-planes
    (circular sector, circular segment)? True iff within range of every arc's
    circle and on the interior side of every edge."""
    n_a = int(bound[1])
    n_e = int(bound[2])
    idx = 3
    for _i in range(n_a):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        idx += 5
        if (x - cx) ** 2 + (y - cy) ** 2 > r * r:
            return False
    for _i in range(n_e):
        ax = bound[idx]
        ay = bound[idx + 1]
        bx = bound[idx + 2]
        by = bound[idx + 3]
        idx += 4
        if (bx - ax) * (y - ay) - (by - ay) * (x - ax) < 0.0:
            return False
    return True


@njit(cache=True)
def inside_arcs_edges_star(x, y, bound):
    """Is (x, y) inside an "arcs + edges" shape whose local origin is always
    strictly interior (Moss's egg, capsule) but whose arcs only cover part of
    their own circles, so the AND-of-full-primitives test above doesn't apply?
    Cast a ray from the origin through (x, y) and compare distances - valid
    because the shape is convex and genuinely star-shaped from the origin."""
    dist = sqrt(x * x + y * y)
    if dist < 1e-9:
        return True
    ux = x / dist
    uy = y / dist
    t_exit, _nx, _ny = arcs_edges_hit(0.0, 0.0, ux, uy, bound)
    return dist <= t_exit


PARITY_DX = cos(0.3719)     # the fixed, generic direction inside_arcs_parity casts along
PARITY_DY = sin(0.3719)     # (an odd angle, so it almost never runs along an edge)


@njit(cache=True)
def inside_arcs_parity(x, y, bound):
    """Is (x, y) inside an "arc loops" shape? These can be concave, so neither the
    AND of their circles nor a ray from the origin works; use the even-odd rule
    instead: a ray from the point crosses the boundary an odd number of times
    exactly when the point is inside."""
    n_a = int(bound[1])
    n_e = int(bound[2])
    count = 0
    idx = 3
    for _i in range(n_a):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        ang_lo = bound[idx + 3]
        span = bound[idx + 4]
        idx += 5
        t_near, t_far = _circle_roots(x, y, PARITY_DX, PARITY_DY, cx, cy, r)
        if EPS < t_near < 1e29 and _angle_in_range(
                atan2(y + t_near * PARITY_DY - cy, x + t_near * PARITY_DX - cx), ang_lo, span):
            count += 1
        if t_far > EPS and _angle_in_range(
                atan2(y + t_far * PARITY_DY - cy, x + t_far * PARITY_DX - cx), ang_lo, span):
            count += 1
    for _i in range(n_e):
        ax = bound[idx]
        ay = bound[idx + 1]
        bx = bound[idx + 2]
        by = bound[idx + 3]
        idx += 4
        if edge_exit_hit(x, y, PARITY_DX, PARITY_DY, ax, ay, bx, by) < 1e29:
            count += 1
    return count % 2 == 1


@njit(cache=True)
def inside_circles(x, y, bound):
    """Is (x, y) inside every one of a "circle intersection" shape's circles
    (two-circle lens / Reuleaux polygon)?"""
    n = int(bound[1])
    idx = 2
    for _i in range(n):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        idx += 3
        if (x - cx) ** 2 + (y - cy) ** 2 > r * r:
            return False
    return True


@njit(cache=True)
def circles_exit_hit(px, py, dx, dy, bound):
    """boundary_hit for the "circle intersection" family, from inside: the
    exit is the smallest of every circle's own far root."""
    n = int(bound[1])
    idx = 2
    best_t = 1e30
    best_nx = 0.0
    best_ny = 0.0
    for _i in range(n):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        idx += 3
        _t_near, t_far = _circle_roots(px, py, dx, dy, cx, cy, r)
        if t_far < best_t:
            best_t = t_far
            best_nx = (px + t_far * dx) - cx
            best_ny = (py + t_far * dy) - cy
    return best_t, best_nx, best_ny


@njit(cache=True)
def circles_entry_hit(px, py, dx, dy, bound):
    """obstacle_hit for the "circle intersection" family, from outside: the
    entry/exit of the intersection are the latest near-root / earliest
    far-root across all circles (Cyrus-Beck-style interval intersection)."""
    n = int(bound[1])
    idx = 2
    t_enter = -1e30
    t_exit = 1e30
    enter_nx = 0.0
    enter_ny = 0.0
    for _i in range(n):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        idx += 3
        t_near, t_far = _circle_roots(px, py, dx, dy, cx, cy, r)
        if t_near > 1e29:                       # this circle's line misses entirely
            return -1.0, 0.0, 0.0
        if t_near > t_enter:
            t_enter = t_near
            enter_nx = (px + t_near * dx) - cx
            enter_ny = (py + t_near * dy) - cy
        if t_far < t_exit:
            t_exit = t_far
    if t_enter <= EPS or t_enter > t_exit:
        return -1.0, 0.0, 0.0
    return t_enter, enter_nx, enter_ny


# --------------------------------------------------------------------------- #
# The squircles and the Archimedean spiral ribbon. Their ray hits are found by root search along the ray,
# not in closed form. Both are worked out in the shape's own frame: bound[1], bound[2] are the cosine and sine
# of its rotation (a point or direction is turned by minus the rotation, a normal turned back), then
#   superelliptical squircle  bound[3], bound[4] = the semi-axes a, b;  bound[5] = the exponent n
#   Fernandez-Guasti squircle bound[3] = bound[4] = the radius r;       bound[5] = the squareness s
#   spiral ribbon             bound[3] = b (radius gained per radian),  bound[4] = the ribbon's width w,
#                             bound[5] = theta_max, the angle the ribbon runs to (0: an empty shape)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def squircle_f(lx, ly, shape, bound):
    """For the two squircles: F(lx, ly) and its gradient, in the shape's own frame. F < 0 inside the shape, > 0
    outside, and F is convex along every straight line (the shapes are convex, and F is their gauge or a sum of
    convex terms), so along a ray F(t) has at most two roots, and Newton's method started from the left of the
    first (or the right of the last) one creeps up on it without ever jumping past it."""
    if shape == SHAPE_SUPERELLIPSE:
        a = bound[3]
        b = bound[4]
        n = bound[5]
        u = abs(lx) / a
        v = abs(ly) / b
        gx = n * u ** (n - 1.0) / a
        gy = n * v ** (n - 1.0) / b
        if lx < 0.0:
            gx = -gx
        elif lx == 0.0:
            gx = 0.0
        if ly < 0.0:
            gy = -gy
        elif ly == 0.0:
            gy = 0.0
        return u ** n + v ** n - 1.0, gx, gy
    # Fernandez-Guasti: with q = x^2 + y^2, F = (q + sqrt(q^2 - 4 s^2 x^2 y^2)) / (2 r^2) - 1, which is (the
    # gauge of the shape)^2 - 1 (see squircle_hit: it is where x^2 + y^2 - s^2 x^2 y^2 / r^2 = r^2 along the
    # radial line through the point, taking the nearer of the two roots)
    r2 = bound[3] * bound[3]
    s2 = bound[5] * bound[5]
    x2 = lx * lx
    y2 = ly * ly
    q = x2 + y2
    root = sqrt(max(q * q - 4.0 * s2 * x2 * y2, 0.0))
    if root > 1e-12:
        drx = 2.0 * lx * (q - 2.0 * s2 * y2) / root
        dry = 2.0 * ly * (q - 2.0 * s2 * x2) / root
    else:
        drx = 0.0       # the corner of the s = 1 square: any subgradient will do for Newton's method
        dry = 0.0
    return (q + root) / (2.0 * r2) - 1.0, (2.0 * lx + drx) / (2.0 * r2), (2.0 * ly + dry) / (2.0 * r2)


@njit(cache=True)
def _slab(p, d, half):
    """Where the line p + t d lies between the planes at -half and +half: (t_near, t_far), or (1e30, -1e30)
    if it never does."""
    if abs(d) < 1e-15:
        if abs(p) > half:
            return 1e30, -1e30
        return -1e30, 1e30
    t1 = (-half - p) / d
    t2 = (half - p) / d
    if t1 > t2:
        return t2, t1
    return t1, t2


@njit(cache=True)
def squircle_hit(px, py, dx, dy, shape, bound, entering):
    """boundary_hit (entering False: the ray starts inside, this is where it leaves) and obstacle_hit (entering
    True: it starts outside, this is where it enters, or t < 0 if it misses) for the two squircles. The shape
    lies inside the box of its semi-axes, so the ray is first clipped to that box, and Newton's method starts
    at the box's end of the ray (see squircle_f for why it cannot overshoot)."""
    c = bound[1]
    s = bound[2]
    lx = c * px + s * py
    ly = -s * px + c * py
    ldx = c * dx + s * dy
    ldy = -s * dx + c * dy
    a_near, a_far = _slab(lx, ldx, bound[3] * (1.0 + 1e-12) + 1e-12)
    b_near, b_far = _slab(ly, ldy, bound[4] * (1.0 + 1e-12) + 1e-12)
    t_near = max(a_near, b_near)
    t_far = min(a_far, b_far)
    if t_near > t_far or t_far <= 0.0:
        if entering:
            return -1.0, 0.0, 0.0
        return 1e30, 0.0, 0.0
    if entering:
        t = max(t_near, 0.0)
        for _i in range(60):
            f, gx, gy = squircle_f(lx + t * ldx, ly + t * ldy, shape, bound)
            if f <= 1e-14:
                break
            slope = gx * ldx + gy * ldy
            if slope >= -1e-300:                 # F is above 0 and not falling: the ray never gets there
                return -1.0, 0.0, 0.0
            t -= f / slope
            if t > t_far:
                return -1.0, 0.0, 0.0
        f, gx, gy = squircle_f(lx + t * ldx, ly + t * ldy, shape, bound)
        if f > 1e-9 or t <= EPS:                 # a graze that never quite arrived, or the ray's own start
            return -1.0, 0.0, 0.0
    else:
        t = t_far
        for _i in range(60):
            f, gx, gy = squircle_f(lx + t * ldx, ly + t * ldy, shape, bound)
            if f <= 1e-14:
                break
            slope = gx * ldx + gy * ldy
            if slope <= 1e-300:
                break
            t -= f / slope
            if t < 0.0:
                t = 0.0
                break
        if t < 0.0:
            t = 0.0
        f, gx, gy = squircle_f(lx + t * ldx, ly + t * ldy, shape, bound)
    return t, c * gx - s * gy, s * gx + c * gy


@njit(cache=True)
def inside_squircle(x, y, shape, bound):
    c = bound[1]
    s = bound[2]
    f, _gx, _gy = squircle_f(c * x + s * y, -s * x + c * y, shape, bound)
    return f < 0.0


@njit(cache=True)
def inside_spiral(x, y, bound):
    """Is (x, y) inside the spiral ribbon: the points (rho, phi) with  b*theta <= rho < b*theta + w  for some
    theta between 0 and theta_max that is phi plus a whole number of turns? Of the thetas that differ from phi
    by whole turns only the largest one with b*theta <= rho can work (the ribbon is under a turn wide)."""
    theta_max = bound[5]
    if theta_max <= 1e-9:
        return False
    c = bound[1]
    s = bound[2]
    lx = c * x + s * y
    ly = -s * x + c * y
    b = bound[3]
    w = bound[4]
    rho = sqrt(lx * lx + ly * ly)
    if rho >= b * theta_max + w:
        return False
    two_pi = 2.0 * pi
    phi = atan2(ly, lx)
    theta = phi + two_pi * floor((rho / b - phi) / two_pi)
    if theta < 0.0 or theta > theta_max:
        return False
    return rho - b * theta < w


@njit(cache=True)
def spiral_d(t, m, c, pp, phi0, inv_b):
    """D(t) = rho(t)/b - phi(t) along a ray, where phi is the polar angle followed continuously (a straight line
    not through the centre sweeps under a half turn in all). The ray is on the inner edge of the ribbon where D is
    a whole number of turns (2 pi k) and on the outer edge where D is k turns plus 2 pi * fill."""
    s = t + m
    return sqrt(s * s + c * c) * inv_b - (phi0 + atan2(t * c, pp + t * m))


@njit(cache=True)
def spiral_slope(t, m, c, inv_b):
    s = t + m
    rr = s * s + c * c
    if rr < 1e-24:
        return 0.0
    return s * inv_b / sqrt(rr) - c / rr


@njit(cache=True)
def spiral_solve(level, lo, hi, d_lo, d_hi, m, c, pp, phi0, inv_b):
    """The t in [lo, hi] where D(t) = level, D being monotone there and d_lo, d_hi (D less the level at each end)
    of opposite sign: Newton's method kept inside the shrinking bracket, falling back to bisection."""
    if d_lo == 0.0:
        return lo
    if d_hi == 0.0:
        return hi
    t = 0.5 * (lo + hi)
    for _i in range(100):
        f = spiral_d(t, m, c, pp, phi0, inv_b) - level
        if f == 0.0:
            return t
        if (f > 0.0) == (d_lo > 0.0):
            lo = t
            d_lo = f
        else:
            hi = t
            d_hi = f
        slope = spiral_slope(t, m, c, inv_b)
        t_new = t - f / slope if slope != 0.0 else 0.5 * (lo + hi)
        if t_new <= lo or t_new >= hi:
            t_new = 0.5 * (lo + hi)
        if abs(t_new - t) <= 1e-15 * (1.0 + abs(t)):
            return t_new
        t = t_new
    return t


@njit(cache=True)
def spiral_edges_hit(lx, ly, ldx, ldy, b, w, theta_max):
    """The nearest t > EPS where a ray, in the ribbon's own frame, meets one of its two curved edges
    (rho = b*theta and rho = b*theta + w, theta from 0 to theta_max), with the normal there; 1e30 if it never does.

    D(t) of spiral_d has exactly one turning point, found in closed form (a quartic in s that is quadratic in s^2),
    so the ray splits into a stretch where D falls and a stretch where it rises, and on each of those the edges are
    the levels D = 2 pi k and D = 2 pi (k + fill) crossed in order. Each level is solved by spiral_solve and is
    accepted if its theta is within the ribbon's range; the first accepted one is the answer."""
    two_pi = 2.0 * pi
    r_out = b * theta_max + w
    t_in, t_out = _circle_roots(lx, ly, ldx, ldy, 0.0, 0.0, r_out * (1.0 + 1e-9))
    t0 = max(t_in, 0.0)
    t1 = t_out
    if t1 <= t0:
        return 1e30, 0.0, 0.0
    inv_b = 1.0 / b
    fill_turn = w * inv_b                       # 2 pi * fill: the ribbon's width, in radians of D
    pp = lx * lx + ly * ly
    m = lx * ldx + ly * ldy
    c = lx * ldy - ly * ldx
    phi0 = atan2(ly, lx) if pp > 1e-18 else atan2(ldy, ldx)
    hh = c * c
    s_turn = sqrt(0.5 * (-hh + sqrt(hh * hh + 4.0 * b * b * hh)))
    t_min = (s_turn if c >= 0.0 else -s_turn) - m       # where D stops falling and starts rising
    mid = min(max(t_min, t0), t1)
    for piece in range(2):
        falling = piece == 0
        u0 = t0 if falling else mid
        u1 = mid if falling else t1
        if u1 <= u0:
            continue
        d0 = spiral_d(u0, m, c, pp, phi0, inv_b)
        d1 = spiral_d(u1, m, c, pp, phi0, inv_b)
        k = floor(d0 / two_pi)
        rem = d0 - two_pi * k
        if falling:                             # the highest level at or below D(u0), then downwards
            n = int(2 * k) + (1 if rem >= fill_turn else 0)
        elif rem <= 0.0:                        # the lowest level at or above D(u0), then upwards
            n = int(2 * k)
        elif rem <= fill_turn:
            n = int(2 * k) + 1
        else:
            n = int(2 * k) + 2
        for _i in range(20000):
            level = two_pi * (n // 2) + (fill_turn if n % 2 == 1 else 0.0)
            if falling:
                if level < d1 - 1e-12:
                    break
            elif level > d1 + 1e-12:
                break
            t = spiral_solve(level, u0, u1, d0 - level, d1 - level, m, c, pp, phi0, inv_b)
            n_hit = n
            n += -1 if falling else 1
            if t <= EPS:
                continue
            x = lx + t * ldx
            y = ly + t * ldy
            rho = sqrt(x * x + y * y)
            if n_hit % 2 == 0:                              # an inner edge: rho = b * theta
                ok = rho <= b * theta_max * (1.0 + 1e-12) + 1e-12
            else:                                           # an outer edge: rho = b * theta + w
                ok = rho >= w * (1.0 - 1e-12) - 1e-12 and rho <= r_out * (1.0 + 1e-12)
            if ok:
                rho = max(rho, 1e-12)
                return t, x + b * y / rho, y - b * x / rho
    return 1e30, 0.0, 0.0


@njit(cache=True)
def spiral_hit(px, py, dx, dy, bound):
    """The nearest t > EPS where a ray touches the spiral ribbon's edge (its two curved edges and the two straight
    ends), from inside or outside alike, with the normal there; 1e30 if it never does."""
    theta_max = bound[5]
    if theta_max <= 1e-9:
        return 1e30, 0.0, 0.0
    c = bound[1]
    s = bound[2]
    b = bound[3]
    w = bound[4]
    lx = c * px + s * py
    ly = -s * px + c * py
    ldx = c * dx + s * dy
    ldy = -s * dx + c * dy
    best_t, nx, ny = spiral_edges_hit(lx, ly, ldx, ldy, b, w, theta_max)
    t = edge_exit_hit(lx, ly, ldx, ldy, 0.0, 0.0, w, 0.0)             # the end the ribbon starts from
    if t < best_t:
        best_t = t
        nx = 0.0
        ny = -w
    cos_t = cos(theta_max)
    sin_t = sin(theta_max)
    ax = b * theta_max * cos_t
    ay = b * theta_max * sin_t
    bx = (b * theta_max + w) * cos_t
    by = (b * theta_max + w) * sin_t
    t = edge_exit_hit(lx, ly, ldx, ldy, ax, ay, bx, by)               # the end it runs out at
    if t < best_t:
        best_t = t
        nx = by - ay
        ny = -(bx - ax)
    return best_t, c * nx - s * ny, s * nx + c * ny


@njit(cache=True)
def squircle_distance_grid(gx, gy, shape, bound):
    """Over a whole grid: the first-order distance |F| / |grad F| from each point to a squircle's edge (for the
    outline and the two ring layers; accurate close to the edge, which is all they need)."""
    c = bound[1]
    s = bound[2]
    h, w = gx.shape
    out = np.empty((h, w), dtype=np.float64)
    for j in range(h):
        for i in range(w):
            f, fx, fy = squircle_f(c * gx[j, i] + s * gy[j, i], -s * gx[j, i] + c * gy[j, i],
                                   shape, bound)
            out[j, i] = abs(f) / max(sqrt(fx * fx + fy * fy), 1e-9)
    return out


@njit(cache=True)
def _segment_distance(x, y, ax, ay, bx, by):
    ex = bx - ax
    ey = by - ay
    along = ((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey)
    along = min(max(along, 0.0), 1.0)
    return sqrt((x - ax - along * ex) ** 2 + (y - ay - along * ey) ** 2)


@njit(cache=True)
def spiral_distance_grid(gx, gy, bound):
    """Over a whole grid: the distance from each point to the spiral ribbon's edge - first order for the two
    curved edges (the offset in rho from the nearest turn, over the size of rho - b*phi's gradient), exact for the
    two straight ends - for the outline's two ring layers."""
    h, w_px = gx.shape
    out = np.full((h, w_px), 1e30)
    theta_max = bound[5]
    if theta_max <= 1e-9:
        return out
    c = bound[1]
    s = bound[2]
    b = bound[3]
    w = bound[4]
    two_pi = 2.0 * pi
    ax = b * theta_max * cos(theta_max)
    ay = b * theta_max * sin(theta_max)
    bx = (b * theta_max + w) * cos(theta_max)
    by = (b * theta_max + w) * sin(theta_max)
    for j in range(h):
        for i in range(w_px):
            lx = c * gx[j, i] + s * gy[j, i]
            ly = -s * gx[j, i] + c * gy[j, i]
            best = min(_segment_distance(lx, ly, 0.0, 0.0, w, 0.0),
                       _segment_distance(lx, ly, ax, ay, bx, by))
            rho = sqrt(lx * lx + ly * ly)
            if rho > 1e-12:
                phi = atan2(ly, lx)
                base = phi + two_pi * floor((rho / b - phi) / two_pi)
                slope = sqrt(1.0 + (b / rho) ** 2)
                for k in range(-1, 2):
                    theta = base + two_pi * k
                    if theta < 0.0 or theta > theta_max:
                        continue
                    best = min(best, abs(rho - b * theta) / slope, abs(rho - b * theta - w) / slope)
            out[j, i] = best
    return out


# --------------------------------------------------------------------------- #
# The snail of Pythagoras (the spiral of Theodorus): the union of n right triangles that share the vertex O at the
# centre; triangle k has the corners O, A_k, A_{k+1}, with the right angle at A_k, |O A_k| = sqrt(k) and
# |A_k A_{k+1}| = 1 (both scaled). Its bound is [area, n, the scale, then the n + 1 corners A_1 .. A_{n+1} as x, y
# pairs, already turned by the rotation]. Each triangle is convex, so a ray meets it in one interval of t; the
# shape is the union of those intervals' triangles, which is star-shaped about O but not convex, so the ray's way
# out is found by chaining the intervals that touch, and its way in is the earliest interval start.
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _snail_clip(px, py, dx, dy, ax, ay, bx, by):
    """The ray p + t d against the triangle O, A, B (counter-clockwise): (ok, t_in, t_out, the normal where it
    enters, the normal where it leaves). Inside is where all three of cross(A, p), cross(p, B) and
    cross(B - A, p - A) are positive; each is a linear function of t. The two straight edges out from O count as
    inside (a point on one shared by two triangles is in both), so the shape has no hairline hole along them."""
    lo = -1e30
    hi = 1e30
    lnx = 0.0
    lny = 0.0
    hnx = 0.0
    hny = 0.0
    ok = True
    for edge in range(3):
        if edge == 0:
            f0 = ax * py - ay * px
            slope = ax * dy - ay * dx
            gx = -ay
            gy = ax
        elif edge == 1:
            f0 = px * by - py * bx
            slope = dx * by - dy * bx
            gx = by
            gy = -bx
        else:
            f0 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            slope = (bx - ax) * dy - (by - ay) * dx
            gx = -(by - ay)
            gy = bx - ax
        if abs(slope) < 1e-14:
            if f0 < -1e-12 if edge < 2 else f0 <= 0.0:       # (a ray along an edge out from O is inside)
                ok = False
            continue
        t = -f0 / slope
        if slope > 0.0:
            if t > lo:
                lo = t
                lnx = gx
                lny = gy
        elif t < hi:
            hi = t
            hnx = gx
            hny = gy
    return ok and lo < hi, lo, hi, lnx, lny, hnx, hny


@njit(cache=True)
def inside_snail(x, y, bound):
    n = int(bound[1])
    for k in range(n):
        ax = bound[3 + 2 * k]
        ay = bound[4 + 2 * k]
        bx = bound[5 + 2 * k]
        by = bound[6 + 2 * k]
        if (ax * y - ay * x >= 0.0 and x * by - y * bx >= 0.0
                and (bx - ax) * (y - ay) - (by - ay) * (x - ax) > 0.0):
            return True
    return False


SNAIL_JOIN = 1e-9        # two triangles' intervals that meet within this are one stretch of the ray


@njit(cache=True)
def snail_hit(px, py, dx, dy, bound, entering):
    """boundary_hit (entering False: the ray starts inside) and obstacle_hit (entering True: it starts outside, t < 0
    if it never enters) for the snail of Pythagoras, with the normal there."""
    n = int(bound[1])
    if entering:
        best_t = 1e30
        best_nx = 0.0
        best_ny = 0.0
        for k in range(n):
            ok, lo, hi, lnx, lny, _hnx, _hny = _snail_clip(
                px, py, dx, dy, bound[3 + 2 * k], bound[4 + 2 * k], bound[5 + 2 * k], bound[6 + 2 * k])
            if ok and lo > EPS and lo < best_t:
                best_t = lo
                best_nx = lnx
                best_ny = lny
        if best_t > 1e29:
            return -1.0, 0.0, 0.0
        return best_t, best_nx, best_ny
    reach = 0.0              # how far along the ray the inside has been followed
    nx = 0.0
    ny = 0.0
    found = False
    for _pass in range(n + 1):
        grew = False
        for k in range(n):
            ok, lo, hi, _lnx, _lny, hnx, hny = _snail_clip(
                px, py, dx, dy, bound[3 + 2 * k], bound[4 + 2 * k], bound[5 + 2 * k], bound[6 + 2 * k])
            if ok and lo <= reach + SNAIL_JOIN and hi > reach + 1e-12 and hi > EPS:
                reach = hi
                nx = hnx
                ny = hny
                found = True
                grew = True
        if not grew:
            break
    if not found:
        return 1e30, 0.0, 0.0
    return reach, nx, ny


@njit(cache=True)
def snail_near_grid(gx, gy, bound, w):
    """Over a whole grid: is each point within `w` of the shape's edge? The edges are those of the triangles, but a
    piece of one that has the shape on both sides of it (the edges two triangles share, and the parts covered by a
    later turn) is not an edge of the shape; the nearest point on each close edge is probed just either side."""
    n = int(bound[1])
    h, wd = gx.shape
    out = np.zeros((h, wd), dtype=np.bool_)
    w2 = w * w
    for j in range(h):
        for i in range(wd):
            x = gx[j, i]
            y = gy[j, i]
            for k in range(n):
                for edge in range(3):
                    if edge == 0:
                        ax = 0.0
                        ay = 0.0
                        bx = bound[3 + 2 * k]
                        by = bound[4 + 2 * k]
                    elif edge == 1:
                        ax = bound[3 + 2 * k]
                        ay = bound[4 + 2 * k]
                        bx = bound[5 + 2 * k]
                        by = bound[6 + 2 * k]
                    else:
                        ax = bound[5 + 2 * k]
                        ay = bound[6 + 2 * k]
                        bx = 0.0
                        by = 0.0
                    ex = bx - ax
                    ey = by - ay
                    length2 = ex * ex + ey * ey
                    u = min(max(((x - ax) * ex + (y - ay) * ey) / length2, 0.0), 1.0)
                    qx = ax + u * ex
                    qy = ay + u * ey
                    if (x - qx) ** 2 + (y - qy) ** 2 < w2:
                        length = sqrt(length2)
                        sx = -ey / length * 1e-7
                        sy = ex / length * 1e-7
                        if inside_snail(qx + sx, qy + sy, bound) != inside_snail(qx - sx, qy - sy, bound):
                            out[j, i] = True
                            break
                if out[j, i]:
                    break
    return out


# --------------------------------------------------------------------------- #
# The Hugelschaffer egg: with u measured from the middle of its length, a ray's points satisfy
#   y^2 (a^2 + d^2 + 2 d u) = b^2 (a^2 - u^2)        (|u| < a: inside where "<")
# (the Newton transform of two circles; (x - d)^2/a^2 + y^2/b^2 (1 + (2dx - d^2)/a^2) = 1 with x = u + d). Its bound
# is [area, cos, sin, a, b, d] and it is worked out in its own frame like the squircles. Along a ray the equation is
# a cubic in t, so the hit is the first root of that cubic in the ray's stretch inside the box |u| <= a, |y| <= b
# (the curve's other branch, which runs off to infinity, is outside it).
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _cubic_root_in(c0, c1, c2, c3, lo, hi, f_lo):
    """The root in [lo, hi] of c3 t^3 + c2 t^2 + c1 t + c0, which changes sign there (f_lo is its value at lo) and
    is monotone: Newton's method kept inside the shrinking bracket, falling back to bisection."""
    if f_lo == 0.0:
        return lo
    t = 0.5 * (lo + hi)
    for _i in range(100):
        f = ((c3 * t + c2) * t + c1) * t + c0
        if f == 0.0:
            return t
        if (f > 0.0) == (f_lo > 0.0):
            lo = t
            f_lo = f
        else:
            hi = t
        slope = (3.0 * c3 * t + 2.0 * c2) * t + c1
        t_new = t - f / slope if slope != 0.0 else 0.5 * (lo + hi)
        if t_new <= lo or t_new >= hi:
            t_new = 0.5 * (lo + hi)
        if abs(t_new - t) <= 1e-15 * (1.0 + abs(t)):
            return t_new
        t = t_new
    return t


@njit(cache=True)
def inside_egg(x, y, bound):
    c = bound[1]
    s = bound[2]
    a = bound[3]
    b = bound[4]
    d = bound[5]
    u = c * x + s * y
    v = -s * x + c * y
    if abs(u) >= a:
        return False
    return v * v * (a * a + d * d + 2.0 * d * u) < b * b * (a * a - u * u)


@njit(cache=True)
def egg_hit(px, py, dx, dy, bound):
    """The nearest t > EPS where a ray meets the egg's edge, from inside or outside alike, with the normal there;
    1e30 if it never does. The ray is clipped to the egg's box, its equation becomes the cubic c3 t^3 + c2 t^2 +
    c1 t + c0 from there, and the cubic's roots are found piece by piece between its turning points (where it is
    monotone, so each piece has at most one)."""
    c = bound[1]
    s = bound[2]
    a = bound[3]
    b = bound[4]
    d = bound[5]
    lx = c * px + s * py
    ly = -s * px + c * py
    p = c * dx + s * dy
    q = -s * dx + c * dy
    a_near, a_far = _slab(lx, p, a * (1.0 + 1e-12) + 1e-12)
    b_near, b_far = _slab(ly, q, b * (1.0 + 1e-12) + 1e-12)
    t_near = max(a_near, b_near)
    t_far = min(a_far, b_far)
    if t_near > t_far or t_far <= EPS:
        return 1e30, 0.0, 0.0
    t_start = max(t_near, 0.0)
    length = t_far - t_start
    u0 = lx + t_start * p
    y0 = ly + t_start * q
    k = a * a + d * d
    d0 = k + 2.0 * d * u0
    e = 2.0 * d * p
    c0 = y0 * y0 * d0 - b * b * (a * a - u0 * u0)
    c1 = y0 * y0 * e + 2.0 * y0 * q * d0 + 2.0 * b * b * u0 * p
    c2 = 2.0 * y0 * q * e + q * q * d0 + b * b * p * p
    c3 = q * q * e
    # the turning points: the roots of the derivative 3 c3 t^2 + 2 c2 t + c1, in order
    n_turn = 0
    turn_a = 0.0
    turn_b = 0.0
    qa = 3.0 * c3
    qb = 2.0 * c2
    if abs(qa) > 1e-13 * (abs(qb) + abs(c1)) + 1e-300:
        disc = qb * qb - 4.0 * qa * c1
        if disc > 0.0:
            root = sqrt(disc)
            r1 = (-qb - root) / (2.0 * qa)
            r2 = (-qb + root) / (2.0 * qa)
            turn_a = min(r1, r2)
            turn_b = max(r1, r2)
            n_turn = 2
    elif qb != 0.0:
        turn_a = -c1 / qb
        n_turn = 1
    prev = 0.0
    f_prev = c0
    for step in range(3):
        if step == 0:
            if n_turn < 1:
                continue
            nxt = turn_a
        elif step == 1:
            if n_turn < 2:
                continue
            nxt = turn_b
        else:
            nxt = length
        if step < 2 and not prev < nxt < length:
            continue
        f_nxt = ((c3 * nxt + c2) * nxt + c1) * nxt + c0
        if (f_prev < 0.0) != (f_nxt < 0.0):
            r = _cubic_root_in(c0, c1, c2, c3, prev, nxt, f_prev)
            t = t_start + r
            if t > EPS:
                u = u0 + r * p
                y = y0 + r * q
                gu = 2.0 * d * y * y + 2.0 * b * b * u
                gv = 2.0 * y * (k + 2.0 * d * u)
                return t, c * gu - s * gv, s * gu + c * gv
        prev = nxt
        f_prev = f_nxt
    return 1e30, 0.0, 0.0


@njit(cache=True)
def egg_distance_grid(gx, gy, bound, w):
    """Over a whole grid: the first-order distance |F| / |grad F| from each point to the egg's edge, for the points
    within `w` of the egg's box (F is the equation above); every other point is reported as far away. The estimate
    is only good near the egg: far out it goes to zero along the vertical line where the curve's other branch
    comes in from infinity (where 2 d u + a^2 + d^2 is 0), which at a large shift is only just past the egg's point
    and drew a straight line right across the image."""
    c = bound[1]
    s = bound[2]
    a = bound[3]
    b = bound[4]
    d = bound[5]
    k = a * a + d * d
    h, width = gx.shape
    out = np.full((h, width), 1e30)
    for j in range(h):
        for i in range(width):
            u = c * gx[j, i] + s * gy[j, i]
            y = -s * gx[j, i] + c * gy[j, i]
            den = k + 2.0 * d * u
            if den > 0.0 and abs(u) <= a + w and abs(y) <= b + w:
                f = y * y * den - b * b * (a * a - u * u)
                gu = 2.0 * d * y * y + 2.0 * b * b * u
                gv = 2.0 * y * den
                out[j, i] = abs(f) / max(sqrt(gu * gu + gv * gv), 1e-9)
    return out


# --------------------------------------------------------------------------- #
# The sine flower: the points (rho, phi), in the shape's own frame, with rho < r(phi) = R (1 + A sin(k phi)); its bound
# is [area, cos, sin, R, A, k] (A below 1, so r stays positive; k a whole number of petals). It is star-shaped about
# the centre but not convex. A straight line at distance h from the centre has rho = h / cos(phi - phi0) (phi0 the
# direction of its nearest point), so it meets the edge where  g(phi) = h - r(phi) cos(phi - phi0)  is 0, and
# g < 0 means inside. g has no division in it and a known bound on its second derivative, so it is searched
# along the stretch of the line that lies between the inner and outer circles of the flower (the edge is nowhere
# else), skipping only intervals proved to hold no root, and every sign change is solved by bracketed Newton.
# --------------------------------------------------------------------------- #
@njit(cache=True)
def inside_flower(x, y, bound):
    c = bound[1]
    s = bound[2]
    u = c * x + s * y
    v = -s * x + c * y
    return sqrt(u * u + v * v) < bound[3] * (1.0 + bound[4] * sin(bound[5] * atan2(v, u)))


@njit(cache=True)
def _flower_root(lo, hi, g_lo, phi0, h, radius, amp, petals):
    """The phi in [lo, hi] where g(phi) = h - r(phi) cos(phi - phi0) is 0, g changing sign there (g_lo is its value
    at lo): Newton's method kept inside the shrinking bracket, falling back to bisection."""
    if g_lo == 0.0:
        return lo
    phi = 0.5 * (lo + hi)
    for _i in range(100):
        r = radius * (1.0 + amp * sin(petals * phi))
        g = h - r * cos(phi - phi0)
        if g == 0.0:
            return phi
        if (g > 0.0) == (g_lo > 0.0):
            lo = phi
            g_lo = g
        else:
            hi = phi
        dr = radius * amp * petals * cos(petals * phi)
        slope = -dr * cos(phi - phi0) + r * sin(phi - phi0)
        phi_new = phi - g / slope if slope != 0.0 else 0.5 * (lo + hi)
        if phi_new <= lo or phi_new >= hi:
            phi_new = 0.5 * (lo + hi)
        if abs(phi_new - phi) <= 1e-15 * (1.0 + abs(phi)):
            return phi_new
        phi = phi_new
    return phi


@njit(cache=True)
def _flower_window(phi_from, phi_to, phi0, h, radius, amp, petals, lx, ly, ldx, ldy):
    """Walk phi from `phi_from` to `phi_to` (either way round) looking for the first sign change of g whose point on
    the ray is further along than EPS: (found, t, the edge's normal there, in the shape's frame).

    The walk starts from steps of a quarter of g's shortest period, and an interval is only passed over once it is
    PROVEN to hold no root, using a bound M on |g''| (g is below its tangent at either end by at most M d^2 / 2, so g
    cannot reach 0 if that bound stays on one side of it) - a shallow dip, like a petal's tip grazed by the line, is
    therefore found however narrow it is. An interval with a sign change is solved once g is known to be monotone
    across it (|g'| at an end beats M times the interval's length); otherwise it is halved, the half met first
    being looked at first."""
    span = abs(phi_to - phi_from)
    if span <= 0.0:
        return False, 0.0, 0.0, 0.0
    bound2 = radius * (amp * petals * petals + 2.0 * amp * petals + 1.0 + amp)     # |g''| <= this
    n0 = max(1, int(span * (petals + 1.0) * 4.0 / (2.0 * pi)) + 1)
    step = (phi_to - phi_from) / n0
    stack_a = np.empty(64)
    stack_b = np.empty(64)
    for i in range(n0):
        stack_a[0] = phi_from + step * i
        stack_b[0] = phi_from + step * (i + 1)
        top = 1
        while top > 0:
            top -= 1
            a = stack_a[top]
            b = stack_b[top]
            ra = radius * (1.0 + amp * sin(petals * a))
            rb = radius * (1.0 + amp * sin(petals * b))
            ga = h - ra * cos(a - phi0)
            gb = h - rb * cos(b - phi0)
            da = -radius * amp * petals * cos(petals * a) * cos(a - phi0) + ra * sin(a - phi0)
            db = -radius * amp * petals * cos(petals * b) * cos(b - phi0) + rb * sin(b - phi0)
            length = abs(b - a)
            slack = 0.5 * bound2 * length * length
            if (ga < 0.0) != (gb < 0.0):
                done = abs(da) > bound2 * length or abs(db) > bound2 * length or length < 1e-9
                if done:
                    if a < b:
                        phi = _flower_root(a, b, ga, phi0, h, radius, amp, petals)
                    else:
                        phi = _flower_root(b, a, gb, phi0, h, radius, amp, petals)
                    r = radius * (1.0 + amp * sin(petals * phi))
                    t = (r * cos(phi) - lx) * ldx + (r * sin(phi) - ly) * ldy
                    if t > EPS:
                        slope = radius * amp * petals * cos(petals * phi) / r
                        return True, t, cos(phi) + slope * sin(phi), sin(phi) - slope * cos(phi)
            elif ga >= 0.0:
                done = ga + da * (b - a) - slack > 0.0 or gb - db * (b - a) - slack > 0.0 or length < 1e-9
            else:
                done = ga + da * (b - a) + slack < 0.0 or gb - db * (b - a) + slack < 0.0 or length < 1e-9
            if not done and top < 62:
                mid = 0.5 * (a + b)
                stack_a[top] = mid
                stack_b[top] = b
                stack_a[top + 1] = a
                stack_b[top + 1] = mid
                top += 2
    return False, 0.0, 0.0, 0.0


@njit(cache=True)
def flower_hit(px, py, dx, dy, bound):
    """The nearest t > EPS where a ray meets the sine flower's edge, from inside or outside alike, with the normal
    there; 1e30 if it never does. The line's stretch between the flower's inner and outer circles is one or two
    windows of phi (one if the line is nearer the centre than the inner circle is, so that it runs through it),
    taken in the order the ray reaches them."""
    c = bound[1]
    s = bound[2]
    radius = bound[3]
    amp = bound[4]
    petals = bound[5]
    lx = c * px + s * py
    ly = -s * px + c * py
    ldx = c * dx + s * dy
    ldy = -s * dx + c * dy
    cross = lx * ldy - ly * ldx
    h = abs(cross)
    r_out = radius * (1.0 + amp) * (1.0 + 1e-7)         # (a hair wider each way, so that a circle - amplitude 0 -
    r_in = radius * (1.0 - amp) * (1.0 - 1e-7)          # still has a window to look in)
    if h >= r_out:
        return 1e30, 0.0, 0.0
    if h < 1e-9:
        # through the centre: the line leaves it along phi_d (to one side) and phi_d + pi (to the other), at distances
        # r(phi) from it, and g has no roots to find (rho = h / cos is 0 / 0 there)
        phi_d = atan2(ldy, ldx)
        t_centre = -(lx * ldx + ly * ldy)
        t1 = t_centre + radius * (1.0 + amp * sin(petals * phi_d))
        t2 = t_centre - radius * (1.0 + amp * sin(petals * (phi_d + pi)))
        phi = phi_d
        t = t1
        if not t1 > EPS or (t2 > EPS and t2 < t1):
            phi = phi_d + pi
            t = t2
        if not t > EPS:
            return 1e30, 0.0, 0.0
        slope = radius * amp * petals * cos(petals * phi) / (radius * (1.0 + amp * sin(petals * phi)))
        nx = cos(phi) + slope * sin(phi)
        ny = sin(phi) - slope * cos(phi)
        return t, c * nx - s * ny, s * nx + c * ny
    phi0 = atan2(-ldx, ldy) if cross >= 0.0 else atan2(ldx, -ldy)       # the direction of the line's nearest point
    alpha_out = acos(h / r_out)
    alpha_in = acos(h / r_in) if h < r_in else 0.0
    forward = cross >= 0.0          # phi grows the way the ray goes
    found = False
    t = 0.0
    nx = 0.0
    ny = 0.0
    if alpha_in == 0.0:
        if forward:
            found, t, nx, ny = _flower_window(phi0 - alpha_out, phi0 + alpha_out, phi0, h, radius, amp, petals,
                                              lx, ly, ldx, ldy)
        else:
            found, t, nx, ny = _flower_window(phi0 + alpha_out, phi0 - alpha_out, phi0, h, radius, amp, petals,
                                              lx, ly, ldx, ldy)
    elif forward:
        found, t, nx, ny = _flower_window(phi0 - alpha_out, phi0 - alpha_in, phi0, h, radius, amp, petals,
                                          lx, ly, ldx, ldy)
        if not found:
            found, t, nx, ny = _flower_window(phi0 + alpha_in, phi0 + alpha_out, phi0, h, radius, amp, petals,
                                              lx, ly, ldx, ldy)
    else:
        found, t, nx, ny = _flower_window(phi0 + alpha_out, phi0 + alpha_in, phi0, h, radius, amp, petals,
                                          lx, ly, ldx, ldy)
        if not found:
            found, t, nx, ny = _flower_window(phi0 - alpha_in, phi0 - alpha_out, phi0, h, radius, amp, petals,
                                              lx, ly, ldx, ldy)
    if not found:
        return 1e30, 0.0, 0.0
    return t, c * nx - s * ny, s * nx + c * ny


@njit(cache=True)
def flower_distance_grid(gx, gy, bound):
    """Over a whole grid: the distance from each point to the sine flower's edge, for the outline's two ring layers.
    F = rho - r(phi) is 0 on the edge, and each point is slid along F's gradient to where F is 0 (Newton's method on
    the point itself, four steps); the distance is how far it moved. A point that does not arrive (too near the centre,
    where the petals are steep) is reported as far away."""
    c = bound[1]
    s = bound[2]
    radius = bound[3]
    amp = bound[4]
    petals = bound[5]
    h, width = gx.shape
    out = np.full((h, width), 1e30)
    for j in range(h):
        for i in range(width):
            x0 = c * gx[j, i] + s * gy[j, i]
            y0 = -s * gx[j, i] + c * gy[j, i]
            x = x0
            y = y0
            f = 1e30
            for _step in range(4):
                rho = sqrt(x * x + y * y)
                if rho < 1e-12:
                    f = 1e30
                    break
                phi = atan2(y, x)
                r = radius * (1.0 + amp * sin(petals * phi))
                slope = radius * amp * petals * cos(petals * phi) / rho
                gxx = cos(phi) + slope * sin(phi)
                gyy = sin(phi) - slope * cos(phi)
                f = rho - r
                norm2 = gxx * gxx + gyy * gyy
                x -= f * gxx / norm2
                y -= f * gyy / norm2
            rho = sqrt(x * x + y * y)
            if rho > 1e-12 and abs(rho - radius * (1.0 + amp * sin(petals * atan2(y, x)))) < 1e-4 * radius:
                out[j, i] = sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return out


@njit(cache=True)
def inside_boundary(x, y, shape, bound):
    """Is the point (x, y) strictly inside the shape (whose geometry is `bound`)?"""
    if shape == SHAPE_ELLIPSE:
        return bound[1] * x * x + 2.0 * bound[2] * x * y + bound[3] * y * y < 1.0
    if shape == SHAPE_POLYGON:
        h = bound[1]
        n = int(bound[2])
        for k in range(n):
            if bound[3 + 2 * k] * x + bound[4 + 2 * k] * y >= h:
                return False
        return True
    if shape == SHAPE_LENS or shape == SHAPE_REULEAUX:
        return inside_circles(x, y, bound)
    if shape == SHAPE_SECTOR or shape == SHAPE_SEGMENT:
        return inside_arcs_edges_and(x, y, bound)
    if shape == SHAPE_SUPERELLIPSE or shape == SHAPE_FG_SQUIRCLE:
        return inside_squircle(x, y, shape, bound)
    if shape == SHAPE_SPIRAL:
        return inside_spiral(x, y, bound)
    if shape == SHAPE_SNAIL:
        return inside_snail(x, y, bound)
    if shape == SHAPE_EGG:
        return inside_egg(x, y, bound)
    if shape == SHAPE_FLOWER:
        return inside_flower(x, y, bound)
    if shape >= SHAPE_FIRST_ARC_LOOP:
        return inside_arcs_parity(x, y, bound)
    return inside_arcs_edges_star(x, y, bound)   # MOSS_EGG, CAPSULE


@njit(cache=True)
def inside_boundary_grid(gx, gy, shape, bound):
    """inside_boundary(), over a whole (H, W) grid at once - compiled, so this
    stays fast even though it's a plain loop; used for drawing shape outlines
    and thumbnails."""
    h, w = gx.shape
    out = np.empty((h, w), dtype=np.bool_)
    for j in range(h):
        for i in range(w):
            out[j, i] = inside_boundary(gx[j, i], gy[j, i], shape, bound)
    return out


@njit(cache=True)
def fires_ray(x, y, obstacles):
    """Does a ray actually start at (x, y)? True if (x, y) is inside any shape
    whose "fire rays inside" is on. Being inside shapes that have it off makes
    no difference."""
    for oi in range(len(obstacles.shapes)):
        if obstacles.fire_inside[oi] and inside_boundary(
                x - obstacles.cx[oi], y - obstacles.cy[oi],
                obstacles.shapes[oi], obstacles.bounds[oi]):
            return True
    return False


@njit(cache=True)
def boundary_hit(px, py, dx, dy, shape, bound):
    """
    For a ray starting at (px, py), inside the shape, heading along the unit
    vector (dx, dy): the distance t to where it meets the boundary, and the
    normal there (not necessarily unit length).
    """
    if shape == SHAPE_ELLIPSE:
        m11 = bound[1]
        m12 = bound[2]
        m22 = bound[3]
        # Solve  A t^2 + 2 B t + C = 0  for the point p + t d on  p^T M p = 1.
        # The ray starts inside, so take the far root.
        qa = m11 * dx * dx + 2.0 * m12 * dx * dy + m22 * dy * dy
        qb = m11 * px * dx + m12 * (px * dy + py * dx) + m22 * py * dy
        qc = m11 * px * px + 2.0 * m12 * px * py + m22 * py * py - 1.0
        disc = qb * qb - qa * qc
        if disc < 0.0:
            disc = 0.0
        t = (-qb + sqrt(disc)) / qa
        hx = px + t * dx
        hy = py + t * dy
        return t, m11 * hx + m12 * hy, m12 * hx + m22 * hy
    if shape == SHAPE_POLYGON:
        # Convex polygon: the ray leaves through the nearest edge it is heading out of
        h = bound[1]
        n = int(bound[2])
        best = 1e30
        bnx = 0.0
        bny = 0.0
        for k in range(n):
            nx = bound[3 + 2 * k]
            ny = bound[4 + 2 * k]
            nd = nx * dx + ny * dy
            if nd > 1e-12:
                t = (h - (nx * px + ny * py)) / nd
                if t < 0.0:
                    t = 0.0
                if t < best:
                    best = t
                    bnx = nx
                    bny = ny
        return best, bnx, bny
    if shape == SHAPE_LENS or shape == SHAPE_REULEAUX:
        return circles_exit_hit(px, py, dx, dy, bound)
    if shape == SHAPE_SUPERELLIPSE or shape == SHAPE_FG_SQUIRCLE:
        return squircle_hit(px, py, dx, dy, shape, bound, False)
    if shape == SHAPE_SPIRAL:
        return spiral_hit(px, py, dx, dy, bound)
    if shape == SHAPE_SNAIL:
        return snail_hit(px, py, dx, dy, bound, False)
    if shape == SHAPE_EGG:
        return egg_hit(px, py, dx, dy, bound)
    if shape == SHAPE_FLOWER:
        return flower_hit(px, py, dx, dy, bound)
    return arcs_edges_hit(px, py, dx, dy, bound)   # MOSS_EGG, SECTOR, SEGMENT, CAPSULE, the arc loops


@njit(cache=True)
def obstacle_hit(px, py, dx, dy, shape, bound):
    """
    For a ray at (px, py), OUTSIDE this solid obstacle, heading along the unit
    vector (dx, dy): the distance t to where it first enters the obstacle (and
    the normal there), or t < 0 if this ray's forward path (t > EPS) never
    reaches it. Ellipses use the near root of the quadratic; convex polygons
    use a Cyrus-Beck-style half-plane clip.
    """
    if shape == SHAPE_ELLIPSE:
        m11 = bound[1]
        m12 = bound[2]
        m22 = bound[3]
        qa = m11 * dx * dx + 2.0 * m12 * dx * dy + m22 * dy * dy
        qb = m11 * px * dx + m12 * (px * dy + py * dx) + m22 * py * dy
        qc = m11 * px * px + 2.0 * m12 * px * py + m22 * py * py - 1.0
        disc = qb * qb - qa * qc
        if disc < 0.0 or qa <= 0.0:
            return -1.0, 0.0, 0.0
        t = (-qb - sqrt(disc)) / qa                # near root: the entry point
        if t <= EPS:
            return -1.0, 0.0, 0.0
        hx = px + t * dx
        hy = py + t * dy
        return t, m11 * hx + m12 * hy, m12 * hx + m22 * hy
    if shape == SHAPE_POLYGON:
        # Convex polygon, as a solid obstacle: clip the ray against every edge's
        # half-plane; the entry time is the latest of the "becomes satisfied"
        # crossings, the exit time the earliest of the "stops being satisfied" ones.
        h = bound[1]
        n = int(bound[2])
        t_enter = -1e30
        t_exit = 1e30
        enter_nx = 0.0
        enter_ny = 0.0
        for k in range(n):
            nx = bound[3 + 2 * k]
            ny = bound[4 + 2 * k]
            nd = nx * dx + ny * dy
            num = h - (nx * px + ny * py)
            if nd > 1e-14:
                if num < 0.0:
                    return -1.0, 0.0, 0.0            # never satisfied for t >= 0
                t = num / nd
                if t < t_exit:
                    t_exit = t
            elif nd < -1e-14:
                if num < 0.0:
                    t = num / nd
                    if t > t_enter:
                        t_enter = t
                        enter_nx = nx
                        enter_ny = ny
            else:
                if num < 0.0:
                    return -1.0, 0.0, 0.0
        if t_enter <= EPS or t_enter > t_exit:
            return -1.0, 0.0, 0.0
        return t_enter, enter_nx, enter_ny
    if shape == SHAPE_LENS or shape == SHAPE_REULEAUX:
        return circles_entry_hit(px, py, dx, dy, bound)
    if shape == SHAPE_SUPERELLIPSE or shape == SHAPE_FG_SQUIRCLE:
        return squircle_hit(px, py, dx, dy, shape, bound, True)
    if shape == SHAPE_SPIRAL:
        t, nx, ny = spiral_hit(px, py, dx, dy, bound)
        if t > 1e29:
            return -1.0, 0.0, 0.0
        return t, nx, ny
    if shape == SHAPE_SNAIL:
        return snail_hit(px, py, dx, dy, bound, True)
    if shape == SHAPE_EGG:
        t, nx, ny = egg_hit(px, py, dx, dy, bound)
        if t > 1e29:
            return -1.0, 0.0, 0.0
        return t, nx, ny
    if shape == SHAPE_FLOWER:
        t, nx, ny = flower_hit(px, py, dx, dy, bound)
        if t > 1e29:
            return -1.0, 0.0, 0.0
        return t, nx, ny
    t, nx, ny = arcs_edges_hit(px, py, dx, dy, bound)   # MOSS_EGG, SECTOR, SEGMENT, CAPSULE, the arc loops
    if t > 1e29:
        return -1.0, 0.0, 0.0
    return t, nx, ny


@njit(cache=True)
def beyond_escape_radius(x, y, escape_radius):
    """Is the point (x, y) `escape_radius` or further from the origin? The escape test of every engine: an orbit
    ends at the first point for which it is true; a ray ends where it first becomes true (escape_exit_t)."""
    return x * x + y * y - escape_radius * escape_radius >= 0.0


@njit(cache=True)
def escape_exit_t(px, py, dx, dy, escape_radius):
    """Distance along the unit direction (dx, dy) from (px, py) to where the
    ray reaches `escape_radius` from the origin (0 if it is already that far
    out or further)."""
    if beyond_escape_radius(px, py, escape_radius):
        return 0.0
    b = px * dx + py * dy
    c = px * px + py * py - escape_radius * escape_radius
    return -b + sqrt(b * b - c)


@njit(cache=True)
def is_confined(path_points, npts, confine):
    """Has the sequence stopped getting anywhere? Judges the last `confine.lookback` of its first `npts` points: they
    are confined when they lie inside a box smaller than `confine.threshold` both ways (CONFINE_BOX), or when the
    segments joining them add up to less than it (CONFINE_PATH). False while there are fewer points than the
    lookback. The confinement test of every engine (for an orbit it is a convergence test)."""
    lookback = confine.lookback
    if npts < lookback:
        return False
    threshold = confine.threshold
    if confine.method == CONFINE_BOX:
        minx = 1e30
        maxx = -1e30
        miny = 1e30
        maxy = -1e30
        for k in range(npts - lookback, npts):
            minx = min(minx, path_points[k, 0])
            maxx = max(maxx, path_points[k, 0])
            miny = min(miny, path_points[k, 1])
            maxy = max(maxy, path_points[k, 1])
        return (maxx - minx) < threshold and (maxy - miny) < threshold
    length = 0.0                                   # CONFINE_PATH
    for k in range(npts - lookback, npts - 1):
        length += sqrt((path_points[k + 1, 0] - path_points[k, 0]) ** 2
                       + (path_points[k + 1, 1] - path_points[k, 1]) ** 2)
    return length < threshold


ESCAPE_OBJ = -3      # trace_path's "what did the ray hit": nothing, it escaped


@njit(cache=True)
def unit_direction(deg):
    """The unit vector (dx, dy) pointing `deg` degrees counterclockwise from
    +x. Exact for multiples of 45 degrees (so straight up is exactly (0, 1) and
    the up-and-right default is exactly (1, 1)/sqrt(2), rather than a rounding
    error away), and otherwise cos/sin of the angle."""
    k = deg / 45.0
    if k == floor(k):
        i = int(k) % 8
        if i == 0:
            return 1.0, 0.0
        if i == 1:
            return INV_SQRT2, INV_SQRT2
        if i == 2:
            return 0.0, 1.0
        if i == 3:
            return -INV_SQRT2, INV_SQRT2
        if i == 4:
            return -1.0, 0.0
        if i == 5:
            return -INV_SQRT2, -INV_SQRT2
        if i == 6:
            return 0.0, -1.0
        return INV_SQRT2, -INV_SQRT2
    rad = deg * pi / 180.0
    return cos(rad), sin(rad)


@njit(nogil=True, cache=True)
def trace_path(x0, y0, path_points, path_point_tags, termination, engine_args, raw):
    """
    Trace one ray, storing the start point and every
    bounce point in path_points (needs at least termination.max_iter + 2 rows, 2 columns), the
    "tag" of each of those points in path_point_tags (TAG_START / TAG_BOUNDARY /
    TAG_SELF / TAG_DISTCUT / TAG_OBSTACLE_TERM / TAG_ESCAPED / TAG_PERPENDICULAR, same length
    as path_points), and
    some raw measurements in raw (N_RAW entries). If the distance limit is
    reached part-way along a segment, the ray's end point (on that segment) is
    appended to path_points (tag TAG_DISTCUT) too.

    `termination` is a NaturalTerminationRule (see natural_termination_rule): what ends the ray without anything in the scene ending it -
    the iteration limit `max_iter` (TERM_ITER), the distance limit `max_dist` (TERM_DIST), the `escape_radius` an
    escaping ray ends at (TERM_ESCAPED) and the `confine` rule (TERM_CONFINED, below). `engine_args` is a RayArgs (see
    ray_args): `seg_mult`, `fuzz`, `emission` and `obstacles`, described below.

    `engine_args.seg_mult` (0 < seg_mult <= 1) shortens every ordinary bounce's recorded
    segment to that fraction of its true length - the endpoint moves back
    towards the segment's start - while the ray still continues on from there
    in the direction the bounce actually calculated (from the true, full-length
    hit point's surface normal), so the ray visibly changes direction in
    mid-air rather than exactly at the boundary/obstacle/its own path. 1 means
    no effect. This only applies to ordinary bounces - a ray that terminates
    (distance limit, a "terminate" obstacle, perpendicular incidence) always
    still reaches its true, unshortened endpoint, since there's no next
    segment to continue into.

    `engine_args.obstacles` is an ObstacleArrays describing every obstacle (parallel
    arrays; each obstacle's own bound is centred on (cx[i], cy[i]); `effects`
    is an (n, 2) array of
    OBSTACLE_EFFECT_* codes: column 0 for rays coming in from outside, column
    1 for rays going out from inside). What happens to a ray as it reaches a
    shape's edge depends only on the effect chosen for that direction. Reflect acts like a
    mirror (using the boundary's fuzzing settings and counting toward the same
    bounce total); Terminate ends the ray the instant the edge is reached;
    Pass through lets the ray cross (flipping whether it is inside that
    shape) without a bounce or a recorded point. A ray starts inside every
    shape that contains its start point. A shape that passes rays through in
    both directions is ignored. If a ray has nothing left ahead to hit, it
    escapes (TERM_ESCAPED): its last point is where it is `escape_radius` from
    the origin (never before the last edge it passed through).

    `termination.confine` is a ConfineRule (see confine_rule): the ray is confined
    (TERM_CONFINED) when the last `lookback` points of its path are inside a box
    smaller than the threshold both ways (CONFINE_BOX), or their segments add up
    to less than it (CONFINE_PATH) - see is_confined.

    `engine_args.emission` is an EmissionRule (see emission_rule): the ray's direction is its
    base direction - straight up (EMIT_UP), or from its start point towards the
    target (EMIT_TOWARDS) - plus the offset in degrees, positive meaning more
    counterclockwise.

    `engine_args.fuzz` is a FuzzRule (see fuzz_rule; angles in degrees, emit_jitter is added
    to the emission direction). When enabled, the RNG is used for the jitter
    terms, so the caller must have seeded it beforehand for reproducibility.

    Returns (term, npts):
      term  why the ray terminated: TERM_CONFINED, TERM_ITER, TERM_DIST,
            TERM_OBSTACLE, TERM_PERPENDICULAR or TERM_ESCAPED
      npts  number of valid points in path_points (the last is the final location)
    """
    max_iter = termination.max_iter
    max_dist = termination.max_dist
    escape_radius = termination.escape_radius
    confine = termination.confine
    seg_mult = engine_args.seg_mult
    fuzz = engine_args.fuzz
    emission = engine_args.emission
    obstacles = engine_args.obstacles
    obst_shapes = obstacles.shapes
    obst_bounds = obstacles.bounds
    obst_cx = obstacles.cx
    obst_cy = obstacles.cy
    obst_effects = obstacles.effects
    fuzzing = fuzz.enabled
    n_obst = len(obst_shapes)

    path_points[0, 0] = x0
    path_points[0, 1] = y0
    path_point_tags[0] = TAG_START
    npts = 1
    n_bounces = 0
    self_k = -1
    self_m = -1
    deflect = 0.0
    dist = 0.0
    last_full = 0.0
    term = TERM_ITER

    # Which shapes the ray is inside right now. It changes only when the ray
    # passes through an edge (see below), and is always decided by this flag
    # rather than re-derived from the geometry, so a ray grazing an edge can't
    # be judged inside by one test and outside by the next.
    inside = np.zeros(n_obst, dtype=np.bool_)
    for oi in range(n_obst):
        inside[oi] = inside_boundary(x0 - obst_cx[oi], y0 - obst_cy[oi],
                                     obst_shapes[oi], obst_bounds[oi])
    max_cross = 2 * n_obst + 2               # a shape passes rays one way only, so it is passed through at most once per segment

    px = x0
    py = y0
    # Emission direction: the base (straight up, or towards the target point -
    # straight up if this ray starts exactly on it), plus the angle offset, plus
    # (with fuzzing) a random jitter.
    if emission.mode == EMIT_TOWARDS and (emission.target_x != x0 or emission.target_y != y0):
        base_deg = atan2(emission.target_y - y0, emission.target_x - x0) * 180.0 / pi
    else:
        base_deg = 90.0
    ray_deg = base_deg + emission.offset
    if fuzzing:
        ej = fuzz.emit_jitter
        ray_deg += np.random.uniform(-ej, ej) if ej > 0.0 else 0.0
    dx, dy = unit_direction(ray_deg)
    last_obj = -2                            # >= 0: a segment; <= OBST_OBJ_BASE: a shape's edge

    for _ in range(max_iter):
        # Find where this straight segment ends: at a reflecting edge, a
        # terminating edge, the ray's own earlier path, or (if there is nothing
        # ahead) the edge of the image. Passing through an edge is not an end:
        # the ray just flips its inside/outside state for that shape and carries
        # on, all as part of the same segment, so a crossing is never a
        # bounce, never a recorded point, and never shortens the segment.
        t_off = 0.0                          # distance already advanced along this segment
        n_cross = 0
        ended = False
        best_t = 1e30
        best_obj = -1
        hit_eff = OBSTACLE_EFFECT_PASS
        wnx = 0.0
        wny = 0.0
        while True:
            qx = px + t_off * dx
            qy = py + t_off * dy
            best_t = 1e30
            best_obj = -1                    # nothing hit (yet)
            hit_eff = OBSTACLE_EFFECT_PASS

            # Distance to every shape's edge: from inside, the way out; from
            # outside, the way in. A shape that passes rays through in both
            # directions is not there at all.
            for si in range(n_obst):
                e_in = obst_effects[si, 0]
                e_out = obst_effects[si, 1]
                if e_in == OBSTACLE_EFFECT_PASS and e_out == OBSTACLE_EFFECT_PASS:
                    continue
                if inside[si]:
                    t, hnx, hny = boundary_hit(qx - obst_cx[si], qy - obst_cy[si], dx, dy,
                                               obst_shapes[si], obst_bounds[si])
                else:
                    t, hnx, hny = obstacle_hit(qx - obst_cx[si], qy - obst_cy[si], dx, dy,
                                               obst_shapes[si], obst_bounds[si])
                if t >= 0.0 and t < best_t:
                    best_t = t
                    best_obj = OBST_OBJ_BASE - si
                    wnx = hnx
                    wny = hny
                    hit_eff = e_out if inside[si] else e_in

            # Distance to earlier path segments: path_points[k] -> path_points[k+1]. A
            # self-segment collision only overrides an edge collision already
            # found above if it's MEANINGFULLY closer (not just closer by
            # floating-point noise) - otherwise, when retracing an old path
            # back through a point that's also on an edge (e.g. after a
            # self-bounce at normal incidence sends the ray straight back the
            # way it came), a near-tie could let the self-segment "win" and
            # the ray would reflect as if the solid wall weren't there,
            # sneaking through it.
            for k in range(npts - 1):
                if k == last_obj or k == npts - 2:   # skip segment we're leaving
                    continue
                ax = path_points[k, 0]
                ay = path_points[k, 1]
                ex = path_points[k + 1, 0] - ax
                ey = path_points[k + 1, 1] - ay
                denom = dx * ey - dy * ex
                if -1e-14 < denom < 1e-14:
                    continue
                wx = ax - qx
                wy = ay - qy
                t = (wx * ey - wy * ex) / denom
                if t <= EPS or t >= best_t - SELF_TIE_EPS:
                    continue
                u = (wx * dy - wy * dx) / denom
                if u < 0.0 or u > 1.0:
                    continue
                best_t = t
                best_obj = k

            if best_t >= 1e29:
                # Nothing at all ahead: the ray escapes. Its segment ends where
                # is escape_radius from the origin (never before the last edge it
                # passed through).
                best_obj = ESCAPE_OBJ
                best_t = max(escape_exit_t(px, py, dx, dy, escape_radius), t_off) - t_off

            if best_obj <= OBST_OBJ_BASE and hit_eff == OBSTACLE_EFFECT_PASS:
                # Through an edge: flip inside/outside for that shape, carry on
                inside[OBST_OBJ_BASE - best_obj] = not inside[OBST_OBJ_BASE - best_obj]
                t_off += best_t
                n_cross += 1
                if n_cross > max_cross:      # coincident edges etc.: cannot happen normally
                    term = TERM_CONFINED
                    ended = True
                    break
                continue
            break                            # a real end to this segment
        if ended:
            break
        total_t = t_off + best_t             # the segment's full length, crossings included

        # Distance limit: cut this segment off where the limit is reached
        if dist + total_t > max_dist:
            rem = max_dist - dist
            if rem > 1e-12:
                path_points[npts, 0] = px + rem * dx
                path_points[npts, 1] = py + rem * dy
                path_point_tags[npts] = TAG_DISTCUT
                npts += 1
                last_full = total_t
            term = TERM_DIST
            break

        # Nothing ahead: the ray leaves for good
        if best_obj == ESCAPE_OBJ:
            path_points[npts, 0] = px + total_t * dx
            path_points[npts, 1] = py + total_t * dy
            path_point_tags[npts] = TAG_ESCAPED
            npts += 1
            term = TERM_ESCAPED
            break

        # A "terminate" edge ends the ray outright, with no reflection
        if best_obj <= OBST_OBJ_BASE and hit_eff == OBSTACLE_EFFECT_TERMINATE:
            path_points[npts, 0] = px + total_t * dx
            path_points[npts, 1] = py + total_t * dy
            path_point_tags[npts] = TAG_OBSTACLE_TERM
            npts += 1
            term = TERM_OBSTACLE
            break

        # Move to the hit point
        hx = px + total_t * dx
        hy = py + total_t * dy

        # Surface normal at the hit point
        if best_obj <= OBST_OBJ_BASE:
            nx = wnx
            ny = wny
        else:
            if self_k < 0:                       # first time the ray touches itself
                self_k = best_obj
                self_m = npts                    # index the hit point will occupy
            nx = -(path_points[best_obj + 1, 1] - path_points[best_obj, 1])
            ny = path_points[best_obj + 1, 0] - path_points[best_obj, 0]
        nlen = sqrt(nx * nx + ny * ny)
        if nlen < 1e-12:
            # A degenerate (zero-length) normal means the ray has become
            # trapped at a single point (e.g. a sharp corner like a sector's
            # apex) rather than crossing a real surface - treat it the same
            # as becoming confined, rather than dividing by zero.
            term = TERM_CONFINED
            break
        nx /= nlen
        ny /= nlen

        # Perfectly perpendicular incidence: the ray would reflect straight
        # back the way it came, retracing its own path - and which side of
        # itself it ends up on from there is undefined, so the ray ends here
        # instead (a distinct termination reason, though it behaves like
        # confinement in that it's a sudden termination rather than a hit on a
        # boundary/obstacle/distance/iteration limit).
        dot = dx * nx + dy * ny
        if dot > 1.0 - RETRO_EPS or dot < -1.0 + RETRO_EPS:
            path_points[npts, 0] = hx
            path_points[npts, 1] = hy
            path_point_tags[npts] = TAG_PERPENDICULAR
            npts += 1
            term = TERM_PERPENDICULAR
            break

        # Perfect-mirror reflection of the incoming direction
        odx = dx
        ody = dy
        rdx = dx - 2.0 * dot * nx
        rdy = dy - 2.0 * dot * ny
        rlen = sqrt(rdx * rdx + rdy * rdy)
        rdx /= rlen
        rdy /= rlen

        # Fuzzing: perturb the deflection angle (incoming -> outgoing) itself,
        # so an unfuzzed bounce (offset 0, jitter 0, multiplier 1) is identical
        # to a plain mirror reflection. Bounces off any reflecting edge and
        # off the ray's own path have separate settings.
        if fuzzing:
            if best_obj <= OBST_OBJ_BASE:
                f_off = fuzz.bnd_offset
                f_jit = fuzz.bnd_jitter
                f_mult = fuzz.bnd_mult
            else:
                f_off = fuzz.self_offset
                f_jit = fuzz.self_jitter
                f_mult = fuzz.self_mult
            cross = odx * rdy - ody * rdx
            dotr = odx * rdx + ody * rdy
            base_ang = atan2(cross, dotr)
            jitter_b = np.random.uniform(-f_jit, f_jit) if f_jit > 0.0 else 0.0
            new_ang = base_ang * f_mult + f_off * pi / 180.0 + jitter_b * pi / 180.0
            ca = cos(new_ang)
            sa = sin(new_ang)
            dx = odx * ca - ody * sa
            dy = odx * sa + ody * ca
            dlen = sqrt(dx * dx + dy * dy)
            dx /= dlen
            dy /= dlen
        else:
            dx = rdx
            dy = rdy

        c = odx * dx + ody * dy
        if c > 1.0:
            c = 1.0
        elif c < -1.0:
            c = -1.0
        deflect += acos(c) * (180.0 / pi)

        # The travelled segment now becomes part of the "mirror" set - shortened
        # to seg_mult of its true length if that's less than 1, though the ray
        # still continues on in the direction just calculated above (from the
        # true, full-length hit point's normal), not recalculated from here.
        if seg_mult < 1.0:
            ex = px + seg_mult * total_t * odx
            ey = py + seg_mult * total_t * ody
        else:
            ex = hx
            ey = hy
        path_points[npts, 0] = ex
        path_points[npts, 1] = ey
        path_point_tags[npts] = TAG_SELF if best_obj >= 0 else TAG_BOUNDARY
        npts += 1
        n_bounces += 1
        dist += seg_mult * total_t
        px = ex
        py = ey
        last_obj = best_obj

        # Pulled back along a segment that passed through edges: the ray is
        # now wherever that is, which may be on the other side of some of
        # them, so re-derive which shapes it is inside from where it actually is.
        if seg_mult < 1.0 and n_cross > 0:
            for oi in range(n_obst):
                inside[oi] = inside_boundary(ex - obst_cx[oi], ey - obst_cy[oi],
                                             obst_shapes[oi], obst_bounds[oi])

        # Confined? Judge the last `confine.lookback` points of the path.
        if is_confined(path_points, npts, confine):
            term = TERM_CONFINED
            break

    raw[RAW_SELF_K] = self_k
    raw[RAW_SELF_M] = self_m
    raw[RAW_DEFLECT] = deflect
    raw[RAW_BOUNCES] = n_bounces
    raw[RAW_TERM] = term
    raw[RAW_LAST_FULL] = last_full
    return term, npts


@njit(cache=True)
def in_sequence(seq, k, path_point_tags):
    """Whether point k of a traced path belongs to the named sequence `seq` (SEQ_*): the one definition of what each
    sequence is, decided by the point's tag alone (`path_point_tags`, as trace_path made them): every end point that is not a
    bounce has a tag of its own, so it is in no bounce sequence."""
    if seq == SEQ_PATH:
        return True
    tag = path_point_tags[k]
    if seq == SEQ_BOUNDARY:
        return tag == TAG_BOUNDARY
    if seq == SEQ_SELF:
        return tag == TAG_SELF
    return tag == TAG_BOUNDARY or tag == TAG_SELF


@njit(cache=True)
def sequence_summary(seq, path_points, npts, path_point_tags, nb, max_iter, out, ray_characteristic_points):
    """Reduce the named sequence `seq` of a traced path, in one pass over its points, to everything that is stored
    about it: its point reductions (PRED_*, into their rows of `ray_characteristic_points`, shape (N_POINTS, 2); a sequence with no points
    reports the path's end point for each) and its scalar reductions (SRED_*, into their slots of `out`: how many
    points it has over the iteration limit, the length and last direction of the polyline through them in order, and
    its share of the path's bounces)."""
    count = 0
    sx = 0.0
    sy = 0.0
    total = 0.0
    prev_x = 0.0
    prev_y = 0.0
    last_dx = 0.0
    last_dy = 0.0
    i_first = -1
    i_last = -1
    i_top = -1
    i_bottom = -1
    i_right = -1
    i_left = -1
    for k in range(npts):
        if not in_sequence(seq, k, path_point_tags):
            continue
        x = path_points[k, 0]
        y = path_points[k, 1]
        sx += x
        sy += y
        if count > 0:
            total += sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            last_dx = x - prev_x
            last_dy = y - prev_y
            if y > path_points[i_top, 1]:
                i_top = k
            if y < path_points[i_bottom, 1]:
                i_bottom = k
            if x > path_points[i_right, 0]:
                i_right = k
            if x < path_points[i_left, 0]:
                i_left = k
        else:
            i_first = k
            i_top = k
            i_bottom = k
            i_right = k
            i_left = k
        prev_x = x
        prev_y = y
        i_last = k
        count += 1
    if count == 0:
        i_first = npts - 1
        i_last = npts - 1
        i_top = npts - 1
        i_bottom = npts - 1
        i_right = npts - 1
        i_left = npts - 1
        mean_x = path_points[npts - 1, 0]
        mean_y = path_points[npts - 1, 1]
    else:
        mean_x = sx / count
        mean_y = sy / count
    ray_characteristic_points[POINT_SLOT[seq, PRED_FIRST], 0] = path_points[i_first, 0]
    ray_characteristic_points[POINT_SLOT[seq, PRED_FIRST], 1] = path_points[i_first, 1]
    ray_characteristic_points[POINT_SLOT[seq, PRED_LAST], 0] = path_points[i_last, 0]
    ray_characteristic_points[POINT_SLOT[seq, PRED_LAST], 1] = path_points[i_last, 1]
    ray_characteristic_points[POINT_SLOT[seq, PRED_MEAN], 0] = mean_x
    ray_characteristic_points[POINT_SLOT[seq, PRED_MEAN], 1] = mean_y
    ray_characteristic_points[POINT_SLOT[seq, PRED_TOP], 0] = path_points[i_top, 0]
    ray_characteristic_points[POINT_SLOT[seq, PRED_TOP], 1] = path_points[i_top, 1]
    ray_characteristic_points[POINT_SLOT[seq, PRED_BOTTOM], 0] = path_points[i_bottom, 0]
    ray_characteristic_points[POINT_SLOT[seq, PRED_BOTTOM], 1] = path_points[i_bottom, 1]
    ray_characteristic_points[POINT_SLOT[seq, PRED_RIGHT], 0] = path_points[i_right, 0]
    ray_characteristic_points[POINT_SLOT[seq, PRED_RIGHT], 1] = path_points[i_right, 1]
    ray_characteristic_points[POINT_SLOT[seq, PRED_LEFT], 0] = path_points[i_left, 0]
    ray_characteristic_points[POINT_SLOT[seq, PRED_LEFT], 1] = path_points[i_left, 1]
    ang = atan2(last_dy, last_dx) if count >= 2 else 0.0
    out[STAT_SLOT[seq, SRED_COUNT]] = min(count / max_iter, 1.0)
    out[STAT_SLOT[seq, SRED_LENGTH]] = atan(total / 5.0) * 2.0 / pi
    out[STAT_SLOT[seq, SRED_LAST_DIR]] = min(max((ang + pi) / (2.0 * pi), 0.0), 1.0)
    out[STAT_SLOT[seq, SRED_SHARE]] = min(count / nb, 1.0) if nb > 0 else 0.0


@njit(cache=True)
def compute_sequences(path_points, npts, path_point_tags, nb, max_iter, out, ray_characteristic_points):
    """Fill every stored reduction of every named sequence (SEQ_*) of a traced path: the points into `ray_characteristic_points` (shape
    (N_POINTS, 2)) and the scalars into `out` (N_STATS slots; compute_run_measurements fills the rest). The bounce points are
    path_points[1] .. path_points[nb]; the start point and a distance-cut end point are not bounce points. A sequence with no points
    (a ray with no bounces of the relevant kind) uses the path's end point. Five of the slots are the old choices
    ST_BOUNCE_COUNT, ST_LENGTH, ST_LAST_DIR, ST_SELF_SHARE and ST_BOUNDARY_SHARE, and thirteen points are the old POINT_*
    choices (see POINT_ALIASES)."""
    for seq in range(N_SEQS):
        sequence_summary(seq, path_points, npts, path_point_tags, nb, max_iter, out, ray_characteristic_points)


@njit(cache=True)
def compute_run_measurements(path_points, npts, raw, max_iter, seg_mult, out):
    """
    Turn a traced path into the metrics of the ray's own run (each in 0..1, written to
    `out`; compute_sequences fills the rest, among them ST_BOUNCE_COUNT, ST_LENGTH,
    ST_LAST_DIR, ST_SELF_SHARE and ST_BOUNDARY_SHARE). Also fills RAW_LENGTH, RAW_LAST, RAW_AREA, RAW_LAST_DIR and
    RAW_DISP in `raw`. The loop-area metric is the loop's area divided by the
    area of a circle with the same perimeter as the loop, so it is
    dimensionless and needs no reference shape (1 is a perfect circle). The point-comparison metrics are written here using
    start->end as a placeholder pair; the real values (using whichever two
    points a channel is actually set to compare) are derived later, once the
    sample points have been computed.
    """
    bounces = int(raw[RAW_BOUNCES])

    # Path lengths
    total = 0.0
    last = 0.0
    for k in range(npts - 1):
        last = sqrt((path_points[k + 1, 0] - path_points[k, 0]) ** 2 + (path_points[k + 1, 1] - path_points[k, 1]) ** 2)
        total += last

    # Direction of the last segment (counterclockwise from pointing right)
    ldx = path_points[npts - 1, 0] - path_points[npts - 2, 0]
    ldy = path_points[npts - 1, 1] - path_points[npts - 2, 1]
    ang = atan2(ldy, ldx)

    # Straight-line distance from the start point to the end point
    sdx = path_points[npts - 1, 0] - path_points[0, 0]
    sdy = path_points[npts - 1, 1] - path_points[0, 1]
    disp = sqrt(sdx * sdx + sdy * sdy)

    # Area of the loop formed when the ray first touches itself: the polygon
    # path_points[k+1] ... path_points[m], where path_points[m] is the hit point on segment k. With
    # the segment length multiplier active, bounce points are systematically
    # pulled back from where the ray's own path actually is, so this polygon
    # no longer has the meaning its name implies - report 0 rather than a
    # number that looks like a real self-intersection area but isn't one.
    area = 0.0
    perimeter = 0.0
    sk = int(raw[RAW_SELF_K])
    sm = int(raw[RAW_SELF_M])
    if sk >= 0 and seg_mult >= 1.0:
        first = sk + 1
        s = 0.0
        for a in range(first, sm + 1):
            b2 = a + 1 if a < sm else first
            s += path_points[a, 0] * path_points[b2, 1] - path_points[b2, 0] * path_points[a, 1]
            perimeter += sqrt((path_points[b2, 0] - path_points[a, 0]) ** 2 + (path_points[b2, 1] - path_points[a, 1]) ** 2)
        area = abs(s) * 0.5

    raw[RAW_LENGTH] = total
    raw[RAW_LAST] = last
    raw[RAW_AREA] = area
    raw[RAW_LAST_DIR] = ang * (180.0 / pi)
    raw[RAW_DISP] = disp

    # Point-comparison metrics: filled here using start->end (this function has
    # no sample-point array to work from); recolor()/component_values() use the
    # actually-chosen point pair instead. Written so the stats array is never
    # left uninitialized, and so the preview list shows a sensible value even
    # for a channel not currently using one of these metrics.
    out[ST_HDIST] = min(abs(sdx) * 0.5, 1.0)
    out[ST_VDIST] = min(abs(sdy) * 0.5, 1.0)
    out[ST_MANHATTAN] = min((abs(sdx) + abs(sdy)) * 0.5, 1.0)
    out[ST_EUCLIDEAN] = min(disp * 0.5, 1.0)

    if bounces > 0:
        out[ST_ANGLES] = min(raw[RAW_DEFLECT] / (180.0 * bounces), 1.0)
    else:                                          # cut off before the first bounce
        out[ST_ANGLES] = 0.0
    if perimeter > 1e-12:
        # the area of a circle with circumference `perimeter` is perimeter^2 / (4 pi)
        out[ST_LOOP_AREA] = min(area / (perimeter * perimeter / (4.0 * pi)), 1.0)
    else:
        out[ST_LOOP_AREA] = 0.0

    term = int(raw[RAW_TERM])
    if term == TERM_ITER:
        out[ST_TERMINATION] = 0.0
    elif term == TERM_CONFINED:
        out[ST_TERMINATION] = 0.2
    elif term == TERM_DIST:
        out[ST_TERMINATION] = 0.4
    elif term == TERM_OBSTACLE:
        out[ST_TERMINATION] = 0.6
    elif term == TERM_PERPENDICULAR:
        out[ST_TERMINATION] = 0.8
    else:                                          # TERM_ESCAPED
        out[ST_TERMINATION] = 1.0

    full = raw[RAW_LAST_FULL]
    if full > 0.0:
        out[ST_LAST_RATIO] = min(last / full, 1.0)
    else:
        out[ST_LAST_RATIO] = 1.0


@njit(cache=True)
def scatter_one_buddha_point(counts, x, y, world_rect):
    """Scatters one point's weight of 1 into `counts` at its own pixel -
    cheap, no blur kernel applied here at all. `counts` covers the world rectangle `world_rect`. Blur (if any) is applied
    once, afterward, to the whole accumulated counts array - see
    blur_counts_image - rather than per point, which is both mathematically
    identical (convolution distributes over sums) and vastly cheaper."""
    h, w = counts.shape
    col = int(floor((x - world_rect.x_start) * (w / (world_rect.x_stop - world_rect.x_start))))
    row = int(floor((world_rect.y_stop - y) * (h / (world_rect.y_stop - world_rect.y_start))))
    if 0 <= row < h and 0 <= col < w:
        counts[row, col] += 1


@njit(cache=True)
def scatter_trace_points(counts, world_rect, path_points, path_point_tags, npts, nb, want_all_self, want_all_mid, want_prog_bounce,
                         want_prog_mid):
    """Scatter into `counts` (in place; it covers the world rectangle `world_rect`) the points of one traced ray that the four
    "multi-point" Buddha options ask for: every self bounce, every segment midpoint, the progressive means of the
    bounce points and of the segment midpoints. (path_points, path_point_tags, npts, nb: as trace_path made them.)"""
    if want_all_self:
        for k in range(1, nb + 1):
            if in_sequence(SEQ_SELF, k, path_point_tags):
                scatter_one_buddha_point(counts, path_points[k, 0], path_points[k, 1], world_rect)

    if want_all_mid:
        for k in range(npts - 1):
            mx = 0.5 * (path_points[k, 0] + path_points[k + 1, 0])
            my = 0.5 * (path_points[k, 1] + path_points[k + 1, 1])
            scatter_one_buddha_point(counts, mx, my, world_rect)

    if want_prog_bounce and nb >= 2:
        sx = path_points[1, 0] + path_points[2, 0]
        sy = path_points[1, 1] + path_points[2, 1]
        scatter_one_buddha_point(counts, sx / 2.0, sy / 2.0, world_rect)
        for c in range(3, nb + 1):
            sx += path_points[c, 0]
            sy += path_points[c, 1]
            scatter_one_buddha_point(counts, sx / c, sy / c, world_rect)

    if want_prog_mid and npts - 1 >= 2:
        sx = 0.5 * (path_points[0, 0] + path_points[1, 0]) + 0.5 * (path_points[1, 0] + path_points[2, 0])
        sy = 0.5 * (path_points[0, 1] + path_points[1, 1]) + 0.5 * (path_points[1, 1] + path_points[2, 1])
        scatter_one_buddha_point(counts, sx / 2.0, sy / 2.0, world_rect)
        cnt = 2
        for k in range(2, npts - 1):
            sx += 0.5 * (path_points[k, 0] + path_points[k + 1, 0])
            sy += 0.5 * (path_points[k, 1] + path_points[k + 1, 1])
            cnt += 1
            scatter_one_buddha_point(counts, sx / cnt, sy / cnt, world_rect)


@njit(cache=True)
def scatter_from_seeds(row0, row1, zone, world_rect, termination, engine_args, rng_seed_base, slots, end_terms,
                       want_all_self, want_all_mid, want_prog_bounce, want_prog_mid, counts):
    """
    Buddha scattering from a grid of seed points. The grid has the shape of `counts` (h rows of w seeds) and is
    laid over the WorldRect `zone` at regular spacings, so its size depends on the output's resolution and not on the
    zone: a seed is the centre of its cell, columns going from zone.x_start to zone.x_stop and rows from
    zone.y_stop (the top row) down to zone.y_start. `counts` itself covers the world rectangle `world_rect` (the image's), which
    is where a scattered point is counted; a zone equal to the world rectangle is exactly the pixels of the image, which is
    what the "camera pixels" seeding uses. For rows [row0, row1), every seed that
    fires a ray (see fires_ray) is traced, with the same fuzzing seed formula as trace_rows, and
    scattered into `counts` (in place; no blur here, it is applied once afterwards): the stored point of each slot
    in `slots` (see POINT_SLOT; an int64 array), the last point if the ray ended for a reason in `end_terms` (TERM_*;
    an int64 array), and whatever the four multi options asked for (see scatter_trace_points).
    Deliberately single-threaded (no prange): a scattered point can land on any pixel, so a parallel version would
    risk concurrent writes. `termination` is a NaturalTerminationRule and `engine_args` a RayArgs (see trace_path).
    """
    max_iter = termination.max_iter
    obstacles = engine_args.obstacles
    fuzzing = engine_args.fuzz.enabled
    h, w = counts.shape
    step_x = (zone.x_stop - zone.x_start) / w
    step_y = (zone.y_stop - zone.y_start) / h
    path_points = np.empty((max_iter + 2, 2))
    path_point_tags = np.empty(max_iter + 2)
    raw = np.empty(N_RAW)
    out = np.empty(N_STATS)
    ray_characteristic_points = np.empty((N_POINTS, 2))
    stored = len(slots) > 0 or len(end_terms) > 0
    for j in range(row0, row1):
        for i in range(w):
            x = zone.x_start + (i + 0.5) * step_x
            y = zone.y_stop - (j + 0.5) * step_y
            if not fires_ray(x, y, obstacles):
                continue
            if fuzzing:
                np.random.seed(rng_seed_base + i * 1000003 + j * 97)
            term, npts = trace_path(x, y, path_points, path_point_tags, termination, engine_args, raw)
            nb = int(raw[RAW_BOUNCES])
            if stored:
                compute_sequences(path_points, npts, path_point_tags, nb, max_iter, out, ray_characteristic_points)
                for slot in slots:
                    scatter_one_buddha_point(counts, ray_characteristic_points[slot, 0], ray_characteristic_points[slot, 1], world_rect)
                for t in end_terms:
                    if term == t:
                        scatter_one_buddha_point(counts, ray_characteristic_points[POINT_END, 0], ray_characteristic_points[POINT_END, 1], world_rect)
            scatter_trace_points(counts, world_rect, path_points, path_point_tags, npts, nb, want_all_self, want_all_mid,
                                 want_prog_bounce, want_prog_mid)


@njit(cache=True)
def scatter_multi_buddha(row0, row1, world_rect, termination, engine_args, rng_seed_base, want_all_self, want_all_mid,
                         want_prog_bounce, want_prog_mid, counts):
    """
    For rows [row0, row1), re-trace every ray that actually fires (starting at the centre of each pixel of
    `counts`' own grid, which covers the world rectangle `world_rect`, and using the exact same seed formula as
    trace_rows, so this reproduces the identical rays already shown, at whatever resolution that
    is) and scatter into `counts` the points of the "multi-point" Buddha options asked for - scatter_from_seeds
    with the world rectangle as its zone and no characteristic points. `termination` is a NaturalTerminationRule and `engine_args` a RayArgs
    (see trace_path).
    """
    no_slots = np.empty(0, dtype=np.int64)
    scatter_from_seeds(row0, row1, world_rect, world_rect, termination, engine_args, rng_seed_base, no_slots, no_slots,
                       want_all_self, want_all_mid, want_prog_bounce, want_prog_mid, counts)


# A numba limitation shapes the parallel driver below: a parallel function cannot take a tuple that contains
# other tuples (a NaturalTerminationRule holds a ConfineRule, a RayArgs holds a FuzzRule, an EmissionRule and an
# ObstacleArrays) and use it inside its prange loop - compiling fails with "'str' object has no attribute 'loc'" (or
# "scalar type ... given for non scalar argument") - but a parallel function that is given their one-level pieces
# can put the whole tuples together again INSIDE its loop (verified). So trace_rows takes the
# rules whole, like trace_path, and hands their pieces to _trace_rows_parallel, which
# rebuilds them for each pixel's trace_path. Any other engine's driver has the same constraint.
@njit(nogil=True, cache=True)
def trace_rows(stats, characteristic_points, pixel_state, row0, row1, out_w, out_h, world_rect, termination, engine_args, rng_seed_base):
    """
    Simulate rows [row0, row1) of an out_w x out_h output grid (the preview image, an exported image, a video frame,
    a thumbnail). For each pixel that actually fires a ray, fill in stats[j, i, :] and points[j, i, :, :] and set
    pixel_state[j, i] = PIXEL_TRACED; for each that does not, set PIXEL_NO_RAY (so no PIXEL_PENDING is left in the
    rows rendered); a ray starts at its pixel's centre. The grid covers the world rectangle `world_rect` (a WorldRect), so an
    output whose shape differs from the rectangle's stretches the picture to fit; the "confined to less than a pixel"
    physics is in world units (the confinement threshold), not in the output's pixels, so a ray behaves identically
    whatever resolution or world rectangle it's sampled with. When fuzzing is enabled, each pixel's ray gets its own
    reproducible seed. `termination` is a NaturalTerminationRule and `engine_args` a RayArgs (see trace_path).
    """
    _trace_rows_parallel(stats, characteristic_points, pixel_state, row0, row1, out_w, out_h, world_rect, termination.max_iter,
                             termination.max_dist, termination.escape_radius, termination.confine, engine_args.seg_mult,
                             engine_args.fuzz, engine_args.emission, engine_args.obstacles, rng_seed_base)


@njit(nogil=True, parallel=True, cache=True)
def _trace_rows_parallel(stats, characteristic_points, pixel_state, row0, row1, out_w, out_h, world_rect, max_iter, max_dist,
                             escape_radius, confine, seg_mult, fuzz, emission, obstacles,
                             rng_seed_base):
    """trace_rows with the rules in pieces (see above)."""
    fuzzing = fuzz.enabled
    x_start = world_rect.x_start
    y_stop = world_rect.y_stop
    step_x = (world_rect.x_stop - world_rect.x_start) / out_w
    step_y = (world_rect.y_stop - world_rect.y_start) / out_h
    n = (row1 - row0) * out_w
    for idx in prange(n):
        j = row0 + idx // out_w
        i = idx % out_w
        x = x_start + (i + 0.5) * step_x
        y = y_stop - (j + 0.5) * step_y
        if not fires_ray(x, y, obstacles):
            pixel_state[j, i] = PIXEL_NO_RAY
            continue
        if fuzzing:
            np.random.seed(rng_seed_base + i * 1000003 + j * 97)
        path_points = np.empty((max_iter + 2, 2))
        path_point_tags = np.empty(max_iter + 2)
        raw = np.empty(N_RAW)
        out = np.empty(N_STATS)
        ray_characteristic_points = np.empty((N_POINTS, 2))
        term, npts = trace_path(x, y, path_points, path_point_tags,
                                NaturalTerminationRule(max_iter, max_dist, escape_radius, confine),
                                RayArgs(seg_mult, fuzz, emission, obstacles), raw)
        compute_run_measurements(path_points, npts, raw, max_iter, seg_mult, out)
        compute_sequences(path_points, npts, path_point_tags, int(raw[RAW_BOUNCES]), max_iter, out, ray_characteristic_points)
        for s in range(N_STATS):
            stats[j, i, s] = out[s]
        for p in range(N_POINTS):
            characteristic_points[j, i, p, 0] = ray_characteristic_points[p, 0]
            characteristic_points[j, i, p, 1] = ray_characteristic_points[p, 1]
        pixel_state[j, i] = PIXEL_TRACED
