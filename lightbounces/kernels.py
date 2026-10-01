"""The simulation kernels: everything numba compiles, and every constant and type they and the code that calls them agree on.

THE RULE (verified: python dev/refactor/numba_cache_check.py): when numba reuses a cached compile it only checks the
modification time of the file the kernel is defined in. So a kernel keeps running STALE code if a constant it reads, or
another @njit function it calls, is changed in a different file. Therefore this one file defines
  * every @njit function (they all call each other),
  * every constant a kernel reads - and the rest of each numbered family (TERM_*, SHAPE_*, ST_*, POINT_*, RAW_*, KIND_*,
    OBSTACLE_EFFECT_*, ...) so that a family is never split between files,
  * the kernel argument types (FuzzRule, ConfineRule, EmissionRule, ObstacleArrays),
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
SIZE = 512                 # image size in pixels
PIXEL = 2.0 / SIZE         # width of one pixel in world units (image spans [-1, 1])

# Why a ray stopped (returned by trace_path as `term`)
TERM_CONFINED = 0   # confined to less than one pixel
TERM_ITER = 1       # iteration limit reached
TERM_DIST = 2       # distance limit reached
TERM_OBSTACLE = 3   # hit an obstacle whose ray effect is "terminate"
TERM_PERPENDICULAR = 4   # hit a reflective surface at exactly perpendicular
                         # incidence (which side it would retrace along is
                         # undefined - see trace_path)
TERM_ESCAPED = 5         # nothing left ahead to hit: the ray leaves for good (its
                         # last point is where it leaves the [-1, 1] image square)

# Shapes
SHAPE_ELLIPSE = 0
SHAPE_POLYGON = 1
SHAPE_MOSS_EGG = 2
SHAPE_SECTOR = 3
SHAPE_SEGMENT = 4
SHAPE_LENS = 5
SHAPE_REULEAUX = 6
SHAPE_CAPSULE = 7
SHAPE_TREFOIL = 8           # from here on: arc loops, which can be concave (see inside_arcs_parity)
SHAPE_QUATREFOIL = 9
SHAPE_LUNE = 10
SHAPE_ARBELOS = 11
SHAPE_FIRST_ARC_LOOP = SHAPE_TREFOIL

# Ray emission direction (a figure-defining setting, not part of fuzzing): each
# ray's base direction is either straight up (90 degrees) or from its start
# pixel towards a target point; the angle offset, positive meaning more
# counterclockwise, is added to that. Fuzzing's emission jitter is added on top.
# The defaults (Up, offset -45) give the up-and-to-the-right (45 degree)
# direction the program has always used.
EMIT_UP = 0
EMIT_TOWARDS = 1

# Confinement: a ray stops (TERM_CONFINED) once its recent path shows it has
# stopped getting anywhere. Two things are chosen: how many of the path's most
# recent points to look at (the window), and the method that judges them:
#   Box          the window's bounding box is smaller than the threshold in BOTH
#                directions
#   Path length  the window's segments (one fewer than its points) add up to
#                less than the threshold
# The threshold is in world units (the image spans [-1, 1]), not pixels, so a
# ray behaves the same at every export resolution. The defaults are what was
# always hard-wired: a 4-point window, the Box method, and one pixel of the
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

KIND_START = 0.0           # a point's "kind", used only for the segment graphic
KIND_BOUNDARY = 1.0
KIND_SELF = 2.0
KIND_DISTCUT = 3.0
KIND_OBSTACLE_TERM = 4.0   # the point where a "terminate" obstacle ended the ray
KIND_ESCAPED = 5.0         # the point where an escaping ray leaves the image square

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
    "window",                                        # how many recent path points are judged (int)
    "method",                                        # CONFINE_BOX / CONFINE_PATH (int)
    "threshold"])                                    # world units
EmissionRule = namedtuple("EmissionRule", [
    "mode",                                          # EMIT_UP / EMIT_TOWARDS (int)
    "target_x", "target_y",                          # the point rays are fired towards
    "offset"])                                       # degrees, positive = counterclockwise

ST_HDIST = 0        # horizontal distance |x1-x2| between two points  [comparison]
ST_VDIST = 1        # vertical distance |y1-y2| between two points    [comparison]
ST_MANHATTAN = 2    # |x1-x2| + |y1-y2| between two points            [comparison]
ST_EUCLIDEAN = 3    # straight-line distance between two points       [comparison]
ST_BOUNCES = 4      # bounces / iteration limit               (linear)
ST_RAY_FRAC = 5     # bounces against ray / bounces           (linear)
ST_BOUNDARY_FRAC = 6  # bounces against the shape / bounces   (linear)
ST_LOOP_AREA = 7    # first self-loop area / same-perimeter circle (linear)
ST_ANGLES = 8       # sum of bounce angles / (180 * bounces)  (linear)
ST_LENGTH = 9       # total ray length                        (arctangent)
ST_LAST_DIR = 10    # direction of the last line segment      (linear angle)
ST_TERMINATION = 11  # why the ray stopped (see TERM_STAT_VALUE)
ST_LAST_RATIO = 12  # last segment length / its uncut length
N_STATS = 13

# Sample points that can be picked out of a ray's path
POINT_START = 0          # the ray's starting point (the pixel it was emitted from)
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
N_POINTS = 13             # the simulation's own per-ray sample point count - NOT the
                         # total number of selectable points; see N_POINT_CHOICES

# Raw measurements filled in by trace_path / compute_stats
RAW_BOUNDARY = 0    # number of bounces off a shape's edge (not off the ray's own path)
RAW_SELF_K = 1      # index of the segment hit at the first self-touch (-1: never)
RAW_SELF_M = 2      # index of the hit point at the first self-touch
RAW_DEFLECT = 3     # sum of deflection angles, degrees
RAW_LENGTH = 4      # total path length
RAW_LAST = 5        # length of the last segment (as drawn)
RAW_AREA = 6        # area of the first self-loop
RAW_LAST_DIR = 7    # direction of the last segment, degrees (-180..180)
RAW_BOUNCES = 8     # number of bounces
RAW_TERM = 9        # why the ray stopped (TERM_*)
RAW_LAST_FULL = 10  # uncut length of a distance-cut last segment (0 if not cut)
RAW_DISP = 11       # straight-line distance from start point to end point
N_RAW = 12

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
def emits_ray(x, y, obstacles):
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
    return arcs_edges_hit(px, py, dx, dy, bound)   # MOSS_EGG, SECTOR, SEGMENT, CAPSULE


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
    t, nx, ny = arcs_edges_hit(px, py, dx, dy, bound)   # MOSS_EGG, SECTOR, SEGMENT, CAPSULE
    if t > 1e29:
        return -1.0, 0.0, 0.0
    return t, nx, ny


@njit(cache=True)
def square_exit_t(px, py, dx, dy):
    """Distance along the unit direction (dx, dy) from (px, py) to where the
    ray leaves the [-1, 1] image square (0 if it is already outside it and
    heading away)."""
    t = 1e30
    if dx > 1e-15:
        t = min(t, (1.0 - px) / dx)
    elif dx < -1e-15:
        t = min(t, (-1.0 - px) / dx)
    if dy > 1e-15:
        t = min(t, (1.0 - py) / dy)
    elif dy < -1e-15:
        t = min(t, (-1.0 - py) / dy)
    return max(t, 0.0)


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
def trace_path(x0, y0, pts, kinds, max_iter, max_dist, seg_mult, fuzz, confine, emission,
               obstacles, raw):
    """
    Trace one ray, storing the start point and every
    bounce point in pts (needs at least max_iter + 2 rows, 2 columns), the
    "kind" of each of those points in kinds (KIND_START / KIND_BOUNDARY /
    KIND_SELF / KIND_DISTCUT / KIND_OBSTACLE_TERM / KIND_ESCAPED, same length
    as pts), and
    some raw measurements in raw (N_RAW entries). If the distance limit is
    reached part-way along a segment, the ray's end point (on that segment) is
    appended to pts (kind KIND_DISTCUT) too.

    `seg_mult` (0 < seg_mult <= 1) shortens every ordinary bounce's recorded
    segment to that fraction of its true length - the endpoint moves back
    towards the segment's start - while the ray still continues on from there
    in the direction the bounce actually calculated (from the true, full-length
    hit point's surface normal), so the ray visibly changes direction in
    mid-air rather than exactly at the boundary/obstacle/its own path. 1 means
    no effect. This only applies to ordinary bounces - a ray that terminates
    (distance limit, a "terminate" obstacle, perpendicular incidence) always
    still reaches its true, unshortened endpoint, since there's no next
    segment to continue into.

    `obstacles` is an ObstacleArrays describing every obstacle (parallel
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
    escapes (TERM_ESCAPED): its last point is where it leaves the [-1, 1]
    image square.

    `confine` is a ConfineRule (see confine_rule): the ray is confined
    (TERM_CONFINED) when the last `window` points of its path are inside a box
    smaller than the threshold both ways (CONFINE_BOX), or their segments add up
    to less than it (CONFINE_PATH).

    `emission` is an EmissionRule (see emission_rule): the ray's direction is its
    base direction - straight up (EMIT_UP), or from its start point towards the
    target (EMIT_TOWARDS) - plus the offset in degrees, positive meaning more
    counterclockwise.

    `fuzz` is a FuzzRule (see fuzz_rule; angles in degrees, emit_jitter is added
    to the emission direction). When enabled, the RNG is used for the jitter
    terms, so the caller must have seeded it beforehand for reproducibility.

    Returns (term, npts):
      term  why the ray stopped: TERM_CONFINED, TERM_ITER, TERM_DIST,
            TERM_OBSTACLE, TERM_PERPENDICULAR or TERM_ESCAPED
      npts  number of valid points in pts (the last is the final location)
    """
    obst_shapes = obstacles.shapes
    obst_bounds = obstacles.bounds
    obst_cx = obstacles.cx
    obst_cy = obstacles.cy
    obst_effects = obstacles.effects
    fuzzing = fuzz.enabled
    n_obst = len(obst_shapes)
    confine_window = confine.window
    confine_method = confine.method
    confine_threshold = confine.threshold

    pts[0, 0] = x0
    pts[0, 1] = y0
    kinds[0] = KIND_START
    npts = 1
    n_bounces = 0
    n_boundary = 0
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
        stop = False
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

            # Distance to earlier path segments: pts[k] -> pts[k+1]. A
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
                ax = pts[k, 0]
                ay = pts[k, 1]
                ex = pts[k + 1, 0] - ax
                ey = pts[k + 1, 1] - ay
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
                # it leaves the image square (never before the last edge it
                # passed through).
                best_obj = ESCAPE_OBJ
                best_t = max(square_exit_t(px, py, dx, dy), t_off) - t_off

            if best_obj <= OBST_OBJ_BASE and hit_eff == OBSTACLE_EFFECT_PASS:
                # Through an edge: flip inside/outside for that shape, carry on
                inside[OBST_OBJ_BASE - best_obj] = not inside[OBST_OBJ_BASE - best_obj]
                t_off += best_t
                n_cross += 1
                if n_cross > max_cross:      # coincident edges etc.: cannot happen normally
                    term = TERM_CONFINED
                    stop = True
                    break
                continue
            break                            # a real end to this segment
        if stop:
            break
        total_t = t_off + best_t             # the segment's full length, crossings included

        # Distance limit: cut this segment off where the limit is reached
        if dist + total_t > max_dist:
            rem = max_dist - dist
            if rem > 1e-12:
                pts[npts, 0] = px + rem * dx
                pts[npts, 1] = py + rem * dy
                kinds[npts] = KIND_DISTCUT
                npts += 1
                last_full = total_t
            term = TERM_DIST
            break

        # Nothing ahead: the ray leaves for good
        if best_obj == ESCAPE_OBJ:
            pts[npts, 0] = px + total_t * dx
            pts[npts, 1] = py + total_t * dy
            kinds[npts] = KIND_ESCAPED
            npts += 1
            term = TERM_ESCAPED
            break

        # A "terminate" edge ends the ray outright, with no reflection
        if best_obj <= OBST_OBJ_BASE and hit_eff == OBSTACLE_EFFECT_TERMINATE:
            pts[npts, 0] = px + total_t * dx
            pts[npts, 1] = py + total_t * dy
            kinds[npts] = KIND_OBSTACLE_TERM
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
            n_boundary += 1                      # any reflecting edge counts as boundary
        else:
            if self_k < 0:                       # first time the ray touches itself
                self_k = best_obj
                self_m = npts                    # index the hit point will occupy
            nx = -(pts[best_obj + 1, 1] - pts[best_obj, 1])
            ny = pts[best_obj + 1, 0] - pts[best_obj, 0]
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
        # confinement in that it's a sudden stop rather than a hit on a
        # boundary/obstacle/distance/iteration limit).
        dot = dx * nx + dy * ny
        if dot > 1.0 - RETRO_EPS or dot < -1.0 + RETRO_EPS:
            pts[npts, 0] = hx
            pts[npts, 1] = hy
            kinds[npts] = KIND_SELF if best_obj >= 0 else KIND_BOUNDARY
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
        pts[npts, 0] = ex
        pts[npts, 1] = ey
        kinds[npts] = KIND_SELF if best_obj >= 0 else KIND_BOUNDARY
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

        # Confined? Judge the last `confine_window` points of the path.
        if npts >= confine_window:
            if confine_method == CONFINE_BOX:
                minx = 1e30
                maxx = -1e30
                miny = 1e30
                maxy = -1e30
                for k in range(npts - confine_window, npts):
                    minx = min(minx, pts[k, 0])
                    maxx = max(maxx, pts[k, 0])
                    miny = min(miny, pts[k, 1])
                    maxy = max(maxy, pts[k, 1])
                if (maxx - minx) < confine_threshold and (maxy - miny) < confine_threshold:
                    term = TERM_CONFINED
                    break
            else:                                  # CONFINE_PATH
                length = 0.0
                for k in range(npts - confine_window, npts - 1):
                    length += sqrt((pts[k + 1, 0] - pts[k, 0]) ** 2
                                   + (pts[k + 1, 1] - pts[k, 1]) ** 2)
                if length < confine_threshold:
                    term = TERM_CONFINED
                    break

    raw[RAW_BOUNDARY] = n_boundary
    raw[RAW_SELF_K] = self_k
    raw[RAW_SELF_M] = self_m
    raw[RAW_DEFLECT] = deflect
    raw[RAW_BOUNCES] = n_bounces
    raw[RAW_TERM] = term
    raw[RAW_LAST_FULL] = last_full
    return term, npts


@njit(cache=True)
def compute_stats(pts, npts, raw, max_iter, seg_mult, out):
    """
    Turn a traced path into the N_STATS metrics (each in 0..1, written to
    `out`). Also fills RAW_LENGTH, RAW_LAST, RAW_AREA, RAW_LAST_DIR and
    RAW_DISP in `raw`. The loop-area metric is the loop's area divided by the
    area of a circle with the same perimeter as the loop, so it is
    dimensionless and needs no reference shape (1 is a perfect circle). The point-comparison metrics are written here using
    start->end as a placeholder pair; the real values (using whichever two
    points a channel is actually set to compare) are derived later, once the
    sample points have been computed.
    """
    bounces = int(raw[RAW_BOUNCES])
    n_boundary = raw[RAW_BOUNDARY]

    # Path lengths
    total = 0.0
    last = 0.0
    for k in range(npts - 1):
        last = sqrt((pts[k + 1, 0] - pts[k, 0]) ** 2 + (pts[k + 1, 1] - pts[k, 1]) ** 2)
        total += last

    # Direction of the last segment (counterclockwise from pointing right)
    ldx = pts[npts - 1, 0] - pts[npts - 2, 0]
    ldy = pts[npts - 1, 1] - pts[npts - 2, 1]
    ang = atan2(ldy, ldx)

    # Straight-line distance from the start point to the end point
    sdx = pts[npts - 1, 0] - pts[0, 0]
    sdy = pts[npts - 1, 1] - pts[0, 1]
    disp = sqrt(sdx * sdx + sdy * sdy)

    # Area of the loop formed when the ray first touches itself: the polygon
    # pts[k+1] ... pts[m], where pts[m] is the hit point on segment k. With
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
            s += pts[a, 0] * pts[b2, 1] - pts[b2, 0] * pts[a, 1]
            perimeter += sqrt((pts[b2, 0] - pts[a, 0]) ** 2 + (pts[b2, 1] - pts[a, 1]) ** 2)
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

    out[ST_BOUNCES] = min(bounces / max_iter, 1.0)
    if bounces > 0:
        out[ST_RAY_FRAC] = (bounces - n_boundary) / bounces
        out[ST_BOUNDARY_FRAC] = n_boundary / bounces
        out[ST_ANGLES] = min(raw[RAW_DEFLECT] / (180.0 * bounces), 1.0)
    else:                                          # cut off before the first bounce
        out[ST_RAY_FRAC] = 0.0
        out[ST_BOUNDARY_FRAC] = 0.0
        out[ST_ANGLES] = 0.0
    if perimeter > 1e-12:
        # the area of a circle with circumference `perimeter` is perimeter^2 / (4 pi)
        out[ST_LOOP_AREA] = min(area / (perimeter * perimeter / (4.0 * pi)), 1.0)
    else:
        out[ST_LOOP_AREA] = 0.0
    out[ST_LENGTH] = atan(total / 5.0) * 2.0 / pi
    out[ST_LAST_DIR] = min(max((ang + pi) / (2.0 * pi), 0.0), 1.0)

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
def compute_points(pts, npts, raw, kinds, sp):
    """
    Pick the N_POINTS candidate sample points out of a traced path and write
    their (x, y) coordinates to sp (shape (N_POINTS, 2)). The bounce points are
    pts[1] .. pts[n_bounces]; the start point and a distance-cut end point are
    not bounce points. A ray with no bounces of the relevant kind uses its end
    point for that mean/extremum (all/boundary/self); a ray with no bounces at
    all uses its end point for every bounce-based choice.
    """
    nb = int(raw[RAW_BOUNCES])
    ex = pts[npts - 1, 0]
    ey = pts[npts - 1, 1]

    sp[POINT_START, 0] = pts[0, 0]
    sp[POINT_START, 1] = pts[0, 1]
    sp[POINT_END, 0] = ex
    sp[POINT_END, 1] = ey

    if nb > 0:
        ax = 0.0
        ay = 0.0
        bx = 0.0
        by = 0.0
        bn = 0
        sxm = 0.0
        sym = 0.0
        sn = 0
        i_r = 1                                    # max x, any kind
        i_l = 1                                    # min x, any kind
        i_t = 1                                    # max y, any kind
        i_b = 1                                    # min y, any kind
        i_rs = -1                                   # max x, self bounces only
        i_ls = -1                                   # min x, self bounces only
        i_ts = -1                                   # max y, self bounces only
        i_bs = -1                                   # min y, self bounces only
        for k in range(1, nb + 1):
            x = pts[k, 0]
            y = pts[k, 1]
            ax += x
            ay += y
            if kinds[k] == KIND_BOUNDARY:
                bx += x
                by += y
                bn += 1
            elif kinds[k] == KIND_SELF:
                sxm += x
                sym += y
                sn += 1
                if i_rs < 0 or x > pts[i_rs, 0]:
                    i_rs = k
                if i_ls < 0 or x < pts[i_ls, 0]:
                    i_ls = k
                if i_ts < 0 or y > pts[i_ts, 1]:
                    i_ts = k
                if i_bs < 0 or y < pts[i_bs, 1]:
                    i_bs = k
            if x > pts[i_r, 0]:
                i_r = k
            if x < pts[i_l, 0]:
                i_l = k
            if y > pts[i_t, 1]:
                i_t = k
            if y < pts[i_b, 1]:
                i_b = k
        sp[POINT_ALL_MEAN, 0] = ax / nb
        sp[POINT_ALL_MEAN, 1] = ay / nb
        if bn > 0:
            sp[POINT_BOUNDARY_MEAN, 0] = bx / bn
            sp[POINT_BOUNDARY_MEAN, 1] = by / bn
        else:
            sp[POINT_BOUNDARY_MEAN, 0] = ex
            sp[POINT_BOUNDARY_MEAN, 1] = ey
        if sn > 0:
            sp[POINT_SELF_MEAN, 0] = sxm / sn
            sp[POINT_SELF_MEAN, 1] = sym / sn
            sp[POINT_SELF_RIGHT, 0] = pts[i_rs, 0]
            sp[POINT_SELF_RIGHT, 1] = pts[i_rs, 1]
            sp[POINT_SELF_LEFT, 0] = pts[i_ls, 0]
            sp[POINT_SELF_LEFT, 1] = pts[i_ls, 1]
            sp[POINT_SELF_TOP, 0] = pts[i_ts, 0]
            sp[POINT_SELF_TOP, 1] = pts[i_ts, 1]
            sp[POINT_SELF_BOTTOM, 0] = pts[i_bs, 0]
            sp[POINT_SELF_BOTTOM, 1] = pts[i_bs, 1]
        else:
            sp[POINT_SELF_MEAN, 0] = ex
            sp[POINT_SELF_MEAN, 1] = ey
            sp[POINT_SELF_RIGHT, 0] = ex
            sp[POINT_SELF_RIGHT, 1] = ey
            sp[POINT_SELF_LEFT, 0] = ex
            sp[POINT_SELF_LEFT, 1] = ey
            sp[POINT_SELF_TOP, 0] = ex
            sp[POINT_SELF_TOP, 1] = ey
            sp[POINT_SELF_BOTTOM, 0] = ex
            sp[POINT_SELF_BOTTOM, 1] = ey
        sp[POINT_RIGHT, 0] = pts[i_r, 0]
        sp[POINT_RIGHT, 1] = pts[i_r, 1]
        sp[POINT_LEFT, 0] = pts[i_l, 0]
        sp[POINT_LEFT, 1] = pts[i_l, 1]
        sp[POINT_TOP, 0] = pts[i_t, 0]
        sp[POINT_TOP, 1] = pts[i_t, 1]
        sp[POINT_BOTTOM, 0] = pts[i_b, 0]
        sp[POINT_BOTTOM, 1] = pts[i_b, 1]
    else:
        for p in range(N_POINTS):
            if p != POINT_START and p != POINT_END:
                sp[p, 0] = ex
                sp[p, 1] = ey


@njit(nogil=True, parallel=True, cache=True)
def render_rows(stats, points, status, row0, row1, max_iter, max_dist, seg_mult, fuzz, confine,
                emission, obstacles, seed_base):
    """
    Simulate rows [row0, row1). For each pixel that actually fires a ray, fill
    in stats[j, i, :] and points[j, i, :, :] and set status[j, i] = 1. When
    fuzzing is enabled, each pixel's ray gets its own reproducible seed.
    """
    fuzzing = fuzz.enabled
    n = (row1 - row0) * SIZE
    for idx in prange(n):
        j = row0 + idx // SIZE
        i = idx % SIZE
        x = (i + 0.5) * PIXEL - 1.0
        y = 1.0 - (j + 0.5) * PIXEL
        if not emits_ray(x, y, obstacles):
            continue
        if fuzzing:
            np.random.seed(seed_base + i * 1000003 + j * 97)
        pts = np.empty((max_iter + 2, 2))
        kinds = np.empty(max_iter + 2)
        raw = np.empty(N_RAW)
        out = np.empty(N_STATS)
        sp = np.empty((N_POINTS, 2))
        term, npts = trace_path(x, y, pts, kinds, max_iter, max_dist, seg_mult, fuzz, confine,
                                emission, obstacles, raw)
        compute_stats(pts, npts, raw, max_iter, seg_mult, out)
        compute_points(pts, npts, raw, kinds, sp)
        for s in range(N_STATS):
            stats[j, i, s] = out[s]
        for p in range(N_POINTS):
            points[j, i, p, 0] = sp[p, 0]
            points[j, i, p, 1] = sp[p, 1]
        status[j, i] = 1


@njit(cache=True)
def scatter_one_buddha_point(counts, x, y, w, h):
    """Scatters one point's weight of 1 into `counts` at its own pixel -
    cheap, no blur kernel applied here at all. Blur (if any) is applied
    once, afterward, to the whole accumulated counts array - see
    blur_counts_image - rather than per point, which is both mathematically
    identical (convolution distributes over sums) and vastly cheaper."""
    col = int(floor((x + 1.0) * (w / 2.0)))
    row = int(floor((1.0 - y) * (h / 2.0)))
    if 0 <= row < h and 0 <= col < w:
        counts[row, col] += 1


@njit(cache=True)
def scatter_multi_buddha(row0, row1, max_iter, max_dist, seg_mult, fuzz, confine, emission,
                         obstacles, seed_base, want_all_self, want_all_mid,
                         want_prog_bounce, want_prog_mid, counts):
    """
    For rows [row0, row1), re-trace every ray that actually fires (starting at the
    centre of each pixel of `counts`' own grid, and using the exact same seed
    formula as render_rows / render_rows_wh, so this reproduces the identical
    rays already shown, at whatever resolution that is) and, for whichever of the four "multi-point" Buddha
    options are requested, scatter every one of that option's points into
    `counts` (in place) - cheap, no blur kernel applied here; blur (if any)
    is applied once, afterward, to the whole accumulated counts array.
    Deliberately single-threaded (no prange): a scattered point can land on
    any pixel, not just ones in this row range, so a parallel version would
    risk concurrent writes to the same pixel.
    """
    fuzzing = fuzz.enabled
    h, w = counts.shape
    step_x = 2.0 / w                         # the grid of `counts` covers [-1, 1] on both axes
    step_y = 2.0 / h                         # (exactly render_rows_wh's rays, at any resolution)
    pts = np.empty((max_iter + 2, 2))
    kinds = np.empty(max_iter + 2)
    raw = np.empty(N_RAW)
    for j in range(row0, row1):
        for i in range(w):
            x = (i + 0.5) * step_x - 1.0
            y = 1.0 - (j + 0.5) * step_y
            if not emits_ray(x, y, obstacles):
                continue
            if fuzzing:
                np.random.seed(seed_base + i * 1000003 + j * 97)
            _term, npts = trace_path(x, y, pts, kinds, max_iter, max_dist, seg_mult, fuzz,
                                     confine, emission, obstacles, raw)
            nb = int(raw[RAW_BOUNCES])

            if want_all_self:
                for k in range(1, nb + 1):
                    if kinds[k] == KIND_SELF:
                        scatter_one_buddha_point(counts, pts[k, 0], pts[k, 1], w, h)

            if want_all_mid:
                for k in range(npts - 1):
                    mx = 0.5 * (pts[k, 0] + pts[k + 1, 0])
                    my = 0.5 * (pts[k, 1] + pts[k + 1, 1])
                    scatter_one_buddha_point(counts, mx, my, w, h)

            if want_prog_bounce and nb >= 2:
                sx = pts[1, 0] + pts[2, 0]
                sy = pts[1, 1] + pts[2, 1]
                scatter_one_buddha_point(counts, sx / 2.0, sy / 2.0, w, h)
                for c in range(3, nb + 1):
                    sx += pts[c, 0]
                    sy += pts[c, 1]
                    scatter_one_buddha_point(counts, sx / c, sy / c, w, h)

            if want_prog_mid and npts - 1 >= 2:
                sx = 0.5 * (pts[0, 0] + pts[1, 0]) + 0.5 * (pts[1, 0] + pts[2, 0])
                sy = 0.5 * (pts[0, 1] + pts[1, 1]) + 0.5 * (pts[1, 1] + pts[2, 1])
                scatter_one_buddha_point(counts, sx / 2.0, sy / 2.0, w, h)
                cnt = 2
                for k in range(2, npts - 1):
                    sx += 0.5 * (pts[k, 0] + pts[k + 1, 0])
                    sy += 0.5 * (pts[k, 1] + pts[k + 1, 1])
                    cnt += 1
                    scatter_one_buddha_point(counts, sx / cnt, sy / cnt, w, h)


@njit(nogil=True, parallel=True, cache=True)
def render_rows_wh(stats, points, status, row0, row1, out_w, out_h, max_iter, max_dist,
                   seg_mult, fuzz, confine, emission, obstacles, seed_base):
    """
    Like render_rows, but for an arbitrary out_w x out_h output grid (used for
    video frames) instead of the fixed SIZE x SIZE main image. The grid still
    covers the same [-1, 1] world extent on both axes, so a non-square output
    stretches the shape to fit; the "confined to less than a pixel" physics
    stays tied to the main image's fixed PIXEL size, not to out_w/out_h, so a
    ray behaves identically regardless of the resolution it's sampled at.
    """
    fuzzing = fuzz.enabled
    step_x = 2.0 / out_w
    step_y = 2.0 / out_h
    n = (row1 - row0) * out_w
    for idx in prange(n):
        j = row0 + idx // out_w
        i = idx % out_w
        x = (i + 0.5) * step_x - 1.0
        y = 1.0 - (j + 0.5) * step_y
        if not emits_ray(x, y, obstacles):
            continue
        if fuzzing:
            np.random.seed(seed_base + i * 1000003 + j * 97)
        pts = np.empty((max_iter + 2, 2))
        kinds = np.empty(max_iter + 2)
        raw = np.empty(N_RAW)
        out = np.empty(N_STATS)
        sp = np.empty((N_POINTS, 2))
        term, npts = trace_path(x, y, pts, kinds, max_iter, max_dist, seg_mult, fuzz, confine,
                                emission, obstacles, raw)
        compute_stats(pts, npts, raw, max_iter, seg_mult, out)
        compute_points(pts, npts, raw, kinds, sp)
        for s in range(N_STATS):
            stats[j, i, s] = out[s]
        for p in range(N_POINTS):
            points[j, i, p, 0] = sp[p, 0]
            points[j, i, p, 1] = sp[p, 1]
        status[j, i] = 1
