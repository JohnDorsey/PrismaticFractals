"""Shape geometry: the packed "bound" arrays the simulation kernels take for each kind of shape, the functions that build
them from a shape's parameters (ellipse_bound ... arbelos_bound), and the helpers that probe or sample them.

The simulation kernels receive a shape code and a float array `bound`:
  bound[0]   area of the shape
  ellipse:   bound[1:4] = m11, m12, m22, where a point is inside when
             m11*x^2 + 2*m12*x*y + m22*y^2 < 1
  polygon:   bound[1] = apothem (distance from centre to each edge),
             bound[2] = number of sides n,
             bound[3+2k], bound[4+2k] = outward unit normal of edge k
             (a point is inside when normal_k . p < apothem for every k)
The other layouts are described above ARC_FIELDS, and the squircle and spiral ones above superellipse_bound. Imports only constants and
kernels: shapes.py (the Shape classes) sits above."""
from math import acos, atan2, cos, gamma, pi, radians, sin, sqrt, tan
import numpy as np
from .constants import (
    ARC_SPAN_MAX, ARC_SPAN_MIN, AXIS_MIN, LOBE_SIZE_MAX, MOSS_ANGLE_MAX, MOSS_ANGLE_MIN,
    SECTOR_SPAN_MAX, SPIRAL_FILL_MAX, SPIRAL_FILL_MIN, SPIRAL_PITCH_MAX, SPIRAL_PITCH_MIN, SPLIT_MAX,
    EGG_SHIFT_MAX, EGG_SHIFT_MIN, FLOWER_AMPLITUDE_MAX, FLOWER_AMPLITUDE_MIN, FLOWER_MAX_PETALS,
    FLOWER_MIN_PETALS, SNAIL_MAX_SEGMENTS, SNAIL_MIN_SEGMENTS, SPLIT_MIN, SQUARENESS_MAX, SQUARENESS_MIN, SUPER_EXPONENT_MAX,
    SUPER_EXPONENT_MIN
)
from .kernels import (
    SHAPE_ARBELOS, SHAPE_FG_SQUIRCLE, SHAPE_LUNE, SHAPE_MOSS_EGG, SHAPE_REULEAUX, SHAPE_SNAIL,
    SHAPE_SUPERELLIPSE, boundary_hit, inside_boundary
)


def ellipse_bound(width, height, rotation):
    """
    Boundary description of an ellipse with full extents `width` x `height`,
    turned counterclockwise by `rotation` degrees. Written as p^T M p < 1 with
    M = R diag(1/a^2, 1/b^2) R^T (a, b = semi-axes). 2.0 x 2.0 with no rotation
    is exactly the unit circle (M = identity).
    """
    a = max(width * 0.5, AXIS_MIN)
    b = max(height * 0.5, AXIS_MIN)
    th = radians(rotation)
    c, s = cos(th), sin(th)
    ia, ib = 1.0 / (a * a), 1.0 / (b * b)
    return np.array([pi * a * b,
                     c * c * ia + s * s * ib,
                     c * s * (ia - ib),
                     s * s * ia + c * c * ib], dtype=np.float64)


def polygon_bound(sides, rotation, radius=1.0):
    """
    Boundary description of a regular polygon with `sides` sides whose
    vertices lie on a circle of the given `radius` (1.0 = the unit circle). With
    rotation 0 one vertex points
    straight up; `rotation` (degrees) turns it counterclockwise.
    """
    n = int(sides)
    th = radians(rotation)
    bound = np.empty(3 + 2 * n, dtype=np.float64)
    bound[0] = 0.5 * n * radius * radius * sin(2.0 * pi / n)   # area
    bound[1] = radius * cos(pi / n)                            # apothem
    bound[2] = n
    for k in range(n):
        # vertex k sits at angle pi/2 + th + 2*pi*k/n; edge k's normal lies
        # half a step further round
        ang = pi / 2.0 + th + pi / n + 2.0 * pi * k / n
        bound[3 + 2 * k] = cos(ang)
        bound[4 + 2 * k] = sin(ang)
    return bound


# shape families. Both keep bound[0] = area, same as ellipse/polygon.
#
# "Arcs + edges" (Moss's egg, circular sector, circular segment, capsule): a
# small closed loop of circular arcs and straight, bounded line segments.
#   bound[1] = number of arcs (n_a); bound[2] = number of edges (n_e)
#   then n_a * (cx, cy, r, ang_lo, span) - a point is "on" the arc when its
#   angle from (cx, cy), normalized into [ang_lo, ang_lo + span), is in range
#   then n_e * (ax, ay, bx, by) - a bounded segment from (ax, ay) to (bx, by)
#
# "Arc loops" (trefoil, quatrefoil, lune, arbelos) use the same arrays as "arcs + edges"
# (with no edges), but the loop may be concave, so a point is inside by the even-odd rule
# (inside_arcs_parity), not by an AND of circles or a ray from the origin.
#
# "Circle intersection" (two-circle lens, Reuleaux polygon): the shape is
# exactly the intersection of a few full circles (so its ray-exit is just the
# minimum of each circle's own, and its obstacle-entry the same style of
# interval-intersection already used for the polygon obstacle).
#   bound[1] = number of circles (n_c); then n_c * (cx, cy, r)
ARC_FIELDS = 5
EDGE_FIELDS = 4
CIRCLE_FIELDS = 3


def pack_arcs_edges(area, arcs, edges):
    """arcs: list of (cx, cy, r, ang_lo, span) with angles in radians. edges:
    list of (ax, ay, bx, by)."""
    n_a, n_e = len(arcs), len(edges)
    bound = np.zeros(3 + ARC_FIELDS * n_a + EDGE_FIELDS * n_e, dtype=np.float64)
    bound[0] = area
    bound[1] = n_a
    bound[2] = n_e
    idx = 3
    for cx, cy, r, ang_lo, span in arcs:
        bound[idx:idx + 5] = (cx, cy, r, ang_lo, span)
        idx += 5
    for ax, ay, bx, by in edges:
        bound[idx:idx + 4] = (ax, ay, bx, by)
        idx += 4
    return bound


def pack_circles(area, circles):
    """circles: list of (cx, cy, r)."""
    n = len(circles)
    bound = np.zeros(2 + CIRCLE_FIELDS * n, dtype=np.float64)
    bound[0] = area
    bound[1] = n
    idx = 2
    for cx, cy, r in circles:
        bound[idx:idx + 3] = (cx, cy, r)
        idx += 3
    return bound


def _rot2(x, y, cos_t, sin_t):
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def estimate_area_numeric(shape, bound, extent, grid=150):
    """Fallback area estimate (grid sampling over a square of half-width
    `extent`) for shapes without a simple closed-form area - cheap enough
    since it only runs once per parameter change, not per ray (inside_boundary
    is the same compiled kernel the simulation itself uses)."""
    step = 2.0 * extent / grid
    count = 0
    for j in range(grid):
        y = (j + 0.5) * step - extent
        for i in range(grid):
            x = (i + 0.5) * step - extent
            if inside_boundary(x, y, shape, bound):
                count += 1
    return count * step * step


MOSS_FIT_SCALE = 0.8   # shrinks Moss's egg so its largest apex angle (180deg,
                       # its flattest/roundest case) leaves a comfortable
                       # margin inside the viewport, rather than exactly
                       # touching its edge


def moss_egg_bound(apex_angle_deg, size=1.0, rotation_deg=0.0):
    """
    Moss's egg (see Wikipedia): an oval of 4 tangent circular arcs, built from
    an isosceles triangle ABC with apex angle ACB = `apex_angle_deg` (60..180).
    AB is normalized to length 2, then the whole shape is re-centred on the
    vertical middle of its own bounding box (rather than left centred on M,
    the base semicircle's own centre, which sits well below the shape's true
    middle), scaled down for a comfortable fit - see MOSS_FIT_SCALE - and then
    by the user-facing `size` on top of that. With no rotation the egg's
    round end is at the bottom (-y), its pointed/capped end at the top (+y).
    """
    angle = max(min(apex_angle_deg, MOSS_ANGLE_MAX), MOSS_ANGLE_MIN)
    half = radians(angle) / 2.0
    L = 1.0 / sin(half)                        # CA = CB
    h = 1.0 / tan(half)                        # height of C above AB
    Ax, Ay = -1.0, 0.0
    Bx, By = 1.0, 0.0
    Cx, Cy = 0.0, h
    Mx, My = 0.0, 0.0
    ux, uy = (Cx - Bx) / L, (Cy - By) / L      # unit vector B -> C
    Dx, Dy = Bx + 2.0 * ux, By + 2.0 * uy
    vx, vy = (Cx - Ax) / L, (Cy - Ay) / L      # unit vector A -> C
    Ex, Ey = Ax + 2.0 * vx, Ay + 2.0 * vy
    CD = abs(2.0 - L)                          # = |C-D| = |C-E| (tangent radius)

    def arc_params(cx, cy, start_x, start_y, end_x, end_y, r):
        a_start = atan2(start_y - cy, start_x - cx)
        a_end = atan2(end_y - cy, end_x - cx)
        ang_lo = a_end
        span = (a_start - a_end) % (2.0 * pi)
        return (cx, cy, r, ang_lo, span)

    # Traversal order (clockwise, per the construction): B -> A (semicircle)
    # -> D (B-centred arc) -> E (C-centred arc, the "cap") -> B (A-centred arc)
    arcs = [
        arc_params(Mx, My, Bx, By, Ax, Ay, 1.0),
        arc_params(Bx, By, Ax, Ay, Dx, Dy, 2.0),
    ]
    if CD > 1e-9:
        arcs.append(arc_params(Cx, Cy, Dx, Dy, Ex, Ey, CD))
    arcs.append(arc_params(Ax, Ay, Ex, Ey, Bx, By, 2.0))

    # Re-centre vertically (the shape spans y = -1 at the bottom to y = h+CD
    # at the top; shift so those are equally far from the local origin) and
    # apply the comfortable-fit scale and the user-facing size on top of it.
    mid_y = (h + CD - 1.0) / 2.0
    s = MOSS_FIT_SCALE * size
    arcs = [(s * cx, s * (cy - mid_y), s * r, ang_lo, span)
           for cx, cy, r, ang_lo, span in arcs]

    rot = radians(rotation_deg)
    if rot != 0.0:
        c, sn = cos(rot), sin(rot)
        arcs = [(cx * c - cy * sn, cx * sn + cy * c, r, ang_lo + rot, span)
               for cx, cy, r, ang_lo, span in arcs]

    bound = pack_arcs_edges(0.0, arcs, [])
    bound[0] = estimate_area_numeric(SHAPE_MOSS_EGG, bound, 2.2 * size)
    return bound


def sector_bound(radius, span_deg, rotation_deg):
    """Circular sector (pie slice): apex at the local origin, radius `radius`,
    angular width `span_deg` (up to 180 degrees), bisector pointing at
    `rotation_deg`."""
    r = max(radius, 1e-6)
    span = radians(max(min(span_deg, SECTOR_SPAN_MAX), ARC_SPAN_MIN))
    rot = radians(rotation_deg)
    ang_lo = rot - span / 2.0
    ang_hi = rot + span / 2.0
    p_lo = (r * cos(ang_lo), r * sin(ang_lo))
    p_hi = (r * cos(ang_hi), r * sin(ang_hi))
    arcs = [(0.0, 0.0, r, ang_lo, span)]
    edges = [(0.0, 0.0, p_lo[0], p_lo[1]), (p_hi[0], p_hi[1], 0.0, 0.0)]
    area = 0.5 * r * r * span
    return pack_arcs_edges(area, arcs, edges)


def segment_bound(radius, span_deg, rotation_deg):
    """Circular segment: the "cap" of a circle of radius `radius` beyond a
    chord subtending `span_deg` at the centre, facing `rotation_deg`."""
    r = max(radius, 1e-6)
    span = radians(max(min(span_deg, ARC_SPAN_MAX), ARC_SPAN_MIN))
    rot = radians(rotation_deg)
    ang_lo = rot - span / 2.0
    ang_hi = rot + span / 2.0
    p_lo = (r * cos(ang_lo), r * sin(ang_lo))
    p_hi = (r * cos(ang_hi), r * sin(ang_hi))
    arcs = [(0.0, 0.0, r, ang_lo, span)]
    edges = [(p_hi[0], p_hi[1], p_lo[0], p_lo[1])]
    area = 0.5 * r * r * (span - sin(span))
    return pack_arcs_edges(area, arcs, edges)


def lens_bound(radius1, radius2, separation, rotation_deg):
    """Two-circle lens: the intersection of two circles of radii `radius1`,
    `radius2`, whose centres are `separation` apart along `rotation_deg`."""
    r1 = max(radius1, 1e-6)
    r2 = max(radius2, 1e-6)
    d = max(separation, 0.0)
    rot = radians(rotation_deg)
    c, s = cos(rot), sin(rot)
    c1 = (-0.5 * d * c, -0.5 * d * s)
    c2 = (0.5 * d * c, 0.5 * d * s)
    if d >= r1 + r2:
        area = 0.0                              # circles don't meet
    elif d <= abs(r1 - r2):
        area = pi * min(r1, r2) ** 2             # one circle sits inside the other
    else:
        a1 = acos(min(1.0, max(-1.0, (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))))
        a2 = acos(min(1.0, max(-1.0, (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))))
        tri = 0.5 * sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
        area = r1 * r1 * a1 + r2 * r2 * a2 - tri
    return pack_circles(area, [(c1[0], c1[1], r1), (c2[0], c2[1], r2)])


def reuleaux_bound(sides, rotation, radius=1.0):
    """Reuleaux polygon on a regular `sides`-gon (odd, >= 3) of circumradius
    `radius`: the intersection of `sides` circles, one centred at each vertex,
    each with the polygon's constant width as its radius."""
    n = int(sides)
    if n % 2 == 0:
        n += 1                                  # must be odd; nudge up
    n = max(n, 3)
    th = radians(rotation)
    w = 2.0 * radius * sin(pi * (n - 1) / (2.0 * n))   # constant width
    circles = []
    for k in range(n):
        ang = pi / 2.0 + th + 2.0 * pi * k / n
        circles.append((radius * cos(ang), radius * sin(ang), w))
    bound = pack_circles(0.0, circles)
    bound[0] = estimate_area_numeric(SHAPE_REULEAUX, bound, radius + w)
    return bound


def capsule_bound(length, radius, rotation_deg):
    """Capsule / stadium: a rectangle of the given `length` (the straight
    part) capped with a semicircle of `radius` at each end, oriented along
    `rotation_deg`."""
    ln = max(length, 0.0)
    r = max(radius, 1e-6)
    rot = radians(rotation_deg)
    c, s = cos(rot), sin(rot)
    half = ln / 2.0
    rx, ry = _rot2(half, 0.0, c, s)             # right cap centre
    lx, ly = _rot2(-half, 0.0, c, s)            # left cap centre
    # Local (pre-rotation) corner points, then rotated into place
    br = _rot2(half, -r, c, s)                  # bottom-right
    tr = _rot2(half, r, c, s)                   # top-right
    tl = _rot2(-half, r, c, s)                  # top-left
    bl = _rot2(-half, -r, c, s)                 # bottom-left
    arcs = [
        (rx, ry, r, rot - pi / 2.0, pi),        # right cap: -90deg..+90deg (rotated)
        (lx, ly, r, rot + pi / 2.0, pi),        # left cap: +90deg..+270deg (rotated)
    ]
    edges = [
        (bl[0], bl[1], br[0], br[1]),           # bottom edge, left -> right
        (tr[0], tr[1], tl[0], tl[1]),           # top edge, right -> left
    ]
    area = 2.0 * r * ln + pi * r * r
    return pack_arcs_edges(area, arcs, edges)


def region_arcs(circles, member):
    """The boundary of a region built from full circles [(cx, cy, r), ...], as a
    list of arcs (cx, cy, r, ang_lo, span): `member(x, y)` says whether a point is in
    the region. Each circle is cut where it crosses the others, and a piece is kept
    when the region is on one side of it only (probed just inside and just outside)."""
    unique = []
    for circle in circles:
        if not any(all(abs(circle[i] - other[i]) < 1e-9 for i in range(3)) for other in unique):
            unique.append(circle)
    tau = 2.0 * pi
    arcs = []
    for ci, (cx, cy, r) in enumerate(unique):
        cuts = [0.0]
        for cj, (ox, oy, other_r) in enumerate(unique):
            d = sqrt((ox - cx) ** 2 + (oy - cy) ** 2)
            if cj == ci or d >= r + other_r or d <= abs(r - other_r) or d == 0.0:
                continue
            half = acos(max(-1.0, min(1.0, (r * r - other_r * other_r + d * d) / (2.0 * d * r))))
            base = atan2(oy - cy, ox - cx)
            cuts += [(base + half) % tau, (base - half) % tau]
        cuts = sorted(set(round(a, 12) for a in cuts)) + [tau]
        eps = 1e-6 * max(r, 1e-3)
        kept = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi - lo < 1e-9:
                continue
            mid = 0.5 * (lo + hi)
            ux, uy = cos(mid), sin(mid)
            if member(cx + (r + eps) * ux, cy + (r + eps) * uy) != \
                    member(cx + (r - eps) * ux, cy + (r - eps) * uy):
                kept.append([lo, hi - lo])
        if len(kept) > 1 and kept[0][0] == 0.0 and abs(kept[-1][0] + kept[-1][1] - tau) < 1e-9:
            kept[-1][1] += kept.pop(0)[1]              # join the pieces either side of angle 0
        arcs += [(cx, cy, r, lo, span) for lo, span in kept]
    return arcs


def arc_loop_bound(shape, arcs, extent):
    """Pack `arcs` as an arc-loops shape of kind `shape`; its area is estimated."""
    bound = pack_arcs_edges(0.0, arcs, [])
    bound[0] = estimate_area_numeric(shape, bound, extent)
    return bound


def lobed_bound(shape, lobes, radius, lobe_size, rotation):
    """A trefoil / quatrefoil: the union of `lobes` equal circles whose centres lie on a
    circle round the origin, one pointing straight up at rotation 0. The farthest point of
    the shape is `radius` from the origin and each lobe's radius is `lobe_size` * `radius`."""
    big = max(radius, 1e-6)
    size = max(min(lobe_size, LOBE_SIZE_MAX), 0.5)
    r = size * big
    ring = big - r
    circles = []
    for k in range(lobes):
        ang = pi / 2.0 + radians(rotation) + 2.0 * pi * k / lobes
        circles.append((ring * cos(ang), ring * sin(ang), r))

    def member(x, y):
        return any((x - cx) ** 2 + (y - cy) ** 2 < rr * rr for cx, cy, rr in circles)
    return arc_loop_bound(shape, region_arcs(circles, member), 1.05 * big)


def lune_bound(radius, cut_radius, cut_offset, rotation):
    """A lune (crescent): the disc of radius `radius` about the origin, minus a second disc of
    radius `cut_radius` whose centre is `cut_offset` away, in the direction `rotation` (0 = +x)."""
    big = max(radius, 1e-6)
    cut = max(cut_radius, 1e-6)
    rot = radians(rotation)
    cut_cx, cut_cy = max(cut_offset, 0.0) * cos(rot), max(cut_offset, 0.0) * sin(rot)
    circles = [(0.0, 0.0, big), (cut_cx, cut_cy, cut)]

    def member(x, y):
        return x * x + y * y < big * big and (x - cut_cx) ** 2 + (y - cut_cy) ** 2 >= cut * cut
    return arc_loop_bound(SHAPE_LUNE, region_arcs(circles, member), 1.05 * big)


def arbelos_bound(radius, split, rotation):
    """An arbelos: the half-disc on a diameter of length 2 * `radius`, minus the half-discs on the two
    parts the diameter is split into (`split` is the first part's share). It is centred
    vertically on its bounding box, with the curved side up at rotation 0."""
    big = max(radius, 1e-6)
    share = max(min(split, SPLIT_MAX), SPLIT_MIN)
    small_a, small_b = share * big, (1.0 - share) * big          # the small radii (they sum to `big`)
    shift = -0.5 * big
    rot = radians(rotation)
    cs, sn = cos(rot), sin(rot)
    arcs = []
    for cx, r in ((0.0, big), (-big + small_a, small_a), (big - small_b, small_b)):
        arcs.append((cx * cs - shift * sn, cx * sn + shift * cs, r, rot, pi))
    return arc_loop_bound(SHAPE_ARBELOS, arcs, 1.05 * big)


# "Squircle" and "spiral" layouts: a convex curve given by an implicit function (the superelliptical and
# Fernandez-Guasti squircles) and the Archimedean spiral ribbon. Their rays are found by root search in the
# shape's own frame (see kernels.squircle_f and kernels.spiral_edges_hit), so each is six numbers:
#   bound[0] = area; bound[1], bound[2] = cos, sin of the rotation;
#   superelliptical squircle  bound[3], bound[4] = semi-axes a, b; bound[5] = the exponent n
#                             (inside when |x/a|^n + |y/b|^n < 1)
#   Fernandez-Guasti squircle bound[3] = bound[4] = the radius r; bound[5] = the squareness s
#                             (inside when x^2 + y^2 - (s^2/r^2) x^2 y^2 < r^2, within the box |x|, |y| < r)
#   spiral ribbon             bound[3] = b, bound[4] = w, bound[5] = theta_max: the points (rho, phi) with
#                             b*theta <= rho < b*theta + w for some theta in [0, theta_max] that is phi plus whole
#                             turns (theta_max 0: the shape is empty)


def superellipse_bound(width, height, exponent, rotation):
    """A superelliptical squircle |x/a|^n + |y/b|^n < 1 with full extents `width` x `height` (a, b the half
    of each) and exponent n (2: an ellipse, 1: a diamond, large: a rectangle), turned counterclockwise by
    `rotation` degrees. Its area is exact: 4ab * gamma(1 + 1/n)^2 / gamma(1 + 2/n)."""
    a = max(width * 0.5, AXIS_MIN)
    b = max(height * 0.5, AXIS_MIN)
    n = min(max(exponent, SUPER_EXPONENT_MIN), SUPER_EXPONENT_MAX)
    th = radians(rotation)
    return np.array([4.0 * a * b * gamma(1.0 + 1.0 / n) ** 2 / gamma(1.0 + 2.0 / n),
                     cos(th), sin(th), a, b, n], dtype=np.float64)


def fg_squircle_bound(radius, squareness, rotation):
    """A Fernandez-Guasti squircle x^2 + y^2 - (s^2/r^2) x^2 y^2 < r^2 of radius r and squareness s (0: the
    circle of radius r; 1: the square of side 2r), turned counterclockwise by `rotation` degrees. Along the
    direction at angle phi its edge is at rho^2 = 2 r^2 / (1 + sqrt(1 - s^2 sin^2(2 phi))), and the area is half
    the integral of rho^2 over the turn (a midpoint sum)."""
    r = max(radius, AXIS_MIN)
    s = min(max(squareness, SQUARENESS_MIN), SQUARENESS_MAX)
    th = radians(rotation)
    steps = 4096
    phi = (np.arange(steps) + 0.5) * (2.0 * pi / steps)
    rho2 = 2.0 * r * r / (1.0 + np.sqrt(1.0 - (s * np.sin(2.0 * phi)) ** 2))
    return np.array([0.5 * rho2.sum() * (2.0 * pi / steps), cos(th), sin(th), r, r, s], dtype=np.float64)


def spiral_bound(radius, pitch, fill, rotation):
    """An Archimedean spiral ribbon: it winds out from the centre, its inner edge rho = b*theta and its
    outer edge rho = b*theta + w, `pitch` (= 2 pi b) apart from one turn to the next, `fill` * pitch wide (w), until
    the outer edge reaches `radius` from the centre (theta_max = (radius - w) / b, never below 0: a radius under
    the ribbon's width leaves nothing). It starts along +x at rotation 0 and winds counterclockwise; each end is a
    straight cut across the ribbon. The area is the integral of ((b theta + w)^2 - (b theta)^2) / 2 up to
    theta_max."""
    p = min(max(pitch, SPIRAL_PITCH_MIN), SPIRAL_PITCH_MAX)
    f = min(max(fill, SPIRAL_FILL_MIN), SPIRAL_FILL_MAX)
    b = p / (2.0 * pi)
    w = f * p
    theta_max = (radius - w) / b
    if theta_max <= 1e-9:
        theta_max = 0.0
    th = radians(rotation)
    return np.array([0.5 * (b * w * theta_max ** 2 + w * w * theta_max), cos(th), sin(th), b, w, theta_max],
                    dtype=np.float64)


def squircle_outline_xy(shape, bound, n=361):
    """(x, y) arrays tracing a squircle's edge exactly (not by casting rays): the superellipse as
    x = a sgn(cos t)|cos t|^(2/n), y = b sgn(sin t)|sin t|^(2/n); the Fernandez-Guasti squircle by its edge
    distance along each direction; each turned by the rotation."""
    t = np.linspace(0.0, 2.0 * pi, n)
    if shape == SHAPE_SUPERELLIPSE:
        ct, st = np.cos(t), np.sin(t)
        x = bound[3] * np.sign(ct) * np.abs(ct) ** (2.0 / bound[5])
        y = bound[4] * np.sign(st) * np.abs(st) ** (2.0 / bound[5])
    else:
        rho = bound[3] * np.sqrt(2.0 / (1.0 + np.sqrt(1.0 - (bound[5] * np.sin(2.0 * t)) ** 2)))
        x, y = rho * np.cos(t), rho * np.sin(t)
    return bound[1] * x - bound[2] * y, bound[2] * x + bound[1] * y


def spiral_outline_xy(bound, per_turn=90):
    """(x, y) arrays tracing the spiral ribbon's edge: out along the inner edge, across the far end, back along
    the outer edge and across the near end to the start. A shape too small to hold its ribbon is the single point
    at the centre."""
    b, w, theta_max = bound[3], bound[4], bound[5]
    if theta_max <= 0.0:
        return np.zeros(2), np.zeros(2)
    theta = np.linspace(0.0, theta_max, 8 + int(per_turn * theta_max / (2.0 * pi)))
    radius = np.concatenate((b * theta, (b * theta + w)[::-1]))
    angle = np.concatenate((theta, theta[::-1]))
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    x, y = np.append(x, x[0]), np.append(y, y[0])
    return bound[1] * x - bound[2] * y, bound[2] * x + bound[1] * y


# "Snail" layout (the snail of Pythagoras): the union of `n` right triangles O, A_k, A_{k+1} sharing the vertex O at
# the centre.   bound[0] = area; bound[1] = n; bound[2] = the scale; then the corners A_1 .. A_{n+1} as x, y pairs
# from bound[3], already turned by the rotation. In the unscaled figure |O A_k| = sqrt(k), the right angle is at
# A_k and |A_k A_{k+1}| = 1, so A_{k+1} is the direction of A_k turned by atan(1 / sqrt(k)); the scale makes the
# farthest corner, A_{n+1}, one unit from the centre.
def snail_bound(segments, rotation):
    """The snail of Pythagoras made of `segments` triangles, turned counterclockwise by `rotation` degrees (the
    first corner A_1 is along +x at rotation 0), scaled so that its farthest corner is 1 from the centre."""
    n = min(max(int(round(segments)), SNAIL_MIN_SEGMENTS), SNAIL_MAX_SEGMENTS)
    theta = np.concatenate(([0.0], np.cumsum(np.arctan(1.0 / np.sqrt(np.arange(1, n + 1))))))
    scale = 1.0 / sqrt(n + 1.0)
    angle = theta + radians(rotation)
    radius = scale * np.sqrt(np.arange(1, n + 2))
    bound = np.zeros(3 + 2 * (n + 1), dtype=np.float64)
    bound[1] = n
    bound[2] = scale
    bound[3:] = np.column_stack((radius * np.cos(angle), radius * np.sin(angle))).ravel()
    bound[0] = estimate_area_numeric(SHAPE_SNAIL, bound, 1.05)
    return bound


def snail_outline_xy(bound):
    """(x, y) arrays tracing the snail's edge. Seen from the centre the shape is the farthest of its triangles'
    outer edges in each direction, so the edge is followed by direction: at each corner's direction, just before and
    just after (where a later turn starts or ends, the edge jumps along a straight line out from the centre), joined by
    straight pieces. A direction no triangle covers is the centre itself."""
    n = int(bound[1])
    corners = bound[3:3 + 2 * (n + 1)].reshape(-1, 2)
    a, b = corners[:-1], corners[1:]
    edge = b - a
    start = np.arctan2(a[:, 1], a[:, 0])
    span = np.mod(np.arctan2(b[:, 1], b[:, 0]) - start, 2.0 * pi)
    turns = np.mod(np.append(start, np.arctan2(b[-1, 1], b[-1, 0])), 2.0 * pi)
    phi = np.sort(np.mod(np.concatenate((turns - 1e-9, turns + 1e-9)), 2.0 * pi))
    far = np.zeros_like(phi)
    for k in range(n):
        covered = np.mod(phi - start[k], 2.0 * pi) <= span[k]
        ux, uy = np.cos(phi), np.sin(phi)
        across = ux * edge[k, 1] - uy * edge[k, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (a[k, 0] * edge[k, 1] - a[k, 1] * edge[k, 0]) / across
        far = np.where(covered & np.isfinite(t), np.maximum(far, t), far)
    x, y = far * np.cos(phi), far * np.sin(phi)
    return np.append(x, x[0]), np.append(y, y[0])


# "Egg" layout (the Hugelschaffer egg): bound[0] = area; bound[1], bound[2] = cos, sin of the rotation;
# bound[3] = a, bound[4] = b, bound[5] = d. With u measured from the middle of its length (so u runs from -a to a) it
# is y^2 (a^2 + d^2 + 2 d u) = b^2 (a^2 - u^2), with |u| < a inside; the page's form, in x = u + d, is
# (x - d)^2 / a^2 + y^2 / b^2 (1 + (2 d x - d^2) / a^2) = 1, and b is exactly its half-width.
def egg_bound(half_length, half_width, shift, rotation):
    """A Hugelschaffer egg of half-length a, half-width b and shift d = `shift` * a (|shift| below 1; 0 is the
    ellipse with semi-axes a and b), turned counterclockwise by `rotation` degrees about the middle of its
    length. Its pointed end is toward +x for a positive shift. The area is the integral of u dy round the
    parametrization of egg_outline_xy."""
    a = max(half_length, AXIS_MIN)
    b = max(half_width, AXIS_MIN)
    d = min(max(shift, EGG_SHIFT_MIN), EGG_SHIFT_MAX) * a
    th = radians(rotation)
    steps = 4096
    t = (np.arange(steps) + 0.5) * (2.0 * pi / steps)
    u = -d * np.sin(t) ** 2 + np.cos(t) * np.sqrt(a * a - (d * np.sin(t)) ** 2)
    area = float(np.sum(u * b * np.cos(t)) * (2.0 * pi / steps))
    return np.array([area, cos(th), sin(th), a, b, d], dtype=np.float64)


def egg_outline_xy(bound, n=361):
    """(x, y) arrays tracing the egg exactly, by the page's parametrization (in u = x - d, so that the egg is
    centred on the middle of its length): u = -d sin(t)^2 + cos(t) sqrt(a^2 - d^2 sin(t)^2), y = b sin(t)."""
    a, b, d = bound[3], bound[4], bound[5]
    t = np.linspace(0.0, 2.0 * pi, n)
    u = -d * np.sin(t) ** 2 + np.cos(t) * np.sqrt(a * a - (d * np.sin(t)) ** 2)
    y = b * np.sin(t)
    return bound[1] * u - bound[2] * y, bound[2] * u + bound[1] * y


# "Flower" layout (the sine flower): bound[0] = area; bound[1], bound[2] = cos, sin of the rotation; bound[3] = R,
# bound[4] = A, bound[5] = k. In the shape's own frame it is rho < R (1 + A sin(k phi)).
def flower_bound(radius, petals, amplitude, rotation):
    """A sine flower of radius R, `petals` petals (k, a whole number) and amplitude A (clamped below 1), turned
    counterclockwise by `rotation` degrees: the points within R (1 + A sin(k phi)) of the centre at angle phi. The
    area is exactly pi R^2 (1 + A^2 / 2)."""
    r = max(radius, 1e-6)
    k = min(max(int(round(petals)), FLOWER_MIN_PETALS), FLOWER_MAX_PETALS)
    a = min(max(amplitude, FLOWER_AMPLITUDE_MIN), FLOWER_AMPLITUDE_MAX)
    th = radians(rotation)
    return np.array([pi * r * r * (1.0 + 0.5 * a * a), cos(th), sin(th), r, a, k], dtype=np.float64)


def flower_outline_xy(bound):
    """(x, y) arrays tracing the sine flower's edge exactly, from its formula, turned by the rotation."""
    k = int(bound[5])
    phi = np.linspace(0.0, 2.0 * pi, max(361, 48 * k + 1))
    rho = bound[3] * (1.0 + bound[4] * np.sin(k * phi))
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    return bound[1] * x - bound[2] * y, bound[2] * x + bound[1] * y


def chain_arcs_xy(bound, per_arc=60):
    """(x, y) arrays tracing the boundary of an arc-loops shape: each arc sampled, then the
    arcs joined end to end (an arc is turned round when its far end is the nearer one)."""
    pieces = []
    for k in range(int(bound[1])):
        cx, cy, r, lo, span = bound[3 + ARC_FIELDS * k:3 + ARC_FIELDS * (k + 1)]
        ang = lo + span * np.linspace(0.0, 1.0, 8 + int(per_arc * span / (2.0 * pi)))
        pieces.append(np.column_stack((cx + r * np.cos(ang), cy + r * np.sin(ang))))
    if not pieces:
        return np.zeros(2), np.zeros(2)
    path = pieces.pop(0)
    while pieces:
        end = path[-1]
        gap = [min(np.hypot(*(p[0] - end)), np.hypot(*(p[-1] - end))) for p in pieces]
        nxt = pieces.pop(int(np.argmin(gap)))
        if np.hypot(*(nxt[-1] - end)) < np.hypot(*(nxt[0] - end)):
            nxt = nxt[::-1]
        path = np.vstack((path, nxt))
    path = np.vstack((path, path[:1]))
    return path[:, 0], path[:, 1]


def is_robustly_interior(shape, bound, x, y, margin):
    """Is (x, y) inside the shape with some room to spare - not just inside,
    but with a small disk around it also entirely inside? A plain
    inside_boundary() check can pass for a corner (like a sector's apex) or a
    point right on an edge, which is useless as a scaling centre or a
    ray-casting origin; this rejects those."""
    if not inside_boundary(x, y, shape, bound):
        return False
    for ang in (0.0, 0.25 * pi, 0.5 * pi, 0.75 * pi, pi, 1.25 * pi, 1.5 * pi, 1.75 * pi):
        if not inside_boundary(x + margin * cos(ang), y + margin * sin(ang), shape, bound):
            return False
    return True


def find_interior_point(shape, bound, extent=2.2):
    """Some point robustly inside the shape (not a corner or a point right on
    the boundary) - a spiral search using the same inside_boundary test the
    simulation itself uses, so it's correct for any of the 8 shapes without
    needing a bespoke formula per family. Used as a ray-casting origin
    (sample_boundary_xy), which needs genuine breathing room to work correctly."""
    margin = max(0.02, extent * 0.02)
    if is_robustly_interior(shape, bound, 0.0, 0.0, margin):
        return 0.0, 0.0
    for r in np.linspace(0.02, extent, 60):
        for ang in np.linspace(0.0, 2.0 * pi, 24, endpoint=False):
            x, y = r * cos(ang), r * sin(ang)
            if is_robustly_interior(shape, bound, x, y, margin):
                return x, y
    # Fall back to a plain (non-robust) inside test, so this always returns
    # something rather than nothing, for an unusually thin/small shape.
    if inside_boundary(0.0, 0.0, shape, bound):
        return 0.0, 0.0
    for r in np.linspace(0.02, extent, 60):
        for ang in np.linspace(0.0, 2.0 * pi, 24, endpoint=False):
            x, y = r * cos(ang), r * sin(ang)
            if inside_boundary(x, y, shape, bound):
                return x, y
    return 0.0, 0.0                            # shouldn't happen for a valid shape


def sample_boundary_xy(shape, bound, n=181):
    """n (x, y) points tracing the shape's boundary, found by casting rays
    from an interior point at evenly-spaced angles and using boundary_hit
    (already correct for all 8 shapes) to find where each one exits."""
    ix, iy = find_interior_point(shape, bound)
    xs = np.empty(n)
    ys = np.empty(n)
    for k in range(n):
        ang = 2.0 * pi * k / (n - 1)
        ux, uy = cos(ang), sin(ang)
        t, _nx, _ny = boundary_hit(ix, iy, ux, uy, shape, bound)
        xs[k] = ix + t * ux
        ys[k] = iy + t * uy
    return xs, ys
