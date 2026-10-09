"""Is the ray-bouncing engine a pure change of scale? Renders a scene in the world [-1, 1], then the same scene with every
length doubled in the world [-2, 2] (shapes, positions, the escape radius, the confinement threshold, the emission
target), and compares the two pixel for pixel. This is what makes the doubled default world (Engine_Refactor_Design.md,
D10) safe: a difference here is a place where the engine is not scale-free.

    python dev/refactor/scale_check.py [scale]        (the scale defaults to 2)

Every shape kind is tried at its default parameters and at mid-range ones, with the default emission and with emission
Towards a point, with segment multipliers 1.0 and 0.7, and with fuzzing on. The base scene's parameters are written out
here, not read from the program's defaults, so the check stays valid when the defaults change.

Scales that are not 1, 2 or 0.5 are informative only: parameters beyond a slider's range are clamped by the bound
builders (the spiral at 4), and anything that is not a power of two is not exact in floating point.

Scales other than 1, 2 and 0.5 are informative only: a bound builder clamps a parameter beyond its slider range (the
spiral at 4), and a scale that is not a power of two is not exact in floating point.

What must be equal: the pixel states, and for every pixel whose path agrees (the characteristic points exactly `scale`
times - doubling is exact in floating point - or to rounding) every metric that does not depend on size. A path that
diverges is counted and classed: the kernel's absolute tolerances (the 1e-14 test for a parallel segment, EPS, SELF_TIE_EPS)
decide ties differently at another scale, which only matters where a ray runs back along its own path (the unit circle at
the default 45 degrees does this) or nearly so, and where a root search stops at an absolute tolerance (the squircles, the
egg, the spiral, the flower); a scene fails if more than MAX_DIVERGED of its traced pixels diverge without such a retrace. The size-dependent metrics are the four point distances and the
squashed path lengths: they are compared as the distance or length they stand for (undone with the divisors the program
uses - DISTANCE_METRIC_DIVISOR and LENGTH_METRIC_DIVISOR if it names them, else the old 2 and 5; the program's are now 4 and 10), skipping a pixel where
either value is clipped. Exit code 1 if a scene has a problem."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np                                                 # noqa: E402
import tests.support as sup                                        # noqa: E402

m = sup.m
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
N = 48
RNG_BASE = 7
MAX_DIVERGED = 0.02             # the most traced pixels of a scene that may diverge without retracing (see compare)
OLD_ESCAPE_RADIUS = 2.0                 # the base scene's own values (not the program's current defaults)
OLD_CONFINE_THRESHOLD = 2.0 / 512
DISTANCE_FIELDS = {"width", "height", "radius", "lens_radius1", "lens_radius2", "lens_sep", "capsule_length",
                   "capsule_radius", "cut_radius", "cut_offset", "egg_a", "egg_b", "moss_size", "spiral_pitch",
                   "r", "cx", "cy"}
DIST_DIV = getattr(m, "DISTANCE_METRIC_DIVISOR", 2.0)
LENGTH_DIV = getattr(m, "LENGTH_METRIC_DIVISOR", 5.0)
POINT_DISTANCES = [m.ST_HDIST, m.ST_VDIST, m.ST_MANHATTAN, m.ST_EUCLIDEAN]
LENGTHS = sorted({int(m.STAT_SLOT[seq, m.SRED_LENGTH]) for seq in range(m.N_SEQS)})
SCALE_FREE = [s for s in range(m.N_STATS) if s not in POINT_DISTANCES and s not in LENGTHS]


def shape_of(cls, variant, scale):
    """An instance of `cls` with the base parameters of `variant`, every distance multiplied by `scale`."""
    o = cls(1)
    for p in cls.PARAMS:
        if variant == "mid" and p.lo is not None:
            v = p.lo + 0.4 * (p.hi - p.lo)
            setattr(o, p.field, int(round(v)) if p.fmt == "int" else float(v))
    if variant == "mid":
        o.rotation = 25.0
        o.r, o.theta, o.cx, o.cy = 0.08, 30.0, 0.05, -0.04
    for field in DISTANCE_FIELDS:
        if field in cls.FIELD_NAMES and isinstance(getattr(o, field), float):
            setattr(o, field, getattr(o, field) * scale)
    o.in_effect = o.out_effect = m.OBSTACLE_EFFECT_REFLECT
    o.fire_inside = True
    return o


def setup(obstacle, scale, seg_mult, fuzzed, towards):
    """(world rectangle, termination rule, ray arguments) of one scene at one scale."""
    ray = m.DEFAULT_RAY_SETTINGS.replace(seg_mult=seg_mult)
    if fuzzed:
        ray = ray.replace(fuzz=m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=3.0, bnd_jitter=2.0, self_jitter=2.0))
    if towards:
        ray = ray.replace(emit_mode=m.EMIT_TOWARDS, emit_target_cx=0.3 * scale, emit_target_cy=0.2 * scale)
    nat = m.DEFAULT_NATURAL_TERMINATION_SETTINGS.replace(max_iter=60, max_dist=0.0, escape_radius=OLD_ESCAPE_RADIUS * scale,
                                                         confine_threshold=OLD_CONFINE_THRESHOLD * scale)
    world = m.WorldRect(-scale, scale, -scale, scale)
    return world, m.settings_natural_termination(nat), m.path_args(m.ENGINE_RAY_BOUNCING, m.settings_ray_args(ray, m.build_obstacle_arrays([obstacle])))


def trace(obstacle, scale, seg_mult, fuzzed, towards):
    world, termination, engine_args = setup(obstacle, scale, seg_mult, fuzzed, towards)
    stats = np.zeros((N, N, m.N_STATS), dtype=np.float32)
    points = np.zeros((N, N, m.N_POINTS, 2), dtype=np.float32)
    state = np.zeros((N, N), dtype=np.uint8)
    m.trace_rows(stats, points, state, 0, N, N, N, world, termination, engine_args, RNG_BASE)
    return stats, points, state


def retraces(points):
    """Does the path run back along an earlier segment? (A segment parallel to an earlier one and on its line: the case
    where which of two coincident hits wins is decided by rounding and by the kernel's absolute tolerances.)"""
    seg = np.diff(points, axis=0)
    for j in range(1, len(seg)):
        length = np.hypot(*seg[j])
        if length < 1e-12:
            continue
        d = seg[j] / length
        for i in range(j - 1):
            e = seg[i]
            if abs(d[0] * e[1] - d[1] * e[0]) < 1e-9 * max(np.hypot(*e), 1e-12):
                r = points[i] - points[j]
                if abs(r[0] * d[1] - r[1] * d[0]) < 1e-9:
                    return True
    return False


def base_path(obstacle, seg_mult, fuzzed, towards, i, j):
    """The base scene's path from pixel (i, j), traced on its own (same seed formula as trace_rows)."""
    world, termination, engine_args = setup(obstacle, 1.0, seg_mult, fuzzed, towards)
    step = (world.x_stop - world.x_start) / N
    x, y = world.x_start + (i + 0.5) * step, world.y_stop - (j + 0.5) * step
    path_points, tags, raw = np.empty((62, 2)), np.empty(62), np.empty(m.N_RAW)
    if fuzzed:
        sup._seed_kernel_rng(RNG_BASE + i * 1000003 + j * 97)
    term, npts = m.trace_path(x, y, path_points, tags, termination, engine_args.ray, raw)
    return path_points[:npts]


DIRECTIONS = sorted({int(m.STAT_SLOT[seq, m.SRED_LAST_DIR]) for seq in range(m.N_SEQS)})


def metric_problems(sa, sb, ok, scale, rtol):
    """Metrics of the pixels in the mask `ok` that are not equal between base and scaled."""
    out = []
    atol = 1e-6 if rtol == 0.0 else 1e-4
    for s in SCALE_FREE:
        if rtol > 0.0 and (s in DIRECTIONS or s == m.ST_ANGLES):
            continue        # at rounding level a nearly zero-length last segment has no reliable direction or turn
        if s in DIRECTIONS:
            # (a ray can end with a segment of almost no length, whose direction is then whatever rounding made it)
            if (sa[ok][:, s] != sb[ok][:, s]).sum() > max(2, 0.002 * ok.sum()):
                out.append(f"metric {s}")
            continue
        if not np.allclose(sa[ok][:, s], sb[ok][:, s], rtol=rtol, atol=atol):
            out.append(f"metric {s}")
    for s in POINT_DISTANCES:
        keep = ok & (sa[..., s] < 1.0) & (sb[..., s] < 1.0)
        if not np.allclose(sa[..., s][keep] * DIST_DIV * scale, sb[..., s][keep] * DIST_DIV, rtol=1e-4, atol=1e-6):
            out.append(f"distance metric {s}")
    for s in LENGTHS:
        keep = ok & (sa[..., s] < 0.99) & (sb[..., s] < 0.99)
        want = LENGTH_DIV * np.tan(sa[..., s][keep].astype(np.float64) * np.pi / 2) * scale
        got = LENGTH_DIV * np.tan(sb[..., s][keep].astype(np.float64) * np.pi / 2)
        if not np.allclose(want, got, rtol=2e-3, atol=1e-5):
            out.append(f"length metric {s}")
    return out


def compare(obstacle_base, a, b, scale, seg_mult, fuzzed, towards):
    """(counts, problems): how many traced pixels agree exactly, to rounding, or diverge, and what is wrong."""
    (sa, pa, ta), (sb, pb, tb) = a, b
    problems = []
    if not np.array_equal(ta, tb):
        problems.append(f"pixel states differ at {int((ta != tb).sum())} pixels")
    traced = ta == m.PIXEL_TRACED
    err = np.abs(pb.astype(np.float64) - pa.astype(np.float64) * scale).max(axis=(2, 3))
    exact = traced & (err == 0.0)
    rounding = traced & (err > 0.0) & (err <= 1e-6 * scale)
    diverged = traced & (err > 1e-6 * scale)
    problems += [p + " (exact pixels)" for p in metric_problems(sa, sb, exact, scale, 0.0)]
    problems += [p + " (rounding-level pixels)" for p in metric_problems(sa, sb, rounding, scale, 1e-4)]
    other = sum(1 for j, i in np.argwhere(diverged) if not retraces(base_path(obstacle_base, seg_mult, fuzzed, towards, i, j)))
    if other > MAX_DIVERGED * max(1, traced.sum()):
        problems.append(f"{other} of {int(traced.sum())} traced pixels diverge without retracing their own path")
    return (int(exact.sum()), int(rounding.sum()), int(diverged.sum()) - other, other), problems


def main():
    bad = 0
    totals = np.zeros(4, dtype=int)
    scenes = 0
    for cls in m.SHAPE_CLASSES:
        for variant in ("default", "mid"):
            for seg_mult, fuzzed, towards in ((1.0, False, False), (0.7, False, False), (1.0, True, False), (1.0, False, True)):
                base = shape_of(cls, variant, 1.0)
                counts, problems = compare(base, trace(base, 1.0, seg_mult, fuzzed, towards),
                                           trace(shape_of(cls, variant, SCALE), SCALE, seg_mult, fuzzed, towards),
                                           SCALE, seg_mult, fuzzed, towards)
                totals += counts
                scenes += 1
                if problems:
                    bad += 1
                    print(f"PROBLEM  {cls.NAME} {variant} seg{seg_mult} fuzz={fuzzed} towards={towards}: {'; '.join(problems[:6])}")
    exact, rounding, retracing, other = (int(t) for t in totals)
    print(f"{scenes} scenes at scale {SCALE:g}; traced pixels: {exact} exactly scaled, {rounding} equal to rounding, "
          f"{retracing} diverged where the path runs back along itself, {other} diverged elsewhere "
          f"({100.0 * (retracing + other) / max(1, sum(totals)):.3f}% in all); "
          f"{'no problems' if not bad else f'{bad} scenes with PROBLEMS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
