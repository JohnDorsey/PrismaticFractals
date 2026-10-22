"""Hashes of what the program draws, to prove that a change which should not alter any image does not.

    python dev/refactor/image_hashes.py <out.json>                  record the hashes of this tree
    python dev/refactor/image_hashes.py <out.json> <baseline.json>  record them and compare with a recording
                                                                    (prints every difference; exit code 1 if any)

Three groups, all on small fixed scenes (a circle, and a circle with an ellipse; segment multipliers 1.0 / 0.7 / 0.8 /
0.9; fuzzing off): "kernel" (the arrays trace_rows fills and the Buddha counts of scatter_multi_buddha), "export" (the
image apply_coloration makes for five colorations - what an export draws) and "live" (the live colouring paths on a fake
app: incremental Buddha counts, recolor, hover values). The scenes are written out here - the world [-1, 1], a circle of
width 2, escape radius 2, confinement threshold 2/512 - and not read from the program's defaults, so a change of the
defaults (the world is going to be twice as large) leaves every hash alone, and so does any change that renames a
setting or reorganises code. What the metrics' divisors change are the four point-distance metrics and the path length:
the "kernel" hashes leave those stat columns out, the colorations hashed read no distance or length metric, and a
fourth group, "distance_dependent" (the program's own default coloration, which reads three distances), is recorded but
not compared. When those divisors change the baseline is re-recorded on purpose (say so in the report)."""
import dataclasses
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dev"))                 # (the tests package and the harness are in dev/)
os.chdir(ROOT)
import numpy as np                                                 # noqa: E402
import tests.support as sup                                        # noqa: E402

m = sup.m


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


OLD_WORLD = m.WorldRect(-1.0, 1.0, -1.0, 1.0)           # the scenes below are drawn in the world the program started with
OLD_ESCAPE_RADIUS = 2.0
OLD_CONFINE = m.confine_rule(4, m.CONFINE_BOX, 2.0 / 512)
DISTANCE_STATS = [m.ST_HDIST, m.ST_VDIST, m.ST_MANHATTAN, m.ST_EUCLIDEAN]
LENGTH_STATS = sorted({int(m.STAT_SLOT[seq, m.SRED_LENGTH]) for seq in range(m.N_SEQS)})
SCALE_FREE_STATS = [s for s in range(m.N_STATS) if s not in DISTANCE_STATS and s not in LENGTH_STATS]


def ellipse():
    e = m.EllipseShape(1)
    e.width, e.height, e.rotation, e.cx, e.cy = 0.7, 0.45, 25.0, 0.15, -0.1
    return e


def circle():
    c = m.EllipseShape(0)
    c.width = c.height = 2.0
    c.in_effect = c.out_effect = m.OBSTACLE_EFFECT_REFLECT
    c.fire_inside = True
    return c


def termination(max_iter):
    return m.natural_termination_rule(max_iter, m.effective_dist(0.0), OLD_ESCAPE_RADIUS, OLD_CONFINE)


def scenes():
    return {"circle": [circle()], "circle+ellipse": [circle(), ellipse()]}


def kernel_hashes():
    res = {}
    for name, scene in scenes().items():
        arrays = m.build_obstacle_arrays(scene)
        for seg in (1.0, 0.7):
            for it in (40, 300):
                stop, ray = termination(it), sup._engine_args(arrays, seg_mult=seg)
                n = 96
                stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
                points = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
                state = np.zeros((n, n), dtype=np.uint8)
                m.trace_rows(stats, points, state, 0, n, n, n, OLD_WORLD, stop, ray, 0)
                res[f"{name} seg{seg} it{it}"] = [h(stats[..., SCALE_FREE_STATS]), h(points),
                                                 h((state == m.PIXEL_TRACED).astype(np.uint8))]
                for opts in ((True, False, False, False), (False, True, False, False), (False, False, True, True)):
                    counts = np.zeros((n, n))
                    m.scatter_multi_buddha(0, n, OLD_WORLD, stop, ray, 0, *opts, False, np.ones(m.N_TERM, dtype=np.bool_), counts)
                    res[f"{name} seg{seg} it{it} buddha{opts}"] = [h(counts)]
    return res


def make_coloration(cname):
    v = m.ColorationConfig(0)
    if cname != "default":                  # (the default coloration reads three point distances)
        for c in range(3):
            v.metric_types[c] = m.METRIC_TYPE_OTHER
            v.other_metric[c] = (m.ST_BOUNCE_COUNT, m.ST_TERMINATION, m.ST_ANGLES)[c]
    if cname.startswith("buddha"):
        sel = (28, 29) if cname == "buddha_simple" else (31, 32, 33, 34)
        v.chan_src[1] = m.CHAN_SRC_BUDDHA
        v.buddha_points[1] = tuple(i in sel for i in range(m.N_BUDDHA_POINTS))
    if cname == "hsv_const":
        v.color_model = "HSV"
        v.chan_src[0] = m.CHAN_SRC_CONSTANT
    return v


def export_hashes():
    res, distance_dependent = {}, {}
    app = type("A", (), {})()
    apply_coloration = m.App.apply_coloration.__get__(app)
    n = 80
    for sname, scene in scenes().items():
        arrays = m.build_obstacle_arrays(scene)
        for seg in (1.0, 0.8):
            stop, ray = termination(60), sup._engine_args(arrays, seg_mult=seg)
            stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
            pts = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
            state = np.zeros((n, n), dtype=np.uint8)
            m.trace_rows(stats, pts, state, 0, n, n, n, OLD_WORLD, stop, ray, 0)
            for cname in ("default", "other_types", "buddha_simple", "buddha_multi", "hsv_const"):
                v = sup._coloration_settings_from_config(make_coloration(cname))
                blur = m.BuddhaBlurSettings(radius=0.02, type=m.BUDDHA_BLUR_GAUSSIAN)
                image = h(apply_coloration(v, stats, pts, state, n, n, blur, stop, ray, 0, None, OLD_WORLD))
                (distance_dependent if cname == "default" else res)[f"{sname} seg{seg} {cname}"] = image
    return res, distance_dependent


def live_hashes():
    n = 40
    world = OLD_WORLD
    rule = termination(60)
    engine_args = sup._engine_args(m.build_obstacle_arrays([circle(), ellipse()]), seg_mult=0.9)
    stats = np.zeros((n, n, m.N_STATS), dtype=np.float32)
    pts = np.zeros((n, n, m.N_POINTS, 2), dtype=np.float32)
    state = np.zeros((n, n), dtype=np.uint8)
    m.trace_rows(stats, pts, state, 0, n, n, n, world, rule, engine_args, 5)

    def make_app(v, seeds):
        app = type("A", (), {})()
        app.live = m.LiveImage(m.DEFAULT_FRAMING, m.FULL_LAYOUT)        # (its grid is then replaced by this scene's)
        app.live.img_h = app.live.img_w = n
        app.live.world_rect = world
        app.image_window = world
        app.live.stats, app.live.characteristic_points, app.live.pixel_state = stats, pts, state
        app.points = pts
        app.live.img = np.zeros((n, n, 3), np.uint8)
        app.live.rows_done = n
        app.live_coloration = lambda: v
        app.mapping = m.make_mapping(v.metric_types, v.point_metric, v.other_metric, v.chan_src, v.constant_choice)
        app.current_blur_settings = lambda: m.BuddhaBlurSettings(radius=0.03, type=m.BUDDHA_BLUR_GAUSSIAN)
        app.current_buddha_seed_settings = lambda: seeds
        app.settings_of_current_image = lambda: m.DEFAULT_RAY_SETTINGS
        app.generated = dataclasses.replace(
            m.generated_figure(m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_TERMINATION_SETTINGS, []),
            termination=rule, engine_args=engine_args)
        app.current_figure = m.App.current_figure.__get__(app)
        for name in ("scatter_buddha_rows", "blurred_buddha_counts", "rebuild_buddha_counts", "recolor", "component_values",
                     "map_color", "world_to_canvas"):
            setattr(app, name, getattr(m.App, name).__get__(app))
        return app

    res = {}
    camera = m.BuddhaSeedSettings(camera_pixels=True, x_start=-1.0, x_stop=1.0, y_start=-1.0, y_stop=1.0)
    zone = m.BuddhaSeedSettings(camera_pixels=False, x_start=-0.5, x_stop=0.8, y_start=-0.6, y_stop=0.5)
    for cname in ("default", "other_types", "buddha_simple", "buddha_multi", "buddha_two_channels", "hsv_const"):
        for sname, seeds in (("camera", camera), ("zone", zone)):
            if sname == "zone" and not cname.startswith("buddha"):
                continue
            v = make_coloration(cname)
            if cname.startswith("buddha"):
                sel = (28, 29) if cname in ("buddha_simple", "buddha_two_channels") else (30, 31, 32, 33, 34)
                v.buddha_points[1] = tuple(i in sel for i in range(m.N_BUDDHA_POINTS))
                v.buddha_brightness[1] = 2
                if cname == "buddha_two_channels":
                    v.chan_src[2] = m.CHAN_SRC_BUDDHA
                    v.buddha_points[2] = tuple(i in (31, 33) for i in range(m.N_BUDDHA_POINTS))
                    v.buddha_compression[2] = 1
            tag = f"{cname}/{sname}"
            app = make_app(v, seeds)
            app.rebuild_buddha_counts()
            app.recolor(0, n)
            res[tag + " full"] = h(app.live.img)
            res[tag + " counts"] = [h(c) if c is not None else None for c in app.live.buddha_counts]
            res[tag + " blurred"] = [h(app.blurred_buddha_counts(c)) if app.blurred_buddha_counts(c) is not None else None
                                     for c in range(3)]
            app = make_app(v, seeds)                       # incrementally, as the tick loop does
            app.live.rows_done = 0
            app.rebuild_buddha_counts()
            for r0, r1 in ((0, 7), (7, 19), (19, 33), (33, n)):
                app.live.rows_done = r1
                app.scatter_buddha_rows(r0, r1)
                app.recolor(0 if any(s == m.CHAN_SRC_BUDDHA for s in v.chan_src) else r0, r1)
            res[tag + " incremental"] = h(app.live.img)
            hover = []
            for (i, j) in ((20, 20), (12, 25), (30, 8), (1, 1), (21, 19)):
                if state[j, i] != m.PIXEL_TRACED:
                    hover.append(None)
                    continue
                x = world.x_start + (i + 0.5) * (world.x_stop - world.x_start) / n
                y = world.y_stop - (j + 0.5) * (world.y_stop - world.y_start) / n
                path_points = np.empty((62, 2))
                tags = np.empty(62)
                raw = np.empty(m.N_RAW)
                term, npts = m.trace_path(x, y, path_points, tags, rule, engine_args.ray, engine_args.obstacles, raw)
                vec = np.zeros(m.N_STATS)
                sp = np.zeros((m.N_POINTS, 2))
                m.compute_run_measurements(path_points, npts, raw, 60, 0.9, vec)
                m.compute_sequences(path_points, npts, tags, int(raw[m.RAW_BOUNCES]), 60, m.RAY_SEQUENCE_TABLES, vec, sp)
                vals = app.component_values(vec, sp, x, y)
                hover.append([repr(float(t)) for t in vals] + [repr(app.map_color(vals))])
            res[tag + " hover"] = hover
    return res


def compare(now, before):
    bad = 0
    for group in sorted((set(now) | set(before)) - {"distance_dependent"}):
        a, b = now.get(group, {}), before.get(group, {})
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                bad += 1
                print(f"DIFFERENT  {group}: {key}")
    return bad


if __name__ == "__main__":
    export, distance_dependent = export_hashes()
    live = live_hashes()
    distance_dependent.update({"live " + key: live.pop(key) for key in [k for k in live if k.startswith("default/")]})
    out = {"kernel": kernel_hashes(), "export": export, "live": live, "distance_dependent": distance_dependent}
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    print({g: len(v) for g, v in out.items()}, "hashes recorded in", sys.argv[1])
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            bad = compare(out, json.load(f))
        print("IDENTICAL to", sys.argv[2] if not bad else f"{bad} DIFFERENCES from {sys.argv[2]}")
        sys.exit(1 if bad else 0)
