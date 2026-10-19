"""Warm run time and first-call compile time of the drivers (run: python dev/refactor/driver_bench.py [ROOT] [--first]).

ROOT is a folder holding `PrismaticFractals/` and `test_harness.py` (here or in its dev/; default: this project), so an old copy of the package can be timed
against the present one, alternately, in separate processes. Without --first it times, after a warm-up call, the best of several runs of
four runs on fixed scenes (a 512 x 512 trace_rows of the starting scene; the same of a scene with two shapes and a segment
multiplier; scatter_multi_buddha, serial, of 256 x 256 seeds with all segment midpoints; a 256 x 256 trace_rows of 2000 iterations) and prints one line of seconds per scene.
With --first it prints the import time and the time of the FIRST calls (compiling, or loading from the numba cache). Written for
the engine refactor's step 7 (`Engine_Refactor_Plan.md`): the engine switch inside the drivers must cost nothing measurable.
"""
import os
import sys
import time

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ROOT = os.path.abspath(args[0]) if args else os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dev"))        # (an old copy of the program may keep test_harness.py in ROOT itself)
import numpy as np                                  # noqa: E402
import test_harness                                  # noqa: E402

started = time.perf_counter()
m = test_harness.load_module()
imported = time.perf_counter()


def engine_args(arrays, seg_mult=1.0):
    """The drivers' engine arguments, whichever form the package under test has them in."""
    ray = m.ray_args(seg_mult, m.fuzz_rule(m.DEFAULT_FUZZ), m.EMISSION_DEFAULT)
    return m.path_args(m.ENGINE_RAY_BOUNCING, ray, arrays)


def scene_two_shapes():
    first = m.default_starting_obstacle(0)
    second = m.EllipseShape(1)
    second.width, second.height, second.cx = 1.2, 0.6, 0.4
    return m.build_obstacle_arrays([first, second])


def render(n, arrays, seg_mult, max_iter):
    stats = np.zeros((n, n, m.N_STATS), np.float32)
    points = np.zeros((n, n, m.N_POINTS, 2), np.float32)
    state = np.zeros((n, n), np.uint8)
    termination = m.natural_termination_rule(max_iter, m.effective_dist(0.0), m.DEFAULT_ESCAPE_RADIUS, m.CONFINE_DEFAULT)
    m.trace_rows(stats, points, state, 0, n, n, n, m.WORLD_RECT_DEFAULT, termination, engine_args(arrays, seg_mult), 0)


def scatter(n, arrays, max_iter):
    counts = np.zeros((n, n))
    termination = m.natural_termination_rule(max_iter, m.effective_dist(0.0), m.DEFAULT_ESCAPE_RADIUS, m.CONFINE_DEFAULT)
    m.scatter_multi_buddha(0, n, m.WORLD_RECT_DEFAULT, termination, engine_args(arrays), 0, False, True, False, False, False,
                           np.ones(m.N_TERM, dtype=np.bool_), counts)


start_arrays = m.build_obstacle_arrays([m.default_starting_obstacle(0)])
two_arrays = scene_two_shapes()
scenes = {
    "trace_rows 512 start": lambda: render(512, start_arrays, 1.0, 256),
    "trace_rows 512 two shapes": lambda: render(512, two_arrays, 0.9, 256),
    "scatter 256 all midpoints": lambda: scatter(256, two_arrays, 120),
    "trace_rows 256 long rays": lambda: render(256, two_arrays, 0.9, 2000),
}

if "--first" in sys.argv:
    first = time.perf_counter()
    render(16, start_arrays, 1.0, 16)
    scatter(16, start_arrays, 16)
    print(f"import {imported - started:.2f}s | first kernel calls {time.perf_counter() - first:.2f}s")
else:
    results = []
    for name, run in scenes.items():
        run()                                                   # compile / load
        times = []
        for _ in range(7):
            t0 = time.perf_counter()
            run()
            times.append(time.perf_counter() - t0)
        results.append(f"{name}: {min(times):.4f}")
    print(" | ".join(results))
