"""How long the first kernel calls take (run: python dev/refactor/kernel_cost.py).

Prints the import time and the time of the first calls of two kernel entry points. Run it twice
(the second run shows the warm-cache time), and once after `touch`ing a source file (shows what a
recompile costs). Baseline at FILE_VERSION 113, before any split: warm 4.0 s; after touching
light_bounces.py without changing any kernel 6.4 s (the extra 2.4 s is the recompile that a split
should stop UI/project edits from causing).
"""
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
import numpy as np                                  # noqa: E402
import test_harness                                  # noqa: E402

started = time.perf_counter()
m = test_harness.load_module()
imported = time.perf_counter()
obstacle = m.default_starting_obstacle(0)
arrays = m.build_obstacle_arrays([obstacle])
n = 16
stats = np.zeros((n, n, m.N_STATS), np.float32)
characteristic_points = np.zeros((n, n, m.N_POINTS, 2), np.float32)
pixel_state = np.zeros((n, n), np.uint8)
termination = m.natural_termination_rule(16, m.effective_dist(0.0), m.DEFAULT_ESCAPE_RADIUS, m.CONFINE_DEFAULT)
engine_args = m.ray_args(1.0, m.fuzz_rule(m.DEFAULT_FUZZ), m.EMISSION_DEFAULT, arrays)
first_call = time.perf_counter()
m.trace_rows(stats, characteristic_points, pixel_state, 0, n, n, n, termination, engine_args, 0)
counts = np.zeros((n, n))
m.scatter_multi_buddha(0, n, m.WORLD_RECT_DEFAULT, termination, engine_args, 0, False, True, False, False, False,
                       np.ones(m.N_TERM, dtype=np.bool_), counts)
done = time.perf_counter()
print(f"import {imported - started:.1f}s | first kernel calls {done - first_call:.1f}s")
