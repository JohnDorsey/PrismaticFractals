"""One large export in a fresh process (so the peak is its own):

    python dev/refactor/big_export.py <default|point_pairs|buddha_simple|buddha_all|mandelbrot_default|mandelbrot_buddha> <full|demand> [w h]

Prints the peak memory of the process (Windows peak working set; about 207 MB of it is the interpreter and numba), the
time and a hash of the image. "demand" is what render_full_image does (it keeps only the stored slots the colorations
read, see lightbounces.stored_layout); "full" stores every slot of every pixel, as it did before FILE_VERSION 151, to
compare with. Defaults to 3840 x 2160, 16 iterations, outlines off. Needs psutil."""
import hashlib
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dev"))                 # (the tests package and the harness are in dev/)
os.chdir(ROOT)
import numpy as np                                                 # noqa: E402
import psutil                                                      # noqa: E402
import tests.support as sup                                        # noqa: E402

m = sup.m
kind, mode = sys.argv[1], sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 3840
h = int(sys.argv[4]) if len(sys.argv) > 4 else 2160

mandelbrot = kind.startswith("mandelbrot")
v = m.ColorationConfig(0)
if mandelbrot:                                     # "mandelbrot_default" (three distances) or "mandelbrot_buddha" (all path points)
    v.metric_types = [0, 0, 0]
    if kind == "mandelbrot_buddha":
        v.chan_src[1] = m.CHAN_SRC_BUDDHA
        v.buddha_points[1] = tuple(label == "All path points" for label in m.BUDDHA_POINT_LABELS)
elif kind == "point_pairs":
    v.point1_choice = [m.POINT_TOP, m.POINT_BOTTOM, m.POINT_LEFT]
    v.point2_choice = [m.POINT_RIGHT, m.CHOICE_ORIGIN, m.POINT_SELF_MEAN]
    v.metric_types = [0, 0, 0]
elif kind == "buddha_all":
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    v.buddha_points[1] = tuple(i < m.N_POINTS for i in range(m.N_BUDDHA_POINTS))
elif kind == "buddha_simple":
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    v.buddha_points[1] = sup._buddha_choice(m.POINT_ALL_MEAN, m.POINT_END)
settings = (sup._coloration_settings_from_config(v),)
values = dict(m.DEFAULT_SETTING_VALUES)
values["iter"] = 64 if mandelbrot else 16
if mandelbrot:
    values["engine"] = m.ENGINE_MANDELBROT
data = m.resolve_frame_settings([m.Keyframe(1, 0, values)], 0, values).replace(outline=False)
start = m.default_starting_obstacle(0)
if mandelbrot:
    start.in_effect = start.out_effect = m.OBSTACLE_EFFECT_PASS
obst = m.build_obstacle_arrays([start])
app = sup._fake_render_app()
proc = psutil.Process()
t0 = time.perf_counter()
if mode == "demand":
    img = app.render_full_image(data, w, h, obst, settings, m.DEFAULT_FRAMING)
else:                                    # every slot of every pixel
    termination, engine_args = m.settings_natural_termination(data.natural_termination), m.path_args(data.engine.engine, m.settings_ray_args(data.ray), obst)
    stats = np.zeros((h, w, m.N_STATS), dtype=np.float32)
    points = np.zeros((h, w, m.N_POINTS, 2), dtype=np.float32)
    state = np.zeros((h, w), dtype=np.uint8)
    m.trace_rows(stats, points, state, 0, h, w, h, m.WORLD_RECT_DEFAULT, termination, engine_args, data.ray.fuzz.rng_seed)
    img = app.apply_coloration(settings[0], stats, points, state, w, h, data.buddha_blur, termination, engine_args,
                               data.ray.fuzz.rng_seed, data.buddha_seed)
dt = time.perf_counter() - t0
peak = proc.memory_info().peak_wset
print("%-14s %-7s %dx%d  peak %7.0f MB  %5.1f s  image %s" % (kind, mode, w, h, peak / 1e6, dt, hashlib.sha256(img.tobytes()).hexdigest()[:12]))
