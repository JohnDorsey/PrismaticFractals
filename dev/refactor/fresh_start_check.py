"""Does a fresh start draw in the default world what it drew in the world half the size?

    python dev/refactor/fresh_start_check.py [size]          (default 256)

Renders the scene the program starts with (the circle of radius 2 in the default world [-2, 2]) with the program's defaults
(escape radius 4, one pixel of confinement 4/512, the point distances divided by DISTANCE_METRIC_DIVISOR), and the scene
it started with before FILE_VERSION 154 (the unit circle in [-1, 1], escape radius 2, 2/512, the old divisor 2: patched into
metrics.py for that render, which reads it when it colours), and counts the pixels that differ, for three colorations: the
default one (three point distances), one of bounce count, termination and bounce angles, and the default with a Buddha
channel. The few that differ are the rays that run back along their own path, where an absolute tolerance of the kernel
(the 1e-14 parallel test, EPS, SELF_TIE_EPS) decides a tie differently at another scale (see scale_check.py)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np                                                 # noqa: E402
import tests.support as sup                                        # noqa: E402

m = sup.m


def main(n):
    app = sup._fake_render_app()
    new_data = m.resolve_frame_settings([], 0, m.DEFAULT_SETTING_VALUES).replace(outline=False)
    old_data = new_data.replace(
        natural_termination=new_data.natural_termination.replace(escape_radius=2.0, confine_threshold=2.0 / 512),
        buddha_seed=new_data.buddha_seed.replace(x_start=-1.0, x_stop=1.0, y_start=-1.0, y_stop=1.0))
    new_scene = m.build_obstacle_arrays([m.default_starting_obstacle(0)])
    old_scene = m.build_obstacle_arrays([sup._unit_circle(0)])
    old_framing = m.FramingSnapshot(id=0, x=0.0, y=0.0, diagonal=2.0 * 2.0 ** 0.5, aspect=1.0)
    counts = m.ColorationConfig(1)
    for c in range(3):
        counts.metric_types[c] = m.METRIC_TYPE_OTHER
    counts.other_metric[:] = [m.ST_BOUNCE_COUNT, m.ST_TERMINATION, m.ST_ANGLES]
    buddha = m.ColorationConfig(2)
    buddha.chan_src[1] = m.CHAN_SRC_BUDDHA
    buddha.buddha_points[1] = sup._buddha_choice(m.POINT_ALL_MEAN, m.POINT_END)
    for name, config in (("default (three point distances)", m.ColorationConfig(0)),
                         ("bounce count, termination, angles", counts), ("default + a Buddha channel", buddha)):
        coloration = sup._coloration_settings_from_config(config)
        new = app.render_full_image(new_data, n, n, new_scene, (coloration,), m.DEFAULT_FRAMING)
        saved = m.metrics.DISTANCE_METRIC_DIVISOR
        m.metrics.DISTANCE_METRIC_DIVISOR = 2.0
        try:
            old = app.render_full_image(old_data, n, n, old_scene, (coloration,), old_framing)
        finally:
            m.metrics.DISTANCE_METRIC_DIVISOR = saved
        differ = (new != old).any(axis=2)
        fired = (new != np.array(m.BACKGROUND, dtype=np.uint8)).any(axis=2)
        print(f"{n}x{n} {name}: {int(differ.sum())} of {int(fired.sum())} drawn pixels differ ({100.0 * differ.sum() / fired.sum():.3f}%)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
