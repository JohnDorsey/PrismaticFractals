"""The real-window check run after every refactoring step (python dev/refactor/project_roundtrip_check.py).

It starts the real program (needs a display), builds a non-default project (extra obstacles, a second
coloration with a Buddha channel, a keyframe, changed settings), generates the image, saves the project,
chooses New, opens the saved file, generates again - and asserts that

  * New gives exactly the starting project and a blank image,
  * Open restores exactly the saved project, again with a blank image,
  * the regenerated image is pixel-identical to the first one,
  * the unsaved-changes mark and prompts work, and a damaged file is refused without a change.

Prints one line per check and exits with an error at the first failure. It writes only into a
temporary folder (the project file, the exports and the project dialogs are all redirected).
"""
import hashlib
import os
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import tkinter as tk                                 # noqa: E402
import numpy as np                                   # noqa: E402
import PrismaticFractals as lb                           # noqa: E402

folder = tempfile.mkdtemp(prefix="lb_check_")
lb.paths.EXPORT_DIR = os.path.join(folder, "exports")
lb.paths.TEMP_DIR = os.path.join(lb.paths.EXPORT_DIR, "TEMP")
lb.paths.PROJECT_DIR = os.path.join(folder, "projects")
path = os.path.join(folder, "check" + lb.PROJECT_EXTENSION)
prompts = []
lb.messagebox.askyesnocancel = lambda *a, **k: prompts.append(a[0]) or False     # "don't save"
lb.messagebox.showerror = lambda *a, **k: prompts.append(("ERROR",) + a)
lb.messagebox.showinfo = lambda *a, **k: prompts.append(("INFO",) + a)


def check(condition, text):
    print(("ok   " if condition else "FAIL ") + text)
    if not condition:
        raise SystemExit(f"check failed: {text}")


root = tk.Tk()
app = lb.App(root)
started = time.time()
while not app.ready and time.time() - started < 180:
    root.update()
check(app.ready, "the numba kernels compiled and the program is ready")
check(not app.is_dirty(), "a fresh program has no unsaved changes")
check(lb.project_signature(app.capture_project()) == app.saved_signature,
      "the starting project is the one captured at startup")


def generate():
    app.start()
    while app.running:
        root.update()
    root.update()
    return hashlib.sha256(app.live.img.tobytes()).hexdigest()


for shape_class in (lb.LuneShape, lb.TrefoilShape):
    app.new_obstacle(shape_class)
app.obstacles[1].cut_offset = 0.7
app.obstacles[1].r = 0.1
app.iter_var.set("40")
app.model.set("emit_offset", -20.0)
app.new_coloration()
app.live_coloration().set_channel(0, lb.ChannelSettings(
    chan_src=lb.CHAN_SRC_BUDDHA, metric_type=lb.METRIC_TYPE_OTHER, metric_slot=lb.ST_BOUNCE_COUNT, point1=0, point2=1,
    buddha_points=(False,) * 8 + (True,) + (False,) * (lb.N_BUDDHA_POINTS - 9),
    buddha_path_kinds=lb.ALL_PATH_KINDS, buddha_compression=0, buddha_brightness=1, constant=lb.NONE_ZERO))
app.refresh_color_controls_for_selected_coloration()
app.rebuild_coloration_list()
app.refresh_live_preview_state()
app.new_keyframe()
app.fps_var.set("12")
BLUR = lb.BuddhaBlurSettings(radius=0.04, type=lb.BUDDHA_BLUR_BOKEH)      # the Buddha blur: a record of its own
app.model.set("buddha_blur_radius", BLUR.radius)
app.buddha_blur_type_var.set(lb.BUDDHA_BLUR_LABELS[BLUR.type])
app.on_buddha_blur_changed()
root.update()
check(app.is_dirty(), "edits make the project dirty")
check(app.current_blur_settings() == BLUR, "the blur controls give a BuddhaBlurSettings beside the ray settings")

first = generate()
check(app.generated is not None and int(app.live.pixel_state.sum()) > 0, "an image was generated")
check(first == generate(), "generating twice in the same state gives the same image")
stats_before = app.live.stats.copy()

check(app.save_project_to(path) and not app.is_dirty(), "Save writes the file and clears the dirty mark")
saved = lb.project_signature(app.capture_project())

app.new_project()
check(lb.project_signature(app.capture_project()) == lb.project_signature(app.default_project)
      and len(app.obstacles) == 1 and len(app.colorations) == 1 and not app.keyframes,
      "New gives exactly the starting project")
check(app.generated is None and int(app.live.pixel_state.sum()) == 0, "New leaves a blank image")
check(app.current_blur_settings() == lb.DEFAULT_BUDDHA_BLUR, "New puts the blur controls back to their defaults")

lb.filedialog.askopenfilename = lambda *a, **k: path
app.open_project()
check(lb.project_signature(app.capture_project()) == saved and not app.is_dirty(),
      "Open restores exactly the saved project, without unsaved changes")
check(app.generated is None and int(app.live.pixel_state.sum()) == 0, "Open leaves a blank image")
check(app.current_blur_settings() == BLUR, "Open restores the Buddha blur controls")
second = generate()
check(second == first and np.array_equal(stats_before, app.live.stats),
      "the regenerated image and simulation data are identical to the first ones")
app.model.set("buddha_blur_radius", 0.08)                    # the blur is live: no Start needed
app.on_buddha_blur_changed()
root.update()
check(hashlib.sha256(app.live.img.tobytes()).hexdigest() != second,
      "changing the blur recolors the generated image at once")
check(np.array_equal(stats_before, app.live.stats) and not app.start_btn.instate(["!disabled"])
      and not app.revert_btn.instate(["!disabled"]),
      "...without touching the simulation data, and Start and Revert are not needed")
app.model.set("buddha_blur_radius", BLUR.radius)
app.on_buddha_blur_changed()
root.update()
check(hashlib.sha256(app.live.img.tobytes()).hexdigest() == second and not app.is_dirty(),
      "putting the blur back gives exactly the first image again")

app.iter_var.set("41")
root.update()
check(app.is_dirty(), "an edit after Open is an unsaved change")
asked = len(prompts)
app.new_project()
check(len(prompts) == asked + 1 and app.iter_var.get() == "256",
      "New asks about unsaved changes first (answered: don't save)")

bad = os.path.join(folder, "bad" + lb.PROJECT_EXTENSION)
with open(bad, "w", encoding="utf-8") as handle:
    handle.write("{ not json")
lb.filedialog.askopenfilename = lambda *a, **k: bad
before = lb.project_signature(app.capture_project())
app.open_project()
check(prompts[-1][0] == "ERROR" and lb.project_signature(app.capture_project()) == before,
      "a damaged project file is refused and nothing changes")
root.destroy()
print("all checks passed")
