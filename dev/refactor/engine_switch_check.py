"""The real-window check of the engine switch (python dev/refactor/engine_switch_check.py [screenshot folder]).

Starts the real program, answers its notices
itself, and asserts that choosing the Mandelbrot engine resets what it does not accept (and says so), that an image and the hover preview
work with it - the Mandelbrot set and a Julia set - that going back and switching back put everything as it was (the ray engine's image is
pixel-identical to the first), that saving the engine into a keyframe offers the fixes for the other keyframes, and that the progress bar can
carry the frame count. With a folder, it also saves screenshots of the Simulation tab and the Colorations tab there (needs a screen).
Prints one line per check and exits with an error at the first failure. Writes only into a temporary folder."""
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
import light_bounces as lb                           # noqa: E402

folder = tempfile.mkdtemp(prefix="lb_engine_check_")
lb.paths.EXPORT_DIR = os.path.join(folder, "exports")
lb.paths.TEMP_DIR = os.path.join(lb.paths.EXPORT_DIR, "TEMP")
lb.paths.PROJECT_DIR = os.path.join(folder, "projects")
shots = sys.argv[1] if len(sys.argv) > 1 else None
notices, answers = [], []


def check(condition, text):
    print(("ok   " if condition else "FAIL ") + text)
    if not condition:
        raise SystemExit(f"check failed: {text}")


root = tk.Tk()
app = lb.App(root)
app.show_notice = lambda message, can_go_back=False, **kw: notices.append((message, can_go_back)) or (answers.pop(0) if answers else True)
started = time.time()
while not app.ready and time.time() - started < 180:
    root.update()
check(app.ready, "the program is ready")
check(app.engine_combo.cget("values") == ("Ray Bouncing", "Mandelbrot") or list(app.engine_combo.cget("values")) == ["Ray Bouncing", "Mandelbrot"],
      "the dropdown offers both engines")


def generate():
    app.start()
    while app.running:
        root.update()
    root.update()
    return hashlib.sha256(app.img.tobytes()).hexdigest()


def choose(name):
    app.engine_var.set(name)
    app.on_engine_changed()
    root.update()


def screenshot(name):
    if not shots:
        return
    from PIL import ImageGrab
    root.update()
    ImageGrab.grab(bbox=(root.winfo_rootx(), root.winfo_rooty(), root.winfo_rootx() + root.winfo_width(),
                         root.winfo_rooty() + root.winfo_height())).save(os.path.join(shots, name))


def select_tab(notebook, text):
    for tab in notebook.tabs():
        if notebook.tab(tab, "text") == text:
            notebook.select(tab)
    root.update()


ray_hash = generate()
start_project = lb.project_signature(app.capture_project())
check(all(o.in_effect == lb.OBSTACLE_EFFECT_REFLECT for o in app.obstacles), "the starting obstacle reflects")

choose("Mandelbrot")
check(len(notices) == 1 and notices[0][1] and "have been reset" in notices[0][0], "choosing Mandelbrot says what it reset, with go back")
check(all(o.in_effect == lb.OBSTACLE_EFFECT_PASS and o.out_effect == lb.OBSTACLE_EFFECT_PASS for o in app.obstacles),
      "the obstacle now passes rays through")
check(app.engine_applied == lb.ENGINE_MANDELBROT and list(app.obs_in_effect_combo.cget("values")) == ["Terminate", "Pass through"],
      "the obstacle effect dropdowns no longer list Reflect")
check(not app.orbit_exponent_entry.instate(["disabled"]) and app.orbit_julia_re_entry.instate(["disabled"]),
      "the Mandelbrot controls are on, the Julia constant is greyed")
select_tab(app.controls_notebook, "Simulation")
screenshot("simulation_mandelbrot.png")
select_tab(app.controls_notebook, "Emissions")
screenshot("emissions_greyed.png")
check(app.emit_mode_combo.instate(["disabled"]) and app.fuzz_enable_check.instate(["disabled"]),
      "the Emissions tab and fuzzing are greyed out under Mandelbrot (nothing is hidden)")
check(not app.seg_mult_scale.widget.instate(["disabled"]) and not app.pre_seg_mult_scale.widget.instate(["disabled"]),
      "both segment length multipliers are on under Mandelbrot")
check(app.iter_entry.instate(["!disabled"]) and app.escape_radius_entry.instate(["!disabled"]), "the Termination tab is not")
select_tab(app.controls_notebook, "Simulation")
mandelbrot_hash = generate()
check(mandelbrot_hash != ray_hash and bool((app.img != np.array(lb.BACKGROUND, dtype=np.uint8)).any()), "an image of the Mandelbrot engine is made")
app.update_preview(300, 256)
check("iterations" in app.info_vars["outcome"].get(), "the hover preview of an orbit: " + app.info_vars["outcome"].get())
app.orbit_mapping_var.set(lb.ORBIT_MAPPING_NAMES[1])
app.orbit_julia_re_var.set("-0.8")
app.orbit_julia_im_var.set("0.156")
app.on_orbit_commit()
check(app.start_btn.instate(["!disabled"]), "changing a Mandelbrot setting is a different figure: Start is enabled")
check(not app.orbit_julia_re_entry.instate(["disabled"]) and app.orbit_z0_combo.instate(["disabled"]), "the pixel is now z_0: the Julia constant is on, z_0 is greyed")
julia_hash = generate()
check(julia_hash not in (ray_hash, mandelbrot_hash), "a Julia set is made")
select_tab(app.color_notebook if hasattr(app, "color_notebook") else app.controls_notebook, "Colorations")
screenshot("colorations_mandelbrot.png")

app.orbit_mapping_var.set(lb.ORBIT_MAPPING_NAMES[0])
app.on_orbit_changed()
notices.clear()
choose("Ray Bouncing")
check(notices == [] and all(o.in_effect == lb.OBSTACLE_EFFECT_REFLECT for o in app.obstacles), "switching back restores the reflecting obstacle, with no notice")
check(not app.emit_mode_combo.instate(["disabled"]) and not app.fuzz_enable_check.instate(["disabled"]) and not app.seg_mult_scale.widget.instate(["disabled"]),
      "and the ray engine's controls are back on")
check(app.pre_seg_mult_scale.widget.instate(["disabled"]), "the pre-calculation length multiplier is greyed under the ray engine (Mandelbrot only)")
check(app.emit_jitter_scale.widget.instate(["disabled"]), "(fuzzing is off, so its own settings stay greyed: the control's own rule)")
check(lb.project_signature(app.capture_project()) != start_project, "(the Julia settings the user changed meanwhile are kept)")
app.orbit_julia_re_var.set("0.0")
app.orbit_julia_im_var.set("0.0")
app.on_orbit_commit()
check(generate() == ray_hash, "the ray engine's image is pixel-identical to the first")

answers.append(False)
choose("Mandelbrot")
check(app.engine_var.get() == "Ray Bouncing" and app.engine_applied == lb.ENGINE_RAY_BOUNCING
      and all(o.in_effect == lb.OBSTACLE_EFFECT_REFLECT for o in app.obstacles), "go back puts the engine and the values back")

# keyframes (D19, D36)
app.new_keyframe()                                                  # frame 0: ray engine, Reflect, everything defined
notices.clear()
app.new_keyframe_empty()                                            # frame 1: defines nothing; selecting it loads what frame 0 animates
choose("Mandelbrot")
notices.clear()
answers.append(True)
app.save_settings(app.keyframes[1], ["engine"])                    # the first keyframe's Reflect is held under it
check(len(notices) == 1 and notices[0][1] and "Keyframe at frame 0" in notices[0][0], "saving the engine offers the fix for the keyframe before it")
check(app.keyframes[0].data[f"obst:{app.obstacles[0].id}:in_effect"] == lb.OBSTACLE_EFFECT_PASS and app.keyframes[1].data == {"engine": 1},
      "[ok] made it, and saved the engine")
app.deselect_keyframe()
choose("Ray Bouncing")
app.keyframes[0].data[f"obst:{app.obstacles[0].id}:in_effect"] = lb.OBSTACLE_EFFECT_REFLECT   # (a project edited by hand: allowed, D19)
app.keyframes[1].data["engine"] = lb.ENGINE_MANDELBROT
notices.clear()
app.select_keyframe(app.keyframes[1])
check(app.engine_var.get() == "Mandelbrot", "selecting the keyframe loads its engine")
app.deselect_keyframe()
check(app.keyframes[0].data[f"obst:{app.obstacles[0].id}:in_effect"] == lb.OBSTACLE_EFFECT_REFLECT, "a keyframe holding an invalid value is left as it is")
app.keyframes[0].data["engine"] = lb.ENGINE_MANDELBROT                # (the first keyframe now holds Reflect under the Mandelbrot engine)
notices.clear()
app.select_keyframe(app.keyframes[0])
check(app.engine_var.get() == "Mandelbrot" and len(notices) == 1 and "keyframe have been clamped or defaulted" in notices[0][0],
      "selecting it loads its values corrected, with a message that only says ok")
app.deselect_keyframe()

# the margin of an export: the settings the engine ignores are grey (looked at in the screenshot)
if shots:
    figure = app.current_figure()
    blocks = lb.figure_info_blocks(figure.ray, figure.natural_termination, list(figure.obstacle_snapshots), lb.EngineSettings(engine=lb.ENGINE_MANDELBROT),
                                   app.current_mandelbrot_settings())
    lb.write_png(os.path.join(shots, "margin_mandelbrot.png"), lb.add_left_info_margin(np.zeros((400, 200, 3), np.uint8), blocks))

# the coloration editor's scrolling area must lie inside the tab (once it was taller than the tab and the last rows could not be reached)
select_tab(app.controls_notebook, "Perception")
select_tab(app.perception_notebook, "Colorations")
for c in range(3):
    app.selected_coloration_obj().chan_src[c] = lb.CHAN_SRC_BUDDHA
app.refresh_color_controls_for_selected_coloration()
root.update()
editor = [w for w in app.colorations_tab.winfo_children() if w.winfo_class() == "TLabelframe"][0]
check(editor.winfo_rooty() + editor.winfo_height() <= app.colorations_tab.winfo_rooty() + app.colorations_tab.winfo_height(),
      "the coloration editor, with three Buddha channels, fits inside its tab")

# an option the engine lacks, ticked under the ray engine, is unticked AND greyed in the editor after the switch, and a click on another
# checkbox does not bring it back (the editor was once never refreshed, so the stale tick was written back and the live image drew it)
midpoints = lb.BUDDHA_POINT_LABELS.index("All self bounces")
choose("Ray Bouncing")
app.buddha_point_vars[1][midpoints].set(True)
app.on_buddha_points_changed(1)
check(app.selected_coloration_obj().buddha_points[1][midpoints], "(the ray engine takes All self bounces)")
choose("Mandelbrot")
check(not app.buddha_point_vars[1][midpoints].get() and app.buddha_point_checks[1][midpoints].instate(["disabled"]),
      "after the switch the editor shows All self bounces unticked and greyed out")
other = lb.BUDDHA_POINT_LABELS.index("Path: Last point")
app.buddha_point_vars[1][other].set(True)
app.on_buddha_points_changed(1)
check(not app.selected_coloration_obj().buddha_points[1][midpoints], "ticking another option does not bring the corrected one back")
# a channel switched to Buddha under Mandelbrot must not keep the ray engine's hidden default (it made the image stop fitting: no recolouring)
app.selected_coloration_obj().chan_src[2] = lb.CHAN_SRC_BUDDHA
app.selected_coloration_obj().buddha_points[2] = tuple(i == lb.POINT_ALL_MEAN for i in range(lb.N_BUDDHA_POINTS))
app.chan_src_vars[2].set(lb.CHAN_SRC_LABELS[lb.CHAN_SRC_BUDDHA])
app.on_chan_src_changed(2)
check([i for i, on in enumerate(app.selected_coloration_obj().buddha_points[2]) if on] == [lb.point_slot(lb.SEQ_PATH, lb.PRED_MEAN)]
      and not app.buddha_point_vars[2][lb.POINT_ALL_MEAN].get(), "a channel switched to Buddha under Mandelbrot is given the engine's own option at once")

# the progress bar's frame count
check(app.video_progress_style is not None, "the progress bar has a style that can carry text")
app.set_video_progress_text("3/10")
select_tab(app.bottom_notebook if hasattr(app, "bottom_notebook") else app.controls_notebook, "Video")
screenshot("video_progress.png")
print("all checks passed" + (f"; screenshots in {shots}" if shots else ""))
root.destroy()
