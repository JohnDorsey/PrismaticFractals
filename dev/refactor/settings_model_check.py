"""The real-window check of the settings model (python dev/refactor/settings_model_check.py).

Starts the real program and types into the natural termination entries (the Termination tab's limits and confinement), the Buddha blur and the Buddha seeding the way a user does
(entry.delete / entry.insert, then a Return or FocusOut event), asserting that the model, the red entry style, the message under the entries,
the Start / Revert buttons, Start's generated figure, Revert, a keyframe's load and its "modified" marks, and the hover preview all follow.
Prints one line per check and exits with an error at the first failure. Writes only into a temporary folder."""
import os
import re
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import tkinter as tk                                 # noqa: E402
import PrismaticFractals as lb                       # noqa: E402

folder = tempfile.mkdtemp(prefix="lb_model_check_")
lb.paths.EXPORT_DIR = os.path.join(folder, "exports")
lb.paths.TEMP_DIR = os.path.join(lb.paths.EXPORT_DIR, "TEMP")
lb.paths.PROJECT_DIR = os.path.join(folder, "projects")


def check(condition, text):
    print(("ok   " if condition else "FAIL ") + text)
    if not condition:
        raise SystemExit(f"check failed: {text}")


root = tk.Tk()
app = lb.App(root)
app.show_notice = lambda message, can_go_back=False, **kw: True
started = time.time()
while not app.ready and time.time() - started < 180:
    root.update()
check(app.ready, "the program is ready")


def type_into(entry, text):
    """Replace the entry's text by typing `text` one character at a time (so the key-by-key validation runs)."""
    entry.focus_force()
    root.update()
    entry.delete(0, "end")
    for ch in text:
        entry.insert("end", ch)
    root.update()


def press(entry, sequence):
    """Run what Tk runs for `sequence` on `entry`: the callable registered for the binding. (A generated key event is delivered to the
    window that has the focus, and this window cannot be given the operating system's focus from a tool, so the binding's own command
    is called instead.)"""
    script = entry.bind(sequence)
    check(bool(script), f"the entry has a binding for {sequence}")
    root.tk.call(re.search(r"\[(\S+) %#", script).group(1))
    root.update()


def style(entry):
    return str(entry.cget("style"))


def enabled(button):
    return not button.instate(["disabled"])


def generate():
    app.start()
    while app.running:
        root.update()
    root.update()


model = app.model
check(set(model.keys()) == set(lb.SETTING_SPECS) and len(model.keys()) == 37, "the model manages every static setting")
check(model.record(lb.NaturalTerminationSettings) == app.current_natural_termination_settings() == lb.DEFAULT_NATURAL_TERMINATION_SETTINGS,
      "at start the model and the controls hold the defaults")

# ---- typing a valid value
generate()
check(not enabled(app.start_btn) and not enabled(app.revert_btn), "after Start, neither Start nor Revert is needed")
type_into(app.iter_entry, "20")
check(model.value("iter") == 20 and model.error("iter") is None, "typing 20 into the iteration limit reaches the model")
check(style(app.iter_entry) == "TEntry" and app.limit_msg_var.get() == "", "a valid entry is not red and shows no message")
check(enabled(app.start_btn) and enabled(app.revert_btn), "the changed limit enables Start and Revert")

# ---- typing something invalid
type_into(app.iter_entry, "0")
check(model.error("iter") is not None and model.value("iter") == 20, "0 is an error and the model keeps 20")
check(style(app.iter_entry) == "Invalid.TEntry", "the iteration entry turns red")
check("Iteration limit must be a whole number" in app.limit_msg_var.get(), "the message says what is wrong")
check(app.current_natural_termination_settings() is None and app.current_generation_settings() is None, "no settings while an entry is invalid")
check(not enabled(app.start_btn), "Start is disabled while an entry is invalid")

type_into(app.confine_threshold_entry, "0")
check("Threshold must be a number" in app.confine_msg_var.get() and style(app.confine_threshold_entry) == "Invalid.TEntry",
      "an invalid confinement threshold shows its own message")

# ---- commit puts the last valid value back
press(app.iter_entry, "<Return>")
check(app.iter_var.get() == "20" and style(app.iter_entry) == "TEntry" and app.limit_msg_var.get() == "", "Return restores the last valid 20")
press(app.confine_threshold_entry, "<FocusOut>")
check(app.confine_threshold_var.get() == str(lb.DEFAULT_CONFINE_THRESHOLD) and app.confine_msg_var.get() == "",
      "focus-out restores the last valid confinement threshold")
type_into(app.confine_threshold_entry, "4E-3")
press(app.confine_threshold_entry, "<Return>")
check(app.confine_threshold_var.get() == "0.004" and model.value("confine_threshold") == 0.004, "committing 4E-3 tidies it to 0.004")

# ---- the dropdown
app.confine_method_var.set(lb.CONFINE_METHOD_NAMES[lb.CONFINE_PATH])
root.update()
check(model.value("confine_method") == lb.CONFINE_PATH and "Path length" in app.confine_threshold_label_var.get(),
      "choosing the Path length method reaches the model and relabels the threshold")

# ---- Start uses what was typed
generate()
figure = app.generated.natural_termination
check((figure.max_iter, figure.confine_method, figure.confine_threshold) == (20, lb.CONFINE_PATH, 0.004), "Start generates with the typed limits")
check(not enabled(app.start_btn) and not enabled(app.revert_btn), "(after Start the baseline is what was typed: neither button is needed)")

# ---- Revert goes back to the baseline of the last Start, writing the controls through the model
type_into(app.dist_entry, "7")
type_into(app.escape_radius_entry, "3.5")
check(model.value("dist") == 7 and model.value("escape_radius") == 3.5, "two more entries are typed")
app.revert_settings()
root.update()
check(app.dist_var.get() == "0" and app.escape_radius_var.get() == str(lb.DEFAULT_ESCAPE_RADIUS) and model.value("dist") == 0,
      "Revert writes the baseline back into the entries and the model")
check(not enabled(app.revert_btn), "Revert is then disabled")

# ---- a keyframe loads values through the model and typing marks them modified
type_into(app.iter_entry, "33")
app.new_keyframe()
root.update()
kf = app.keyframes[-1]
check(kf.data["iter"] == 33, "a new keyframe captures what was typed")
app.deselect_keyframe()                   # (a new keyframe is selected; reselecting the selected one changes nothing)
type_into(app.iter_entry, "44")
app.select_keyframe(kf)
root.update()
check(app.iter_var.get() == "33" and model.value("iter") == 33, "selecting the keyframe loads 33 into the entry and the model")
check("iter" not in app.kf_modified, "loading it is not an edit")
type_into(app.iter_entry, "55")
app.check_modified()
check("iter" in app.kf_modified, "typing with the keyframe selected marks the setting modified")
app.deselect_keyframe()

# ---- an out-of-range value from a keyframe is an error like typed text, and Start is refused
app.apply_setting_values({"iter": 99999})
root.update()
check(model.error("iter") is not None and style(app.iter_entry) == "Invalid.TEntry", "a keyframe value out of range shows as invalid")
app.revert_settings()
root.update()
check(model.error("iter") is None and style(app.iter_entry) == "TEntry", "Revert clears it")

# ---- the hover preview follows the model
type_into(app.confine_lookback_entry, "6")
app.generated = None                      # (before the first Start the preview uses the controls' last valid values)
check(app.preview_natural_termination_settings().confine_lookback == 6, "the preview reads the model's value")
type_into(app.confine_lookback_entry, "1")
check(app.preview_natural_termination_settings().confine_lookback == 6, "...and keeps the last valid one while the entry is invalid")

# ---- the Buddha blur
type_into(app.buddha_blur_radius_entry, "0.2")
check(model.value("buddha_blur_radius") == 0.2 and app.current_blur_settings().radius == 0.2, "typing a blur radius reaches the model at once")
type_into(app.buddha_blur_radius_entry, "0.9")
check(model.value("buddha_blur_radius") == lb.BUDDHA_BLUR_RADIUS_MAX, "a radius above the maximum counts as the maximum")
type_into(app.buddha_blur_radius_entry, "-")
check(model.error("buddha_blur_radius") is None and style(app.buddha_blur_radius_entry) == "TEntry" and model.value("buddha_blur_radius") == lb.BUDDHA_BLUR_RADIUS_MAX,
      "half-typed text is ignored without an error (the last valid value stays)")
press(app.buddha_blur_radius_entry, "<Return>")
check(app.buddha_blur_radius_var.get() == "0.5", "committing puts the last valid radius back")
check(app.blur_job is None, "...and the recolouring it started has run")
app.buddha_blur_type_var.set(lb.BUDDHA_BLUR_LABELS[2])
check(app.current_blur_settings() == lb.BuddhaBlurSettings(radius=0.5, type=2), "choosing a blur type reaches the model")

# ---- the Buddha seeding
check(app.model.value("buddha_seed_camera") is True and app.buddha_seed_x_start_entry.instate(["disabled"]), "at start the camera pixels are the seeds and the zone is greyed out")
app.buddha_seed_camera_check.invoke()
root.update()
check(app.model.value("buddha_seed_camera") is False and not app.buddha_seed_x_start_entry.instate(["disabled"]), "unticking the checkbox reaches the model and enables the zone")
check(app.seed_job is None, "...and the recolouring it started has run")
type_into(app.buddha_seed_x_start_entry, "-1.5")
check(model.value("buddha_seed_x_start") == -1.5 and app.current_buddha_seed_settings().x_start == -1.5, "typing a zone value reaches the model")
type_into(app.buddha_seed_x_start_entry, "-")
check(model.error("buddha_seed_x_start") is None and app.current_buddha_seed_settings().x_start == -1.5, "a half-typed zone value counts as the last valid one")
press(app.buddha_seed_x_start_entry, "<FocusOut>")
check(app.buddha_seed_x_start_var.get() == "-1.5", "committing puts the last valid zone value back")
type_into(app.buddha_seed_y_stop_entry, ".25")
press(app.buddha_seed_y_stop_entry, "<Return>")
check(app.buddha_seed_y_stop_var.get() == "0.25", "committing tidies .25 to 0.25")

# ---- the ray-bouncing engine's settings
type_into(app.confine_lookback_entry, "6")           # (an earlier step left this entry invalid)
check(app.emit_target_r_entry.instate(["disabled"]) and app.model.value("emit_mode") == lb.EMIT_UP, "with the direction Up the target entries are greyed out")
app.emit_mode_var.set(lb.EMIT_MODE_NAMES[lb.EMIT_TOWARDS])
root.update()
check(model.value("emit_mode") == lb.EMIT_TOWARDS and not app.emit_target_r_entry.instate(["disabled"]), "choosing Towards point reaches the model and enables the target entries")
type_into(app.emit_target_theta_entry, "-")
check(model.error("emit_target_theta") is not None and style(app.emit_target_theta_entry) == "Invalid.TEntry", "a half-typed target angle turns its entry red")
check("Target position values must be numbers" in app.emit_msg_var.get(), "...and shows the message")
check(app.current_ray_settings() is None and app.current_generation_settings() is None and not enabled(app.start_btn), "...and there are no ray settings, so Start is disabled")
press(app.emit_target_theta_entry, "<Return>")
check(app.emit_target_theta_var.get() == "0.0" and app.emit_msg_var.get() == "" and style(app.emit_target_theta_entry) == "TEntry", "committing puts the last valid target angle back")
type_into(app.emit_target_theta_entry, "-720")
check(app.current_ray_settings().emit_target_theta == -720.0, "a valid target angle reaches the ray settings")
type_into(app.emit_offset_entry, "12.345")
check(app.current_ray_settings().emit_offset == 12.35, "the emission offset is rounded to two places in the settings")
press(app.emit_offset_entry, "<Return>")
check(app.emit_offset_var.get() == "12.35", "committing tidies the offset to its two-place text")
type_into(app.seg_mult_entry, "5")
check(app.current_ray_settings().seg_mult == lb.SEGMENT_MULT_MAX, "a segment multiplier above its range counts as the maximum")
type_into(app.seg_mult_entry, "0.5")
press(app.seg_mult_entry, "<Return>")
check(app.seg_mult_var.get() == "0.50" and model.value("seg_mult") == 0.5, "committing tidies the segment multiplier to 0.50")
check(app.bnd_jitter_entry.instate(["disabled"]) and app.rng_seed_entry.instate(["disabled"]), "with fuzzing off its entries are greyed out")
app.fuzz_enable_check.invoke()
root.update()
check(model.value("fuzz_on") is True and not app.bnd_jitter_entry.instate(["disabled"]) and not app.rng_seed_entry.instate(["disabled"]),
      "ticking Enable fuzzing reaches the model and enables the entries")
type_into(app.bnd_jitter_entry, "999")
check(app.current_ray_settings().fuzz.bnd_jitter == 180.0, "a jitter above 180 counts as 180")
press(app.bnd_jitter_entry, "<Return>")
check(app.bnd_jitter_var.get() == "180.0", "committing tidies it to 180.0")
type_into(app.rng_seed_entry, "12")
check(model.value("seed") == 12 and style(app.rng_seed_entry) == "TEntry", "typing a seed reaches the model")
before = model.value("seed")
app.randomize_rng_seed()
root.update()
check(app.rng_seed_var.get() == str(model.value("seed")) and 0 <= model.value("seed") <= 999999999, "Randomize writes a seed into the model and the entry")
generate()
ray = app.generated.ray
check((ray.emit_mode, ray.emit_target_theta, ray.emit_offset, ray.seg_mult, ray.fuzz.enabled, ray.fuzz.bnd_jitter, ray.fuzz.rng_seed)
      == (lb.EMIT_TOWARDS, -720.0, 12.35, 0.5, True, 180.0, model.value("seed")), "Start generates with the typed ray settings")
type_into(app.seg_mult_entry, "0.9")
type_into(app.emit_offset_entry, "3")
app.revert_settings()
root.update()
check(app.seg_mult_var.get() == "0.50" and app.emit_offset_var.get() == "12.35" and app.fuzz_enabled_var.get() is True, "Revert writes the ray settings back from the baseline")
app.deselect_keyframe()
app.new_keyframe()
root.update()
kf = app.keyframes[-1]
check(kf.data["seg_mult"] == 0.5 and kf.data["emit_offset"] == 12.35 and kf.data["bnd_jitter"] == 180.0, "a new keyframe captures the ray settings from the model")
type_into(app.seg_mult_entry, "0.7")
app.check_modified()
check("seg_mult" in app.kf_modified, "typing with the keyframe selected marks the segment multiplier modified")
app.deselect_keyframe()
app.fuzz_enable_check.invoke()                       # (fuzzing off again, so that it does not slow what follows)
root.update()
check(model.value("fuzz_on") is False and app.bnd_jitter_entry.instate(["disabled"]), "unticking it greys the entries again")

# ---- the engine and the Mandelbrot engine's settings
check(model.value("engine") == lb.ENGINE_RAY_BOUNCING and app.current_engine_settings() == lb.DEFAULT_ENGINE_SETTINGS, "the ray engine is selected at start")
check(app.orbit_exponent_entry.instate(["disabled"]) and app.pre_seg_mult_entry.instate(["disabled"]), "under the ray engine the Mandelbrot controls are greyed out")
app.engine_var.set("Mandelbrot")
app.on_engine_changed()
root.update()
check(model.value("engine") == lb.ENGINE_MANDELBROT and app.engine_applied == lb.ENGINE_MANDELBROT and app.current_engine_description().uses_orbit_settings,
      "choosing Mandelbrot in the dropdown reaches the model and the engine switch ran")
check(not app.orbit_exponent_entry.instate(["disabled"]) and not app.pre_seg_mult_entry.instate(["disabled"]), "...and the Mandelbrot controls are enabled")
check(app.orbit_julia_re_entry.instate(["disabled"]) and not app.orbit_z0_combo.instate(["disabled"]), "with the pixel as c, the Julia constant is greyed out and z_0 is not")
type_into(app.orbit_exponent_entry, "3")
type_into(app.orbit_exponent_entry, "-")
check(app.current_mandelbrot_settings().exponent == 3.0 and model.error("orbit_exponent") is None and style(app.orbit_exponent_entry) == "TEntry",
      "a half-typed exponent counts as the last valid one, with no error")
press(app.orbit_exponent_entry, "<Return>")
check(app.orbit_exponent_var.get() == "3.0", "committing puts the last valid exponent back")
app.orbit_mapping_var.set(lb.ORBIT_MAPPING_NAMES[lb.ORBIT_MAPPING_JULIA])
app.on_orbit_changed()
root.update()
check(model.value("orbit_mapping") == lb.ORBIT_MAPPING_JULIA and not app.orbit_julia_re_entry.instate(["disabled"]) and app.orbit_z0_combo.instate(["disabled"]),
      "choosing the Julia mapping reaches the model, enables the Julia constant and greys out z_0")
type_into(app.pre_seg_mult_entry, "0.5")
press(app.pre_seg_mult_entry, "<Return>")
check(app.pre_seg_mult_var.get() == "0.50" and app.current_mandelbrot_settings().pre_seg_mult == 0.5, "the before-multiplier is committed to 0.50 and read by the settings")
check(app.mandelbrot_defines_figure(lb.DEFAULT_MANDELBROT_SETTINGS) and enabled(app.start_btn), "a Mandelbrot setting is figure-defining under Mandelbrot: Start is enabled")
app.engine_var.set("Ray Bouncing")
app.on_engine_changed()
root.update()
check(model.value("engine") == lb.ENGINE_RAY_BOUNCING and app.orbit_exponent_entry.instate(["disabled"]) and app.pre_seg_mult_entry.instate(["disabled"]),
      "choosing the ray engine again greys the Mandelbrot controls out")
check(app.orbit_exponent_var.get() == "3.0" and app.current_mandelbrot_settings().pre_seg_mult == 0.5, "...but they keep their values")

# ---- a project captures and applies them
type_into(app.confine_lookback_entry, "6")           # (an earlier step left this entry invalid, and a project cannot be captured then)
project = app.capture_project()
check(project.buddha_seed.x_start == -1.5 and project.buddha_seed.camera_pixels is False and project.buddha_blur == lb.BuddhaBlurSettings(radius=0.5, type=2),
      "the project captures the blur and the seeding from the model")
app.model.set("buddha_seed_x_start", 0.75)
app.model.set("buddha_blur_radius", 0.1)
root.update()
check(app.buddha_seed_x_start_var.get() == "0.75" and app.buddha_blur_radius_var.get() == "0.1", "a program write shows in the controls")
app.apply_project(lb.project_from_dict(lb.project_to_dict(project)))
root.update()
check(app.buddha_seed_x_start_var.get() == "-1.5" and app.buddha_blur_radius_var.get() == "0.5", "applying a project writes them back into the controls")

root.destroy()
print("all checks passed")
