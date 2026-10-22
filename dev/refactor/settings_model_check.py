"""The real-window check of the settings model (python dev/refactor/settings_model_check.py).

Starts the real program and types into the natural termination entries (the Termination tab's limits and confinement) the way a user does
(entry.delete / entry.insert, then a Return or FocusOut event), asserting that the model, the red entry style, the message under the entries,
the Start / Revert buttons, Start's generated figure, Revert, a keyframe's load and its "modified" marks, and the hover preview all follow.
Prints one line per check and exits with an error at the first failure. Writes only into a temporary folder."""
import os
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
    """Do what Tk does for `sequence` on `entry`. Generated key events are not delivered to a window that has no real focus (as when this
    runs from a tool), so the check asserts that the entry has a binding for it and calls the program's commit handler, which every
    one of these bindings is."""
    check(bool(entry.bind(sequence)), f"the entry has a binding for {sequence}")
    app.on_limit_commit()
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
check(set(model.keys()) == set(lb.NATURAL_TERMINATION_SETTING_SPECS), "the model manages the six natural termination settings")
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

root.destroy()
print("all checks passed")
