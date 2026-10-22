"""Changing the path engine, and keeping every value valid for the engine in use (Engine_Refactor_Design.md D7, D18-D20, D35, D36): the notice
dialog, the engine switch with its pop-up and "go back", the engine memory, what selecting a keyframe with values the engine does not accept
does, the guard on saving values into a keyframe, and the correction made when a render starts. The rules themselves are the pure functions of
engines.py; this is the part that reads and writes the controls."""
import tkinter as tk
from tkinter import ttk
from .colorations import coloration_channel_value
from .constants import DEFAULT_ENGINE, ENGINE_NAMES
from .engines import (
    EMPTY_ENGINE_MEMORY, KEYFRAME_HEADER, correct_channel, correct_settings, correction_message, engine_description, go_back,
    keyframe_fixes, switch_engine
)
from .shapes import obstacle_tab_title

RENDER_HEADER = "Some values were clamped or defaulted before rendering:"
SAVE_HEADER = "Saving this would leave some keyframe values invalid for the engine in effect. These changes are suggested:"


class EngineFlowMixin:
    # ---- the notice -------------------------------------------------------------------- #
    def show_notice(self, message, can_go_back=False, title="Some settings changed"):
        """Show `message` in a window of its own and wait: True when the user pressed [ok], False for [go back] (only offered with
        `can_go_back`; closing the window then counts as going back)."""
        answer = {"ok": not can_go_back}
        top = tk.Toplevel(self.root)
        top.title(title)
        top.transient(self.root)
        top.resizable(False, False)
        text = tk.Text(top, width=96, height=min(max(message.count("\n") + 2, 4), 26), wrap="word", borderwidth=0, padx=8, pady=8)
        text.insert("1.0", message)
        text.configure(state="disabled")
        text.grid(row=0, column=0, columnspan=2, sticky="nsew")

        def close(ok):
            answer["ok"] = ok
            top.destroy()
        buttons = ttk.Frame(top)
        buttons.grid(row=1, column=0, sticky="e", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="ok", command=lambda: close(True)).pack(side="left", padx=4)
        if can_go_back:
            ttk.Button(buttons, text="go back", command=lambda: close(False)).pack(side="left", padx=4)
        top.protocol("WM_DELETE_WINDOW", lambda: close(not can_go_back))
        top.grab_set()
        self.root.wait_window(top)
        return answer["ok"]

    def engine_titles(self):
        """The names the corrections' wording gives the obstacles and colorations (as the keyframe list does)."""
        titles = {f"obst:{o.id}": obstacle_tab_title(i, o) for i, o in enumerate(self.obstacles)}
        titles.update({f"coloration:{v.id}": f"Coloration {i + 1}" for i, v in enumerate(self.colorations)})
        return titles

    # ---- writing values into the controls ------------------------------------------------ #
    def apply_setting_values(self, values):
        """Write `values` ({setting key: value}, any mix of static settings, obstacle fields and coloration channels) into the controls,
        then refresh what depends on them once. Not an edit the keyframe being edited should see as one while it is done."""
        self.kf_applying = True
        try:
            with self.model.batch():
                for key, value in values.items():
                    self.load_value(key, value)
            self.on_limit_var_changed()
            self.on_shape_param_changed()
            self.on_fuzz_changed()
            self.on_buddha_blur_changed()
            self.on_buddha_seeds_changed()
            self.on_orbit_changed()
            self.on_framings_loaded()
        finally:
            self.kf_applying = False
        self.apply_engine_offers()
        self.refresh_timeline_diffs()

    # ---- switching engine (D7, D18) --------------------------------------------------------- #
    def switch_to_selected_engine(self):
        """The engine dropdown now shows another engine than the one the program has been using (`engine_applied`): everything the new
        engine does not accept is reset (what it had comes back where nothing has changed since), the user is told - and may go back."""
        new_code = ENGINE_NAMES.index(self.engine_var.get())
        old_code = self.engine_applied
        settings = None if new_code == old_code else self.capture_all_settings(show_error=False)
        if settings is None:                                # nothing to switch, or an entry holds nonsense for the moment
            self.apply_engine_offers()
            return
        settings["engine"] = old_code                       # (the dropdown already shows the new one: what is being left is the old)
        result = switch_engine(settings, engine_description(old_code), engine_description(new_code), self.engine_memory, self.engine_titles())
        self.apply_setting_values({key: result.settings[key] for key in result.touched})
        self.engine_memory = result.memory
        if result.corrections and not self.show_notice(correction_message(result.corrections), can_go_back=True):
            undone, memory = go_back(result, self.capture_all_settings(show_error=False) or result.settings)
            self.apply_setting_values({key: undone[key] for key in result.touched if key in undone})
            self.engine_memory = memory

    # ---- a keyframe's values (D19, D36) ----------------------------------------------------- #
    def animated_value(self, kf, key, sorted_kfs=None):
        """The value `key` resolves to at kf's own frame, blended from the keyframes that define it (kf itself does not)."""
        from .settings import resolve_value
        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame_index) if sorted_kfs is None else sorted_kfs
        return resolve_value(key, sorted_kfs, kf.frame_index, self.setting_fallback(key), self.setting_is_float(key))

    def correct_loaded_values(self, values, header=KEYFRAME_HEADER):
        """`values` ({setting key: value}, what selecting a keyframe is about to load) corrected for the engine they put in effect
        (the one among them, else the program's): (corrected values, the message to show or None)."""
        code = values.get("engine", ENGINE_NAMES.index(self.engine_var.get()))
        corrected, corrections = correct_settings(values, engine_description(code), self.engine_titles())
        return corrected, (correction_message(corrections, header) if corrections else None)

    def commit_to_keyframe(self, kf, new_values, is_new=False):
        """Put `new_values` ({key: value}) into keyframe `kf` - unless that leaves keyframes holding values the engines in effect do not
        accept (D19): then the changes that would put it right are shown with [ok] and [go back]; [ok] makes them along with the save,
        [go back] cancels the save (False). `is_new`: `kf` is not in the list yet (it is added when True is returned)."""
        from .settings import Keyframe
        prospective = [Keyframe(k.id, k.frame_index, dict(k.data)) for k in self.keyframes if k is not kf]
        prospective.append(Keyframe(kf.id, kf.frame_index, {**kf.data, **new_values}))
        fixes = keyframe_fixes(prospective, ENGINE_NAMES.index(self.engine_var.get()), saved={(kf.id, key) for key in new_values},
                               is_float=self.setting_is_float, titles=self.engine_titles())
        if fixes and not self.show_notice(correction_message([fix.correction for fix in fixes], SAVE_HEADER), can_go_back=True):
            return False
        kf.data.update(new_values)
        by_id = {k.id: k for k in self.keyframes}
        by_id[kf.id] = kf
        for fix in fixes:
            by_id[fix.kf_id].data[fix.key] = fix.new
        if is_new:
            self.keyframes.append(kf)
        return True

    # ---- when a render starts (D35) ---------------------------------------------------------- #
    def correct_controls_for_render(self):
        """Before Start: whatever the controls hold that the engine in use does not accept is corrected in the controls, with a notice
        (rare: the switch and loading already correct what they bring in)."""
        settings = self.capture_all_settings(show_error=False)
        if settings is None:
            return
        corrected, corrections = correct_settings(settings, engine_description(ENGINE_NAMES.index(self.engine_var.get())), self.engine_titles())
        if corrections:
            self.apply_setting_values({c.key: corrected[c.key] for c in corrections})
            self.show_notice(correction_message(corrections, RENDER_HEADER))

    def correct_new_coloration(self, coloration):
        """A coloration just made has the usual defaults, which the engine in use may not offer (the default one counts bounces): give it
        the engine's own (D39). Not a reset, so no notice."""
        description = engine_description(ENGINE_NAMES.index(self.engine_var.get()) if hasattr(self, "engine_var") else DEFAULT_ENGINE)
        for c in range(3):
            channel, _changes = correct_channel(coloration_channel_value(coloration, c), description, c)
            coloration.set_channel(c, channel)
