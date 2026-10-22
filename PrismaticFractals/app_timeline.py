"""The timeline: keyframes, the list of every setting a keyframe can define, loading and saving settings to a keyframe, the blue 'modified' marks, and the keyframe list and detail panel."""
import tkinter as tk
from tkinter import messagebox, ttk
from .constants import MODIFIED_POLL_MS, PAD, SELECTED_OUTLINE, SELECTED_OUTLINE_PX, TIMELINE_W
from .kernels import PREVIEW_BOX
from .metrics import CHAN_SRC_BUDDHA, make_mapping
from .settings import (
    DEFAULT_SETTING_VALUES, Keyframe, SETTING_FLOAT_KEYS, SETTING_INFO, SETTING_SPECS,
    SETTING_TABS_ORDER, SETTING_WIDGET_ATTRS, SettingEntry, remove_keyframe_keys, resolve_value,
    settings_to_dict
)
from .shapes import OBST_FIELD_IS_FLOAT, obstacle_tab_title
from .ui_text import (
    DIFF_PRESENCE_CHOICES, diff_section_header_text, setting_passes_presence_filter,
    short_value_text
)
from .colorations import COLORATION_FIELD_INFO, coloration_channel_value, coloration_field_value
from .framings import FRAMING_FIELD_INFO, FRAMING_FIELDS, framing_is_usable
from .widgets import MODIFIED_OUTLINE, MODIFIED_OUTLINE_PX, ModifiedOutline, detect_modified_keys


class TimelineMixin:
    def init_timeline_state(self):
        """The keyframes, which one is selected, the settings changed since it was selected or last saved (the blue outlines), and the
        keyframe settings list's expanded sections (everything starts collapsed)."""
        self.keyframes = []
        self._kf_counter = 0
        self.selected_keyframe = None
        self.kf_modified = set()
        self.kf_last_seen = None          # capture_all_settings() at the previous check
        self.kf_applying = False          # True while a keyframe's values are being loaded
        self.outlines = {}                # widget -> ModifiedOutline
        self.diff_expanded = set()

    def purge_keyframe_settings(self, prefix):
        """Removes a just-deleted obstacle's, coloration's or framing's settings from every
        keyframe right away (see remove_keyframe_keys), so no keyframe is ever
        left holding settings for something that no longer exists - they
        would otherwise be invisible in the Timeline's settings list, and so
        impossible for the user to remove."""
        remove_keyframe_keys(self.keyframes, prefix)
        self.refresh_timeline_diffs()

    def build_timeline_panel(self):
        """Top right: the keyframe list, and beside it, a
        list of every setting, showing how each relates to the selected
        keyframe, grouped by tab."""
        tl = ttk.LabelFrame(self.root, text="Timeline", width=TIMELINE_W, height=PREVIEW_BOX)
        tl.grid(row=0, column=2, sticky="nsew", padx=(PAD // 2, PAD), pady=(PAD, PAD // 2))
        tl.grid_propagate(False)
        tl.columnconfigure(0, weight=1)
        tl.columnconfigure(1, weight=1)
        tl.rowconfigure(2, weight=1)

        new_row = ttk.Frame(tl)
        new_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        new_row.columnconfigure((0, 1), weight=1)
        ttk.Button(new_row, text="New keyframe (from current settings)",
                  command=self.new_keyframe).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(new_row, text="New keyframe (empty)",
                  command=self.new_keyframe_empty).grid(row=0, column=1, sticky="ew",
                                                        padx=(2, 0))

        ttk.Label(tl, text="Keyframes:", foreground="#666").grid(
            row=1, column=0, sticky="w", padx=8)

        diff_header = ttk.Frame(tl)
        diff_header.grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Label(diff_header, text="Settings:", foreground="#666").pack(side="left")
        self.diff_delete_all_btn = ttk.Button(diff_header, text="Delete all",
                                              command=self.delete_all_diffs)
        self.diff_delete_all_btn.pack(side="right")
        self.diff_save_all_btn = ttk.Button(diff_header, text="Save all modified",
                                            command=self.save_modified_settings)
        self.diff_save_all_btn.pack(side="right", padx=(0, 4))

        # Keyframe list (left column): one label per keyframe, click to select
        list_container = ttk.Frame(tl)
        list_container.grid(row=2, column=0, sticky="nsew", padx=8, pady=(2, 4))
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)
        kf_canvas = tk.Canvas(list_container, highlightthickness=0)
        vsb = ttk.Scrollbar(list_container, orient="vertical", command=kf_canvas.yview)
        kf_canvas.configure(yscrollcommand=vsb.set)
        kf_canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.kf_list_frame = ttk.Frame(kf_canvas)
        kf_window = kf_canvas.create_window((0, 0), window=self.kf_list_frame, anchor="nw")
        self.kf_list_frame.bind(
            "<Configure>", lambda _e: kf_canvas.configure(scrollregion=kf_canvas.bbox("all")))
        kf_canvas.bind(
            "<Configure>", lambda e: kf_canvas.itemconfigure(kf_window, width=e.width))
        for background in (kf_canvas, self.kf_list_frame):      # empty background deselects
            background.bind("<Button-1>", lambda _e: self.deselect_keyframe())

        # Settings list (right column): every setting, grouped by tab
        diff_container = ttk.Frame(tl)
        diff_container.grid(row=2, column=1, sticky="nsew", padx=8, pady=(2, 4))
        diff_container.columnconfigure(0, weight=1)
        diff_container.rowconfigure(1, weight=1)
        self.diff_presence_var = tk.StringVar(value=DIFF_PRESENCE_CHOICES[-1])
        filters = ttk.Frame(diff_container)
        filters.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))
        ttk.Label(filters, text="Show", font=self.small_font).pack(side="left")
        combo = ttk.Combobox(filters, textvariable=self.diff_presence_var,
                             values=DIFF_PRESENCE_CHOICES, state="readonly", width=17)
        combo.pack(side="left", padx=(4, 0))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_timeline_diffs())
        diff_canvas = self.diff_canvas = tk.Canvas(diff_container, highlightthickness=0)
        dvsb = ttk.Scrollbar(diff_container, orient="vertical", command=diff_canvas.yview)
        diff_canvas.configure(yscrollcommand=dvsb.set)
        diff_canvas.grid(row=1, column=0, sticky="nsew")
        dvsb.grid(row=1, column=1, sticky="ns")
        self.diff_list_frame = ttk.Frame(diff_canvas)
        self.built.add("diff_list_frame")
        diff_window = diff_canvas.create_window((0, 0), window=self.diff_list_frame, anchor="nw")
        self.diff_list_frame.bind(
            "<Configure>", lambda _e: diff_canvas.configure(scrollregion=diff_canvas.bbox("all")))
        diff_canvas.bind(
            "<Configure>", lambda e: diff_canvas.itemconfigure(diff_window, width=e.width))

        det = ttk.LabelFrame(tl, text="Selected keyframe")
        det.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 8))
        det.columnconfigure(1, weight=1)
        self.kf_detail_label_var = tk.StringVar(value="(none selected)")
        ttk.Label(det, textvariable=self.kf_detail_label_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 4))

        ttk.Label(det, text="Frame index:").grid(
            row=1, column=0, sticky="w", padx=8, pady=2)
        self.kf_frame_var = tk.StringVar(value="")
        vcmd = (self.root.register(self.validate_digits), "%P")
        self.kf_frame_entry = ttk.Entry(det, textvariable=self.kf_frame_var, width=8,
                                        justify="right", validate="key",
                                        validatecommand=vcmd)
        self.kf_frame_entry.grid(row=1, column=1, sticky="w", padx=(0, 8), pady=2)
        self.kf_frame_entry.bind("<FocusOut>", self.on_kf_frame_commit)
        self.kf_frame_entry.bind("<Return>", self.on_kf_frame_commit)
        self.kf_frame_entry.bind("<KP_Enter>", self.on_kf_frame_commit)

        btnrow = ttk.Frame(det)
        btnrow.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 8))
        self.kf_split_btn = ttk.Button(btnrow, text="Split",
                                       command=self.split_selected_keyframe)
        self.kf_split_btn.pack(side="left", padx=(0, 4))
        self.kf_delete_btn = ttk.Button(btnrow, text="Delete",
                                        command=self.delete_selected_keyframe)
        self.kf_delete_btn.pack(side="left")

        ttk.Label(tl, foreground="#666", wraplength=TIMELINE_W - 30,
                  text="Selecting a keyframe loads its settings (and the settings "
                       "its neighbors animate through it) into the controls; a setting "
                       "you then change gets a blue outline until you save it or "
                       "deselect (click the empty part of the keyframe list). Only float "
                       "values are interpolated (linearly) between keyframes. Other "
                       "values are held until the next keyframe."
                 ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))

        self.update_kf_detail_panel()
        self.rebuild_keyframe_list()
        self.root.after(MODIFIED_POLL_MS, self.poll_modified)
        self.refresh_timeline_diffs()

    # ---- timeline: keyframes ------------------------------------------------ #
    def capture_all_settings(self, show_error=True):
        """dict of every canonical setting's current value (plus every current
        obstacle's fields, every current coloration's and every current framing's, as dynamic
        "obst:<id>:<field>" / "coloration:<id>:<field>" / "framing:<id>:<field>" keys), or None (with an
        error dialog, unless show_error is False - for passive/reactive
        callers like refresh_timeline_diffs, which already show their own
        gentle inline message and run on every keystroke, where a popup
        would be disruptive) if the limit fields are currently invalid."""
        snap = self.current_generation_settings()
        if snap is None:
            if show_error:
                messagebox.showerror("Invalid settings",
                                     "Fix the iteration/distance limit fields first.")
            return None
        d = settings_to_dict(*snap, self.current_blur_settings(), self.current_buddha_seed_settings(),
                             self.current_engine_settings(), self.current_mandelbrot_settings())
        d["outline"] = self.deco["outline"].get()
        for o in self.obstacles:
            for field in o.FIELD_NAMES:
                d[f"obst:{o.id}:{field}"] = getattr(o, field)
        for v in self.colorations:
            d[f"coloration:{v.id}:color_model"] = v.color_model
            for c in range(3):
                d[f"coloration:{v.id}:chan{c}"] = coloration_channel_value(v, c)
        for f in self.framings:
            for field in FRAMING_FIELDS:
                d[f"framing:{f.id}:{field}"] = getattr(f, field)
        return d

    def new_keyframe(self):
        """A new keyframe that defines every setting, from the controls' current
        values."""
        data = self.capture_all_settings()
        if data is None:
            return
        frame_index = (0 if not self.keyframes
                      else max(kf.frame_index for kf in self.keyframes) + 1)
        kf = Keyframe(self._kf_counter, frame_index, {})
        if not self.commit_to_keyframe(kf, data, is_new=True):
            return
        self._kf_counter += 1
        self.select_keyframe(kf, apply=False)      # it holds exactly the current values

    def new_keyframe_empty(self):
        """A new keyframe that defines no settings at all; selecting it loads
        whatever its neighbors animate through it."""
        frame_index = (0 if not self.keyframes
                      else max(kf.frame_index for kf in self.keyframes) + 1)
        kf = Keyframe(self._kf_counter, frame_index, {})
        self._kf_counter += 1
        self.keyframes.append(kf)
        self.select_keyframe(kf)

    def load_value(self, key, val):
        """Apply one resolved value to whichever control(s) hold that setting.
        Does NOT itself refresh dependent state - callers batch that after."""
        if key.startswith("obst:"):
            _prefix, obst_id, field = key.split(":", 2)
            obst_id = int(obst_id)
            o = next((o for o in self.obstacles if o.id == obst_id), None)
            if o is not None and field in o.FIELD_NAMES:
                want = o.FIELD_TYPES[field]
                setattr(o, field, want(val) if want is float else
                        int(round(val)) if want is int else bool(val))
                if o is self.selected_obstacle:
                    self.refresh_obstacle_editor()
                self.rebuild_obstacle_list()
            return
        if key.startswith("coloration:"):
            _prefix, coloration_id, field = key.split(":", 2)
            v = self.coloration_by_id(int(coloration_id))
            if v is None:
                return
            if field == "color_model":
                v.color_model = val
            else:
                c = int(field[-1])
                v.set_channel(c, val)
            if v is self.selected_coloration_obj():
                self.refresh_color_controls_for_selected_coloration()
            if v is self.live_coloration():
                self.mapping = make_mapping(v.metric_types, v.point_metric, v.other_metric,
                                            v.chan_src, v.constant_choice)
                if field != "color_model" and v.chan_src[int(field[-1])] == CHAN_SRC_BUDDHA:
                    self.rebuild_buddha_counts()   # the color model never affects Buddha scattering
                                                   # at all, and an unrelated channel's
                                                   # own edit can't either - see
                                                   # apply_mapping_change's identical fix
                self.update_stat_tags()
                self.recolor(0, self.rows_done)
                self.draw()
                if self.pinned_px is not None:
                    self.update_preview(*self.pinned_px)
            self.rebuild_coloration_list()
            return
        if key.startswith("framing:"):
            _prefix, framing_id, field = key.split(":", 2)
            f = self.framing_by_id(int(framing_id))
            if f is not None and field in FRAMING_FIELDS:
                setattr(f, field, float(val))
                self.framings_loaded = True        # (on_framings_loaded, once the batch is in)
            return
        if key == "outline":
            self.deco["outline"].set(val)
            self.on_decoration_changed("outline")
        else:
            self.set_setting_control(key, val)

    def on_framings_loaded(self):
        """Framings' numbers were just loaded from a keyframe (by load_value, which only sets them): show the
        selected one's, make the image an image of it - one new generation however many of its four numbers
        changed - and refresh the thumbnails. A framing made unusable by the mix of numbers (no window to see
        through) is left on screen as it was."""
        if not self.framings_loaded:
            return
        self.framings_loaded = False
        self.refresh_framing_controls_for_selected_framing()
        if framing_is_usable(self.selected_framing_obj()):
            self.apply_selected_framing()
        self.rebuild_coloration_list()             # (which also rebuilds the framings' list)

    def set_setting_control(self, key, val):
        """Write a setting (a SETTING_SPECS key) into its control, through the settings model (which holds every one of them and
        writes the control as its kind needs it). Does NOT refresh what depends on it."""
        self.model.set(key, val)

    def live_setting_values(self):
        """{static setting key: the value its control holds right now}: what an export (a video, or a new image) uses for
        a setting that no keyframe defines - the values the controls held when the export began, never the program's
        defaults. (If the limit fields hold nonsense the controls cannot be read, and the defaults stand in; this is
        only for showing a value in the keyframe list - an export refuses to start then.)"""
        current = self.capture_all_settings(show_error=False)
        return DEFAULT_SETTING_VALUES if current is None else current

    def setting_fallback(self, key):
        """The value to fall back to when no keyframe defines `key` at all, which is the value an export would use for
        it: what the controls hold now for a normal setting (live_setting_values), that obstacle's own current
        value (None if it no longer exists) for an obst:<id>:<field> key, or
        that coloration's (or framing's) own current value (None if it no longer exists)
        for a coloration:<id>:<field> (framing:<id>:<field>) key. An export takes a copy of these when it begins
        (resolve_frame_settings' `base`, and the obstacle, coloration and framing snapshots)."""
        if key.startswith("obst:"):
            _prefix, obst_id, field = key.split(":", 2)
            o = next((o for o in self.obstacles if o.id == int(obst_id)), None)
            return getattr(o, field) if o is not None else None
        if key.startswith("coloration:"):
            _prefix, coloration_id, field = key.split(":", 2)
            v = self.coloration_by_id(int(coloration_id))
            return coloration_field_value(v, field) if v is not None else None
        if key.startswith("framing:"):
            _prefix, framing_id, field = key.split(":", 2)
            f = self.framing_by_id(int(framing_id))
            return getattr(f, field) if f is not None else None
        return self.live_setting_values()[key]

    def setting_is_float(self, key):
        if key.startswith("obst:"):
            return OBST_FIELD_IS_FLOAT[key.split(":", 2)[2]]
        if key.startswith("coloration:"):
            field = key.split(":", 2)[2]
            return next(is_f for f, _label, is_f in COLORATION_FIELD_INFO if f == field)
        if key.startswith("framing:"):
            field = key.split(":", 2)[2]
            return next(is_f for f, _label, is_f in FRAMING_FIELD_INFO if f == field)
        return key in SETTING_FLOAT_KEYS

    def load_animated_setting(self, kf, key, refresh=True):
        """Load the value `key` resolves to at kf's own frame index, blended
        from whichever surrounding keyframes actually define it (kf itself
        doesn't) - without changing kf. `refresh=False` skips the refresh
        side effects (each of which re-walks every current setting, so
        they're too expensive to repeat once per key) - callers loading
        several settings at once should pass False here and do one shared
        refresh afterwards (see apply_keyframe_settings)."""
        self.load_value(key, self.animated_value(kf, key))
        if refresh:
            self.on_limit_var_changed()
            self.on_framings_loaded()
            self.on_shape_param_changed()
            self.on_fuzz_changed()
            self.on_buddha_blur_changed()
            self.on_buddha_seeds_changed()
            self.on_orbit_changed()
            self.refresh_timeline_diffs()

    def save_settings(self, kf, keys):
        """Save several of the controls' current settings into `kf` (adding
        any it didn't already define)."""
        current = self.capture_all_settings()
        if current is None:
            return
        if not self.commit_to_keyframe(kf, {key: current[key] for key in keys if key in current}):
            return                                  # (the keyframes would have been left invalid, and the user went back)
        self.kf_modified.difference_update(keys)
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()
        self.refresh_modified_outlines()

    def delete_setting(self, kf, key):
        """Remove `key` from `kf`, so it goes back to being either animated by
        surrounding keyframes or undefined."""
        if key in kf.data:
            del kf.data[key]
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

    def setting_status(self, kf, key, sorted_kfs, idx_kf):
        """How `key` relates to `kf`: 'present' (kf defines it), 'animated'
        (kf doesn't, but some other keyframe does - on either side or both, so
        its value at kf comes from the neighbors: blended between the nearest
        one on each side, or, with one side only, held from the first/last
        keyframe defining it), or 'none' (no keyframe defines it)."""
        if key in kf.data:
            return "present"
        others = sorted_kfs[:idx_kf] + sorted_kfs[idx_kf + 1:]
        return "animated" if any(key in k2.data for k2 in others) else "none"

    def select_keyframe(self, kf, apply=True):
        """Select `kf`: equivalent to deselecting the current keyframe first (its
        unsaved modifications stop counting as modifications; the values stay in
        the controls until overwritten below), then, with `apply`, loading into
        the controls every setting `kf` defines and every setting its
        neighbors animate through it. Selecting the selected keyframe again
        does nothing."""
        if kf is self.selected_keyframe:
            return
        self.selected_keyframe = kf
        self.kf_modified = set()
        if apply:
            self.apply_keyframe_settings(kf)
        self.kf_last_seen = self.capture_all_settings(show_error=False)
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()
        self.refresh_modified_outlines()

    def deselect_keyframe(self):
        """No keyframe selected. The controls keep their values, but none of them
        counts as modified any more, so selecting another keyframe overwrites
        them freely."""
        if self.selected_keyframe is None:
            return
        self.selected_keyframe = None
        self.kf_modified = set()
        self.kf_last_seen = None
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()
        self.refresh_modified_outlines()

    def apply_keyframe_settings(self, kf):
        """Load into the controls every setting `kf` defines, and every one it
        lacks but its neighbors animate through its frame index, then refresh
        what depends on them once."""
        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame_index)
        idx_kf = sorted_kfs.index(kf)
        values = {}
        for entry in self.all_setting_entries():
            key = entry.key
            status = self.setting_status(kf, key, sorted_kfs, idx_kf)
            if status == "present":
                values[key] = kf.data[key]
            elif status == "animated":
                values[key] = self.animated_value(kf, key, sorted_kfs)
        values, notice = self.correct_loaded_values(values)         # (D36: what the engine it puts in effect does not accept)
        self.apply_setting_values(values)
        if notice is not None:
            self.show_notice(notice)

    def poll_modified(self):
        """Runs every MODIFIED_POLL_MS for the life of the program."""
        try:
            self.check_modified()
            self.refresh_modified_outlines()
        finally:
            self.root.after(MODIFIED_POLL_MS, self.poll_modified)

    def check_modified(self):
        """With a keyframe selected, mark every setting whose value changed since
        the previous check as modified (see detect_modified_keys). Polling the
        settings, rather than hooking every control, catches every way a setting
        can be edited (entries, sliders, dropdowns, the obstacle and coloration
        editors) and costs well under a millisecond. Returns whether the set of
        modified settings changed."""
        if self.selected_keyframe is None or self.kf_applying:
            return False
        current = self.capture_all_settings(show_error=False)
        if current is None:                       # a limit field is mid-edit
            return False
        last = self.kf_last_seen if self.kf_last_seen is not None else current
        new = detect_modified_keys(last, current, self.kf_modified)
        self.kf_last_seen = current
        if new == self.kf_modified:
            return False
        self.kf_modified = new
        self.refresh_timeline_diffs()
        return True

    def widgets_for_setting(self, key):
        """The main control(s) currently showing setting `key`: [] when it has
        none visible now (an obstacle, coloration or framing other than the one being edited)."""
        def attr(name):
            obj = getattr(self, name, None)
            return getattr(obj, "widget", obj) if obj is not None else None

        if key.startswith("obst:"):
            _prefix, obst_id, field = key.split(":", 2)
            o = self.selected_obstacle
            if o is None or o.id != int(obst_id):
                return []
            widget = self.obstacle_widget(type(o), field)
            widget = getattr(widget, "widget", widget)
            return [widget] if widget is not None else []
        if key.startswith("coloration:"):
            _prefix, coloration_id, field = key.split(":", 2)
            v = self.selected_coloration_obj()
            if v is None or v.id != int(coloration_id):
                return []
            if field == "color_model":
                widget = attr("color_model_combo")
            else:
                blocks = getattr(self, "chan_blocks", [])
                widget = blocks[int(field[-1])] if int(field[-1]) < len(blocks) else None
            return [widget] if widget is not None else []
        if key.startswith("framing:"):
            _prefix, framing_id, field = key.split(":", 2)
            if self.selected_framing_id != int(framing_id):
                return []
            widget = attr(f"framing_{field}_entry")
            return [widget] if widget is not None else []
        widget = attr(SETTING_WIDGET_ATTRS[key])
        return [widget] if widget is not None else []

    def refresh_modified_outlines(self):
        """Show a blue outline around the control of every modified setting, and
        remove the outline from every other."""
        want = set()
        for key in self.kf_modified:
            want.update(self.widgets_for_setting(key))
        for widget in want:
            if widget not in self.outlines:
                self.outlines[widget] = ModifiedOutline(widget)
        for widget, outline in self.outlines.items():
            outline.show(widget in want)

    def rebuild_keyframe_list(self):
        for w in self.kf_list_frame.winfo_children():
            w.destroy()
        for idx, kf in enumerate(self.keyframes):
            selected = kf is self.selected_keyframe
            row = tk.Frame(self.kf_list_frame,
                          highlightthickness=SELECTED_OUTLINE_PX if selected else 0,
                          highlightbackground=SELECTED_OUTLINE,
                          highlightcolor=SELECTED_OUTLINE)
            row.grid(row=idx, column=0, sticky="ew", pady=2, padx=1)
            row.columnconfigure(0, weight=1)
            row.bind("<Button-1>", lambda _e, kf=kf: self.select_keyframe(kf))

            lbl = tk.Label(row, text=f"#{idx + 1}  frame {kf.frame_index}",
                          anchor="w", cursor="hand2", font=self.small_font)
            lbl.grid(row=0, column=0, sticky="ew", padx=4, pady=6)
            lbl.bind("<Button-1>", lambda _e, kf=kf: self.select_keyframe(kf))
        self.update_kf_detail_panel()

    def all_setting_entries(self):
        """[SettingEntry, ...] for every static setting plus every current
        obstacle's and coloration's fields, in display order. Each entry's `fallback` is the value an export would use
        if no keyframe defined the setting: the controls' value now (see setting_fallback)."""
        live = self.live_setting_values()
        entries = [SettingEntry(key=key, tab=tab, label=label, is_float=key in SETTING_FLOAT_KEYS,
                                fallback=live[key])
                   for key, tab, label in SETTING_INFO]
        for idx, o in enumerate(self.obstacles):
            tab = obstacle_tab_title(idx, o)
            for field, label, is_float in o.KEYFRAME_FIELDS:
                entries.append(SettingEntry(key=f"obst:{o.id}:{field}", tab=tab, label=label,
                                            is_float=is_float, fallback=getattr(o, field)))
        for idx, v in enumerate(self.colorations):
            tab = f"Coloration {idx}"
            for field, label, is_float in COLORATION_FIELD_INFO:
                entries.append(SettingEntry(key=f"coloration:{v.id}:{field}", tab=tab, label=label,
                                            is_float=is_float,
                                            fallback=coloration_field_value(v, field)))
        for idx, f in enumerate(self.framings):
            for field, label, is_float in FRAMING_FIELD_INFO:
                entries.append(SettingEntry(key=f"framing:{f.id}:{field}", tab=f"Framing {idx}", label=label,
                                            is_float=is_float, fallback=getattr(f, field)))
        return entries

    def all_setting_tabs(self):
        """Tab names in display order: the fixed ones, then one per
        obstacle, then one per coloration, then one per framing."""
        return (SETTING_TABS_ORDER + [obstacle_tab_title(i, o) for i, o in enumerate(self.obstacles)]
               + [f"Coloration {i}" for i in range(len(self.colorations))]
               + [f"Framing {i}" for i in range(len(self.framings))])

    def refresh_timeline_diffs(self):
        """Rebuild the settings list (see _rebuild_timeline_diffs), then refit its
        scrolling to the new content."""
        self._rebuild_timeline_diffs()
        self.fit_diff_scroll()

    def fit_diff_scroll(self):
        """Make the settings list's scroll area match its content right now, and keep
        the coloration inside it. Waiting for the list frame's <Configure> event is not
        enough: when a rebuild leaves the list shorter than the current scroll
        position (a filter or a collapsed section shrinking it), the canvas shows
        none of the frame, never resizes it, and so never sends that event - the
        list stayed scrolled past its end, blank."""
        canvas = getattr(self, "diff_canvas", None)
        if canvas is None:
            return
        self.diff_list_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.yview_moveto(canvas.yview()[0])      # clamps into the new scroll area

    def _rebuild_timeline_diffs(self):
        """Rebuild the settings list for the selected keyframe: every setting,
        grouped into collapsible sections (by the tab it lives on, or the obstacle
        or coloration it belongs to; all collapsed until expanded, see
        toggle_diff_section), each setting showing whether it's defined here,
        animated by surrounding keyframes, or neither."""
        if "diff_list_frame" not in self.built:
            return                                 # Timeline panel not built yet
        for w in self.diff_list_frame.winfo_children():
            w.destroy()
        kf = self.selected_keyframe

        if kf is None:
            self.diff_save_all_btn.state(["disabled"])
            self.diff_delete_all_btn.state(["disabled"])
            ttk.Label(self.diff_list_frame, text="(select a keyframe)",
                     foreground="#666", font=self.small_font).grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return
        current = self.capture_all_settings(show_error=False)
        if current is None:
            self.diff_save_all_btn.state(["disabled"])
            self.diff_delete_all_btn.state(["disabled"])
            ttk.Label(self.diff_list_frame, text="(fix the limit fields first)",
                     foreground="#666", font=self.small_font).grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return

        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame_index)
        idx_kf = sorted_kfs.index(kf)
        self.diff_save_all_btn.state(["!disabled"] if self.kf_modified else ["disabled"])
        self.diff_delete_all_btn.state(["!disabled"] if kf.data else ["disabled"])

        row = 0
        entries = self.all_setting_entries()
        modified = self.kf_modified
        presence = self.diff_presence_var.get()

        def passes(key):
            return setting_passes_presence_filter(key in kf.data, presence)

        for tab in self.all_setting_tabs():
            all_tab_entries = [e for e in entries if e.tab == tab]
            tab_entries = [e for e in all_tab_entries if passes(e.key)]
            if not tab_entries:
                continue                       # nothing here matches the filters
            expanded = tab in self.diff_expanded
            n_set = sum(1 for e in all_tab_entries if e.key in kf.data)
            n_modified = sum(1 for e in all_tab_entries if e.key in modified)
            header = ttk.Label(self.diff_list_frame, cursor="hand2",
                               text=diff_section_header_text(tab, expanded, n_set,
                                                             len(all_tab_entries),
                                                             n_modified),
                               foreground=MODIFIED_OUTLINE if n_modified else "",
                               font=self.small_bold_font)
            header.grid(row=row, column=0, sticky="w", padx=4, pady=(6, 1))
            header.bind("<Button-1>", lambda _e, t=tab: self.toggle_diff_section(t))
            row += 1
            if not expanded:
                continue
            for entry in tab_entries:
                self.add_setting_row(row, kf, entry.key, entry.label, entry.is_float,
                                     entry.fallback, current, sorted_kfs, idx_kf)
                row += 1
        if row == 0:
            ttk.Label(self.diff_list_frame, text="(no settings match the filters)",
                     foreground="#666", font=self.small_font).grid(
                row=0, column=0, sticky="w", padx=4, pady=4)

    def toggle_diff_section(self, tab):
        """Expand or collapse one section of the keyframe settings list. Sections
        are remembered by title, and all start collapsed."""
        if tab in self.diff_expanded:
            self.diff_expanded.discard(tab)
        else:
            self.diff_expanded.add(tab)
        self.refresh_timeline_diffs()

    def add_setting_row(self, row, kf, key, label, is_float, fallback, current, sorted_kfs,
                        idx_kf):
        """One setting's row in the keyframe settings list: its label, and Save /
        Delete buttons as they apply, outlined in blue while the setting is
        modified."""
        status = self.setting_status(kf, key, sorted_kfs, idx_kf)
        r = tk.Frame(self.diff_list_frame,
                     highlightthickness=MODIFIED_OUTLINE_PX if key in self.kf_modified else 0,
                     highlightbackground=MODIFIED_OUTLINE, highlightcolor=MODIFIED_OUTLINE)
        r.grid(row=row, column=0, sticky="ew", padx=4, pady=(1, 3))
        ttk.Label(r, text=label, font=self.small_font).grid(row=0, column=0, sticky="w")
        btns = ttk.Frame(r)
        btns.grid(row=1, column=0, sticky="w")

        save_text = f"Save {short_value_text(key, current[key])}"
        ttk.Button(btns, text=save_text,
                  command=lambda kf=kf, k=key: self.save_settings(
                      kf, [k])).pack(side="left", padx=2)

        if status == "present":
            ttk.Button(btns, text="Delete",
                      command=lambda kf=kf, k=key: self.delete_setting(
                          kf, k)).pack(side="left", padx=(2, 0))
        elif status == "animated":
            ttk.Label(btns, text="(animated by neighbors)",
                     foreground="#666", font=self.small_font).pack(
                side="left", padx=(4, 0))

    def save_modified_settings(self):
        """Save into the selected keyframe every setting marked modified."""
        kf = self.selected_keyframe
        if kf is None or not self.kf_modified:
            return
        self.save_settings(kf, sorted(self.kf_modified))

    def delete_all_diffs(self):
        """Delete every setting the selected keyframe defines, leaving it
        empty (every setting then falls back to being animated by neighbors,
        or undefined, just like a freshly emptied single setting would)."""
        kf = self.selected_keyframe
        if kf is None:
            return
        kf.data.clear()
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

    def update_kf_detail_panel(self):
        kf = self.selected_keyframe
        if kf is None:
            self.kf_detail_label_var.set("(none selected)")
            self.kf_frame_var.set("")
            self.kf_frame_entry.state(["disabled"])
            self.kf_split_btn.state(["disabled"])
            self.kf_delete_btn.state(["disabled"])
            return
        idx = self.keyframes.index(kf)
        self.kf_detail_label_var.set(f"Keyframe #{idx + 1}")
        self.kf_frame_var.set(str(kf.frame_index))
        self.kf_frame_entry.state(["!disabled"])
        self.kf_split_btn.state(["!disabled"])
        self.kf_delete_btn.state(["!disabled"])

    def on_kf_frame_commit(self, _event=None):
        kf = self.selected_keyframe
        if kf is None:
            return
        text = self.kf_frame_var.get()
        if text == "":
            self.kf_frame_var.set(str(kf.frame_index))  # every keyframe needs an index
            return
        kf.frame_index = int(text)
        self.reorder_keyframes()
        self.rebuild_keyframe_list()

    def reorder_keyframes(self):
        """Sort keyframes ascending by frame index (stable, so ties keep their
        relative order)."""
        self.keyframes.sort(key=lambda kf: kf.frame_index)

    def split_selected_keyframe(self):
        kf = self.selected_keyframe
        if kf is None:
            return
        new_kf = Keyframe(self._kf_counter, kf.frame_index, dict(kf.data))
        self._kf_counter += 1
        idx = self.keyframes.index(kf)
        self.keyframes.insert(idx + 1, new_kf)
        self.rebuild_keyframe_list()

    def delete_selected_keyframe(self):
        kf = self.selected_keyframe
        if kf is None:
            return
        self.keyframes.remove(kf)
        self.deselect_keyframe()
