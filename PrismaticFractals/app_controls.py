"""The Termination, Emissions and Mutations tabs: building the limit, confinement, emission, fuzzing and segment-length controls, validating and committing what is typed, reading the controls into a RaySettings, and the Start / Revert button states."""
import random
import re
import tkinter as tk
from tkinter import ttk
from .constants import (
    CONFINE_METHOD_NAMES,
    DEFAULT_CONFINE_METHOD, DEFAULT_CONFINE_THRESHOLD, DEFAULT_CONFINE_LOOKBACK, DEFAULT_DIST,
    DEFAULT_ESCAPE_RADIUS,
    DEFAULT_EMIT_MODE, DEFAULT_EMIT_TARGET_CX, DEFAULT_EMIT_TARGET_CY, DEFAULT_ENGINE, ENGINE_NAMES,
    OFFERED_ENGINES, ORBIT_ESCAPE_NAMES, ORBIT_MAPPING_JULIA, ORBIT_MAPPING_NAMES, ORBIT_Z0_NAMES, ORBIT_Z0_NOTE,
    DEFAULT_EMIT_TARGET_R, DEFAULT_EMIT_TARGET_THETA, DEFAULT_RNG_SEED, DEFAULT_ITER,
    DEFAULT_SEGMENT_MULT, EMIT_MODE_NAMES, EMIT_TARGET_FIELDS,
    EMIT_TARGET_LABELS, ERROR_RED, RIGHT_W, SEGMENT_MULT_MAX, SEGMENT_MULT_MIN
)
from .engines import effect_names, engine_description
from .figure import settings_of
from .kernels import CONFINE_BOX, EMIT_TOWARDS, DEFAULT_PIXEL_WORLD_WIDTH
from .parsing import parse_resolution_dimension, parse_target_coordinate
from .settings import (
    DEFAULT_SETTING_VALUES, NATURAL_TERMINATION_SETTING_SPECS, SETTING_SPECS, BuddhaBlurSettings, BuddhaSeedSettings, DEFAULT_FUZZ, DEFAULT_MANDELBROT_SETTINGS, EngineSettings,
    MandelbrotSettings, NaturalTerminationSettings, RaySettings, settings_to_dict
)
from .shapes import obstacles_signature
from .settings_model import format_value
from .widgets import FloatEntryHandle, ModelBinding


class ControlsMixin:
    def build_limits_controls(self, parent):
        """Three validated number-entry fields: iteration limit, distance limit and escape radius."""
        lim = ttk.LabelFrame(parent, text="Termination limits")
        lim.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 6))
        lim.columnconfigure(0, weight=1)
        iter_vcmd = (self.root.register(self.validate_digits), "%P")
        dist_vcmd = (self.root.register(self.validate_dist_digits), "%P")
        escape_vcmd = (self.root.register(self.validate_positive_float), "%P")

        self.iter_var = tk.StringVar(value=str(DEFAULT_ITER))
        self.dist_var = tk.StringVar(value=str(DEFAULT_DIST))
        self.escape_radius_var = tk.StringVar(value=str(DEFAULT_ESCAPE_RADIUS))
        rows = [("Iteration limit (bounces):", self.iter_var, "iter_entry", iter_vcmd),
                ("Distance limit (units, 0 = none):", self.dist_var, "dist_entry", dist_vcmd),
                ("Escape radius (units):", self.escape_radius_var, "escape_radius_entry",
                 escape_vcmd)]
        for r, (text, var, attr, vcmd) in enumerate(rows):
            ttk.Label(lim, text=text).grid(
                row=r, column=0, sticky="w", padx=(8, 6), pady=(6 if r == 0 else 2, 2))
            entry = ttk.Entry(lim, textvariable=var, width=8, justify="right",
                              validate="key", validatecommand=vcmd)
            entry.grid(row=r, column=1, padx=(0, 8), pady=(6 if r == 0 else 2, 2))
            entry.bind("<FocusOut>", self.on_limit_commit)
            entry.bind("<Return>", self.on_limit_commit)
            entry.bind("<KP_Enter>", self.on_limit_commit)
            setattr(self, attr, entry)

        self.limit_msg_var = tk.StringVar()
        ttk.Label(lim, textvariable=self.limit_msg_var, foreground=ERROR_RED,
                  wraplength=RIGHT_W - 70).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8)
        ttk.Label(lim, foreground="#666", wraplength=RIGHT_W - 70,
                  text="A ray also terminates when it is confined (see Confinement below), "
                       "or when it escapes (has nothing left to hit): an escaping ray "
                       "terminates where it is that escape radius from the origin (the "
                       "image's edge is 1 unit from the origin along each axis). "
                       "One unit is half the image width; a segment that would pass "
                       "the distance limit is cut off where it is reached. A "
                       "distance limit of 0 means no distance limit. Large limits "
                       "(or fuzzing) can take a very long time."
                  ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

    def build_segment_mult_controls(self, parent):
        """The two segment length multipliers (bounded entries). After (both engines): how much of each step's true segment length is
        actually travelled or recorded, once whatever the step needs has been calculated at its end. Before (the Mandelbrot engine
        only): how much of the step z -> f(z) is taken, right after it is calculated."""
        sm = ttk.LabelFrame(parent, text="Segment length")
        sm.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        sm.columnconfigure(1, weight=1)
        self.pre_seg_mult_scale = self.add_angle_entry(
            sm, 0, "Length multiplier, before the calculation (Mandelbrot only)", DEFAULT_SEGMENT_MULT, on_change=self.on_pre_seg_mult_changed,
            lo=SEGMENT_MULT_MIN, hi=SEGMENT_MULT_MAX, fmt="{:.2f}")
        self.add_model_entry(sm, 1, "Length multiplier, after the calculation", "seg_mult", on_commit=self.on_seg_mult_changed)
        ttk.Label(sm, foreground="#666", wraplength=RIGHT_W - 70,
                  text="After: shortens every ordinary bounce's segment to this fraction of "
                       "its true length, pulling the end back towards the start - but "
                       "the ray still continues on from there in the direction that "
                       "bounce calculated, so it visibly changes course in mid-air. "
                       "For the Mandelbrot engine it only moves the point added to the "
                       "path, along the step from the true z; the true z still goes on to the "
                       "next iteration. 1 means no effect. While active, the self-bounce loop area "
                       "metric always reads 0, since that polygon no longer reflects "
                       "the ray's own path.\n"
                       "Before (Mandelbrot engine): scales the step from z to z^p + c as soon as it is "
                       "calculated, so the shortened point is both the one added to the path and the "
                       "z of the next iteration (a damped orbit)."
                  ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 6))

    def build_confinement_controls(self, parent):
        """The Confinement section: how many of a ray's most recent path points
        are judged (the lookback), by which method, against what threshold (see
        CONFINE_BOX / CONFINE_PATH). The threshold accepts any number above
        zero, however small."""
        cf = ttk.LabelFrame(parent, text="Confinement")
        cf.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        cf.columnconfigure(0, weight=1)
        lookback_vcmd = (self.root.register(self.validate_lookback_digits), "%P")
        threshold_vcmd = (self.root.register(self.validate_positive_float), "%P")

        self.confine_lookback_var = tk.StringVar(value=str(DEFAULT_CONFINE_LOOKBACK))
        self.confine_method_var = tk.StringVar(
            value=CONFINE_METHOD_NAMES[DEFAULT_CONFINE_METHOD])
        self.confine_threshold_var = tk.StringVar(
            value=str(DEFAULT_CONFINE_THRESHOLD))
        self.confine_threshold_label_var = tk.StringVar()

        ttk.Label(cf, text="Lookback (path points):").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=(6, 2))
        self.confine_lookback_entry = ttk.Entry(
            cf, textvariable=self.confine_lookback_var, width=8, justify="right",
            validate="key", validatecommand=lookback_vcmd)
        self.confine_lookback_entry.grid(row=0, column=1, padx=(0, 8), pady=(6, 2))

        ttk.Label(cf, text="Detection method:").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=2)
        method_combo = ttk.Combobox(cf, textvariable=self.confine_method_var,
                                    values=CONFINE_METHOD_NAMES, state="readonly", width=12)
        method_combo.grid(row=1, column=1, padx=(0, 8), pady=2)
        method_combo.bind("<<ComboboxSelected>>", lambda _e: method_combo.selection_clear())
        self.confine_method_combo = method_combo

        ttk.Label(cf, textvariable=self.confine_threshold_label_var).grid(
            row=2, column=0, sticky="w", padx=(8, 6), pady=2)
        self.confine_threshold_entry = ttk.Entry(
            cf, textvariable=self.confine_threshold_var, width=14, justify="right",
            validate="key", validatecommand=threshold_vcmd)
        self.confine_threshold_entry.grid(row=2, column=1, padx=(0, 8), pady=2)
        for entry in (self.confine_lookback_entry, self.confine_threshold_entry):
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, self.on_limit_commit)

        self.confine_msg_var = tk.StringVar()
        ttk.Label(cf, textvariable=self.confine_msg_var, foreground=ERROR_RED,
                  wraplength=RIGHT_W - 70).grid(row=3, column=0, columnspan=2,
                                                sticky="w", padx=8)
        ttk.Label(cf, foreground="#666", wraplength=RIGHT_W - 70,
                  text="A ray terminates as confined when its last few path points (the "
                       "lookback) show it is going nowhere. Box: they fit in a box "
                       "smaller than the threshold in both directions. Path length: "
                       "the segments between them add up to less than the threshold. "
                       "The threshold is in world units (the image spans -1 to 1) and "
                       "can be any number above zero; one pixel of the 512-pixel "
                       f"preview is {DEFAULT_PIXEL_WORLD_WIDTH:g}, the default."
                  ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        self.update_confine_threshold_label()

    def build_emission_controls(self, parent):
        """The Ray emission section: whether rays are fired straight up or from
        their start pixel towards a target point (any coordinates at all), and
        the angle offset added to that direction (positive = more
        counterclockwise)."""
        em = ttk.LabelFrame(parent, text="Ray emission")
        em.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 6))
        em.columnconfigure(0, weight=1)
        coord_vcmd = (self.root.register(self.validate_signed_float), "%P")

        self.emit_mode_var = tk.StringVar(value=EMIT_MODE_NAMES[DEFAULT_EMIT_MODE])
        for field, default in zip(EMIT_TARGET_FIELDS,
                                  (DEFAULT_EMIT_TARGET_R, DEFAULT_EMIT_TARGET_THETA,
                                   DEFAULT_EMIT_TARGET_CX, DEFAULT_EMIT_TARGET_CY)):
            setattr(self, f"emit_target_{field}_var", tk.StringVar(value=str(default)))

        ttk.Label(em, text="Direction:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=(6, 2))
        mode_combo = ttk.Combobox(em, textvariable=self.emit_mode_var, values=EMIT_MODE_NAMES,
                                  state="readonly", width=14)
        mode_combo.grid(row=0, column=1, padx=(0, 8), pady=(6, 2))
        mode_combo.bind("<<ComboboxSelected>>", lambda _e: mode_combo.selection_clear())
        self.emit_mode_combo = mode_combo

        self.emit_target_entries = []
        for r, (field, text) in enumerate(zip(EMIT_TARGET_FIELDS, EMIT_TARGET_LABELS),
                                          start=1):
            ttk.Label(em, text="Target " + text[0].lower() + text[1:]).grid(
                row=r, column=0, sticky="w", padx=(8, 6), pady=2)
            entry = ttk.Entry(em, textvariable=getattr(self, f"emit_target_{field}_var"),
                              width=14, justify="right", validate="key",
                              validatecommand=coord_vcmd)
            entry.grid(row=r, column=1, padx=(0, 8), pady=2)
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, self.on_limit_commit)
            self.emit_target_entries.append(entry)
            setattr(self, f"emit_target_{field}_entry", entry)

        offset_frame = ttk.Frame(em)
        offset_frame.grid(row=5, column=0, columnspan=2, sticky="w")
        self.add_model_entry(offset_frame, 0, "Angle offset (\u00b0, + = counterclockwise)", "emit_offset", on_commit=self.on_emission_changed)

        self.emit_msg_var = tk.StringVar()
        ttk.Label(em, textvariable=self.emit_msg_var, foreground=ERROR_RED,
                  wraplength=RIGHT_W - 70).grid(row=6, column=0, columnspan=2,
                                                sticky="w", padx=8)
        ttk.Label(em, foreground="#666", wraplength=RIGHT_W - 70,
                  text="Each ray starts out pointing straight up, or from its own "
                       "start pixel towards the target point (which can be anywhere, "
                       "positioned like an obstacle: a distance r at angle \u03b8 from "
                       "the origin, plus an offset X and Y; a pixel exactly on it points "
                       "up). The angle offset is then "
                       "added, positive meaning more counterclockwise: with Up, the "
                       "default -45\u00b0 gives the up-and-to-the-right direction "
                       "the program has always used. Fuzzing's emission jitter is "
                       "added on top."
                  ).grid(row=7, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        self.update_emission_state()

    def update_emission_state(self):
        """The target point only matters when rays are fired towards it."""
        state = ["!disabled"] if self.model.value("emit_mode") == EMIT_TOWARDS else ["disabled"]
        for entry in self.emit_target_entries:
            entry.state(state)

    def on_emission_changed(self):
        """The angle offset entry committed a new value."""
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def update_confine_threshold_label(self, *_args):
        """The threshold means a box size or a path length, depending on the method."""
        box = self.confine_method_var.get() == CONFINE_METHOD_NAMES[CONFINE_BOX]
        self.confine_threshold_label_var.set(
            "Box size (world units):" if box else "Path length (world units):")

    def on_confine_method_changed(self, *_args):
        self.update_confine_threshold_label()

    def add_slider_row(self, parent, row, label, lo, hi, resolution, default, fmt,
                       on_change=None):
        """One compact row: label | slider | current value. Returns the slider."""
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(8, 8), pady=1)
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=resolution,
                         orient="horizontal", showvalue=False,
                         highlightthickness=0, width=14, sliderlength=18)
        scale.set(default)
        scale.grid(row=row, column=1, sticky="ew", pady=1)
        value_label = ttk.Label(parent, text=fmt.format(default), width=6,
                                anchor="e")
        value_label.grid(row=row, column=2, sticky="e", padx=(6, 8))

        callback = on_change or self.on_shape_param_changed

        def changed(value, value_label=value_label, fmt=fmt, callback=callback):
            value_label.configure(text=fmt.format(float(value)))
            callback()

        scale.configure(command=changed)           # only after the set() above
        return scale

    def add_slider_pair(self, parent, row, spec_a, spec_b, on_change=None):
        """Two compact sliders side by side on one row, to save space. Each
        spec is (label, lo, hi, resolution, default, fmt). Returns (scale_a,
        scale_b)."""
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=0, columnspan=3, sticky="ew")
        row_frame.columnconfigure((0, 1), weight=1)
        fa = ttk.Frame(row_frame)
        fa.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        fa.columnconfigure(1, weight=1)
        fb = ttk.Frame(row_frame)
        fb.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        fb.columnconfigure(1, weight=1)
        scale_a = self.add_slider_row(fa, 0, *spec_a, on_change=on_change)
        scale_b = self.add_slider_row(fb, 0, *spec_b, on_change=on_change)
        return scale_a, scale_b

    def add_angle_entry(self, parent, row, label, default, on_change=None,
                        lo=None, hi=None, fmt="{:.2f}", exact=False):
        """A label and a text entry on one snug row - a drop-in .get()/.set()
        replacement for a tk.Scale. With no bounds (the default), accepts any
        float, so many-revolution angles can be typed or animated; pass lo/hi
        to clamp into a range instead - used for settings that keep a
        slider-like range but as a text field (jitter/multiplier). Returns a
        FloatEntryHandle."""
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Label(row_frame, text=label).pack(side="left", padx=(8, 6))
        handle = FloatEntryHandle(None, fmt, default, lo=lo, hi=hi, exact=exact)
        var = tk.StringVar(value=handle.text(handle.last_valid))
        handle.var = var
        vcmd = (self.root.register(self.validate_float_free), "%P")
        entry = ttk.Entry(row_frame, textvariable=var, width=9, justify="right",
                          validate="key", validatecommand=vcmd)
        entry.pack(side="left")

        def commit(_event=None):
            handle.commit()
            if on_change is not None:
                on_change()

        entry.bind("<FocusOut>", commit)
        entry.bind("<Return>", commit)
        entry.bind("<KP_Enter>", commit)
        handle.widget = entry
        handle.row = row_frame                      # (the label and the entry: for greying both)
        return handle

    def add_model_entry(self, parent, row, label, key, on_commit=None):
        """A label and a text entry on one snug row (in the manner of add_angle_entry) for the setting `key` of self.model: what is
        typed is reported to the model, and committing it (Enter, focus out) puts the last valid value back and calls `on_commit`.
        The variable and entry are the attributes the setting's spec row names. Returns the row's frame."""
        spec = SETTING_SPECS[key]
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Label(row_frame, text=label).pack(side="left", padx=(8, 6))
        var = tk.StringVar(value=format_value(key, DEFAULT_SETTING_VALUES[key]))
        vcmd = (self.root.register(self.validate_float_free), "%P")
        entry = ttk.Entry(row_frame, textvariable=var, width=9, justify="right", validate="key", validatecommand=vcmd)
        entry.pack(side="left")

        def commit(_event=None):
            self.bindings[key].commit()
            if on_commit is not None:
                on_commit()

        for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
            entry.bind(event, commit)
        setattr(self, spec.control, var)
        setattr(self, spec.widget, entry)
        return row_frame

    def build_fuzzing_controls(self, parent):
        """Angle controls for ray emission, bounces off the boundary, and
        bounces off the ray itself (laid out in a 2x2 grid). Everything except
        the Enable checkbox is greyed out while fuzzing is off."""
        fz = ttk.LabelFrame(parent, text="Fuzzing")
        fz.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 6))
        fz.columnconfigure((0, 1), weight=1, uniform="fuzzcols")

        self.fuzz_enabled_var = tk.BooleanVar(value=DEFAULT_FUZZ.enabled)
        self.fuzz_enable_check = ttk.Checkbutton(
            fz, text="Enable fuzzing", variable=self.fuzz_enabled_var,
            command=self.on_fuzz_changed)
        self.fuzz_enable_check.grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 4))
        ttk.Label(fz, foreground="#666", wraplength=RIGHT_W - 70,
                  text="Perturbs the ray's angles. Emission jitter is added to the "
                       "ray's starting direction (set in the Emissions tab). Each "
                       "bounce group perturbs the "
                       "deflection angle at that kind of bounce: Offset always "
                       "added, Jitter a fresh random amount within \u00b1 the range "
                       "each time, Multiplier scales the angle before those are "
                       "added (\u00d71 = unchanged)."
                  ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        emit_group = ttk.LabelFrame(fz, text="Ray emission")
        emit_group.columnconfigure(1, weight=1)
        self.add_model_entry(emit_group, 0, "Jitter (\u00b1\u00b0)", "emit_jitter", on_commit=self.on_fuzz_changed)
        emit_group.grid(row=2, column=0, sticky="nsew", padx=(6, 3), pady=(0, 6))
        self.fuzz_ttk_widgets += [self.emit_jitter_entry]

        def bounce_group(title, prefix):
            g = ttk.LabelFrame(fz, text=title)
            g.columnconfigure(1, weight=1)
            for row, (field, label) in enumerate((("offset", "Offset (\u00b0)"), ("jitter", "Jitter (\u00b1\u00b0)"),
                                                  ("mult", "Multiplier"))):
                self.add_model_entry(g, row, label, f"{prefix}_{field}", on_commit=self.on_fuzz_changed)
                self.fuzz_ttk_widgets.append(getattr(self, f"{prefix}_{field}_entry"))
            return g

        bounce_group("Boundary bounces", "bnd").grid(
            row=2, column=1, sticky="nsew", padx=(3, 6), pady=(0, 6))
        bounce_group("Self bounces", "self").grid(
            row=3, column=0, sticky="nsew", padx=(6, 3), pady=(0, 6))

        rng_seed_frame = ttk.LabelFrame(fz, text="Random seed")
        rng_seed_frame.grid(row=3, column=1, sticky="nsew", padx=(3, 6), pady=(0, 6))
        rng_seed_frame.columnconfigure(0, weight=1)
        row = ttk.Frame(rng_seed_frame)
        row.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        self.rng_seed_var = tk.StringVar(value=str(DEFAULT_RNG_SEED))
        vcmd = (self.root.register(self.validate_rng_seed_digits), "%P")
        self.rng_seed_entry = ttk.Entry(row, textvariable=self.rng_seed_var, width=11,
                                    justify="right", validate="key",
                                    validatecommand=vcmd)
        self.rng_seed_entry.pack(side="left", padx=(0, 8))
        self.rng_seed_entry.bind("<FocusOut>", self.on_rng_seed_commit)
        self.rng_seed_entry.bind("<Return>", self.on_rng_seed_commit)
        self.rng_seed_entry.bind("<KP_Enter>", self.on_rng_seed_commit)
        randomize_btn = ttk.Button(row, text="Randomize", command=self.randomize_rng_seed)
        randomize_btn.pack(side="left")
        self.fuzz_ttk_widgets += [self.rng_seed_entry, randomize_btn]
        ttk.Label(rng_seed_frame, foreground="#666", wraplength=RIGHT_W // 2 - 60,
                  text="Same seed and settings give the same image every time."
                  ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 6))

    # ---- limits: number entries with validation ---------------------------- #
    @staticmethod
    def validate_digits(proposed):
        """Key-by-key check: only plain digits, at most four of them. (Whether
        the value is in range is judged by parse_iter_limit.)"""
        return re.fullmatch(r"[0-9]{0,4}", proposed) is not None

    @staticmethod
    def validate_dist_digits(proposed):
        """Key-by-key check for the distance limit: digits, at most four
        before the point, and an optional decimal point with at most two
        digits after it - so it accepts non-integer distances while typing,
        without ever accepting a character that could never be part of a
        valid one. (Whether the final value is in range is judged by
        parse_dist_limit.)"""
        return re.fullmatch(r"[0-9]{0,4}(\.[0-9]{0,2})?", proposed) is not None

    @staticmethod
    def validate_signed_float(proposed):
        """Key-by-key check for a coordinate: a partially typed number with an
        optional leading '-' and optional exponent, nothing that could never
        become a valid one. Whether it is finite is judged by
        parse_target_coordinate."""
        return re.fullmatch(r"-?[0-9]*\.?[0-9]*(?:[eE][-+]?[0-9]*)?", proposed) is not None

    @staticmethod
    def validate_lookback_digits(proposed):
        """Key-by-key check for the confinement lookback: up to three digits."""
        return re.fullmatch(r"[0-9]{0,3}", proposed) is not None

    @staticmethod
    def validate_positive_float(proposed):
        """Key-by-key check for the confinement threshold: a partially typed
        plain or scientific-notation number (no sign at the front - it must be
        above zero - but an exponent may be negative), nothing that could never
        become a valid one. Whether it is actually above zero is judged by
        parse_confine_threshold."""
        return re.fullmatch(r"[0-9]*\.?[0-9]*(?:[eE][-+]?[0-9]*)?", proposed) is not None

    @staticmethod
    def validate_rng_seed_digits(proposed):
        return re.fullmatch(r"[0-9]{0,9}", proposed) is not None

    @staticmethod
    def validate_bitrate_digits(proposed):
        return re.fullmatch(r"[0-9]{0,6}", proposed) is not None

    @staticmethod
    def validate_float_free(proposed):
        """Key-by-key check for an unrestricted float entry (angles): allows
        an optional leading '-' and a partially-typed number, but nothing
        that could never become a valid float."""
        return re.fullmatch(r"-?[0-9]*\.?[0-9]*", proposed) is not None

    def commit_resolution_var(self, var):
        """A resolution field (New image or video width/height) lost focus
        or had Enter pressed: rewrite its text to the rounded value it will
        actually be read as (see parse_resolution_dimension), so what's
        displayed always matches what an export will actually use. Leaves
        invalid text alone - the existing error messaging when the value is
        actually read covers that case."""
        rounded = parse_resolution_dimension(var.get())
        if rounded is not None:
            var.set(str(rounded))

    def bind_model_controls(self):
        """Connect the controls of every setting the model holds (the natural termination limits and confinement, the Buddha blur and
        the Buddha seeding) to it: what is typed goes to the model, what the program sets comes back through on_model_changed."""
        self.bindings = {}
        for key in self.model.keys():
            spec = SETTING_SPECS[key]
            entry = None if spec.kind in ("choice", "bool") else getattr(self, spec.widget)       # (a dropdown or checkbox has no entry to colour)
            self.bindings[key] = ModelBinding(self.model, key, getattr(self, spec.control), entry, lambda value, key=key: format_value(key, value))
        self.model.subscribe(self.on_model_changed)

    def on_model_changed(self, changes):
        """A setting of the model changed ({key: "submit" or "set"}): show it, flag what is invalid, and refresh what depends on it."""
        for key, origin in changes.items():
            if origin == "set":
                self.bindings[key].write_value()
            self.bindings[key].show_error()
        self.on_ray_model_changed(changes)
        if NATURAL_TERMINATION_SETTING_SPECS.keys().isdisjoint(changes):          # (the other settings react when committed)
            return
        errors = lambda keys: " ".join(self.model.error(key) for key in keys if self.model.error(key) is not None)
        self.limit_msg_var.set(errors(("iter", "dist", "escape_radius")))
        self.confine_msg_var.set(errors(("confine_lookback", "confine_threshold")))
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def on_ray_model_changed(self, changes):
        """An emission setting or the fuzzing seed changed in the model (typed, or set by the program): what the old traces did."""
        if "emit_mode" in changes:
            self.update_emission_state()
        if "seed" in changes:
            self.on_fuzz_changed()
        if any(key in changes for key in ("emit_mode", "emit_target_r", "emit_target_theta", "emit_target_cx", "emit_target_cy")):
            self.on_limit_var_changed()

    def on_limit_var_changed(self, *_args):
        """The emission settings changed (an entry was typed in, or settings were loaded): say what is wrong with the target, and refresh
        what depends on them."""
        self.emit_msg_var.set(next((self.model.error(f"emit_target_{field}") for field in EMIT_TARGET_FIELDS
                                    if self.model.error(f"emit_target_{field}") is not None), ""))
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def on_limit_commit(self, _event=None):
        """Enter pressed or focus left an entry: tidy the text, and put back the
        last valid value if the field holds something invalid."""
        for key, spec in NATURAL_TERMINATION_SETTING_SPECS.items():
            if spec.kind != "choice":
                self.bindings[key].commit()
        for field in EMIT_TARGET_FIELDS:
            self.bindings[f"emit_target_{field}"].commit()

    def on_pre_seg_mult_changed(self):
        """The pre-calculation segment length multiplier entry committed a new value."""
        self.invalidate_preview()
        self.on_orbit_changed()

    def on_seg_mult_changed(self):
        """The segment length multiplier entry committed a new value (already
        parsed and clamped into range by its FloatEntryHandle)."""
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def on_rng_seed_commit(self, _event=None):
        self.bindings["seed"].commit()

    def randomize_rng_seed(self):
        self.rng_seed_var.set(str(random.randint(0, 999999999)))
        self.on_rng_seed_commit()

    # ---- change handling (obstacle shape parameters and the like) -------------- #
    def on_shape_param_changed(self, _value=None):
        self.invalidate_preview()
        self.schedule_outline_refresh()           # live outlines before first Start
        self.update_pending_outline()             # dashed outlines afterwards
        self.update_buttons()
        self.refresh_timeline_diffs()

    # ---- fuzzing controls ---------------------------------------------------- #
    def update_fuzz_enabled_state(self):
        """Grey out every Fuzzing-tab control except the Enable checkbox itself
        whenever fuzzing is disabled."""
        engine_off = self.current_engine_description().ignores("fuzz_on")
        self.fuzz_enable_check.state(["disabled"] if engine_off else ["!disabled"])
        enabled = self.model.value("fuzz_on") and not engine_off
        tk_state = tk.NORMAL if enabled else tk.DISABLED
        for w in self.fuzz_tk_widgets:
            w.configure(state=tk_state)
        ttk_state = ["!disabled"] if enabled else ["disabled"]
        for w in self.fuzz_ttk_widgets:
            w.state(ttk_state)

    def on_fuzz_changed(self, *_args):
        self.update_fuzz_enabled_state()
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def schedule_outline_refresh(self):
        """Coalesce rapid slider events into one outline redraw."""
        if self.outline_job is None:
            self.outline_job = self.root.after_idle(self.refresh_outline)

    def refresh_outline(self):
        self.outline_job = None
        if self.generated is None:                # nothing generated yet: live preview
            self.draw()

    # ---- settings, Start and Revert ---------------------------------------- #
    def current_ray_settings(self):
        """The ray-bouncing engine's control values (segment length, fuzzing, emission), or None
        if an emission target entry is invalid."""
        return self.model.record(RaySettings)

    def current_natural_termination_settings(self):
        """The termination limits' and confinement controls' values (a NaturalTerminationSettings), or
        None if one of their entries is invalid."""
        return self.model.record(NaturalTerminationSettings)

    def current_generation_settings(self):
        """(RaySettings, NaturalTerminationSettings) - everything that, with the obstacles, determines
        the generated image - or None if any entry of either is invalid."""
        ray = self.current_ray_settings()
        natural_termination = self.current_natural_termination_settings()
        if ray is None or natural_termination is None:
            return None
        return ray, natural_termination

    def build_engine_controls(self, parent):
        """The Simulation tab: which path engine makes the figure. A choice can never be invalid, and a different engine is
        a different figure, so changing it enables Start like any figure-defining setting. Only the engines in
        OFFERED_ENGINES are listed."""
        frame = ttk.LabelFrame(parent, text="Path engine")
        frame.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Label(frame, text="Engine:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
        self.engine_var = tk.StringVar(value=ENGINE_NAMES[DEFAULT_ENGINE])
        self.engine_combo = ttk.Combobox(frame, textvariable=self.engine_var, state="readonly", width=18,
                                         values=[ENGINE_NAMES[e] for e in OFFERED_ENGINES])
        self.engine_combo.grid(row=0, column=1, sticky="w", padx=(0, 8), pady=8)
        self.engine_combo.bind("<<ComboboxSelected>>", lambda _e: self.on_engine_changed())
        self.build_mandelbrot_controls(parent)

    def build_mandelbrot_controls(self, parent):
        """The Mandelbrot engine's own settings, under the engine dropdown (D6, D14): the exponent, where the orbit starts (z_0),
        whether the pixel is c (Mandelbrot) or z_0 (Julia), the Julia constant and how an escaping orbit ends. They exist whichever
        engine is selected (they are saved and animated all the same) and are greyed out while they are of no use: all of them
        unless the engine is Mandelbrot, the Julia constant unless the pixel is z_0, z_0 when it is. Figure-defining."""
        frame = ttk.LabelFrame(parent, text="Mandelbrot engine")
        frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.orbit_labels, self.orbit_widgets = {}, {}
        float_vcmd = (self.root.register(self.validate_signed_float), "%P")
        self.last_orbit = {"exponent": DEFAULT_MANDELBROT_SETTINGS.exponent, "julia_re": DEFAULT_MANDELBROT_SETTINGS.julia_re,
                           "julia_im": DEFAULT_MANDELBROT_SETTINGS.julia_im}
        rows = (("exponent", "Exponent:", "entry"), ("z0", "Starting value z_0:", ORBIT_Z0_NAMES),
                ("mapping", "Pixel maps to:", ORBIT_MAPPING_NAMES), ("julia_re", "Julia constant, real part:", "entry"),
                ("julia_im", "Julia constant, imaginary part:", "entry"), ("escape", "An escaping orbit ends:", ORBIT_ESCAPE_NAMES))
        for row, (field, label, kind) in enumerate(rows):
            name = ttk.Label(frame, text=label)
            name.grid(row=row, column=0, sticky="w", padx=(8, 6), pady=3)
            if kind == "entry":
                var = tk.StringVar(value=str(getattr(DEFAULT_MANDELBROT_SETTINGS, field)))
                widget = ttk.Entry(frame, textvariable=var, width=14, justify="right", validate="key", validatecommand=float_vcmd)
                for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                    widget.bind(event, self.on_orbit_commit)
                setattr(self, f"orbit_{field}_entry", widget)
            else:
                var = tk.StringVar(value=kind[getattr(DEFAULT_MANDELBROT_SETTINGS, field)])
                widget = ttk.Combobox(frame, textvariable=var, state="readonly", width=26, values=kind)
                widget.bind("<<ComboboxSelected>>", lambda _e, w=widget: (w.selection_clear(), self.on_orbit_changed()))
                setattr(self, f"orbit_{field}_combo", widget)
            setattr(self, f"orbit_{field}_var", var)
            widget.grid(row=row, column=1, sticky="w", padx=(0, 8), pady=3)
            self.orbit_labels[field], self.orbit_widgets[field] = name, widget
        self.orbit_note = ttk.Label(frame, text=ORBIT_Z0_NOTE, foreground="#666", wraplength=RIGHT_W - 70)
        self.orbit_note.grid(row=len(rows), column=0, columnspan=2, sticky="w", padx=8, pady=(2, 6))
        self.update_orbit_state()

    def update_orbit_state(self):
        """Grey out the Mandelbrot engine's controls that are of no use now (nothing is hidden, D4)."""
        mandelbrot_on = self.current_engine_description().uses_orbit_settings
        julia = self.orbit_mapping_var.get() == ORBIT_MAPPING_NAMES[ORBIT_MAPPING_JULIA]
        useless = {"julia_re": not julia, "julia_im": not julia, "z0": julia}
        for field, widget in self.orbit_widgets.items():
            off = not mandelbrot_on or useless.get(field, False)
            for w in (widget, self.orbit_labels[field]):
                w.state(["disabled"] if off else ["!disabled"])
        self.orbit_note.state(["disabled"] if not mandelbrot_on or julia else ["!disabled"])
        self.set_tree_enabled(self.pre_seg_mult_scale.row, mandelbrot_on)          # (the Segment length section's own row)

    def on_orbit_commit(self, _event=None):
        """Enter pressed or focus left one of the exponent / Julia constant entries: an invalid one goes back to its last valid
        value."""
        for field in self.last_orbit:
            var = getattr(self, f"orbit_{field}_var")
            value = parse_target_coordinate(var.get())
            var.set(str(self.last_orbit[field] if value is None else value))
        self.on_orbit_changed()

    def on_orbit_changed(self):
        """A Mandelbrot engine setting changed (by hand, or loaded from a keyframe or a project): remember the entries' valid
        values and refresh what depends on it."""
        for field in self.last_orbit:
            value = parse_target_coordinate(getattr(self, f"orbit_{field}_var").get())
            if value is not None:
                self.last_orbit[field] = value
        self.update_orbit_state()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def on_engine_changed(self):
        """The engine dropdown was used: switch to the engine it shows (see switch_to_selected_engine)."""
        self.engine_combo.selection_clear()
        self.switch_to_selected_engine()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def current_engine_description(self):
        """What the engine selected in the dropdown offers and accepts (an EngineDescription)."""
        if not hasattr(self, "engine_var"):                  # (the Simulation tab is built after the Colorations tab)
            return engine_description(DEFAULT_ENGINE)
        return engine_description(self.current_engine_settings().engine)

    def set_tree_enabled(self, widget, enabled):
        """Enable or grey out `widget` and everything inside it (ttk widgets by their state, classic Tk ones by their `state` option)."""
        for child in widget.winfo_children():
            if hasattr(child, "instate"):                  # (a ttk widget: classic ones have no `instate`)
                child.state(["!disabled"] if enabled else ["disabled"])
            else:
                try:
                    child.configure(state=tk.NORMAL if enabled else tk.DISABLED)
                except tk.TclError:
                    pass                                # (a frame or canvas has none)
            self.set_tree_enabled(child, enabled)

    def refresh_enabled_states(self):
        """THE resolver of whether a control is enabled (D4, D5, D13): the engine in use makes use of it AND its own rule allows it.
        The tabs the engine ignores (Emissions for Mandelbrot) are greyed out whole, nothing is hidden; the settings it ignores in a
        tab it otherwise uses (fuzzing for Mandelbrot) and the controls of the other engine (the Mandelbrot box and the pre-calculation
        length multiplier under the ray engine) by their own rules; for the others
        the controls' own rules decide (fuzzing's settings while it is off, the Julia constant unless the pixel is z_0, ...). A greyed
        setting is still saved, loaded and animated."""
        for tab, frame in self.engine_tab_frames.items():
            self.set_tree_enabled(frame, not self.engine_ignores_tab(tab))
        self.update_fuzz_enabled_state()
        self.update_orbit_state()

    def engine_ignores_tab(self, tab):
        """Whether the engine selected makes no use of what is in the tab named `tab`."""
        return tab in self.current_engine_description().ignored_tabs

    def apply_engine_offers(self):
        """Show what the selected engine offers: the obstacle effect dropdowns list only its effects, and the Colorations tab's
        metric and point dropdowns and Buddha checkboxes only its choices (a checkbox it does not offer is greyed out). The
        values already chosen are not changed here: that is what correcting them is for (engines.correct_settings)."""
        self.engine_applied = self.current_engine_settings().engine
        description = self.current_engine_description()
        self.refresh_enabled_states()
        names = effect_names(description)
        for combo in (self.obs_in_effect_combo, self.obs_out_effect_combo):
            combo.configure(values=names)
        self.refresh_color_controls_for_selected_coloration()

    def current_mandelbrot_settings(self):
        """The Mandelbrot engine controls' values (a MandelbrotSettings; never invalid: an entry holding nonsense for the moment
        counts as its last valid value)."""
        number = {}
        for field, last in self.last_orbit.items():
            value = parse_target_coordinate(getattr(self, f"orbit_{field}_var").get())
            number[field] = last if value is None else value
        return MandelbrotSettings(
            z0=ORBIT_Z0_NAMES.index(self.orbit_z0_var.get()), mapping=ORBIT_MAPPING_NAMES.index(self.orbit_mapping_var.get()),
            escape=ORBIT_ESCAPE_NAMES.index(self.orbit_escape_var.get()), pre_seg_mult=self.pre_seg_mult_scale.get(), **number)

    def mandelbrot_defines_figure(self, mandelbrot):
        """Whether `mandelbrot` (a MandelbrotSettings) matters to a figure made with the engine selected now: only when that is the
        Mandelbrot engine."""
        return engine_description(self.current_engine_settings().engine).uses_orbit_settings and self.current_mandelbrot_settings() != mandelbrot

    def current_engine_settings(self):
        """The Simulation tab's engine (an EngineSettings; never invalid)."""
        return EngineSettings(engine=ENGINE_NAMES.index(self.engine_var.get()))

    def current_blur_settings(self):
        """The Buddha blur controls' values (a BuddhaBlurSettings; a slider and a list of
        choices can never hold an invalid value, so unlike current_generation_settings this is
        never None). Always what Buddha channels are colored with: the blur is not baked
        into a generated image, so there is no "generated" blur to remember."""
        return self.model.record(BuddhaBlurSettings)

    def current_buddha_seed_settings(self):
        """The Buddha seeding controls' values (a BuddhaSeedSettings). Like the blur it is never None: a zone entry
        that holds something invalid for the moment counts as its last valid value. Not part of what a generated
        image is made of, so there is no "generated" seeding to remember either."""
        return self.model.record(BuddhaSeedSettings)

    def update_buttons(self):
        """Start: enabled if the entries are valid and either nothing has been
        generated yet or the settings (or obstacles) differ from those the
        image was started with. Revert: enabled whenever any simulation
        control differs from the baseline (obstacles are scene furniture, not
        part of Revert). Export/Export As: enabled once a figure has actually
        finished generating (started, not currently running, all rows done) -
        exporting before that would just export the blank background, which
        is never useful, so the button explains what to do instead."""
        cur = self.current_generation_settings()
        start_ok = (self.ready and cur is not None
                    and (self.generated is None
                         or cur != settings_of(self.generated)
                         or self.current_engine_settings() != self.generated.engine
                         or self.mandelbrot_defines_figure(self.generated.mandelbrot)
                         or obstacles_signature(self.obstacles) != self.generated.obstacles_signature))
        revert_ok = (cur is None or cur != (self.baseline, self.baseline_natural_termination)
                     or self.current_engine_settings() != self.baseline_engine
                     or self.current_mandelbrot_settings() != self.baseline_mandelbrot)
        self.start_btn.state(["!disabled"] if start_ok else ["disabled"])
        self.revert_btn.state(["!disabled"] if revert_ok else ["disabled"])
        if hasattr(self, "save_btn"):
            has_image = (self.generated is not None and not self.running
                         and self.rows_done >= self.img_h)
            self.save_btn.state(["!disabled"] if has_image else ["disabled"])
            self.save_as_btn.state(["!disabled"] if has_image else ["disabled"])
            tip = ("" if has_image else
                  "Press Start and wait for generation to finish before exporting.")
            self.save_btn_tooltip.update(tip)
            self.save_as_btn_tooltip.update(tip)

    def revert_settings(self):
        """Put every simulation control back to the baseline: its value when the
        program started, or when the last generation was started."""
        for key, value in settings_to_dict(self.baseline, self.baseline_natural_termination, self.baseline_engine,
                                   self.baseline_mandelbrot).items():
            self.set_setting_control(key, value)
        self.on_limit_var_changed()
        self.on_shape_param_changed()
        self.on_fuzz_changed()
