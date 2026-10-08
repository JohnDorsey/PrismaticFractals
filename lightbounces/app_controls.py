"""The Termination, Emissions and Mutations tabs: building the limit, confinement, emission, fuzzing and segment-length controls, validating and committing what is typed, reading the controls into a RaySettings, and the Start / Revert button states."""
import random
import re
import tkinter as tk
from tkinter import ttk
from .constants import (
    BUDDHA_BLUR_LABELS, CONFINE_METHOD_NAMES, CONFINE_LOOKBACK_MAX, CONFINE_LOOKBACK_MIN,
    DEFAULT_CONFINE_METHOD, DEFAULT_CONFINE_THRESHOLD, DEFAULT_CONFINE_LOOKBACK, DEFAULT_DIST,
    DEFAULT_ESCAPE_RADIUS,
    DEFAULT_EMIT_MODE, DEFAULT_EMIT_OFFSET, DEFAULT_EMIT_TARGET_CX, DEFAULT_EMIT_TARGET_CY,
    DEFAULT_EMIT_TARGET_R, DEFAULT_EMIT_TARGET_THETA, DEFAULT_RNG_SEED, DEFAULT_ITER,
    DEFAULT_SEGMENT_MULT, DIST_LIMIT_MAX, DIST_LIMIT_MIN, EMIT_MODE_NAMES, EMIT_TARGET_FIELDS,
    EMIT_TARGET_LABELS, ERROR_RED, FUZZ_JITTER_MAX, FUZZ_JITTER_MIN, FUZZ_MULT_MAX,
    FUZZ_MULT_MIN, ITER_LIMIT_MAX, ITER_LIMIT_MIN, RIGHT_W, SEGMENT_MULT_MAX, SEGMENT_MULT_MIN
)
from .kernels import CONFINE_BOX, EMIT_TOWARDS, PIXEL
from .parsing import (
    parse_confine_threshold, parse_confine_lookback, parse_dist_limit, parse_escape_radius,
    parse_iter_limit,
    parse_resolution_dimension, parse_rng_seed, parse_target_coordinate
)
from .settings import (
    BuddhaBlurSettings, BuddhaSeedSettings, DEFAULT_FUZZ, FuzzParams, NaturalStopSettings, RaySettings,
    settings_to_dict
)
from .shapes import obstacles_signature
from .widgets import FloatEntryHandle


class ControlsMixin:
    def build_limits_controls(self, parent):
        """Three validated number-entry fields: iteration limit, distance limit and escape radius."""
        lim = ttk.LabelFrame(parent, text="Stopping limits")
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
                  text="A ray also stops when it is confined (see Confinement below), "
                       "or when it escapes (has nothing left to hit): an escaping ray "
                       "stops where it is that escape radius from the origin (the "
                       "image's edge is 1 unit from the origin along each axis). "
                       "One unit is half the image width; a segment that would pass "
                       "the distance limit is cut off where it is reached. A "
                       "distance limit of 0 means no distance limit. Large limits "
                       "(or fuzzing) can take a very long time."
                  ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

    def build_segment_mult_controls(self, parent):
        """A single bounded entry: how much of each bounce's true segment
        length is actually travelled before continuing in the newly-bounced
        direction."""
        sm = ttk.LabelFrame(parent, text="Segment length")
        sm.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        sm.columnconfigure(1, weight=1)
        self.seg_mult_scale = self.add_angle_entry(
            sm, 0, "Length multiplier", DEFAULT_SEGMENT_MULT, on_change=self.on_seg_mult_changed,
            lo=SEGMENT_MULT_MIN, hi=SEGMENT_MULT_MAX, fmt="{:.2f}")
        ttk.Label(sm, foreground="#666", wraplength=RIGHT_W - 70,
                  text="Shortens every ordinary bounce's segment to this fraction of "
                       "its true length, pulling the end back towards the start - but "
                       "the ray still continues on from there in the direction that "
                       "bounce calculated, so it visibly changes course in mid-air. "
                       "1 means no effect. While active, the self-bounce loop area "
                       "metric always reads 0, since that polygon no longer reflects "
                       "the ray's own path."
                  ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 6))

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
                  text="A ray stops as confined when its last few path points (the "
                       "lookback) show it is going nowhere. Box: they fit in a box "
                       "smaller than the threshold in both directions. Path length: "
                       "the segments between them add up to less than the threshold. "
                       "The threshold is in world units (the image spans -1 to 1) and "
                       "can be any number above zero; one pixel of the 512-pixel "
                       f"preview is {PIXEL:g}, the default."
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
        self.emit_offset_scale = self.add_angle_entry(
            offset_frame, 0, "Angle offset (\u00b0, + = counterclockwise)", DEFAULT_EMIT_OFFSET,
            on_change=self.on_emission_changed, fmt="{:.2f}")

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
        state = ["!disabled"] if self.emit_mode_var.get() == EMIT_MODE_NAMES[EMIT_TOWARDS] \
            else ["disabled"]
        for entry in self.emit_target_entries:
            entry.state(state)

    def on_emit_mode_changed(self, *_args):
        self.update_emission_state()
        self.on_limit_var_changed()

    def on_emission_changed(self):
        """The angle offset entry committed a new value (already parsed and
        clamped into range by its FloatEntryHandle)."""
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
        self.on_limit_var_changed()

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
        return handle

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
        self.emit_jitter_scale = self.add_angle_entry(
            emit_group, 0, "Jitter (\u00b1\u00b0)", 0.0, on_change=self.on_fuzz_changed,
            lo=FUZZ_JITTER_MIN, hi=FUZZ_JITTER_MAX, fmt="{:.1f}")
        emit_group.grid(row=2, column=0, sticky="nsew", padx=(6, 3), pady=(0, 6))
        self.fuzz_ttk_widgets += [self.emit_jitter_scale.widget]

        def bounce_group(title, prefix):
            g = ttk.LabelFrame(fz, text=title)
            g.columnconfigure(1, weight=1)
            off = self.add_angle_entry(g, 0, "Offset (\u00b0)", 0.0,
                                       on_change=self.on_fuzz_changed)
            jit = self.add_angle_entry(g, 1, "Jitter (\u00b1\u00b0)", 0.0,
                                       on_change=self.on_fuzz_changed,
                                       lo=FUZZ_JITTER_MIN, hi=FUZZ_JITTER_MAX, fmt="{:.1f}")
            mult = self.add_angle_entry(g, 2, "Multiplier", 1.0,
                                        on_change=self.on_fuzz_changed,
                                        lo=FUZZ_MULT_MIN, hi=FUZZ_MULT_MAX, fmt="{:.2f}")
            setattr(self, f"{prefix}_offset_scale", off)
            setattr(self, f"{prefix}_jitter_scale", jit)
            setattr(self, f"{prefix}_mult_scale", mult)
            self.fuzz_ttk_widgets += [off.widget, jit.widget, mult.widget]
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

    def on_limit_var_changed(self, *_args):
        """Either entry's text changed: flag bad values, track the last good ones."""
        it = parse_iter_limit(self.iter_var.get())
        ds = parse_dist_limit(self.dist_var.get())
        er = parse_escape_radius(self.escape_radius_var.get())
        cw = parse_confine_lookback(self.confine_lookback_var.get())
        ct = parse_confine_threshold(self.confine_threshold_var.get())
        if it is not None:
            self.last_iter = it
        if ds is not None:
            self.last_dist = ds
        if er is not None:
            self.last_escape_radius = er
        if cw is not None:
            self.last_confine_lookback = cw
        if ct is not None:
            self.last_confine_threshold = ct
        self.iter_entry.configure(style="TEntry" if it is not None else "Invalid.TEntry")
        self.dist_entry.configure(style="TEntry" if ds is not None else "Invalid.TEntry")
        self.escape_radius_entry.configure(style="TEntry" if er is not None else "Invalid.TEntry")
        self.confine_lookback_entry.configure(
            style="TEntry" if cw is not None else "Invalid.TEntry")
        self.confine_threshold_entry.configure(
            style="TEntry" if ct is not None else "Invalid.TEntry")
        emit_msgs = []
        for field in EMIT_TARGET_FIELDS:
            value = parse_target_coordinate(getattr(self, f"emit_target_{field}_var").get())
            if value is not None:
                setattr(self, f"last_emit_target_{field}", value)
            getattr(self, f"emit_target_{field}_entry").configure(
                style="TEntry" if value is not None else "Invalid.TEntry")
            if value is None and not emit_msgs:
                emit_msgs.append("Target position values must be numbers (any size, "
                                 "positive or negative, e.g. -0.5 or 3e8).")
        self.emit_msg_var.set(" ".join(emit_msgs))
        confine_msgs = []
        if cw is None:
            confine_msgs.append(f"Lookback must be a whole number from "
                                f"{CONFINE_LOOKBACK_MIN} to {CONFINE_LOOKBACK_MAX}.")
        if ct is None:
            confine_msgs.append("Threshold must be a number greater than 0 "
                                "(any size, e.g. 0.004 or 1e-9).")
        self.confine_msg_var.set(" ".join(confine_msgs))
        msgs = []
        if it is None:
            msgs.append(f"Iteration limit must be a whole number from "
                        f"{ITER_LIMIT_MIN} to {ITER_LIMIT_MAX}.")
        if ds is None:
            msgs.append(f"Distance limit must be a whole number from "
                        f"{DIST_LIMIT_MIN} to {DIST_LIMIT_MAX} (0 = no limit).")
        if er is None:
            msgs.append("Escape radius must be a number greater than 0 (e.g. 2 or 1.5e3).")
        self.limit_msg_var.set(" ".join(msgs))
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def on_limit_commit(self, _event=None):
        """Enter pressed or focus left an entry: tidy the text, and put back the
        last valid value if the field holds something invalid."""
        for var, last, parser in (
                (self.iter_var, self.last_iter, parse_iter_limit),
                (self.dist_var, self.last_dist, parse_dist_limit),
                (self.escape_radius_var, self.last_escape_radius, parse_escape_radius),
                (self.confine_lookback_var, self.last_confine_lookback, parse_confine_lookback),
                (self.confine_threshold_var, self.last_confine_threshold,
                 parse_confine_threshold),
                *[(getattr(self, f"emit_target_{field}_var"),
                   getattr(self, f"last_emit_target_{field}"), parse_target_coordinate)
                  for field in EMIT_TARGET_FIELDS]):
            value = parser(var.get())
            var.set(str(last if value is None else value))

    def on_seg_mult_changed(self):
        """The segment length multiplier entry committed a new value (already
        parsed and clamped into range by its FloatEntryHandle)."""
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_timeline_diffs()

    def on_rng_seed_var_changed(self, *_args):
        rng_seed = parse_rng_seed(self.rng_seed_var.get())
        if rng_seed is not None:
            self.last_rng_seed = rng_seed
        self.rng_seed_entry.configure(style="TEntry" if rng_seed is not None else "Invalid.TEntry")
        self.on_fuzz_changed()

    def on_rng_seed_commit(self, _event=None):
        rng_seed = parse_rng_seed(self.rng_seed_var.get())
        self.rng_seed_var.set(str(self.last_rng_seed if rng_seed is None else rng_seed))

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
    def read_fuzz_controls(self):
        """FuzzParams from the current widget values."""
        rng_seed = parse_rng_seed(self.rng_seed_var.get())
        return FuzzParams(
            enabled=bool(self.fuzz_enabled_var.get()),
            emit_jitter=round(float(self.emit_jitter_scale.get()), 2),
            bnd_offset=round(float(self.bnd_offset_scale.get()), 2),
            bnd_jitter=round(float(self.bnd_jitter_scale.get()), 2),
            bnd_mult=round(float(self.bnd_mult_scale.get()), 3),
            self_offset=round(float(self.self_offset_scale.get()), 2),
            self_jitter=round(float(self.self_jitter_scale.get()), 2),
            self_mult=round(float(self.self_mult_scale.get()), 3),
            rng_seed=self.last_rng_seed if rng_seed is None else rng_seed)

    def update_fuzz_enabled_state(self):
        """Grey out every Fuzzing-tab control except the Enable checkbox itself
        whenever fuzzing is disabled."""
        enabled = self.fuzz_enabled_var.get()
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
        if self.generated_settings is None:       # nothing generated yet: live preview
            self.draw()

    # ---- settings, Start and Revert ---------------------------------------- #
    def current_ray_settings(self):
        """The ray-bouncing engine's control values (segment length, fuzzing, emission), or None
        if an emission target entry is invalid."""
        target = [parse_target_coordinate(getattr(self, f"emit_target_{field}_var").get())
                  for field in EMIT_TARGET_FIELDS]
        if None in target:
            return None
        return RaySettings(seg_mult=self.seg_mult_scale.get(), fuzz=self.read_fuzz_controls(),
                           emit_mode=EMIT_MODE_NAMES.index(self.emit_mode_var.get()),
                           emit_target_r=target[0], emit_target_theta=target[1],
                           emit_target_cx=target[2], emit_target_cy=target[3],
                           emit_offset=round(float(self.emit_offset_scale.get()), 2))

    def current_natural_stop_settings(self):
        """The stopping limits' and confinement controls' values (a NaturalStopSettings), or
        None if one of their entries is invalid."""
        it = parse_iter_limit(self.iter_var.get())
        ds = parse_dist_limit(self.dist_var.get())
        er = parse_escape_radius(self.escape_radius_var.get())
        cw = parse_confine_lookback(self.confine_lookback_var.get())
        ct = parse_confine_threshold(self.confine_threshold_var.get())
        if it is None or ds is None or er is None or cw is None or ct is None:
            return None
        return NaturalStopSettings(
            max_iter=it, max_dist=ds, escape_radius=er, confine_lookback=cw,
            confine_method=CONFINE_METHOD_NAMES.index(self.confine_method_var.get()),
            confine_threshold=ct)

    def current_generation_settings(self):
        """(RaySettings, NaturalStopSettings) - everything that, with the obstacles, determines
        the generated image - or None if any entry of either is invalid."""
        ray = self.current_ray_settings()
        natural_stop = self.current_natural_stop_settings()
        if ray is None or natural_stop is None:
            return None
        return ray, natural_stop

    def current_blur_settings(self):
        """The Buddha blur controls' values (a BuddhaBlurSettings; a slider and a list of
        choices can never hold an invalid value, so unlike current_generation_settings this is
        never None). Always what Buddha channels are colored with: the blur is not baked
        into a generated image, so there is no "generated" blur to remember."""
        return BuddhaBlurSettings(radius=self.buddha_blur_radius_scale.get(),
                                  type=BUDDHA_BLUR_LABELS.index(self.buddha_blur_type_var.get()))

    def current_buddha_seed_settings(self):
        """The Buddha seeding controls' values (a BuddhaSeedSettings). Like the blur it is never None: a zone entry
        that holds something invalid for the moment counts as its last valid value. Not part of what a generated
        image is made of, so there is no "generated" seeding to remember either."""
        zone = {}
        for field, last in self.last_buddha_seed.items():
            value = parse_target_coordinate(getattr(self, f"buddha_seed_{field}_var").get())
            zone[field] = last if value is None else value
        return BuddhaSeedSettings(camera_pixels=bool(self.buddha_seed_camera_var.get()), **zone)

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
                    and (self.generated_settings is None
                         or cur != (self.generated_settings, self.generated_natural_stop)
                         or obstacles_signature(self.obstacles) != self.gen_obstacles_sig))
        revert_ok = cur is None or cur != (self.baseline, self.baseline_natural_stop)
        self.start_btn.state(["!disabled"] if start_ok else ["disabled"])
        self.revert_btn.state(["!disabled"] if revert_ok else ["disabled"])
        if hasattr(self, "save_btn"):
            has_image = (self.generated_settings is not None and not self.running
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
        for key, value in settings_to_dict(self.baseline, self.baseline_natural_stop).items():
            self.set_setting_control(key, value)
        self.on_limit_var_changed()
        self.on_shape_param_changed()
        self.on_fuzz_changed()
