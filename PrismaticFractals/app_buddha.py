"""Buddha coloration on the live image: the Buddha seeding and Buddha blur tabs (a change recolours at once), a channel's Buddha point, compression and brightness choices, and the visit-count buffers with their blur cache. How the counts are made and read is in buddha.py."""
import tkinter as tk
from tkinter import ttk
import numpy as np
from .constants import (
    DEFAULT_BUDDHA_SEED_CAMERA, DEFAULT_BUDDHA_SEED_X_START, DEFAULT_BUDDHA_SEED_X_STOP,
    DEFAULT_BUDDHA_SEED_Y_START, DEFAULT_BUDDHA_SEED_Y_STOP, BUDDHA_BLUR_LABELS,
    DEFAULT_BUDDHA_BLUR_TYPE, RIGHT_W
)
from .kernels import PIXEL_TRACED
from .metrics import BUDDHA_BRIGHTNESS_OPTIONS, BUDDHA_COMPRESSION_LABELS, CHAN_SRC_BUDDHA
from .buddha import blur_counts_image, blur_kernel, scatter_buddha_channel
from .color_pipeline import coloration_fits

SEED_ZONE_FIELDS = ("x_start", "x_stop", "y_start", "y_stop")        # the fields of the Buddha seeding zone


class BuddhaMixin:
    # ---- a channel's Buddha choices ------------------------------------------------ #
    def on_buddha_points_changed(self, c):
        """One of channel c's 8 Buddha point checkboxes was toggled."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.buddha_points[c] = tuple(v.get() for v in self.buddha_point_vars[c])
        self.apply_mapping_change(c)

    def on_buddha_path_kinds_changed(self, c):
        """One of channel c's three path-kind checkboxes (escaping / non-escaping / terminated by obstacle) was toggled."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.buddha_path_kinds[c] = tuple(bool(v.get()) for v in self.buddha_kind_vars[c])
        self.apply_mapping_change(c)

    def on_buddha_compression_changed(self, c):
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.buddha_compression[c] = BUDDHA_COMPRESSION_LABELS.index(
            self.buddha_compression_vars[c].get())
        self.buddha_compression_combos[c].selection_clear()
        self.apply_mapping_change(c)

    def on_buddha_brightness_changed(self, c):
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.buddha_brightness[c] = BUDDHA_BRIGHTNESS_OPTIONS.index(
            self.buddha_brightness_vars[c].get())
        self.buddha_brightness_combos[c].selection_clear()
        self.apply_mapping_change(c)

    # ---- the Buddha seeding and Buddha blur tabs ----------------------------------- #
    def build_buddha_seeding_controls(self, parent):
        """The Buddha seeding tab: where the rays that Buddha channels scatter start (their seed points - not to be
        confused with the Emissions tab, which is about what direction each ray is fired in). With the checkbox
        they start at the camera's pixels, the same places as the image's own rays; without it, on a grid laid
        over the zone below with one seed per pixel of the output, however large or small the zone. Nothing
        here changes what a generated image is made of, so a change recolours at once, no Start."""
        self.buddha_seed_camera_var = tk.BooleanVar(value=DEFAULT_BUDDHA_SEED_CAMERA)
        self.buddha_seed_camera_check = ttk.Checkbutton(
            parent, text="use camera pixels as emission seed points",
            variable=self.buddha_seed_camera_var, command=self.on_buddha_seed_camera_changed)
        self.buddha_seed_camera_check.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        zone = ttk.LabelFrame(parent, text="Seed point grid")
        zone.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        zone.columnconfigure(0, weight=1)
        coord_vcmd = (self.root.register(self.validate_signed_float), "%P")
        self.buddha_seed_labels = []
        self.buddha_seed_entries = []
        for row, (field, label, default) in enumerate(
                (("x_start", "X start", DEFAULT_BUDDHA_SEED_X_START),
                 ("x_stop", "X stop", DEFAULT_BUDDHA_SEED_X_STOP),
                 ("y_start", "Y start", DEFAULT_BUDDHA_SEED_Y_START),
                 ("y_stop", "Y stop", DEFAULT_BUDDHA_SEED_Y_STOP))):
            var = tk.StringVar(value=str(default))
            setattr(self, f"buddha_seed_{field}_var", var)
            name = ttk.Label(zone, text=label + " (world units):")
            name.grid(row=row, column=0, sticky="w", padx=(8, 6), pady=2)
            entry = ttk.Entry(zone, textvariable=var, width=14, justify="right", validate="key",
                              validatecommand=coord_vcmd)
            entry.grid(row=row, column=1, padx=(0, 8), pady=2)
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, self.on_buddha_seed_commit)
            setattr(self, f"buddha_seed_{field}_entry", entry)
            self.buddha_seed_labels.append(name)
            self.buddha_seed_entries.append(entry)
        note = ttk.Label(zone, foreground="#666", wraplength=RIGHT_W - 70,
                         text="Seed points are spread over this zone at regular spacings, one for each pixel "
                              "of the image (the live image or an exported one), so shrinking the zone packs "
                              "the same number of seeds into less space. A seed that is not inside a shape "
                              "that fires rays fires nothing. The zone, dashed in purple on the preview while "
                              "this tab is open, does not have to be inside the image.")
        note.grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 6))
        self.buddha_seed_labels.append(note)
        self.update_buddha_seed_state()

    def update_buddha_seed_state(self):
        """The zone's controls are greyed out while the camera's pixels are the seeds."""
        state = ["disabled"] if self.model.value("buddha_seed_camera") else ["!disabled"]
        for widget in self.buddha_seed_entries + self.buddha_seed_labels:
            widget.state(state)

    def on_buddha_seed_camera_changed(self):
        self.on_buddha_seeds_changed()

    def on_buddha_seed_commit(self, _event=None):
        """Enter pressed or focus left a zone entry: tidy its text (an invalid one goes back to the last valid
        value) and, if the zone changed, recolour."""
        for field in SEED_ZONE_FIELDS:
            self.bindings[f"buddha_seed_{field}"].commit()
        self.on_buddha_seeds_changed()

    def on_buddha_seeds_changed(self):
        """The seeding changed (by hand, or loaded from a keyframe or a project): refresh what shows it now, and
        recolour - which takes a moment when a Buddha channel has to trace its rays again - once the changes of this
        moment have all arrived."""
        self.update_buddha_seed_state()
        self.update_seed_zone_outline()
        self.refresh_timeline_diffs()
        if self.seed_job is None:
            self.seed_job = self.root.after_idle(self.refresh_buddha_seeds)

    def refresh_buddha_seeds(self):
        """Scatter every Buddha channel again from the seeds as they now are, and recolour (like refresh_blur, but
        the scattered counts themselves are new). Costs nothing when no coloration has a Buddha channel."""
        self.seed_job = None
        self.coloration_thumb_cache = {}
        if not any(src == CHAN_SRC_BUDDHA for v in self.colorations for src in v.chan_src):
            return
        self.rebuild_buddha_counts()
        self.recolor(0, self.rows_done)
        self.draw()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)
        self.rebuild_coloration_list()

    def build_buddha_blur_controls(self, parent):
        """Radius slider and blur-type dropdown for Buddha coloration's
        point-scattering blur - one setting for the whole image (a
        BuddhaBlurSettings, beside the ray simulation's settings and not part of
        them), not a per-coloration/per-channel one, so it lives in its own tab rather
        than inside the Colorations tab's per-channel Buddha controls. A change
        recolors the image at once (the blur is applied to the stored scatter
        counts, never to the simulation, so no Start is needed)."""
        rf = ttk.LabelFrame(parent, text="Radius")
        rf.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 6))
        rf.columnconfigure(1, weight=1)
        self.add_model_entry(rf, 0, "Radius", "buddha_blur_radius", on_commit=self.on_buddha_blur_changed)
        ttk.Label(rf, foreground="#666", wraplength=RIGHT_W - 70,
                  text="A multiple of the image's smaller side, whatever framing it shows: 0.1 "
                       "means a blur radius of a tenth of it. 0 means no blur at all - "
                       "each point still lands on exactly one pixel, as before."
                  ).grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 6))

        tf = ttk.LabelFrame(parent, text="Blur type")
        tf.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        tf.columnconfigure(0, weight=1)
        self.buddha_blur_type_var = tk.StringVar(value=BUDDHA_BLUR_LABELS[
            DEFAULT_BUDDHA_BLUR_TYPE])
        self.buddha_blur_type_combo = ttk.Combobox(
            tf, textvariable=self.buddha_blur_type_var, values=BUDDHA_BLUR_LABELS,
            state="readonly", width=22)
        self.buddha_blur_type_combo.grid(row=0, column=0, sticky="w", padx=8, pady=(6, 4))
        self.buddha_blur_type_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self.on_buddha_blur_changed())
        ttk.Label(tf, foreground="#666", wraplength=RIGHT_W - 70,
                  text="All four share the radius above. Gaussian treats it as a "
                       "standard deviation, so it can affect pixels somewhat beyond "
                       "that distance; the other three stop exactly at it. Bokeh is "
                       "a uniform disc; catadioptric bokeh is a ring, like a mirror "
                       "lens's characteristic donut-shaped out-of-focus highlights."
                  ).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 6))

    def on_buddha_blur_changed(self):
        """The Buddha blur radius or type changed (by hand, or loaded from a keyframe):
        the keyframe list is refreshed now, the recoloring (which can take a moment with
        several Buddha channels) once the slider events of this moment have all arrived."""
        self.refresh_timeline_diffs()
        if self.blur_job is None:
            self.blur_job = self.root.after_idle(self.refresh_blur)

    def refresh_blur(self):
        """Recolor everything that shows a Buddha channel with the blur now set: every
        channel's cached blurred counts and every non-live coloration's cached thumbnail are for
        the old blur; the live coloration is recolored, redrawn and its thumbnail rebuilt. Costs
        nothing visible when no coloration has a Buddha channel."""
        self.blur_job = None
        self.buddha_counts_blurred = [None] * 3
        self.coloration_thumb_cache = {}
        if not any(src == CHAN_SRC_BUDDHA for v in self.colorations for src in v.chan_src):
            return
        self.recolor(0, self.rows_done)
        self.draw()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)
        self.rebuild_coloration_list()

    # ---- the visit counts ---------------------------------------------------------- #
    def scatter_buddha_rows(self, r0, r1):
        """Add rows [r0, r1)'s contribution to every Buddha-mode channel's RAW (unblurred) visit-count buffer (see
        buddha.scatter_buddha_channel: the stored characteristic points for the "simple" options, rays traced again for
        the "multi" ones, or a seed grid when the seeding is not the camera's pixels). Blur (if any) is never applied
        here - it's applied once, lazily, to the whole accumulated buffer when a colorized value is actually needed
        (see blurred_buddha_counts) - so this stays cheap regardless of the blur radius; each channel whose raw counts
        changed here has its cached blurred version invalidated."""
        coloration = self.live_coloration()
        if r1 <= r0 or not any(s == CHAN_SRC_BUDDHA for s in coloration.chan_src):
            return
        if not coloration_fits(coloration, self.current_figure().layout, self.current_buddha_seed_settings()):
            return                                  # (the image waits for Start: it is of another engine than the controls')
        seeds = self.current_buddha_seed_settings()
        fired = self.pixel_state[r0:r1] == PIXEL_TRACED
        stats = self.stats[r0:r1]
        characteristic_points = self.characteristic_points[r0:r1]
        figure = self.current_figure()
        for c in range(3):
            if coloration.chan_src[c] != CHAN_SRC_BUDDHA or self.buddha_counts[c] is None:
                continue
            if scatter_buddha_channel(self.buddha_counts[c], coloration.buddha_points[c], r0, r1, self.image_world_rect,
                                      seeds, fired, stats, characteristic_points, figure.termination,
                                      figure.engine_args, figure.rng_seed, figure.layout, coloration.buddha_path_kinds[c]):
                self.buddha_counts_blurred[c] = None

    def blurred_buddha_counts(self, c):
        """Channel c's blurred Buddha counts, computed once (via a single whole-image FFT pass, independent of how many
        points were scattered or how large the blur radius is) and cached until the next scatter_buddha_rows call
        invalidates it. None if the channel isn't in Buddha mode at all."""
        if self.buddha_counts[c] is None:
            return None
        if self.buddha_counts_blurred[c] is None:
            kernel = blur_kernel(self.current_blur_settings(), self.img_w, self.img_h)
            self.buddha_counts_blurred[c] = blur_counts_image(self.buddha_counts[c], kernel)
        return self.buddha_counts_blurred[c]

    def rebuild_buddha_counts(self):
        """Full reset + rescatter of every Buddha-mode channel's visit-count buffer, from every row computed so far.
        Used when the Buddha configuration itself changes (which points, or which channels are in Buddha mode at all) -
        the existing counts would otherwise be for the wrong selection - and when starting a fresh render."""
        self.buddha_counts = [
            np.zeros((self.img_h, self.img_w), dtype=np.float64) if s == CHAN_SRC_BUDDHA else None
            for s in self.live_coloration().chan_src]
        self.buddha_counts_blurred = [None] * 3
        self.scatter_buddha_rows(0, self.rows_done)
