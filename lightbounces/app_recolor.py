"""Colour mapping on the live image: the Colorations tab's handlers (model, source, metric, points...), the Buddha count buffers and their blur cache, and recolouring rows from the stored per-pixel data."""
import numpy as np
from .constants import BACKGROUND, TERM_STAT_VALUE
from .kernels import N_STATS, PIXEL_TRACED, ST_TERMINATION, scatter_multi_buddha
from .metrics import (
    BUDDHA_BRIGHTNESS_OPTIONS, BUDDHA_BRIGHTNESS_VALUES, BUDDHA_COMPRESSION_LABELS,
    BUDDHA_MULTI, BUDDHA_MULTI_ALL_MID, BUDDHA_MULTI_ALL_SELF, BUDDHA_MULTI_PROG_BOUNCE,
    BUDDHA_MULTI_PROG_MID, BUDDHA_POINT_CATALOG, BUDDHA_POINT_LABELS, CHAN_BUDDHA,
    CHAN_SRC_BUDDHA, CHAN_SRC_CONSTANT, CHAN_SRC_LABELS, METRIC_TYPE_LABELS, METRIC_TYPE_POINT_PAIR, NONE_ONE,
    NONE_ZERO, OTHER_LABELS, POINT_DESCRIPTIONS, POINT_LABELS, POINT_METRIC_LABELS, POINT_NOTE,
    POINT_STATS, STAT_DESCRIPTIONS, STAT_LABELS, blur_counts_image, buddha_compress,
    build_buddha_blur_kernel, constant_label_to_stat, constant_stat_to_label, extract_point_xy,
    make_mapping, other_label_to_stat, other_stat_to_label, point_comparison_metric,
    resolve_point_xy, scatter_points_raw, scatter_seeded_buddha, world_radius_to_pixels
)
from .ui_text import default_map_help
from .colorations import MODES, channels_to_rgb


class RecolorMixin:
    # ---- color mapping ---------------------------------------------------- #
    def on_mode_changed(self, _event=None):
        """RGB <-> HSV: relabel the table rows and, if the edited coloration is the
        live one, recolor right away."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.mode = self.mode_var.get()
        for c, (name, color) in enumerate(MODES[coloration.mode]):
            self.comp_labels[c].configure(text=name, foreground=color)
        self.map_help_var.set(default_map_help(coloration.mode))
        self.mode_combo.selection_clear()
        if coloration is self.live_coloration():
            self.update_stat_tags()
            self.recolor(0, self.rows_done)
            self.draw()
            if self.pinned_px is not None:
                self.update_preview(*self.pinned_px)
        self.rebuild_coloration_list()
        self.refresh_timeline_diffs()

    def refresh_row(self, c):
        """Show channel c's Query, Buddha, or Constant controls (whichever
        its Source is set to), and within Query, set up the Metric and Point
        dropdowns to match its Type: point comparison -> the comparison
        metrics plus Point 1/2 dropdowns; other -> the remaining metrics and
        no Point dropdowns. Reflects the selected coloration (whichever one the
        Color mapping panel is currently showing/editing)."""
        coloration = self.selected_coloration_obj()
        src = coloration.chan_src[c]
        self.query_frames[c].grid_remove()
        self.buddha_frames[c].grid_remove()
        self.constant_frames[c].grid_remove()
        if src == CHAN_SRC_BUDDHA:
            self.buddha_frames[c].grid()
            return
        if src == CHAN_SRC_CONSTANT:
            self.constant_frames[c].grid()
            self.constant_vars[c].set(constant_stat_to_label(coloration.constant_choice[c]))
            return
        self.query_frames[c].grid()
        self.metric_type_vars[c].set(METRIC_TYPE_LABELS[coloration.metric_types[c]])
        if coloration.metric_types[c] == METRIC_TYPE_POINT_PAIR:
            self.map_combos[c].configure(values=POINT_METRIC_LABELS)
            self.map_vars[c].set(STAT_LABELS[coloration.point_metric[c]])
            self.point1_combos[c].grid()
            self.point2_combos[c].grid()
        else:
            self.map_combos[c].configure(values=OTHER_LABELS)
            self.map_vars[c].set(other_stat_to_label(coloration.other_metric[c]))
            self.point1_combos[c].grid_remove()
            self.point2_combos[c].grid_remove()

    def update_stat_tags(self):
        """Tag each metric in the preview list, after its value, with the
        channel letter(s) it currently drives, e.g. '-> B'. Always about the
        live coloration specifically - this describes the top-left preview image,
        not whichever coloration is being edited."""
        letters = [name[0] for name, _color in MODES[self.live_coloration().mode]]
        for s in range(N_STATS):
            used = [letters[c] for c in range(3) if self.mapping[c] == s]
            self.stat_tags[s].configure(text="-> " + ", ".join(used) if used else "")

    def on_chan_src_changed(self, c):
        """Source dropdown changed (Query <-> Buddha <-> Constant)."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.chan_src[c] = CHAN_SRC_LABELS.index(self.chan_src_vars[c].get())
        self.chan_src_combos[c].selection_clear()
        self.refresh_row(c)
        self.apply_mapping_change(c)

    def on_constant_changed(self, c):
        """The "Constant" source's own dropdown changed (Always 0 <-> Always 1)."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.constant_choice[c] = constant_label_to_stat(self.constant_vars[c].get())
        self.constant_combos[c].selection_clear()
        self.apply_mapping_change(c)

    def on_metric_type_changed(self, c):
        """Type dropdown changed (point comparison <-> other)."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.metric_types[c] = METRIC_TYPE_LABELS.index(self.metric_type_vars[c].get())
        self.metric_type_combos[c].selection_clear()
        self.refresh_row(c)
        self.apply_mapping_change(c)

    def on_metric_selected(self, c):
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        label = self.map_vars[c].get()
        if coloration.metric_types[c] == METRIC_TYPE_POINT_PAIR:
            coloration.point_metric[c] = POINT_STATS[POINT_METRIC_LABELS.index(label)]
        else:
            coloration.other_metric[c] = other_label_to_stat(label)
        self.apply_mapping_change(c)

    def on_point_selected(self, c):
        if self._color_editor_loading:
            return
        self.apply_mapping_change(c)

    def on_buddha_points_changed(self, c):
        """One of channel c's 8 Buddha point checkboxes was toggled."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.buddha_points[c] = tuple(v.get() for v in self.buddha_point_vars[c])
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

    def apply_mapping_change(self, c):
        """Channel c's Source, Type, Metric, Point 1/2, Buddha points,
        Buddha compression or Buddha brightness changed, for whichever coloration
        is selected: update its own data and the panel's help text always;
        if the edited coloration is also the live one, additionally update
        everything the live preview depends on (self.mapping is always the
        LIVE coloration's mapping specifically, so it's only touched in that
        case - the help text below is computed fresh from `coloration` instead,
        so it's correct regardless of which coloration is being edited)."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.point1_choice = [POINT_LABELS.index(v.get()) for v in self.point1_vars]
        coloration.point2_choice = [POINT_LABELS.index(v.get()) for v in self.point2_vars]
        coloration_mapping = make_mapping(coloration.metric_types, coloration.point_metric, coloration.other_metric,
                                    coloration.chan_src, coloration.constant_choice)

        name = MODES[coloration.mode][c][0]
        if coloration.chan_src[c] == CHAN_SRC_BUDDHA:
            chosen = [BUDDHA_POINT_LABELS[i] for i, v in enumerate(coloration.buddha_points[c]) if v]
            comp = BUDDHA_COMPRESSION_LABELS[coloration.buddha_compression[c]]
            bright = BUDDHA_BRIGHTNESS_OPTIONS[coloration.buddha_brightness[c]]
            if chosen:
                text = (f"{name}: Buddha coloration - a visit count builds up at "
                       f"whichever pixel each ray's {', '.join(chosen).lower()} lands "
                       f"in, compressed to 0..255 with {comp.lower()}"
                       + (f", boosted {bright}." if bright != "1x" else "."))
            else:
                text = f"{name}: Buddha coloration, but no points are selected - always 0."
        else:
            stat = coloration_mapping[c]
            if stat == NONE_ZERO:
                text = f"{name}: always 0."
            elif stat == NONE_ONE:
                text = f"{name}: always 1."
            else:
                text = f"{name}: {STAT_DESCRIPTIONS[stat]}"
                if stat in POINT_STATS:
                    text += (f" Point 1: {POINT_DESCRIPTIONS[coloration.point1_choice[c]]} "
                            f"Point 2: {POINT_DESCRIPTIONS[coloration.point2_choice[c]]}"
                            + POINT_NOTE)
        self.map_help_var.set(text)
        self.map_combos[c].selection_clear()
        self.point1_combos[c].selection_clear()
        self.point2_combos[c].selection_clear()

        if coloration is self.live_coloration():
            self.mapping = coloration_mapping
            self.update_stat_tags()
            if coloration.chan_src[c] == CHAN_SRC_BUDDHA:
                self.rebuild_buddha_counts()   # only the edited channel's Buddha
                                               # state could possibly need this -
                                               # editing an unrelated (Query/
                                               # Constant) channel never does, and
                                               # skipping it there is what keeps a
                                               # color change fast even when some
                                               # OTHER channel uses an expensive
                                               # "multi" Buddha point option
            self.recolor(0, self.rows_done)
            self.draw()
            if self.pinned_px is not None:
                self.update_preview(*self.pinned_px)
        self.rebuild_coloration_list()
        self.refresh_timeline_diffs()

    def scatter_buddha_rows(self, r0, r1):
        """Add rows [r0, r1)'s contribution to every Buddha-mode channel's
        RAW (unblurred) visit-count buffer. For each such channel's selected
        points: a "simple" point increments the count at whichever pixel
        its (already computed) sample point lands in, for every ray that
        fired (and, if the option is restricted to a termination reason,
        only for rays that ended that way); a "multi" point re-traces the
        row's rays to reach parts of the path the per-pixel summary doesn't
        keep (see scatter_multi_buddha). Blur (if any) is never applied
        here - it's applied once, lazily, to the whole accumulated buffer
        when a colorized value is actually needed (see
        blurred_buddha_counts) - so this stays cheap regardless of the
        blur radius; each channel whose raw counts changed here has its
        cached blurred version invalidated."""
        coloration = self.live_coloration()
        if r1 <= r0 or not any(s == CHAN_SRC_BUDDHA for s in coloration.chan_src):
            return
        h, w = self.img_h, self.img_w
        world_rect = self.image_world_rect
        seeds = self.current_buddha_seed_settings()
        fired = self.pixel_state[r0:r1] == PIXEL_TRACED
        if seeds.camera_pixels and not fired.any():
            return
        term_stat = self.stats[r0:r1, :, ST_TERMINATION]
        for c in range(3):
            if coloration.chan_src[c] != CHAN_SRC_BUDDHA or self.buddha_counts[c] is None:
                continue
            if not seeds.camera_pixels:
                # seeded from their own grid: every chosen point comes from rays fired there (rows of the grid
                # correspond to the rows of the image as they are done)
                if scatter_seeded_buddha(self.buddha_counts[c], world_rect, seeds, r0, r1, coloration.buddha_points[c],
                                         self.gen_stop, self.gen_ray,
                                         self.settings_of_current_image().fuzz.rng_seed):
                    self.buddha_counts_blurred[c] = None
                continue
            want_multi = {BUDDHA_MULTI_ALL_SELF: False, BUDDHA_MULTI_ALL_MID: False,
                         BUDDHA_MULTI_PROG_BOUNCE: False, BUDDHA_MULTI_PROG_MID: False}
            touched = False
            for i, on in enumerate(coloration.buddha_points[c]):
                if not on:
                    continue
                _label, style, ref, term_filter = BUDDHA_POINT_CATALOG[i]
                if style == BUDDHA_MULTI:
                    want_multi[ref] = True
                    continue
                mask = fired
                if term_filter is not None:
                    target = TERM_STAT_VALUE[term_filter]
                    mask = mask & (np.abs(term_stat - target) < 0.01)
                if not mask.any():
                    continue
                xs = self.characteristic_points[r0:r1, :, ref, 0][mask]
                ys = self.characteristic_points[r0:r1, :, ref, 1][mask]
                col = np.floor((xs - world_rect.x_start) * (w / (world_rect.x_stop - world_rect.x_start))).astype(np.int64)
                row = np.floor((world_rect.y_stop - ys) * (h / (world_rect.y_stop - world_rect.y_start))).astype(np.int64)
                valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
                scatter_points_raw(self.buddha_counts[c], row, col, valid)
                touched = True
            if any(want_multi.values()):
                g = self.settings_of_current_image()
                scatter_multi_buddha(
                    r0, r1, world_rect, self.gen_stop, self.gen_ray, g.fuzz.rng_seed,
                    want_multi[BUDDHA_MULTI_ALL_SELF], want_multi[BUDDHA_MULTI_ALL_MID],
                    want_multi[BUDDHA_MULTI_PROG_BOUNCE], want_multi[BUDDHA_MULTI_PROG_MID],
                    self.buddha_counts[c])
                touched = True
            if touched:
                self.buddha_counts_blurred[c] = None

    def blurred_buddha_counts(self, c):
        """Channel c's blurred Buddha counts, computed once (via a single
        whole-image FFT pass, independent of how many points were
        scattered or how large the blur radius is) and cached until the
        next scatter_buddha_rows call invalidates it. None if the channel
        isn't in Buddha mode at all."""
        if self.buddha_counts[c] is None:
            return None
        if self.buddha_counts_blurred[c] is None:
            blur = self.current_blur_settings()
            radius_px = world_radius_to_pixels(blur.radius, self.img_w, self.img_h)
            kernel = build_buddha_blur_kernel(radius_px, blur.type)
            self.buddha_counts_blurred[c] = blur_counts_image(self.buddha_counts[c], kernel)
        return self.buddha_counts_blurred[c]

    def rebuild_buddha_counts(self):
        """Full reset + rescatter of every Buddha-mode channel's visit-count
        buffer, from every row computed so far. Used when the Buddha
        configuration itself changes (which points, or which channels are in
        Buddha mode at all) - the existing counts would otherwise be for the
        wrong selection - and when starting a fresh render."""
        self.buddha_counts = [
            np.zeros((self.img_h, self.img_w), dtype=np.float64) if s == CHAN_SRC_BUDDHA else None
            for s in self.live_coloration().chan_src]
        self.buddha_counts_blurred = [None] * 3
        self.scatter_buddha_rows(0, self.rows_done)

    def recolor(self, r0, r1):
        """Turn stored metrics and sample points into pixel colors, rows [r0, r1)."""
        if r1 <= r0:
            return
        coloration = self.live_coloration()
        st = self.stats[r0:r1]
        sp = self.characteristic_points[r0:r1]
        computed = self.pixel_state[r0:r1] == PIXEL_TRACED
        out = self.img[r0:r1]
        out[:] = BACKGROUND
        n = int(computed.sum())
        if n:
            vals = []
            for c in range(3):
                s = self.mapping[c]
                if s == CHAN_BUDDHA:
                    counts = self.blurred_buddha_counts(c)
                    if counts is None:
                        vals.append(np.zeros(n))
                    else:
                        sub = counts[r0:r1][computed].astype(np.float64)
                        mult = BUDDHA_BRIGHTNESS_VALUES[coloration.buddha_brightness[c]]
                        vals.append(buddha_compress(sub, coloration.buddha_compression[c], mult))
                elif s == NONE_ZERO:
                    vals.append(np.zeros(n))
                elif s == NONE_ONE:
                    vals.append(np.ones(n))
                elif s in POINT_STATS:
                    x1, y1 = extract_point_xy(sp, coloration.point1_choice[c], computed, n)
                    x2, y2 = extract_point_xy(sp, coloration.point2_choice[c], computed, n)
                    vals.append(point_comparison_metric(s, x1, y1, x2, y2))
                else:
                    vals.append(np.clip(st[..., s][computed].astype(np.float64),
                                        0.0, 1.0))
            r, g, b = channels_to_rgb(coloration.mode, *vals)
            rgb = np.stack((r, g, b), axis=-1)
            out[computed] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    def component_values(self, vec, sp, x0, y0):
        """The three channel values (0..1) for one ray under the current
        mapping. vec: its metrics, sp: its sample points, (x0, y0): the
        pixel it was emitted from (used to look up a Buddha channel's
        already-accumulated count at that same pixel)."""
        coloration = self.live_coloration()
        vals = []
        for c in range(3):
            s = self.mapping[c]
            if s == CHAN_BUDDHA:
                counts = self.blurred_buddha_counts(c)
                if counts is None:
                    v = 0.0
                else:
                    cx, cy = self.world_to_canvas(x0, y0)
                    col, row = int(np.floor(cx)), int(np.floor(cy))
                    if 0 <= row < self.img_h and 0 <= col < self.img_w:
                        mult = BUDDHA_BRIGHTNESS_VALUES[coloration.buddha_brightness[c]]
                        v = float(buddha_compress(counts[row, col], coloration.buddha_compression[c],
                                                  mult))
                    else:
                        v = 0.0
            elif s == NONE_ZERO:
                v = 0.0
            elif s == NONE_ONE:
                v = 1.0
            elif s in POINT_STATS:
                x1, y1 = resolve_point_xy(sp, coloration.point1_choice[c])
                x2, y2 = resolve_point_xy(sp, coloration.point2_choice[c])
                v = float(point_comparison_metric(s, x1, y1, x2, y2))
            else:
                v = min(max(float(vec[s]), 0.0), 1.0)
            vals.append(v)
        return vals

    def map_color(self, vals):
        """Color (r, g, b) from three channel values under the current model."""
        r, g, b = channels_to_rgb(self.live_coloration().mode, *[np.array([v]) for v in vals])
        return tuple(int(np.clip(np.rint(ch[0]), 0, 255)) for ch in (r, g, b))
