"""Colour mapping on the live image: the Colorations tab's handlers (model, source, metric, points...) and recolouring rows from the stored per-pixel data (the colour chain itself is in color_pipeline.py, the Buddha counts in app_buddha.py and buddha.py)."""
import numpy as np
from .kernels import N_STATS, PIXEL_TRACED
from .metrics import (
    BUDDHA_BRIGHTNESS_OPTIONS, BUDDHA_COMPRESSION_LABELS, BUDDHA_POINT_LABELS, CHAN_BUDDHA,
    CHAN_SRC_BUDDHA, CHAN_SRC_CONSTANT, CHAN_SRC_LABELS, METRIC_TYPE_LABELS,
    METRIC_TYPE_POINT_PAIR, NONE_ONE, NONE_ZERO, OTHER_LABELS, POINT_DESCRIPTIONS, POINT_LABELS,
    POINT_METRIC_LABELS, POINT_NOTE, POINT_METRICS, METRIC_DESCRIPTIONS, METRIC_LABELS,
    constant_label_to_metric, constant_metric_to_label, make_mapping, other_label_to_metric,
    other_metric_to_label
)
from .color_pipeline import coloration_fits, paint, ray_values
from .engines import other_metric_labels, point_menu_labels, point_metric_labels
from .stored_layout import FULL_LAYOUT
from .ui_text import default_map_help
from .colorations import COLOR_MODELS, channels_to_rgb


class RecolorMixin:
    # ---- color mapping ---------------------------------------------------- #
    def on_color_model_changed(self, _event=None):
        """RGB <-> HSV: relabel the table rows and, if the edited coloration is the
        live one, recolor right away."""
        if self._color_editor_loading:
            return
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        coloration.color_model = self.color_model_var.get()
        for c, (name, color) in enumerate(COLOR_MODELS[coloration.color_model]):
            self.comp_labels[c].configure(text=name, foreground=color)
        self.map_help_var.set(default_map_help(coloration.color_model))
        self.color_model_combo.selection_clear()
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
            self.constant_vars[c].set(constant_metric_to_label(coloration.constant_choice[c]))
            return
        self.query_frames[c].grid()
        self.metric_type_vars[c].set(METRIC_TYPE_LABELS[coloration.metric_types[c]])
        offered = self.current_engine_description()
        for combo in (self.point1_combos[c], self.point2_combos[c]):
            combo.configure(values=point_menu_labels(offered))
        if coloration.metric_types[c] == METRIC_TYPE_POINT_PAIR:
            self.map_combos[c].configure(values=point_metric_labels(offered))
            self.map_vars[c].set(METRIC_LABELS[coloration.point_metric[c]])
            self.point1_combos[c].grid()
            self.point2_combos[c].grid()
        else:
            self.map_combos[c].configure(values=other_metric_labels(offered))
            self.map_vars[c].set(other_metric_to_label(coloration.other_metric[c]))
            self.point1_combos[c].grid_remove()
            self.point2_combos[c].grid_remove()

    def update_stat_tags(self):
        """Tag each metric in the preview list, after its value, with the
        channel letter(s) it currently drives, e.g. '-> B'. Always about the
        live coloration specifically - this describes the top-left preview image,
        not whichever coloration is being edited."""
        letters = [name[0] for name, _color in COLOR_MODELS[self.live_coloration().color_model]]
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
        coloration.constant_choice[c] = constant_label_to_metric(self.constant_vars[c].get())
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
            coloration.point_metric[c] = POINT_METRICS[POINT_METRIC_LABELS.index(label)]
        else:
            coloration.other_metric[c] = other_label_to_metric(label)
        self.apply_mapping_change(c)

    def on_point_selected(self, c):
        if self._color_editor_loading:
            return
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

        name = COLOR_MODELS[coloration.color_model][c][0]
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
            metric_slot = coloration_mapping[c]
            if metric_slot == NONE_ZERO:
                text = f"{name}: always 0."
            elif metric_slot == NONE_ONE:
                text = f"{name}: always 1."
            else:
                text = f"{name}: {METRIC_DESCRIPTIONS[metric_slot]}"
                if metric_slot in POINT_METRICS:
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

    def recolor(self, r0, r1):
        """Turn stored metrics and characteristic points into pixel colors, rows [r0, r1) (see color_pipeline)."""
        if r1 <= r0 or not coloration_fits(self.live_coloration(), self.current_figure().layout, self.current_buddha_seed_settings()):
            return                                  # (the image waits for Start: it is of another engine than the controls')
        computed = self.pixel_state[r0:r1] == PIXEL_TRACED
        blurred = [None] * 3
        if computed.any():
            for c in range(3):
                counts = self.blurred_buddha_counts(c) if self.mapping[c] == CHAN_BUDDHA else None
                blurred[c] = None if counts is None else counts[r0:r1]
        paint(self.img[r0:r1], self.live_coloration(), self.mapping, self.stats[r0:r1],
              self.characteristic_points[r0:r1], computed, blurred, self.current_figure().layout)

    def component_values(self, vec, ray_characteristic_points, x0, y0, layout=FULL_LAYOUT):
        """The three channel values (0..1) for one ray under the current mapping. vec: its metrics,
        ray_characteristic_points: its characteristic points, (x0, y0): the pixel it was fired from (used to look up a
        Buddha channel's already-accumulated count at that same pixel). `layout` says which metrics and points they keep."""
        if not coloration_fits(self.live_coloration(), layout, self.current_buddha_seed_settings()):
            return (0.0, 0.0, 0.0)                  # (the controls hold another engine's choices than this ray's engine stores)
        at_pixel = [None] * 3
        for c in range(3):
            if self.mapping[c] != CHAN_BUDDHA:
                continue
            counts = self.blurred_buddha_counts(c)
            if counts is not None:
                cx, cy = self.world_to_canvas(x0, y0)
                col, row = int(np.floor(cx)), int(np.floor(cy))
                inside = 0 <= row < self.img_h and 0 <= col < self.img_w
                at_pixel[c] = counts[row, col] if inside else 0.0
        return ray_values(self.live_coloration(), self.mapping, vec, ray_characteristic_points, at_pixel, layout)

    def map_color(self, vals):
        """Color (r, g, b) from three channel values under the current model."""
        r, g, b = channels_to_rgb(self.live_coloration().color_model, *[np.array([v]) for v in vals])
        return tuple(int(np.clip(np.rint(ch[0]), 0, 255)) for ch in (r, g, b))
