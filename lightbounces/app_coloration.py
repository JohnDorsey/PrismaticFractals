"""Colorations and the Colorations tab: the list of colorations with thumbnails, creating/duplicating/deleting/selecting a coloration, and the colour editor (the Buddha tabs are in app_buddha.py)."""
import tkinter as tk
from tkinter import ttk
from .constants import RIGHT_W, SELECTED_OUTLINE, SELECTED_OUTLINE_PX, THUMB_SIZE
from .image_io import encode_png_bytes
from .widgets import ThumbnailStrip
from .kernels import N_POINTS, N_PRED, N_SEQS
from .metrics import (
    BUDDHA_BRIGHTNESS_OPTIONS, BUDDHA_COMPRESSION_LABELS, BUDDHA_POINT_LABELS,
    CHAN_CONSTANT_LABELS, CHAN_SRC_LABELS, METRIC_TYPE_LABELS, N_BUDDHA_POINTS, PATH_KIND_LABELS, POINT_LABELS,
    POINT_MENU_LABELS, POINT_OF_SLOT, POINT_REDUCTION_SHORT, SEQUENCE_SHORT,
    constant_metric_to_label, make_mapping
)
from .ui_text import default_map_help
from .colorations import COLOR_MODELS, COLOR_MODEL_NAMES, ColorationConfig


class ColorationMixin:
    # ---- colorations (Colorations tab) -------------------------------------------- #
    def coloration_by_id(self, coloration_id):
        """The live ColorationConfig for one coloration id, or None if it doesn't exist."""
        for v in self.colorations:
            if v.id == coloration_id:
                return v
        return None

    def live_coloration(self):
        """The coloration the top-left preview shows: whichever coloration is currently
        selected in the Colorations tab. Selecting a coloration always shows it
        here - there's no separate "focus" step."""
        return self.selected_coloration_obj()

    def refresh_live_preview_state(self):
        """Recompute everything that depends on which coloration is currently live
        (self.mapping, the running Buddha counts, the displayed image) -
        called whenever self.live_coloration() changes (selecting a different
        coloration, or creating/deleting one changes the selection)."""
        live = self.live_coloration()
        self.mapping = make_mapping(live.metric_types, live.point_metric, live.other_metric,
                                    live.chan_src, live.constant_choice)
        self.rebuild_buddha_counts()
        self.update_stat_tags()
        self.recolor(0, self.rows_done)
        self.draw()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)

    def selected_coloration_obj(self):
        return self.coloration_by_id(self.selected_coloration_id)

    def all_coloration_ids(self):
        """Every coloration's id, in display order (self.colorations' own list order)."""
        return [v.id for v in self.colorations]

    def new_coloration(self):
        """Add a new (default-settings) coloration right after the selected one,
        and select it - which also makes it the one shown in the preview."""
        v = ColorationConfig(self._coloration_counter)
        self._coloration_counter += 1
        idx = self.all_coloration_ids().index(self.selected_coloration_id)
        self.colorations.insert(idx + 1, v)
        self.selected_coloration_id = v.id
        self.coloration_thumb_cache.pop(v.id, None)
        self.refresh_live_preview_state()
        self.rebuild_coloration_list()
        self.refresh_color_controls_for_selected_coloration()

    def duplicate_coloration(self):
        """A copy of the selected coloration, inserted right after it, selected,
        and shown in the preview."""
        src = self.selected_coloration_obj()
        if src is None:
            return
        new_v = ColorationConfig(self._coloration_counter)
        self._coloration_counter += 1
        for field in ColorationConfig.__slots__:
            if field == "id":
                continue
            val = getattr(src, field)
            setattr(new_v, field, list(val) if isinstance(val, list) else val)
        idx = self.all_coloration_ids().index(self.selected_coloration_id)
        self.colorations.insert(idx + 1, new_v)
        self.selected_coloration_id = new_v.id
        self.coloration_thumb_cache.pop(new_v.id, None)
        self.refresh_live_preview_state()
        self.rebuild_coloration_list()
        self.refresh_color_controls_for_selected_coloration()

    def delete_coloration(self):
        """Removes the selected coloration, unless it's the only one left. The
        deleted coloration was always the one shown in the preview (selection and
        preview are the same thing), so self.colorations[0] becomes the new
        selection, and the preview is refreshed to match it right away."""
        if len(self.colorations) <= 1:
            return
        v = self.coloration_by_id(self.selected_coloration_id)
        if v is None:
            return
        self.colorations.remove(v)
        self.coloration_thumb_cache.pop(v.id, None)
        self.purge_keyframe_settings(f"coloration:{v.id}:")
        self.selected_coloration_id = self.colorations[0].id
        self.coloration_thumb_cache.pop(self.selected_coloration_id, None)
        self.refresh_live_preview_state()
        self.rebuild_coloration_list()
        self.refresh_color_controls_for_selected_coloration()

    def select_coloration(self, coloration_id):
        """Selects a coloration for editing - which also makes it the one the
        top-left panel shows."""
        if coloration_id == self.selected_coloration_id:
            return
        self.selected_coloration_id = coloration_id
        self.coloration_thumb_cache.pop(coloration_id, None)   # about to become live (and possibly
                                                    # edited) - any cached thumbnail from
                                                    # before must not be reused once it's
                                                    # non-live again
        self.refresh_live_preview_state()
        self.rebuild_coloration_list()
        self.refresh_color_controls_for_selected_coloration()

    def refresh_color_controls_for_selected_coloration(self):
        """Populate every Color mapping control from the selected coloration (or
        leave them as-is if the Colorations tab hasn't been built yet, or no
        coloration is selected). Mirrors refresh_obstacle_editor."""
        if not hasattr(self, "mode_var"):
            return                                  # Colorations tab not built yet
        coloration = self.selected_coloration_obj()
        if coloration is None:
            return
        self._color_editor_loading = True
        try:
            self.color_model_var.set(coloration.color_model)
            for c, (name, color) in enumerate(COLOR_MODELS[coloration.color_model]):
                self.comp_labels[c].configure(text=name, foreground=color)
            self.map_help_var.set(default_map_help(coloration.color_model))
            for c in range(3):
                self.chan_src_vars[c].set(CHAN_SRC_LABELS[coloration.chan_src[c]])
                self.point1_vars[c].set(POINT_LABELS[coloration.point1_choice[c]])
                self.point2_vars[c].set(POINT_LABELS[coloration.point2_choice[c]])
                for i, bvar in enumerate(self.buddha_point_vars[c]):
                    bvar.set(coloration.buddha_points[c][i])
                for k, kvar in enumerate(self.buddha_kind_vars[c]):
                    kvar.set(coloration.buddha_path_kinds[c][k])
                self.buddha_compression_vars[c].set(
                    BUDDHA_COMPRESSION_LABELS[coloration.buddha_compression[c]])
                self.buddha_brightness_vars[c].set(
                    BUDDHA_BRIGHTNESS_OPTIONS[coloration.buddha_brightness[c]])
                self.refresh_row(c)
        finally:
            self._color_editor_loading = False

    def build_colorations_list(self, parent):
        """New/Dupe/Del stacked to the left of a horizontally-scrolling strip
        of coloration thumbnails (a ThumbnailStrip, which the Framing tab's list is too) - visually the Obstacles
        tab's list turned sideways, since (unlike obstacles) colorations are always shown in a fixed
        left-to-right order in an exported image, so browsing them side by side mirrors how they'll actually
        be laid out."""
        self.coloration_strip = ThumbnailStrip(
            parent, THUMB_SIZE, self.small_font, SELECTED_OUTLINE, SELECTED_OUTLINE_PX, self.new_coloration,
            self.duplicate_coloration, self.delete_coloration, self.select_coloration)
        self.coloration_strip.frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))
        self.rebuild_coloration_list()

    def rebuild_coloration_list(self):
        """Show every coloration's thumbnail (of framing 0), the selected one framed. The framings' list is rebuilt
        too: its thumbnails are coloured by coloration 0, and every change that comes here may be one of it."""
        items = [(coloration_id, f"coloration {idx}",
                  encode_png_bytes(self.render_coloration_thumbnail(self.coloration_by_id(coloration_id), THUMB_SIZE)),
                  coloration_id == self.selected_coloration_id)
                 for idx, coloration_id in enumerate(self.all_coloration_ids())]
        self.coloration_strip.rebuild(items)
        if hasattr(self, "framing_strip"):
            self.rebuild_framing_list()

    def build_color_controls(self, parent):
        """The colorations list (New/Dupe/Del + a horizontal strip of coloration
        thumbnails - unlabeled, like the Obstacles tab's list), then the
        "Edit selected coloration" panel: a scrollable list of the color model row
        followed by one block per channel: a Source (Query/Buddha) row, then
        that source's own controls on the row(s) below it. Shows/edits
        whichever coloration is selected in the list above
        (self.selected_coloration_obj()) - see
        refresh_color_controls_for_selected_coloration()."""
        self._color_editor_loading = False
        self.build_colorations_list(parent)

        coloration = self.selected_coloration_obj()
        mp = ttk.LabelFrame(parent, text="Edit selected coloration")
        mp.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))
        mp.columnconfigure(0, weight=1)

        # Everything scrolls together - the color model row plus the three
        # channel blocks - since Buddha's controls make a channel several
        # rows tall and it can all overflow the panel.
        list_container = ttk.Frame(mp)
        list_container.grid(row=0, column=0, sticky="nsew", padx=(8, 2), pady=(6, 4))
        mp.rowconfigure(0, weight=1)
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)
        chan_canvas = tk.Canvas(list_container, height=300, highlightthickness=0)
        chan_vsb = ttk.Scrollbar(list_container, orient="vertical",
                                 command=chan_canvas.yview)
        chan_canvas.configure(yscrollcommand=chan_vsb.set)
        chan_canvas.grid(row=0, column=0, sticky="nsew")
        chan_vsb.grid(row=0, column=1, sticky="ns")
        chan_list_frame = ttk.Frame(chan_canvas)
        chan_window = chan_canvas.create_window((0, 0), window=chan_list_frame, anchor="nw")
        chan_list_frame.bind(
            "<Configure>",
            lambda _e: chan_canvas.configure(scrollregion=chan_canvas.bbox("all")))
        chan_canvas.bind(
            "<Configure>", lambda e: chan_canvas.itemconfigure(chan_window, width=e.width))
        chan_list_frame.columnconfigure(0, weight=1)

        top = ttk.Frame(chan_list_frame)
        top.grid(row=0, column=0, sticky="w", padx=(0, 4), pady=(0, 6))
        ttk.Label(top, text="Color model:").pack(side="left", padx=(0, 10))
        self.color_model_var = tk.StringVar(value=coloration.color_model)
        self.color_model_combo = ttk.Combobox(top, textvariable=self.color_model_var,
                                       values=COLOR_MODEL_NAMES, state="readonly", width=8)
        self.color_model_combo.pack(side="left")
        self.color_model_combo.bind("<<ComboboxSelected>>", self.on_color_model_changed)

        self.comp_labels = []
        self.chan_src_vars = []
        self.chan_src_combos = []
        self.query_frames = []
        self.metric_type_vars = []
        self.metric_type_combos = []
        self.map_vars = []
        self.map_combos = []
        self.point1_vars = []
        self.point1_combos = []
        self.point2_vars = []
        self.point2_combos = []
        self.buddha_frames = []
        self.buddha_point_vars = []
        self.buddha_point_checks = []
        self.buddha_kind_vars = []
        self.buddha_compression_vars = []
        self.buddha_compression_combos = []
        self.buddha_brightness_vars = []
        self.buddha_brightness_combos = []
        self.constant_frames = []
        self.constant_vars = []
        self.constant_combos = []
        self.chan_blocks = []

        for c, (name, color) in enumerate(COLOR_MODELS[coloration.color_model]):
            block = ttk.Frame(chan_list_frame, relief="groove", borderwidth=1)
            block.grid(row=c + 1, column=0, sticky="ew", padx=(0, 4), pady=4)
            block.columnconfigure(2, weight=1)

            lbl = ttk.Label(block, text=name, foreground=color)
            lbl.grid(row=0, column=0, sticky="w", padx=(6, 10), pady=(4, 2))
            ttk.Label(block, text="Source:").grid(row=0, column=1, sticky="w", pady=(4, 2))
            svar = tk.StringVar(value=CHAN_SRC_LABELS[coloration.chan_src[c]])
            scombo = ttk.Combobox(block, textvariable=svar, values=CHAN_SRC_LABELS,
                                  state="readonly", width=9)
            scombo.grid(row=0, column=2, sticky="w", padx=(6, 6), pady=(4, 2))
            scombo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_chan_src_changed(c))

            # ---- Query sub-controls: Type, Metric, Point 1, Point 2 ----
            qf = ttk.Frame(block)
            qf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(6, 6), pady=(0, 4))
            qf.columnconfigure(1, weight=1)
            qf.columnconfigure(3, weight=1)

            metric_type_var = tk.StringVar(value=METRIC_TYPE_LABELS[coloration.metric_types[c]])
            metric_type_combo = ttk.Combobox(qf, textvariable=metric_type_var, values=METRIC_TYPE_LABELS,
                                  state="readonly", width=11)
            metric_type_combo.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
            metric_type_combo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_metric_type_changed(c))

            mvar = tk.StringVar()
            mcombo = ttk.Combobox(qf, textvariable=mvar, state="readonly", width=18)
            mcombo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(0, 6), pady=2)
            mcombo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_metric_selected(c))

            ttk.Label(qf, text="Point 1:").grid(row=1, column=0, sticky="w", pady=2)
            p1var = tk.StringVar(value=POINT_LABELS[coloration.point1_choice[c]])
            p1combo = ttk.Combobox(qf, textvariable=p1var, values=POINT_MENU_LABELS,
                                   state="readonly", width=14)
            p1combo.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=2)
            p1combo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_point_selected(c))

            ttk.Label(qf, text="Point 2:").grid(row=2, column=0, sticky="w", pady=2)
            p2var = tk.StringVar(value=POINT_LABELS[coloration.point2_choice[c]])
            p2combo = ttk.Combobox(qf, textvariable=p2var, values=POINT_MENU_LABELS,
                                   state="readonly", width=14)
            p2combo.grid(row=2, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=2)
            p2combo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_point_selected(c))

            # ---- Buddha sub-controls: Points (multi-select), Compression,
            # brightness multiplier ----
            bf = ttk.Frame(block)
            bf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(6, 6), pady=(0, 4))
            ttk.Label(bf, text="Points:").grid(row=0, column=0, sticky="nw", pady=2)
            pts_grid = ttk.Frame(bf)
            pts_grid.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=2)
            # every stored point is one reduction (a column) of one sequence (a row); the points of the
            # rays a limit ended, and the options that scatter many points per ray, are listed below
            for red in range(N_PRED):
                ttk.Label(pts_grid, text=POINT_REDUCTION_SHORT[red], font=self.small_font).grid(
                    row=0, column=1 + red, padx=2)
            for seq in range(N_SEQS):
                ttk.Label(pts_grid, text=SEQUENCE_SHORT[seq], font=self.small_font).grid(
                    row=1 + seq, column=0, sticky="w", padx=(0, 8))
            bvars = []
            bchecks = []
            for i in range(N_BUDDHA_POINTS):
                bvar = tk.BooleanVar(value=coloration.buddha_points[c][i])
                if i < N_POINTS:
                    seq, red = POINT_OF_SLOT[i]
                    chk = ttk.Checkbutton(pts_grid, variable=bvar,
                                          command=lambda c=c: self.on_buddha_points_changed(c))
                    chk.grid(row=1 + seq, column=1 + red)
                else:
                    chk = ttk.Checkbutton(pts_grid, text=BUDDHA_POINT_LABELS[i], variable=bvar,
                                          command=lambda c=c: self.on_buddha_points_changed(c))
                    chk.grid(row=N_SEQS + 1 + i - N_POINTS, column=0, columnspan=N_PRED + 1, sticky="w")
                bvars.append(bvar)
                bchecks.append(chk)
            kinds_row = ttk.Frame(bf)                    # the paths this channel scatters (all three ticked: every path)
            kinds_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
            ttk.Label(kinds_row, text="Paths:").pack(side="left", padx=(0, 4))
            kvars = []
            for k, label in enumerate(PATH_KIND_LABELS):
                kvar = tk.BooleanVar(value=coloration.buddha_path_kinds[c][k])
                ttk.Checkbutton(kinds_row, text=label, variable=kvar,
                                command=lambda c=c: self.on_buddha_path_kinds_changed(c)).pack(side="left", padx=(0, 8))
                kvars.append(kvar)
            settings_row = ttk.Frame(bf)
            settings_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
            ttk.Label(settings_row, text="Compression:").pack(side="left", padx=(0, 4))
            cvar = tk.StringVar(value=BUDDHA_COMPRESSION_LABELS[coloration.buddha_compression[c]])
            ccombo = ttk.Combobox(settings_row, textvariable=cvar,
                                  values=BUDDHA_COMPRESSION_LABELS, state="readonly", width=11)
            ccombo.pack(side="left", padx=(0, 12))
            ccombo.bind("<<ComboboxSelected>>",
                        lambda _e, c=c: self.on_buddha_compression_changed(c))
            ttk.Label(settings_row, text="Brightness:").pack(side="left", padx=(0, 4))
            brvar = tk.StringVar(value=BUDDHA_BRIGHTNESS_OPTIONS[coloration.buddha_brightness[c]])
            brcombo = ttk.Combobox(settings_row, textvariable=brvar,
                                   values=BUDDHA_BRIGHTNESS_OPTIONS, state="readonly", width=5)
            brcombo.pack(side="left")
            brcombo.bind("<<ComboboxSelected>>",
                        lambda _e, c=c: self.on_buddha_brightness_changed(c))

            # ---- Constant sub-controls: a single dropdown, no simulation
            # data involved at all ----
            cnf = ttk.Frame(block)
            cnf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(6, 6), pady=(0, 4))
            ttk.Label(cnf, text="Value:").pack(side="left", padx=(0, 4))
            const_var = tk.StringVar(
                value=constant_metric_to_label(coloration.constant_choice[c]))
            const_combo = ttk.Combobox(cnf, textvariable=const_var,
                                       values=CHAN_CONSTANT_LABELS, state="readonly", width=9)
            const_combo.pack(side="left")
            const_combo.bind("<<ComboboxSelected>>",
                             lambda _e, c=c: self.on_constant_changed(c))

            self.chan_blocks.append(block)
            self.comp_labels.append(lbl)
            self.chan_src_vars.append(svar)
            self.chan_src_combos.append(scombo)
            self.query_frames.append(qf)
            self.metric_type_vars.append(metric_type_var)
            self.metric_type_combos.append(metric_type_combo)
            self.map_vars.append(mvar)
            self.map_combos.append(mcombo)
            self.point1_vars.append(p1var)
            self.point1_combos.append(p1combo)
            self.point2_vars.append(p2var)
            self.point2_combos.append(p2combo)
            self.buddha_frames.append(bf)
            self.buddha_point_vars.append(bvars)
            self.buddha_point_checks.append(bchecks)
            self.buddha_kind_vars.append(kvars)
            self.buddha_compression_vars.append(cvar)
            self.buddha_compression_combos.append(ccombo)
            self.buddha_brightness_vars.append(brvar)
            self.buddha_brightness_combos.append(brcombo)
            self.constant_frames.append(cnf)
            self.constant_vars.append(const_var)
            self.constant_combos.append(const_combo)
            self.refresh_row(c)

        self.map_help_var = tk.StringVar(value=default_map_help(coloration.color_model))
        ttk.Label(mp, textvariable=self.map_help_var, foreground="#666",
                  wraplength=RIGHT_W - 70, justify="left").grid(
            row=1, column=0, sticky="w", padx=8, pady=(4, 6))
