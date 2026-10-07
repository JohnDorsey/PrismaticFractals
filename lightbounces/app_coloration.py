"""Views and the Coloration tab: the list of views with thumbnails, creating/duplicating/deleting/selecting a view, the colour editor, and the Buddha blur tab (a change recolours at once)."""
import tkinter as tk
from tkinter import ttk
from .constants import (
    DEFAULT_BUDDHA_SEED_CAMERA, DEFAULT_BUDDHA_SEED_X_START, DEFAULT_BUDDHA_SEED_X_STOP,
    DEFAULT_BUDDHA_SEED_Y_START, DEFAULT_BUDDHA_SEED_Y_STOP, BUDDHA_BLUR_LABELS, BUDDHA_BLUR_RADIUS_MAX, BUDDHA_BLUR_RADIUS_MIN,
    DEFAULT_BUDDHA_BLUR_RADIUS, DEFAULT_BUDDHA_BLUR_TYPE, RIGHT_W, SELECTED_OUTLINE,
    SELECTED_OUTLINE_PX, THUMB_SIZE
)
from .image_io import encode_png_bytes
from .parsing import parse_target_coordinate
from .kernels import N_POINTS, N_PRED, N_SEQS
from .metrics import (
    BUDDHA_BRIGHTNESS_OPTIONS, BUDDHA_COMPRESSION_LABELS, BUDDHA_POINT_LABELS, CHAN_CONSTANT_LABELS,
    CHAN_SRC_BUDDHA, CHAN_SRC_LABELS, KIND_LABELS, N_BUDDHA_POINTS, POINT_LABELS, POINT_MENU_LABELS,
    POINT_OF_SLOT, POINT_REDUCTION_SHORT, SEQUENCE_SHORT, constant_stat_to_label, make_mapping
)
from .ui_text import default_map_help
from .views import MODES, MODE_NAMES, ViewConfig


class ColorationMixin:
    # ---- views (Coloration tab) -------------------------------------------- #
    def view_by_id(self, view_id):
        """The live ViewConfig for one view id, or None if it doesn't exist."""
        for v in self.views:
            if v.id == view_id:
                return v
        return None

    def live_view(self):
        """The view the top-left preview shows: whichever view is currently
        selected in the Coloration tab. Selecting a view always shows it
        here - there's no separate "focus" step."""
        return self.selected_view_obj()

    def refresh_live_preview_state(self):
        """Recompute everything that depends on which view is currently live
        (self.mapping, the running Buddha counts, the displayed image) -
        called whenever self.live_view() changes (selecting a different
        view, or creating/deleting one changes the selection)."""
        live = self.live_view()
        self.mapping = make_mapping(live.kinds, live.point_metric, live.other_metric,
                                    live.chan_src, live.constant_choice)
        self.rebuild_buddha_counts()
        self.update_stat_tags()
        self.recolor(0, self.rows_done)
        self.draw()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)

    def selected_view_obj(self):
        return self.view_by_id(self.selected_view_id)

    def all_view_ids(self):
        """Every view's id, in display order (self.views' own list order)."""
        return [v.id for v in self.views]

    def new_view(self):
        """Add a new (default-settings) view right after the selected one,
        and select it - which also makes it the one shown in the preview."""
        v = ViewConfig(self._view_counter)
        self._view_counter += 1
        idx = self.all_view_ids().index(self.selected_view_id)
        self.views.insert(idx + 1, v)
        self.selected_view_id = v.id
        self.view_thumb_cache.pop(v.id, None)
        self.refresh_live_preview_state()
        self.rebuild_view_list()
        self.refresh_color_controls_for_selected_view()

    def duplicate_view(self):
        """A copy of the selected view, inserted right after it, selected,
        and shown in the preview."""
        src = self.selected_view_obj()
        if src is None:
            return
        new_v = ViewConfig(self._view_counter)
        self._view_counter += 1
        for field in ViewConfig.__slots__:
            if field == "id":
                continue
            val = getattr(src, field)
            setattr(new_v, field, list(val) if isinstance(val, list) else val)
        idx = self.all_view_ids().index(self.selected_view_id)
        self.views.insert(idx + 1, new_v)
        self.selected_view_id = new_v.id
        self.view_thumb_cache.pop(new_v.id, None)
        self.refresh_live_preview_state()
        self.rebuild_view_list()
        self.refresh_color_controls_for_selected_view()

    def delete_view(self):
        """Removes the selected view, unless it's the only one left. The
        deleted view was always the one shown in the preview (selection and
        preview are the same thing), so self.views[0] becomes the new
        selection, and the preview is refreshed to match it right away."""
        if len(self.views) <= 1:
            return
        v = self.view_by_id(self.selected_view_id)
        if v is None:
            return
        self.views.remove(v)
        self.view_thumb_cache.pop(v.id, None)
        self.purge_keyframe_settings(f"view:{v.id}:")
        self.selected_view_id = self.views[0].id
        self.view_thumb_cache.pop(self.selected_view_id, None)
        self.refresh_live_preview_state()
        self.rebuild_view_list()
        self.refresh_color_controls_for_selected_view()

    def select_view(self, view_id):
        """Selects a view for editing - which also makes it the one the
        top-left panel shows."""
        if view_id == self.selected_view_id:
            return
        self.selected_view_id = view_id
        self.view_thumb_cache.pop(view_id, None)   # about to become live (and possibly
                                                    # edited) - any cached thumbnail from
                                                    # before must not be reused once it's
                                                    # non-live again
        self.refresh_live_preview_state()
        self.rebuild_view_list()
        self.refresh_color_controls_for_selected_view()

    def refresh_color_controls_for_selected_view(self):
        """Populate every Color mapping control from the selected view (or
        leave them as-is if the Coloration tab hasn't been built yet, or no
        view is selected). Mirrors refresh_obstacle_editor."""
        if not hasattr(self, "mode_var"):
            return                                  # Coloration tab not built yet
        view = self.selected_view_obj()
        if view is None:
            return
        self._color_editor_loading = True
        try:
            self.mode_var.set(view.mode)
            for c, (name, color) in enumerate(MODES[view.mode]):
                self.comp_labels[c].configure(text=name, foreground=color)
            self.map_help_var.set(default_map_help(view.mode))
            for c in range(3):
                self.chan_src_vars[c].set(CHAN_SRC_LABELS[view.chan_src[c]])
                self.point1_vars[c].set(POINT_LABELS[view.point1_choice[c]])
                self.point2_vars[c].set(POINT_LABELS[view.point2_choice[c]])
                for i, bvar in enumerate(self.buddha_point_vars[c]):
                    bvar.set(view.buddha_points[c][i])
                self.buddha_compression_vars[c].set(
                    BUDDHA_COMPRESSION_LABELS[view.buddha_compression[c]])
                self.buddha_brightness_vars[c].set(
                    BUDDHA_BRIGHTNESS_OPTIONS[view.buddha_brightness[c]])
                self.refresh_row(c)
        finally:
            self._color_editor_loading = False

    def build_views_list(self, parent):
        """New/Dupe/Del stacked to the left of a horizontally-scrolling strip
        of view thumbnails - visually the Obstacles tab's list turned
        sideways, since (unlike obstacles) views are always shown in a fixed
        left-to-right order in an exported image, so browsing them side by
        side mirrors how they'll actually be laid out."""
        top_row = ttk.Frame(parent)
        top_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))
        top_row.columnconfigure(1, weight=1)

        btn_col = ttk.Frame(top_row)
        btn_col.grid(row=0, column=0, sticky="n", padx=(0, 6))
        ttk.Button(btn_col, text="New", width=6, command=self.new_view).pack(
            fill="x", pady=(0, 3))
        ttk.Button(btn_col, text="Dupe", width=6, command=self.duplicate_view).pack(
            fill="x", pady=(0, 3))
        self.view_del_btn = ttk.Button(btn_col, text="Del", width=6, command=self.delete_view)
        self.view_del_btn.pack(fill="x")

        list_container = ttk.Frame(top_row)
        list_container.grid(row=0, column=1, sticky="ew")
        list_container.columnconfigure(0, weight=1)
        view_canvas = tk.Canvas(list_container, height=THUMB_SIZE + 34, highlightthickness=0)
        view_hsb = ttk.Scrollbar(list_container, orient="horizontal",
                                 command=view_canvas.xview)
        view_canvas.configure(xscrollcommand=view_hsb.set)
        view_canvas.grid(row=0, column=0, sticky="ew")
        view_hsb.grid(row=1, column=0, sticky="ew")
        self.view_list_frame = ttk.Frame(view_canvas)
        view_canvas.create_window((0, 0), window=self.view_list_frame, anchor="nw")
        self.view_list_frame.bind(
            "<Configure>",
            lambda _e: view_canvas.configure(scrollregion=view_canvas.bbox("all")))
        # (unlike the vertical obstacle list, the inner frame's WIDTH is never
        # pinned to the canvas's - it needs to grow past it, since scrolling
        # sideways to reach it is the whole point)

        self.rebuild_view_list()

    def rebuild_view_list(self):
        for w in self.view_list_frame.winfo_children():
            w.destroy()
        for idx, view_id in enumerate(self.all_view_ids()):
            selected = view_id == self.selected_view_id
            cell = tk.Frame(self.view_list_frame,
                            highlightthickness=SELECTED_OUTLINE_PX if selected else 0,
                            highlightbackground=SELECTED_OUTLINE,
                            highlightcolor=SELECTED_OUTLINE)
            cell.grid(row=0, column=idx, sticky="n", padx=2, pady=1)

            view_obj = self.view_by_id(view_id)
            thumb = tk.PhotoImage(
                data=encode_png_bytes(self.render_view_thumbnail(view_obj, THUMB_SIZE)),
                format="png")
            thumb_lbl = tk.Label(cell, image=thumb, bd=0, cursor="hand2")
            thumb_lbl.image = thumb              # keep a reference alive
            thumb_lbl.grid(row=0, column=0, padx=2, pady=(2, 0))
            thumb_lbl.bind("<Button-1>", lambda _e, vid=view_id: self.select_view(vid))

            lbl = tk.Label(cell, text=f"view {idx}", cursor="hand2", font=self.small_font)
            lbl.grid(row=1, column=0, pady=(0, 2))
            lbl.bind("<Button-1>", lambda _e, vid=view_id: self.select_view(vid))

        if hasattr(self, "view_del_btn"):
            self.view_del_btn.state(
                ["disabled"] if len(self.views) <= 1 else ["!disabled"])

    def build_color_controls(self, parent):
        """The views list (New/Dupe/Del + a horizontal strip of view
        thumbnails - unlabeled, like the Obstacles tab's list), then the
        "Edit selected view" panel: a scrollable list of the color model row
        followed by one block per channel: a Source (Query/Buddha) row, then
        that source's own controls on the row(s) below it. Shows/edits
        whichever view is selected in the list above
        (self.selected_view_obj()) - see
        refresh_color_controls_for_selected_view()."""
        self._color_editor_loading = False
        self.build_views_list(parent)

        view = self.selected_view_obj()
        mp = ttk.LabelFrame(parent, text="Edit selected view")
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
        self.mode_var = tk.StringVar(value=view.mode)
        self.mode_combo = ttk.Combobox(top, textvariable=self.mode_var,
                                       values=MODE_NAMES, state="readonly", width=8)
        self.mode_combo.pack(side="left")
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)

        self.comp_labels = []
        self.chan_src_vars = []
        self.chan_src_combos = []
        self.query_frames = []
        self.kind_vars = []
        self.kind_combos = []
        self.map_vars = []
        self.map_combos = []
        self.point1_vars = []
        self.point1_combos = []
        self.point2_vars = []
        self.point2_combos = []
        self.buddha_frames = []
        self.buddha_point_vars = []
        self.buddha_point_checks = []
        self.buddha_compression_vars = []
        self.buddha_compression_combos = []
        self.buddha_brightness_vars = []
        self.buddha_brightness_combos = []
        self.constant_frames = []
        self.constant_vars = []
        self.constant_combos = []
        self.chan_blocks = []

        for c, (name, color) in enumerate(MODES[view.mode]):
            block = ttk.Frame(chan_list_frame, relief="groove", borderwidth=1)
            block.grid(row=c + 1, column=0, sticky="ew", padx=(0, 4), pady=4)
            block.columnconfigure(2, weight=1)

            lbl = ttk.Label(block, text=name, foreground=color)
            lbl.grid(row=0, column=0, sticky="w", padx=(6, 10), pady=(4, 2))
            ttk.Label(block, text="Source:").grid(row=0, column=1, sticky="w", pady=(4, 2))
            svar = tk.StringVar(value=CHAN_SRC_LABELS[view.chan_src[c]])
            scombo = ttk.Combobox(block, textvariable=svar, values=CHAN_SRC_LABELS,
                                  state="readonly", width=9)
            scombo.grid(row=0, column=2, sticky="w", padx=(6, 6), pady=(4, 2))
            scombo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_chan_src_changed(c))

            # ---- Query sub-controls: Type, Metric, Point 1, Point 2 ----
            qf = ttk.Frame(block)
            qf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(6, 6), pady=(0, 4))
            qf.columnconfigure(1, weight=1)
            qf.columnconfigure(3, weight=1)

            kvar = tk.StringVar(value=KIND_LABELS[view.kinds[c]])
            kcombo = ttk.Combobox(qf, textvariable=kvar, values=KIND_LABELS,
                                  state="readonly", width=11)
            kcombo.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
            kcombo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_kind_changed(c))

            mvar = tk.StringVar()
            mcombo = ttk.Combobox(qf, textvariable=mvar, state="readonly", width=18)
            mcombo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(0, 6), pady=2)
            mcombo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_metric_selected(c))

            ttk.Label(qf, text="Point 1:").grid(row=1, column=0, sticky="w", pady=2)
            p1var = tk.StringVar(value=POINT_LABELS[view.point1_choice[c]])
            p1combo = ttk.Combobox(qf, textvariable=p1var, values=POINT_MENU_LABELS,
                                   state="readonly", width=14)
            p1combo.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(4, 0), pady=2)
            p1combo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_point_selected(c))

            ttk.Label(qf, text="Point 2:").grid(row=2, column=0, sticky="w", pady=2)
            p2var = tk.StringVar(value=POINT_LABELS[view.point2_choice[c]])
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
                bvar = tk.BooleanVar(value=view.buddha_points[c][i])
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
            settings_row = ttk.Frame(bf)
            settings_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
            ttk.Label(settings_row, text="Compression:").pack(side="left", padx=(0, 4))
            cvar = tk.StringVar(value=BUDDHA_COMPRESSION_LABELS[view.buddha_compression[c]])
            ccombo = ttk.Combobox(settings_row, textvariable=cvar,
                                  values=BUDDHA_COMPRESSION_LABELS, state="readonly", width=11)
            ccombo.pack(side="left", padx=(0, 12))
            ccombo.bind("<<ComboboxSelected>>",
                        lambda _e, c=c: self.on_buddha_compression_changed(c))
            ttk.Label(settings_row, text="Brightness:").pack(side="left", padx=(0, 4))
            brvar = tk.StringVar(value=BUDDHA_BRIGHTNESS_OPTIONS[view.buddha_brightness[c]])
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
                value=constant_stat_to_label(view.constant_choice[c]))
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
            self.kind_vars.append(kvar)
            self.kind_combos.append(kcombo)
            self.map_vars.append(mvar)
            self.map_combos.append(mcombo)
            self.point1_vars.append(p1var)
            self.point1_combos.append(p1combo)
            self.point2_vars.append(p2var)
            self.point2_combos.append(p2combo)
            self.buddha_frames.append(bf)
            self.buddha_point_vars.append(bvars)
            self.buddha_point_checks.append(bchecks)
            self.buddha_compression_vars.append(cvar)
            self.buddha_compression_combos.append(ccombo)
            self.buddha_brightness_vars.append(brvar)
            self.buddha_brightness_combos.append(brcombo)
            self.constant_frames.append(cnf)
            self.constant_vars.append(const_var)
            self.constant_combos.append(const_combo)
            self.refresh_row(c)

        self.map_help_var = tk.StringVar(value=default_map_help(view.mode))
        ttk.Label(mp, textvariable=self.map_help_var, foreground="#666",
                  wraplength=RIGHT_W - 70, justify="left").grid(
            row=1, column=0, sticky="w", padx=8, pady=(4, 6))

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
        self.last_buddha_seed = {}
        for row, (field, label, default) in enumerate(
                (("x_start", "X start", DEFAULT_BUDDHA_SEED_X_START),
                 ("x_stop", "X stop", DEFAULT_BUDDHA_SEED_X_STOP),
                 ("y_start", "Y start", DEFAULT_BUDDHA_SEED_Y_START),
                 ("y_stop", "Y stop", DEFAULT_BUDDHA_SEED_Y_STOP))):
            var = tk.StringVar(value=str(default))
            setattr(self, f"buddha_seed_{field}_var", var)
            self.last_buddha_seed[field] = default
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
        state = ["disabled"] if self.buddha_seed_camera_var.get() else ["!disabled"]
        for widget in self.buddha_seed_entries + self.buddha_seed_labels:
            widget.state(state)

    def on_buddha_seed_camera_changed(self):
        self.on_buddha_seeds_changed()

    def on_buddha_seed_commit(self, _event=None):
        """Enter pressed or focus left a zone entry: tidy its text (an invalid one goes back to the last valid
        value) and, if the zone changed, recolour."""
        for field in self.last_buddha_seed:
            var = getattr(self, f"buddha_seed_{field}_var")
            value = parse_target_coordinate(var.get())
            var.set(str(self.last_buddha_seed[field] if value is None else value))
        self.on_buddha_seeds_changed()

    def on_buddha_seeds_changed(self):
        """The seeding changed (by hand, or loaded from a keyframe or a project): remember the zone's valid values,
        refresh what shows it now, and recolour - which takes a moment when a Buddha channel has to trace its
        rays again - once the changes of this moment have all arrived."""
        for field in self.last_buddha_seed:
            value = parse_target_coordinate(getattr(self, f"buddha_seed_{field}_var").get())
            if value is not None:
                self.last_buddha_seed[field] = value
        self.update_buddha_seed_state()
        self.update_seed_zone_outline()
        self.refresh_timeline_diffs()
        if self.seed_job is None:
            self.seed_job = self.root.after_idle(self.refresh_buddha_seeds)

    def refresh_buddha_seeds(self):
        """Scatter every Buddha channel again from the seeds as they now are, and recolour (like refresh_blur, but
        the scattered counts themselves are new). Costs nothing when no view has a Buddha channel."""
        self.seed_job = None
        self.view_thumb_cache = {}
        if not any(src == CHAN_SRC_BUDDHA for v in self.views for src in v.chan_src):
            return
        self.rebuild_buddha_counts()
        self.recolor(0, self.rows_done)
        self.draw()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)
        self.rebuild_view_list()

    def build_buddha_blur_controls(self, parent):
        """Radius slider and blur-type dropdown for Buddha coloration's
        point-scattering blur - one setting for the whole image (a
        BuddhaBlurSettings, beside the ray simulation's settings and not part of
        them), not a per-view/per-channel one, so it lives in its own tab rather
        than inside the Coloration tab's per-channel Buddha controls. A change
        recolors the image at once (the blur is applied to the stored scatter
        counts, never to the simulation, so no Start is needed)."""
        rf = ttk.LabelFrame(parent, text="Radius")
        rf.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 6))
        rf.columnconfigure(1, weight=1)
        self.buddha_blur_radius_scale = self.add_angle_entry(
            rf, 0, "Radius", DEFAULT_BUDDHA_BLUR_RADIUS, on_change=self.on_buddha_blur_changed,
            lo=BUDDHA_BLUR_RADIUS_MIN, hi=BUDDHA_BLUR_RADIUS_MAX, fmt="{:.2f}",
            exact=True)
        ttk.Label(rf, foreground="#666", wraplength=RIGHT_W - 70,
                  text="A multiple of the view's smallest dimension (2 units): 0.1 "
                       "means a blur radius of 0.2 units. 0 means no blur at all - "
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
        channel's cached blurred counts and every non-live view's cached thumbnail are for
        the old blur; the live view is recolored, redrawn and its thumbnail rebuilt. Costs
        nothing visible when no view has a Buddha channel."""
        self.blur_job = None
        self.buddha_counts_blurred = [None] * 3
        self.view_thumb_cache = {}
        if not any(src == CHAN_SRC_BUDDHA for v in self.views for src in v.chan_src):
            return
        self.recolor(0, self.rows_done)
        self.draw()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)
        self.rebuild_view_list()
