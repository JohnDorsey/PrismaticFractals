"""Building the window: the image, controls, info and status quadrants, the decorations tab and its handler."""
import tkinter as tk
from tkinter import ttk
import numpy as np
from .constants import (
    BOTTOM_H, DECO_IMAGE, DECO_PREVIEW, GUIDE_RGBA, KIND_COLOR, PAD, RIGHT_W, SEG_CANVAS_H,
    SEG_CANVAS_W
)
from .export_text import COLOR_INFO_LOCATIONS, FIGURE_INFO_LOCATIONS, SAVE_SOURCE_OPTIONS
from .image_io import SAVE_NEW_FORMATS, lerp_color, png_rgba
from .kernels import (
    KIND_BOUNDARY, KIND_ESCAPED, KIND_OBSTACLE_TERM, KIND_SELF, KIND_START, N_STATS, SIZE
)
from .metrics import STAT_SHORT
from .widgets import JxlQualityControls, Tooltip, make_scrollable_tab


class LayoutMixin:
    # ---- layout ------------------------------------------------------------ #
    def build_image_quadrant(self):
        """Top left: the image. Hover overlay lives on the canvas, not in the image."""
        self.canvas = tk.Canvas(self.root, width=SIZE, height=SIZE, bd=0,
                                highlightthickness=0, bg="black",
                                cursor="crosshair")
        self.canvas.grid(row=0, column=0, padx=(PAD, PAD // 2), pady=(PAD, PAD // 2))
        self.photo = tk.PhotoImage(width=SIZE, height=SIZE)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

        # One-pixel-thick images with a uniform alpha give true semi-transparent
        # guide lines (Tk canvas lines have no alpha of their own).
        self.hline_img = None
        self.vline_img = None
        try:
            self.hline_img = tk.PhotoImage(
                data=png_rgba(SIZE, 1, GUIDE_RGBA), format="png")
            self.vline_img = tk.PhotoImage(
                data=png_rgba(1, SIZE, GUIDE_RGBA), format="png")
        except tk.TclError:
            self.hline_img = self.vline_img = None   # fall back to stippled lines

    def build_controls_quadrant(self):
        """Top middle: four tabs of controls, with Start / Revert below them."""
        ctl = ttk.LabelFrame(self.root, text="Controls", width=RIGHT_W, height=SIZE)
        ctl.grid(row=0, column=1, sticky="nsew",
                 padx=(PAD // 2, PAD // 2), pady=(PAD, PAD // 2))
        ctl.grid_propagate(False)
        ctl.columnconfigure(0, weight=1)
        ctl.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(ctl)
        notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=(6, 4))
        sim_tab = ttk.Frame(notebook)
        fuzz_tab = ttk.Frame(notebook)
        color_tab = ttk.Frame(notebook)
        buddha_blur_tab = ttk.Frame(notebook)
        deco_tab = ttk.Frame(notebook)
        obst_tab = ttk.Frame(notebook)
        notebook.add(sim_tab, text="Simulation")
        notebook.add(fuzz_tab, text="Fuzzing")
        notebook.add(color_tab, text="Coloration")
        notebook.add(buddha_blur_tab, text="Buddha blur")
        notebook.add(deco_tab, text="Decorations")
        notebook.add(obst_tab, text="Obstacles")
        for tab in (sim_tab, fuzz_tab, color_tab, buddha_blur_tab, deco_tab, obst_tab):
            tab.columnconfigure(0, weight=1)

        # The Simulation and Obstacles tabs hold more than fits in their height
        # (and how much depends on the user's font size and display scaling), so
        # they scroll, like the Coloration tab's channel editor.
        self.sim_canvas, sim_inner = make_scrollable_tab(sim_tab)
        self.build_limits_controls(sim_inner)
        self.build_segment_mult_controls(sim_inner)
        self.build_confinement_controls(sim_inner)
        self.build_emission_controls(sim_inner)
        self.build_fuzzing_controls(fuzz_tab)
        self.build_color_controls(color_tab)
        self.build_buddha_blur_controls(buddha_blur_tab)
        self.build_decoration_controls(deco_tab)
        self.obst_canvas, obst_inner = make_scrollable_tab(obst_tab)
        self.build_obstacles_controls(obst_inner)

        buttons = ttk.Frame(ctl)
        buttons.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 10))
        buttons.columnconfigure((0, 1), weight=1)
        self.start_btn = ttk.Button(buttons, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.revert_btn = ttk.Button(buttons, text="Revert settings",
                                     command=self.revert_settings)
        self.revert_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def build_decoration_controls(self, parent):
        """Checkboxes for the things drawn on / over the image, in two groups."""
        groups = [("Affect the output image", DECO_IMAGE,
                   "Drawn into the image itself, so they also appear in saved "
                   "files and keyframes."),
                  ("Preview only (never saved)", DECO_PREVIEW,
                   "Drawn over the image on screen while you hover or edit "
                   "settings; saved files and keyframes never contain them.")]
        for g, (title, items, note) in enumerate(groups):
            frame = ttk.LabelFrame(parent, text=title)
            frame.grid(row=g, column=0, sticky="ew", padx=6,
                       pady=(8 if g == 0 else 0, 6))
            frame.columnconfigure(0, weight=1)
            for r, (key, label, default) in enumerate(items):
                var = tk.BooleanVar(value=default)
                check = ttk.Checkbutton(
                    frame, text=label, variable=var,
                    command=lambda k=key: self.on_decoration_changed(k))
                check.grid(row=r, column=0, sticky="w", padx=10, pady=2)
                setattr(self, f"deco_{key}_check", check)
                self.deco[key] = var
            ttk.Label(frame, text=note, foreground="#666",
                      wraplength=RIGHT_W - 70).grid(
                row=len(items), column=0, sticky="w", padx=10, pady=(2, 6))

    def build_info_quadrant(self):
        """Bottom left: the previewed ray, in Numbers and Graphics tabs."""
        info = ttk.LabelFrame(self.root, text="Ray preview", width=SIZE, height=BOTTOM_H)
        info.grid(row=1, column=0, sticky="nsew",
                  padx=(PAD, PAD // 2), pady=(PAD // 2, PAD))
        info.grid_propagate(False)
        info.columnconfigure(0, weight=1)
        info.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(info)
        notebook.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        numbers_tab = ttk.Frame(notebook)
        graphics_tab = ttk.Frame(notebook)
        notebook.add(numbers_tab, text="Numbers")
        notebook.add(graphics_tab, text="Graphics")
        numbers_tab.columnconfigure(0, weight=1)
        graphics_tab.columnconfigure(0, weight=1)

        self.build_numbers_tab(numbers_tab)
        self.build_graphics_tab(graphics_tab)

    def build_numbers_tab(self, tab):
        """Every detail and metric of the previewed ray."""
        self.info_placeholder = ttk.Label(
            tab, foreground="#666",
            text="Hover the mouse over a pixel inside the shape to preview its "
                 "ray. Click to pin it there and free up the mouse.")
        self.info_placeholder.grid(row=0, column=0, sticky="nw", padx=6, pady=8)

        rows = ttk.Frame(tab)
        rows.grid(row=0, column=0, sticky="nw", padx=2, pady=2)
        self.info_rows = rows
        right = ttk.Frame(rows)
        right.grid(row=0, column=0, sticky="nw", padx=(4, 8))
        left = ttk.Frame(rows)
        left.grid(row=0, column=1, sticky="nw")

        fields = [("Start", "start"), ("Outcome", "outcome"),
                  ("Bounces", "bounces"), ("Final", "final"),
                  ("Length", "length"), ("Last seg", "lastseg"),
                  ("Self-loop", "loop"), ("Components", "components"),
                  ("Color", "color")]
        self.info_vars = {}
        for r, (label, key) in enumerate(fields):
            ttk.Label(left, text=label + ":", foreground="#666").grid(
                row=r, column=0, sticky="nw", padx=(0, 8))
            var = tk.StringVar()
            self.info_vars[key] = var
            if key == "color":
                holder = ttk.Frame(left)
                holder.grid(row=r, column=1, sticky="w")
                self.swatch = tk.Label(holder, width=4, bd=1, relief="solid")
                self.swatch.pack(side="left")
                ttk.Label(holder, textvariable=var, wraplength=130).pack(
                    side="left", padx=8)
            else:
                ttk.Label(left, textvariable=var, wraplength=205).grid(
                    row=r, column=1, sticky="w")

        # Metric list: name | value | "-> R" tag for channels currently using it
        self.stat_vars = []
        self.stat_tags = []
        for s in range(N_STATS):
            ttk.Label(right, text=STAT_SHORT[s], font=self.small_font).grid(
                row=s, column=0, sticky="w", padx=(0, 10))
            var = tk.StringVar()
            self.stat_vars.append(var)
            ttk.Label(right, textvariable=var, font=self.small_font).grid(
                row=s, column=1, sticky="e")
            tag = ttk.Label(right, text="", font=self.small_font, width=8)
            tag.grid(row=s, column=2, sticky="w", padx=(6, 0))
            self.stat_tags.append(tag)
        rows.grid_remove()
        self.update_stat_tags()

    def build_graphics_tab(self, tab):
        """A plot of the previewed ray's segment lengths, earliest on the left."""
        legend = ttk.Frame(tab)
        legend.grid(row=0, column=0, sticky="w", padx=6, pady=(8, 2))
        ttk.Label(legend, text="Every segment, earliest to latest, as a bar "
                              "(one unit tall; longer segments are truncated). "
                              "Color marks each end's kind: "
                  ).pack(side="left")
        for rgb, text in [(KIND_COLOR[KIND_START], "start/distance-cut"),
                          (KIND_COLOR[KIND_BOUNDARY], "boundary bounce"),
                          (KIND_COLOR[KIND_SELF], "self bounce"),
                          (KIND_COLOR[KIND_OBSTACLE_TERM], "edge termination"),
                          (KIND_COLOR[KIND_ESCAPED], "escape")]:
            sw = tk.Label(legend, text="  ", bg=lerp_color(rgb, rgb, 0.0))
            sw.pack(side="left", padx=(4, 2))
            ttk.Label(legend, text=text).pack(side="left", padx=(0, 8))
        self.seg_canvas = tk.Canvas(tab, width=SEG_CANVAS_W, height=SEG_CANVAS_H,
                                    bd=0, highlightthickness=1,
                                    highlightbackground="#444", bg="black")
        self.seg_canvas.grid(row=1, column=0, sticky="w", padx=6, pady=(2, 8))

    def draw_segments(self, pts, kinds, n):
        """Redraw the segment-lengths graphic for the ray pts[0..n-1], bars
        growing downward from the top, gradient-colored by each end's kind."""
        cv = self.seg_canvas
        cv.delete("all")
        nseg = n - 1
        if nseg <= 0:
            return
        px_per_unit = SEG_CANVAS_H              # canvas height = 1 world unit
        colw = SEG_CANVAS_W / nseg
        halfw = max(colw / 2.0, 0.75)
        for k in range(nseg):
            length = float(np.hypot(pts[k + 1, 0] - pts[k, 0], pts[k + 1, 1] - pts[k, 1]))
            h = min(length * px_per_unit, SEG_CANVAS_H)
            if h <= 0:
                continue
            x = (k + 0.5) * colw
            top_rgb = KIND_COLOR[float(kinds[k])]
            bot_rgb = KIND_COLOR[float(kinds[k + 1])]
            if top_rgb == bot_rgb:
                cv.create_rectangle(x - halfw, 0, x + halfw, h,
                                    fill=lerp_color(top_rgb, top_rgb, 0.0), width=0)
                continue
            steps = max(2, min(64, int(h)))
            for s in range(steps):
                t0 = s / steps
                t1 = (s + 1) / steps
                color = lerp_color(top_rgb, bot_rgb, (t0 + t1) / 2.0)
                cv.create_rectangle(x - halfw, t0 * h, x + halfw, t1 * h,
                                    fill=color, width=0)

    def build_status_quadrant(self):
        """Bottom middle: the Export designer (whether to include one view
        or all of them in exported output), then the Export controls (what
        to export, its format, and the Export / Export As buttons)."""
        container = ttk.Frame(self.root, width=RIGHT_W, height=BOTTOM_H)
        container.grid(row=1, column=1, sticky="nsew",
                       padx=(PAD // 2, PAD // 2), pady=(PAD // 2, PAD))
        container.grid_propagate(False)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        vcmd_wide = (self.root.register(self.validate_bitrate_digits), "%P")

        # ---- Export designer: which view(s) end up in the output ----
        designer = ttk.LabelFrame(container, text="Export designer")
        designer.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.include_all_views_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(designer, text="Include all views",
                        variable=self.include_all_views_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))

        color_info_row = ttk.Frame(designer)
        color_info_row.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(color_info_row, text="Color info location:").pack(side="left", padx=(0, 6))
        self.color_info_location_var = tk.StringVar(value=COLOR_INFO_LOCATIONS[0])
        ttk.Combobox(color_info_row, textvariable=self.color_info_location_var,
                    values=COLOR_INFO_LOCATIONS, state="readonly", width=13).pack(side="left")

        figure_info_row = ttk.Frame(designer)
        figure_info_row.grid(row=2, column=0, sticky="w", padx=8, pady=(2, 6))
        ttk.Label(figure_info_row, text="Figure settings location:").pack(
            side="left", padx=(0, 6))
        self.figure_info_location_var = tk.StringVar(value=FIGURE_INFO_LOCATIONS[0])
        ttk.Combobox(figure_info_row, textvariable=self.figure_info_location_var,
                    values=FIGURE_INFO_LOCATIONS, state="readonly", width=13).pack(side="left")

        # ---- Export: what to export, its format, and the buttons ----
        exp = ttk.LabelFrame(container, text="Export")
        exp.grid(row=1, column=0, sticky="nsew")
        exp.columnconfigure(0, weight=1)

        # Export buttons are packed at the bottom FIRST, so they always get
        # their space no matter how tall everything else below ends up
        # being (a fixed-height panel plus content that can grow otherwise
        # risks the classic "everything after the tall content gets
        # squeezed out" grid/pack failure mode). Disabled (with a tooltip
        # explaining why) until a figure has actually finished generating -
        # see update_buttons().
        self.save_frame = ttk.Frame(exp)
        self.save_frame.columnconfigure((0, 1), weight=1)
        self.save_btn = ttk.Button(self.save_frame, text="Export", command=self.save)
        self.save_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.save_as_btn = ttk.Button(self.save_frame, text="Export As...",
                                      command=self.save_as)
        self.save_as_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.save_frame.pack(side="bottom", fill="x", padx=10, pady=(6, 8))
        self.save_btn_tooltip = Tooltip(self.save_btn)
        self.save_as_btn_tooltip = Tooltip(self.save_as_btn)

        # Everything else stacks above it, top to bottom
        rest = ttk.Frame(exp)
        rest.pack(side="top", fill="both", expand=True)
        rest.columnconfigure(0, weight=1)

        # ---- what to export and in what format (both apply whichever
        # source is chosen) ----
        source_row = ttk.Frame(rest)
        source_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 1))
        ttk.Label(source_row, text="Image to export:").pack(side="left", padx=(0, 6))
        self.save_source_var = tk.StringVar(value=SAVE_SOURCE_OPTIONS[0])
        save_source_combo = ttk.Combobox(
            source_row, textvariable=self.save_source_var, values=SAVE_SOURCE_OPTIONS,
            state="readonly", width=19)
        save_source_combo.pack(side="left")
        save_source_combo.bind("<<ComboboxSelected>>", self.on_save_source_changed)

        fmt_row = ttk.Frame(rest)
        fmt_row.grid(row=1, column=0, sticky="ew", padx=10, pady=1)
        ttk.Label(fmt_row, text="Format:").pack(side="left", padx=(0, 4))
        self.save_format_var = tk.StringVar(value=SAVE_NEW_FORMATS[0])
        format_combo = ttk.Combobox(fmt_row, textvariable=self.save_format_var,
                                    values=SAVE_NEW_FORMATS, state="readonly", width=6)
        format_combo.pack(side="left", padx=(0, 10))
        format_combo.bind("<<ComboboxSelected>>", self.on_save_format_changed)
        self.save_jxl_controls = JxlQualityControls(fmt_row)
        self.save_jxl_mode_var = self.save_jxl_controls.mode_var
        self.save_jxl_quality_scale = self.save_jxl_controls.scale
        self.on_save_format_changed()          # (the default format is jxl)

        # "New image" only: its own resolution (an existing preview already
        # has one, so this only makes sense when generating a fresh image) -
        # this is per view: with "Include all views" on, each view is
        # rendered at this resolution and they're placed side by side, so
        # the final output is wider than this.
        self.save_new_frame = ttk.Frame(rest)
        self.save_new_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(1, 8))
        dims_row = ttk.Frame(self.save_new_frame)
        dims_row.grid(row=0, column=0, sticky="w")
        self.save_new_w_var = tk.StringVar(value="1080")
        self.save_new_h_var = tk.StringVar(value="1080")
        for label, var in [("Width (px):", self.save_new_w_var),
                           ("Height (px):", self.save_new_h_var)]:
            ttk.Label(dims_row, text=label).pack(side="left", padx=(0, 4))
            entry = ttk.Entry(dims_row, textvariable=var, width=6, justify="right",
                              validate="key", validatecommand=vcmd_wide)
            entry.pack(side="left", padx=(0, 10))
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, lambda _e, var=var: self.commit_resolution_var(var))
        self.save_new_frame.grid_remove()

    # ---- decorations -------------------------------------------------------- #
    def on_decoration_changed(self, key):
        """A decoration checkbox was toggled."""
        self.hover_px = None                      # redraw any overlay on next hover
        if key == "outline":                      # part of the image itself
            self.draw()
            self.refresh_timeline_diffs()
        elif key == "pending":
            self.update_pending_outline()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)
