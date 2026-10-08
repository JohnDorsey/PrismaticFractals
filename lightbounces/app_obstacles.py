"""The Obstacles tab: the list, the new... menu, the per-kind parameter controls, and creating, selecting, editing, duplicating and deleting obstacles (and which of them are pending since the last Start)."""
import tkinter as tk
from tkinter import ttk
from .constants import (
    DEFAULT_ROTATION, OBSTACLE_EFFECT_NAMES, PENDING_OUTLINE, RAY_HELP_TEXT, RIGHT_W,
    SELECTED_OUTLINE, SELECTED_OUTLINE_PX, THUMB_SIZE
)
from .image_io import encode_png_bytes
from .kernels import OBSTACLE_EFFECT_PASS, OBSTACLE_EFFECT_REFLECT
from .shapes import POSITION_PARAMS, SHAPE_CLASSES, Shape, render_obstacle_thumbnail


class ObstaclesMixin:
    def build_obstacles_controls(self, parent):
        """Obstacle list (scrollable, with "new..." - a button opening a menu of the shape kinds -
        Dupe and Del stacked to its left) plus an editor for whichever obstacle is
        selected: its shape's name (fixed when it was made), a rotation, the
        controls of its own shape kind (one group per kind, built from that
        kind's PARAMS table; only the selected obstacle's is shown), a polar
        position (r, theta) with an x/y offset, incoming and outgoing ray-effect
        dropdowns, and a "fire rays inside" checkbox. Every one of an obstacle's
        settings can be animated between keyframes, just like the main
        Emissions/Termination/Mutations/Coloration settings. `self.obs_widgets` maps
        (Shape, field) to the control of a field every kind has, and (a kind,
        field) to the control of one of that kind's own."""
        self._obst_editor_loading = False
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        top_row = ttk.Frame(parent)
        top_row.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        top_row.columnconfigure(1, weight=1)
        top_row.rowconfigure(0, weight=1)

        btn_col = ttk.Frame(top_row)
        btn_col.grid(row=0, column=0, sticky="n", padx=(0, 6))
        # An ordinary button (so it matches Dupe and Del; a ttk Menubutton is shorter than
        # a Button) that opens the menu of kinds just below itself.
        new_button = ttk.Button(btn_col, text="new...", width=6)
        new_menu = tk.Menu(new_button, tearoff=0)
        for shape_class in SHAPE_CLASSES:
            new_menu.add_command(label=shape_class.NAME,
                                 command=lambda cls=shape_class: self.new_obstacle(cls))
        new_button.configure(command=lambda: self.show_new_obstacle_menu(new_button))
        new_button.pack(fill="x", pady=(0, 3))
        self.obs_new_menu = new_menu
        ttk.Button(btn_col, text="Dupe", width=6, command=self.duplicate_obstacle).pack(
            fill="x", pady=(0, 3))
        ttk.Button(btn_col, text="Del", width=6, command=self.delete_obstacle).pack(
            fill="x")

        list_container = ttk.Frame(top_row)
        list_container.grid(row=0, column=1, sticky="nsew")
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)
        obst_canvas = tk.Canvas(list_container, height=130, highlightthickness=0)
        obst_vsb = ttk.Scrollbar(list_container, orient="vertical",
                                 command=obst_canvas.yview)
        obst_canvas.configure(yscrollcommand=obst_vsb.set)
        obst_canvas.grid(row=0, column=0, sticky="nsew")
        obst_vsb.grid(row=0, column=1, sticky="ns")
        self.obst_list_frame = ttk.Frame(obst_canvas)
        obst_window = obst_canvas.create_window((0, 0), window=self.obst_list_frame,
                                                anchor="nw")
        self.obst_list_frame.bind(
            "<Configure>",
            lambda _e: obst_canvas.configure(scrollregion=obst_canvas.bbox("all")))
        obst_canvas.bind(
            "<Configure>", lambda e: obst_canvas.itemconfigure(obst_window, width=e.width))

        editor = ttk.LabelFrame(parent, text="Selected obstacle")
        editor.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))
        editor.columnconfigure(0, weight=1)

        top = ttk.Frame(editor)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))
        ttk.Label(top, text="Shape:").pack(side="left")
        self.obs_shape_label = ttk.Label(top, text="")
        self.obs_shape_label.pack(side="left", padx=(8, 0))

        self.obs_widgets = {}
        rotg = ttk.Frame(editor)
        rotg.grid(row=1, column=0, sticky="ew")
        rotg.columnconfigure(1, weight=1)
        self.obs_widgets[(Shape, "rotation")] = self.add_angle_entry(
            rotg, 0, "Rotation (\u00b0)", DEFAULT_ROTATION,
            on_change=self.on_obstacle_param_changed)

        self.obs_shape_groups = {}
        for shape_class in SHAPE_CLASSES:
            group = ttk.Frame(editor)
            group.grid(row=2, column=0, sticky="ew")
            group.columnconfigure(1, weight=1)
            for row, p in enumerate(shape_class.PARAMS):
                self.obs_widgets[(shape_class, p.field)] = self.add_shape_param_row(group, row, p)
            self.obs_shape_groups[shape_class] = group

        pos_group = ttk.Frame(editor)
        pos_group.grid(row=3, column=0, sticky="ew")
        pos_group.columnconfigure((0, 1), weight=1)
        r_param, theta_param, cx_param, cy_param = POSITION_PARAMS
        self.obs_widgets[(Shape, "r")] = self.add_shape_param_row(pos_group, 0, r_param)
        self.obs_widgets[(Shape, "theta")] = self.add_angle_entry(
            pos_group, 1, "Angle (\u03b8, \u00b0)", theta_param.default,
            on_change=self.on_obstacle_param_changed)
        (self.obs_widgets[(Shape, "cx")], self.obs_widgets[(Shape, "cy")]) = self.add_slider_pair(
            pos_group, 2,
            *[(p.label, p.lo, p.hi, p.step, p.default, p.slider_fmt)
              for p in (cx_param, cy_param)],
            on_change=self.on_obstacle_param_changed)

        in_row = ttk.Frame(editor)
        in_row.grid(row=4, column=0, sticky="ew", padx=8, pady=(4, 2))
        ttk.Label(in_row, text="Incoming ray effect:", width=20).pack(side="left")
        self.obs_in_effect_var = tk.StringVar(
            value=OBSTACLE_EFFECT_NAMES[OBSTACLE_EFFECT_REFLECT])
        self.obs_in_effect_combo = ttk.Combobox(in_row, textvariable=self.obs_in_effect_var,
                                                values=OBSTACLE_EFFECT_NAMES,
                                                state="readonly", width=12)
        self.obs_in_effect_combo.pack(side="left", padx=(8, 0))
        self.obs_in_effect_combo.bind("<<ComboboxSelected>>", self.on_obstacle_effect_changed)
        self.obs_widgets[(Shape, "in_effect")] = self.obs_in_effect_combo

        out_row = ttk.Frame(editor)
        out_row.grid(row=5, column=0, sticky="ew", padx=8, pady=(2, 2))
        ttk.Label(out_row, text="Outgoing ray effect:", width=20).pack(side="left")
        self.obs_out_effect_var = tk.StringVar(
            value=OBSTACLE_EFFECT_NAMES[OBSTACLE_EFFECT_REFLECT])
        self.obs_out_effect_combo = ttk.Combobox(out_row, textvariable=self.obs_out_effect_var,
                                                 values=OBSTACLE_EFFECT_NAMES,
                                                 state="readonly", width=12)
        self.obs_out_effect_combo.pack(side="left", padx=(8, 0))
        self.obs_out_effect_combo.bind("<<ComboboxSelected>>", self.on_obstacle_effect_changed)
        self.obs_widgets[(Shape, "out_effect")] = self.obs_out_effect_combo

        self.obs_fire_inside_var = tk.BooleanVar(value=True)
        self.obs_fire_inside_check = ttk.Checkbutton(
            editor, text="Fire rays inside", variable=self.obs_fire_inside_var,
            command=self.on_obstacle_fire_inside_changed)
        self.obs_fire_inside_check.grid(row=6, column=0, sticky="w", padx=8, pady=2)
        self.obs_widgets[(Shape, "fire_inside")] = self.obs_fire_inside_check

        ttk.Label(parent, foreground="#666", wraplength=RIGHT_W - 70,
                  text="Every one of an obstacle's settings can be saved "
                       "into keyframes and animated, just like the other "
                       "tabs - but the set of obstacles that exist is fixed "
                       "for a whole render, taken from whichever obstacles "
                       "exist when Start is pressed."
                 ).grid(row=2, column=0, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(parent, foreground="#666", wraplength=RIGHT_W - 70,
                  text=RAY_HELP_TEXT).grid(row=3, column=0, sticky="w", padx=8, pady=(0, 8))

        self.show_obstacle_shape_controls()
        self.refresh_obstacle_editor()
        self.rebuild_obstacle_list()

    def show_new_obstacle_menu(self, button):
        """Open the menu of shape kinds just below `button`."""
        try:
            self.obs_new_menu.tk_popup(button.winfo_rootx(),
                                       button.winfo_rooty() + button.winfo_height())
        finally:
            self.obs_new_menu.grab_release()

    def add_shape_param_row(self, parent, row, p):
        """The control for one ShapeParam: a slider row, or an angle entry when the
        parameter has no range."""
        if p.lo is None:
            return self.add_angle_entry(parent, row, f"{p.label} (\u00b0)", p.default,
                                        on_change=self.on_obstacle_param_changed)
        label = f"{p.label} (\u00b0)" if p.fmt == "angle" else p.label
        return self.add_slider_row(parent, row, label, p.lo, p.hi, p.step, p.default,
                                   p.slider_fmt, on_change=self.on_obstacle_param_changed)

    def obstacle_widget(self, shape_class, field):
        """The editor's control for `field` of an obstacle of kind `shape_class`
        (None if there is none)."""
        widget = self.obs_widgets.get((shape_class, field))
        return widget if widget is not None else self.obs_widgets.get((Shape, field))

    def show_obstacle_shape_controls(self):
        """Show the control group of the selected obstacle's kind, and hide the others."""
        o = self.selected_obstacle
        shown = type(o) if o is not None else None
        for shape_class, group in self.obs_shape_groups.items():
            if shape_class is shown:
                group.grid()
            else:
                group.grid_remove()

    def refresh_obstacle_editor(self):
        """Populate the editor controls from the selected obstacle (or leave
        them as-is, disabled-by-convention, if none is selected). Every numeric
        field is set table-driven off the obstacle's own parameter tables."""
        o = self.selected_obstacle
        if o is None:
            return
        self._obst_editor_loading = True
        try:
            self.obs_shape_label.configure(text=o.NAME)
            for p in o.NUMERIC_PARAMS:
                self.obstacle_widget(type(o), p.field).set(getattr(o, p.field))
            self.obs_in_effect_var.set(OBSTACLE_EFFECT_NAMES[o.in_effect])
            self.obs_out_effect_var.set(OBSTACLE_EFFECT_NAMES[o.out_effect])
            self.obs_fire_inside_var.set(o.fire_inside)
            self.show_obstacle_shape_controls()
        finally:
            self._obst_editor_loading = False

    def new_obstacle(self, shape_class):
        """Add an obstacle of kind `shape_class` (the kind is fixed from now on)."""
        o = shape_class(self._obst_counter)
        self._obst_counter += 1
        self.obstacles.append(o)
        self.selected_obstacle = o
        self.rebuild_obstacle_list()
        self.refresh_obstacle_editor()
        self.on_shape_param_changed()             # obstacles affect the image too

    def select_obstacle(self, o):
        self.selected_obstacle = o
        self.rebuild_obstacle_list()
        self.refresh_obstacle_editor()

    def rebuild_obstacle_list(self):
        for w in self.obst_list_frame.winfo_children():
            w.destroy()
        for idx, o in enumerate(self.obstacles):
            selected = o is self.selected_obstacle
            row = tk.Frame(self.obst_list_frame,
                          highlightthickness=SELECTED_OUTLINE_PX if selected else 0,
                          highlightbackground=SELECTED_OUTLINE,
                          highlightcolor=SELECTED_OUTLINE)
            row.grid(row=idx, column=0, sticky="ew", pady=2, padx=1)
            row.columnconfigure(1, weight=1)

            thumb = tk.PhotoImage(
                data=encode_png_bytes(render_obstacle_thumbnail(o, THUMB_SIZE)), format="png")
            thumb_lbl = tk.Label(row, image=thumb, bd=0, cursor="hand2")
            thumb_lbl.image = thumb                  # keep a reference alive
            thumb_lbl.grid(row=0, column=0, padx=(2, 6), pady=2)
            thumb_lbl.bind("<Button-1>", lambda _e, o=o: self.select_obstacle(o))

            lbl = tk.Label(row, text=f"#{idx + 1}  {o.describe()}",
                          anchor="w", cursor="hand2", font=self.small_font)
            lbl.grid(row=0, column=1, sticky="ew")
            lbl.bind("<Button-1>", lambda _e, o=o: self.select_obstacle(o))

    def on_obstacle_param_changed(self):
        if self._obst_editor_loading:
            return
        o = self.selected_obstacle
        if o is None:
            return
        for p in o.NUMERIC_PARAMS:
            raw = self.obstacle_widget(type(o), p.field).get()
            setattr(o, p.field,
                   int(round(float(raw))) if p.fmt == "int" else round(float(raw), 2))
        self.rebuild_obstacle_list()
        self.on_shape_param_changed()

    def on_obstacle_effect_changed(self, _event=None):
        if self._obst_editor_loading:
            return
        o = self.selected_obstacle
        if o is None:
            return
        o.in_effect = OBSTACLE_EFFECT_NAMES.index(self.obs_in_effect_var.get())
        o.out_effect = OBSTACLE_EFFECT_NAMES.index(self.obs_out_effect_var.get())
        self.obs_in_effect_combo.selection_clear()
        self.obs_out_effect_combo.selection_clear()
        self.rebuild_obstacle_list()
        self.on_shape_param_changed()

    def on_obstacle_fire_inside_changed(self):
        if self._obst_editor_loading:
            return
        o = self.selected_obstacle
        if o is None:
            return
        o.fire_inside = bool(self.obs_fire_inside_var.get())
        self.on_shape_param_changed()

    def duplicate_obstacle(self):
        o = self.selected_obstacle
        if o is None:
            return
        new_o = o.duplicate(self._obst_counter)
        self._obst_counter += 1
        idx = self.obstacles.index(o)
        self.obstacles.insert(idx + 1, new_o)
        self.selected_obstacle = new_o
        self.rebuild_obstacle_list()
        self.refresh_obstacle_editor()
        self.on_shape_param_changed()

    def delete_obstacle(self):
        o = self.selected_obstacle
        if o is None:
            return
        self.obstacles.remove(o)
        self.selected_obstacle = None
        self.purge_keyframe_settings(f"obst:{o.id}:")
        self.rebuild_obstacle_list()
        self.on_shape_param_changed()

    def obstacle_is_pending(self, o):
        """Should `o` get the dashed editing guide? Yes if it does nothing to
        rays (passes them through both ways - regardless of whether it
        changed), is new since the last Start, or its settings differ from
        what was last generated."""
        if o.in_effect == OBSTACLE_EFFECT_PASS and o.out_effect == OBSTACLE_EFFECT_PASS:
            return True
        frozen = self.gen_obstacles_by_id.get(o.id)
        return frozen is None or o.state() != frozen

    def update_pending_outline(self):
        """Dashed outline of every obstacle that is pending: doing nothing, new,
        or different from what's actually been generated."""
        self.canvas.delete("pending")
        if not self.deco["pending"].get():
            return
        g = self.generated_settings
        if g is None:
            return                                 # nothing generated yet: nothing "pending"
        for o in self.obstacles:
            if self.obstacle_is_pending(o):
                self.canvas.create_line(*o.outline_coords(self.image_world_rect, self.img_w, self.img_h),
                                        fill=PENDING_OUTLINE, dash=(5, 4), tags="pending")
