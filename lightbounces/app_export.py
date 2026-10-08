"""Exporting a still image: the Export designer's save choices, Save / Save As for the current image, and generating a new image at its own resolution on a background thread."""
import os
import threading
from tkinter import filedialog, messagebox
from .export_text import SAVE_SOURCE_OPTIONS, apply_info_locations, image_filename
from .image_io import encode_and_write
from .parsing import parse_resolution_dimension
from .paths import ensure_export_dir
from .settings import Keyframe, resolve_frame_settings, resolve_obstacles
from .shapes import build_obstacle_arrays
from .colorations import resolve_colorations, snapshot_coloration
from .framings import resolve_framing


class ExportMixin:
    # ---- saving ------------------------------------------------------------ #
    def on_save_source_changed(self, _event=None):
        if self.save_source_var.get() == SAVE_SOURCE_OPTIONS[1]:   # "New image"
            self.save_new_frame.grid()
        else:
            self.save_new_frame.grid_remove()

    def on_save_format_changed(self, _event=None):
        if self.save_format_var.get() == "jxl":
            self.save_jxl_controls.frame.pack(side="left")
        else:
            self.save_jxl_controls.frame.pack_forget()

    def save(self):
        if self.save_source_var.get() == SAVE_SOURCE_OPTIONS[1]:
            self.save_new_image(None)
        else:
            fmt = self.save_format_var.get()
            self._write(os.path.join(ensure_export_dir(), image_filename(
                self.generated_settings, self.generated_natural_termination, fmt,
                self.save_jxl_mode_var.get() == "Lossless",
                int(round(float(self.save_jxl_quality_scale.get()))))))

    def save_as(self):
        if self.save_source_var.get() == SAVE_SOURCE_OPTIONS[1]:
            self.save_new_image_as()
        else:
            fmt = self.save_format_var.get()
            label = {"png": "PNG image", "qoi": "QOI image", "jxl": "JPEG XL image"}[fmt]
            path = filedialog.asksaveasfilename(
                parent=self.root, title="Save image as",
                initialdir=ensure_export_dir(),
                initialfile=image_filename(
                    self.generated_settings, self.generated_natural_termination, fmt,
                    self.save_jxl_mode_var.get() == "Lossless",
                    int(round(float(self.save_jxl_quality_scale.get())))),
                defaultextension="." + fmt, filetypes=[(label, f"*.{fmt}")])
            if path:
                self._write(path)

    def colorations_to_export(self):
        """Which colorations to include in an export, per the Export designer's
        "Include all colorations" checkbox: all of them, in list/composition
        order, or just the one currently selected/shown in the preview."""
        if self.include_all_colorations_var.get():
            return list(self.colorations)
        return [self.live_coloration()]

    def _write(self, path):
        fmt = self.save_format_var.get()
        jxl_lossless = self.save_jxl_mode_var.get() == "Lossless"
        jxl_quality = int(round(float(self.save_jxl_quality_scale.get())))
        colorations = self.colorations_to_export()
        # self.frame already holds exactly the single-coloration case (with the
        # outline if that decoration is on, no preview overlays) - only
        # recompute when more than one coloration needs to be composed in.
        rgb = self.frame if len(colorations) == 1 else self.render_current_image(colorations)
        rgb = apply_info_locations(
            rgb, colorations, rgb.shape[1] // len(colorations), self.generated_settings,
            self.generated_natural_termination, self.gen_obstacle_snapshots, self.color_info_location_var.get(),
            self.figure_info_location_var.get(), self.image_framing, self.framing_info_location_var.get(),
            mutate_ok=len(colorations) > 1)   # len(colorations) > 1 means rgb is already a fresh
                                       # array from render_current_image, not self.frame
        try:
            encode_and_write(rgb, path, fmt, jxl_lossless, jxl_quality)
        except Exception as exc:                   # noqa: BLE001
            messagebox.showerror("Save failed", f"Could not save:\n{path}\n\n{exc}")
            return

    def save_new_image_as(self):
        """Ask for a path (before generating anything, so cancelling costs
        nothing), then generate and save a "New image" there."""
        fmt = self.save_format_var.get()
        label = {"png": "PNG image", "qoi": "QOI image", "jxl": "JPEG XL image"}[fmt]
        initial = image_filename(
            self.generated_settings, self.generated_natural_termination, fmt,
            self.save_jxl_mode_var.get() == "Lossless",
            int(round(float(self.save_jxl_quality_scale.get()))))
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save new image as",
            initialdir=ensure_export_dir(), initialfile=initial,
            defaultextension="." + fmt, filetypes=[(label, f"*.{fmt}")])
        if path:
            self.save_new_image(path)

    def save_new_image(self, path_or_none):
        """Generate a fresh image at the New-figure controls' own resolution
        (per coloration - with "Include all colorations" on, each coloration is rendered at
        this size and they're placed side by side) from the CURRENT (live)
        settings - the same per-frame rendering path video export uses,
        treating the current state as a single one-off "keyframe" - and save
        it in the chosen format. Runs on a background thread (like video
        rendering) since generating a frame at a large resolution isn't
        instant, unlike saving the already-rendered preview."""
        w = parse_resolution_dimension(self.save_new_w_var.get(), max_digits=6)
        h = parse_resolution_dimension(self.save_new_h_var.get(), max_digits=6)
        if w is None or h is None:
            messagebox.showerror("Invalid resolution",
                                 "Enter a width and height of at least 1 pixel.")
            return
        settings_dict = self.capture_all_settings()
        if settings_dict is None:
            return
        fmt = self.save_format_var.get()
        jxl_lossless = self.save_jxl_mode_var.get() == "Lossless"
        jxl_quality = int(round(float(self.save_jxl_quality_scale.get())))
        fake_kf = Keyframe(-1, 0, settings_dict)
        data = resolve_frame_settings([fake_kf], 0)
        obst_snapshots = [o.snapshot() for o in self.obstacles]
        obst_values = resolve_obstacles(obst_snapshots, [fake_kf], 0)
        obst_arrays = build_obstacle_arrays(obst_values)
        coloration_snapshots = [snapshot_coloration(v) for v in self.colorations_to_export()]
        colorations = resolve_colorations(coloration_snapshots, [fake_kf], 0)
        framing = resolve_framing(self.selected_framing_snapshot(), [fake_kf], 0)
        color_location = self.color_info_location_var.get()
        figure_location = self.figure_info_location_var.get()
        framing_location = self.framing_info_location_var.get()
        if path_or_none is None:
            path = os.path.join(ensure_export_dir(),
                                image_filename(data.ray, data.natural_termination, fmt, jxl_lossless,
                                               jxl_quality))
        else:
            path = path_or_none
        self.save_btn.state(["disabled"])
        self.save_as_btn.state(["disabled"])
        thread = threading.Thread(
            target=self._save_new_image_worker,
            args=(data, w, h, obst_arrays, obst_values, colorations, framing, fmt, jxl_lossless,
                 jxl_quality, path, color_location, figure_location, framing_location),
            daemon=True)
        thread.start()

    def _save_new_image_worker(self, data, w, h, obst_arrays, obst_values, colorations, framing, fmt,
                               jxl_lossless, jxl_quality, path, color_location,
                               figure_location, framing_location):
        """Runs on a background thread: never touches a Tk widget directly,
        only via self.root.after(0, ...)."""
        try:
            rgb = self.render_full_image(data, w, h, obst_arrays, colorations, framing)
            rgb = apply_info_locations(rgb, colorations, w, data.ray, data.natural_termination,
                                       obst_values, color_location, figure_location,
                                       framing, framing_location, mutate_ok=True)
            encode_and_write(rgb, path, fmt, jxl_lossless, jxl_quality)
        except Exception as exc:                    # noqa: BLE001
            self.root.after(0, lambda m=str(exc): self._save_new_image_failed(path, m))
            return
        self.root.after(0, lambda: self._save_new_image_done(path))

    def _save_new_image_done(self, path):
        self.save_btn.state(["!disabled"])
        self.save_as_btn.state(["!disabled"])

    def _save_new_image_failed(self, path, message):
        self.save_btn.state(["!disabled"])
        self.save_as_btn.state(["!disabled"])
        messagebox.showerror("Save failed", f"Could not save:\n{path}\n\n{message}")
