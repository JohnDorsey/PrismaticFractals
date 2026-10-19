"""The Video panel: render settings, rendering every frame on a background thread, downsampling, encoding with ffmpeg (and cjxl for animated JPEG XL), re-encoding existing frames, progress and clearing the temporary frames."""
import os
import re
import shutil
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from .engines import correct_frame, correction_message
from .export_text import VIDEO_BITRATE_MODES, VIDEO_FORMATS, apply_info_locations
from .external import cjxl_available
from .image_io import (
    DOWNSAMPLE_FACTORS, DOWNSAMPLE_FACTOR_VALUES, DOWNSAMPLE_METHODS, convert_apng_to_jxl,
    downsample_image, jxl_quality_tag, read_png, write_png
)
from .parsing import parse_frame_number, parse_positive_int, parse_resolution_dimension
from .paths import ensure_export_dir
from .settings import resolve_frame_settings, resolve_obstacles, snapshot_keyframes
from .shapes import build_obstacle_arrays
from .colorations import resolve_colorations, snapshot_coloration
from .framings import resolve_framing
from .widgets import HelpButton, JxlQualityControls
from . import external, paths


def frames_to_render(keyframe_indices, starting_frame):
    """(first, last) frame to render: from the starting frame (but never before the first keyframe,
    since the frames before it would only repeat its values) to the last keyframe. None when the
    starting frame is past the last keyframe, so there is nothing to render."""
    first = max(min(keyframe_indices), starting_frame)
    last = max(keyframe_indices)
    return (first, last) if first <= last else None


def frame_file_offset(first_frame, number_from_start):
    """The number of the first frame's file: the frame's own number when frame files are numbered from
    the starting frame, else 0 (ffmpeg reads a sequence from any number: -start_number)."""
    return first_frame if number_from_start else 0


class VideoMixin:
    def build_video_panel(self):
        """The Video tab (`self.video_tab`, made by build_status_quadrant), in rows: resolution and which frames to render, the
        encoding (framerate, format, downsampling, bitrate), the render buttons with the progress bar and a "?" button that holds
        the long explanation, and below them a preview of the latest rendered frame beside the status box."""
        vp = self.video_tab
        vcmd_wide = (self.root.register(self.validate_bitrate_digits), "%P")

        # ---- Render settings: resolution and which frames to render ----
        dims_row = ttk.Frame(vp)
        dims_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 2))
        self.video_w_var = tk.StringVar(value="1080")
        self.video_h_var = tk.StringVar(value="1080")
        for label, var, width in [("Width (px):", self.video_w_var, 6),
                                  ("Height (px):", self.video_h_var, 6)]:
            ttk.Label(dims_row, text=label).pack(side="left", padx=(0, 4))
            entry = ttk.Entry(dims_row, textvariable=var, width=width, justify="right",
                              validate="key", validatecommand=vcmd_wide)
            entry.pack(side="left", padx=(0, 10))
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, lambda _e, var=var: self.commit_resolution_var(var))
        self.start_frame_var = tk.StringVar(value="0")
        ttk.Label(dims_row, text="Starting frame:").pack(side="left", padx=(0, 4))
        ttk.Entry(dims_row, textvariable=self.start_frame_var, width=7, justify="right",
                  validate="key", validatecommand=vcmd_wide).pack(side="left")
        self.number_files_from_start_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(vp, text="Number frame files from the starting frame",
                        variable=self.number_files_from_start_var).grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 2))

        # ---- Encoding: framerate, format, downsampling, bitrate ----
        fmt_row = ttk.Frame(vp)
        fmt_row.grid(row=2, column=0, sticky="ew", padx=8, pady=2)
        self.fps_var = tk.StringVar(value="30")
        ttk.Label(fmt_row, text="FPS:").pack(side="left", padx=(0, 4))
        ttk.Entry(fmt_row, textvariable=self.fps_var, width=5, justify="right",
                 validate="key", validatecommand=vcmd_wide).pack(side="left", padx=(0, 10))
        ttk.Label(fmt_row, text="Format:").pack(side="left", padx=(0, 4))
        self.export_format_var = tk.StringVar(value="mp4 (h264)")
        format_combo = ttk.Combobox(fmt_row, textvariable=self.export_format_var,
                                    values=VIDEO_FORMATS,
                                    state="readonly", width=6)
        format_combo.pack(side="left", padx=(0, 10))
        format_combo.bind("<<ComboboxSelected>>", self.on_export_format_changed)
        ttk.Label(fmt_row, text="Downsample:").pack(side="left", padx=(0, 4))
        self.downsample_factor_var = tk.StringVar(value=DOWNSAMPLE_FACTORS[0])
        downsample_combo = ttk.Combobox(
            fmt_row, textvariable=self.downsample_factor_var,
            values=DOWNSAMPLE_FACTORS, state="readonly", width=9)
        downsample_combo.pack(side="left", padx=(0, 10))
        downsample_combo.bind("<<ComboboxSelected>>", self.on_downsample_factor_changed)
        self.downsample_method_label = ttk.Label(fmt_row, text="Method:")
        self.downsample_method_label.pack(side="left", padx=(0, 4))
        self.downsample_method_var = tk.StringVar(value=DOWNSAMPLE_METHODS[0])
        self.downsample_method_combo = ttk.Combobox(
            fmt_row, textvariable=self.downsample_method_var,
            values=DOWNSAMPLE_METHODS, state="readonly", width=7)
        self.downsample_method_combo.pack(side="left")

        mode_line = ttk.Frame(vp)
        mode_line.grid(row=3, column=0, sticky="ew", padx=8, pady=2)
        self.bitrate_mode_row = ttk.Frame(mode_line)
        self.bitrate_mode_row.pack(side="left")
        ttk.Label(self.bitrate_mode_row, text="Bitrate mode:").pack(
            side="left", padx=(0, 4))
        self.bitrate_mode_var = tk.StringVar(value="Constant quality")
        mode_combo = ttk.Combobox(
            self.bitrate_mode_row, textvariable=self.bitrate_mode_var,
            values=VIDEO_BITRATE_MODES,
            state="readonly", width=15)
        mode_combo.pack(side="left", padx=(0, 10))
        mode_combo.bind("<<ComboboxSelected>>", self.on_bitrate_mode_changed)

        self.video_jxl_controls = JxlQualityControls(mode_line)

        self.quality_row = ttk.Frame(self.bitrate_mode_row)
        self.quality_scale = self.add_slider_row(
            self.quality_row, 0, "Quality", 0, 100, 5, 75, "{:.0f}",
            on_change=lambda: None)

        self.bitrate_row = ttk.Frame(self.bitrate_mode_row)
        ttk.Label(self.bitrate_row, text="Bitrate (kbps):").pack(side="left", padx=(0, 4))
        self.bitrate_var = tk.StringVar(value="512")
        ttk.Entry(self.bitrate_row, textvariable=self.bitrate_var, width=8,
                 justify="right", validate="key", validatecommand=vcmd_wide).pack(
            side="left")

        # Variable bitrate: target and maximum on the same line
        self.vbr_row = ttk.Frame(self.bitrate_mode_row)
        self.vbr_target_var = tk.StringVar(value="512")
        self.vbr_max_var = tk.StringVar(value="1024")
        for label, var in [("Target (kbps):", self.vbr_target_var),
                           ("Max (kbps):", self.vbr_max_var)]:
            ttk.Label(self.vbr_row, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(self.vbr_row, textvariable=var, width=7, justify="right",
                     validate="key", validatecommand=vcmd_wide).pack(
                side="left", padx=(0, 10))

        self.on_bitrate_mode_changed()
        self.on_export_format_changed()
        self.on_downsample_factor_changed()

        # ---- the buttons, the progress bar and the "?" that holds the long explanation ----
        render_row = ttk.Frame(vp)
        render_row.grid(row=4, column=0, sticky="ew", padx=8, pady=(4, 4))
        render_row.columnconfigure(3, weight=1)
        self.render_btn = ttk.Button(render_row, text="Start Render",
                                     command=self.start_or_cancel_render)
        self.render_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.reencode_btn = ttk.Button(render_row, text="Re-encode",
                                       command=self.start_reencode)
        self.reencode_btn.grid(row=0, column=1, sticky="ew", padx=3)
        self.clear_btn = ttk.Button(render_row, text="Clear",
                                    command=self.clear_temp_frames)
        self.clear_btn.grid(row=0, column=2, sticky="ew", padx=3)
        self.video_progress_style = self.make_labeled_progress_style()
        self.video_progress = ttk.Progressbar(render_row, maximum=1, mode="determinate",
                                              **({"style": self.video_progress_style} if self.video_progress_style else {}))
        self.video_progress.grid(row=0, column=3, sticky="ew", padx=(3, 3))
        self.video_help = HelpButton(render_row, self.video_help_text())
        self.video_help.grid(row=0, column=4, padx=(3, 0))

        # ---- a preview of the latest rendered frame, beside the status box ----
        lower = ttk.Frame(vp)
        lower.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 6))
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)
        vp.rowconfigure(5, weight=1)

        self.preview_label = tk.Label(lower, bd=1, relief="solid")
        self.preview_label.grid(row=0, column=0, sticky="n", padx=(0, 8))
        self.set_preview_image(None)

        status_frame = ttk.Frame(lower)
        status_frame.grid(row=0, column=1, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        self.video_status_text = tk.Text(status_frame, height=6, wrap="word",
                                         font=self.small_font, state="disabled",
                                         bd=1, relief="solid")
        status_vsb = ttk.Scrollbar(status_frame, orient="vertical",
                                   command=self.video_status_text.yview)
        self.video_status_text.configure(yscrollcommand=status_vsb.set)
        self.video_status_text.grid(row=0, column=0, sticky="nsew")
        status_vsb.grid(row=0, column=1, sticky="ns")
        self.set_video_status("" if external.ffmpeg is not None else
                              "ffmpeg-python was not found - install it with:  pip install ffmpeg-python")

    def video_help_text(self):
        """What the "?" button of the Video tab shows."""
        def near(path):                                      # (relative to the program, unless that is on another drive)
            try:
                return os.path.relpath(path, paths.SCRIPT_DIR).replace(os.sep, "/")
            except ValueError:
                return path.replace(os.sep, "/")
        temp, exports = near(paths.TEMP_DIR), near(paths.EXPORT_DIR)
        note = "" if external.ffmpeg is not None else ("\nffmpeg-python was not found - install "
                                                      "it with:  pip install ffmpeg-python")
        return (f"Frames are written as PNGs to {temp}/ next to this program at the resolution "
                "above (same world area as the viewport), then combined with ffmpeg into an "
                f".mp4, .gif or .apng saved in {exports}/. Downsampling "
                "shrinks only that combined video, not the PNG sequence. "
                "Re-encode combines whatever frames are already in TEMP with "
                "the current encoding settings (ignoring width/height - it "
                "uses the existing frames' own resolution). Nothing is ever "
                "deleted automatically - clear TEMP yourself between "
                f"renders.{note}")

    def on_downsample_factor_changed(self, _event=None):
        """The Method dropdown only matters once downsampling is actually on."""
        if self.downsample_factor_var.get() == DOWNSAMPLE_FACTORS[0]:
            self.downsample_method_label.state(["disabled"])
            self.downsample_method_combo.state(["disabled"])
        else:
            self.downsample_method_label.state(["!disabled"])
            self.downsample_method_combo.state(["!disabled"])

    # ---- video rendering ------------------------------------------------------ #
    def set_video_status(self, text):
        """Replace the scrollable render-status text (so long messages, like a
        full error, are never cut off - just scroll)."""
        self.video_status_text.configure(state="normal")
        self.video_status_text.delete("1.0", "end")
        self.video_status_text.insert("1.0", text)
        self.video_status_text.configure(state="disabled")

    def append_video_status(self, text):
        """Add a line to the render-status text and scroll to it. The text is cleared only when a new export starts (D21), so
        everything one export said stays to be read."""
        self.video_status_text.configure(state="normal")
        if self.video_status_text.get("1.0", "end-1c"):
            self.video_status_text.insert("end", "\n")
        self.video_status_text.insert("end", text)
        self.video_status_text.see("end")
        self.video_status_text.configure(state="disabled")

    def begin_video_status(self, text):
        """A new export (a render or a re-encode) starts: the status text starts again with `text`, and the progress bar from nothing."""
        self.set_video_status(text)
        self.set_video_progress_text("")

    def make_labeled_progress_style(self):
        """A style of progress bar with a text over it (the frame count), or None where the theme cannot make one (the count is then a
        line of the status text for every frame)."""
        try:
            style = ttk.Style()
            name = "Labeled.Horizontal.TProgressbar"
            style.layout(name, [("Horizontal.Progressbar.trough", {"children": [("Horizontal.Progressbar.pbar",
                                                                                  {"side": "left", "sticky": "ns"})],
                                                                    "sticky": "nswe"}),
                                ("Horizontal.Progressbar.label", {"sticky": "nswe"})])
            style.configure(name, text="", anchor="center")
            return name
        except tk.TclError:
            return None

    def set_video_progress_text(self, text):
        """The frame count (or whatever `text` says) over the progress bar; without a labelled bar, a new line of the status text."""
        if self.video_progress_style:
            ttk.Style().configure(self.video_progress_style, text=text)
        elif text:
            self.append_video_status(f"Rendering frames: {text}")

    def describe_output_file(self, out_path, num_frames, fps):
        """'X.XX MB, N.NN Mbps (T.TTs)' for a just-written output file - size
        read straight off disk, bitrate computed from that actual size and
        the video's duration (num_frames / fps), not the bitrate that was
        requested when encoding: those can differ, especially for GIF/APNG
        (which ignore the bitrate settings entirely) or a quality/CRF-based
        MP4 encode (which doesn't target a specific bitrate at all)."""
        try:
            size_bytes = os.path.getsize(out_path)
        except OSError:
            return ""
        size_mb = size_bytes / (1024.0 * 1024.0)
        if fps <= 0 or num_frames <= 0:
            return f"{size_mb:.2f} MB"
        duration_s = num_frames / fps
        bitrate_bps = size_bytes * 8.0 / duration_s
        if bitrate_bps >= 1_000_000.0:
            bitrate_text = f"{bitrate_bps / 1_000_000.0:.2f} Mbps"
        else:
            bitrate_text = f"{bitrate_bps / 1000.0:.0f} kbps"
        return f"{size_mb:.2f} MB, {bitrate_text} ({duration_s:.2f}s)"

    def on_export_format_changed(self, _event=None):
        fmt = self.export_format_var.get()
        self.bitrate_mode_row.pack_forget()
        self.video_jxl_controls.frame.pack_forget()
        if fmt == "mp4 (h264)":
            self.bitrate_mode_row.pack(side="left")
        elif fmt == "JXL":
            self.video_jxl_controls.frame.pack(side="left")

    def on_bitrate_mode_changed(self, _event=None):
        mode = self.bitrate_mode_var.get()
        self.quality_row.pack_forget()
        self.bitrate_row.pack_forget()
        self.vbr_row.pack_forget()
        if mode == "Constant quality":
            self.quality_row.pack(side="left")
        elif mode == "Target bitrate":
            self.bitrate_row.pack(side="left")
        else:
            self.vbr_row.pack(side="left")

    def ffmpeg_output_kwargs(self, fmt, mode, quality, bitrate_kbps, vbr_target_kbps,
                             vbr_max_kbps):
        """The ffmpeg.output() keyword arguments for the chosen export format
        and (for MP4) bitrate mode."""
        if fmt in ("GIF", "APNG", "JXL"):
            return {}                # let ffmpeg's defaults for these formats handle it -
                                     # JXL's ffmpeg stage produces an intermediate APNG,
                                     # so it uses the exact same defaults as APNG itself
        if fmt == "mkv (ffv1)":
            # Lossless, and "gbrp" (FFV1's planar RGB pixel format) keeps the full
            # RGB precision rather than yuv420p's chroma subsampling - there's no
            # bitrate/quality concept for a lossless codec, so nothing else to set.
            return {"vcodec": "ffv1", "pix_fmt": "gbrp"}
        kwargs = {"pix_fmt": "yuv420p", "vcodec": "libx264"}
        if mode == "Constant quality":
            kwargs["crf"] = max(0, min(51, round(51 * (1.0 - quality / 100.0))))
        elif mode == "Target bitrate":
            kwargs["video_bitrate"] = f"{bitrate_kbps}k"
        else:                                        # Variable bitrate
            kwargs["video_bitrate"] = f"{vbr_target_kbps}k"
            kwargs["maxrate"] = f"{vbr_max_kbps}k"
            kwargs["bufsize"] = f"{2 * vbr_max_kbps}k"
        return kwargs

    def read_render_settings(self, require_resolution=True):
        """Framerate, resolution and bitrate/format settings from the Video
        panel, or None (with an error dialog already shown) if anything
        required is invalid. Returns a dict. With require_resolution=False
        (Re-encode), the width/height fields aren't read or validated at all,
        since re-encoding always uses the existing frames' own resolution."""
        fps = parse_positive_int(self.fps_var.get())
        if fps is None:
            messagebox.showerror("Invalid framerate", "Enter a framerate of at least 1.")
            return None
        video_w = video_h = None
        if require_resolution:
            video_w = parse_resolution_dimension(self.video_w_var.get(), max_digits=6)
            video_h = parse_resolution_dimension(self.video_h_var.get(), max_digits=6)
            if video_w is None or video_h is None:
                messagebox.showerror("Invalid resolution",
                                     "Enter a video width and height of at least "
                                     "1 pixel.")
                return None
        fmt = self.export_format_var.get()
        mode = self.bitrate_mode_var.get()
        quality = int(round(float(self.quality_scale.get())))
        bitrate = vbr_target = vbr_max = 0
        if fmt == "mp4 (h264)" and mode == "Target bitrate":
            bitrate_text = self.bitrate_var.get()
            bitrate = int(bitrate_text) if bitrate_text else 0
            if bitrate <= 0:
                messagebox.showerror("Invalid bitrate",
                                     "Enter a bitrate greater than 0 kbps.")
                return None
        elif fmt == "mp4 (h264)" and mode == "Variable bitrate":
            target_text = self.vbr_target_var.get()
            max_text = self.vbr_max_var.get()
            vbr_target = int(target_text) if target_text else 0
            vbr_max = int(max_text) if max_text else 0
            if vbr_target <= 0 or vbr_max <= 0:
                messagebox.showerror("Invalid bitrate",
                                     "Enter a target and a maximum bitrate, "
                                     "both greater than 0 kbps.")
                return None
            if vbr_max < vbr_target:
                messagebox.showerror("Invalid bitrate",
                                     "The maximum bitrate must be at least as "
                                     "high as the target bitrate.")
                return None
        if external.ffmpeg is None:
            messagebox.showerror("ffmpeg-python not found",
                                 "This needs the ffmpeg-python package and the "
                                 "ffmpeg command-line tool. Install with:\n"
                                 "  pip install ffmpeg-python")
            return None
        if fmt == "JXL" and not cjxl_available():
            messagebox.showerror("cjxl not found",
                                 "JXL video export needs the 'cjxl' command-line "
                                 "tool (part of libjxl) to convert the rendered "
                                 "frames to an animated JXL file. Install it and "
                                 "make sure it's on your PATH.")
            return None
        return {"fps": fps, "video_w": video_w, "video_h": video_h, "fmt": fmt,
               "mode": mode, "quality": quality, "bitrate": bitrate,
               "vbr_target": vbr_target, "vbr_max": vbr_max,
               "jxl_lossless": self.video_jxl_controls.lossless(),
               "jxl_quality": self.video_jxl_controls.quality(),
               "downsample_factor": DOWNSAMPLE_FACTOR_VALUES[self.downsample_factor_var.get()],
               "downsample_method": self.downsample_method_var.get()}

    def busy_with_render_or_reencode(self):
        return self.render_thread is not None and self.render_thread.is_alive()

    def start_or_cancel_render(self):
        if self.busy_with_render_or_reencode():
            self.render_cancel.set()
            self.append_video_status("Cancelling (finishing the current frame)...")
            return

        if not self.keyframes:
            messagebox.showerror("No keyframes",
                                 "Add at least one keyframe before rendering.")
            return
        indices = [kf.frame_index for kf in self.keyframes]
        if len(indices) != len(set(indices)):
            messagebox.showerror("Duplicate frame indices",
                                 "Two or more keyframes share the same frame "
                                 "index. Give each a unique frame index (in the "
                                 "Timeline panel) before rendering.")
            return
        rs = self.read_render_settings()
        if rs is None:
            return

        os.makedirs(paths.TEMP_DIR, exist_ok=True)
        existing = os.listdir(paths.TEMP_DIR)
        if existing:
            messagebox.showerror("TEMP folder is not empty",
                                 f"{paths.TEMP_DIR}\nalready has {len(existing)} file(s) "
                                 "in it. This program never deletes files, "
                                 "including its own temporary ones - please clear "
                                 "the TEMP folder yourself, then try again.")
            return

        starting_frame = parse_frame_number(self.start_frame_var.get())
        if starting_frame is None:
            messagebox.showerror("Invalid starting frame",
                                 "Enter a starting frame: a whole number, 0 or more.")
            return
        frame_range = frames_to_render(indices, starting_frame)
        if frame_range is None:
            messagebox.showerror("Nothing to render",
                                 f"The starting frame ({starting_frame}) is after the last "
                                 f"keyframe (frame {max(indices)}), so there is nothing to "
                                 "render. Lower the starting frame or add a later keyframe.")
            return
        frame_lo, frame_hi = frame_range
        file_offset = frame_file_offset(frame_lo, self.number_files_from_start_var.get())
        kf_snapshot = snapshot_keyframes(self.keyframes)   # independent copies: edits made
                                                   # while rendering must not leak in
        base_settings = self.capture_all_settings()        # (a copy: a fresh dict) - what no keyframe defines
        if base_settings is None:                          # the limit fields are invalid (it said so)
            return
        obst_snapshot = [o.snapshot() for o in self.obstacles]
        coloration_snapshot = [snapshot_coloration(v) for v in self.colorations_to_export()]
        framing_snapshot = self.selected_framing_snapshot()
        color_location = self.color_info_location_var.get()
        figure_location = self.figure_info_location_var.get()
        framing_location = self.framing_info_location_var.get()

        self.render_cancel = threading.Event()
        self.render_btn.configure(text="Cancel Render")
        self.reencode_btn.state(["disabled"])
        self.clear_btn.state(["disabled"])
        self.begin_video_status("Starting...")
        total = frame_hi - frame_lo + 1
        self.video_progress.configure(maximum=total)
        self.video_progress["value"] = 0
        self.render_thread = threading.Thread(
            target=self._render_worker,
            args=(kf_snapshot, frame_lo, frame_hi, rs, base_settings, obst_snapshot, coloration_snapshot, framing_snapshot,
                 color_location, figure_location, framing_location, file_offset),
            daemon=True)
        self.render_thread.start()

    def output_extension_and_suffix(self, rs):
        """(extension, filename suffix) for the chosen export format."""
        fmt = rs["fmt"]
        if fmt == "GIF":
            return "gif", ""
        if fmt == "APNG":
            return "apng", ""
        if fmt == "JXL":
            return "jxl", "." + jxl_quality_tag(rs["jxl_lossless"], rs["jxl_quality"])
        if fmt == "mkv (ffv1)":
            return "mkv", ".ffv1"     # MKV can hold many codecs, so the codec goes in
                                      # the filename itself for clarity
        return "mp4", ""

    def prepare_ffmpeg_source(self, start_number, num_frames, rs):
        """(pattern, start_number, scratch_dir_or_None) for ffmpeg to read
        frames from. With no downsampling, or with the "ffmpeg" downsample
        method (which scales natively during encoding instead - see
        apply_downsample_filter), reads directly from TEMP_DIR, unchanged.
        With the "Mean" or "Median" method, reads
        frame_{start_number}.png .. frame_{start_number+num_frames-1}.png
        from TEMP_DIR, shrinks each one in Python, and writes the shrunk
        copies to a scratch subdirectory for ffmpeg to read instead -
        downsampling is an encoding setting, so the original PNG sequence in
        TEMP_DIR is never touched by it. Call cleanup_scratch_dir() with the
        third value once encoding finishes (successfully or not)."""
        factor = rs["downsample_factor"]
        method = rs["downsample_method"]
        if factor <= 1 or method == "ffmpeg":
            return os.path.join(paths.TEMP_DIR, "frame_%05d.png"), start_number, None
        scratch_dir = os.path.join(paths.TEMP_DIR, "_downsample_scratch")
        if os.path.isdir(scratch_dir):
            shutil.rmtree(scratch_dir)
        os.makedirs(scratch_dir, exist_ok=True)
        for i in range(num_frames):
            src = os.path.join(paths.TEMP_DIR, f"frame_{start_number + i:05d}.png")
            small = downsample_image(read_png(src), factor, method)
            write_png(os.path.join(scratch_dir, f"frame_{i:05d}.png"), small)
        return os.path.join(scratch_dir, "frame_%05d.png"), 0, scratch_dir

    def apply_downsample_filter(self, stream, rs):
        """Apply ffmpeg's own scale filter to `stream` when downsampling is
        set to the "ffmpeg" method - fast, since it's done natively during
        encoding rather than pre-processing every frame in Python first. Uses
        ffmpeg's iw/ih (input width/height) expressions, so it works without
        needing to know the actual frame resolution here (important for
        Re-encode, which doesn't read it). "area" is ffmpeg's box-filter/
        averaging algorithm - the closest native equivalent to the "Mean"
        method, and appropriate for shrinking by an exact integer factor."""
        factor = rs["downsample_factor"]
        if factor > 1 and rs["downsample_method"] == "ffmpeg":
            stream = external.ffmpeg.filter(stream, "scale", f"iw/{factor}", f"ih/{factor}",
                                   flags="area")
        return stream

    def cleanup_scratch_dir(self, scratch_dir):
        if scratch_dir is not None and os.path.isdir(scratch_dir):
            shutil.rmtree(scratch_dir, ignore_errors=True)

    def ffmpeg_error_detail(self, exc, out_path, pattern):
        """Build the render-failure message, including ffmpeg's own stderr
        text when the exception is an ffmpeg.Error (quiet=True still captures
        it, it's just not shown unless we pull it out explicitly)."""
        stderr_text = ""
        if isinstance(exc, external.ffmpeg.Error) and getattr(exc, "stderr", None):
            decoded = exc.stderr.decode("utf-8", errors="replace").strip()
            if decoded:
                stderr_text = "\nffmpeg said:\n" + decoded[-4000:]
        return (f"{exc}{stderr_text}\n"
               f"Frames folder: {paths.TEMP_DIR}\n"
               f"Frame pattern: {pattern}\n"
               f"Output file: {out_path}\n"
               "A 'cannot find the path specified'/'No such file or "
               "directory' error here usually means the ffmpeg command-line "
               "program itself isn't installed or isn't on your PATH - check "
               "that 'ffmpeg' runs from a terminal.")

    def encode_video_output(self, pattern, start_number, rs, out_path):
        """Runs ffmpeg over the frame sequence at `pattern`, producing
        `out_path` in whichever format rs["fmt"] selects. Shared by a fresh
        render and a re-encode of existing frames - the two differ only in
        where the frames themselves came from, not in how they get encoded.

        JXL is a special case: ffmpeg's own libjxl wrapper only supports a
        single still frame, not animation (see write_jxl), so there is no
        way to have ffmpeg produce an animated .jxl file directly. Instead,
        ffmpeg renders to a temporary APNG here (which it handles natively
        and losslessly), and the external `cjxl` tool (part of libjxl,
        checked for availability back in read_render_settings) converts
        that APNG's full animation to the final JXL file, which it does
        support (lossless, or lossy at the quality chosen in the video panel) - the
        temporary APNG is removed either way."""
        kwargs = self.ffmpeg_output_kwargs(rs["fmt"], rs["mode"], rs["quality"],
                                           rs["bitrate"], rs["vbr_target"],
                                           rs["vbr_max"])
        ffmpeg_out_path = out_path
        jxl_apng_temp = None
        if rs["fmt"] == "JXL":
            jxl_apng_temp = os.path.join(paths.TEMP_DIR, "_jxl_video_source.apng")
            ffmpeg_out_path = jxl_apng_temp
        stream = external.ffmpeg.input(pattern, framerate=rs["fps"], start_number=start_number)
        stream = self.apply_downsample_filter(stream, rs)
        stream = external.ffmpeg.output(stream, ffmpeg_out_path, **kwargs)
        external.ffmpeg.run(stream, overwrite_output=True, quiet=True)
        if jxl_apng_temp is not None:
            self.root.after(0, lambda: self.append_video_status("Converting to JXL with cjxl..."))
            try:
                convert_apng_to_jxl(jxl_apng_temp, out_path, rs["jxl_lossless"],
                                    rs["jxl_quality"])
            finally:
                if os.path.exists(jxl_apng_temp):
                    os.remove(jxl_apng_temp)

    def _render_worker(self, keyframes, frame_lo, frame_hi, rs, base_settings, obst_snapshot, coloration_snapshot,
                       framing_snapshot, color_location, figure_location, framing_location, file_offset):
        """Runs on a background thread: never touches a Tk widget directly, only
        via self.root.after(0, ...). Frame `fn` is written to the file numbered
        fn - frame_lo + file_offset (see frame_file_offset). `base_settings` is every static setting as the controls
        held it when the render started: the value of a setting no keyframe defines."""
        ext, suffix = self.output_extension_and_suffix(rs)
        out_name = f"{paths.PROGRAM_NAME}-video-{time.strftime('%Y%m%d-%H%M%S')}{suffix}.{ext}"
        out_path = os.path.join(ensure_export_dir(), out_name)
        pattern = os.path.join(paths.TEMP_DIR, "frame_%05d.png")
        scratch_dir = None
        try:
            assigned_sorted = sorted(keyframes, key=lambda k: k.frame_index)
            last_found = None
            total = frame_hi - frame_lo + 1
            for n, fn in enumerate(range(frame_lo, frame_hi + 1)):
                if self.render_cancel.is_set():
                    self.root.after(0, lambda: self.append_video_status("Cancelled."))
                    return
                data = resolve_frame_settings(assigned_sorted, fn, base_settings)
                obst_values = resolve_obstacles(obst_snapshot, assigned_sorted, fn)
                colorations = resolve_colorations(coloration_snapshot, assigned_sorted, fn)
                # D20: what the engine of this frame does not accept (keyframes may hold such values) is corrected before
                # the frame is drawn and before its margin text is made, and the status box says so
                data, obst_values, colorations, corrections = correct_frame(data, obst_values, colorations)
                if corrections:
                    found = [(c.key, c.field, c.old, c.new) for c in corrections]
                    if found != last_found:
                        message = correction_message(corrections, f"Frame {fn}: some values were clamped or defaulted:")
                    else:                           # (not the whole list again for every frame of a long video)
                        message = f"Frame {fn}: the same values were clamped or defaulted."
                    self.root.after(0, lambda t=message: self.append_video_status(t))
                    last_found = found
                obst_arrays = build_obstacle_arrays(obst_values)
                framing = resolve_framing(framing_snapshot, assigned_sorted, fn)
                rgb = self.render_full_image(data, rs["video_w"], rs["video_h"], obst_arrays,
                                             colorations, framing)
                # The figure's settings can be keyframe-animated, so this is
                # resolved per frame - but the margin's own size depends only on
                # the fixed video_w/video_h and the number of obstacles, so every
                # frame still comes out the same size.
                rgb = apply_info_locations(rgb, colorations, rs["video_w"],
                                           data.ray, data.natural_termination, obst_values,
                                           color_location, figure_location, framing, framing_location,
                                           mutate_ok=True, engine=data.engine, mandelbrot=data.mandelbrot)
                path = os.path.join(paths.TEMP_DIR, f"frame_{fn - frame_lo + file_offset:05d}.png")
                write_png(path, rgb)
                done = n + 1
                self.root.after(0, lambda d=done, t=total, im=rgb:
                                self._update_render_progress(d, t, im))

            if rs["downsample_factor"] > 1 and rs["downsample_method"] != "ffmpeg":
                self.root.after(0, lambda: self.append_video_status("Downsampling frames..."))
                pattern, start_number, scratch_dir = self.prepare_ffmpeg_source(
                    file_offset, total, rs)
            else:
                start_number = file_offset
            self.root.after(0, lambda: self.append_video_status("Encoding video with ffmpeg..."))
            self.encode_video_output(pattern, start_number, rs, out_path)
            info = self.describe_output_file(out_path, total, rs["fps"])
            self.root.after(0, lambda: self.append_video_status(f"Done: {out_path}\n{info}"))
        except Exception as exc:                    # noqa: BLE001
            detail = self.ffmpeg_error_detail(exc, out_path, pattern)
            self.root.after(0, lambda m=detail: self.append_video_status(f"Render failed: {m}"))
        finally:
            self.cleanup_scratch_dir(scratch_dir)
            self.root.after(0, self._render_finished)

    def start_reencode(self):
        """Combine whatever frames are already in TEMP into a new video, at
        whatever resolution the first frame happens to be (the width/height
        fields are ignored for this)."""
        if self.busy_with_render_or_reencode():
            messagebox.showerror("Busy", "A render or re-encode is already running.")
            return
        if not os.path.isdir(paths.TEMP_DIR):
            messagebox.showerror("No frames",
                                 f"{paths.TEMP_DIR}\ndoesn't exist yet - render a video "
                                 "first, or place frame PNGs there yourself.")
            return
        frame_files = sorted(f for f in os.listdir(paths.TEMP_DIR)
                             if re.fullmatch(r"frame_\d+\.png", f))
        if not frame_files:
            messagebox.showerror("No frames",
                                 f"{paths.TEMP_DIR}\nhas no frame_NNNNN.png files to "
                                 "combine. Render a video first, or place frame "
                                 "PNGs there yourself.")
            return
        rs = self.read_render_settings(require_resolution=False)
        if rs is None:
            return
        start_number = min(int(re.fullmatch(r"frame_(\d+)\.png", f).group(1))
                           for f in frame_files)

        self.render_cancel = threading.Event()      # unused here, but shared plumbing
        self.render_btn.state(["disabled"])
        self.reencode_btn.state(["disabled"])
        self.clear_btn.state(["disabled"])
        self.begin_video_status(f"Re-encoding {len(frame_files)} existing frame(s)...")
        self.video_progress.configure(maximum=1)
        self.video_progress["value"] = 0
        self.render_thread = threading.Thread(
            target=self._reencode_worker, args=(rs, start_number, len(frame_files)),
            daemon=True)
        self.render_thread.start()

    def _reencode_worker(self, rs, start_number, num_frames):
        """Runs on a background thread: never touches a Tk widget directly, only
        via self.root.after(0, ...)."""
        ext, suffix = self.output_extension_and_suffix(rs)
        out_name = f"{paths.PROGRAM_NAME}-video-{time.strftime('%Y%m%d-%H%M%S')}{suffix}.{ext}"
        out_path = os.path.join(ensure_export_dir(), out_name)
        pattern = os.path.join(paths.TEMP_DIR, "frame_%05d.png")
        scratch_dir = None
        try:
            if rs["downsample_factor"] > 1 and rs["downsample_method"] != "ffmpeg":
                self.root.after(0, lambda: self.append_video_status("Downsampling frames..."))
                pattern, start_number, scratch_dir = self.prepare_ffmpeg_source(
                    start_number, num_frames, rs)
            self.root.after(0, lambda: self.append_video_status("Encoding video with ffmpeg..."))
            self.encode_video_output(pattern, start_number, rs, out_path)
            info = self.describe_output_file(out_path, num_frames, rs["fps"])
            self.root.after(0, lambda: self.append_video_status(f"Done: {out_path}\n{info}"))
        except Exception as exc:                    # noqa: BLE001
            detail = self.ffmpeg_error_detail(exc, out_path, pattern)
            self.root.after(0, lambda m=detail: self.append_video_status(f"Render failed: {m}"))
        finally:
            self.cleanup_scratch_dir(scratch_dir)
            self.root.after(0, self._render_finished)

    def _update_render_progress(self, done, total, latest_frame):
        self.video_progress.configure(maximum=total)
        self.video_progress["value"] = done
        self.set_video_progress_text(f"{done}/{total}")
        self.set_preview_image(latest_frame)

    def clear_temp_frames(self):
        """Deletes only frame_NNNNN.png files from TEMP_DIR - the exact same
        pattern start_reencode itself looks for, and the only file shape
        this program's own rendering ever writes there. Nothing else in the
        folder is touched, however it got there, and this is the one place
        in the whole program that deletes anything at all - everywhere else
        deliberately leaves that to the user (see start_or_cancel_render's
        own "TEMP folder is not empty" check)."""
        if self.busy_with_render_or_reencode():
            messagebox.showerror("Busy", "A render or re-encode is already running.")
            return
        if not os.path.isdir(paths.TEMP_DIR):
            messagebox.showinfo("Nothing to clear", f"{paths.TEMP_DIR}\ndoesn't exist yet.")
            return
        frame_files = sorted(f for f in os.listdir(paths.TEMP_DIR)
                             if re.fullmatch(r"frame_\d+\.png", f))
        if not frame_files:
            messagebox.showinfo("Nothing to clear",
                                f"{paths.TEMP_DIR}\nhas no frame_NNNNN.png files to delete.")
            return
        if not messagebox.askyesno(
                "Delete frame files?",
                f"Delete {len(frame_files)} frame_NNNNN.png file(s) from\n{paths.TEMP_DIR}?\n\n"
                "Only files matching that exact name pattern are touched - anything "
                "else in the folder is left alone. This can't be undone."):
            return
        errors = []
        deleted = 0
        for f in frame_files:
            try:
                os.remove(os.path.join(paths.TEMP_DIR, f))
                deleted += 1
            except OSError as exc:
                errors.append(f"{f}: {exc}")
        if errors:
            shown = "\n".join(errors[:10]) + ("\n..." if len(errors) > 10 else "")
            messagebox.showerror(
                "Some files couldn't be deleted",
                f"Deleted {deleted} of {len(frame_files)} file(s). Failures:\n{shown}")
        else:
            self.append_video_status(f"Cleared {deleted} frame file(s) from TEMP.")

    def _render_finished(self):
        self.render_thread = None
        self.render_btn.configure(text="Start Render")
        self.render_btn.state(["!disabled"])
        self.reencode_btn.state(["!disabled"])
        self.clear_btn.state(["!disabled"])
        self.set_preview_image(None)              # back to the placeholder
