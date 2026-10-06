"""The picture on screen: drawing it with its overlays and obstacle outlines, and the hover / pinned ray demonstration."""
import tkinter as tk
import numpy as np
from .constants import (
    CONFINE_METHOD_NAMES, EMIT_MODE_NAMES, GUIDE_FALLBACK_HEX, PREVIEW_SIZE, SAMPLE_COLORS,
    TERM_END_COLOR, TERM_OUTCOME_TEXT, VIEW_NAME_OVERLAY_BG, VIEW_NAME_OVERLAY_FG,
    VIEW_NAME_OVERLAY_MARGIN, VIEW_NAME_OVERLAY_PAD, VIEW_NAME_OVERLAY_SCALE
)
from .image_io import bilinear_resize, encode_png_bytes, preview_placeholder, render_text_line
from .kernels import (
    N_STATS, PIXEL, RAW_AREA, RAW_BOUNCES, RAW_BOUNDARY, RAW_DISP, RAW_LAST, RAW_LAST_DIR,
    RAW_LAST_FULL, RAW_LENGTH, RAW_SELF_K, SIZE, compute_points, compute_stats, emits_ray,
    trace_path
)
from .metrics import POINT_STATS, resolve_point_xy
from .settings import NaturalStopSettings, RaySettings, settings_natural_stop, settings_ray_args
from .shapes import ObstacleRing, build_obstacle_arrays, draw_obstacle_outline
from .views import MODES


class PreviewMixin:
    def set_preview_image(self, rgb):
        """Show `rgb` (an (H, W, 3) uint8 array) bilinearly resized into the
        frame-preview box, or the 'preview will go here' placeholder if rgb
        is None (i.e. nothing is currently being rendered)."""
        if rgb is None:
            small = preview_placeholder(PREVIEW_SIZE, PREVIEW_SIZE)
        else:
            small = bilinear_resize(rgb, PREVIEW_SIZE, PREVIEW_SIZE)
        photo = tk.PhotoImage(data=encode_png_bytes(small), format="png")
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo            # keep a reference alive

    def obstacle_ring_specs(self):
        """[ObstacleRing, ...] for obstacles to draw a solid two-layer outline for:
        the frozen set if an image has been generated (matching what was actually
        simulated/saved), else the live (current) set."""
        if self.generated_settings is not None:
            g = self.gen_ray.obstacles
            return [ObstacleRing(shape=int(g.shapes[i]), bound=g.bounds[i], cx=float(g.cx[i]),
                                 cy=float(g.cy[i]), in_effect=int(g.effects[i, 0]),
                                 out_effect=int(g.effects[i, 1]))
                    for i in range(len(g.shapes))]
        rings = []
        for o in self.obstacles:
            cx, cy = o.center()
            rings.append(ObstacleRing(shape=o.KIND, bound=o.bound(),
                                      cx=cx, cy=cy, in_effect=o.in_effect,
                                      out_effect=o.out_effect))
        return rings

    def draw(self):
        """Push the current image to the screen, with the obstacle outlines
        drawn into it if that decoration is enabled
        (two layers: outer = incoming effect, inner = outgoing effect; brown for
        Terminate, pale light blue for Reflect, black for Pass through),
        then the live view's name in the bottom-left corner - a preview-only
        decoration drawn into self.display_frame, a separate buffer from
        self.frame (which is what gets exported/saved, and so must never
        carry it)."""
        np.copyto(self.frame, self.img)
        outline = self.deco.get("outline")
        if outline is None or outline.get():
            for ring in self.obstacle_ring_specs():
                draw_obstacle_outline(self.frame, ring.shape, ring.bound, self.gx - ring.cx,
                                      self.gy - ring.cy, ring.in_effect, ring.out_effect)
        np.copyto(self.display_frame, self.frame)
        self.draw_view_name_overlay()
        header = f"P6 {SIZE} {SIZE} 255\n".encode()
        self.photo.configure(data=header + self.display_frame.tobytes(), format="PPM")

    def draw_view_name_overlay(self):
        """Stamps the live view's display name ("view N") into the bottom-left
        corner of self.display_frame, in the tiny 5x7 pixel font, on a solid
        background box so it stays legible over any image content. Never
        touches self.frame - this is preview-only and must never end up in
        a saved/exported image."""
        idx = self.all_view_ids().index(self.live_view().id)
        mask = render_text_line(f"view {idx}", VIEW_NAME_OVERLAY_SCALE)
        h, w = mask.shape
        pad = VIEW_NAME_OVERLAY_PAD
        margin = VIEW_NAME_OVERLAY_MARGIN
        y1 = SIZE - margin
        y0 = y1 - h - 2 * pad
        x0 = margin
        x1 = x0 + w + 2 * pad
        self.display_frame[y0:y1, x0:x1] = VIEW_NAME_OVERLAY_BG
        region = self.display_frame[y0 + pad:y0 + pad + h, x0 + pad:x0 + pad + w]
        region[mask] = VIEW_NAME_OVERLAY_FG

    # ---- hover demonstration ---------------------------------------------- #
    def preview_settings(self):
        """Settings for previews: those of the image being generated, or the
        current entries/controls (last valid values) if nothing has been started."""
        if self.generated_settings is not None:
            return self.generated_settings
        return RaySettings(seg_mult=self.seg_mult_scale.get(), fuzz=self.read_fuzz_controls(),
                        emit_mode=EMIT_MODE_NAMES.index(self.emit_mode_var.get()),
                        emit_target_r=self.last_emit_target_r,
                        emit_target_theta=self.last_emit_target_theta,
                        emit_target_cx=self.last_emit_target_cx,
                        emit_target_cy=self.last_emit_target_cy,
                        emit_offset=round(float(self.emit_offset_scale.get()), 2))

    def preview_natural_stop_settings(self):
        """The natural stopping settings for previews, chosen the same way as preview_settings():
        those of the image being generated, or the controls' last valid values."""
        if self.generated_natural_stop is not None:
            return self.generated_natural_stop
        return NaturalStopSettings(
            max_iter=self.last_iter, max_dist=self.last_dist, escape_radius=self.last_escape_radius,
            confine_window=self.last_confine_window,
            confine_method=CONFINE_METHOD_NAMES.index(self.confine_method_var.get()),
            confine_threshold=self.last_confine_threshold)

    def invalidate_preview(self):
        """Called whenever a control that feeds preview_settings() changes,
        before the first Start. A pinned preview is recomputed immediately (so
        it tracks the new settings even though the mouse hasn't moved); a live
        hover is just cleared, to be recomputed on the next mouse move."""
        if self.generated_settings is not None:
            return                                 # preview is frozen to the image
        self.hover_px = None
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)

    def on_motion(self, event):
        if self.pinned_px is not None:
            return                                 # preview is pinned: ignore the mouse
        self.update_preview(event.x, event.y)

    def on_click(self, event):
        i, j = event.x, event.y
        if not (0 <= i < SIZE and 0 <= j < SIZE):
            return
        x = (i + 0.5) * PIXEL - 1.0
        y = 1.0 - (j + 0.5) * PIXEL
        ps = self.preview_settings()
        obstacles = build_obstacle_arrays(self.obstacles)
        if (self.pinned_px != (i, j) and not emits_ray(x, y, obstacles)):
            return                                 # clicking where no ray fires does nothing -
                                                   # except on the pinned pixel, to unpin it
        if self.pinned_px == (i, j):
            self.pinned_px = None                  # unpin: resume following the mouse
            self.canvas.configure(cursor="crosshair")
            self.update_preview(i, j)
        else:
            self.pinned_px = (i, j)
            self.canvas.configure(cursor="hand2")
            self.hover_px = None                   # force a redraw even if unchanged
            self.update_preview(i, j)

    def update_preview(self, i, j):
        """(Re)compute and draw the preview for pixel (i, j) of the canvas."""
        if not (0 <= i < SIZE and 0 <= j < SIZE):
            self.clear_overlay()
            return
        x = (i + 0.5) * PIXEL - 1.0
        y = 1.0 - (j + 0.5) * PIXEL
        ps = self.preview_settings()
        obstacles = build_obstacle_arrays(self.obstacles)
        # A hovered pixel that fires no ray has no preview - but a pinned one
        # stays visible even if the scene changed under it so that it no longer
        # fires: it then shows the ray such a pixel would trace anyway.
        if (self.pinned_px != (i, j) and not emits_ray(x, y, obstacles)):
            self.clear_overlay()                   # not a ray-emitting pixel
            return
        if (i, j) == self.hover_px:
            return

        self.hover_px = (i, j)
        pn = self.preview_natural_stop_settings()
        if ps.fuzz.enabled:
            np.random.seed(ps.fuzz.seed + i * 1000003 + j * 97)
        pts, kinds = self.hover_pts, self.hover_kinds
        raw, vec, sp = self.hover_raw, self.hover_vec, self.hover_sp
        term, n = trace_path(x, y, pts, kinds, settings_natural_stop(pn),
                             settings_ray_args(ps, obstacles), raw)
        compute_stats(pts, n, raw, pn.max_iter, ps.seg_mult, vec)
        compute_points(pts, n, raw, kinds, sp)
        vals = self.component_values(vec, sp, x, y)
        self.draw_segments(pts, kinds, n)

        # ---- overlay on the canvas (never part of the image) ----
        half = SIZE / 2.0
        xs = (pts[:n, 0] + 1.0) * half
        ys = (1.0 - pts[:n, 1]) * half
        coords = np.column_stack((xs, ys)).ravel().tolist()
        show = {key: var.get() for key, var in self.deco.items()}

        cv = self.canvas
        cv.delete("ray")

        # Semi-transparent grey guide lines through the end of the ray
        if show["guides"]:
            gx = min(max(int(xs[-1]), 0), SIZE - 1)
            gy = min(max(int(ys[-1]), 0), SIZE - 1)
            if self.hline_img is not None:
                cv.create_image(0, gy, anchor="nw", image=self.hline_img, tags="ray")
                cv.create_image(gx, 0, anchor="nw", image=self.vline_img, tags="ray")
            else:                                   # no alpha support: stipple instead
                cv.create_line(0, gy + 0.5, SIZE, gy + 0.5, fill=GUIDE_FALLBACK_HEX,
                               stipple="gray50", tags="ray")
                cv.create_line(gx + 0.5, 0, gx + 0.5, SIZE, fill=GUIDE_FALLBACK_HEX,
                               stipple="gray50", tags="ray")

        # The ray's path
        if show["ray_path"]:
            cv.create_line(*coords, fill="#ffe14d", width=1, tags="ray")

        # Start marker (green), end marker: red if the ray was confined, orange
        # if the distance limit ended it, grey for the iteration limit, yellow
        # if a "terminate" obstacle ended it
        if show["ray_start"]:
            cv.create_oval(xs[0] - 3, ys[0] - 3, xs[0] + 3, ys[0] + 3,
                           fill="#3ddc63", outline="black", tags="ray")
        if show["ray_end"]:
            end_col = TERM_END_COLOR[term]
            cv.create_oval(xs[-1] - 3, ys[-1] - 3, xs[-1] + 3, ys[-1] + 3,
                           fill=end_col, outline="black", tags="ray")

        # Sample points: two rings (point 1, point 2) per channel that uses a
        # point-comparison metric
        if show["samples"]:
            live_view = self.live_view()
            for c in range(3):
                if self.mapping[c] in POINT_STATS:
                    base_rad = 8 - 2 * c
                    for p, rad in ((live_view.point1_choice[c], base_rad),
                                  (live_view.point2_choice[c], base_rad + 3)):
                        px, py = resolve_point_xy(sp, p)
                        cx = (px + 1.0) * half
                        cy = (1.0 - py) * half
                        cv.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, width=2,
                                      outline=SAMPLE_COLORS[live_view.mode][c], tags="ray")

        # ---- info panel ----
        bounces = int(raw[RAW_BOUNCES])
        n_boundary = int(raw[RAW_BOUNDARY])
        fx, fy = float(pts[n - 1, 0]), float(pts[n - 1, 1])
        r, g, b = self.map_color(vals)
        hexcol = f"#{r:02x}{g:02x}{b:02x}"

        v = self.info_vars
        pin_tag = "  [pinned]" if self.pinned_px == (i, j) else ""
        v["start"].set(f"({i}, {j})  [{x:+.3f}, {y:+.3f}]{pin_tag}")
        v["outcome"].set(TERM_OUTCOME_TEXT[term].format(bounces=bounces))
        v["bounces"].set(f"{bounces}: {n_boundary} boundary, "
                         f"{bounces - n_boundary} own path")
        v["final"].set(f"({fx:+.4f}, {fy:+.4f})")
        v["length"].set(f"{raw[RAW_LENGTH]:.4f} (start-end {raw[RAW_DISP]:.4f})")
        seg = f"len {raw[RAW_LAST]:.3g}"
        if raw[RAW_LAST_FULL] > 0:
            seg += f" of {raw[RAW_LAST_FULL]:.3g}"
        seg += f", dir {raw[RAW_LAST_DIR]:.1f}\u00b0"
        v["lastseg"].set(seg)
        if raw[RAW_SELF_K] >= 0:
            v["loop"].set(f"area {raw[RAW_AREA]:.4f}")
        else:
            v["loop"].set("never touches itself")
        letters = [name[0] for name, _color in MODES[self.live_view().mode]]
        v["components"].set("   ".join(f"{letters[c]} {vals[c]:.3f}" for c in range(3)))
        v["color"].set(hexcol)
        for s in range(N_STATS):
            self.stat_vars[s].set(f"{vec[s]:.3f}")
        self.swatch.configure(bg=hexcol)
        self.info_placeholder.grid_remove()
        self.info_rows.grid()

    def on_leave(self, _event):
        if self.pinned_px is None:                 # a pinned preview stays put
            self.clear_overlay()

    def clear_overlay(self):
        self.canvas.delete("ray")
        self.hover_px = None
        self.info_rows.grid_remove()
        self.info_placeholder.grid()
        self.seg_canvas.delete("all")
