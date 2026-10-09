"""The picture on screen: drawing it with its overlays and obstacle outlines, and the hover / pinned ray demonstration."""
import tkinter as tk
import numpy as np
from .constants import (
    CONFINE_METHOD_NAMES, EMIT_MODE_NAMES, GUIDE_FALLBACK_HEX, VIDEO_PREVIEW_SIZE, SAMPLE_COLORS, SEED_ZONE_OUTLINE,
    TERM_END_COLOR, TERM_OUTCOME_TEXT, COLORATION_NAME_OVERLAY_BG, COLORATION_NAME_OVERLAY_FG,
    COLORATION_NAME_OVERLAY_MARGIN, COLORATION_NAME_OVERLAY_PAD, COLORATION_NAME_OVERLAY_SCALE, BACKGROUND
)
from .image_io import bilinear_resize, encode_png_bytes, make_grid, preview_placeholder, render_text_line
from .framings import framing_image_size, framing_outline_pixel, framing_world_rect
from .kernels import (
    N_POINTS, N_STATS, PIXEL_PENDING, RAW_AREA, RAW_BOUNCES, RAW_DISP, RAW_LAST, RAW_LAST_DIR,
    RAW_LAST_FULL, RAW_LENGTH, RAW_SELF_K, SEQ_BOUNDARY, fires_ray, in_sequence, measure_path, trace_engine_path
)
from .engines import layout_of_engine, tables_of_engine
from .metrics import POINT_METRICS, resolve_point_xy
from .settings import NaturalTerminationSettings, RaySettings, path_args, settings_natural_termination, settings_ray_args
from .shapes import ObstacleRing, build_obstacle_arrays, draw_obstacle_outline
from .colorations import COLOR_MODELS


class PreviewMixin:
    # ---- the image on screen: its framing, its pixels and the world they cover ---------------------------- #
    def set_image_framing(self, framing):
        """The image on screen is, from now on, an image of `framing` (a FramingSnapshot): the pixel grid is
        its aspect ratio's size (see framing_image_size), covers the world rectangle it makes (image_world_rect), and
        everything computed so far is discarded (nothing is computed for the new grid yet: rows_done is 0). The
        canvas takes the new size. Every conversion between a pixel and the world goes through image_world_rect,
        img_w and img_h (pixel_center, world_to_canvas)."""
        self.image_framing = framing
        self.image_world_rect = framing_world_rect(framing)
        self.img_w, self.img_h = framing_image_size(framing)
        self.outline_pixel = framing_outline_pixel(framing)
        self.rows_done = 0
        self.make_image_buffers()
        if hasattr(self, "canvas"):
            self.canvas.configure(width=self.img_w, height=self.img_h)
            self.photo.configure(width=self.img_w, height=self.img_h)

    def make_image_buffers(self):
        """The per-pixel data and the image buffers for an img_w x img_h image of image_world_rect, all empty (the
        buffers of the same size as the last are cleared, not made again) and the grid of the world coordinates of
        every pixel centre (which is also each ray's start point)."""
        shape = (self.img_h, self.img_w)
        layout = self.current_figure().layout
        if (getattr(self, "pixel_state", None) is not None and self.pixel_state.shape == shape
                and self.stats.shape[-1] == layout.n_stats and self.characteristic_points.shape[-2] == layout.n_points):
            self.stats[:] = 0
            self.characteristic_points[:] = 0
            self.pixel_state[:] = PIXEL_PENDING
            self.img[:] = BACKGROUND
        else:
            self.stats, self.characteristic_points = self.current_figure().layout.allocate(*shape)
            self.pixel_state = np.full(shape, PIXEL_PENDING, dtype=np.uint8)
            self.img = np.empty(shape + (3,), dtype=np.uint8)
            self.img[:] = BACKGROUND
            self.image_for_export = np.empty_like(self.img)        # img + the outlines; this is what gets exported/saved, so
                                                        # it must never hold a preview-only decoration
            self.image_for_screen = np.empty_like(self.img)   # self.image_for_export + the coloration-name overlay; only ever
                                                           # pushed to self.photo, never written to disk
        self.gx, self.gy = make_grid(self.img_w, self.img_h, self.image_world_rect)

    def pixel_center(self, i, j):
        """The world coordinates of the centre of pixel (i, j) of the image on screen: where its ray starts (the same
        arithmetic as trace_rows's, so a hover ray is exactly the one the image has)."""
        world_rect = self.image_world_rect
        return (world_rect.x_start + (i + 0.5) * ((world_rect.x_stop - world_rect.x_start) / self.img_w),
                world_rect.y_stop - (j + 0.5) * ((world_rect.y_stop - world_rect.y_start) / self.img_h))

    def world_to_canvas(self, x, y):
        """The canvas (pixel) coordinates of the world point (x, y) - numbers or arrays - on the image on screen."""
        world_rect = self.image_world_rect
        return ((x - world_rect.x_start) * (self.img_w / (world_rect.x_stop - world_rect.x_start)),
                (world_rect.y_stop - y) * (self.img_h / (world_rect.y_stop - world_rect.y_start)))

    def set_preview_image(self, rgb):
        """Show `rgb` (an (H, W, 3) uint8 array) bilinearly resized into the
        frame-preview box, or the 'preview will go here' placeholder if rgb
        is None (i.e. nothing is currently being rendered)."""
        if rgb is None:
            small = preview_placeholder(VIDEO_PREVIEW_SIZE, VIDEO_PREVIEW_SIZE)
        else:
            small = bilinear_resize(rgb, VIDEO_PREVIEW_SIZE, VIDEO_PREVIEW_SIZE)
        photo = tk.PhotoImage(data=encode_png_bytes(small), format="png")
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo            # keep a reference alive

    def obstacle_ring_specs(self):
        """[ObstacleRing, ...] for obstacles to draw a solid two-layer outline for:
        the frozen set if an image has been generated (matching what was actually
        simulated/saved), else the live (current) set."""
        if self.generated is not None:
            g = self.generated.engine_args.obstacles
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
        then the live coloration's name in the bottom-left corner - a preview-only
        decoration drawn into self.image_for_screen, a separate buffer from
        self.image_for_export (which is what gets exported/saved, and so must never
        carry it)."""
        np.copyto(self.image_for_export, self.img)
        outline = self.deco.get("outline")
        if outline is None or outline.get():
            for ring in self.obstacle_ring_specs():
                draw_obstacle_outline(self.image_for_export, ring.shape, ring.bound, self.gx - ring.cx,
                                      self.gy - ring.cy, ring.in_effect, ring.out_effect, self.outline_pixel)
        np.copyto(self.image_for_screen, self.image_for_export)
        self.draw_coloration_name_overlay()
        header = f"P6 {self.img_w} {self.img_h} 255\n".encode()
        self.photo.configure(data=header + self.image_for_screen.tobytes(), format="PPM")

    def draw_coloration_name_overlay(self):
        """Stamps the live coloration's display name ("coloration N") into the bottom-left
        corner of self.image_for_screen, in the tiny 5x7 pixel font, on a solid
        background box so it stays legible over any image content. Never
        touches self.image_for_export - this is preview-only and must never end up in
        a saved/exported image."""
        idx = self.all_coloration_ids().index(self.live_coloration().id)
        mask = render_text_line(f"coloration {idx}", COLORATION_NAME_OVERLAY_SCALE)
        h, w = mask.shape
        pad = COLORATION_NAME_OVERLAY_PAD
        margin = COLORATION_NAME_OVERLAY_MARGIN
        y1 = self.img_h - margin
        y0 = y1 - h - 2 * pad
        x0 = margin
        x1 = x0 + w + 2 * pad
        if y0 < 0 or x1 > self.img_w:
            return                                 # the image is too small to hold it
        self.image_for_screen[y0:y1, x0:x1] = COLORATION_NAME_OVERLAY_BG
        region = self.image_for_screen[y0 + pad:y0 + pad + h, x0 + pad:x0 + pad + w]
        region[mask] = COLORATION_NAME_OVERLAY_FG

    # ---- hover demonstration ---------------------------------------------- #
    def preview_settings(self):
        """Settings for previews: those of the image being generated, or the
        current entries/controls (last valid values) if nothing has been started."""
        if self.generated is not None:
            return self.generated.ray
        return RaySettings(seg_mult=self.seg_mult_scale.get(), fuzz=self.read_fuzz_controls(),
                        emit_mode=EMIT_MODE_NAMES.index(self.emit_mode_var.get()),
                        emit_target_r=self.last_emit_target_r,
                        emit_target_theta=self.last_emit_target_theta,
                        emit_target_cx=self.last_emit_target_cx,
                        emit_target_cy=self.last_emit_target_cy,
                        emit_offset=round(float(self.emit_offset_scale.get()), 2))

    def preview_natural_termination_settings(self):
        """The natural termination settings for previews, chosen the same way as preview_settings():
        those of the image being generated, or the controls' last valid values."""
        if self.generated is not None:
            return self.generated.natural_termination
        return NaturalTerminationSettings(
            max_iter=self.last_iter, max_dist=self.last_dist, escape_radius=self.last_escape_radius,
            confine_lookback=self.last_confine_lookback,
            confine_method=CONFINE_METHOD_NAMES.index(self.confine_method_var.get()),
            confine_threshold=self.last_confine_threshold)

    def preview_engine_settings(self):
        """The path engine for previews, chosen the same way as preview_settings(): that of the image being generated, or the
        dropdown's."""
        return self.generated.engine if self.generated is not None else self.current_engine_settings()

    def invalidate_preview(self):
        """Called whenever a control that feeds preview_settings() changes,
        before the first Start. A pinned preview is recomputed immediately (so
        it tracks the new settings even though the mouse hasn't moved); a live
        hover is just cleared, to be recomputed on the next mouse move."""
        if self.generated is not None:
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
        if not (0 <= i < self.img_w and 0 <= j < self.img_h):
            return
        x, y = self.pixel_center(i, j)
        ps = self.preview_settings()
        obstacles = build_obstacle_arrays(self.obstacles)
        if (self.pinned_px != (i, j) and not fires_ray(x, y, obstacles)):
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
        if not (0 <= i < self.img_w and 0 <= j < self.img_h):
            self.clear_overlay()
            return
        x, y = self.pixel_center(i, j)
        ps = self.preview_settings()
        obstacles = build_obstacle_arrays(self.obstacles)
        # A hovered pixel that fires no ray has no preview - but a pinned one
        # stays visible even if the scene changed under it so that it no longer
        # fires: it then shows the ray such a pixel would trace anyway.
        if (self.pinned_px != (i, j) and not fires_ray(x, y, obstacles)):
            self.clear_overlay()                   # not a pixel that fires a ray
            return
        if (i, j) == self.hover_px:
            return

        self.hover_px = (i, j)
        pn = self.preview_natural_termination_settings()
        if ps.fuzz.enabled:
            np.random.seed(ps.fuzz.rng_seed + i * 1000003 + j * 97)
        path_points, path_point_tags = self.hover_path_points, self.hover_path_point_tags
        raw, vec, ray_characteristic_points = self.hover_raw, self.hover_vec, self.hover_ray_characteristic_points
        engine = self.preview_engine_settings().engine
        args = path_args(engine, settings_ray_args(ps), obstacles, tables=tables_of_engine(engine))
        term, n = trace_engine_path(x, y, path_points, path_point_tags, settings_natural_termination(pn), args, raw)
        measure_path(engine, path_points, n, path_point_tags, raw, pn.max_iter, ps.seg_mult, args.tables, vec, ray_characteristic_points)
        layout = layout_of_engine(engine)
        vals = self.component_values(vec, ray_characteristic_points, x, y, layout)
        self.draw_segments(path_points, path_point_tags, n)
        self.draw_corkscrew(path_points, path_point_tags, n)

        # ---- overlay on the canvas (never part of the image) ----
        xs, ys = self.world_to_canvas(path_points[:n, 0], path_points[:n, 1])
        coords = np.column_stack((xs, ys)).ravel().tolist()
        show = {key: var.get() for key, var in self.deco.items()}

        cv = self.canvas
        cv.delete("ray")

        # Semi-transparent grey guide lines through the end of the ray
        if show["guides"]:
            gx = min(max(int(xs[-1]), 0), self.img_w - 1)
            gy = min(max(int(ys[-1]), 0), self.img_h - 1)
            if self.hline_img is not None:
                cv.create_image(0, gy, anchor="nw", image=self.hline_img, tags="ray")
                cv.create_image(gx, 0, anchor="nw", image=self.vline_img, tags="ray")
            else:                                   # no alpha support: stipple instead
                cv.create_line(0, gy + 0.5, self.img_w, gy + 0.5, fill=GUIDE_FALLBACK_HEX,
                               stipple="gray50", tags="ray")
                cv.create_line(gx + 0.5, 0, gx + 0.5, self.img_h, fill=GUIDE_FALLBACK_HEX,
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
            live_coloration = self.live_coloration()
            for c in range(3):
                if self.mapping[c] in POINT_METRICS:
                    base_rad = 8 - 2 * c
                    for p, rad in ((live_coloration.point1_choice[c], base_rad),
                                  (live_coloration.point2_choice[c], base_rad + 3)):
                        px, py = resolve_point_xy(ray_characteristic_points, p)
                        cx, cy = self.world_to_canvas(px, py)
                        cv.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, width=2,
                                      outline=SAMPLE_COLORS[live_coloration.color_model][c], tags="ray")

        # ---- info panel ----
        bounces = int(raw[RAW_BOUNCES])
        n_boundary = sum(in_sequence(SEQ_BOUNDARY, k, path_point_tags) for k in range(n))
        fx, fy = float(path_points[n - 1, 0]), float(path_points[n - 1, 1])
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
        letters = [name[0] for name, _color in COLOR_MODELS[self.live_coloration().color_model]]
        v["components"].set("   ".join(f"{letters[c]} {vals[c]:.3f}" for c in range(3)))
        v["color"].set(hexcol)
        for s in range(N_STATS):
            self.stat_vars[s].set(f"{vec[layout.stat_position(s)]:.3f}" if layout.has_stat(s) else "-")
        self.swatch.configure(bg=hexcol)
        self.info_placeholder.grid_remove()
        self.info_rows.grid()

    def seed_tab_focused(self):
        """Whether the Buddha seeding tab is the one being looked at (its tab of the Buddha tab, which is itself
        the selected tab of the controls)."""
        return (self.controls_notebook.select() == str(self.buddha_tab)
                and self.buddha_notebook.select() == str(self.buddha_seeding_tab))

    def update_seed_zone_outline(self):
        """Dashed purple outline of the zone the Buddha seed points are spread over, on the preview, while - and only
        while - the Buddha seeding tab is being looked at. (It is not a decoration with a checkbox of its own, for
        now.) The zone may reach outside the image."""
        if not hasattr(self, "canvas") or not hasattr(self, "buddha_seeding_tab"):
            return                                 # the window is still being built
        self.canvas.delete("seedzone")
        if not self.seed_tab_focused():
            return
        zone = self.current_buddha_seed_settings()
        left, top = self.world_to_canvas(zone.x_start, zone.y_stop)
        right, bottom = self.world_to_canvas(zone.x_stop, zone.y_start)
        self.canvas.create_rectangle(left, top, right, bottom,
                                     outline=SEED_ZONE_OUTLINE, width=2, dash=(6, 4), tags="seedzone")

    def on_leave(self, _event):
        if self.pinned_px is None:                 # a pinned preview stays put
            self.clear_overlay()

    def clear_overlay(self):
        self.canvas.delete("ray")
        self.hover_px = None
        self.info_rows.grid_remove()
        self.info_placeholder.grid()
        self.seg_canvas.delete("all")
        self.clear_corkscrew()
