"""Turning the stored simulation data (or a freshly traced frame) into coloured images: a whole frame at any resolution, the current image composed from several views, one view's colours, and view thumbnails."""
import numpy as np
from .constants import BACKGROUND, OBST_EFFECT_RGB, TERM_STAT_VALUE
from .image_io import make_grid
from .kernels import N_POINTS, N_STATS, SIZE, ST_TERMINATION, render_rows_wh, scatter_multi_buddha
from .metrics import (
    BUDDHA_BRIGHTNESS_VALUES, BUDDHA_MULTI, BUDDHA_MULTI_ALL_MID, BUDDHA_MULTI_ALL_SELF,
    BUDDHA_MULTI_PROG_BOUNCE, BUDDHA_MULTI_PROG_MID, BUDDHA_POINT_CATALOG, CHAN_BUDDHA,
    NONE_ONE, NONE_ZERO, POINT_STATS, blur_counts_image, buddha_compress,
    build_buddha_blur_kernel, extract_point_xy, make_mapping, point_comparison_metric,
    scatter_points_raw, world_radius_to_pixels
)
from .settings import settings_natural_stop, settings_ray_args
from .shapes import build_obstacle_arrays, draw_obstacle_outline, ring_layer_masks
from .views import channels_to_rgb


class RenderMixin:
    def render_current_image(self, views):
        """The CURRENT (already fully-traced) simulation, colorized by each
        of `views` and composed side by side at the live preview's own
        resolution (SIZE x SIZE per view) - reusing self.stats/self.points/
        self.status as already computed, no re-simulation. Each view gets
        the outline decoration drawn in if enabled, exactly like
        self.frame/draw(). Only called for more than one view - the
        single-view case is just self.frame, already rendered."""
        images = []
        for v in views:
            images.append(self.colorize_view(v.settings(), self.stats, self.points, self.status,
                                             SIZE, SIZE, self.current_blur_settings(),
                                             self.gen_stop, self.gen_ray,
                                             self.settings_of_current_image().fuzz.seed))
        outline = self.deco.get("outline")
        if outline is None or outline.get():
            for img in images:
                for ring in self.obstacle_ring_specs():
                    draw_obstacle_outline(img, ring.shape, ring.bound, self.gx - ring.cx,
                                          self.gy - ring.cy, ring.in_effect, ring.out_effect)
        return images[0] if len(images) == 1 else np.concatenate(images, axis=1)

    def render_full_image(self, data, out_w, out_h, obst_arrays, views):
        """The full uint8 image for one frame, covering the same [-1, 1]
        world extent as the main image regardless of out_w/out_h - safe to
        call from a background thread (touches only numpy arrays, never a Tk
        widget). `data` is the resolved simulation (KeyframeData - shape,
        limits, fuzzing); `obst_arrays` is an ObstacleArrays built on the main
        thread beforehand.

        `views` is every view to render (see resolve_views()), in order -
        each one rendered as its own full (out_h, out_w, 3) image from the
        SAME traced rays (ray-tracing depends only on the simulation, never
        on how it's colored), then placed side by side: the returned image
        is (out_h, out_w * len(views), 3). With a single view (the ordinary
        case before this feature existed), this is exactly the one image
        it's always been."""
        stop = settings_natural_stop(data.natural_stop)
        ray = settings_ray_args(data.ray, obst_arrays)
        seed = data.ray.fuzz.seed
        stats = np.zeros((out_h, out_w, N_STATS), dtype=np.float32)
        points = np.zeros((out_h, out_w, N_POINTS, 2), dtype=np.float32)
        status = np.zeros((out_h, out_w), dtype=np.uint8)
        render_rows_wh(stats, points, status, 0, out_h, out_w, out_h, stop, ray, seed)

        images = [self.colorize_view(v, stats, points, status, out_w, out_h, data.buddha_blur,
                                     stop, ray, seed)
                 for v in views]

        if data.outline:
            gx, gy = make_grid(out_w, out_h)
            ob = obst_arrays
            obst_layers = [ring_layer_masks(ob.shapes[oi], ob.bounds[oi],
                                            gx - ob.cx[oi], gy - ob.cy[oi])
                           + (ob.effects[oi, 0], ob.effects[oi, 1])
                           for oi in range(len(ob.shapes))]
            for img in images:
                for outer, inner, in_e, out_e in obst_layers:
                    img[outer] = OBST_EFFECT_RGB[int(in_e)]
                    img[inner] = OBST_EFFECT_RGB[int(out_e)]

        return images[0] if len(images) == 1 else np.concatenate(images, axis=1)

    def colorize_view(self, v, stats, points, status, out_w, out_h, blur, stop, ray, seed_base):
        """One view's (out_h, out_w, 3) uint8 image (background color where no
        ray was computed, no outline - that's applied uniformly afterwards)
        from already-traced stats/points/status. `v` is a ViewSettings: this
        is the same coloring computation every view goes through, whichever
        one it is. `blur` (a BuddhaBlurSettings) is how Buddha channels are blurred;
        `stop` (a NaturalStopRule), `ray` (a RayArgs) and `seed_base` (the fuzzing seed) are the
        simulation's own, used only to re-trace the rays for the "multi" Buddha point options."""
        mapping = make_mapping(v.kinds, v.point_metric, v.other_metric, v.chan_src,
                               v.constant_choice)
        img = np.empty((out_h, out_w, 3), dtype=np.uint8)
        img[:] = BACKGROUND
        computed = status == 1
        n = int(computed.sum())
        if n:
            radius_px = world_radius_to_pixels(blur.radius, out_w, out_h)
            buddha_kernel = build_buddha_blur_kernel(radius_px, blur.type)
            vals = []
            for c in range(3):
                s = mapping[c]
                if s == CHAN_BUDDHA:
                    counts = np.zeros((out_h, out_w), dtype=np.float64)
                    want_multi = {BUDDHA_MULTI_ALL_SELF: False, BUDDHA_MULTI_ALL_MID: False,
                                 BUDDHA_MULTI_PROG_BOUNCE: False, BUDDHA_MULTI_PROG_MID: False}
                    term_stat = stats[..., ST_TERMINATION]
                    for i, on in enumerate(v.buddha_points[c]):
                        if not on:
                            continue
                        _label, style, ref, term_filter = BUDDHA_POINT_CATALOG[i]
                        if style == BUDDHA_MULTI:
                            want_multi[ref] = True
                            continue
                        mask = computed
                        if term_filter is not None:
                            target = TERM_STAT_VALUE[term_filter]
                            mask = mask & (np.abs(term_stat - target) < 0.01)
                        if not mask.any():
                            continue
                        xs = points[..., ref, 0][mask]
                        ys = points[..., ref, 1][mask]
                        col = np.floor((xs + 1.0) * (out_w / 2.0)).astype(np.int64)
                        row = np.floor((1.0 - ys) * (out_h / 2.0)).astype(np.int64)
                        valid = (col >= 0) & (col < out_w) & (row >= 0) & (row < out_h)
                        scatter_points_raw(counts, row, col, valid)
                    if any(want_multi.values()):
                        scatter_multi_buddha(
                            0, out_h, stop, ray, seed_base,
                            want_multi[BUDDHA_MULTI_ALL_SELF], want_multi[BUDDHA_MULTI_ALL_MID],
                            want_multi[BUDDHA_MULTI_PROG_BOUNCE],
                            want_multi[BUDDHA_MULTI_PROG_MID], counts)
                    counts = blur_counts_image(counts, buddha_kernel)
                    sub = counts[computed].astype(np.float64)
                    mult = BUDDHA_BRIGHTNESS_VALUES[v.buddha_brightness[c]]
                    vals.append(buddha_compress(sub, v.buddha_compression[c], mult))
                elif s == NONE_ZERO:
                    vals.append(np.zeros(n))
                elif s == NONE_ONE:
                    vals.append(np.ones(n))
                elif s in POINT_STATS:
                    x1, y1 = extract_point_xy(points, v.point1_choice[c], computed, n)
                    x2, y2 = extract_point_xy(points, v.point2_choice[c], computed, n)
                    vals.append(point_comparison_metric(s, x1, y1, x2, y2))
                else:
                    vals.append(np.clip(stats[..., s][computed].astype(np.float64),
                                        0.0, 1.0))
            r, g, b = channels_to_rgb(v.mode, *vals)
            rgb = np.stack((r, g, b), axis=-1)
            img[computed] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return img

    def render_view_thumbnail(self, view_obj, size):
        """A size x size RGB thumbnail of how `view_obj` currently colors the
        live simulation, shrunk down (nearest-neighbor - a thumbnail doesn't
        need better). For the live view specifically, self.img already
        holds exactly this (kept current by recolor(), which every caller
        of rebuild_view_list runs first) - reusing it skips colorize_view
        entirely, which matters because a "multi" Buddha point option (All
        self bounces, segment midpoints, progressive means) makes
        colorize_view re-trace every fired ray from scratch. For any other
        (non-live) view, self.view_thumb_cache is checked next: as long as
        self.rows_done hasn't advanced since it was computed, that view's
        settings can't have changed either (only the live view is ever
        edited), so the cached thumbnail is still exactly correct - and
        reusing it is what keeps editing one view fast even when another
        view uses an expensive "multi" Buddha point option, since
        rebuild_view_list would otherwise redo that re-trace, unchanged,
        every single time regardless of which view was actually edited."""
        if view_obj is self.live_view():
            ys = (np.arange(size) * SIZE / size).astype(np.int64)
            xs = (np.arange(size) * SIZE / size).astype(np.int64)
            return self.img[np.ix_(ys, xs)]
        cached = self.view_thumb_cache.get(view_obj.id)
        if cached is not None and cached[0] == (self.rows_done, size):
            return cached[1]
        v = view_obj.settings()
        full = self.colorize_view(v, self.stats, self.points, self.status, SIZE, SIZE,
                                  self.current_blur_settings(), self.gen_stop, self.gen_ray,
                                  self.settings_of_current_image().fuzz.seed)
        ys = (np.arange(size) * SIZE / size).astype(np.int64)
        xs = (np.arange(size) * SIZE / size).astype(np.int64)
        thumb = full[np.ix_(ys, xs)]
        self.view_thumb_cache[view_obj.id] = ((self.rows_done, size), thumb)
        return thumb
