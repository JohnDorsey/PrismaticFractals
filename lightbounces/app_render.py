"""Turning the stored simulation data (or a freshly traced frame) into coloured images: a whole frame at any resolution, the current image composed from several colorations, one coloration's colours, and coloration thumbnails."""
import numpy as np
from .constants import BACKGROUND, OBST_EFFECT_RGB, THUMB_BACKGROUND
from .image_io import make_grid
from .framings import framing_image_size, framing_outline_pixel, framing_world_rect, same_framing
from .kernels import N_POINTS, N_STATS, PIXEL_TRACED, ST_TERMINATION, WORLD_RECT_DEFAULT, trace_rows
from .metrics import CHAN_BUDDHA, make_mapping
from .buddha import blur_counts_image, blur_kernel, scatter_buddha_channel
from .color_pipeline import paint
from .settings import settings_natural_termination, settings_ray_args
from .shapes import draw_obstacle_outline, ring_layer_masks


class RenderMixin:
    def render_current_image(self, colorations):
        """The CURRENT (already fully-traced) simulation, colorized by each
        of `colorations` and composed side by side at the live preview's own
        resolution (img_w x img_h per coloration) - reusing self.stats/self.characteristic_points/
        self.pixel_state as already computed, no re-simulation. Each coloration gets
        the outline decoration drawn in if enabled, exactly like
        self.image_for_export/draw(). Only called for more than one coloration - the
        single-coloration case is just self.image_for_export, already rendered."""
        images = []
        for v in colorations:
            images.append(self.apply_coloration(v.settings(), self.stats, self.characteristic_points, self.pixel_state,
                                             self.img_w, self.img_h, self.current_blur_settings(),
                                             self.gen_termination, self.gen_engine_args,
                                             self.settings_of_current_image().fuzz.rng_seed,
                                             self.current_buddha_seed_settings(), self.image_world_rect))
        outline = self.deco.get("outline")
        if outline is None or outline.get():
            for img in images:
                for ring in self.obstacle_ring_specs():
                    draw_obstacle_outline(img, ring.shape, ring.bound, self.gx - ring.cx,
                                          self.gy - ring.cy, ring.in_effect, ring.out_effect, self.outline_pixel)
        return images[0] if len(images) == 1 else np.concatenate(images, axis=1)

    def render_full_image(self, data, out_w, out_h, obst_arrays, colorations, framing):
        """The full uint8 image for one frame, covering the world rectangle of `framing` (a FramingSnapshot)
        regardless of out_w/out_h (a shape that differs from the framing's stretches it) - safe to
        call from a background thread (touches only numpy arrays, never a Tk
        widget). `data` is the resolved simulation (KeyframeData - shape,
        limits, fuzzing); `obst_arrays` is an ObstacleArrays built on the main
        thread beforehand.

        `colorations` is every coloration to render (see resolve_colorations()), in order -
        each one rendered as its own full (out_h, out_w, 3) image from the
        SAME traced rays (ray-tracing depends only on the simulation, never
        on how it's colored), then placed side by side: the returned image
        is (out_h, out_w * len(colorations), 3). With a single coloration (the ordinary
        case before this feature existed), this is exactly the one image
        it's always been."""
        termination = settings_natural_termination(data.natural_termination)
        engine_args = settings_ray_args(data.ray, obst_arrays)
        rng_seed = data.ray.fuzz.rng_seed
        stats = np.zeros((out_h, out_w, N_STATS), dtype=np.float32)
        characteristic_points = np.zeros((out_h, out_w, N_POINTS, 2), dtype=np.float32)
        pixel_state = np.zeros((out_h, out_w), dtype=np.uint8)
        world_rect = framing_world_rect(framing)
        trace_rows(stats, characteristic_points, pixel_state, 0, out_h, out_w, out_h, world_rect, termination, engine_args, rng_seed)

        images = [self.apply_coloration(v, stats, characteristic_points, pixel_state, out_w, out_h, data.buddha_blur,
                                     termination, engine_args, rng_seed, data.buddha_seed, world_rect)
                 for v in colorations]

        if data.outline:
            gx, gy = make_grid(out_w, out_h, world_rect)
            ob = obst_arrays
            pixel = framing_outline_pixel(framing)
            obst_layers = [ring_layer_masks(ob.shapes[oi], ob.bounds[oi],
                                            gx - ob.cx[oi], gy - ob.cy[oi], pixel=pixel)
                           + (ob.effects[oi, 0], ob.effects[oi, 1])
                           for oi in range(len(ob.shapes))]
            for img in images:
                for outer, inner, in_e, out_e in obst_layers:
                    img[outer] = OBST_EFFECT_RGB[int(in_e)]
                    img[inner] = OBST_EFFECT_RGB[int(out_e)]

        return images[0] if len(images) == 1 else np.concatenate(images, axis=1)

    def apply_coloration(self, v, stats, characteristic_points, pixel_state, out_w, out_h, blur, termination, engine_args, rng_seed_base, seeds=None,
                         world_rect=WORLD_RECT_DEFAULT):
        """One coloration's (out_h, out_w, 3) uint8 image (background color where no
        ray was computed, no outline - that's applied uniformly afterwards)
        from already-traced stats/characteristic_points/pixel_state. `v` is a ColorationSettings: this
        is the same coloring computation every coloration goes through, whichever
        one it is. `blur` (a BuddhaBlurSettings) is how Buddha channels are blurred;
        `termination` (a NaturalTerminationRule), `engine_args` (a RayArgs) and `rng_seed_base` (the fuzzing seed) are the
        simulation's own, used only to re-trace the rays for the "multi" Buddha point options. `seeds` (a
        BuddhaSeedSettings; None means the camera pixels) says where the rays that Buddha channels scatter start: by
        default the pixels of this very image, whose characteristic points serve the single-point options; otherwise a seed
        grid with this image's rows and columns over the seeding zone, every chosen point of which comes from
        rays traced there. `world_rect` (a WorldRect) is the world rectangle the image covers."""
        mapping = make_mapping(v.metric_types, v.point_metric, v.other_metric, v.chan_src,
                               v.constant_choice)
        img = np.empty((out_h, out_w, 3), dtype=np.uint8)
        computed = pixel_state == PIXEL_TRACED
        blurred = [None] * 3
        if computed.any():
            kernel = None
            term_stat = stats[..., ST_TERMINATION]
            for c in range(3):
                if mapping[c] == CHAN_BUDDHA:
                    if kernel is None:
                        kernel = blur_kernel(blur, out_w, out_h)
                    counts = np.zeros((out_h, out_w), dtype=np.float64)
                    scatter_buddha_channel(counts, v.buddha_points[c], 0, out_h, world_rect, seeds, computed, term_stat,
                                           characteristic_points, termination, engine_args, rng_seed_base)
                    blurred[c] = blur_counts_image(counts, kernel)
        paint(img, v, mapping, stats, characteristic_points, computed, blurred)
        return img

    def render_coloration_thumbnail(self, coloration_obj, size):
        """A size x size RGB thumbnail of how `coloration_obj` currently colors the
        live simulation, shrunk down (nearest-neighbor - a thumbnail doesn't
        need better). For the live coloration specifically, self.img already
        holds exactly this (kept current by recolor(), which every caller
        of rebuild_coloration_list runs first) - reusing it skips apply_coloration
        entirely, which matters because a "multi" Buddha point option (All
        self bounces, segment midpoints, progressive means) makes
        apply_coloration re-trace every fired ray from scratch. For any other
        (non-live) coloration, self.coloration_thumb_cache is checked next: as long as
        self.rows_done hasn't advanced since it was computed, that coloration's
        settings can't have changed either (only the live coloration is ever
        edited), so the cached thumbnail is still exactly correct - and
        reusing it is what keeps editing one coloration fast even when another
        coloration uses an expensive "multi" Buddha point option, since
        rebuild_coloration_list would otherwise redo that re-trace, unchanged,
        every single time regardless of which coloration was actually edited.

        A coloration's thumbnail is always of framing 0, though: when the image on screen is of another framing,
        its data is of no use here, and the thumbnail is a small simulation of framing 0's own
        (render_thumbnail)."""
        framing_zero = self.framings[0]
        if not same_framing(self.image_framing, framing_zero):
            return self.render_thumbnail(framing_zero, coloration_obj.settings(), size)
        if coloration_obj is self.live_coloration():
            ys = (np.arange(size) * self.img_h / size).astype(np.int64)
            xs = (np.arange(size) * self.img_w / size).astype(np.int64)
            return self.img[np.ix_(ys, xs)]
        cached = self.coloration_thumb_cache.get(coloration_obj.id)
        if cached is not None and cached[0] == (self.rows_done, size):
            return cached[1]
        v = coloration_obj.settings()
        full = self.apply_coloration(v, self.stats, self.characteristic_points, self.pixel_state, self.img_w, self.img_h,
                                  self.current_blur_settings(), self.gen_termination, self.gen_engine_args,
                                  self.settings_of_current_image().fuzz.rng_seed,
                                  self.current_buddha_seed_settings(), self.image_world_rect)
        ys = (np.arange(size) * self.img_h / size).astype(np.int64)
        xs = (np.arange(size) * self.img_w / size).astype(np.int64)
        thumb = full[np.ix_(ys, xs)]
        self.coloration_thumb_cache[coloration_obj.id] = ((self.rows_done, size), thumb)
        return thumb

    def simulated_thumbnail_data(self, framing, size):
        """The generated figure as `framing` sees it, simulated on a grid that fits a `size` pixel square:
        (stats, characteristic_points, pixel_state, world_rect, width, height). Made once per framing and size until the next generation
        (self.thumb_sims), since every coloration's thumbnail of it shares it."""
        key = (framing.x, framing.y, framing.diagonal, framing.aspect, size)
        data = self.thumb_sims.get(key)
        if data is None:
            w, h = framing_image_size(framing, size)
            world_rect = framing_world_rect(framing)
            stats = np.zeros((h, w, N_STATS), dtype=np.float32)
            characteristic_points = np.zeros((h, w, N_POINTS, 2), dtype=np.float32)
            pixel_state = np.zeros((h, w), dtype=np.uint8)
            trace_rows(stats, characteristic_points, pixel_state, 0, h, w, h, world_rect, self.gen_termination, self.gen_engine_args,
                           self.settings_of_current_image().fuzz.rng_seed)
            data = (stats, characteristic_points, pixel_state, world_rect, w, h)
            self.thumb_sims[key] = data
        return data

    def render_thumbnail(self, framing, coloration_settings, size):
        """A size x size RGB thumbnail of what `framing` sees of the generated figure, coloured by a
        ColorationSettings: a small simulation of its own (see simulated_thumbnail_data), in the framing's shape
        centred on a grey square. Before anything has been generated, the blank image in that shape."""
        picture = np.empty((size, size, 3), dtype=np.uint8)
        picture[:] = THUMB_BACKGROUND
        w, h = framing_image_size(framing, size)
        y0, x0 = (size - h) // 2, (size - w) // 2
        if self.generated_settings is None:
            picture[y0:y0 + h, x0:x0 + w] = BACKGROUND
            return picture
        stats, characteristic_points, pixel_state, world_rect, w, h = self.simulated_thumbnail_data(framing, size)
        picture[y0:y0 + h, x0:x0 + w] = self.apply_coloration(
            coloration_settings, stats, characteristic_points, pixel_state, w, h, self.current_blur_settings(), self.gen_termination,
            self.gen_engine_args, self.settings_of_current_image().fuzz.rng_seed, self.current_buddha_seed_settings(), world_rect)
        return picture

    def render_framing_thumbnail(self, framing_obj, size):
        """A framing's thumbnail: what it sees, coloured by coloration 0."""
        return self.render_thumbnail(framing_obj, self.colorations[0].settings(), size)
