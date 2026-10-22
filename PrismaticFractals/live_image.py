"""The image being generated, apart from any window: its framing and pixel grid, the per-pixel data the simulation stores for it, the colours
made from that data, and the Buddha visit counts. Everything here is plain numpy and takes its inputs as arguments (the figure, the
coloration and its mapping, the Buddha seeding and the blur), so a window, a script or another front end can drive it the same way: `advance`
traces rows, `recolor` turns stored rows into colours, `scatter_buddha_rows` / `blurred_buddha_counts` keep a Buddha channel's counts.

Pure: imports no Tk."""
import time
import numpy as np
from .buddha import blur_counts_image, blur_kernel, scatter_buddha_channel
from .color_pipeline import coloration_fits, paint
from .constants import BACKGROUND
from .framings import framing_image_size, framing_outline_pixel, framing_world_rect
from .image_io import make_grid
from .kernels import PIXEL_PENDING, PIXEL_TRACED, trace_rows
from .metrics import CHAN_BUDDHA, CHAN_SRC_BUDDHA


class LiveImage:
    """The image of one framing of a figure, filling up row by row. Its attributes are the pixel grid (`img_w`, `img_h`, the `world_rect` it
    covers, `gx` / `gy`: the world coordinates of every pixel centre, which are also the rays' start points, and the `outline_pixel` size
    that obstacle outlines are drawn with), the per-pixel data (`stats`, `characteristic_points`, `pixel_state`: laid out as the figure's
    `layout` says), the colours (`img`, uint8 RGB), `rows_done`, and per channel the Buddha visit counts (`buddha_counts`, raw) and their
    lazily blurred versions (`buddha_counts_blurred`)."""

    def __init__(self, framing, layout):
        self.stats = self.characteristic_points = self.pixel_state = self.img = None
        self.reset(framing, layout)

    def reset(self, framing, layout):
        """Become a blank image of `framing` (a FramingSnapshot) with the per-pixel data laid out as `layout` says: the grid takes the framing's
        size and world rectangle, and everything computed so far is discarded (`rows_done` is 0, the Buddha counts are gone). Buffers of the
        same shape and layout as the last are cleared in place, not made again. Returns whether new buffers were made (a window then makes
        its own, of the same shape)."""
        self.framing = framing
        self.world_rect = framing_world_rect(framing)
        self.img_w, self.img_h = framing_image_size(framing)
        self.outline_pixel = framing_outline_pixel(framing)
        self.rows_done = 0
        self.batch_rows = 1
        self.buddha_counts = [None, None, None]
        self.buddha_counts_blurred = [None, None, None]
        shape = (self.img_h, self.img_w)
        reuse = (self.pixel_state is not None and self.pixel_state.shape == shape
                 and self.stats.shape[-1] == layout.n_stats and self.characteristic_points.shape[-2] == layout.n_points)
        if reuse:
            self.stats[:] = 0
            self.characteristic_points[:] = 0
            self.pixel_state[:] = PIXEL_PENDING
            self.img[:] = BACKGROUND
        else:
            self.stats, self.characteristic_points = layout.allocate(*shape)
            self.pixel_state = np.full(shape, PIXEL_PENDING, dtype=np.uint8)
            self.img = np.empty(shape + (3,), dtype=np.uint8)
            self.img[:] = BACKGROUND
        self.gx, self.gy = make_grid(self.img_w, self.img_h, self.world_rect)
        return not reuse

    # ---- pixels and the world ---------------------------------------------------------------------- #
    def pixel_center(self, i, j):
        """The world coordinates of the centre of pixel (i, j): where its ray starts (the same arithmetic as trace_rows's, so a hover ray is
        exactly the one the image has)."""
        world_rect = self.world_rect
        return (world_rect.x_start + (i + 0.5) * ((world_rect.x_stop - world_rect.x_start) / self.img_w),
                world_rect.y_stop - (j + 0.5) * ((world_rect.y_stop - world_rect.y_start) / self.img_h))

    def to_canvas(self, x, y):
        """The pixel coordinates of the world point (x, y) - numbers or arrays - on this image."""
        world_rect = self.world_rect
        return ((x - world_rect.x_start) * (self.img_w / (world_rect.x_stop - world_rect.x_start)),
                (world_rect.y_stop - y) * (self.img_h / (world_rect.y_stop - world_rect.y_start)))

    # ---- the simulation ---------------------------------------------------------------------------- #
    def advance(self, figure, budget):
        """Trace rows of `figure` (a GeneratedFigure) for about `budget` seconds, adapting the batch size so that a batch takes about a third of
        it. Returns the first row traced (the rows from there to `rows_done` are new)."""
        t_start = time.perf_counter()
        target = budget / 3.0
        rows_before = self.rows_done
        while self.rows_done < self.img_h:
            t0 = time.perf_counter()
            row1 = min(self.rows_done + self.batch_rows, self.img_h)
            trace_rows(self.stats, self.characteristic_points, self.pixel_state, self.rows_done, row1, self.img_w, self.img_h,
                       self.world_rect, figure.termination, figure.engine_args, figure.rng_seed)
            self.rows_done = row1
            now = time.perf_counter()
            dt = max(now - t0, 1e-6)
            scale = min(max(target / dt, 0.5), 2.0)
            self.batch_rows = max(1, int(self.batch_rows * scale))
            if now - t_start >= budget:
                break
        return rows_before

    # ---- colours ----------------------------------------------------------------------------------- #
    def recolor(self, r0, r1, coloration, mapping, figure, seeds, blur):
        """Turn the stored metrics and characteristic points of rows [r0, r1) into pixel colours (see color_pipeline). `coloration` is the one
        shown, `mapping` its metric mapping, `figure` what the data was made from, `seeds` and `blur` the Buddha seeding and blur. Nothing
        happens while the data cannot hold what the coloration reads (it is of another engine: the image waits for Start)."""
        if r1 <= r0 or not coloration_fits(coloration, figure.layout, seeds):
            return
        computed = self.pixel_state[r0:r1] == PIXEL_TRACED
        blurred = [None] * 3
        if computed.any():
            for c in range(3):
                counts = self.blurred_buddha_counts(c, blur) if mapping[c] == CHAN_BUDDHA else None
                blurred[c] = None if counts is None else counts[r0:r1]
        paint(self.img[r0:r1], coloration, mapping, self.stats[r0:r1], self.characteristic_points[r0:r1], computed, blurred, figure.layout)

    # ---- the Buddha visit counts ------------------------------------------------------------------- #
    def scatter_buddha_rows(self, r0, r1, coloration, figure, seeds):
        """Add rows [r0, r1)'s contribution to the RAW (unblurred) visit-count buffer of every Buddha channel of `coloration` (see
        buddha.scatter_buddha_channel: the stored characteristic points for the "simple" options, rays traced again for the "multi" ones, or a
        seed grid when the seeding is not the camera's pixels). The blur is never applied here - it is applied once, lazily, to the whole
        accumulated buffer when a colour is needed (blurred_buddha_counts) - so this stays cheap whatever the blur radius; a channel whose raw
        counts changed has its cached blurred version dropped."""
        if r1 <= r0 or not any(s == CHAN_SRC_BUDDHA for s in coloration.chan_src):
            return
        if not coloration_fits(coloration, figure.layout, seeds):
            return
        fired = self.pixel_state[r0:r1] == PIXEL_TRACED
        stats = self.stats[r0:r1]
        characteristic_points = self.characteristic_points[r0:r1]
        for c in range(3):
            if coloration.chan_src[c] != CHAN_SRC_BUDDHA or self.buddha_counts[c] is None:
                continue
            if scatter_buddha_channel(self.buddha_counts[c], coloration.buddha_points[c], r0, r1, self.world_rect, seeds, fired, stats,
                                      characteristic_points, figure.termination, figure.engine_args, figure.rng_seed, figure.layout,
                                      coloration.buddha_path_kinds[c]):
                self.buddha_counts_blurred[c] = None

    def blurred_buddha_counts(self, c, blur):
        """Channel c's blurred Buddha counts (`blur`: a BuddhaBlurSettings), computed once - by a single whole-image FFT pass, whatever the number
        of points or the radius - and kept until the counts change or `invalidate_blur`. None if the channel is not in Buddha mode."""
        if self.buddha_counts[c] is None:
            return None
        if self.buddha_counts_blurred[c] is None:
            kernel = blur_kernel(blur, self.img_w, self.img_h)
            self.buddha_counts_blurred[c] = blur_counts_image(self.buddha_counts[c], kernel)
        return self.buddha_counts_blurred[c]

    def invalidate_blur(self):
        """The blur settings changed: every channel's blurred counts are stale."""
        self.buddha_counts_blurred = [None, None, None]

    def rebuild_buddha_counts(self, coloration, figure, seeds):
        """Reset and scatter again every Buddha channel of `coloration` from every row computed so far. Used when the Buddha configuration
        itself changes (which points, which channels are in Buddha mode) - the counts would be for the wrong selection - and when a fresh
        generation starts."""
        self.buddha_counts = [np.zeros((self.img_h, self.img_w), dtype=np.float64) if s == CHAN_SRC_BUDDHA else None
                              for s in coloration.chan_src]
        self.buddha_counts_blurred = [None, None, None]
        self.scatter_buddha_rows(0, self.rows_done, coloration, figure, seeds)
