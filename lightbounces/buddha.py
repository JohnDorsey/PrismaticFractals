"""How Buddha counts are made and read. A Buddha channel colours a pixel by a visit count: a histogram of where chosen
points of many rays landed. This module makes the counts (from the characteristic points already stored, or from rays
traced again from the pixels or from a seed grid), blurs them, and compresses a count to a value in 0..1. It is made of
functions of arrays only - no window state - so the live image, an export and a video frame make their counts the same
way. What can be chosen (the catalog of points, the compression and brightness options) is in metrics.py."""
from math import ceil, pi
import numpy as np
from .constants import (
    BUDDHA_BLUR_BOKEH, BUDDHA_BLUR_CATADIOPTRIC_INNER_RATIO, BUDDHA_BLUR_GAUSSIAN,
    BUDDHA_BLUR_GAUSSIAN_TRUNCATE, BUDDHA_BLUR_TRIANGULAR, TERM_STAT_VALUE
)
from .kernels import ST_TERMINATION, WorldRect, scatter_from_seeds, scatter_multi_buddha
from .stored_layout import FULL_LAYOUT
from .metrics import (
    ALL_PATH_KINDS, BUDDHA_ARCTAN_SCALE, BUDDHA_COMPRESS_ARCTAN, BUDDHA_MULTI, BUDDHA_MULTI_ALL_MID, BUDDHA_MULTI_ALL_PATH,
    BUDDHA_MULTI_ALL_SELF, BUDDHA_MULTI_PROG_BOUNCE, BUDDHA_MULTI_PROG_MID, BUDDHA_POINT_CATALOG, PATH_KIND_OF_TERM,
    term_allowed_flags
)


def world_radius_to_pixels(radius_control, out_w, out_h):
    """The blur radius control's value -> a radius in pixels for an out_w x
    out_h output. The control is a multiple of the output's smallest
    dimension (the framing's whole picture, whatever world it shows), which for
    a same-scale (never-distorted) circular blur footprint is
    radius_px = radius_control * min(out_w, out_h)."""
    return radius_control * min(out_w, out_h)


def build_buddha_blur_kernel(radius_px, blur_type):
    """The (row_offset, col_offset, weight) neighborhood for scattering one
    point's weight of 1 across nearby pixels instead of just its own, as a
    float64 (K, 3) array with weights normalized to sum to 1 - or exactly
    [[0, 0, 1.0]] when radius_px is negligible (no blur at all, the
    original single-pixel behavior). Only Gaussian extends past radius_px
    itself (truncated at BUDDHA_BLUR_GAUSSIAN_TRUNCATE standard
    deviations); the other three kernels are exactly zero beyond it."""
    if radius_px < 0.5:
        return np.array([[0.0, 0.0, 1.0]])
    extent_mult = BUDDHA_BLUR_GAUSSIAN_TRUNCATE if blur_type == BUDDHA_BLUR_GAUSSIAN else 1.0
    extent = int(ceil(extent_mult * radius_px))
    rows, cols = [], []
    for dr in range(-extent, extent + 1):
        for dc in range(-extent, extent + 1):
            rows.append(dr)
            cols.append(dc)
    dr_arr = np.array(rows, dtype=np.float64)
    dc_arr = np.array(cols, dtype=np.float64)
    d = np.hypot(dr_arr, dc_arr)
    if blur_type == BUDDHA_BLUR_GAUSSIAN:
        w = np.exp(-(d * d) / (2.0 * radius_px * radius_px))
        w[d > BUDDHA_BLUR_GAUSSIAN_TRUNCATE * radius_px] = 0.0
    elif blur_type == BUDDHA_BLUR_TRIANGULAR:
        w = np.clip(1.0 - d / radius_px, 0.0, None)
    elif blur_type == BUDDHA_BLUR_BOKEH:
        w = (d <= radius_px).astype(np.float64)
    else:                                              # BUDDHA_BLUR_CATADIOPTRIC
        inner = BUDDHA_BLUR_CATADIOPTRIC_INNER_RATIO * radius_px
        w = ((d >= inner) & (d <= radius_px)).astype(np.float64)
    keep = w > 0.0
    total = w[keep].sum()
    if total <= 0.0:
        return np.array([[0.0, 0.0, 1.0]])      # degenerate (e.g. a sub-pixel ring) ->
                                                # fall back to no blur rather than
                                                # silently dropping the point entirely
    return np.stack([dr_arr[keep], dc_arr[keep], w[keep] / total], axis=1)


def scatter_points_raw(counts, rows, cols, valid):
    """Scatters a plain +1 at each of the given integer pixel positions
    (rows[valid], cols[valid]) into `counts` (in place) - cheap, no blur
    kernel involved at all. Blur (if any) is applied once, afterwards, to
    the whole accumulated histogram - see blur_counts_image - rather than
    per point, since summing many individually-blurred points is
    mathematically identical to blurring the sum once (convolution
    distributes over sums), and the latter costs nothing extra per point
    scattered."""
    np.add.at(counts, (rows[valid], cols[valid]), 1)


def blur_counts_image(counts, kernel):
    """Blurs the whole `counts` array once with `kernel` (the same (K, 3)
    offset+weight table build_buddha_blur_kernel returns), via FFT - so the
    cost depends on the image size, not on the kernel size (a large blur
    radius is no more expensive than a small one), unlike blurring each
    scattered point individually. Returns a new array; never mutates
    `counts`. The no-blur kernel ([[0, 0, 1.0]]) is a fast-path skip."""
    if kernel.shape[0] == 1 and kernel[0, 0] == 0.0 and kernel[0, 1] == 0.0:
        return counts
    h, w = counts.shape
    max_offset = int(np.max(np.abs(kernel[:, :2])))
    ksize = 2 * max_offset + 1
    dense = np.zeros((ksize, ksize))
    for dr, dc, wt in kernel:
        dense[int(dr) + max_offset, int(dc) + max_offset] = wt
    fft_h = h + ksize - 1
    fft_w = w + ksize - 1
    freq_counts = np.fft.rfft2(counts, s=(fft_h, fft_w))
    freq_kernel = np.fft.rfft2(dense, s=(fft_h, fft_w))
    full = np.fft.irfft2(freq_counts * freq_kernel, s=(fft_h, fft_w))
    return full[max_offset:max_offset + h, max_offset:max_offset + w]


def _engine_point(engine_args, point):
    """The position of characteristic point `point` (an id) in the per-path points of the engine `engine_args` (a PathArgs) is for."""
    position = int(engine_args.tables.point_pos[point])
    if position < 0:
        raise KeyError(f"the engine does not store the characteristic point {point}")
    return position


def scatter_seeded_buddha(counts, world_rect, seeds, row0, row1, chosen, termination, engine_args, rng_seed_base,
                          kinds=ALL_PATH_KINDS):
    """Scatter, into `counts` (in place; it covers the world rectangle `world_rect`), the Buddha points `chosen` (a channel's tuple
    of flags, one per entry of BUDDHA_POINT_CATALOG) of the rays fired from the seed grid `seeds` describes (a
    BuddhaSeedSettings: its zone, laid over counts' own rows and columns), for seed rows [row0, row1). Every chosen
    option - the characteristic points too - is made from those rays, not from the image's own; only from the rays whose
    path is of a kind in `kinds` (a channel's three flags, PATH_KIND_* order). Returns whether anything was chosen."""
    slots, end_terms = [], []
    want_multi = {BUDDHA_MULTI_ALL_SELF: False, BUDDHA_MULTI_ALL_MID: False,
                  BUDDHA_MULTI_PROG_BOUNCE: False, BUDDHA_MULTI_PROG_MID: False, BUDDHA_MULTI_ALL_PATH: False}
    for index, on in enumerate(chosen):
        if not on:
            continue
        _label, style, ref, term_filter = BUDDHA_POINT_CATALOG[index]
        if style == BUDDHA_MULTI:
            want_multi[ref] = True
        elif term_filter is None:
            slots.append(_engine_point(engine_args, ref))
        else:
            end_terms.append(term_filter)
    if not (slots or end_terms or any(want_multi.values())):
        return False
    scatter_from_seeds(row0, row1, WorldRect(seeds.x_start, seeds.x_stop, seeds.y_start, seeds.y_stop), world_rect,
                       termination, engine_args, rng_seed_base,
                       np.array(slots, dtype=np.int64), np.array(end_terms, dtype=np.int64),
                       want_multi[BUDDHA_MULTI_ALL_SELF], want_multi[BUDDHA_MULTI_ALL_MID],
                       want_multi[BUDDHA_MULTI_PROG_BOUNCE], want_multi[BUDDHA_MULTI_PROG_MID],
                       want_multi[BUDDHA_MULTI_ALL_PATH], term_allowed_flags(kinds), counts)
    return True


def buddha_compress(count, method, brightness_mult=1.0):
    """A visit count (a non-negative number, or an array of them) -> 0..1 with `method` (BUDDHA_COMPRESS_ARCTAN or
    _MODULUS), then multiplied by `brightness_mult` and clipped to 0..1 (a multiplier of 1 changes nothing)."""
    if method == BUDDHA_COMPRESS_ARCTAN:
        v = (2.0 / pi) * np.arctan(count / BUDDHA_ARCTAN_SCALE)
    else:
        v = (np.asarray(count) % 256) / 255.0          # BUDDHA_COMPRESS_MODULUS
    if brightness_mult != 1.0:
        v = np.clip(v * brightness_mult, 0.0, 1.0)
    return v


def blur_kernel(blur, out_w, out_h):
    """The blur kernel (see build_buddha_blur_kernel) for a BuddhaBlurSettings on an out_w x out_h image."""
    return build_buddha_blur_kernel(world_radius_to_pixels(blur.radius, out_w, out_h), blur.type)


def _kind_mask(term_stat, kinds):
    """Which pixels' stored termination metric (ST_TERMINATION) is the value of a reason whose kind of path is ticked in
    `kinds` (three flags, PATH_KIND_* order)."""
    mask = np.zeros(term_stat.shape, dtype=bool)
    for term, value in TERM_STAT_VALUE.items():
        if kinds[PATH_KIND_OF_TERM[term]]:
            mask |= np.abs(term_stat - value) < 0.01
    return mask


def scatter_stored_points(counts, chosen, row0, row1, world_rect, fired, stats, characteristic_points, layout=FULL_LAYOUT,
                          kinds=ALL_PATH_KINDS):
    """Scatter into `counts` (in place; the whole image's visit counts, covering `world_rect`) the characteristic
    points that rows [row0, row1) already stored, for the "simple" options in `chosen` (a channel's tuple of flags, one
    per entry of BUDDHA_POINT_CATALOG): one point per ray that fired, and for an option restricted to a termination
    reason only for rays that ended that way. `fired` is the rows' pixel mask, `stats` and `characteristic_points` their
    stored metrics and points (kept as `layout`, a stored_layout.StoredLayout, says; the termination metric is read only
    for an option that asks for a termination reason, and when `kinds` - a channel's three path-kind flags, PATH_KIND_*
    order - leaves a kind out, since a ray whose path is of such a kind scatters nothing). Returns whether any point was
    scattered."""
    h, w = counts.shape
    touched = False
    simple = any(on and BUDDHA_POINT_CATALOG[index][1] != BUDDHA_MULTI for index, on in enumerate(chosen))
    if not all(kinds) and simple:                 # (only the simple options read the stored data: the multi ones trace again)
        term_stat = stats[..., layout.stat_position(ST_TERMINATION)]
        fired = fired & _kind_mask(term_stat, kinds)
    for index, on in enumerate(chosen):
        if not on:
            continue
        _label, style, ref, term_filter = BUDDHA_POINT_CATALOG[index]
        if style == BUDDHA_MULTI:
            continue
        mask = fired
        if term_filter is not None:
            term_stat = stats[..., layout.stat_position(ST_TERMINATION)]
            mask = mask & (np.abs(term_stat - TERM_STAT_VALUE[term_filter]) < 0.01)
        if not mask.any():
            continue
        position = layout.point_position(ref)
        xs = characteristic_points[..., position, 0][mask]
        ys = characteristic_points[..., position, 1][mask]
        col = np.floor((xs - world_rect.x_start) * (w / (world_rect.x_stop - world_rect.x_start))).astype(np.int64)
        row = np.floor((world_rect.y_stop - ys) * (h / (world_rect.y_stop - world_rect.y_start))).astype(np.int64)
        valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        scatter_points_raw(counts, row, col, valid)
        touched = True
    return touched


def scatter_buddha_channel(counts, chosen, row0, row1, world_rect, seeds, fired, stats, characteristic_points,
                           termination, engine_args, rng_seed_base, layout=FULL_LAYOUT, kinds=ALL_PATH_KINDS):
    """Add to `counts` (in place; one Buddha channel's visit counts for the whole image, covering `world_rect`) the
    contribution of rows [row0, row1), for the options `chosen` (a tuple of flags, one per entry of
    BUDDHA_POINT_CATALOG). `seeds` (a BuddhaSeedSettings, or None for the camera pixels) says where the rays start:
    from the pixels of the image itself (the "simple" options then read the characteristic points the rows already
    stored - `fired`, `stats` and `characteristic_points` are those rows', kept as `layout` says - and the "multi" options trace the rays
    again), or from a seed grid over the seeding zone (every option is then made from rays traced there).
    `termination`, `engine_args` and `rng_seed_base` are the simulation's own, for tracing again. `kinds` (three flags,
    PATH_KIND_* order) limits the channel to the paths of the kinds ticked; all three ticked is no limit. Returns whether
    `counts` may have changed."""
    if seeds is not None and not seeds.camera_pixels:
        return scatter_seeded_buddha(counts, world_rect, seeds, row0, row1, chosen, termination, engine_args, rng_seed_base,
                                     kinds)
    if not fired.any():
        return False
    touched = scatter_stored_points(counts, chosen, row0, row1, world_rect, fired, stats, characteristic_points, layout, kinds)
    want_multi = {BUDDHA_MULTI_ALL_SELF: False, BUDDHA_MULTI_ALL_MID: False,
                  BUDDHA_MULTI_PROG_BOUNCE: False, BUDDHA_MULTI_PROG_MID: False, BUDDHA_MULTI_ALL_PATH: False}
    for index, on in enumerate(chosen):
        if on and BUDDHA_POINT_CATALOG[index][1] == BUDDHA_MULTI:
            want_multi[BUDDHA_POINT_CATALOG[index][2]] = True
    if any(want_multi.values()):
        scatter_multi_buddha(row0, row1, world_rect, termination, engine_args, rng_seed_base,
                             want_multi[BUDDHA_MULTI_ALL_SELF], want_multi[BUDDHA_MULTI_ALL_MID],
                             want_multi[BUDDHA_MULTI_PROG_BOUNCE], want_multi[BUDDHA_MULTI_PROG_MID],
                             want_multi[BUDDHA_MULTI_ALL_PATH], term_allowed_flags(kinds), counts)
        touched = True
    return touched
