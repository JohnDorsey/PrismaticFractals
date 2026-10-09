"""The colour chain, once: from the stored data of pixels to their colours. A coloration maps each of its three channels
to a value in 0..1 (a metric of the ray, a distance between two of its characteristic points, a constant, or a
compressed Buddha visit count), and its color model turns the three values into a colour. The live image, an export, a
thumbnail and the number panel of one hovered ray all go through these functions (a hovered ray is an image of one
pixel). They are functions of arrays only; the Buddha counts they read are made in buddha.py."""
import numpy as np
from .constants import BACKGROUND
from .colorations import channels_to_rgb
from .metrics import (
    BUDDHA_BRIGHTNESS_VALUES, CHAN_BUDDHA, NONE_ONE, NONE_ZERO, POINT_METRICS, extract_point_xy,
    point_comparison_metric
)
from .buddha import buddha_compress
from .kernels import ST_TERMINATION
from .metrics import BUDDHA_MULTI, BUDDHA_POINT_CATALOG, CONSTANT_POINT_COORDS, make_mapping
from .stored_layout import FULL_LAYOUT, StoredLayout


def channel_values(coloration, mapping, stats, characteristic_points, computed, blurred_counts, layout=FULL_LAYOUT):
    """The three channels' values (each an array of 0..1 values, one per pixel in `computed`) of a coloration (a
    ColorationSettings or a ColorationConfig). `mapping` is what each channel uses (metrics.make_mapping), `stats` and
    `characteristic_points` the stored data of the pixels (any leading shape, which `computed` - a bool array of that
    shape: the pixels that have data - selects from), and `blurred_counts` a list of three arrays of the same leading
    shape: the blurred Buddha visit counts of each Buddha channel (None for any other, or when it has none: that
    channel is 0). `layout` (a stored_layout.StoredLayout) says which slots `stats` and `characteristic_points` keep;
    the default is all of them."""
    n = int(computed.sum())
    vals = []
    for c in range(3):
        s = mapping[c]
        if s == CHAN_BUDDHA:
            counts = blurred_counts[c]
            if counts is None:
                vals.append(np.zeros(n))
            else:
                mult = BUDDHA_BRIGHTNESS_VALUES[coloration.buddha_brightness[c]]
                vals.append(buddha_compress(counts[computed].astype(np.float64), coloration.buddha_compression[c], mult))
        elif s == NONE_ZERO:
            vals.append(np.zeros(n))
        elif s == NONE_ONE:
            vals.append(np.ones(n))
        elif s in POINT_METRICS:
            x1, y1 = extract_point_xy(characteristic_points, coloration.point1_choice[c], computed, n, layout)
            x2, y2 = extract_point_xy(characteristic_points, coloration.point2_choice[c], computed, n, layout)
            vals.append(point_comparison_metric(s, x1, y1, x2, y2))
        else:
            vals.append(np.clip(stats[..., layout.stat_position(s)][computed].astype(np.float64), 0.0, 1.0))
    return vals


def color_pixels(coloration, mapping, stats, characteristic_points, computed, blurred_counts, layout=FULL_LAYOUT):
    """The colours, an (n, 3) uint8 array, of the n pixels in `computed` (see channel_values for the arguments)."""
    r, g, b = channels_to_rgb(coloration.color_model,
                              *channel_values(coloration, mapping, stats, characteristic_points, computed, blurred_counts, layout))
    return np.clip(np.rint(np.stack((r, g, b), axis=-1)), 0, 255).astype(np.uint8)


def paint(out, coloration, mapping, stats, characteristic_points, computed, blurred_counts, layout=FULL_LAYOUT):
    """Fill `out` (an (h, w, 3) uint8 image) with the background colour, and the pixels in `computed` with their
    colours (see channel_values for the arguments)."""
    out[:] = BACKGROUND
    if computed.any():
        out[computed] = color_pixels(coloration, mapping, stats, characteristic_points, computed, blurred_counts, layout)


def ray_values(coloration, mapping, metrics, ray_characteristic_points, blurred_counts_at_pixel, layout=FULL_LAYOUT):
    """The three channel values (floats in 0..1) of ONE ray: `metrics` is its array of metrics,
    `ray_characteristic_points` its array of points (both kept as `layout` says: all of the ray engine's by default), and
    `blurred_counts_at_pixel` a list of three: for
    each Buddha channel the blurred visit count at the pixel the ray was fired from (None for any other, or when it
    has none)."""
    computed = np.ones((1, 1), dtype=bool)
    counts = [None if count is None else np.full((1, 1), count, dtype=np.float64) for count in blurred_counts_at_pixel]
    return [float(v[0]) for v in channel_values(coloration, mapping, np.asarray(metrics, dtype=np.float64)[None, None, :],
                                                np.asarray(ray_characteristic_points)[None, None, :, :], computed, counts, layout)]


def demand_layout(colorations, seeds=None):
    """The StoredLayout that keeps exactly the slots the colorations (ColorationSettings, as an export renders them) read,
    for Buddha seeding `seeds` (a BuddhaSeedSettings; None means the camera's pixels): the characteristic points and
    metrics of the channels that read them (channel_values), and what a Buddha channel scatters
    (buddha.scatter_stored_points: its simple options' points, and the termination metric for an option restricted to a
    termination reason or a channel limited to some kinds of path - the "multi" options and a seed grid read nothing that
    is stored). Whatever channel_values or
    scatter_stored_points reads must be listed here;
    test_a_demand_layout_keeps_everything_the_colour_chain_reads compares the two on random colorations."""
    stat_slots, point_slots = set(), set()
    camera = seeds is None or seeds.camera_pixels
    for v in colorations:
        mapping = make_mapping(v.metric_types, v.point_metric, v.other_metric, v.chan_src, v.constant_choice)
        for c in range(3):
            s = mapping[c]
            if s == CHAN_BUDDHA:
                if not camera:
                    continue
                for index, on in enumerate(v.buddha_points[c]):
                    if on:
                        _label, style, ref, term_filter = BUDDHA_POINT_CATALOG[index]
                        if style != BUDDHA_MULTI:
                            point_slots.add(ref)
                            if term_filter is not None or not all(v.buddha_path_kinds[c]):
                                stat_slots.add(ST_TERMINATION)      # (an option for one reason, or a channel limited to some kinds of path)
            elif s in (NONE_ZERO, NONE_ONE):
                continue
            elif s in POINT_METRICS:
                point_slots.update(p for p in (v.point1_choice[c], v.point2_choice[c]) if p not in CONSTANT_POINT_COORDS)
            else:
                stat_slots.add(s)
    return StoredLayout.of(stat_slots, point_slots)
