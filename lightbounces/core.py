"""Everything that has not yet been moved into a module of its own (see dev/refactor_plan.md).

The program's feature description is the docstring of the package (lightbounces/__init__.py)."""


import dataclasses
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import traceback
import time
import tkinter as tk
import tkinter.font as tkfont
from collections import namedtuple
from math import acos, atan2, ceil, cos, pi, radians, sin, sqrt, tan
from tkinter import filedialog, messagebox, ttk

import numpy as np


from . import external, paths
from .external import cjxl_available
from .paths import ensure_export_dir, ensure_project_dir
from .version import FILE_VERSION
from .image_io import (
    DOWNSAMPLE_FACTORS, DOWNSAMPLE_FACTOR_VALUES, DOWNSAMPLE_METHODS,
    JXL_MODES, SAVE_NEW_FORMATS, bilinear_resize, convert_apng_to_jxl, downsample_image,
    encode_and_write, encode_png_bytes, jxl_quality_tag, lerp_color, make_grid, png_rgba,
    preview_placeholder, read_png, render_text_line, write_png
)
from .widgets import (
    FloatEntryHandle, JxlQualityControls, MODIFIED_OUTLINE, MODIFIED_OUTLINE_PX, ModifiedOutline,
    Tooltip, detect_modified_keys, make_scrollable_tab
)
from .kernels import (
    CONFINE_BOX, CONFINE_PATH, ConfineRule, EMIT_TOWARDS, EMIT_UP, EmissionRule, FuzzRule,
    KIND_BOUNDARY, KIND_DISTCUT, KIND_ESCAPED, KIND_OBSTACLE_TERM, KIND_SELF, KIND_START,
    N_POINTS, N_RAW, N_STATS, OBSTACLE_EFFECT_PASS, OBSTACLE_EFFECT_REFLECT,
    OBSTACLE_EFFECT_TERMINATE, ObstacleArrays, PIXEL, POINT_ALL_MEAN, POINT_BOTTOM,
    POINT_BOUNDARY_MEAN, POINT_END, POINT_LEFT, POINT_RIGHT, POINT_SELF_BOTTOM, POINT_SELF_LEFT,
    POINT_SELF_MEAN, POINT_SELF_RIGHT, POINT_SELF_TOP, POINT_START, POINT_TOP, RAW_AREA,
    RAW_BOUNCES, RAW_BOUNDARY, RAW_DISP, RAW_LAST, RAW_LAST_DIR, RAW_LAST_FULL, RAW_LENGTH,
    RAW_SELF_K, SHAPE_ARBELOS, SHAPE_CAPSULE, SHAPE_ELLIPSE, SHAPE_LENS, SHAPE_LUNE,
    SHAPE_MOSS_EGG, SHAPE_POLYGON, SHAPE_QUATREFOIL, SHAPE_REULEAUX, SHAPE_SECTOR, SHAPE_SEGMENT,
    SHAPE_TREFOIL, SIZE, ST_ANGLES, ST_BOUNCES, ST_BOUNDARY_FRAC, ST_EUCLIDEAN, ST_HDIST,
    ST_LAST_DIR, ST_LAST_RATIO, ST_LENGTH, ST_LOOP_AREA, ST_MANHATTAN, ST_RAY_FRAC,
    ST_TERMINATION, ST_VDIST, TERM_CONFINED, TERM_DIST, TERM_ESCAPED, TERM_ITER, TERM_OBSTACLE,
    TERM_PERPENDICULAR, boundary_hit, compute_points, compute_stats, emits_ray, inside_boundary,
    inside_boundary_grid, render_rows, render_rows_wh, scatter_multi_buddha, trace_path
)

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
ITER_LIMIT_MIN = 1         # allowed range (whole numbers) for the iteration limit
ITER_LIMIT_MAX = 9999
DIST_LIMIT_MIN = 0         # allowed range for the distance limit; 0 = no limit
DIST_LIMIT_MAX = 9999
DEFAULT_ITER = 256         # default iteration limit (bounces)
DEFAULT_DIST = 0           # default distance limit (units): 0 = no limit
SEGMENT_MULT_MIN = 0.01    # how much of each bounce's true segment length is
SEGMENT_MULT_MAX = 1.0     # actually travelled before continuing in the newly
DEFAULT_SEGMENT_MULT = 1.0 # -calculated (bounced) direction; 1 = no effect
WARM_ITER = 8              # iteration limit used for the compile warm-up

# Buddha blur: spreads each scattered point's weight across a neighborhood of
# pixels instead of a single one, applied only at the moment a point is
# recorded into a Buddha data channel - never touches the ray simulation
# itself. The radius control is a multiple of the view area's smallest
# dimension IN WORLD UNITS (always 2.0 in this program, since the simulation
# always runs in the fixed [-1, 1] square on both axes): control value 0.1 ->
# world radius 0.1 * 2.0 = 0.2 units, matching the same "smallest dimension"
# reference regardless of whether that 2.0 comes from a 2x2 or a 2x3 area.
BUDDHA_BLUR_GAUSSIAN = 0
BUDDHA_BLUR_TRIANGULAR = 1
BUDDHA_BLUR_BOKEH = 2
BUDDHA_BLUR_CATADIOPTRIC = 3
BUDDHA_BLUR_LABELS = ["Gaussian distribution", "Triangular distribution", "Bokeh",
                     "Catadioptric bokeh"]
BUDDHA_BLUR_RADIUS_MIN = 0.0
BUDDHA_BLUR_RADIUS_MAX = 0.5
DEFAULT_BUDDHA_BLUR_RADIUS = 0.0     # no blur - exactly the pre-blur single-pixel behavior
DEFAULT_BUDDHA_BLUR_TYPE = BUDDHA_BLUR_GAUSSIAN
BUDDHA_BLUR_GAUSSIAN_TRUNCATE = 3.0   # gaussian is truncated past this many standard
                                     # deviations - negligible weight beyond it, but
                                     # per the spec it genuinely does extend past the
                                     # nominal radius, unlike the other three kernels
BUDDHA_BLUR_CATADIOPTRIC_INNER_RATIO = 0.5   # the "donut" hole's inner edge, as a
                                             # fraction of the outer radius


def world_radius_to_pixels(radius_control, out_w, out_h):
    """The blur radius control's value -> a radius in pixels for an out_w x
    out_h output. The control is a multiple of the world view's smallest
    dimension (always 2.0 units on both axes here), and pixels-per-unit is
    min(out_w, out_h) / 2.0 for a same-scale (never-distorted) circular
    blur footprint - the 2.0s cancel, leaving radius_px = radius_control *
    min(out_w, out_h)."""
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


# The ST_TERMINATION metric's 0..1 value for each termination reason (see
# compute_stats) - also used outside njit context to filter by termination
# reason (Buddha's two termination-point options).
TERM_STAT_VALUE = {TERM_ITER: 0.0, TERM_CONFINED: 0.2, TERM_DIST: 0.4,
                   TERM_OBSTACLE: 0.6, TERM_PERPENDICULAR: 0.8, TERM_ESCAPED: 1.0}

# The canonical set of termination codes trace_path can return. Anything that
# needs to handle every termination reason - a dict keyed by term, or a
# validation check - is checked against this tuple in run_self_checks()
# below, so a future termination reason that's added to trace_path but
# forgotten somewhere else (as TERM_PERPENDICULAR once was, in two separate
# places) fails loudly at import time instead of only surfacing as a
# KeyError or a silently wrong label the first time that path is hit.
TERM_CODES = (TERM_ITER, TERM_CONFINED, TERM_DIST, TERM_OBSTACLE, TERM_PERPENDICULAR,
              TERM_ESCAPED)

# Ray-end marker color, by termination reason (preview overlay)
TERM_END_COLOR = {TERM_CONFINED: "#ff4d4d", TERM_DIST: "#ffa500", TERM_ITER: "#999999",
                  TERM_OBSTACLE: "#ffeb3b", TERM_PERPENDICULAR: "#7c4dff",
                  TERM_ESCAPED: "#00e5ff"}

# "Outcome" text template, by termination reason (Numbers tab) - .format(bounces=...)
TERM_OUTCOME_TEXT = {
    TERM_CONFINED: "Confined after {bounces} bounces",
    TERM_DIST: "Distance limit reached ({bounces} bounces)",
    TERM_OBSTACLE: "Terminated by an obstacle ({bounces} bounces)",
    TERM_PERPENDICULAR: "Perpendicular collision ({bounces} bounces)",
    TERM_ESCAPED: "Escaped ({bounces} bounces)",
    TERM_ITER: "Iteration limit reached ({bounces} bounces)",
}


# How a shape's geometry is packed into its bound array (documented above ellipse_bound);
# every Shape subclass declares its LAYOUT, and the Python code that has to treat a packing
# differently (offset_bound, ring_layer_masks) branches on it rather than on the kind.
LAYOUT_ELLIPSE = 0          # a quadratic form
LAYOUT_POLYGON = 1          # half-planes
LAYOUT_CIRCLES = 2          # circles (the shape is their intersection)
LAYOUT_ARCS_EDGES = 3       # circular arcs and straight edges
LAYOUT_ARC_LOOPS = 4        # circular arcs forming a closed loop that may be concave; inside by even-odd
LAYOUTS = (LAYOUT_ELLIPSE, LAYOUT_POLYGON, LAYOUT_CIRCLES, LAYOUT_ARCS_EDGES, LAYOUT_ARC_LOOPS)

# Ellipse controls: full width / height (0..2), rotation in degrees
ELLIPSE_MIN = 0.0
ELLIPSE_MAX = 2.0
DEFAULT_WIDTH = 2.0        # 2.0 x 2.0, no rotation = the unit circle
DEFAULT_HEIGHT = 2.0
DEFAULT_RADIUS = 1.0         # a new polygon / Reuleaux / sector / segment's radius
DEFAULT_ROTATION = 0.0       # a new obstacle's rotation
OBST_POS_MIN = -1.0         # obstacle centre (cx, cy) slider range
OBST_POS_MAX = 1.0
OBST_RADIUS_MIN = 0.0       # the radius slider range (polygon, Reuleaux, sector, segment)
OBST_RADIUS_MAX = 2.0
AXIS_MIN = 1e-3            # smallest semi-axis used in the maths (avoids 1/0)

# Regular-polygon controls (circumradius is fixed at 1: vertices on the unit circle)
POLY_MIN_SIDES = 3
POLY_MAX_SIDES = 64
DEFAULT_SIDES = 5

# Obstacles: a fixed-width row per obstacle (padded with zeros) so their bounds
# can live in one ordinary numpy array for the njit kernels.
OBST_BOUND_LEN = 3 + 2 * POLY_MAX_SIDES

# Moss's egg: apex angle ACB (the isosceles triangle's apex angle), size
MOSS_ANGLE_MIN = 60.0      # below this the construction degenerates further
MOSS_ANGLE_MAX = 180.0
DEFAULT_MOSS_ANGLE = 90.0  # the classic right-isosceles-triangle egg
MOSS_SIZE_MIN = 0.0
MOSS_SIZE_MAX = 2.0
DEFAULT_MOSS_SIZE = 0.9    # at 1.0 the default egg's top edge passed y = 1.03, outside
                           # the view; 0.9 keeps every apex angle inside with a margin

# Circular sector / segment: angular span (degrees); the radius is the shared one
ARC_SPAN_MIN = 1.0         # degrees; kept off 0 and 360 to avoid degenerate wedges
ARC_SPAN_MAX = 359.0       # circular segment's range
SECTOR_SPAN_MAX = 180.0    # circular sector's range: capped at a straight angle,
                          # since a sector's two straight edges are only a valid
                          # half-plane pair (its "inside" test) up to 180 degrees
DEFAULT_SECTOR_SPAN = 90.0
DEFAULT_SEGMENT_SPAN = 120.0

# Two-circle lens: two radii and a center separation
LENS_RADIUS_MIN = 0.0
LENS_RADIUS_MAX = 2.0
DEFAULT_LENS_RADIUS = 1.0
LENS_SEP_MIN = 0.0
LENS_SEP_MAX = 2.0
DEFAULT_LENS_SEP = 1.0

# Reuleaux polygon: odd number of sides, like a regular polygon
REULEAUX_MIN_SIDES = 3
REULEAUX_MAX_SIDES = 21    # kept odd in the UI (3, 5, 7, ...)
DEFAULT_REULEAUX_SIDES = 3

# Capsule (stadium): straight-part length, cap radius
CAPSULE_LENGTH_MIN = 0.0
CAPSULE_LENGTH_MAX = 2.0
DEFAULT_CAPSULE_LENGTH = 1.0
CAPSULE_RADIUS_MIN = 0.0
CAPSULE_RADIUS_MAX = 1.0
DEFAULT_CAPSULE_RADIUS = 0.5

# Trefoil / quatrefoil: lobe size as a fraction of the radius (the distance from the
# centre to the farthest point); below 0.5 the lobes would leave a hole in the middle
LOBE_SIZE_MIN = 0.52
LOBE_SIZE_MAX = 1.0
DEFAULT_LOBE_SIZE = 0.6

# Lune: the outer circle's radius is the shared radius; a second circle cuts a bite out of it
CUT_RADIUS_MIN = 0.0
CUT_RADIUS_MAX = 2.0
DEFAULT_CUT_RADIUS = 0.8
CUT_OFFSET_MIN = 0.0
CUT_OFFSET_MAX = 2.0
DEFAULT_CUT_OFFSET = 0.6

# Arbelos: where the diameter is split, as a fraction of its length from one end
SPLIT_MIN = 0.05
SPLIT_MAX = 0.95
DEFAULT_SPLIT = 0.35

# Fuzzing controls
FUZZ_JITTER_MIN = 0.0      # jitter range (+/-), degrees
FUZZ_JITTER_MAX = 180.0
FUZZ_MULT_MIN = 0.0        # bounce angle multiplier range
FUZZ_MULT_MAX = 3.0
DEFAULT_FUZZ_SEED = 0
def strict_record(cls):
    """Class decorator for the program's records of settings. The result is a frozen
    dataclass whose every field is required and must be given by keyword (no
    positional arguments, no defaults), with __slots__, no indexing, iteration or
    unpacking (so a field can only ever be reached by name), and every value checked
    against its annotation: int, float (an int is fine, a bool is not), bool, str,
    tuple (a list is accepted and stored as a tuple) or a class. A violation raises
    TypeError naming the record and field. `replace(**changes)` makes a modified
    copy and `as_dict()` gives the fields as a dict. (The check is attached before @dataclass runs, or dataclass would never
    call it.)"""
    hints = dict(cls.__annotations__)
    owner = cls.__name__

    def __post_init__(self):
        for name, want in hints.items():
            value = getattr(self, name)
            if want is float:
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif want is int:
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif want is tuple:
                if isinstance(value, list):
                    value = tuple(value)
                    object.__setattr__(self, name, value)
                ok = isinstance(value, tuple)
            else:
                ok = isinstance(value, want)
            if not ok:
                raise TypeError(f"{owner}.{name} must be {want.__name__}, "
                                f"got {type(value).__name__}: {value!r}")

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    def as_dict(self):
        """{field name: value}, one level deep (nested records stay records)."""
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)

    cls.__post_init__ = __post_init__
    cls.replace = replace
    cls.as_dict = as_dict
    record = dataclasses.dataclass(frozen=True, kw_only=True, slots=True)(cls)
    # Pickle by field name (dataclass's own state for a slotted frozen class is a bare
    # list of values, which would make a dump or saved file depend on field order).
    # Assigned after the decorator, which would otherwise install its own versions.
    record.__getstate__ = __getstate__
    record.__setstate__ = __setstate__
    return record


@strict_record
class FuzzParams:
    """Fuzzing: random perturbation of the ray's angles."""
    enabled: bool
    emit_jitter: float                               # ray emission (added to its direction)
    bnd_offset: float                                # bounces off the boundary
    bnd_jitter: float
    bnd_mult: float
    self_offset: float                               # bounces off the ray itself
    self_jitter: float
    self_mult: float
    seed: int


DEFAULT_FUZZ = FuzzParams(enabled=False, emit_jitter=0.0, bnd_offset=0.0, bnd_jitter=0.0,
                          bnd_mult=1.0, self_offset=0.0, self_jitter=0.0, self_mult=1.0,
                          seed=DEFAULT_FUZZ_SEED)

EMIT_MODE_NAMES = ["Up", "Towards point"]
DEFAULT_EMIT_MODE = EMIT_UP
# The target point is given like an obstacle's position: a distance r at angle theta
# (degrees) from the origin, plus an x and a y offset (see obstacle_center_from).
EMIT_TARGET_FIELDS = ("r", "theta", "cx", "cy")
EMIT_TARGET_LABELS = ["Distance (r):", "Angle (\u03b8, \u00b0):", "Offset X:", "Offset Y:"]
DEFAULT_EMIT_TARGET_R = 0.0
DEFAULT_EMIT_TARGET_THETA = 0.0
DEFAULT_EMIT_TARGET_CX = 0.0
DEFAULT_EMIT_TARGET_CY = 0.0
DEFAULT_EMIT_OFFSET = -45.0

CONFINE_METHOD_NAMES = ["Box", "Path length"]
CONFINE_WINDOW_MIN = 2
CONFINE_WINDOW_MAX = 999
DEFAULT_CONFINE_WINDOW = 4
DEFAULT_CONFINE_METHOD = CONFINE_BOX
DEFAULT_CONFINE_THRESHOLD = PIXEL
REFRESH_HZ = 10            # redraws per second during generation
COMPUTE_FRACTION = 0.55    # share of each refresh period spent computing

BACKGROUND = (12, 12, 16)
PENDING_OUTLINE = "#7fdbff"   # dashed outline of shape settings not yet generated

# Semi-transparent grey guide lines through the end point of the previewed ray
GUIDE_RGBA = (200, 200, 200, 110)
GUIDE_FALLBACK_HEX = "#c8c8c8"     # used with a stipple if alpha images fail

# The segment-lengths graphic: same px-per-unit scale as the main image, one
# unit tall (so it matches the main image's scale, just turned into a plot of
# length rather than of position). Each bar is colored by what kind of point
# lies at its top (earlier) and bottom (later) end.
SEG_CANVAS_W = SIZE
SEG_CANVAS_H = SIZE // 2   # 1 world unit, at the main image's pixels-per-unit
KIND_COLOR = {KIND_START: (67, 160, 71), KIND_DISTCUT: (67, 160, 71),   # green
             KIND_BOUNDARY: (229, 57, 53),                              # red
             KIND_SELF: (30, 136, 229),                                 # blue
             KIND_OBSTACLE_TERM: (255, 235, 59),                        # yellow
             KIND_ESCAPED: (0, 229, 255)}                               # cyan

# Solid outline colors for obstacles (part of the saved image/video, drawn with
# the same "outline" decoration flag), by ray effect
OBST_TERMINATE_RGB = (121, 85, 72)     # brown
OBST_REFLECT_RGB = (179, 229, 252)     # pale light blue
OBST_PASS_RGB = (0, 0, 0)              # black
OBST_TERMINATE_HEX = "#795548"
OBST_REFLECT_HEX = "#b3e5fc"
OBST_PASS_HEX = "#000000"
# An obstacle's outline is two layers, each this many pixels thick: the outer
# one (just outside its edge) shows its incoming effect, the inner one (just
# inside) its outgoing effect.
OBST_RING_LAYER_PX = 1.5

OBSTACLE_EFFECT_NAMES = ["Reflect", "Terminate", "Pass through"]
OBST_EFFECT_RGB = {OBSTACLE_EFFECT_REFLECT: OBST_REFLECT_RGB,
                   OBSTACLE_EFFECT_TERMINATE: OBST_TERMINATE_RGB,
                   OBSTACLE_EFFECT_PASS: OBST_PASS_RGB}

# Obstacle and view thumbnails
THUMB_SIZE = 48            # small resolution, per pixel
THUMB_BACKGROUND = (110, 110, 110)   # obstacle thumbnails: background and shape fill
THUMB_FILL = (150, 150, 150)
PREVIEW_SIZE = 150         # side length of the video-panel frame preview
SELECTED_OUTLINE = "#ffeb3b"       # yellow
SELECTED_OUTLINE_PX = 4


# The live preview's "view N" name overlay (bottom-left corner, drawn with
# FONT_5X7 - see draw_view_name_overlay). Preview-only: never part of an
# exported image.
VIEW_NAME_OVERLAY_SCALE = 2
VIEW_NAME_OVERLAY_PAD = 3          # blank border between the text and its background box
VIEW_NAME_OVERLAY_MARGIN = 6       # gap between the background box and the image edge
VIEW_NAME_OVERLAY_BG = (0, 0, 0)
VIEW_NAME_OVERLAY_FG = (255, 255, 255)

# Window layout
PAD = 8
RIGHT_W = 580              # width of the Controls / Status column
TIMELINE_W = 780           # width of the Timeline / Video column (holds two lists)
BOTTOM_H = 400             # height of the bottom row of panels

ERROR_RED = "#c62828"

# Shown in the Obstacles tab, under the ray-effect controls
RAY_HELP_TEXT = ("Incoming: what happens to a ray reaching a shape's edge from "
                 "outside; outgoing: from inside. Pass through both ways and rays "
                 "ignore the shape. A ray is fired from every pixel that is inside "
                 "any shape with 'Fire rays inside' ticked; a ray with nothing left "
                 "to hit escapes.")


MAX_CRASH_DUMPS = 3        # per launch: an error repeating in a timer must not fill the disk

@strict_record
class BuddhaBlurSettings:
    """The blur applied to Buddha coloration's scattered points."""
    radius: float                                    # a multiple of the view's smaller dimension
    type: int                                        # index into BUDDHA_BLUR_LABELS


DEFAULT_BUDDHA_BLUR = BuddhaBlurSettings(radius=DEFAULT_BUDDHA_BLUR_RADIUS,
                                         type=DEFAULT_BUDDHA_BLUR_TYPE)


@strict_record
class RaySettings:
    """Every value of every simulation control - what determines the generated image,
    together with the obstacles (the scene, the obstacles and the views, is tracked
    separately). Start compares these, "Revert" restores them, and a keyframe resolves
    to one for each video frame."""
    max_iter: int
    max_dist: float                                  # 0 = no limit; see effective_dist
    seg_mult: float
    fuzz: FuzzParams
    buddha_blur: BuddhaBlurSettings
    confine_window: int
    confine_method: int
    confine_threshold: float
    emit_mode: int
    emit_target_r: float
    emit_target_theta: float
    emit_target_cx: float
    emit_target_cy: float
    emit_offset: float


DEFAULT_RAY_SETTINGS = RaySettings(
    max_iter=DEFAULT_ITER, max_dist=DEFAULT_DIST, seg_mult=DEFAULT_SEGMENT_MULT,
    fuzz=DEFAULT_FUZZ, buddha_blur=DEFAULT_BUDDHA_BLUR, confine_window=DEFAULT_CONFINE_WINDOW,
    confine_method=DEFAULT_CONFINE_METHOD, confine_threshold=DEFAULT_CONFINE_THRESHOLD,
    emit_mode=DEFAULT_EMIT_MODE, emit_target_r=DEFAULT_EMIT_TARGET_R,
    emit_target_theta=DEFAULT_EMIT_TARGET_THETA, emit_target_cx=DEFAULT_EMIT_TARGET_CX,
    emit_target_cy=DEFAULT_EMIT_TARGET_CY, emit_offset=DEFAULT_EMIT_OFFSET)


# A resolved frame's complete simulation settings (what render_full_image
# needs, aside from the views and obstacles - each resolved separately, the
# same way: a fixed set established when the render starts, each one's own
# fields independently held/animated from the keyframes). Built by
# resolve_frame_settings() from whichever keyframes define each setting.
@strict_record
class KeyframeData:
    settings: RaySettings
    outline: bool


MODIFIED_POLL_MS = 100
PROJECT_POLL_MS = 1000     # how often the window title's unsaved-changes mark is refreshed


class Keyframe:
    """One entry in the timeline. `data` is a dict of setting-key -> value -
    only for the settings this keyframe actually defines (see SETTING_INFO);
    a keyframe may define any subset, which is how different settings can
    start/stop animating at different points on the timeline. Between two
    keyframes a setting that is a float slider (SETTING_FLOAT_KEYS) is
    interpolated linearly; every other setting holds the earlier value."""
    __slots__ = ("id", "frame_index", "data")

    def __init__(self, id_, frame_index, data):
        self.id = id_
        self.frame_index = frame_index
        self.data = data


def remove_keyframe_keys(keyframes, prefix):
    """Deletes every setting whose key starts with `prefix` from every
    keyframe's data, returning how many were removed. Used when an obstacle
    or view is deleted, with a prefix like "obst:3:" - the trailing colon
    matters, so obstacle 3's cleanup can't touch obstacle 30's keys."""
    removed = 0
    for kf in keyframes:
        stale = [key for key in kf.data if key.startswith(prefix)]
        for key in stale:
            del kf.data[key]
        removed += len(stale)
    return removed


def snapshot_keyframes(keyframes):
    """Independent copies of `keyframes` (own data dicts), so a running render is unaffected by anything done
    to the real keyframes meanwhile - including the cleanup of a deleted
    obstacle's or view's settings."""
    return [Keyframe(kf.id, kf.frame_index, dict(kf.data)) for kf in keyframes]


# --------------------------------------------------------------------------- #
# Shapes: the obstacles. Every obstacle is an instance of one subclass of Shape and
# stays that kind for its whole life (the kind is chosen when it is created); a
# subclass has only the parameters its own geometry needs.
# --------------------------------------------------------------------------- #
# One row per parameter of a shape (and of the position every shape shares):
#   field        the attribute name (and the keyframe key's last part)
#   label        its name in the obstacle editor and the keyframe settings list
#   default      its value on a new obstacle
#   animate      True for a float (interpolated between keyframes), False for a
#                whole number (held)
#   fmt          how short_value_text shows it: "size" (N.NN), "angle" (N.N and a
#                degree sign), "offset" (+N.NN) or "int"
#   lo, hi, step, slider_fmt   the editor slider; lo None means a free angle entry
#                (any number of degrees) instead of a slider
ShapeParam = namedtuple("ShapeParam", ["field", "label", "default", "animate", "fmt",
                                       "lo", "hi", "step", "slider_fmt"])


def _slider(field, label, default, lo, hi, step=0.01, fmt="size", slider_fmt="{:.2f}"):
    return ShapeParam(field, label, default, True, fmt, lo, hi, step, slider_fmt)


def _count(field, label, default, lo, hi, step=1):
    return ShapeParam(field, label, default, False, "int", lo, hi, step, "{:.0f}")


def _free_angle(field, label, default=0.0):
    return ShapeParam(field, label, default, True, "angle", None, None, None, None)


# What every shape has besides its own parameters.
ROTATION_PARAM = _free_angle("rotation", "Rotation", DEFAULT_ROTATION)
POSITION_PARAMS = (
    _slider("r", "Distance (r)", 0.0, 0.0, OBST_POS_MAX),
    _free_angle("theta", "Angle (θ)"),
    _slider("cx", "Offset X", 0.0, OBST_POS_MIN, OBST_POS_MAX, fmt="offset",
            slider_fmt="{:+.2f}"),
    _slider("cy", "Offset Y", 0.0, OBST_POS_MIN, OBST_POS_MAX, fmt="offset",
            slider_fmt="{:+.2f}"),
)
# The other fields of every shape, which are not numbers: (field, label, type).
EFFECT_FIELDS = (("in_effect", "Incoming ray effect", int),
                 ("out_effect", "Outgoing ray effect", int),
                 ("fire_inside", "Fire rays inside", bool))
RADIUS_PARAM = _slider("radius", "Radius", DEFAULT_RADIUS, OBST_RADIUS_MIN, OBST_RADIUS_MAX)


class Shape:
    """One obstacle: a shape that rays can reflect off, be terminated by, or pass
    through - separately for rays crossing its edge from outside (in_effect) and
    from inside (out_effect). This is the base class; the kinds are its
    subclasses below, each declaring only its own parameters in PARAMS (and the
    same names in __slots__; a self-check keeps the two in step). Its position is
    given in polar coordinates (r, theta) from the origin, with (cx, cy) as an
    additional plain x/y offset from that polar position.

    A field only ever takes a number of the type its row declares (a float field
    any real number, an int field a whole number), or it raises TypeError.
    snapshot() / with_values() give frozen copies - the same class, but with
    every assignment refused - for the background threads (video render, new-image
    export) and for the settings the image on screen was generated with, so a
    thread can read an obstacle while the GUI is editing the live one. A frozen
    copy has every method of the live shape."""
    KIND = None             # the SHAPE_* code the kernels know this kind by
    KEY = None              # the permanent name a project file calls this kind by
    LAYOUT = None           # the LAYOUT_* its bound array is packed as
    NAME = None             # shown to the user
    PARAMS = ()             # this kind's own ShapeParam rows
    __slots__ = ("id", "rotation", "r", "theta", "cx", "cy", "in_effect", "out_effect",
                 "fire_inside", "_frozen")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NUMERIC_PARAMS = (ROTATION_PARAM,) + tuple(cls.PARAMS) + POSITION_PARAMS
        # (field, label, is_float) for every field a keyframe can hold, in display order
        cls.KEYFRAME_FIELDS = (
            tuple((p.field, p.label, p.animate) for p in cls.NUMERIC_PARAMS)
            + tuple((field, label, False) for field, label, _type in EFFECT_FIELDS))
        cls.FIELD_NAMES = tuple(field for field, _label, _is_float in cls.KEYFRAME_FIELDS)
        types = {p.field: (float if p.animate else int) for p in cls.NUMERIC_PARAMS}
        types.update({field: type_ for field, _label, type_ in EFFECT_FIELDS})
        types["id"] = int
        cls.FIELD_TYPES = types

    def __init__(self, id_):
        object.__setattr__(self, "_frozen", False)
        self.id = id_
        for p in self.NUMERIC_PARAMS:
            setattr(self, p.field, p.default)
        self.in_effect = OBSTACLE_EFFECT_REFLECT
        self.out_effect = OBSTACLE_EFFECT_REFLECT
        self.fire_inside = True

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(f"{type(self).__name__} is a frozen copy: can't set {name}")
        want = self.FIELD_TYPES.get(name)
        if want is None:
            raise AttributeError(f"{type(self).__name__} has no field {name!r}")
        if want is bool:
            ok = isinstance(value, (bool, np.bool_))
        elif want is int:
            ok = isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, (int, float, np.integer, np.floating)) \
                and not isinstance(value, bool)
        if not ok:
            raise TypeError(f"{type(self).__name__}.{name} must be {want.__name__}, "
                            f"got {type(value).__name__}: {value!r}")
        object.__setattr__(self, name, value)

    def _copy(self, id_, values, frozen):
        unknown = set(values) - set(self.FIELD_NAMES)
        if unknown:
            raise AttributeError(f"{type(self).__name__} has no field(s) {sorted(unknown)}")
        copy = type(self)(id_)
        for name in self.FIELD_NAMES:
            setattr(copy, name, values[name] if name in values else getattr(self, name))
        object.__setattr__(copy, "_frozen", frozen)
        return copy

    def __getstate__(self):
        # (None, {slot: value}) is the state pickle already gives a slotted object, which
        # keeps the dump viewer's reading of a shape unchanged; copy and pickle restore it
        # through __setstate__, so every value is checked again and a frozen copy stays frozen
        return (None, {name: getattr(self, name)
                       for name in ("id",) + self.FIELD_NAMES + ("_frozen",)})

    def __setstate__(self, state):
        values = state[1]
        object.__setattr__(self, "_frozen", False)
        for name, value in values.items():
            if name != "_frozen":
                setattr(self, name, value)
        object.__setattr__(self, "_frozen", values["_frozen"])

    def snapshot(self):
        """A frozen copy (see the class docstring)."""
        return self._copy(self.id, {}, True)

    def with_values(self, **values):
        """A frozen copy with some fields replaced (the values a video frame
        resolves from the keyframes)."""
        return self._copy(self.id, values, True)

    def duplicate(self, new_id):
        """A live, editable copy under a new id, of the same kind."""
        return self._copy(new_id, {}, False)

    def state(self):
        """Every field but the id, with the kind - for telling whether an obstacle
        has changed since the last Start."""
        return (self.KIND,) + tuple(getattr(self, f) for f in self.FIELD_NAMES)

    def center(self):
        """World (x, y) of the obstacle's centre: the polar position (r,
        theta) from the origin, plus the plain (cx, cy) offset."""
        th = radians(self.theta)
        return self.r * cos(th) + self.cx, self.r * sin(th) + self.cy

    def describe(self):
        """The one-line description shown in the obstacle list."""
        pos_txt = f"r{self.r:.2f} @ {self.theta:.1f}°"
        if self.cx or self.cy:
            pos_txt += f" +({self.cx:+.2f}, {self.cy:+.2f})"
        return (f"{self.shape_text()}, {pos_txt}, in: {OBSTACLE_EFFECT_NAMES[self.in_effect]}, "
                f"out: {OBSTACLE_EFFECT_NAMES[self.out_effect]}")

    def info_lines(self, index):
        """Every setting that defines the obstacle (`index` is its number in the
        list), as text lines no longer than FIGURE_INFO_COLUMN_CHARS, using only
        glyphs the pixel font has."""
        return [f"Obstacle #{index}: {self.NAME}", self.info_param_line(),
                f"  rot {self.rotation:.1f}°",
                f"  pos r{self.r:.2f} θ{self.theta:.1f}°",
                f"  offset {self.cx:+.2f} {self.cy:+.2f}",
                f"  in:{OBSTACLE_EFFECT_NAMES[int(self.in_effect)]}",
                f"  out:{OBSTACLE_EFFECT_NAMES[int(self.out_effect)]}",
                f"  fire:{'On' if self.fire_inside else 'Off'}"]

    # What each kind provides:
    def params(self):
        """The tuple its bound function takes (see bound_from_params)."""
        raise NotImplementedError

    @classmethod
    def bound_from_params(cls, params):
        """The packed shape array (see make_bound) for a params() tuple."""
        raise NotImplementedError

    def shape_text(self):
        """The shape's part of its one-line description in the obstacle list."""
        raise NotImplementedError

    def extent(self):
        """A reasonable half-width (world units) to zoom its thumbnail to fit."""
        raise NotImplementedError

    def info_param_line(self):
        """The one line of the figure-settings margin that gives its own parameters."""
        raise NotImplementedError

    def bound(self):
        return self.bound_from_params(self.params())

    def boundary_xy(self):
        """(x, y) arrays tracing its edge around its own centre."""
        return sample_boundary_xy(self.KIND, self.bound())

    def outline_coords(self):
        """Canvas coordinates (flat list) of a polygon tracing the shape, at its
        centre (polar position plus x/y offset)."""
        wx, wy = self.boundary_xy()
        ocx, ocy = self.center()
        xs = (wx + ocx + 1.0) * (SIZE / 2.0)
        ys = (1.0 - (wy + ocy)) * (SIZE / 2.0)
        return np.column_stack((xs, ys)).ravel().tolist()


class EllipseShape(Shape):
    KIND = SHAPE_ELLIPSE
    LAYOUT = LAYOUT_ELLIPSE
    KEY = "ellipse"
    NAME = "Ellipse"
    PARAMS = (_slider("width", "Width", 1.0, ELLIPSE_MIN, ELLIPSE_MAX),
              _slider("height", "Height", 1.0, ELLIPSE_MIN, ELLIPSE_MAX))
    __slots__ = ("width", "height")

    def params(self):
        return (self.width, self.height, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return ellipse_bound(*params)

    def shape_text(self):
        return f"Ellipse {self.width:.2f}×{self.height:.2f}, rot {self.rotation:.1f}°"

    def extent(self):
        return max(self.width, self.height) / 2.0

    def info_param_line(self):
        return f"  w{self.width:.2f} h{self.height:.2f}"

    def boundary_xy(self):
        t = np.linspace(0.0, 2.0 * pi, 181)
        ex, ey = 0.5 * self.width * np.cos(t), 0.5 * self.height * np.sin(t)
        th = radians(self.rotation)
        c, s = cos(th), sin(th)
        return c * ex - s * ey, s * ex + c * ey


class RegularPolygonShape(Shape):
    KIND = SHAPE_POLYGON
    LAYOUT = LAYOUT_POLYGON
    KEY = "regular_polygon"
    NAME = "Regular polygon"
    PARAMS = (_count("sides", "Sides", DEFAULT_SIDES, POLY_MIN_SIDES, POLY_MAX_SIDES),
              RADIUS_PARAM)
    __slots__ = ("sides", "radius")

    def params(self):
        return (self.sides, self.rotation, self.radius)

    @classmethod
    def bound_from_params(cls, params):
        return polygon_bound(*params)

    def shape_text(self):
        return f"{self.sides}-gon r{self.radius:.2f}, rot {self.rotation:.1f}°"

    def extent(self):
        return self.radius

    def info_param_line(self):
        return f"  sides {int(round(self.sides))} r{self.radius:.2f}"

    def boundary_xy(self):
        ang = (pi / 2.0 + radians(self.rotation)
               + 2.0 * pi * np.arange(self.sides + 1) / self.sides)
        return self.radius * np.cos(ang), self.radius * np.sin(ang)


class MossEggShape(Shape):
    KIND = SHAPE_MOSS_EGG
    LAYOUT = LAYOUT_ARCS_EDGES
    KEY = "moss_egg"
    NAME = "Moss's egg"
    PARAMS = (_slider("moss_angle", "Angle ACB", DEFAULT_MOSS_ANGLE, MOSS_ANGLE_MIN,
                      MOSS_ANGLE_MAX, 0.5, "angle", "{:.1f}"),
              _slider("moss_size", "Size", DEFAULT_MOSS_SIZE, MOSS_SIZE_MIN, MOSS_SIZE_MAX))
    __slots__ = ("moss_angle", "moss_size")

    def params(self):
        return (self.moss_angle, self.moss_size, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return moss_egg_bound(*params)

    def shape_text(self):
        return (f"Moss's egg {self.moss_angle:.0f}°, size {self.moss_size:.2f}, "
                f"rot {self.rotation:.1f}°")

    def extent(self):
        return 2.2 * self.moss_size

    def info_param_line(self):
        return f"  angle {self.moss_angle:.1f}° size {self.moss_size:.2f}"


class CircularSectorShape(Shape):
    KIND = SHAPE_SECTOR
    LAYOUT = LAYOUT_ARCS_EDGES
    KEY = "circular_sector"
    NAME = "Circular sector"
    PARAMS = (RADIUS_PARAM,
              _slider("sector_span", "Span", DEFAULT_SECTOR_SPAN, ARC_SPAN_MIN,
                      SECTOR_SPAN_MAX, 0.5, "angle", "{:.1f}"))
    __slots__ = ("radius", "sector_span")

    def params(self):
        return (self.radius, self.sector_span, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return sector_bound(*params)

    def shape_text(self):
        return f"Sector r{self.radius:.2f} {self.sector_span:.0f}°"

    def extent(self):
        return self.radius

    def info_param_line(self):
        return f"  r{self.radius:.2f} span {self.sector_span:.1f}°"


class CircularSegmentShape(Shape):
    KIND = SHAPE_SEGMENT
    LAYOUT = LAYOUT_ARCS_EDGES
    KEY = "circular_segment"
    NAME = "Circular segment"
    PARAMS = (RADIUS_PARAM,
              _slider("segment_span", "Span", DEFAULT_SEGMENT_SPAN, ARC_SPAN_MIN,
                      ARC_SPAN_MAX, 0.5, "angle", "{:.1f}"))
    __slots__ = ("radius", "segment_span")

    def params(self):
        return (self.radius, self.segment_span, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return segment_bound(*params)

    def shape_text(self):
        return f"Segment r{self.radius:.2f} {self.segment_span:.0f}°"

    def extent(self):
        return self.radius

    def info_param_line(self):
        return f"  r{self.radius:.2f} span {self.segment_span:.1f}°"


class LensShape(Shape):
    KIND = SHAPE_LENS
    LAYOUT = LAYOUT_CIRCLES
    KEY = "two_circle_lens"
    NAME = "Two-circle lens"
    PARAMS = (_slider("lens_radius1", "Radius 1", DEFAULT_LENS_RADIUS, LENS_RADIUS_MIN,
                      LENS_RADIUS_MAX),
              _slider("lens_radius2", "Radius 2", DEFAULT_LENS_RADIUS, LENS_RADIUS_MIN,
                      LENS_RADIUS_MAX),
              _slider("lens_sep", "Separation", DEFAULT_LENS_SEP, LENS_SEP_MIN, LENS_SEP_MAX))
    __slots__ = ("lens_radius1", "lens_radius2", "lens_sep")

    def params(self):
        return (self.lens_radius1, self.lens_radius2, self.lens_sep, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return lens_bound(*params)

    def shape_text(self):
        return (f"Lens r{self.lens_radius1:.2f}/{self.lens_radius2:.2f} "
                f"d{self.lens_sep:.2f}")

    def extent(self):
        return max(self.lens_radius1, self.lens_radius2) + self.lens_sep / 2.0

    def info_param_line(self):
        return (f"  r1 {self.lens_radius1:.2f} r2 {self.lens_radius2:.2f} "
                f"sep {self.lens_sep:.2f}")


class ReuleauxPolygonShape(Shape):
    KIND = SHAPE_REULEAUX
    LAYOUT = LAYOUT_CIRCLES
    KEY = "reuleaux_polygon"
    NAME = "Reuleaux polygon"
    PARAMS = (_count("reuleaux_sides", "Sides (odd)", DEFAULT_REULEAUX_SIDES,
                     REULEAUX_MIN_SIDES, REULEAUX_MAX_SIDES, step=2),
              RADIUS_PARAM)
    __slots__ = ("reuleaux_sides", "radius")

    def params(self):
        return (self.reuleaux_sides, self.rotation, self.radius)

    @classmethod
    def bound_from_params(cls, params):
        return reuleaux_bound(*params)

    def shape_text(self):
        return f"Reuleaux-{self.reuleaux_sides} r{self.radius:.2f}"

    def extent(self):
        return self.radius * 2.0

    def info_param_line(self):
        return f"  sides {int(round(self.reuleaux_sides))} r{self.radius:.2f}"


class CapsuleShape(Shape):
    KIND = SHAPE_CAPSULE
    LAYOUT = LAYOUT_ARCS_EDGES
    KEY = "capsule"
    NAME = "Capsule"
    PARAMS = (_slider("capsule_length", "Length", DEFAULT_CAPSULE_LENGTH,
                      CAPSULE_LENGTH_MIN, CAPSULE_LENGTH_MAX),
              _slider("capsule_radius", "Radius", DEFAULT_CAPSULE_RADIUS,
                      CAPSULE_RADIUS_MIN, CAPSULE_RADIUS_MAX))
    __slots__ = ("capsule_length", "capsule_radius")

    def params(self):
        return (self.capsule_length, self.capsule_radius, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return capsule_bound(*params)

    def shape_text(self):
        return f"Capsule L{self.capsule_length:.2f} r{self.capsule_radius:.2f}"

    def extent(self):
        return self.capsule_length / 2.0 + self.capsule_radius

    def info_param_line(self):
        return f"  len {self.capsule_length:.2f} r{self.capsule_radius:.2f}"


class ArcLoopShape(Shape):
    """Base of the kinds whose edge is a closed loop of circular arcs that may be concave
    (a kind that only subclasses this is not a kind of its own: see SHAPE_CLASSES)."""
    LAYOUT = LAYOUT_ARC_LOOPS

    def boundary_xy(self):
        return chain_arcs_xy(self.bound())


class LobedShape(ArcLoopShape):
    """Base of the trefoil and quatrefoil: a union of LOBES circles."""
    LOBES = None
    PARAMS = (RADIUS_PARAM,
              _slider("lobe_size", "Lobe size", DEFAULT_LOBE_SIZE, LOBE_SIZE_MIN, LOBE_SIZE_MAX))
    __slots__ = ("radius", "lobe_size")

    def params(self):
        return (self.radius, self.lobe_size, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return lobed_bound(cls.KIND, cls.LOBES, *params)

    def shape_text(self):
        return (f"{self.NAME} r{self.radius:.2f} lobe {self.lobe_size:.2f}, "
                f"rot {self.rotation:.1f}\u00b0")

    def extent(self):
        return self.radius

    def info_param_line(self):
        return f"  r{self.radius:.2f} lobe {self.lobe_size:.2f}"


class TrefoilShape(LobedShape):
    KIND = SHAPE_TREFOIL
    KEY = "trefoil"
    NAME = "Trefoil"
    LOBES = 3
    __slots__ = ()


class QuatrefoilShape(LobedShape):
    KIND = SHAPE_QUATREFOIL
    KEY = "quatrefoil"
    NAME = "Quatrefoil"
    LOBES = 4
    __slots__ = ()


class LuneShape(ArcLoopShape):
    KIND = SHAPE_LUNE
    KEY = "lune"
    NAME = "Lune"
    PARAMS = (RADIUS_PARAM,
              _slider("cut_radius", "Cut radius", DEFAULT_CUT_RADIUS, CUT_RADIUS_MIN, CUT_RADIUS_MAX),
              _slider("cut_offset", "Cut offset", DEFAULT_CUT_OFFSET, CUT_OFFSET_MIN, CUT_OFFSET_MAX))
    __slots__ = ("radius", "cut_radius", "cut_offset")

    def params(self):
        return (self.radius, self.cut_radius, self.cut_offset, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return lune_bound(*params)

    def shape_text(self):
        return (f"Lune r{self.radius:.2f} cut {self.cut_radius:.2f} at {self.cut_offset:.2f}, "
                f"rot {self.rotation:.1f}\u00b0")

    def extent(self):
        return self.radius

    def info_param_line(self):
        return f"  r{self.radius:.2f} cut {self.cut_radius:.2f} off {self.cut_offset:.2f}"


class ArbelosShape(ArcLoopShape):
    KIND = SHAPE_ARBELOS
    KEY = "arbelos"
    NAME = "Arbelos"
    PARAMS = (RADIUS_PARAM, _slider("split", "Split", DEFAULT_SPLIT, SPLIT_MIN, SPLIT_MAX))
    __slots__ = ("radius", "split")

    def params(self):
        return (self.radius, self.split, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return arbelos_bound(*params)

    def shape_text(self):
        return f"Arbelos r{self.radius:.2f} split {self.split:.2f}, rot {self.rotation:.1f}\u00b0"

    def extent(self):
        return self.radius

    def info_param_line(self):
        return f"  r{self.radius:.2f} split {self.split:.2f}"


# Every kind, indexed by its SHAPE_* code (the order of the "new..." menu).
SHAPE_CLASSES = [EllipseShape, RegularPolygonShape, MossEggShape, CircularSectorShape,
                 CircularSegmentShape, LensShape, ReuleauxPolygonShape, CapsuleShape,
                 TrefoilShape, QuatrefoilShape, LuneShape, ArbelosShape]
SHAPE_NAMES = [cls.NAME for cls in SHAPE_CLASSES]

# What a keyframe key's field name says about the field, whichever kind it is on
# (a self-check keeps a name that several kinds share consistent).
OBST_FIELD_FMT = {p.field: p.fmt for cls in SHAPE_CLASSES for p in cls.NUMERIC_PARAMS}
OBST_FIELD_IS_FLOAT = {field: is_float for cls in SHAPE_CLASSES
                       for field, _label, is_float in cls.KEYFRAME_FIELDS}


def offset_bound(shape, bound, thickness):
    """A copy of `bound` moved outward by a fixed absolute `thickness`
    (negative moves it inward): every arc's radius changes by `thickness`,
    and every edge shifts along its own outward normal by `thickness`, so it gives
    a ring of CONSTANT width everywhere (scaling the shape instead would shift each
    point in proportion to its distance from the scaling centre, a tiny fraction of
    a pixel near that centre). Only for the circles and arcs+edges layouts."""
    b = bound.copy()
    layout = SHAPE_CLASSES[shape].LAYOUT
    if layout == LAYOUT_CIRCLES:
        n = int(b[1])
        idx = 2
        for _ in range(n):
            b[idx + 2] = max(b[idx + 2] + thickness, 1e-9)   # r
            idx += 3
    elif layout == LAYOUT_ARCS_EDGES:
        n_a = int(b[1])
        n_e = int(b[2])
        idx = 3
        for _ in range(n_a):
            b[idx + 2] = max(b[idx + 2] + thickness, 1e-9)   # r
            idx += 5
        for _ in range(n_e):
            ax, ay, bx, by = b[idx], b[idx + 1], b[idx + 2], b[idx + 3]
            ex, ey = bx - ax, by - ay
            elen = sqrt(ex * ex + ey * ey)
            if elen > 1e-12:
                # outward normal, matching arcs_edges_hit's own convention
                nx, ny = ey / elen, -ex / elen
                b[idx] = ax + nx * thickness
                b[idx + 1] = ay + ny * thickness
                b[idx + 2] = bx + nx * thickness
                b[idx + 3] = by + ny * thickness
            idx += 4
    return b


def make_bound(shape, params):
    """The packed shape array for SHAPE_* code `shape` and its params() tuple."""
    return SHAPE_CLASSES[shape].bound_from_params(params)


def obstacle_tab_title(index, o):
    """The title of obstacle `o`'s section of the keyframe settings list (`index` is
    its place in the list, from 0)."""
    return f"Obstacle #{index + 1} ({o.NAME})"


def render_obstacle_thumbnail(o, size):
    """A size x size RGB thumbnail of just obstacle `o`'s own shape (not a
    piece of the simulated image), zoomed so it fills the thumbnail
    regardless of its actual world size, drawn as the same two-layer outline
    it gets in the image (outer layer = incoming effect, inner layer =
    outgoing effect) around a plain grey fill, on a mid-grey background so
    all three effect colors (including black) show up."""
    extent = max(o.extent(), 1e-6)
    scale = (size / 2.0) * 0.86 / extent            # zoom to fill, with a small margin
    ys, xs = np.mgrid[0:size, 0:size]
    wx = (xs + 0.5 - size / 2.0) / scale
    wy = -(ys + 0.5 - size / 2.0) / scale
    bound = o.bound()
    mask = inside_boundary_grid(wx, wy, o.KIND, bound)
    layer = 2                                       # pixels per outline layer
    eroded, dilated = mask.copy(), mask.copy()
    for dy, dx in ((-layer, 0), (layer, 0), (0, -layer), (0, layer)):
        eroded &= np.roll(mask, (dy, dx), axis=(0, 1))
        dilated |= np.roll(mask, (dy, dx), axis=(0, 1))
    canvas = np.full((size, size, 3), THUMB_BACKGROUND, dtype=np.uint8)
    canvas[mask] = THUMB_FILL
    canvas[dilated & ~mask] = OBST_EFFECT_RGB[o.in_effect]
    canvas[mask & ~eroded] = OBST_EFFECT_RGB[o.out_effect]
    return canvas


def arc_loop_distance(bound, x, y):
    """On the grid (x, y): the distance from each point to the nearest arc of an arc-loops
    shape (to the arc itself where the point's angle is within it, else to its end points)."""
    best = np.full(x.shape, 1e30)
    for k in range(int(bound[1])):
        cx, cy, r, lo, span = bound[3 + ARC_FIELDS * k:3 + ARC_FIELDS * (k + 1)]
        dx, dy = x - cx, y - cy
        within = np.mod(np.arctan2(dy, dx) - lo, 2.0 * pi) <= span
        d = np.where(within, np.abs(np.hypot(dx, dy) - r), 1e30)
        for end in (lo, lo + span):
            d = np.minimum(d, np.hypot(x - (cx + r * cos(end)), y - (cy + r * sin(end))))
        best = np.minimum(best, d)
    return best


def ring_layer_masks(shape, bound, x, y, layer_px=OBST_RING_LAYER_PX):
    """(outer, inner) boolean masks on the world-coordinate grid (x, y) for a
    shape's two-layer outline: `outer` is the band just outside its edge and
    `inner` the band just inside it, each `layer_px` pixels thick."""
    w = layer_px * PIXEL
    layout = SHAPE_CLASSES[shape].LAYOUT
    if layout == LAYOUT_ELLIPSE:
        m11, m12, m22 = bound[1], bound[2], bound[3]
        q = m11 * x * x + 2.0 * m12 * x * y + m22 * y * y
        grad = 2.0 * np.sqrt((m11 * x + m12 * y) ** 2 + (m12 * x + m22 * y) ** 2)
        sd = (q - 1.0) / np.maximum(grad, 1e-9)    # first-order signed distance (+ outside)
        return (sd >= 0.0) & (sd < w), (sd < 0.0) & (sd >= -w)
    if layout == LAYOUT_POLYGON:
        h = bound[1]
        n = int(bound[2])
        f = np.full(x.shape, -1e9)
        for k in range(n):
            np.maximum(f, bound[3 + 2 * k] * x + bound[4 + 2 * k] * y, out=f)
        sd = f - h
        return (sd >= 0.0) & (sd < w), (sd < 0.0) & (sd >= -w)
    inside = inside_boundary_grid(x, y, shape, bound)
    if layout == LAYOUT_ARC_LOOPS:
        near = arc_loop_distance(bound, x, y) < w       # (offsetting an arc outward would move a
        return near & ~inside, near & inside            # concave arc into the shape)
    bigger = inside_boundary_grid(x, y, shape, offset_bound(shape, bound, w))
    smaller = inside_boundary_grid(x, y, shape, offset_bound(shape, bound, -w))
    return bigger & ~inside, inside & ~smaller


def draw_obstacle_outline(img, shape, bound, x, y, in_effect, out_effect):
    """Draws an obstacle's two-layer outline into `img` (in place), on the
    world-coordinate grid (x, y) already shifted to be relative to the
    obstacle's centre: the outer layer in its incoming effect's color, the
    inner layer in its outgoing effect's."""
    outer, inner = ring_layer_masks(shape, bound, x, y)
    img[outer] = OBST_EFFECT_RGB[int(in_effect)]
    img[inner] = OBST_EFFECT_RGB[int(out_effect)]


def default_starting_obstacle(id_):
    """The obstacle the app starts with: the unit circle at the origin (which
    fills the image), reflecting rays both ways, and firing rays inside
    itself."""
    o = EllipseShape(id_)
    o.width = o.height = DEFAULT_WIDTH
    o.in_effect = o.out_effect = OBSTACLE_EFFECT_REFLECT
    o.fire_inside = True
    return o


def build_obstacle_arrays(obstacles):
    """[Shape, ...] (live obstacles, or frozen copies such as resolved video-frame
    values) -> an ObstacleArrays: parallel numpy arrays, one entry per obstacle (an obstacle that passes
    rays through in both directions is kept, since it can still fire rays and
    still has an outline). `effects` is an (n, 2) array: column 0 the
    incoming effect, column 1 the outgoing effect. The returned cx/cy are
    each obstacle's actual resolved centre (from its polar position plus its
    x/y offset), not the raw offset fields."""
    active = list(obstacles)
    n = len(active)
    shapes = np.empty(n, dtype=np.int64)
    bounds = np.zeros((n, OBST_BOUND_LEN), dtype=np.float64)
    cx = np.empty(n, dtype=np.float64)
    cy = np.empty(n, dtype=np.float64)
    effects = np.empty((n, 2), dtype=np.int64)
    fire_inside = np.empty(n, dtype=np.bool_)
    for idx, o in enumerate(active):
        shapes[idx] = o.KIND
        b = o.bound()
        bounds[idx, :len(b)] = b
        cx[idx], cy[idx] = o.center()
        effects[idx, 0] = o.in_effect
        effects[idx, 1] = o.out_effect
        fire_inside[idx] = o.fire_inside
    return ObstacleArrays(shapes=shapes, bounds=bounds, cx=cx, cy=cy, effects=effects,
                          fire_inside=fire_inside)


def obstacles_signature(obstacles):
    """A plain-data snapshot of every obstacle's settings, for detecting
    whether the obstacle list has changed since the last Start."""
    return tuple(o.state() for o in obstacles)


def resolve_obstacle_arrays(obstacle_snapshots, keyframes_sorted, frame_number):
    """An ObstacleArrays (parallel arrays) for one
    video frame - see resolve_obstacles."""
    return build_obstacle_arrays(
        resolve_obstacles(obstacle_snapshots, keyframes_sorted, frame_number))


@strict_record
class ObstacleRing:
    """What draw_obstacle_outline needs to outline one obstacle: its packed shape, where it
    is, and the effects the outline's two layers show."""
    shape: int
    bound: np.ndarray
    cx: float
    cy: float
    in_effect: int
    out_effect: int


@strict_record
class SettingEntry:
    """One row of the keyframe settings list: a setting (a static one, or one field of
    an obstacle or a view), where it is listed and what it falls back to."""
    key: str
    tab: str
    label: str
    is_float: bool
    fallback: object             # its value when no keyframe defines it


@strict_record
class RaySettingSpec:
    """What the program knows about one simulation setting - one row per setting, so
    that adding a RaySettings field means adding one row here rather than touching a
    chain of places."""
    path: tuple            # where the value lives in a RaySettings, e.g. ("fuzz", "seed")
    control: str           # the App attribute holding the control that shows the value
    widget: str            # the App attribute of the widget its modified-outline surrounds
    kind: str              # how set_setting_control writes the control: "text", "int_text",
                           # "float_text", "number", "bool" or "choice"
    names: tuple           # for "choice": the names the value indexes into
    tab: str               # the keyframe settings list section it is listed under
    label: str             # its name there


def _spec(path, control, widget, kind, names=()):
    return RaySettingSpec(path=path, control=control, widget=widget, kind=kind, names=names)


def _spec(path, control, widget, kind, tab, label, names=()):
    return RaySettingSpec(path=path, control=control, widget=widget, kind=kind, names=names,
                          tab=tab, label=label)


# Every simulation setting, in the order the keyframe settings list shows them within
# each tab. run_self_checks() makes sure these rows and the RaySettings fields are the
# same set.
RAY_SETTING_SPECS = {
    "iter": _spec(("max_iter",), "iter_var", "iter_entry", "text",
                  "Simulation", "Iteration limit"),
    "dist": _spec(("max_dist",), "dist_var", "dist_entry", "text",
                  "Simulation", "Distance limit"),
    "seg_mult": _spec(("seg_mult",), "seg_mult_scale", "seg_mult_scale", "number",
                      "Simulation", "Segment length multiplier"),
    "emit_mode": _spec(("emit_mode",), "emit_mode_var", "emit_mode_combo", "choice",
                       "Simulation", "Emission direction", EMIT_MODE_NAMES),
    "emit_target_r": _spec(("emit_target_r",), "emit_target_r_var", "emit_target_r_entry",
                           "float_text", "Simulation", "Emission target distance (r)"),
    "emit_target_theta": _spec(("emit_target_theta",), "emit_target_theta_var",
                               "emit_target_theta_entry", "float_text", "Simulation",
                               "Emission target angle (\u03b8)"),
    "emit_target_cx": _spec(("emit_target_cx",), "emit_target_cx_var", "emit_target_cx_entry",
                            "float_text", "Simulation", "Emission target offset X"),
    "emit_target_cy": _spec(("emit_target_cy",), "emit_target_cy_var", "emit_target_cy_entry",
                            "float_text", "Simulation", "Emission target offset Y"),
    "emit_offset": _spec(("emit_offset",), "emit_offset_scale", "emit_offset_scale", "number",
                         "Simulation", "Emission angle offset"),
    "confine_window": _spec(("confine_window",), "confine_window_var", "confine_window_entry",
                            "int_text", "Simulation", "Confinement window"),
    "confine_method": _spec(("confine_method",), "confine_method_var", "confine_method_combo",
                            "choice", "Simulation", "Confinement method", CONFINE_METHOD_NAMES),
    "confine_threshold": _spec(("confine_threshold",), "confine_threshold_var",
                               "confine_threshold_entry", "float_text", "Simulation",
                               "Confinement threshold"),
    "fuzz_on": _spec(("fuzz", "enabled"), "fuzz_enabled_var", "fuzz_enable_check", "bool",
                     "Fuzzing", "Enabled"),
    "emit_jitter": _spec(("fuzz", "emit_jitter"), "emit_jitter_scale", "emit_jitter_scale",
                         "number", "Fuzzing", "Emission jitter"),
    "bnd_offset": _spec(("fuzz", "bnd_offset"), "bnd_offset_scale", "bnd_offset_scale",
                        "number", "Fuzzing", "Boundary offset"),
    "bnd_jitter": _spec(("fuzz", "bnd_jitter"), "bnd_jitter_scale", "bnd_jitter_scale",
                        "number", "Fuzzing", "Boundary jitter"),
    "bnd_mult": _spec(("fuzz", "bnd_mult"), "bnd_mult_scale", "bnd_mult_scale", "number",
                      "Fuzzing", "Boundary multiplier"),
    "self_offset": _spec(("fuzz", "self_offset"), "self_offset_scale", "self_offset_scale",
                         "number", "Fuzzing", "Self offset"),
    "self_jitter": _spec(("fuzz", "self_jitter"), "self_jitter_scale", "self_jitter_scale",
                         "number", "Fuzzing", "Self jitter"),
    "self_mult": _spec(("fuzz", "self_mult"), "self_mult_scale", "self_mult_scale", "number",
                       "Fuzzing", "Self multiplier"),
    "seed": _spec(("fuzz", "seed"), "seed_var", "seed_entry", "text", "Fuzzing", "Seed"),
    "buddha_blur_radius": _spec(("buddha_blur", "radius"), "buddha_blur_radius_scale",
                                "buddha_blur_radius_scale", "number", "Buddha blur", "Radius"),
    "buddha_blur_type": _spec(("buddha_blur", "type"), "buddha_blur_type_var",
                              "buddha_blur_type_combo", "choice", "Buddha blur", "Blur type",
                              BUDDHA_BLUR_LABELS),
}

# Every setting a keyframe can define: (key, tab it lives on, short label for the
# settings list), in the order shown within each tab group - the RAY_SETTING_SPECS
# rows, then the one decoration that is not part of a RaySettings.
SETTING_INFO = ([(key, spec.tab, spec.label) for key, spec in RAY_SETTING_SPECS.items()]
                + [("outline", "Decorations", "Outline")])
SETTING_TABS_ORDER = ["Simulation", "Fuzzing", "Buddha blur", "Decorations"]
SETTING_LABEL = {key: label for key, _tab, label in SETTING_INFO}
SETTING_TAB = {key: tab for key, tab, _label in SETTING_INFO}


def _field_types(record_class):
    return {f.name: f.type for f in dataclasses.fields(record_class)}


def ray_setting_leaf_paths(record_class=None, prefix=()):
    """Every value a RaySettings holds, as a path: ("max_iter",), ("fuzz", "seed"), ..."""
    paths = []
    for f in dataclasses.fields(record_class or RaySettings):
        if dataclasses.is_dataclass(f.type):
            paths += ray_setting_leaf_paths(f.type, prefix + (f.name,))
        else:
            paths.append(prefix + (f.name,))
    return paths


def ray_settings_to_dict(settings):
    """{setting key: value} for every RAY_SETTING_SPECS row, read from a RaySettings."""
    values = {}
    for key, spec in RAY_SETTING_SPECS.items():
        value = settings
        for name in spec.path:
            value = getattr(value, name)
        values[key] = value
    return values


def ray_settings_from_dict(values):
    """A RaySettings from {setting key: value} (for example the values resolved from
    keyframes, where an int setting may arrive as a float): each value is converted
    to its field's type - int fields rounded, bool fields made bool, anything else
    kept as it is (the record then checks it)."""
    top, nested = {}, {}
    for key, spec in RAY_SETTING_SPECS.items():
        *outer, leaf = spec.path
        assert len(outer) <= 1, spec.path
        record_class = RaySettings
        for name in outer:
            record_class = _field_types(record_class)[name]
        want = _field_types(record_class)[leaf]
        value = values[key]
        if want is int:
            value = int(round(value))
        elif want is bool:
            value = bool(value)
        (nested.setdefault(outer[0], {}) if outer else top)[leaf] = value
    for name, fields in nested.items():
        top[name] = _field_types(RaySettings)[name](**fields)
    return RaySettings(**top)


# Only the float settings are ever smoothly interpolated; everything else (which
# shape, colors, metrics, sides, flags, seed) always holds the value of the nearest
# earlier keyframe that defines it. Derived: a float field of RaySettings is a float
# setting.
def _is_float_setting(spec):
    record_class = RaySettings
    for name in spec.path[:-1]:
        record_class = _field_types(record_class)[name]
    return _field_types(record_class)[spec.path[-1]] is float


SETTING_FLOAT_KEYS = {key for key, spec in RAY_SETTING_SPECS.items() if _is_float_setting(spec)}

# Which attribute of App holds the main control of each SETTING_INFO setting (a
# widget, or a handle whose `.widget` is the entry) - where its blue "modified"
# outline is drawn. run_self_checks() keeps this in step with SETTING_INFO.
SETTING_WIDGET_ATTRS = {**{key: spec.widget for key, spec in RAY_SETTING_SPECS.items()},
                        "outline": "deco_outline_check"}


DEFAULT_SETTING_VALUES = {**ray_settings_to_dict(DEFAULT_RAY_SETTINGS), "outline": True}


def confine_rule(window, method, threshold):
    """The confinement rule in the form the simulation kernels expect."""
    return ConfineRule(window=int(window), method=int(method), threshold=float(threshold))


# What was always hard-wired (a 4-point window, the Box method, one pixel).
CONFINE_DEFAULT = confine_rule(DEFAULT_CONFINE_WINDOW, DEFAULT_CONFINE_METHOD,
                               DEFAULT_CONFINE_THRESHOLD)


def obstacle_center_from(r, theta, cx, cy):
    """Same as Shape.center(), for already-resolved field values (used
    when animating obstacles between keyframes)."""
    th = radians(theta)
    return r * cos(th) + cx, r * sin(th) + cy


def emission_rule(mode, target_x, target_y, offset):
    """The emission direction rule in the form the simulation kernels expect."""
    return EmissionRule(mode=int(mode), target_x=float(target_x), target_y=float(target_y),
                        offset=float(offset))


# What was always hard-wired: emit up and to the right (Up, offset -45).
EMISSION_DEFAULT = emission_rule(
    DEFAULT_EMIT_MODE,
    *obstacle_center_from(DEFAULT_EMIT_TARGET_R, DEFAULT_EMIT_TARGET_THETA,
                          DEFAULT_EMIT_TARGET_CX, DEFAULT_EMIT_TARGET_CY),
    DEFAULT_EMIT_OFFSET)


def settings_emission(settings):
    """emission_rule() for a RaySettings (its target's four-part position turned
    into the point the kernels aim at)."""
    tx, ty = obstacle_center_from(settings.emit_target_r, settings.emit_target_theta,
                                  settings.emit_target_cx, settings.emit_target_cy)
    return emission_rule(settings.emit_mode, tx, ty, settings.emit_offset)


def settings_confine(settings):
    """confine_rule() for a RaySettings."""
    return confine_rule(settings.confine_window, settings.confine_method,
                        settings.confine_threshold)


def resolve_value(key, keyframes_sorted, frame_number, fallback, is_float):
    """The value of one setting at `frame_number`, using only the keyframes
    (already sorted by frame_index) that actually define `key` - keyframes
    lacking it are skipped entirely, both for holding and for interpolation,
    so different settings can animate over different stretches of the
    timeline. `fallback` is used when no keyframe defines it at all;
    `is_float` says whether it's ever linearly interpolated (True) or always
    held (False)."""
    have = [kf for kf in keyframes_sorted if key in kf.data]
    if not have:
        return fallback

    active = have[0]
    active_i = 0
    for i, kf in enumerate(have):
        if kf.frame_index <= frame_number:
            active = kf
            active_i = i
        else:
            break

    if (not is_float or frame_number <= active.frame_index
            or active_i + 1 >= len(have)):
        return active.data[key]      # held (before the first / after the last too)
    nxt = have[active_i + 1]
    if nxt.frame_index == active.frame_index:
        return active.data[key]
    t = (frame_number - active.frame_index) / (nxt.frame_index - active.frame_index)
    a, b = active.data[key], nxt.data[key]
    return a + (b - a) * t


def resolve_obstacles(obstacle_snapshots, keyframes_sorted, frame_number):
    """[frozen Shape, ...] for one video frame: the set of obstacles is
    fixed (whichever existed when the render started), but every field of
    every one of them is independently resolved (animated or held) from the
    keyframes, exactly like the other settings - falling back to that
    obstacle's own value (at render-start time) wherever no keyframe defines
    a given field."""
    resolved = []
    for snap in obstacle_snapshots:
        values = {field: resolve_value(f"obst:{snap.id}:{field}", keyframes_sorted,
                                       frame_number, getattr(snap, field), is_float)
                  for field, _label, is_float in snap.KEYFRAME_FIELDS}
        resolved.append(snap.with_values(**values))
    return resolved


def resolve_setting(key, keyframes_sorted, frame_number):
    """resolve_value() for one of the static settings in SETTING_INFO."""
    return resolve_value(key, keyframes_sorted, frame_number,
                         DEFAULT_SETTING_VALUES[key], key in SETTING_FLOAT_KEYS)


def resolve_frame_settings(keyframes_sorted, frame_number):
    """A full KeyframeData for one video frame's simulation, with every
    setting resolved independently via resolve_setting(). Coloration isn't
    part of this - see resolve_views(), called separately (mirroring
    resolve_obstacle_arrays), since views are a dynamic list like obstacles,
    not a fixed set of settings."""
    v = {key: resolve_setting(key, keyframes_sorted, frame_number)
        for key, _tab, _label in SETTING_INFO}
    settings = ray_settings_from_dict(v)
    return KeyframeData(settings=settings, outline=bool(v["outline"]))


def effective_dist(max_dist):
    """The distance limit passed to the simulation: infinite when 0 (no limit)."""
    return float("inf") if max_dist == 0 else float(max_dist)


def fuzz_rule(f):
    """FuzzParams -> the FuzzRule the simulation kernels expect (the seed is handled
    separately, by seeding the RNG before each ray is traced)."""
    return FuzzRule(enabled=bool(f.enabled), emit_jitter=float(f.emit_jitter),
                    bnd_offset=float(f.bnd_offset), bnd_jitter=float(f.bnd_jitter),
                    bnd_mult=float(f.bnd_mult), self_offset=float(f.self_offset),
                    self_jitter=float(f.self_jitter), self_mult=float(f.self_mult))


def settings_suffix(settings):
    """Simulation settings as a filename fragment: plain characters, no spaces,
    e.g. 'iter32-distNONE'."""
    parts = [f"iter{settings.max_iter}",
             "distNONE" if settings.max_dist == 0 else f"dist{settings.max_dist}"]
    f = settings.fuzz
    if f.enabled:
        parts += ["fuzz",
                  f"ej{f.emit_jitter:.1f}",
                  f"bo{f.bnd_offset:.1f}", f"bj{f.bnd_jitter:.1f}",
                  f"bm{f.bnd_mult:.2f}",
                  f"so{f.self_offset:.1f}", f"sj{f.self_jitter:.1f}",
                  f"sm{f.self_mult:.2f}", f"seed{f.seed}"]
    return "-".join(parts)


def shift_into_area(rect, area):
    """The smallest (dx, dy) that moves `rect` (left, top, right, bottom) inside
    `area` (same form). A rect larger than the area is lined up by its left and
    top edges instead, so the title bar stays reachable."""
    left, top, right, bottom = rect
    a_left, a_top, a_right, a_bottom = area
    dx = dy = 0
    if right > a_right:
        dx = a_right - right
    if bottom > a_bottom:
        dy = a_bottom - bottom
    if left + dx < a_left:
        dx = a_left - left
    if top + dy < a_top:
        dy = a_top - top
    return dx, dy


def parse_geometry_position(geometry):
    """(x, y) from a Tk geometry string like "1904x936+0+104" or "1904x936-8+-3";
    None if it has no position."""
    match = re.fullmatch(r"\d+x\d+([+-])(-?\d+)([+-])(-?\d+)", geometry)
    if match is None:
        return None
    sign_x, x, sign_y, y = match.groups()
    return (int(x) if sign_x == "+" else -int(x)), (int(y) if sign_y == "+" else -int(y))


def window_rect_and_work_area(root):
    """((left, top, right, bottom) of the whole window including its frame, and the
    (left, top, right, bottom) of the usable area - the work area, taskbar excluded -
    of the monitor it is mostly on), in screen pixels. On Windows both come from
    the operating system; elsewhere the window's client rectangle and the whole
    screen are the best available."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class MonitorInfo(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                            ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(root.winfo_id())
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            info = MonitorInfo()
            info.cbSize = ctypes.sizeof(MonitorInfo)
            monitor = user32.MonitorFromWindow(hwnd, 2)       # MONITOR_DEFAULTTONEAREST
            user32.GetMonitorInfoW(monitor, ctypes.byref(info))
            work = info.rcWork
            return ((rect.left, rect.top, rect.right, rect.bottom),
                    (work.left, work.top, work.right, work.bottom))
        except Exception:                           # noqa: BLE001
            pass
    x, y = root.winfo_rootx(), root.winfo_rooty()
    return ((x, y, x + root.winfo_width(), y + root.winfo_height()),
            (0, 0, root.winfo_screenwidth(), root.winfo_screenheight()))


def unused_path(path):
    """`path`, or with "-2", "-3", ... before its extension if it already exists
    (two captures in the same second must not overwrite each other)."""
    base, ext = os.path.splitext(path)
    candidate, n = path, 1
    while os.path.exists(candidate):
        n += 1
        candidate = f"{base}-{n}{ext}"
    return candidate


def take_ui_screenshot(root, cause):
    """Development aid: grabs the app's own window from the screen and saves
    it as DEV_DIR/screenshots/<FILE_VERSION>-YYYYMMDD-hhmmss-<cause>.png
    (`cause` is "startup" or "manual"), returning the
    path - but only if the DEV_DIR folder already exists (this never creates
    it) and Pillow is installed; otherwise does nothing and returns None. It
    grabs the actual screen pixels inside the window's rectangle, so anything
    covering the window at that moment appears in the shot instead."""
    if cause not in ("startup", "manual"):
        raise ValueError(f"screenshot cause must be startup or manual, not {cause!r}")
    if external.ImageGrab is None or not os.path.isdir(paths.DEV_DIR):
        return None
    root.update()                                  # finish drawing, settle geometry
    x, y = root.winfo_rootx(), root.winfo_rooty()
    img = external.ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height()))
    shots_dir = os.path.join(paths.DEV_DIR, "screenshots")
    os.makedirs(shots_dir, exist_ok=True)
    path = unused_path(os.path.join(
        shots_dir, f"{FILE_VERSION}-{time.strftime('%Y%m%d-%H%M%S')}-{cause}.png"))
    img.save(path)
    return path


def dump_app_state(app, cause, include_large_arrays, exc_info=None):
    """Development aid: pickles everything the application holds into
    DEV_DIR/dumps/<FILE_VERSION>-YYYYMMDD-hhmmss-<cause>.pkl (`cause` is "startup",
    "crash" or "manual") using dev/debug_dump.py, and returns the path; without
    `include_large_arrays` every big array in it is replaced by a stand-in
    (dev/debug_dump.py has the size rule). Does
    nothing and returns None unless the DEV_DIR folder and that file exist.
    Never raises: a failure is reported on stderr and otherwise ignored, so a
    debugging aid can never add a crash (or hide the one being dumped)."""
    if cause not in ("startup", "crash", "manual"):
        raise ValueError(f"dump cause must be startup, crash or manual, not {cause!r}")
    try:
        dumper_path = os.path.join(paths.DEV_DIR, "debug_dump.py")
        if not os.path.isfile(dumper_path):
            return None
        import importlib.util
        spec = importlib.util.spec_from_file_location("debug_dump", dumper_path)
        dumper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dumper)
        dumps_dir = os.path.join(paths.DEV_DIR, "dumps")
        os.makedirs(dumps_dir, exist_ok=True)
        path = unused_path(os.path.join(
            dumps_dir, f"{FILE_VERSION}-{time.strftime('%Y%m%d-%H%M%S')}-{cause}.pkl"))
        return dumper.dump_everything(
            app, globals(), path, cause,
            include_large_arrays=include_large_arrays, exc_info=exc_info,
            file_version=FILE_VERSION)
    except Exception as exc:                        # noqa: BLE001
        print(f"Debug dump failed: {exc}", file=sys.stderr)
        return None


def parse_iter_limit(text):
    """Whole number in [ITER_LIMIT_MIN, ITER_LIMIT_MAX] from entry text, else None."""
    if re.fullmatch(r"[0-9]{1,4}", text):
        value = int(text)
        if ITER_LIMIT_MIN <= value <= ITER_LIMIT_MAX:
            return value
    return None


def parse_dist_limit(text):
    """Non-negative number in [DIST_LIMIT_MIN, DIST_LIMIT_MAX] from entry
    text (0 = no limit), else None. A whole number is returned as an int (so
    it displays cleanly, e.g. "5" not "5.0"); anything with a fractional
    part stays a float, rounded to 2 decimal places."""
    try:
        value = round(float(text), 2)
    except ValueError:
        return None
    if DIST_LIMIT_MIN <= value <= DIST_LIMIT_MAX:
        return int(value) if value == int(value) else value
    return None


def parse_confine_window(text):
    """Whole number of path points in [CONFINE_WINDOW_MIN, CONFINE_WINDOW_MAX]
    from entry text, else None."""
    if re.fullmatch(r"[0-9]{1,3}", text):
        value = int(text)
        if CONFINE_WINDOW_MIN <= value <= CONFINE_WINDOW_MAX:
            return value
    return None


def parse_confine_threshold(text):
    """The confinement threshold from entry text: any finite number strictly
    greater than zero, however small (plain or scientific notation, e.g.
    0.004 or 1e-300), else None. Deliberately not rounded or clamped."""
    if not re.fullmatch(r"(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?", text):
        return None
    value = float(text)
    return value if math.isfinite(value) and value > 0.0 else None


def parse_target_coordinate(text):
    """A coordinate of the emission target point from entry text: any finite
    number, negative or positive, however large or small (plain or scientific
    notation, e.g. -0.5 or 3e8), else None. Not rounded or clamped."""
    if not re.fullmatch(r"-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?", text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def parse_seed(text):
    """Whole number seed from entry text (blank counts as 0), else None."""
    if text == "":
        return 0
    if re.fullmatch(r"[0-9]{1,9}", text):
        return int(text)
    return None


def parse_positive_int(text, max_digits=6):
    """A whole number of at least 1 (up to max_digits digits), else None."""
    if re.fullmatch(rf"[0-9]{{1,{max_digits}}}", text):
        value = int(text)
        if value >= 1:
            return value
    return None


def round_to_multiple_of_4(n):
    """The nearest positive multiple of 4 to n (minimum 4, never 0) - ffmpeg
    needs even width/height to encode most video formats; rounding to a
    multiple of 4 (rather than just 2) gives some safety margin over that
    bare minimum for codecs pickier still (some require a multiple of 4 or
    8, e.g. under chroma subsampling with certain block sizes)."""
    return max(4, round(n / 4.0) * 4)


def parse_resolution_dimension(text, max_digits=6):
    """A resolution field's value: parsed the same way as any other
    positive whole number, then rounded to the nearest multiple of 4 - or
    None if the text isn't a valid whole number at all. Every width/height
    export control (New image, video) reads its value through this, so an
    exported frame's dimensions are always a multiple of 4, regardless of
    exactly what was typed."""
    value = parse_positive_int(text, max_digits=max_digits)
    if value is None:
        return None
    return round_to_multiple_of_4(value)


# Saving a freshly generated image (as opposed to the live preview): what to
# render, and which format to encode it in
SAVE_SOURCE_OPTIONS = ["Current image", "New image"]
VIDEO_FORMATS = ["mp4 (h264)", "GIF", "APNG", "mkv (ffv1)", "JXL"]
VIDEO_BITRATE_MODES = ["Constant quality", "Target bitrate", "Variable bitrate"]

# Export designer: where (if anywhere) each kind of info text goes
COLOR_INFO_LOCATIONS = ["None", "Overlay", "Bottom margin"]
FIGURE_INFO_LOCATIONS = ["None", "Left margin"]


def default_filename(settings=None, ext="png", quality_tag=None):
    """programname-YYYYMMDD-hhmmss-<settings>[.<quality_tag>].<ext>, stamped with the
    current time."""
    name = f"{paths.PROGRAM_NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
    if settings is not None:
        name += "-" + settings_suffix(settings)
    if quality_tag:
        name += "." + quality_tag
    return f"{name}.{ext}"


def image_filename(settings, fmt, jxl_lossless, jxl_quality):
    """The default name of a saved still image: a JPEG XL file also says its quality
    right before the extension (name.q90.jxl; q100 for lossless)."""
    tag = jxl_quality_tag(jxl_lossless, jxl_quality) if fmt == "jxl" else None
    return default_filename(settings, fmt, tag)


# --------------------------------------------------------------------------- #
# Shape geometry
#
# The simulation kernels receive a shape code and a float array `bound`:
#   bound[0]   area of the shape
#   ellipse:   bound[1:4] = m11, m12, m22, where a point is inside when
#              m11*x^2 + 2*m12*x*y + m22*y^2 < 1
#   polygon:   bound[1] = apothem (distance from centre to each edge),
#              bound[2] = number of sides n,
#              bound[3+2k], bound[4+2k] = outward unit normal of edge k
#              (a point is inside when normal_k . p < apothem for every k)
# --------------------------------------------------------------------------- #
def ellipse_bound(width, height, rotation):
    """
    Boundary description of an ellipse with full extents `width` x `height`,
    turned counterclockwise by `rotation` degrees. Written as p^T M p < 1 with
    M = R diag(1/a^2, 1/b^2) R^T (a, b = semi-axes). 2.0 x 2.0 with no rotation
    is exactly the unit circle (M = identity).
    """
    a = max(width * 0.5, AXIS_MIN)
    b = max(height * 0.5, AXIS_MIN)
    th = radians(rotation)
    c, s = cos(th), sin(th)
    ia, ib = 1.0 / (a * a), 1.0 / (b * b)
    return np.array([pi * a * b,
                     c * c * ia + s * s * ib,
                     c * s * (ia - ib),
                     s * s * ia + c * c * ib], dtype=np.float64)


def polygon_bound(sides, rotation, radius=1.0):
    """
    Boundary description of a regular polygon with `sides` sides whose
    vertices lie on a circle of the given `radius` (1.0 = the unit circle). With
    rotation 0 one vertex points
    straight up; `rotation` (degrees) turns it counterclockwise.
    """
    n = int(sides)
    th = radians(rotation)
    bound = np.empty(3 + 2 * n, dtype=np.float64)
    bound[0] = 0.5 * n * radius * radius * sin(2.0 * pi / n)   # area
    bound[1] = radius * cos(pi / n)                            # apothem
    bound[2] = n
    for k in range(n):
        # vertex k sits at angle pi/2 + th + 2*pi*k/n; edge k's normal lies
        # half a step further round
        ang = pi / 2.0 + th + pi / n + 2.0 * pi * k / n
        bound[3 + 2 * k] = cos(ang)
        bound[4 + 2 * k] = sin(ang)
    return bound


# ---- shared bound layouts for the "arcs + edges" and "circle intersection" ---- #
# shape families. Both keep bound[0] = area, same as ellipse/polygon.
#
# "Arcs + edges" (Moss's egg, circular sector, circular segment, capsule): a
# small closed loop of circular arcs and straight, bounded line segments.
#   bound[1] = number of arcs (n_a); bound[2] = number of edges (n_e)
#   then n_a * (cx, cy, r, ang_lo, span) - a point is "on" the arc when its
#   angle from (cx, cy), normalized into [ang_lo, ang_lo + span), is in range
#   then n_e * (ax, ay, bx, by) - a bounded segment from (ax, ay) to (bx, by)
#
# "Arc loops" (trefoil, quatrefoil, lune, arbelos) use the same arrays as "arcs + edges"
# (with no edges), but the loop may be concave, so a point is inside by the even-odd rule
# (inside_arcs_parity), not by an AND of circles or a ray from the origin.
#
# "Circle intersection" (two-circle lens, Reuleaux polygon): the shape is
# exactly the intersection of a few full circles (so its ray-exit is just the
# minimum of each circle's own, and its obstacle-entry the same style of
# interval-intersection already used for the polygon obstacle).
#   bound[1] = number of circles (n_c); then n_c * (cx, cy, r)
ARC_FIELDS = 5
EDGE_FIELDS = 4
CIRCLE_FIELDS = 3


def pack_arcs_edges(area, arcs, edges):
    """arcs: list of (cx, cy, r, ang_lo, span) with angles in radians. edges:
    list of (ax, ay, bx, by)."""
    n_a, n_e = len(arcs), len(edges)
    bound = np.zeros(3 + ARC_FIELDS * n_a + EDGE_FIELDS * n_e, dtype=np.float64)
    bound[0] = area
    bound[1] = n_a
    bound[2] = n_e
    idx = 3
    for cx, cy, r, ang_lo, span in arcs:
        bound[idx:idx + 5] = (cx, cy, r, ang_lo, span)
        idx += 5
    for ax, ay, bx, by in edges:
        bound[idx:idx + 4] = (ax, ay, bx, by)
        idx += 4
    return bound


def pack_circles(area, circles):
    """circles: list of (cx, cy, r)."""
    n = len(circles)
    bound = np.zeros(2 + CIRCLE_FIELDS * n, dtype=np.float64)
    bound[0] = area
    bound[1] = n
    idx = 2
    for cx, cy, r in circles:
        bound[idx:idx + 3] = (cx, cy, r)
        idx += 3
    return bound


def _rot2(x, y, cos_t, sin_t):
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def estimate_area_numeric(shape, bound, extent, grid=150):
    """Fallback area estimate (grid sampling over a square of half-width
    `extent`) for shapes without a simple closed-form area - cheap enough
    since it only runs once per parameter change, not per ray (inside_boundary
    is the same compiled kernel the simulation itself uses)."""
    step = 2.0 * extent / grid
    count = 0
    for j in range(grid):
        y = (j + 0.5) * step - extent
        for i in range(grid):
            x = (i + 0.5) * step - extent
            if inside_boundary(x, y, shape, bound):
                count += 1
    return count * step * step


MOSS_FIT_SCALE = 0.8   # shrinks Moss's egg so its largest apex angle (180deg,
                       # its flattest/roundest case) leaves a comfortable
                       # margin inside the viewport, rather than exactly
                       # touching its edge


def moss_egg_bound(apex_angle_deg, size=1.0, rotation_deg=0.0):
    """
    Moss's egg (see Wikipedia): an oval of 4 tangent circular arcs, built from
    an isosceles triangle ABC with apex angle ACB = `apex_angle_deg` (60..180).
    AB is normalized to length 2, then the whole shape is re-centred on the
    vertical middle of its own bounding box (rather than left centred on M,
    the base semicircle's own centre, which sits well below the shape's true
    middle), scaled down for a comfortable fit - see MOSS_FIT_SCALE - and then
    by the user-facing `size` on top of that. With no rotation the egg's
    round end is at the bottom (-y), its pointed/capped end at the top (+y).
    """
    angle = max(min(apex_angle_deg, MOSS_ANGLE_MAX), MOSS_ANGLE_MIN)
    half = radians(angle) / 2.0
    L = 1.0 / sin(half)                        # CA = CB
    h = 1.0 / tan(half)                        # height of C above AB
    Ax, Ay = -1.0, 0.0
    Bx, By = 1.0, 0.0
    Cx, Cy = 0.0, h
    Mx, My = 0.0, 0.0
    ux, uy = (Cx - Bx) / L, (Cy - By) / L      # unit vector B -> C
    Dx, Dy = Bx + 2.0 * ux, By + 2.0 * uy
    vx, vy = (Cx - Ax) / L, (Cy - Ay) / L      # unit vector A -> C
    Ex, Ey = Ax + 2.0 * vx, Ay + 2.0 * vy
    CD = abs(2.0 - L)                          # = |C-D| = |C-E| (tangent radius)

    def arc_params(cx, cy, start_x, start_y, end_x, end_y, r):
        a_start = atan2(start_y - cy, start_x - cx)
        a_end = atan2(end_y - cy, end_x - cx)
        ang_lo = a_end
        span = (a_start - a_end) % (2.0 * pi)
        return (cx, cy, r, ang_lo, span)

    # Traversal order (clockwise, per the construction): B -> A (semicircle)
    # -> D (B-centred arc) -> E (C-centred arc, the "cap") -> B (A-centred arc)
    arcs = [
        arc_params(Mx, My, Bx, By, Ax, Ay, 1.0),
        arc_params(Bx, By, Ax, Ay, Dx, Dy, 2.0),
    ]
    if CD > 1e-9:
        arcs.append(arc_params(Cx, Cy, Dx, Dy, Ex, Ey, CD))
    arcs.append(arc_params(Ax, Ay, Ex, Ey, Bx, By, 2.0))

    # Re-centre vertically (the shape spans y = -1 at the bottom to y = h+CD
    # at the top; shift so those are equally far from the local origin) and
    # apply the comfortable-fit scale and the user-facing size on top of it.
    mid_y = (h + CD - 1.0) / 2.0
    s = MOSS_FIT_SCALE * size
    arcs = [(s * cx, s * (cy - mid_y), s * r, ang_lo, span)
           for cx, cy, r, ang_lo, span in arcs]

    rot = radians(rotation_deg)
    if rot != 0.0:
        c, sn = cos(rot), sin(rot)
        arcs = [(cx * c - cy * sn, cx * sn + cy * c, r, ang_lo + rot, span)
               for cx, cy, r, ang_lo, span in arcs]

    bound = pack_arcs_edges(0.0, arcs, [])
    bound[0] = estimate_area_numeric(SHAPE_MOSS_EGG, bound, 2.2 * size)
    return bound


def sector_bound(radius, span_deg, rotation_deg):
    """Circular sector (pie slice): apex at the local origin, radius `radius`,
    angular width `span_deg` (up to 180 degrees), bisector pointing at
    `rotation_deg`."""
    r = max(radius, 1e-6)
    span = radians(max(min(span_deg, SECTOR_SPAN_MAX), ARC_SPAN_MIN))
    rot = radians(rotation_deg)
    ang_lo = rot - span / 2.0
    ang_hi = rot + span / 2.0
    p_lo = (r * cos(ang_lo), r * sin(ang_lo))
    p_hi = (r * cos(ang_hi), r * sin(ang_hi))
    arcs = [(0.0, 0.0, r, ang_lo, span)]
    edges = [(0.0, 0.0, p_lo[0], p_lo[1]), (p_hi[0], p_hi[1], 0.0, 0.0)]
    area = 0.5 * r * r * span
    return pack_arcs_edges(area, arcs, edges)


def segment_bound(radius, span_deg, rotation_deg):
    """Circular segment: the "cap" of a circle of radius `radius` beyond a
    chord subtending `span_deg` at the centre, facing `rotation_deg`."""
    r = max(radius, 1e-6)
    span = radians(max(min(span_deg, ARC_SPAN_MAX), ARC_SPAN_MIN))
    rot = radians(rotation_deg)
    ang_lo = rot - span / 2.0
    ang_hi = rot + span / 2.0
    p_lo = (r * cos(ang_lo), r * sin(ang_lo))
    p_hi = (r * cos(ang_hi), r * sin(ang_hi))
    arcs = [(0.0, 0.0, r, ang_lo, span)]
    edges = [(p_hi[0], p_hi[1], p_lo[0], p_lo[1])]
    area = 0.5 * r * r * (span - sin(span))
    return pack_arcs_edges(area, arcs, edges)


def lens_bound(radius1, radius2, separation, rotation_deg):
    """Two-circle lens: the intersection of two circles of radii `radius1`,
    `radius2`, whose centres are `separation` apart along `rotation_deg`."""
    r1 = max(radius1, 1e-6)
    r2 = max(radius2, 1e-6)
    d = max(separation, 0.0)
    rot = radians(rotation_deg)
    c, s = cos(rot), sin(rot)
    c1 = (-0.5 * d * c, -0.5 * d * s)
    c2 = (0.5 * d * c, 0.5 * d * s)
    if d >= r1 + r2:
        area = 0.0                              # circles don't meet
    elif d <= abs(r1 - r2):
        area = pi * min(r1, r2) ** 2             # one circle sits inside the other
    else:
        a1 = acos(min(1.0, max(-1.0, (d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1))))
        a2 = acos(min(1.0, max(-1.0, (d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2))))
        tri = 0.5 * sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
        area = r1 * r1 * a1 + r2 * r2 * a2 - tri
    return pack_circles(area, [(c1[0], c1[1], r1), (c2[0], c2[1], r2)])


def reuleaux_bound(sides, rotation, radius=1.0):
    """Reuleaux polygon on a regular `sides`-gon (odd, >= 3) of circumradius
    `radius`: the intersection of `sides` circles, one centred at each vertex,
    each with the polygon's constant width as its radius."""
    n = int(sides)
    if n % 2 == 0:
        n += 1                                  # must be odd; nudge up
    n = max(n, 3)
    th = radians(rotation)
    w = 2.0 * radius * sin(pi * (n - 1) / (2.0 * n))   # constant width
    circles = []
    for k in range(n):
        ang = pi / 2.0 + th + 2.0 * pi * k / n
        circles.append((radius * cos(ang), radius * sin(ang), w))
    bound = pack_circles(0.0, circles)
    bound[0] = estimate_area_numeric(SHAPE_REULEAUX, bound, radius + w)
    return bound


def capsule_bound(length, radius, rotation_deg):
    """Capsule / stadium: a rectangle of the given `length` (the straight
    part) capped with a semicircle of `radius` at each end, oriented along
    `rotation_deg`."""
    ln = max(length, 0.0)
    r = max(radius, 1e-6)
    rot = radians(rotation_deg)
    c, s = cos(rot), sin(rot)
    half = ln / 2.0
    rx, ry = _rot2(half, 0.0, c, s)             # right cap centre
    lx, ly = _rot2(-half, 0.0, c, s)            # left cap centre
    # Local (pre-rotation) corner points, then rotated into place
    br = _rot2(half, -r, c, s)                  # bottom-right
    tr = _rot2(half, r, c, s)                   # top-right
    tl = _rot2(-half, r, c, s)                  # top-left
    bl = _rot2(-half, -r, c, s)                 # bottom-left
    arcs = [
        (rx, ry, r, rot - pi / 2.0, pi),        # right cap: -90deg..+90deg (rotated)
        (lx, ly, r, rot + pi / 2.0, pi),        # left cap: +90deg..+270deg (rotated)
    ]
    edges = [
        (bl[0], bl[1], br[0], br[1]),           # bottom edge, left -> right
        (tr[0], tr[1], tl[0], tl[1]),           # top edge, right -> left
    ]
    area = 2.0 * r * ln + pi * r * r
    return pack_arcs_edges(area, arcs, edges)


def region_arcs(circles, member):
    """The boundary of a region built from full circles [(cx, cy, r), ...], as a
    list of arcs (cx, cy, r, ang_lo, span): `member(x, y)` says whether a point is in
    the region. Each circle is cut where it crosses the others, and a piece is kept
    when the region is on one side of it only (probed just inside and just outside)."""
    unique = []
    for circle in circles:
        if not any(all(abs(circle[i] - other[i]) < 1e-9 for i in range(3)) for other in unique):
            unique.append(circle)
    tau = 2.0 * pi
    arcs = []
    for ci, (cx, cy, r) in enumerate(unique):
        cuts = [0.0]
        for cj, (ox, oy, other_r) in enumerate(unique):
            d = sqrt((ox - cx) ** 2 + (oy - cy) ** 2)
            if cj == ci or d >= r + other_r or d <= abs(r - other_r) or d == 0.0:
                continue
            half = acos(max(-1.0, min(1.0, (r * r - other_r * other_r + d * d) / (2.0 * d * r))))
            base = atan2(oy - cy, ox - cx)
            cuts += [(base + half) % tau, (base - half) % tau]
        cuts = sorted(set(round(a, 12) for a in cuts)) + [tau]
        eps = 1e-6 * max(r, 1e-3)
        kept = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi - lo < 1e-9:
                continue
            mid = 0.5 * (lo + hi)
            ux, uy = cos(mid), sin(mid)
            if member(cx + (r + eps) * ux, cy + (r + eps) * uy) != \
                    member(cx + (r - eps) * ux, cy + (r - eps) * uy):
                kept.append([lo, hi - lo])
        if len(kept) > 1 and kept[0][0] == 0.0 and abs(kept[-1][0] + kept[-1][1] - tau) < 1e-9:
            kept[-1][1] += kept.pop(0)[1]              # join the pieces either side of angle 0
        arcs += [(cx, cy, r, lo, span) for lo, span in kept]
    return arcs


def arc_loop_bound(shape, arcs, extent):
    """Pack `arcs` as an arc-loops shape of kind `shape`; its area is estimated."""
    bound = pack_arcs_edges(0.0, arcs, [])
    bound[0] = estimate_area_numeric(shape, bound, extent)
    return bound


def lobed_bound(shape, lobes, radius, lobe_size, rotation):
    """A trefoil / quatrefoil: the union of `lobes` equal circles whose centres lie on a
    circle round the origin, one pointing straight up at rotation 0. The farthest point of
    the shape is `radius` from the origin and each lobe's radius is `lobe_size` * `radius`."""
    big = max(radius, 1e-6)
    size = max(min(lobe_size, LOBE_SIZE_MAX), 0.5)
    r = size * big
    ring = big - r
    circles = []
    for k in range(lobes):
        ang = pi / 2.0 + radians(rotation) + 2.0 * pi * k / lobes
        circles.append((ring * cos(ang), ring * sin(ang), r))

    def member(x, y):
        return any((x - cx) ** 2 + (y - cy) ** 2 < rr * rr for cx, cy, rr in circles)
    return arc_loop_bound(shape, region_arcs(circles, member), 1.05 * big)


def lune_bound(radius, cut_radius, cut_offset, rotation):
    """A lune (crescent): the disc of radius `radius` about the origin, minus a second disc of
    radius `cut_radius` whose centre is `cut_offset` away, in the direction `rotation` (0 = +x)."""
    big = max(radius, 1e-6)
    cut = max(cut_radius, 1e-6)
    rot = radians(rotation)
    cut_cx, cut_cy = max(cut_offset, 0.0) * cos(rot), max(cut_offset, 0.0) * sin(rot)
    circles = [(0.0, 0.0, big), (cut_cx, cut_cy, cut)]

    def member(x, y):
        return x * x + y * y < big * big and (x - cut_cx) ** 2 + (y - cut_cy) ** 2 >= cut * cut
    return arc_loop_bound(SHAPE_LUNE, region_arcs(circles, member), 1.05 * big)


def arbelos_bound(radius, split, rotation):
    """An arbelos: the half-disc on a diameter of length 2 * `radius`, minus the half-discs on the two
    parts the diameter is split into (`split` is the first part's share). It is centred
    vertically on its bounding box, with the curved side up at rotation 0."""
    big = max(radius, 1e-6)
    share = max(min(split, SPLIT_MAX), SPLIT_MIN)
    small_a, small_b = share * big, (1.0 - share) * big          # the small radii (they sum to `big`)
    shift = -0.5 * big
    rot = radians(rotation)
    cs, sn = cos(rot), sin(rot)
    arcs = []
    for cx, r in ((0.0, big), (-big + small_a, small_a), (big - small_b, small_b)):
        arcs.append((cx * cs - shift * sn, cx * sn + shift * cs, r, rot, pi))
    return arc_loop_bound(SHAPE_ARBELOS, arcs, 1.05 * big)


def chain_arcs_xy(bound, per_arc=60):
    """(x, y) arrays tracing the boundary of an arc-loops shape: each arc sampled, then the
    arcs joined end to end (an arc is turned round when its far end is the nearer one)."""
    pieces = []
    for k in range(int(bound[1])):
        cx, cy, r, lo, span = bound[3 + ARC_FIELDS * k:3 + ARC_FIELDS * (k + 1)]
        ang = lo + span * np.linspace(0.0, 1.0, 8 + int(per_arc * span / (2.0 * pi)))
        pieces.append(np.column_stack((cx + r * np.cos(ang), cy + r * np.sin(ang))))
    if not pieces:
        return np.zeros(2), np.zeros(2)
    path = pieces.pop(0)
    while pieces:
        end = path[-1]
        gap = [min(np.hypot(*(p[0] - end)), np.hypot(*(p[-1] - end))) for p in pieces]
        nxt = pieces.pop(int(np.argmin(gap)))
        if np.hypot(*(nxt[-1] - end)) < np.hypot(*(nxt[0] - end)):
            nxt = nxt[::-1]
        path = np.vstack((path, nxt))
    path = np.vstack((path, path[:1]))
    return path[:, 0], path[:, 1]


def is_robustly_interior(shape, bound, x, y, margin):
    """Is (x, y) inside the shape with some room to spare - not just inside,
    but with a small disk around it also entirely inside? A plain
    inside_boundary() check can pass for a corner (like a sector's apex) or a
    point right on an edge, which is useless as a scaling centre or a
    ray-casting origin; this rejects those."""
    if not inside_boundary(x, y, shape, bound):
        return False
    for ang in (0.0, 0.25 * pi, 0.5 * pi, 0.75 * pi, pi, 1.25 * pi, 1.5 * pi, 1.75 * pi):
        if not inside_boundary(x + margin * cos(ang), y + margin * sin(ang), shape, bound):
            return False
    return True


def find_interior_point(shape, bound, extent=2.2):
    """Some point robustly inside the shape (not a corner or a point right on
    the boundary) - a spiral search using the same inside_boundary test the
    simulation itself uses, so it's correct for any of the 8 shapes without
    needing a bespoke formula per family. Used as a ray-casting origin
    (sample_boundary_xy), which needs genuine breathing room to work correctly."""
    margin = max(0.02, extent * 0.02)
    if is_robustly_interior(shape, bound, 0.0, 0.0, margin):
        return 0.0, 0.0
    for r in np.linspace(0.02, extent, 60):
        for ang in np.linspace(0.0, 2.0 * pi, 24, endpoint=False):
            x, y = r * cos(ang), r * sin(ang)
            if is_robustly_interior(shape, bound, x, y, margin):
                return x, y
    # Fall back to a plain (non-robust) inside test, so this always returns
    # something rather than nothing, for an unusually thin/small shape.
    if inside_boundary(0.0, 0.0, shape, bound):
        return 0.0, 0.0
    for r in np.linspace(0.02, extent, 60):
        for ang in np.linspace(0.0, 2.0 * pi, 24, endpoint=False):
            x, y = r * cos(ang), r * sin(ang)
            if inside_boundary(x, y, shape, bound):
                return x, y
    return 0.0, 0.0                            # shouldn't happen for a valid shape


def sample_boundary_xy(shape, bound, n=181):
    """n (x, y) points tracing the shape's boundary, found by casting rays
    from an interior point at evenly-spaced angles and using boundary_hit
    (already correct for all 8 shapes) to find where each one exits."""
    ix, iy = find_interior_point(shape, bound)
    xs = np.empty(n)
    ys = np.empty(n)
    for k in range(n):
        ang = 2.0 * pi * k / (n - 1)
        ux, uy = cos(ang), sin(ang)
        t, _nx, _ny = boundary_hit(ix, iy, ux, uy, shape, bound)
        xs[k] = ix + t * ux
        ys[k] = iy + t * uy
    return xs, ys


# --------------------------------------------------------------------------- #
# Ray metrics (each is scaled to 0..1 so it can drive a color component)
# --------------------------------------------------------------------------- #

# Metrics that compare two sample points (the others ignore the point choices).
# They are evaluated at coloring time from the stored points; the copies stored
# in the metrics array are for the end of the ray and feed the preview list.
POINT_STATS = (ST_HDIST, ST_VDIST, ST_MANHATTAN, ST_EUCLIDEAN)
OTHER_STATS = tuple(s for s in range(N_STATS) if s not in POINT_STATS)

STAT_LABELS = [     # shown in the color-mapping table
    "Horizontal distance",
    "Vertical distance",
    "Manhattan distance",
    "Euclidean distance",
    "Bounces / iteration limit",
    "Ray-bounces / bounces",
    "Boundary-bounces / bounces",
    "Loop area / same-perimeter circle",
    "Mean bounce angle / 180\u00b0",
    "Total ray length (arctan)",
    "Last segment direction",
    "Termination reason",
    "Last segment / uncut length",
]
STAT_SHORT = [      # compact names for the ray-preview panel (end-point values)
    "H dist", "V dist", "Manhattan", "Euclidean", "Bounces/limit", "Ray hits/bounces",
    "Boundary hits/bounces", "Loop area/circle", "Angle sum/(180\u00b7n)",
    "Length (arctan)", "Last seg direction", "Termination reason",
    "Last seg / uncut",
]
STAT_DESCRIPTIONS = [
    "horizontal distance between point 1 and point 2 (|x1 - x2|), divided by "
    "2 and clipped to 0..1.",
    "vertical distance between point 1 and point 2 (|y1 - y2|), divided by 2 "
    "and clipped to 0..1.",
    "Manhattan (taxicab) distance between point 1 and point 2 "
    "(|x1-x2| + |y1-y2|), divided by 2 and clipped to 0..1.",
    "straight-line (Euclidean) distance between point 1 and point 2, divided "
    "by 2 and clipped to 0..1.",
    "number of bounces divided by the iteration limit.",
    "bounces off the ray's own earlier path, divided by total bounces "
    "(0 if there were none).",
    "bounces off the edge of a shape (a reflecting edge), divided by total bounces "
    "(0 if there were none).",
    "area of the loop formed when the ray first strikes its own path, "
    "divided by the area of a circle with the same perimeter as the loop "
    "(1 = a perfect circle; smaller = more elongated or jagged; 0 if it "
    "never touches itself).",
    "the average deflection angle over all bounces (the angle in degrees "
    "between incoming and outgoing directions), divided by 180 "
    "(i.e. angle sum / (180 \u00d7 bounces)).",
    "total path length, squashed into 0..1 with (2/\u03c0)\u00b7arctan(length/5).",
    "direction of the final segment, as an angle from -180\u00b0 to 180\u00b0 "
    "(counterclockwise from pointing right) mapped to 0..1: 0.5 = right, "
    "0.75 = up, 0.25 = down, 0 or 1 = left.",
    "why the ray stopped: 0 = iteration limit, 0.2 = confined to less than "
    "a pixel, 0.4 = distance limit, 0.6 = reached an edge set to "
    "\"terminate\", 0.8 = hit a reflective surface perfectly perpendicularly, "
    "1 = escaped (nothing left to hit).",
    "length of the last segment divided by the length it would have had "
    "without the distance limit (1 unless the distance limit cut it short).",
]

# Constant "metrics" a channel can be given instead of a real one (listed among
# the "Other" metrics)
NONE_ZERO = -1
NONE_ONE = -2
CHAN_BUDDHA = -3          # self.mapping[c] sentinel: channel c uses Buddha
                         # coloration, not a Query stat at all

# Dropdown contents for the coloration table
KIND_POINT = 0         # the channel uses a point-comparison metric (+ two points)
KIND_OTHER = 1         # the channel uses any other metric
KIND_LABELS = ["Point comparison", "Other"]
POINT_METRIC_LABELS = [STAT_LABELS[s] for s in POINT_STATS]
OTHER_LABELS = [STAT_LABELS[s] for s in OTHER_STATS]

# Each channel's coloration source: "Query" (the system above - a metric is
# evaluated once per ray, for the pixel it was emitted from), "Buddha" (named
# after the Buddhabrot: every ray's characteristic points are scattered into a
# visit-count histogram, and a pixel's brightness comes from how many times
# ITS OWN location was landed on by any ray's points - see
# BUDDHA_POINT_CATALOG below), or "Constant" (always the same value, taking no
# part in the simulation at all).
CHAN_SRC_QUERY = 0
CHAN_SRC_BUDDHA = 1
CHAN_SRC_CONSTANT = 2
CHAN_SRC_LABELS = ["Query", "Buddha", "Constant"]

# "Constant" source: a channel's value never depends on the simulation at
# all. Stored in the same slot other_metric[c] already occupies (as
# NONE_ZERO/NONE_ONE) rather than a separate field, since only one of the
# two ever applies to a given channel at once.
CHAN_CONSTANT_LABELS = ["Always 0", "Always 1"]


def constant_label_to_stat(label):
    return NONE_ZERO if label == CHAN_CONSTANT_LABELS[0] else NONE_ONE


def constant_stat_to_label(stat):
    return CHAN_CONSTANT_LABELS[0] if stat == NONE_ZERO else CHAN_CONSTANT_LABELS[1]


def other_label_to_stat(label):
    """'Other' dropdown text -> metric index."""
    return OTHER_STATS[OTHER_LABELS.index(label)]


def other_stat_to_label(stat):
    return STAT_LABELS[stat]


def point_comparison_metric(stat, p1x, p1y, p2x, p2y):
    """Value (0..1) of a point-comparison metric between sample points 1 and
    2. Works on scalars or arrays."""
    dx = p1x - p2x
    dy = p1y - p2y
    if stat == ST_HDIST:
        v = np.abs(dx) * 0.5
    elif stat == ST_VDIST:
        v = np.abs(dy) * 0.5
    elif stat == ST_MANHATTAN:
        v = (np.abs(dx) + np.abs(dy)) * 0.5
    else:                                           # ST_EUCLIDEAN
        v = np.hypot(dx, dy) * 0.5
    return np.clip(v, 0.0, 1.0)


# Two fixed reference points, unrelated to any ray - always the same regardless of
# where the ray went. Not part of the simulation's own N_POINTS sample points (they're
# never written into the points array the simulation kernels fill), so any code
# reading a chosen point's coordinates must check CONSTANT_POINT_COORDS first.
POINT_ORIGIN = 13             # the fixed point (0, 0)
POINT_TOP_LEFT_CORNER = 14    # the fixed point (-1, 1): the top-left corner of the
                              # [-1, 1] square the simulation runs in
N_POINT_CHOICES = 15          # total selectable points (13 real + 2 fixed)
CONSTANT_POINT_COORDS = {POINT_ORIGIN: (0.0, 0.0), POINT_TOP_LEFT_CORNER: (-1.0, 1.0)}

POINT_LABELS = [
    "Start point",
    "End point",
    "All bounces mean",
    "Boundary bounces mean",
    "Self bounces mean",
    "Uppermost bounce",
    "Lowermost bounce",
    "Rightmost bounce",
    "Leftmost bounce",
    "Uppermost self bounce",
    "Lowermost self bounce",
    "Rightmost self bounce",
    "Leftmost self bounce",
    "Origin",
    "Top left corner",
]
POINT_DESCRIPTIONS = [
    "the ray's starting point (the pixel it was emitted from).",
    "the last point of the ray (if the distance limit ended it part-way "
    "along a segment, the cut point).",
    "the average of every point where the ray bounced, of any kind.",
    "the average of the points where the ray bounced off the edge of a "
    "shape (a reflecting edge).",
    "the average of the points where the ray bounced off its own earlier "
    "path.",
    "the bounce point with the largest vertical coordinate.",
    "the bounce point with the smallest vertical coordinate.",
    "the bounce point with the largest horizontal coordinate.",
    "the bounce point with the smallest horizontal coordinate.",
    "the bounce point with the largest vertical coordinate, among only "
    "the bounces off the ray's own earlier path.",
    "the bounce point with the smallest vertical coordinate, among only "
    "the bounces off the ray's own earlier path.",
    "the bounce point with the largest horizontal coordinate, among only "
    "the bounces off the ray's own earlier path.",
    "the bounce point with the smallest horizontal coordinate, among only "
    "the bounces off the ray's own earlier path.",
    "the fixed point (0, 0) - unrelated to the ray, always the same.",
    "the fixed point (-1, 1), the top-left corner of the simulation's "
    "[-1, 1] square - unrelated to the ray, always the same.",
]
POINT_NOTE = (" (Start and end always exist, and Origin/Top left corner don't depend "
             "on the ray at all; the rest fall back to the end point if the ray "
             "never bounced that way.)")
POINT_SHORT = ["Start", "End", "AllMean", "BoundMean", "SelfMean",
              "Top", "Bottom", "Right", "Left",
              "SelfTop", "SelfBottom", "SelfRight", "SelfLeft",
              "Origin", "TopLeft"]

# The order the two point dropdowns list the choices in: the two fixed points
# (which don't depend on the ray at all) first, then the points picked out of
# the ray's path. Only the display order - a point's id (its index in
# POINT_LABELS, and into the per-ray sample array for the real ones) never
# changes, and choosing by label still maps back through POINT_LABELS.index().
POINT_MENU_ORDER = [POINT_ORIGIN, POINT_TOP_LEFT_CORNER] + list(range(N_POINTS))
POINT_MENU_LABELS = [POINT_LABELS[i] for i in POINT_MENU_ORDER]


def extract_point_xy(points_arr, point_idx, computed, n):
    """(x, y) arrays of length n holding the chosen point's coordinates for
    every computed pixel: read from `points_arr` (an (H, W, N_POINTS, 2)
    array) for a real simulation point, or a constant-filled array for
    POINT_ORIGIN/POINT_TOP_LEFT_CORNER, which was never written into
    `points_arr` at all."""
    if point_idx in CONSTANT_POINT_COORDS:
        cx, cy = CONSTANT_POINT_COORDS[point_idx]
        return np.full(n, cx), np.full(n, cy)
    xy = points_arr[..., point_idx, :][computed].astype(np.float64)
    return xy[:, 0], xy[:, 1]


def resolve_point_xy(sp, point_idx):
    """(x, y) for a single ray's chosen point: read from `sp` (that ray's
    own (N_POINTS, 2) sample points) for a real simulation point, or one of
    the two fixed constant points."""
    if point_idx in CONSTANT_POINT_COORDS:
        return CONSTANT_POINT_COORDS[point_idx]
    return float(sp[point_idx, 0]), float(sp[point_idx, 1])


# Buddha coloration: the points a channel can choose to scatter into its
# visit-count histogram (any number of them, at once). Most are "simple" -
# one point per ray, read straight from the already-computed sample points,
# optionally only for rays that ended with a specific termination reason -
# but the last four are "multi": every one of several points per ray (found
# by re-tracing, since the per-pixel summary doesn't keep the whole path).
BUDDHA_SIMPLE = 0
BUDDHA_MULTI = 1
BUDDHA_MULTI_ALL_SELF = "all_self"           # every bounce off the ray's own path
BUDDHA_MULTI_ALL_MID = "all_mid"             # every segment's midpoint
BUDDHA_MULTI_PROG_BOUNCE = "prog_bounce"     # progressive means of bounce points
BUDDHA_MULTI_PROG_MID = "prog_mid"           # progressive means of segment midpoints

# (label, style, point index (BUDDHA_SIMPLE) or multi-kind (BUDDHA_MULTI),
#  required termination reason (BUDDHA_SIMPLE only) or None for no filter)
BUDDHA_POINT_CATALOG = [
    ("All bounces mean", BUDDHA_SIMPLE, POINT_ALL_MEAN, None),
    ("Self bounces mean", BUDDHA_SIMPLE, POINT_SELF_MEAN, None),
    ("Boundary bounces mean", BUDDHA_SIMPLE, POINT_BOUNDARY_MEAN, None),
    ("Termination point", BUDDHA_SIMPLE, POINT_END, None),
    ("Termination point (distance limit)", BUDDHA_SIMPLE, POINT_END, TERM_DIST),
    ("Termination point (perpendicular collision)", BUDDHA_SIMPLE, POINT_END,
    TERM_PERPENDICULAR),
    ("Termination point (confined)", BUDDHA_SIMPLE, POINT_END, TERM_CONFINED),
    ("All self bounces", BUDDHA_MULTI, BUDDHA_MULTI_ALL_SELF, None),
    ("All segment midpoints", BUDDHA_MULTI, BUDDHA_MULTI_ALL_MID, None),
    ("Progressive means of bounce points", BUDDHA_MULTI, BUDDHA_MULTI_PROG_BOUNCE, None),
    ("Progressive means of segment midpoints", BUDDHA_MULTI, BUDDHA_MULTI_PROG_MID, None),
]
N_BUDDHA_POINTS = len(BUDDHA_POINT_CATALOG)
BUDDHA_POINT_LABELS = [c[0] for c in BUDDHA_POINT_CATALOG]
BUDDHA_POINT_SHORT = ["AllMean", "SelfMean", "BoundMean", "Term", "Term(dist)", "Term(perp)",
                      "Term(confined)", "AllSelf", "AllMid", "ProgBounce", "ProgMid"]
DEFAULT_BUDDHA_POINTS = (True,) + (False,) * (N_BUDDHA_POINTS - 1)   # "All bounces mean" only

# Buddha coloration: how a pixel's (unbounded, non-negative integer) visit
# count maps down to the 0..255 a color channel needs.
BUDDHA_COMPRESS_ARCTAN = 0
BUDDHA_COMPRESS_MODULUS = 1
BUDDHA_COMPRESSION_LABELS = ["Arctangent", "Modulus"]
BUDDHA_ARCTAN_SCALE = 8.0   # count=8 maps to the arctan curve's midpoint (0.5);
                           # a reasonable default giving visible gradation
                           # across small-to-moderate visit counts

# Buddha coloration: an optional final brightness boost, applied (and
# clamped back to 0..1) after compression - useful since compressed values
# often cluster near the low end for sparse histograms.
BUDDHA_BRIGHTNESS_OPTIONS = ["1x", "2x", "4x", "8x", "16x"]
BUDDHA_BRIGHTNESS_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_BUDDHA_BRIGHTNESS = 0    # index into the lists above: "1x", no change


def buddha_compress(count, method, brightness_mult=1.0):
    """A visit count (a non-negative int, or an array of them) -> 0..1."""
    if method == BUDDHA_COMPRESS_ARCTAN:
        v = (2.0 / pi) * np.arctan(count / BUDDHA_ARCTAN_SCALE)
    else:
        v = (np.asarray(count) % 256) / 255.0          # BUDDHA_COMPRESS_MODULUS
    if brightness_mult != 1.0:
        v = np.clip(v * brightness_mult, 0.0, 1.0)
    return v


DIFF_PRESENCE_CHOICES = ("present", "absent", "present or absent")


def setting_passes_presence_filter(in_keyframe, presence):
    """Whether a setting shows in the keyframe settings list under the "Show"
    filter. `in_keyframe`: the selected keyframe itself defines it; `presence` is
    a value from DIFF_PRESENCE_CHOICES."""
    if presence == "present":
        return in_keyframe
    if presence == "absent":
        return not in_keyframe
    return True


def diff_section_header_text(title, expanded, n_set, n_total, n_modified=0):
    """A section header of the keyframe settings list: an arrow showing whether
    it is expanded, its title, and how many of its settings the selected
    keyframe itself defines (so a collapsed section still says something)."""
    arrow = "\u25bc" if expanded else "\u25b6"
    modified = f", {n_modified} modified" if n_modified else ""
    return f"{arrow} {title}  ({n_set} of {n_total} set{modified})"


def short_value_text(key, val):
    """A compact display of one setting's value, for the Load/Save button
    text in the keyframe settings list."""
    if key.startswith("obst:"):
        field = key.split(":", 2)[2]
        if field in ("in_effect", "out_effect"):
            return OBSTACLE_EFFECT_NAMES[val]
        if field == "fire_inside":
            return "On" if val else "Off"
        fmt = OBST_FIELD_FMT.get(field)
        if fmt == "size":
            return f"{val:.2f}"
        if fmt == "offset":
            return f"{val:+.2f}"
        if fmt == "angle":
            return f"{val:.1f}\u00b0"
        return str(val)                          # "int" fields, and anything else
    if key.startswith("view:"):
        field = key.split(":", 2)[2]
        if field == "mode":
            return val
        # field is "chan0"/"chan1"/"chan2"
        if val.chan_src == CHAN_SRC_BUDDHA:
            chosen = [BUDDHA_POINT_LABELS[i].split()[0]
                      for i, on in enumerate(val.buddha_points) if on]
            comp = BUDDHA_COMPRESSION_LABELS[val.buddha_compression]
            bright = BUDDHA_BRIGHTNESS_OPTIONS[val.buddha_brightness]
            return f"Buddha:{'+'.join(chosen) if chosen else 'none'}/{comp}/{bright}"
        if val.chan_src == CHAN_SRC_CONSTANT:
            return f"Const:{'0' if val.constant == NONE_ZERO else '1'}"
        if val.kind == KIND_POINT:
            return f"{STAT_SHORT[val.stat]}@{POINT_SHORT[val.point1]}/{POINT_SHORT[val.point2]}"
        if val.stat in (NONE_ZERO, NONE_ONE):
            raise ValueError(
                f"channel value has stat={val.stat} (a Constant-only sentinel) but "
                f"chan_src is {val.chan_src}, not CHAN_SRC_CONSTANT")
        return STAT_SHORT[val.stat]
    if key in ("iter", "dist", "seed", "confine_window"):
        return str(val)
    if key == "confine_method":
        return CONFINE_METHOD_NAMES[val]
    if key == "confine_threshold":
        return f"{val:.6g}"
    if key == "emit_offset":
        return f"{val:+.1f}\u00b0"
    if key == "emit_mode":
        return EMIT_MODE_NAMES[val]
    if key == "emit_target_theta":
        return f"{val:.6g}\u00b0"
    if key in ("emit_target_r", "emit_target_cx", "emit_target_cy"):
        return f"{val:.6g}"
    if key in ("emit_jitter", "bnd_offset", "bnd_jitter", "self_offset", "self_jitter"):
        return f"\u00b1{val:.1f}\u00b0" if "jitter" in key else f"{val:+.1f}\u00b0"
    if key in ("bnd_mult", "self_mult", "seg_mult"):
        return f"\u00d7{val:.2f}"
    if key == "fuzz_on":
        return "On" if val else "Off"
    if key == "outline":
        return "On" if val else "Off"
    return str(val)

# Defaults per channel. Each channel remembers a point-comparison metric (plus
# the two points it compares) and an "other" metric, so switching its Type
# back and forth restores what was chosen.
DEFAULT_KINDS = (KIND_POINT, KIND_POINT, KIND_OTHER)
DEFAULT_POINT_METRICS = (ST_HDIST, ST_VDIST, ST_MANHATTAN)
DEFAULT_OTHER_METRICS = (ST_BOUNCES, ST_BOUNCES, ST_BOUNCES)
DEFAULT_POINT1S = (POINT_START, POINT_START, POINT_START)  # point 1 per channel
DEFAULT_POINT2S = (POINT_END, POINT_END, POINT_END)        # point 2 per channel
DEFAULT_CHAN_SRCS = (CHAN_SRC_QUERY, CHAN_SRC_QUERY, CHAN_SRC_QUERY)   # per channel
DEFAULT_CONSTANT_CHOICES = (NONE_ZERO, NONE_ZERO, NONE_ZERO)          # per channel
DEFAULT_BUDDHA_POINTS_PER_CHAN = (DEFAULT_BUDDHA_POINTS,) * 3
DEFAULT_BUDDHA_COMPRESSIONS = (BUDDHA_COMPRESS_ARCTAN,) * 3
DEFAULT_BUDDHA_BRIGHTNESSES = (DEFAULT_BUDDHA_BRIGHTNESS,) * 3
DEFAULT_MODE = "RGB"

# Color models: channel names (and label colors) for the three table rows
MODES = {
    "RGB": [("Red", "#c62828"), ("Green", "#2e7d32"), ("Blue", "#1565c0")],
    "HSV": [("Hue", "#6a1b9a"), ("Saturation", "#00838f"), ("Value", "#616161")],
}
MODE_NAMES = list(MODES)


# ----------------------------------------------------------------------- #
# Views: showing more than 3 channels' worth of information at once, by
# rendering the same simulation multiple times with different colorations
# and placing the results side by side in an exported image or video. Each
# view is exactly what the Coloration tab already controls for a single
# image - a color model plus up to 3 channels - bundled as a ViewConfig.
#
# self.views is one uniform list of ViewConfig objects. The top-left preview
# shows self.live_view() - whichever view is currently selected
# (self.selected_view_id) in the Coloration tab; selecting a view for
# editing is what makes it the preview too, there's no separate step. Every
# view uses the same dynamic "view:<id>:..." keyframe keys, mirroring
# Shape/self.obstacles and "obst:<id>:<field>" exactly, since the two
# problems are the same shape: a dynamic list of same-structure things,
# each with its own independently keyframeable fields, where the SET of
# things is fixed for a given render but every field of every one of them
# can still be held or animated. Deleting the selected (live) view falls
# back to self.views[0] - no promotion step and no keyframe-key renaming is
# ever needed, because no view's keyframe data is tied to its list position.
class ViewConfig:
    """One view's live state: a color model plus up to 3 channels - exactly
    what the Coloration tab controls. Position in self.views has no special
    meaning; which one the top-left preview shows is whichever is selected
    (see live_view/select_view), not necessarily self.views[0]."""
    __slots__ = ("id", "mode", "chan_src", "kinds", "point_metric", "other_metric",
                "point1_choice", "point2_choice", "buddha_points", "buddha_compression",
                "buddha_brightness", "constant_choice")

    def __init__(self, id_):
        self.id = id_
        self.mode = DEFAULT_MODE
        self.chan_src = list(DEFAULT_CHAN_SRCS)
        self.kinds = list(DEFAULT_KINDS)
        self.point_metric = list(DEFAULT_POINT_METRICS)
        self.other_metric = list(DEFAULT_OTHER_METRICS)
        self.point1_choice = list(DEFAULT_POINT1S)
        self.point2_choice = list(DEFAULT_POINT2S)
        self.buddha_points = list(DEFAULT_BUDDHA_POINTS_PER_CHAN)
        self.buddha_compression = list(DEFAULT_BUDDHA_COMPRESSIONS)
        self.buddha_brightness = list(DEFAULT_BUDDHA_BRIGHTNESSES)
        self.constant_choice = list(DEFAULT_CONSTANT_CHOICES)

    def set_channel(self, c, channel):
        """Make channel `c` what the ChannelSettings record `channel` says (a loaded keyframe
        or project)."""
        self.chan_src[c] = channel.chan_src
        self.kinds[c] = channel.kind
        if channel.kind == KIND_POINT:
            self.point_metric[c] = channel.stat
        else:
            self.other_metric[c] = channel.stat
        self.point1_choice[c] = channel.point1
        self.point2_choice[c] = channel.point2
        self.buddha_points[c] = channel.buddha_points
        self.buddha_compression[c] = channel.buddha_compression
        self.buddha_brightness[c] = channel.buddha_brightness
        self.constant_choice[c] = channel.constant

    def settings(self):
        """This view as the ViewSettings the colouring code takes."""
        return ViewSettings(mode=self.mode, kinds=self.kinds, point_metric=self.point_metric,
                            other_metric=self.other_metric, point1_choice=self.point1_choice,
                            point2_choice=self.point2_choice, chan_src=self.chan_src,
                            buddha_points=self.buddha_points,
                            buddha_compression=self.buddha_compression,
                            buddha_brightness=self.buddha_brightness,
                            constant_choice=self.constant_choice)


@strict_record
class ViewSnapshot:
    """A frozen copy of one ViewConfig's fields (see snapshot_view). Its fields
    mirror ViewConfig.__slots__ (a test keeps them in step); the per-channel lists
    are stored as tuples."""
    id: int
    mode: str
    chan_src: tuple
    kinds: tuple
    point_metric: tuple
    other_metric: tuple
    point1_choice: tuple
    point2_choice: tuple
    buddha_points: tuple
    buddha_compression: tuple
    buddha_brightness: tuple
    constant_choice: tuple


def snapshot_view(v):
    """A plain, immutable copy of one ViewConfig's fields - safe to read from
    a background render thread while the GUI thread might still be editing
    the live ViewConfig object (mirrors Shape.snapshot). ViewConfig's
    per-channel fields are mutable lists (the UI mutates them in place, e.g.
    self.chan_src[c] = ...), so each one is copied here rather than shared -
    a shared reference would let a background thread see a live edit
    mid-render, exactly what a "snapshot" is meant to prevent (the record stores
    the per-channel lists as tuples)."""
    return ViewSnapshot(**{f: getattr(v, f) for f in ViewConfig.__slots__})


@strict_record
class ChannelSettings:
    """One color channel of a view, as a keyframe stores it (the value of a
    "view:<id>:chan<c>" key). `constant` is a field of its own even though it only
    means something when chan_src is CHAN_SRC_CONSTANT - it is never packed into
    `stat`, whose meaning already depends on `kind`, so reading it never requires
    guessing which of several interpretations currently applies."""
    chan_src: int
    kind: int
    stat: int                  # the point metric if kind is KIND_POINT, else the other metric
    point1: int
    point2: int
    buddha_points: tuple
    buddha_compression: int
    buddha_brightness: int
    constant: int


def view_channel_value(v, c):
    """Channel c's current setting for a ViewConfig or ViewSnapshot `v`, as
    stored in a keyframe (a ChannelSettings)."""
    stat = v.point_metric[c] if v.kinds[c] == KIND_POINT else v.other_metric[c]
    return ChannelSettings(
        chan_src=v.chan_src[c], kind=v.kinds[c], stat=stat, point1=v.point1_choice[c],
        point2=v.point2_choice[c], buddha_points=v.buddha_points[c],
        buddha_compression=v.buddha_compression[c], buddha_brightness=v.buddha_brightness[c],
        constant=v.constant_choice[c])


# Every field a view can define in a keyframe: (field, label, is_float) -
# mirrors the shapes' KEYFRAME_FIELDS. None of these are float (a color model or a whole
# channel's configuration is held between keyframes, never interpolated).
VIEW_FIELD_INFO = [("mode", "Color model", False), ("chan0", "Channel 1", False),
                   ("chan1", "Channel 2", False), ("chan2", "Channel 3", False)]


def view_field_value(v, field):
    """The current value of one of a view's VIEW_FIELD_INFO fields, in the
    same shape a keyframe stores it in."""
    if field == "mode":
        return v.mode
    return view_channel_value(v, int(field[-1]))


@strict_record
class ViewSettings:
    """One view's resolved coloration, as colorize_view reads it (per-channel values
    are 3-tuples)."""
    mode: str
    kinds: tuple
    point_metric: tuple
    other_metric: tuple
    point1_choice: tuple
    point2_choice: tuple
    chan_src: tuple
    buddha_points: tuple
    buddha_compression: tuple
    buddha_brightness: tuple
    constant_choice: tuple


def decompose_chans(mode, chans):
    """A color model plus 3 channels' ChannelSettings (as stored in a keyframe, or
    built by view_channel_value) -> a ViewSettings, decomposed into the separate
    per-channel fields render_full_image's coloring step actually reads. Used by
    resolve_views() for every view alike."""
    return ViewSettings(
        mode=mode,
        kinds=tuple(c.kind for c in chans),
        point_metric=tuple(c.stat if c.kind == KIND_POINT else DEFAULT_POINT_METRICS[i]
                           for i, c in enumerate(chans)),
        other_metric=tuple(c.stat if c.kind == KIND_OTHER else DEFAULT_OTHER_METRICS[i]
                           for i, c in enumerate(chans)),
        point1_choice=tuple(c.point1 for c in chans),
        point2_choice=tuple(c.point2 for c in chans),
        chan_src=tuple(c.chan_src for c in chans),
        buddha_points=tuple(c.buddha_points for c in chans),
        buddha_compression=tuple(c.buddha_compression for c in chans),
        buddha_brightness=tuple(c.buddha_brightness for c in chans),
        constant_choice=tuple(c.constant for c in chans))


def resolve_views(view_snapshots, keyframes_sorted, frame_number):
    """tuple of ViewSettings, one per view, in the same order as
    view_snapshots - list/export order, unrelated to which view the
    top-left preview happens to be showing (see live_view/select_view):
    the SET of views is fixed (whichever existed when the render started),
    but each one's color model and channels are independently resolved
    (held, never smoothly interpolated) from the keyframes - falling back
    to that view's own value (at render-start time) wherever no keyframe
    defines it. Mirrors resolve_obstacle_arrays."""
    result = []
    for snap in view_snapshots:
        mode = resolve_value(f"view:{snap.id}:mode", keyframes_sorted, frame_number,
                             snap.mode, False)
        chans = [resolve_value(f"view:{snap.id}:chan{c}", keyframes_sorted, frame_number,
                               view_channel_value(snap, c), False)
                for c in range(3)]
        result.append(decompose_chans(mode, chans))
    return tuple(result)


# --------------------------------------------------------------------------- #
# Projects: saving and loading everything the user has set up
#
# A project file is pretty-printed, versioned JSON (PROJECT_EXTENSION). It holds the
# controls' simulation settings, the decorations, every obstacle, view and keyframe, and
# the Export designer and video panel choices - not the generated image, which Start makes
# again (fuzzing is seeded, so the same settings give the same image).
#
# Every choice in the file is a readable string KEY ("euclidean_distance", "reflect",
# "lune"), never a number. A key is permanent: once a project file can contain it, it is
# never renamed or reused, and a Catalog translates it to the program's own code (an int
# the kernels index arrays with, which may be renumbered freely). So the program is free
# to reorder, insert and renumber its codes; test_the_keys_in_project_files_never_change
# pins the keys. Object ids (obstacles, views, keyframes) stay the whole numbers they are.
# --------------------------------------------------------------------------- #
PROJECT_EXTENSION = ".lightbounce"
PROJECT_FORMAT_NAME = "light-bounces-project"
PROJECT_FORMAT_VERSION = 2
PROJECT_FILE_TYPES = [("Light bounces project", f"*{PROJECT_EXTENSION}")]
PROJECT_KEY_PATTERN = re.compile(r"[a-z0-9_]+")


class ProjectError(Exception):
    """A project that cannot be saved or loaded; the message is shown to the user."""


class Catalog:
    """The choices of one kind as a project file stores them: (key, code) pairs. A key is a
    permanent name (lower-case letters, digits, underscores) and all the file holds; the code
    is what the program uses inside - an int (renumberable) or, for a choice made in a
    control, the text the control holds."""

    def __init__(self, name, entries):
        self.name = name
        self.entries = tuple(entries)
        self.by_key = dict(self.entries)
        self.by_code = {code: key for key, code in self.entries}
        assert len(self.by_key) == len(self.by_code) == len(self.entries), name
        assert all(PROJECT_KEY_PATTERN.fullmatch(key) for key in self.by_key), name

    @property
    def codes(self):
        return list(self.by_code)

    def key(self, code):
        if code not in self.by_code:
            raise ProjectError(f"This {self.name} ({code!r}) cannot be saved in a project.")
        return self.by_code[code]

    def code(self, key, where):
        if not isinstance(key, str) or key not in self.by_key:
            raise ProjectError(f"{where}: {key!r} is not a {self.name} this version of the "
                               "program knows (was the project made by a newer version?).")
        return self.by_key[key]


def _string_catalog(name, keys, labels):
    """A Catalog for a choice made in a control: key -> the text the control holds."""
    assert len(keys) == len(labels), name
    return Catalog(name, zip(keys, labels))


CAT_CHAN_SOURCE = Catalog("channel source", [
    ("query", CHAN_SRC_QUERY), ("buddha", CHAN_SRC_BUDDHA), ("constant", CHAN_SRC_CONSTANT)])
CAT_CHAN_KIND = Catalog("channel kind", [("point_metric", KIND_POINT), ("other_metric", KIND_OTHER)])
CAT_METRIC = Catalog("metric", [
    ("horizontal_distance", ST_HDIST), ("vertical_distance", ST_VDIST),
    ("manhattan_distance", ST_MANHATTAN), ("euclidean_distance", ST_EUCLIDEAN),
    ("bounces", ST_BOUNCES), ("ray_bounce_fraction", ST_RAY_FRAC),
    ("boundary_bounce_fraction", ST_BOUNDARY_FRAC), ("loop_area", ST_LOOP_AREA),
    ("mean_bounce_angle", ST_ANGLES), ("ray_length", ST_LENGTH),
    ("last_segment_direction", ST_LAST_DIR), ("termination_reason", ST_TERMINATION),
    ("last_segment_ratio", ST_LAST_RATIO)])
CAT_POINT = Catalog("point", [
    ("start_point", POINT_START), ("end_point", POINT_END), ("all_bounces_mean", POINT_ALL_MEAN),
    ("boundary_bounces_mean", POINT_BOUNDARY_MEAN), ("self_bounces_mean", POINT_SELF_MEAN),
    ("uppermost_bounce", POINT_TOP), ("lowermost_bounce", POINT_BOTTOM),
    ("rightmost_bounce", POINT_RIGHT), ("leftmost_bounce", POINT_LEFT),
    ("uppermost_self_bounce", POINT_SELF_TOP), ("lowermost_self_bounce", POINT_SELF_BOTTOM),
    ("rightmost_self_bounce", POINT_SELF_RIGHT), ("leftmost_self_bounce", POINT_SELF_LEFT),
    ("origin", POINT_ORIGIN), ("top_left_corner", POINT_TOP_LEFT_CORNER)])
CAT_BUDDHA_POINT = Catalog("Buddha point", [          # the code is the place in BUDDHA_POINT_CATALOG
    ("all_bounces_mean", 0), ("self_bounces_mean", 1), ("boundary_bounces_mean", 2),
    ("termination_point", 3), ("termination_point_distance_limit", 4),
    ("termination_point_perpendicular_collision", 5), ("termination_point_confined", 6),
    ("all_self_bounces", 7), ("all_segment_midpoints", 8),
    ("progressive_means_of_bounce_points", 9), ("progressive_means_of_segment_midpoints", 10)])
CAT_BUDDHA_COMPRESSION = Catalog("Buddha compression", [
    ("arctangent", BUDDHA_COMPRESS_ARCTAN), ("modulus", BUDDHA_COMPRESS_MODULUS)])
CAT_BUDDHA_BRIGHTNESS = Catalog("Buddha brightness", [      # the code is the place in the options
    ("1x", 0), ("2x", 1), ("4x", 2), ("8x", 3), ("16x", 4)])
CAT_CONSTANT = Catalog("constant", [("zero", NONE_ZERO), ("one", NONE_ONE)])
CAT_EMIT_MODE = Catalog("emission mode", [("up", EMIT_UP), ("towards_point", EMIT_TOWARDS)])
CAT_CONFINE_METHOD = Catalog("confinement method", [
    ("box", CONFINE_BOX), ("path_length", CONFINE_PATH)])
CAT_BLUR_TYPE = Catalog("blur type", [
    ("gaussian", BUDDHA_BLUR_GAUSSIAN), ("triangular", BUDDHA_BLUR_TRIANGULAR),
    ("bokeh", BUDDHA_BLUR_BOKEH), ("catadioptric_bokeh", BUDDHA_BLUR_CATADIOPTRIC)])
CAT_EFFECT = Catalog("ray effect", [
    ("reflect", OBSTACLE_EFFECT_REFLECT), ("terminate", OBSTACLE_EFFECT_TERMINATE),
    ("pass_through", OBSTACLE_EFFECT_PASS)])
CAT_SHAPE_KIND = Catalog("obstacle kind", [(cls.KEY, cls) for cls in SHAPE_CLASSES])
# The simulation settings that are choices, by setting key
RAY_CHOICE_CATALOGS = {"emit_mode": CAT_EMIT_MODE, "confine_method": CAT_CONFINE_METHOD,
                       "buddha_blur_type": CAT_BLUR_TYPE}
# Choices made in the Export designer and video panel
CAT_COLOR_INFO = _string_catalog("colour info location", ["none", "overlay", "bottom_margin"],
                                 COLOR_INFO_LOCATIONS)
CAT_FIGURE_INFO = _string_catalog("figure info location", ["none", "left_margin"],
                                  FIGURE_INFO_LOCATIONS)
CAT_IMAGE_SOURCE = _string_catalog("image to export", ["current_image", "new_image"],
                                   SAVE_SOURCE_OPTIONS)
CAT_IMAGE_FORMAT = _string_catalog("image format", ["jxl", "png", "qoi"], SAVE_NEW_FORMATS)
CAT_JXL_MODE = _string_catalog("JPEG XL mode", ["lossless", "lossy"], JXL_MODES)
CAT_VIDEO_FORMAT = _string_catalog("video format", ["mp4_h264", "gif", "apng", "mkv_ffv1", "jxl"],
                                   VIDEO_FORMATS)
CAT_BITRATE_MODE = _string_catalog(
    "bitrate mode", ["constant_quality", "target_bitrate", "variable_bitrate"],
    VIDEO_BITRATE_MODES)
CAT_DOWNSAMPLE_FACTOR = _string_catalog("downsample factor", ["none", "2x2", "4x4", "8x8"],
                                        DOWNSAMPLE_FACTORS)
CAT_DOWNSAMPLE_METHOD = _string_catalog(
    "downsample method", ["mean", "median", "sum", "max", "ffmpeg"], DOWNSAMPLE_METHODS)

# The Export designer and video panel choices a project keeps:
#   (key in the file, the App attribute holding the control - a Tk variable or slider with
#    .get()/.set(), possibly a dotted path - what it holds, its Catalog if it is a choice)
# "choice" holds one of the catalog's texts, "text" any string, "bool", "number" an int or float.
EXPORT_OPTIONS = [
    ("include_all_views", "include_all_views_var", "bool", None),
    ("color_info_location", "color_info_location_var", "choice", CAT_COLOR_INFO),
    ("figure_info_location", "figure_info_location_var", "choice", CAT_FIGURE_INFO),
    ("image_source", "save_source_var", "choice", CAT_IMAGE_SOURCE),
    ("image_format", "save_format_var", "choice", CAT_IMAGE_FORMAT),
    ("image_jxl_mode", "save_jxl_mode_var", "choice", CAT_JXL_MODE),
    ("image_jxl_quality", "save_jxl_quality_scale", "number", None),
    ("image_width", "save_new_w_var", "text", None),
    ("image_height", "save_new_h_var", "text", None),
    ("video_fps", "fps_var", "text", None),
    ("video_width", "video_w_var", "text", None),
    ("video_height", "video_h_var", "text", None),
    ("video_skip_first_frame", "skip_first_frame_var", "bool", None),
    ("video_format", "export_format_var", "choice", CAT_VIDEO_FORMAT),
    ("video_bitrate_mode", "bitrate_mode_var", "choice", CAT_BITRATE_MODE),
    ("video_quality", "quality_scale", "number", None),
    ("video_bitrate", "bitrate_var", "text", None),
    ("video_vbr_target", "vbr_target_var", "text", None),
    ("video_vbr_max", "vbr_max_var", "text", None),
    ("video_downsample_factor", "downsample_factor_var", "choice", CAT_DOWNSAMPLE_FACTOR),
    ("video_downsample_method", "downsample_method_var", "choice", CAT_DOWNSAMPLE_METHOD),
    ("video_jxl_mode", "video_jxl_controls.mode_var", "choice", CAT_JXL_MODE),
    ("video_jxl_quality", "video_jxl_controls.scale", "number", None),
]


@strict_record
class Project:
    """Everything a project file holds, as plain values (the obstacles, views and keyframes
    are independent frozen copies). Made by App.capture_project or project_from_dict."""
    ray: RaySettings                    # the simulation controls
    decorations: dict                   # decoration key -> bool
    obstacles: tuple                    # frozen Shapes, in list order
    views: tuple                        # ViewSnapshots, in list order
    keyframes: tuple                    # Keyframes, in list order
    export: dict                        # EXPORT_OPTIONS key -> the control's value
    obstacle_counter: int               # the next new obstacle's / view's / keyframe's id
    view_counter: int
    keyframe_counter: int
    selected_obstacle_id: int           # -1 for none
    selected_view_id: int


def view_config_from_snapshot(snapshot):
    """A live ViewConfig with the contents of a ViewSnapshot."""
    view = ViewConfig(snapshot.id)
    view.mode = snapshot.mode
    for c in range(3):
        view.set_channel(c, view_channel_value(snapshot, c))
    return view


def project_signature(project):
    """What a project's content amounts to, comparable with == : exactly what its file would say,
    apart from the program version, the id counters and which obstacle or view is selected. Used
    to tell whether there are unsaved changes (so a setting a file does not keep, like the
    Buddha compression of a channel that is not a Buddha channel, never counts as one)."""
    content = project_to_dict(project)
    for name in ("program_version", "counters", "selected"):
        del content[name]
    return json.dumps(content, sort_keys=True, default=_json_default)


def _json_default(value):
    """json.dumps's fallback: numpy numbers become Python ones."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} cannot be saved in a project")


def _channel_to_json(channel):
    """A colour channel as the file holds it: its source and only the settings that source uses -
    a query channel its metric (and the two points, if the metric compares points), a Buddha
    channel its points, compression and brightness, a constant channel its constant."""
    out = {"source": CAT_CHAN_SOURCE.key(channel.chan_src)}
    if channel.chan_src == CHAN_SRC_QUERY:
        out["kind"] = CAT_CHAN_KIND.key(channel.kind)
        out["metric"] = CAT_METRIC.key(channel.stat)
        if channel.kind == KIND_POINT:
            out["point1"] = CAT_POINT.key(channel.point1)
            out["point2"] = CAT_POINT.key(channel.point2)
    elif channel.chan_src == CHAN_SRC_BUDDHA:
        out["buddha_points"] = [CAT_BUDDHA_POINT.key(i)
                                for i, chosen in enumerate(channel.buddha_points) if chosen]
        out["buddha_compression"] = CAT_BUDDHA_COMPRESSION.key(channel.buddha_compression)
        out["buddha_brightness"] = CAT_BUDDHA_BRIGHTNESS.key(channel.buddha_brightness)
    else:
        out["constant"] = CAT_CONSTANT.key(channel.constant)
    return out


def _setting_catalog(key):
    """The Catalog for the values of a simulation or keyframe setting `key` that is a choice."""
    if key in RAY_CHOICE_CATALOGS:
        return RAY_CHOICE_CATALOGS[key]
    if key.startswith("obst:") and key.rsplit(":", 1)[-1] in ("in_effect", "out_effect"):
        return CAT_EFFECT
    return None


def _setting_to_json(key, value):
    if isinstance(value, ChannelSettings):
        return _channel_to_json(value)
    catalog = _setting_catalog(key)
    return catalog.key(value) if catalog is not None else value


def project_to_dict(project):
    """The JSON-ready dict of a Project (the file's contents, see project_from_dict)."""
    export = {}
    for key, _path, _kind, catalog in EXPORT_OPTIONS:
        if key in project.export:
            export[key] = catalog.key(project.export[key]) if catalog else project.export[key]
    return {
        "format": PROJECT_FORMAT_NAME,
        "format_version": PROJECT_FORMAT_VERSION,
        "program_version": FILE_VERSION,
        "simulation": {key: _setting_to_json(key, value)
                       for key, value in ray_settings_to_dict(project.ray).items()},
        "decorations": dict(project.decorations),
        "obstacles": [{"kind": type(o).KEY, "id": o.id,
                       "fields": {name: _setting_to_json(f"obst:{o.id}:{name}", getattr(o, name))
                                  for name in o.FIELD_NAMES}}
                      for o in project.obstacles],
        "views": [{"id": v.id, "mode": v.mode,
                   "channels": [_channel_to_json(view_channel_value(v, c)) for c in range(3)]}
                  for v in project.views],
        "keyframes": [{"id": kf.id, "frame_index": kf.frame_index,
                       "settings": {key: _setting_to_json(key, value)
                                    for key, value in kf.data.items()}}
                      for kf in project.keyframes],
        "export": export,
        "counters": {"obstacle": project.obstacle_counter, "view": project.view_counter,
                     "keyframe": project.keyframe_counter},
        "selected": {"obstacle": (project.selected_obstacle_id
                                  if project.selected_obstacle_id >= 0 else None),
                     "view": project.selected_view_id},
    }


def project_text(project):
    """The text of a project file: pretty-printed JSON (two-space indent, one value per
    line), ending with a newline."""
    try:
        return json.dumps(project_to_dict(project), indent=2, ensure_ascii=False,
                          allow_nan=False, default=_json_default) + "\n"
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"The project could not be saved: {exc}") from exc


def save_project(path, project):
    """Write `project` to `path` (an existing file there is replaced). The whole text is made
    first, so a project that cannot be saved never leaves a half-written file."""
    text = project_text(project)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except OSError as exc:
        raise ProjectError(f"The project could not be written to {path}:\n{exc}") from exc


def load_project(path):
    """The Project in the file at `path`; ProjectError (with a message fit to show) if it
    cannot be read or is not a usable project."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ProjectError(f"The file could not be read:\n{exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectError(f"This is not a valid project file (the JSON is damaged at line "
                           f"{exc.lineno}, column {exc.colno}: {exc.msg}).") from exc
    except UnicodeDecodeError as exc:
        raise ProjectError("This is not a project file (it is not UTF-8 text).") from exc
    return project_from_dict(data)


def _expect(condition, message):
    if not condition:
        raise ProjectError(message)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _channel_from_json(data, where, c):
    """Channel number `c` (0-2) from its JSON. Only the settings its source uses have to be there;
    any other setting that is there is kept, and one that is not takes the channel's default."""
    _expect(isinstance(data, dict), f"{where}: a colour channel must be an object")
    _expect("source" in data, f"{where}: the channel has no 'source'")
    default = view_channel_value(ViewConfig(0), c)
    source = CAT_CHAN_SOURCE.code(data["source"], where)
    values = {"chan_src": source, "kind": default.kind, "stat": default.stat,
              "point1": default.point1, "point2": default.point2,
              "buddha_points": default.buddha_points,
              "buddha_compression": default.buddha_compression,
              "buddha_brightness": default.buddha_brightness, "constant": default.constant}
    for name, field, catalog in (("kind", "kind", CAT_CHAN_KIND), ("metric", "stat", CAT_METRIC),
                                 ("point1", "point1", CAT_POINT), ("point2", "point2", CAT_POINT),
                                 ("buddha_compression", "buddha_compression", CAT_BUDDHA_COMPRESSION),
                                 ("buddha_brightness", "buddha_brightness", CAT_BUDDHA_BRIGHTNESS),
                                 ("constant", "constant", CAT_CONSTANT)):
        if name in data:
            values[field] = catalog.code(data[name], where)
    if "buddha_points" in data:
        chosen = data["buddha_points"]
        _expect(isinstance(chosen, list), f"{where}: buddha_points must be a list")
        codes = {CAT_BUDDHA_POINT.code(key, where) for key in chosen}
        values["buddha_points"] = tuple(code in codes for code in range(N_BUDDHA_POINTS))
    needed = {CHAN_SRC_QUERY: ["kind", "metric"] + (["point1", "point2"]
                                                    if values["kind"] == KIND_POINT else []),
              CHAN_SRC_BUDDHA: ["buddha_points", "buddha_compression", "buddha_brightness"],
              CHAN_SRC_CONSTANT: ["constant"]}[source]
    for name in needed:
        _expect(name in data, f"{where}: the channel has no {name!r}")
    if source == CHAN_SRC_QUERY:
        _expect(values["stat"] in (POINT_STATS if values["kind"] == KIND_POINT else OTHER_STATS),
                f"{where}: the metric {data['metric']!r} does not suit a {data['kind']!r} channel")
    return ChannelSettings(**values)


def _setting_from_json(key, value, where):
    """The value of simulation or keyframe setting `key`, as the program holds it."""
    catalog = _setting_catalog(key)
    if catalog is not None:
        return catalog.code(value, where)
    _expect(isinstance(value, (bool, int, float, str)),
            f"{where}: the value of {key!r} must be a number, true/false or text")
    return value


def _shape_from_json(data, seen_ids):
    _expect(isinstance(data, dict), "an obstacle must be an object")
    kind, shape_id, fields = data.get("kind"), data.get("id"), data.get("fields")
    shape_class = CAT_SHAPE_KIND.code(kind, "an obstacle's kind")
    _expect(isinstance(shape_id, int) and not isinstance(shape_id, bool) and shape_id >= 0,
            f"obstacle {kind}: its id must be a whole number")
    _expect(shape_id not in seen_ids, f"two obstacles have the id {shape_id}")
    seen_ids.add(shape_id)
    _expect(isinstance(fields, dict), f"obstacle #{shape_id}: fields must be an object")
    shape = shape_class(shape_id)
    ranges = {p.field: p for p in shape_class.NUMERIC_PARAMS if p.lo is not None}
    for name, value in fields.items():
        _expect(name in shape_class.FIELD_NAMES,
                f"obstacle #{shape_id} ({kind}) has a field {name!r} that this kind does not have")
        value = _setting_from_json(f"obst:{shape_id}:{name}", value, f"obstacle #{shape_id}")
        try:
            setattr(shape, name, value)
        except TypeError as exc:
            raise ProjectError(f"obstacle #{shape_id} ({kind}): {exc}") from exc
        if name in ranges:
            p = ranges[name]
            _expect(p.lo <= value <= p.hi,
                    f"obstacle #{shape_id} ({kind}): {name} = {value} is outside {p.lo} to {p.hi}")
    return shape.snapshot()


def _view_from_json(data, seen_ids):
    _expect(isinstance(data, dict), "a view must be an object")
    view_id, mode, channels = data.get("id"), data.get("mode"), data.get("channels")
    _expect(isinstance(view_id, int) and not isinstance(view_id, bool) and view_id >= 0,
            "a view's id must be a whole number")
    _expect(view_id not in seen_ids, f"two views have the id {view_id}")
    seen_ids.add(view_id)
    _expect(mode in MODES, f"view {view_id}: unknown colour model {mode!r}")
    _expect(isinstance(channels, list) and len(channels) == 3,
            f"view {view_id}: it needs exactly 3 channels")
    view = ViewConfig(view_id)
    view.mode = mode
    for c, channel in enumerate(channels):
        view.set_channel(c, _channel_from_json(channel, f"view {view_id} channel {c + 1}", c))
    return snapshot_view(view)


def _keyframe_from_json(data, seen_ids):
    _expect(isinstance(data, dict), "a keyframe must be an object")
    kf_id, frame_index, settings = data.get("id"), data.get("frame_index"), data.get("settings")
    _expect(isinstance(kf_id, int) and not isinstance(kf_id, bool) and kf_id >= 0,
            "a keyframe's id must be a whole number")
    _expect(kf_id not in seen_ids, f"two keyframes have the id {kf_id}")
    seen_ids.add(kf_id)
    _expect(isinstance(frame_index, int) and not isinstance(frame_index, bool),
            f"keyframe {kf_id}: its frame index must be a whole number")
    _expect(isinstance(settings, dict), f"keyframe {kf_id}: settings must be an object")
    values = {}
    for key, value in settings.items():
        if key.startswith("view:") and ":chan" in key:
            channel_number = key.rsplit(":chan", 1)[1]
            _expect(channel_number in ("0", "1", "2"), f"keyframe {kf_id}: {key!r} is not a channel")
            values[key] = _channel_from_json(value, f"keyframe {kf_id} {key}", int(channel_number))
        else:
            values[key] = _setting_from_json(key, value, f"keyframe {kf_id}")
    return Keyframe(kf_id, frame_index, values)


def _migrate_project_1_to_2(data):
    """Format 1 stored choices as the program's numbers (and obstacle kinds as class names); format 2
    stores their keys. Converts a parsed format-1 file."""
    import copy
    data = copy.deepcopy(data)
    kind_keys = {cls.__name__: cls.KEY for cls in SHAPE_CLASSES}
    choice = {key: catalog for key, catalog in RAY_CHOICE_CATALOGS.items()}

    def channel(old, where):
        _expect(isinstance(old, dict), f"{where}: a colour channel must be an object")
        points = old.get("buddha_points")
        _expect(isinstance(points, list), f"{where}: buddha_points must be a list")
        padded = tuple(bool(p) for p in points)[:N_BUDDHA_POINTS] + \
            (False,) * max(0, N_BUDDHA_POINTS - len(points))
        return _channel_to_json(ChannelSettings(
            chan_src=old["chan_src"], kind=old["kind"], stat=old["stat"], point1=old["point1"],
            point2=old["point2"], buddha_points=padded,
            buddha_compression=old["buddha_compression"],
            buddha_brightness=old["buddha_brightness"], constant=old["constant"]))

    def setting(key, value, where):
        if key.startswith("view:") and ":chan" in key:
            return channel(value, where)
        catalog = _setting_catalog(key)
        return catalog.key(value) if catalog is not None else value

    simulation = data.get("simulation")
    if isinstance(simulation, dict):
        data["simulation"] = {key: (choice[key].key(value) if key in choice else value)
                              for key, value in simulation.items()}
    for obstacle in data.get("obstacles", []):
        obstacle["kind"] = kind_keys.get(obstacle.get("kind"), obstacle.get("kind"))
        fields = obstacle.get("fields")
        if isinstance(fields, dict):
            for name in ("in_effect", "out_effect"):
                if name in fields:
                    fields[name] = CAT_EFFECT.key(fields[name])
    for view in data.get("views", []):
        view["channels"] = [channel(ch, f"view {view.get('id')} channel {c + 1}")
                            for c, ch in enumerate(view.get("channels", []))]
    for kf in data.get("keyframes", []):
        kf["settings"] = {key: setting(key, value, f"keyframe {kf.get('id')} {key}")
                          for key, value in kf.get("settings", {}).items()}
    export = data.get("export")
    if isinstance(export, dict):
        for key, _path, _kind, catalog in EXPORT_OPTIONS:
            if catalog is not None and export.get(key) in catalog.by_code:
                export[key] = catalog.by_code[export[key]]
    data["format_version"] = 2
    return data


PROJECT_MIGRATIONS = {1: _migrate_project_1_to_2}      # format n -> format n + 1


def project_from_dict(data):
    """A Project from the parsed JSON of a project file. Anything unusable raises ProjectError;
    nothing is partly applied. Tolerant of a file from another version of the program: settings
    it lacks take their defaults and things it has that this version does not know are ignored -
    except a key this version has no code for (an unknown kind of obstacle, metric...), which
    is refused."""
    try:
        return _project_from_dict(data)
    except ProjectError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ProjectError(f"The project file is damaged ({type(exc).__name__}: {exc}).") from exc


def _project_from_dict(data):
    _expect(isinstance(data, dict) and data.get("format") == PROJECT_FORMAT_NAME,
            "This is not a Light bounces project file.")
    version = data.get("format_version")
    _expect(isinstance(version, int) and not isinstance(version, bool) and version >= 1,
            "The project file has no valid format version.")
    _expect(version <= PROJECT_FORMAT_VERSION,
            f"This project was saved by a newer version of the program (project format "
            f"{version}; this version reads up to {PROJECT_FORMAT_VERSION}).")
    while version < PROJECT_FORMAT_VERSION:
        data = PROJECT_MIGRATIONS[version](data)
        version += 1

    file_simulation = {key: value for key, value in (data.get("simulation") or {}).items()
                       if key in RAY_SETTING_SPECS}
    simulation = {**ray_settings_to_dict(DEFAULT_RAY_SETTINGS),
                  **{key: _setting_from_json(key, value, "the simulation settings")
                     for key, value in file_simulation.items()}}
    ray = ray_settings_from_dict(simulation)

    decorations = {key: value for key, value in (data.get("decorations") or {}).items()
                   if isinstance(value, bool)}
    export = {}
    for key, _path, _kind, catalog in EXPORT_OPTIONS:
        value = (data.get("export") or {}).get(key)
        if catalog is not None:                       # an unknown choice is dropped (the control
            if isinstance(value, str) and value in catalog.by_key:     # keeps its own value)
                export[key] = catalog.by_key[value]
        elif isinstance(value, (bool, int, float, str)):
            export[key] = value

    obstacle_ids, view_ids, keyframe_ids = set(), set(), set()
    obstacles = tuple(_shape_from_json(o, obstacle_ids) for o in data.get("obstacles", []))
    views = tuple(_view_from_json(v, view_ids) for v in data.get("views", []))
    _expect(views, "The project has no views.")
    keyframes = tuple(_keyframe_from_json(kf, keyframe_ids) for kf in data.get("keyframes", []))

    counters = data.get("counters") or {}
    selected = data.get("selected") or {}
    selected_obstacle = selected.get("obstacle")
    selected_view = selected.get("view")
    return Project(
        ray=ray, decorations=decorations, obstacles=obstacles, views=views,
        keyframes=keyframes, export=export,
        obstacle_counter=max(counters.get("obstacle", 0), max(obstacle_ids, default=-1) + 1),
        view_counter=max(counters.get("view", 0), max(view_ids) + 1),
        keyframe_counter=max(counters.get("keyframe", 0), max(keyframe_ids, default=-1) + 1),
        selected_obstacle_id=selected_obstacle if selected_obstacle in obstacle_ids else -1,
        selected_view_id=selected_view if selected_view in view_ids else views[0].id)


def run_self_checks():
    """Consistency checks between tables that must independently agree (each
    setting/field/termination-reason has to be listed correctly in several
    separate places - for example SETTING_INFO vs DEFAULT_SETTING_VALUES vs
    SETTING_FLOAT_KEYS)
    but have nothing else enforcing that agreement. Run once at import time,
    so a table that's missing an entry (as happened twice already this
    project - TERM_PERPENDICULAR once missing from a color dict and once
    from an outcome-text dict, and "dist" once missing from
    SETTING_FLOAT_KEYS) fails loudly here, rather than silently producing a
    KeyError or a wrong value the first time that particular path is hit."""
    for name, mapping in [("TERM_STAT_VALUE", TERM_STAT_VALUE),
                          ("TERM_END_COLOR", TERM_END_COLOR),
                          ("TERM_OUTCOME_TEXT", TERM_OUTCOME_TEXT)]:
        missing = [t for t in TERM_CODES if t not in mapping]
        if missing:
            raise AssertionError(f"{name} is missing termination code(s): {missing}")

    info_keys = {key for key, _tab, _label in SETTING_INFO}
    default_keys = set(DEFAULT_SETTING_VALUES.keys())
    if info_keys != default_keys:
        raise AssertionError(
            "SETTING_INFO and DEFAULT_SETTING_VALUES disagree - "
            f"only in SETTING_INFO: {info_keys - default_keys or None}; "
            f"only in DEFAULT_SETTING_VALUES: {default_keys - info_keys or None}")
    unknown_float_keys = SETTING_FLOAT_KEYS - info_keys
    if unknown_float_keys:
        raise AssertionError(f"SETTING_FLOAT_KEYS references unknown setting(s): "
                             f"{unknown_float_keys}")

    names_seen = {}
    for index, cls in enumerate(SHAPE_CLASSES):
        if cls.KIND != index:
            raise AssertionError(f"{cls.__name__}.KIND is {cls.KIND} but it is item {index} "
                                 "of SHAPE_CLASSES")
        slots = {s for klass in cls.__mro__ for s in getattr(klass, "__slots__", ())}
        slots -= {"id", "_frozen"}
        if slots != set(cls.FIELD_NAMES):
            raise AssertionError(
                f"{cls.__name__}: __slots__ and PARAMS disagree - "
                f"only in __slots__: {slots - set(cls.FIELD_NAMES) or None}; "
                f"only in the parameter tables: {set(cls.FIELD_NAMES) - slots or None}")
        if cls.LAYOUT not in LAYOUTS:
            raise AssertionError(f"{cls.__name__}.LAYOUT is {cls.LAYOUT!r}, not a LAYOUT_* code")
        if len(set(cls.FIELD_NAMES)) != len(cls.FIELD_NAMES):
            raise AssertionError(f"{cls.__name__} has a duplicate field name")
        for p in cls.NUMERIC_PARAMS:
            if p.fmt not in ("size", "angle", "offset", "int"):
                raise AssertionError(f"{cls.__name__}.{p.field} has invalid fmt {p.fmt!r}")
            if not isinstance(p.default, int if not p.animate else (int, float)):
                raise AssertionError(f"{cls.__name__}.{p.field}'s default is {p.default!r}")
            if (p.lo is None) != (p.hi is None) or (p.lo is None) != (p.step is None):
                raise AssertionError(f"{cls.__name__}.{p.field}: lo, hi and step go together")
            if p.lo is not None and not p.lo <= p.default <= p.hi:
                raise AssertionError(f"{cls.__name__}.{p.field}'s default is outside its range")
            seen = names_seen.setdefault(p.field, (p.fmt, p.animate))
            if seen != (p.fmt, p.animate):
                raise AssertionError(f"{p.field} means different things on different shapes "
                                     f"(fmt/animate {seen} vs {(p.fmt, p.animate)})")
    if len({cls.NAME for cls in SHAPE_CLASSES}) != len(SHAPE_CLASSES):
        raise AssertionError("two shape kinds have the same NAME")

    if len(DEFAULT_BUDDHA_POINTS) != N_BUDDHA_POINTS:
        raise AssertionError(
            f"DEFAULT_BUDDHA_POINTS has {len(DEFAULT_BUDDHA_POINTS)} entries but "
            f"BUDDHA_POINT_CATALOG (N_BUDDHA_POINTS) has {N_BUDDHA_POINTS}")
    if len(BUDDHA_POINT_SHORT) != N_BUDDHA_POINTS:
        raise AssertionError(
            f"BUDDHA_POINT_SHORT has {len(BUDDHA_POINT_SHORT)} entries but "
            f"BUDDHA_POINT_CATALOG (N_BUDDHA_POINTS) has {N_BUDDHA_POINTS}")
    for _c in range(3):
        if len(DEFAULT_BUDDHA_POINTS_PER_CHAN[_c]) != N_BUDDHA_POINTS:
            raise AssertionError(
                f"DEFAULT_BUDDHA_POINTS_PER_CHAN[{_c}] doesn't match "
                f"N_BUDDHA_POINTS ({N_BUDDHA_POINTS})")
    for _name, _lst in (("POINT_LABELS", POINT_LABELS), ("POINT_DESCRIPTIONS",
                        POINT_DESCRIPTIONS), ("POINT_SHORT", POINT_SHORT)):
        if len(_lst) != N_POINT_CHOICES:
            raise AssertionError(
                f"{_name} has {len(_lst)} entries but N_POINT_CHOICES is "
                f"{N_POINT_CHOICES}")
    if sorted(POINT_MENU_ORDER) != list(range(N_POINT_CHOICES)):
        raise AssertionError("POINT_MENU_ORDER must list every point choice exactly once")
    if POINT_MENU_ORDER[:2] != [POINT_ORIGIN, POINT_TOP_LEFT_CORNER]:
        raise AssertionError("the two fixed points come first in the point dropdowns")
    if set(CONSTANT_POINT_COORDS) != {POINT_ORIGIN, POINT_TOP_LEFT_CORNER}:
        raise AssertionError("CONSTANT_POINT_COORDS doesn't match the fixed point ids")
    if len(CHAN_SRC_LABELS) != 3:
        raise AssertionError(
            f"CHAN_SRC_LABELS has {len(CHAN_SRC_LABELS)} entries but there are 3 "
            "channel sources (Query, Buddha, Constant)")
    if len(CHAN_CONSTANT_LABELS) != 2:
        raise AssertionError(
            f"CHAN_CONSTANT_LABELS has {len(CHAN_CONSTANT_LABELS)} entries but a "
            "Constant channel only ever holds NONE_ZERO or NONE_ONE")
    if "constant_choice" not in ViewConfig.__slots__:
        raise AssertionError("ViewConfig is missing its dedicated constant_choice field")

    kind_keys = [cls.KEY for cls in SHAPE_CLASSES]
    if (None in kind_keys or len(set(kind_keys)) != len(kind_keys)
            or not all(PROJECT_KEY_PATTERN.fullmatch(key) for key in kind_keys)):
        raise AssertionError("every shape kind needs its own KEY (lower-case letters, digits "
                             f"and underscores), which a project file calls it by: {kind_keys}")
    for catalog, wanted in (
            (CAT_CHAN_SOURCE, {CHAN_SRC_QUERY, CHAN_SRC_BUDDHA, CHAN_SRC_CONSTANT}),
            (CAT_CHAN_KIND, {KIND_POINT, KIND_OTHER}),
            (CAT_METRIC, set(range(N_STATS))), (CAT_POINT, set(range(N_POINT_CHOICES))),
            (CAT_BUDDHA_POINT, set(range(N_BUDDHA_POINTS))),
            (CAT_BUDDHA_COMPRESSION, set(range(len(BUDDHA_COMPRESSION_LABELS)))),
            (CAT_BUDDHA_BRIGHTNESS, set(range(len(BUDDHA_BRIGHTNESS_OPTIONS)))),
            (CAT_CONSTANT, {NONE_ZERO, NONE_ONE}),
            (CAT_EFFECT, set(range(len(OBSTACLE_EFFECT_NAMES))))):
        if set(catalog.codes) != wanted:
            raise AssertionError(
                f"the {catalog.name} catalog (the keys project files use) does not cover exactly "
                f"the program's codes - missing a key for {wanted - set(catalog.codes) or None}, "
                f"a key for a code that does not exist: {set(catalog.codes) - wanted or None}")
    choice_settings = {key for key, spec in RAY_SETTING_SPECS.items() if spec.kind == "choice"}
    if set(RAY_CHOICE_CATALOGS) != choice_settings:
        raise AssertionError("RAY_CHOICE_CATALOGS and the choice settings in RAY_SETTING_SPECS "
                             f"disagree: {set(RAY_CHOICE_CATALOGS) ^ choice_settings}")
    for key, catalog in RAY_CHOICE_CATALOGS.items():
        if set(catalog.codes) != set(range(len(RAY_SETTING_SPECS[key].names))):
            raise AssertionError(f"the catalog of the {key} setting does not cover its choices")
    if set(SETTING_WIDGET_ATTRS) != info_keys:
        raise AssertionError(
            "SETTING_WIDGET_ATTRS and SETTING_INFO disagree - "
            f"only in SETTING_WIDGET_ATTRS: {set(SETTING_WIDGET_ATTRS) - info_keys or None}; "
            f"only in SETTING_INFO: {info_keys - set(SETTING_WIDGET_ATTRS) or None}")
    if set(RAY_SETTING_SPECS) != info_keys - {"outline"}:
        raise AssertionError(
            "RAY_SETTING_SPECS and SETTING_INFO disagree - "
            f"only in RAY_SETTING_SPECS: {set(RAY_SETTING_SPECS) - info_keys or None}; "
            f"only in SETTING_INFO: {info_keys - {'outline'} - set(RAY_SETTING_SPECS) or None}")
    spec_paths = sorted(spec.path for spec in RAY_SETTING_SPECS.values())
    if spec_paths != sorted(ray_setting_leaf_paths()):
        raise AssertionError(
            "RAY_SETTING_SPECS and the RaySettings fields disagree - "
            f"fields without a row: {set(ray_setting_leaf_paths()) - set(spec_paths) or None}; "
            f"rows without a field: {set(spec_paths) - set(ray_setting_leaf_paths()) or None}")
    bad_kinds = [k for k, spec in RAY_SETTING_SPECS.items()
                 if spec.kind not in ("text", "int_text", "float_text", "number", "bool", "choice")
                 or bool(spec.names) != (spec.kind == "choice")]
    if bad_kinds:
        raise AssertionError(f"RAY_SETTING_SPECS rows with a bad kind or names: {bad_kinds}")


run_self_checks()


def make_mapping(kinds, point_metrics, other_metrics, chan_src=None, constant_choices=None):
    """The metric (or NONE_*/CHAN_BUDDHA) each channel is currently using."""
    def one(c):
        if chan_src is not None and chan_src[c] == CHAN_SRC_BUDDHA:
            return CHAN_BUDDHA
        if chan_src is not None and chan_src[c] == CHAN_SRC_CONSTANT:
            if constant_choices is None:
                raise ValueError(
                    f"channel {c} is CHAN_SRC_CONSTANT but make_mapping was called "
                    "without constant_choices")
            return constant_choices[c]
        return point_metrics[c] if kinds[c] == KIND_POINT else other_metrics[c]
    return tuple(one(c) for c in range(3))


# Brighter colors for the on-image sample-point markers (dark image behind them)
SAMPLE_COLORS = {"RGB": ["#ff5252", "#4cff6a", "#5aa0ff"],
                  "HSV": ["#c77dff", "#3de0e8", "#d8d8d8"]}

# The "Include coloration info" export overlay: pure black background (per
# request), and a light gray - never pure white - for the text.
COLORATION_INFO_BG = (0, 0, 0)
COLORATION_INFO_FG = (210, 210, 210)
COLORATION_INFO_SCALE = 1
COLORATION_INFO_MARGIN = 4
COLORATION_INFO_LINE_GAP = 2


def coloration_info_lines(view):
    """One view's coloration settings as a list of text lines, mirroring how
    the Coloration tab itself is organized: the color model, then each
    channel's Source and either its Metric (+ Point 1/2, if it's a point-
    comparison metric) or its Buddha points (one per line, as in the
    checkbox list itself) and compression/brightness. Uses the same long
    names shown in the Coloration tab's own dropdowns - the metric's Type
    (point comparison/other) is never shown on its own, though; it's
    implied by which metric is named."""
    lines = [f"Mode:{view.mode}"]
    for c in range(3):
        letter = MODES[view.mode][c][0][0]
        if view.chan_src[c] == CHAN_SRC_BUDDHA:
            lines.append(f"{letter}:Buddha")
            chosen = [BUDDHA_POINT_LABELS[i] for i, on in enumerate(view.buddha_points[c]) if on]
            if chosen:
                lines.extend(f"  {name}" for name in chosen)
            else:
                lines.append("  (none)")
            comp = BUDDHA_COMPRESSION_LABELS[view.buddha_compression[c]]
            bright = BUDDHA_BRIGHTNESS_OPTIONS[view.buddha_brightness[c]]
            lines.append(f"  {comp} {bright}")
        elif view.chan_src[c] == CHAN_SRC_CONSTANT:
            lines.append(f"{letter}:Constant")
            lines.append(f"  {constant_stat_to_label(view.constant_choice[c])}")
        else:
            stat = view.point_metric[c] if view.kinds[c] == KIND_POINT else view.other_metric[c]
            if stat in (NONE_ZERO, NONE_ONE):
                raise ValueError(
                    f"channel {c} has stat={stat} (a Constant-only sentinel) but "
                    f"chan_src is {view.chan_src[c]}, not CHAN_SRC_CONSTANT")
            name = STAT_LABELS[stat]
            lines.append(f"{letter}:{name}")
            if stat in POINT_STATS:
                p1 = POINT_SHORT[view.point1_choice[c]]
                p2 = POINT_SHORT[view.point2_choice[c]]
                lines.append(f"  P1:{p1} P2:{p2}")
    return lines


FIGURE_INFO_COLUMN_CHARS = 32        # a column of figure settings holds lines up to this long


def figure_info_blocks(settings, obstacles):
    """Every setting that defines the figure - the mathematical result of
    perfectly simulating these rules, of which each generated image is only a
    view - as a list of blocks of text lines: the stopping limits and segment
    length, the emission direction, the confinement rule, the fuzzing, and every
    obstacle. (Coloration, decorations and the
    Buddha blur only process or decorate the generated data, so they aren't
    part of it.) The blocks' sizes depend only on how many obstacles there
    are, never on the settings' values - so a video's frames, whose settings
    may be animated, all lay out identically."""
    limits = [f"Iter limit:{settings.max_iter}",
              "Dist limit:" + ("none" if settings.max_dist == 0
                               else f"{settings.max_dist:.2f}"),
              f"Segment:x{settings.seg_mult:.2f}"]
    toward = settings.emit_mode == EMIT_TOWARDS
    emission = [f"Emit:{EMIT_MODE_NAMES[settings.emit_mode]}",
                f"  target r {settings.emit_target_r:.9g}" if toward else " ",
                f"  target angle {settings.emit_target_theta:.9g}" if toward else " ",
                f"  offset x {settings.emit_target_cx:.9g}" if toward else " ",
                f"  offset y {settings.emit_target_cy:.9g}" if toward else " ",
                f"  angle offset {settings.emit_offset:+.2f}\u00b0"]
    confinement = [f"Confine:{CONFINE_METHOD_NAMES[settings.confine_method]}",
                   f"  window {settings.confine_window}",
                   ("  box size " if settings.confine_method == CONFINE_BOX
                    else "  path length ") + f"{settings.confine_threshold:.9g}"]
    f = settings.fuzz
    if not f.enabled:
        fuzz = ["Fuzz:Off"] + [" "] * 4
    else:
        fuzz = ["Fuzz:On",
                f"  Emit jitter \u00b1{f.emit_jitter:.1f}\u00b0",
                f"  Bound {f.bnd_offset:+.1f}\u00b0 \u00b1{f.bnd_jitter:.1f}\u00b0 "
                f"\u00d7{f.bnd_mult:.2f}",
                f"  Self {f.self_offset:+.1f}\u00b0 \u00b1{f.self_jitter:.1f}\u00b0 "
                f"\u00d7{f.self_mult:.2f}",
                f"  Seed {f.seed}"]
    blocks = [limits, emission, confinement, fuzz]
    if obstacles:
        blocks += [o.info_lines(i + 1) for i, o in enumerate(obstacles)]
    else:
        blocks.append(["no obstacles"])
    return blocks


def figure_info_lines(settings, obstacles):
    """figure_info_blocks(), flattened into one list of lines."""
    return [line for block in figure_info_blocks(settings, obstacles) for line in block]


def layout_info_columns(blocks, lines_per_column):
    """Packs blocks of text lines into columns of at most `lines_per_column`
    lines, top to bottom then left to right, keeping each block together
    where it fits (a blank line separates blocks within a column) and
    splitting a block only if it is taller than a whole column."""
    columns = [[]]
    for block in blocks:
        col = columns[-1]
        if col and len(col) + 1 + len(block) > lines_per_column and len(block) <= lines_per_column:
            columns.append([])
            col = columns[-1]
        if col:
            col.append(" ")                        # a blank line between blocks
        for line in block:
            if len(col) >= lines_per_column:
                columns.append([])
                col = columns[-1]
            col.append(line)
    return columns


def resolution_scale(size):
    """The pixel-font scale factor for info text relating to something
    `size` pixels across: never smaller than the SIZE=512 baseline scale,
    growing in whole steps above it, so text stays about the same size
    relative to whatever it's describing rather than shrinking to
    illegibility at high export resolutions."""
    return COLORATION_INFO_SCALE * max(1, round(size / SIZE))


def _draw_text_block(img, lines, x0, y0, avail_w, avail_h, scale, draw_bg):
    """Draws `lines` top to bottom into `img` starting at (x0, y0), each in
    the tiny pixel font at `scale`. draw_bg=True paints an individual solid
    background box behind each line first (for drawing over existing image
    content); draw_bg=False skips that (for a margin area that's already a
    solid color). Clips safely to (avail_w, avail_h) and to img's own
    bounds - never raises even if the text would overflow."""
    line_h = 7 * scale + COLORATION_INFO_LINE_GAP * scale
    y = y0
    y_limit = min(y0 + avail_h, img.shape[0])
    x_limit = min(x0 + avail_w, img.shape[1])
    for line in lines:
        if y >= y_limit:
            break
        mask = render_text_line(line, scale)
        h, w = mask.shape
        draw_h = min(h, y_limit - y)
        draw_w = min(w, x_limit - x0)
        if draw_w <= 0:
            y += line_h
            continue
        if draw_bg:
            bg_y1 = min(y + h + 2 * scale, y_limit)
            bg_x1 = min(x0 + w + 2 * scale, x_limit)
            img[max(y0, y - scale):bg_y1, max(x0, x0 - scale):bg_x1] = COLORATION_INFO_BG
        region = img[y:y + draw_h, x0:x0 + draw_w]
        region[mask[:draw_h, :draw_w]] = COLORATION_INFO_FG
        y += line_h


def draw_coloration_info(img, views, view_width):
    """Draws each of `views`' coloration_info_lines onto its own
    view_width-wide portion of the already-composed `img` (side by side, in
    the same order as `views`), top-left, in the tiny pixel font on a solid
    background box drawn over the existing image content. The text scales
    up with the per-view resolution (the smaller of view_width and the
    image height, so an unusually elongated export doesn't get oversized
    text) - see resolution_scale."""
    scale = resolution_scale(min(view_width, img.shape[0]))
    m = COLORATION_INFO_MARGIN * scale
    for i, v in enumerate(views):
        x_lo = i * view_width
        _draw_text_block(img, coloration_info_lines(v), x_lo + m, m,
                         view_width - 2 * m, img.shape[0] - 2 * m, scale, draw_bg=True)


def add_bottom_color_margin(img, views, view_width):
    """Returns a new, taller copy of `img`: a solid black strip appended
    below it, view_width wide per view (so each view's own coloration info
    sits directly beneath that view, matching its own side-by-side
    position) and a quarter that tall, holding that view's
    coloration_info_lines. Never mutates `img`."""
    h, w, _ = img.shape
    margin_h = view_width // 4
    out = np.empty((h + margin_h, w, 3), dtype=img.dtype)
    out[:h] = img
    out[h:] = COLORATION_INFO_BG
    scale = resolution_scale(view_width)   # the view's own size, not the margin's
                                           # (always smaller, one quarter of it) -
                                           # otherwise the text would barely ever
                                           # grow even at large export resolutions
    m = COLORATION_INFO_MARGIN * scale
    for i, v in enumerate(views):
        x_lo = i * view_width
        _draw_text_block(out, coloration_info_lines(v), x_lo + m, h + m,
                         view_width - 2 * m, margin_h - 2 * m, scale, draw_bg=False)
    return out


def add_left_figure_margin(img, settings, obstacles):
    """Returns a new, wider copy of `img`: a solid black strip added to its
    left edge, as tall as `img` itself, holding every setting that defines
    the figure (see figure_info_blocks). The strip is half as wide as `img`
    - or wider, with more columns of text, if the settings need more room
    than that - and its size depends only on `img`'s size and the number of
    obstacles, so every frame of a video gets the same one. Never mutates
    `img`."""
    h, w, _ = img.shape
    scale = resolution_scale(h)   # the view's own size, not the margin's (see
                                  # add_bottom_color_margin's identical fix)
    m = COLORATION_INFO_MARGIN * scale
    line_h = 7 * scale + COLORATION_INFO_LINE_GAP * scale
    columns = layout_info_columns(figure_info_blocks(settings, obstacles),
                                  max(1, (h - 2 * m) // line_h))
    col_w = 6 * FIGURE_INFO_COLUMN_CHARS * scale
    col_gap = 12 * scale
    margin_w = max(w // 2, 2 * m + len(columns) * col_w + (len(columns) - 1) * col_gap)
    out = np.empty((h, margin_w + w, 3), dtype=img.dtype)
    out[:, margin_w:] = img
    out[:, :margin_w] = COLORATION_INFO_BG
    for k, lines in enumerate(columns):
        _draw_text_block(out, lines, m + k * (col_w + col_gap), m, col_w, h - 2 * m, scale,
                         draw_bg=False)
    return out


def apply_info_locations(rgb, views, view_width, settings, obstacles, color_location,
                         figure_location, mutate_ok):
    """Applies the Export designer's Color info location and Figure settings
    location choices to `rgb` (already composed, before any margin
    expansion), returning the (possibly larger) final image - a plain
    module-level function, safe to call from a background thread, since it
    never reads a Tk variable directly (callers capture color_location/
    figure_location on the main thread first). `settings` and `obstacles` are
    what defines the figure being exported (see figure_info_blocks). mutate_ok=False never draws
    into `rgb` in place for the "Overlay" case (a defensive copy is made
    first instead) - needed when `rgb` might be a shared, reused buffer
    like self.frame."""
    if color_location == "Overlay":
        if not mutate_ok:
            rgb = rgb.copy()
        draw_coloration_info(rgb, views, view_width)
    elif color_location == "Bottom margin":
        rgb = add_bottom_color_margin(rgb, views, view_width)
    if figure_location == "Left margin":
        rgb = add_left_figure_margin(rgb, settings, obstacles)
    return rgb


def default_map_help(mode):
    text = ("For each channel, choose the Type of metric: point comparison "
            "metrics measure the distance between two sample points of the "
            "ray (Point 1 and Point 2); other metrics don't need a point. "
            "Changes recolor the image immediately.")
    if mode == "HSV":
        text += " Hue 0 and hue 1 are both red."
    return text


# Decorations: (key, label, default). Shown in two groups on the Decorations tab.
DECO_IMAGE = [      # drawn into the image: they appear in saved files (and keyframes)
    ("outline", "Obstacle outlines", True),
]
DECO_PREVIEW = [    # drawn over the image on screen only: never saved, never in a keyframe
    ("pending", "Pending shape outline (dashed)", True),
    ("ray_path", "Ray path", True),
    ("ray_start", "Ray start marker", True),
    ("ray_end", "Ray endpoint marker", True),
    ("guides", "Endpoint guide lines", True),
    ("samples", "Color sample points (one ring per point-based channel)", False),
]


# --------------------------------------------------------------------------- #
# Color models
# --------------------------------------------------------------------------- #
def hsv_to_rgb_np(h, s, v):
    """Vectorised HSV -> RGB. Inputs/outputs are arrays in 0..1."""
    h6 = (h % 1.0) * 6.0
    fl = np.floor(h6)
    f = h6 - fl
    i = fl.astype(np.int64) % 6
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return r, g, b


def channels_to_rgb(mode, v0, v1, v2):
    """
    Convert three arrays of channel values (0..1) in the given color model
    to red, green, blue arrays on a 0..255 scale.
      RGB: the values are red, green, blue.
      HSV: the values are hue, saturation, value.
    """
    if mode == "HSV":
        r, g, b = hsv_to_rgb_np(v0, v1, v2)
        return r * 255.0, g * 255.0, b * 255.0
    return v0 * 255.0, v1 * 255.0, v2 * 255.0


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App:
    PERIOD = 1.0 / REFRESH_HZ

    def __init__(self, root):
        self.root = root
        root.title(os.path.basename(sys.argv[0]) or "Light bounces")
        root.resizable(False, False)
        self.build_menu_bar()
        self._crash_dumps = 0
        root.report_callback_exception = self.on_callback_exception
        self.install_crash_hooks()
        root.after(100, self.keep_window_on_screen)   # once the window has its real size

        # Slightly smaller font for the dense metrics list
        base_font = tkfont.nametofont("TkDefaultFont")
        self.small_font = base_font.copy()
        size = base_font.cget("size")
        self.small_font.configure(size=size - 1 if size > 0 else size + 1)
        self.small_bold_font = self.small_font.copy()
        self.small_bold_font.configure(weight="bold")

        # Style used to flag an invalid number entry
        ttk.Style().configure("Invalid.TEntry", foreground=ERROR_RED)

        # ---- state ---- (defined before any widget can call back into us)
        self.ready = False                # numba kernels compiled?
        self.running = False              # generation in progress?
        self.done = False                 # image fully generated?
        self.generated_settings = None    # settings the current image was started with
        self.baseline = DEFAULT_RAY_SETTINGS  # what "Revert" restores
        # What the current image was generated with is generated_settings (see
        # settings_of_current_image); these are only the kernel-ready arrays made from it.
        self.gen_fuzz = fuzz_rule(DEFAULT_FUZZ)
        self.gen_confine = CONFINE_DEFAULT
        self.gen_emission = EMISSION_DEFAULT
        self.last_iter = DEFAULT_ITER     # most recent valid entry values
        self.last_dist = DEFAULT_DIST
        self.last_emit_target_r = DEFAULT_EMIT_TARGET_R
        self.last_emit_target_theta = DEFAULT_EMIT_TARGET_THETA
        self.last_emit_target_cx = DEFAULT_EMIT_TARGET_CX
        self.last_emit_target_cy = DEFAULT_EMIT_TARGET_CY
        self.last_confine_window = DEFAULT_CONFINE_WINDOW
        self.last_confine_threshold = DEFAULT_CONFINE_THRESHOLD
        self.last_seed = DEFAULT_FUZZ_SEED
        self.views = [ViewConfig(0)]      # list[ViewConfig], in export/composition order
        self._view_counter = 1
        self.selected_view_id = 0         # which view is selected - both for editing in
                                           # the Coloration tab AND shown in the preview
        self.mapping = make_mapping(self.live_view().kinds, self.live_view().point_metric,
                                    self.live_view().other_metric, self.live_view().chan_src,
                                    self.live_view().constant_choice)
        self.buddha_counts = [None, None, None]   # per channel of the live view: (SIZE, SIZE)
                                                   # float64 RAW (unblurred) visit counts,
                                                   # or None while unused/before Start
        self.buddha_counts_blurred = [None, None, None]   # per channel: the lazily-computed,
                                                           # cached blurred version - see
                                                           # blurred_buddha_counts()
        self.view_thumb_cache = {}        # non-live view id -> (rows_done, thumbnail array)
                                          # at the time it was computed - see
                                          # render_view_thumbnail(). The live view is
                                          # never cached here; it always reuses self.img
                                          # directly instead.
        self.deco = {}                    # decoration key -> BooleanVar
        self.rows_done = 0
        self.batch_rows = 1
        self.tick_id = None
        self.outline_job = None
        self.start_time = 0.0
        self.elapsed = 0.0
        self.next_tick = 0.0
        self.hover_px = None
        self.pinned_px = None             # pixel pinned by a click, or None
        self.fuzz_tk_widgets = []         # classic Tk widgets on the Fuzzing tab
        self.fuzz_ttk_widgets = []        # ttk widgets on the Fuzzing tab
        self.hover_pts = np.empty((ITER_LIMIT_MAX + 2, 2))
        self.hover_kinds = np.empty(ITER_LIMIT_MAX + 2)
        self.hover_raw = np.empty(N_RAW)
        self.hover_vec = np.empty(N_STATS)
        self.hover_sp = np.empty((N_POINTS, 2))

        # ---- timeline / video state ----
        self.keyframes = []               # list[Keyframe], in the order shown
        self._kf_counter = 0
        self.selected_keyframe = None
        self.kf_modified = set()          # setting keys the user changed since the selected
                                          # keyframe was selected / last saved (blue outlines)
        self.kf_last_seen = None          # capture_all_settings() at the previous check
        self.kf_applying = False          # True while a keyframe's values are being loaded
        self.outlines = {}                # widget -> ModifiedOutline
        self.diff_expanded = set()        # titles of the expanded sections of the keyframe
                                          # settings list (everything starts collapsed)
        self.render_thread = None
        self.render_cancel = None

        # ---- obstacles ---- (the scene: not part of Settings, but every obstacle's
        # own fields are keyframeable). The app starts with one obstacle: the
        # unit circle, which reflects rays and fires them inside itself.
        self.obstacles = [default_starting_obstacle(0)]   # list[Shape], in list order
        self._obst_counter = 1
        self.selected_obstacle = self.obstacles[0]
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_bounds = np.empty((0, OBST_BOUND_LEN), dtype=np.float64)
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=np.bool_)
        self.gen_obstacles = ObstacleArrays(      # the obstacles as generated (see start)
            shapes=empty_i64, bounds=empty_bounds, cx=empty_f64, cy=empty_f64,
            effects=np.empty((0, 2), dtype=np.int64), fire_inside=empty_bool)
        self.gen_obstacles_sig = ()
        self.gen_obstacle_snapshots = []   # frozen copies of the obstacles as generated,
                                           # for the export's figure settings text
        self.gen_obstacles_by_id = {}      # obstacle id -> frozen settings, for the
                                           # dashed "changed since last generation" guide

        # ---- per-pixel data ----
        self.stats = np.zeros((SIZE, SIZE, N_STATS), dtype=np.float32)
        self.points = np.zeros((SIZE, SIZE, N_POINTS, 2), dtype=np.float32)
        self.status = np.zeros((SIZE, SIZE), dtype=np.uint8)   # 0 = not computed yet

        # ---- image buffers ----
        self.img = np.empty((SIZE, SIZE, 3), dtype=np.uint8)
        self.img[:] = BACKGROUND
        self.frame = np.empty_like(self.img)      # img + white outline; this is what
                                                   # gets exported/saved, so it must never
                                                   # hold a preview-only decoration
        self.display_frame = np.empty_like(self.img)   # self.frame + the view-name
                                                        # overlay; only ever pushed to
                                                        # self.photo, never written to disk

        # World coordinates of every pixel centre (also each ray's start point)
        self.gx, self.gy = make_grid(SIZE, SIZE)

        self.build_image_quadrant()
        self.build_controls_quadrant()
        self.build_timeline_panel()
        self.build_info_quadrant()
        self.build_status_quadrant()
        self.build_video_panel()
        self.update_fuzz_enabled_state()

        # Watch the number entries (registered last, once everything exists)
        self.iter_var.trace_add("write", self.on_limit_var_changed)
        self.dist_var.trace_add("write", self.on_limit_var_changed)
        self.confine_window_var.trace_add("write", self.on_limit_var_changed)
        self.confine_threshold_var.trace_add("write", self.on_limit_var_changed)
        self.confine_method_var.trace_add("write", self.on_confine_method_changed)
        for field in EMIT_TARGET_FIELDS:
            getattr(self, f"emit_target_{field}_var").trace_add(
                "write", self.on_limit_var_changed)
        self.emit_mode_var.trace_add("write", self.on_emit_mode_changed)
        self.seed_var.trace_add("write", self.on_seed_var_changed)

        self.draw()
        self.update_status()
        self.update_buttons()
        self.start_project_handling()
        root.update()                              # show the window before compiling
        root.after(20, self.warm_up)

    # ---- projects ---------------------------------------------------------- #
    def start_project_handling(self):
        """Called once the window is built: the starting state is the "new project" one;
        remember it (File > New returns to it) and when a project is saved or has unsaved
        changes."""
        self.project_path = None
        self.default_project = self.capture_project()
        self.saved_signature = project_signature(self.default_project)
        self.title_shown = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)
        for sequence, command in (("<Control-n>", self.new_project),
                                  ("<Control-o>", self.open_project),
                                  ("<Control-s>", self.save_project),
                                  ("<Control-Shift-S>", self.save_project_as)):
            self.root.bind_all(sequence, lambda _event, command=command: command())
        self.update_title()
        self.root.after(PROJECT_POLL_MS, self.poll_project_title)

    def export_option_control(self, path):
        control = self
        for name in path.split("."):
            control = getattr(control, name)
        return control

    def capture_export_options(self):
        """{EXPORT_OPTIONS key: value} from the Export designer and video panel controls."""
        values = {}
        for key, path, kind, _allowed in EXPORT_OPTIONS:
            value = self.export_option_control(path).get()
            if kind == "number":
                value = float(value)
                value = int(value) if value.is_integer() else value
            elif kind == "bool":
                value = bool(value)
            else:
                value = str(value)
            values[key] = value
        return values

    def apply_export_options(self, values):
        """Put saved Export designer / video choices into their controls. A value that is
        missing, of the wrong kind, or not one of the allowed choices leaves its control alone."""
        for key, path, kind, allowed in EXPORT_OPTIONS:
            value = values.get(key)
            if kind == "bool":
                ok = isinstance(value, bool)
            elif kind == "number":
                ok = _is_number(value)
            elif kind == "choice":
                ok = value in allowed.codes
            else:
                ok = isinstance(value, str)
            if ok:
                self.export_option_control(path).set(value)

    def capture_project(self, show_error=True):
        """A Project holding everything now set up (None, after telling the user unless
        `show_error` is False, while an entry holds an invalid value)."""
        ray = self.current_ray_settings()
        if ray is None:
            if show_error:
                messagebox.showerror("Invalid settings", "Fix the highlighted entries first.")
            return None
        selected = self.selected_obstacle
        return Project(
            ray=ray, decorations={key: bool(var.get()) for key, var in self.deco.items()},
            obstacles=tuple(o.snapshot() for o in self.obstacles),
            views=tuple(snapshot_view(v) for v in self.views),
            keyframes=tuple(snapshot_keyframes(self.keyframes)),
            export=self.capture_export_options(), obstacle_counter=self._obst_counter,
            view_counter=self._view_counter, keyframe_counter=self._kf_counter,
            selected_obstacle_id=selected.id if selected is not None else -1,
            selected_view_id=self.selected_view_id)

    def discard_generated_image(self):
        """Back to the blank state before the first Start (stopping a generation in progress)."""
        if self.tick_id is not None:
            self.root.after_cancel(self.tick_id)
            self.tick_id = None
        self.running = False
        self.done = False
        self.generated_settings = None
        self.gen_fuzz = fuzz_rule(DEFAULT_FUZZ)
        self.gen_confine = CONFINE_DEFAULT
        self.gen_emission = EMISSION_DEFAULT
        self.gen_obstacles = build_obstacle_arrays([])
        self.gen_obstacles_sig = ()
        self.gen_obstacle_snapshots = []
        self.gen_obstacles_by_id = {}
        self.stats[:] = 0
        self.points[:] = 0
        self.status[:] = 0
        self.img[:] = BACKGROUND
        self.rows_done = 0
        self.batch_rows = 1
        self.view_thumb_cache = {}
        self.canvas.delete("pending")
        self.pinned_px = None
        self.clear_overlay()

    def apply_project(self, project):
        """Make the program hold exactly `project`: the controls, obstacles, views, keyframes
        and export choices are replaced, and the image goes blank until Start. Used by
        File > Open and File > New alike, so a loaded project cannot differ from a fresh one."""
        self.discard_generated_image()
        self.kf_applying = True
        try:
            for key, value in ray_settings_to_dict(project.ray).items():
                self.set_setting_control(key, value)
            self.baseline = project.ray
            for key, value in project.decorations.items():
                if key in self.deco:
                    self.deco[key].set(value)
            self.obstacles = [o.duplicate(o.id) for o in project.obstacles]
            self._obst_counter = project.obstacle_counter
            self.selected_obstacle = next(
                (o for o in self.obstacles if o.id == project.selected_obstacle_id),
                self.obstacles[0] if self.obstacles else None)
            self.views = [view_config_from_snapshot(v) for v in project.views]
            self._view_counter = project.view_counter
            self.selected_view_id = project.selected_view_id
            self.keyframes = snapshot_keyframes(project.keyframes)
            self._kf_counter = project.keyframe_counter
            self.selected_keyframe = None
            self.kf_modified = set()
            self.kf_last_seen = None
            self.apply_export_options(project.export)
        finally:
            self.kf_applying = False
        self.on_limit_var_changed()
        self.on_fuzz_changed()
        self.update_fuzz_enabled_state()
        self.on_save_source_changed()
        self.on_save_format_changed()
        self.on_export_format_changed()
        self.on_bitrate_mode_changed()
        self.on_downsample_factor_changed()
        self.refresh_obstacle_editor()
        self.rebuild_obstacle_list()
        self.refresh_color_controls_for_selected_view()
        self.rebuild_view_list()
        self.refresh_live_preview_state()
        self.on_decoration_changed("outline")
        self.on_decoration_changed("pending")
        self.on_shape_param_changed()
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()
        self.refresh_modified_outlines()
        self.update_buttons()
        self.draw()

    def project_display_name(self):
        return os.path.basename(self.project_path) if self.project_path else "Untitled"

    def is_dirty(self):
        """Whether there are changes not saved in the project (an entry holding an invalid
        value counts as one)."""
        project = self.capture_project(show_error=False)
        if project is None:
            return True
        try:
            return project_signature(project) != self.saved_signature
        except ProjectError:                       # something that cannot be saved is a change
            return True

    def update_title(self):
        title = f"{'*' if self.is_dirty() else ''}{self.project_display_name()} - " \
                f"{os.path.basename(sys.argv[0]) or 'Light bounces'}"
        if title != self.title_shown:
            self.title_shown = title
            self.root.title(title)

    def poll_project_title(self):
        try:
            self.update_title()
        finally:
            self.root.after(PROJECT_POLL_MS, self.poll_project_title)

    def mark_project_saved(self):
        """The project as it now is counts as saved (after New, Open and Save)."""
        project = self.capture_project(show_error=False)
        if project is not None:
            self.saved_signature = project_signature(project)
        self.update_title()

    def can_replace_project(self, action, check_busy=True):
        """Whether the current project may be replaced (or the program closed): not while a
        video renders, and if it has unsaved changes only once the user has chosen to save
        them or not."""
        if check_busy and self.busy_with_render_or_reencode():
            messagebox.showinfo("A video is rendering",
                                f"Wait for the video render to finish (or cancel it) before "
                                f"you {action}.")
            return False
        if not self.is_dirty():
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved changes",
            f"Save the changes to {self.project_display_name()} before you {action}?")
        if answer is None:
            return False
        return self.save_project() if answer else True

    def new_project(self):
        if not self.can_replace_project("start a new project"):
            return
        self.apply_project(self.default_project)
        self.project_path = None
        self.mark_project_saved()

    def open_project(self):
        path = filedialog.askopenfilename(
            parent=self.root, title="Open project", initialdir=self.project_dialog_dir(),
            filetypes=PROJECT_FILE_TYPES)
        if not path:
            return
        try:
            project = load_project(path)
        except ProjectError as exc:
            messagebox.showerror("Could not open project", f"{path}\n\n{exc}")
            return
        if not self.can_replace_project("open another project"):
            return
        self.apply_project(project)
        self.project_path = path
        self.mark_project_saved()

    def project_dialog_dir(self):
        return os.path.dirname(self.project_path) if self.project_path else ensure_project_dir()

    def save_project(self):
        """Save to the project's file (asking for one first if it has none). True if saved."""
        if self.project_path is None:
            return self.save_project_as()
        return self.save_project_to(self.project_path)

    def save_project_as(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save project as", initialdir=self.project_dialog_dir(),
            initialfile=self.project_display_name() if self.project_path
            else "untitled" + PROJECT_EXTENSION,
            defaultextension=PROJECT_EXTENSION, filetypes=PROJECT_FILE_TYPES)
        return bool(path) and self.save_project_to(path)

    def save_project_to(self, path):
        project = self.capture_project()
        if project is None:
            return False
        try:
            save_project(path, project)
        except ProjectError as exc:
            messagebox.showerror("Could not save project", str(exc))
            return False
        self.project_path = path
        self.saved_signature = project_signature(project)
        self.update_title()
        return True

    def on_close_window(self):
        if self.can_replace_project("close the program", check_busy=False):
            self.root.destroy()

    # ---- layout ------------------------------------------------------------ #
    def build_image_quadrant(self):
        """Top left: the image. Hover overlay lives on the canvas, not in the image."""
        self.canvas = tk.Canvas(self.root, width=SIZE, height=SIZE, bd=0,
                                highlightthickness=0, bg="black",
                                cursor="crosshair")
        self.canvas.grid(row=0, column=0, padx=(PAD, PAD // 2), pady=(PAD, PAD // 2))
        self.photo = tk.PhotoImage(width=SIZE, height=SIZE)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", self.on_leave)
        self.canvas.bind("<Button-1>", self.on_click)

        # One-pixel-thick images with a uniform alpha give true semi-transparent
        # guide lines (Tk canvas lines have no alpha of their own).
        self.hline_img = None
        self.vline_img = None
        try:
            self.hline_img = tk.PhotoImage(
                data=png_rgba(SIZE, 1, GUIDE_RGBA), format="png")
            self.vline_img = tk.PhotoImage(
                data=png_rgba(1, SIZE, GUIDE_RGBA), format="png")
        except tk.TclError:
            self.hline_img = self.vline_img = None   # fall back to stippled lines

    def build_controls_quadrant(self):
        """Top middle: four tabs of controls, with Start / Revert below them."""
        ctl = ttk.LabelFrame(self.root, text="Controls", width=RIGHT_W, height=SIZE)
        ctl.grid(row=0, column=1, sticky="nsew",
                 padx=(PAD // 2, PAD // 2), pady=(PAD, PAD // 2))
        ctl.grid_propagate(False)
        ctl.columnconfigure(0, weight=1)
        ctl.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(ctl)
        notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=(6, 4))
        sim_tab = ttk.Frame(notebook)
        fuzz_tab = ttk.Frame(notebook)
        color_tab = ttk.Frame(notebook)
        buddha_blur_tab = ttk.Frame(notebook)
        deco_tab = ttk.Frame(notebook)
        obst_tab = ttk.Frame(notebook)
        notebook.add(sim_tab, text="Simulation")
        notebook.add(fuzz_tab, text="Fuzzing")
        notebook.add(color_tab, text="Coloration")
        notebook.add(buddha_blur_tab, text="Buddha blur")
        notebook.add(deco_tab, text="Decorations")
        notebook.add(obst_tab, text="Obstacles")
        for tab in (sim_tab, fuzz_tab, color_tab, buddha_blur_tab, deco_tab, obst_tab):
            tab.columnconfigure(0, weight=1)

        # The Simulation and Obstacles tabs hold more than fits in their height
        # (and how much depends on the user's font size and display scaling), so
        # they scroll, like the Coloration tab's channel editor.
        self.sim_canvas, sim_inner = make_scrollable_tab(sim_tab)
        self.build_limits_controls(sim_inner)
        self.build_segment_mult_controls(sim_inner)
        self.build_confinement_controls(sim_inner)
        self.build_emission_controls(sim_inner)
        self.build_fuzzing_controls(fuzz_tab)
        self.build_color_controls(color_tab)
        self.build_buddha_blur_controls(buddha_blur_tab)
        self.build_decoration_controls(deco_tab)
        self.obst_canvas, obst_inner = make_scrollable_tab(obst_tab)
        self.build_obstacles_controls(obst_inner)

        buttons = ttk.Frame(ctl)
        buttons.grid(row=1, column=0, sticky="ew", padx=10, pady=(4, 10))
        buttons.columnconfigure((0, 1), weight=1)
        self.start_btn = ttk.Button(buttons, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.revert_btn = ttk.Button(buttons, text="Revert settings",
                                     command=self.revert_settings)
        self.revert_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def build_limits_controls(self, parent):
        """Two validated number-entry fields: iteration and distance limits."""
        lim = ttk.LabelFrame(parent, text="Stopping limits")
        lim.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 6))
        lim.columnconfigure(0, weight=1)
        iter_vcmd = (self.root.register(self.validate_digits), "%P")
        dist_vcmd = (self.root.register(self.validate_dist_digits), "%P")

        self.iter_var = tk.StringVar(value=str(DEFAULT_ITER))
        self.dist_var = tk.StringVar(value=str(DEFAULT_DIST))
        rows = [("Iteration limit (bounces):", self.iter_var, "iter_entry", iter_vcmd),
                ("Distance limit (units, 0 = none):", self.dist_var, "dist_entry", dist_vcmd)]
        for r, (text, var, attr, vcmd) in enumerate(rows):
            ttk.Label(lim, text=text).grid(
                row=r, column=0, sticky="w", padx=(8, 6), pady=(6 if r == 0 else 2, 2))
            entry = ttk.Entry(lim, textvariable=var, width=8, justify="right",
                              validate="key", validatecommand=vcmd)
            entry.grid(row=r, column=1, padx=(0, 8), pady=(6 if r == 0 else 2, 2))
            entry.bind("<FocusOut>", self.on_limit_commit)
            entry.bind("<Return>", self.on_limit_commit)
            entry.bind("<KP_Enter>", self.on_limit_commit)
            setattr(self, attr, entry)

        self.limit_msg_var = tk.StringVar()
        ttk.Label(lim, textvariable=self.limit_msg_var, foreground=ERROR_RED,
                  wraplength=RIGHT_W - 70).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=8)
        ttk.Label(lim, foreground="#666", wraplength=RIGHT_W - 70,
                  text="A ray also stops when it is confined (see Confinement below), "
                       "or when it escapes (has nothing left to hit). "
                       "One unit is half the image width; a segment that would pass "
                       "the distance limit is cut off where it is reached. A "
                       "distance limit of 0 means no distance limit. Large limits "
                       "(or fuzzing) can take a very long time."
                  ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

    def build_segment_mult_controls(self, parent):
        """A single bounded entry: how much of each bounce's true segment
        length is actually travelled before continuing in the newly-bounced
        direction."""
        sm = ttk.LabelFrame(parent, text="Segment length")
        sm.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        sm.columnconfigure(1, weight=1)
        self.seg_mult_scale = self.add_angle_entry(
            sm, 0, "Length multiplier", DEFAULT_SEGMENT_MULT, on_change=self.on_seg_mult_changed,
            lo=SEGMENT_MULT_MIN, hi=SEGMENT_MULT_MAX, fmt="{:.2f}")
        ttk.Label(sm, foreground="#666", wraplength=RIGHT_W - 70,
                  text="Shortens every ordinary bounce's segment to this fraction of "
                       "its true length, pulling the end back towards the start - but "
                       "the ray still continues on from there in the direction that "
                       "bounce calculated, so it visibly changes course in mid-air. "
                       "1 means no effect. While active, the self-bounce loop area "
                       "metric always reads 0, since that polygon no longer reflects "
                       "the ray's own path."
                  ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 6))

    def build_confinement_controls(self, parent):
        """The Confinement section: how many of a ray's most recent path points
        are judged (window), by which method, against what threshold (see
        CONFINE_BOX / CONFINE_PATH). The threshold accepts any number above
        zero, however small."""
        cf = ttk.LabelFrame(parent, text="Confinement")
        cf.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        cf.columnconfigure(0, weight=1)
        window_vcmd = (self.root.register(self.validate_window_digits), "%P")
        threshold_vcmd = (self.root.register(self.validate_positive_float), "%P")

        self.confine_window_var = tk.StringVar(value=str(DEFAULT_CONFINE_WINDOW))
        self.confine_method_var = tk.StringVar(
            value=CONFINE_METHOD_NAMES[DEFAULT_CONFINE_METHOD])
        self.confine_threshold_var = tk.StringVar(
            value=str(DEFAULT_CONFINE_THRESHOLD))
        self.confine_threshold_label_var = tk.StringVar()

        ttk.Label(cf, text="Window size (path points):").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=(6, 2))
        self.confine_window_entry = ttk.Entry(
            cf, textvariable=self.confine_window_var, width=8, justify="right",
            validate="key", validatecommand=window_vcmd)
        self.confine_window_entry.grid(row=0, column=1, padx=(0, 8), pady=(6, 2))

        ttk.Label(cf, text="Detection method:").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=2)
        method_combo = ttk.Combobox(cf, textvariable=self.confine_method_var,
                                    values=CONFINE_METHOD_NAMES, state="readonly", width=12)
        method_combo.grid(row=1, column=1, padx=(0, 8), pady=2)
        method_combo.bind("<<ComboboxSelected>>", lambda _e: method_combo.selection_clear())
        self.confine_method_combo = method_combo

        ttk.Label(cf, textvariable=self.confine_threshold_label_var).grid(
            row=2, column=0, sticky="w", padx=(8, 6), pady=2)
        self.confine_threshold_entry = ttk.Entry(
            cf, textvariable=self.confine_threshold_var, width=14, justify="right",
            validate="key", validatecommand=threshold_vcmd)
        self.confine_threshold_entry.grid(row=2, column=1, padx=(0, 8), pady=2)
        for entry in (self.confine_window_entry, self.confine_threshold_entry):
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, self.on_limit_commit)

        self.confine_msg_var = tk.StringVar()
        ttk.Label(cf, textvariable=self.confine_msg_var, foreground=ERROR_RED,
                  wraplength=RIGHT_W - 70).grid(row=3, column=0, columnspan=2,
                                                sticky="w", padx=8)
        ttk.Label(cf, foreground="#666", wraplength=RIGHT_W - 70,
                  text="A ray stops as confined when its last few path points (the "
                       "window) show it is going nowhere. Box: they fit in a box "
                       "smaller than the threshold in both directions. Path length: "
                       "the segments between them add up to less than the threshold. "
                       "The threshold is in world units (the image spans -1 to 1) and "
                       "can be any number above zero; one pixel of the 512-pixel "
                       f"preview is {PIXEL:g}, the default."
                  ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        self.update_confine_threshold_label()

    def build_emission_controls(self, parent):
        """The Ray emission section: whether rays are fired straight up or from
        their start pixel towards a target point (any coordinates at all), and
        the angle offset added to that direction (positive = more
        counterclockwise)."""
        em = ttk.LabelFrame(parent, text="Ray emission")
        em.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))
        em.columnconfigure(0, weight=1)
        coord_vcmd = (self.root.register(self.validate_signed_float), "%P")

        self.emit_mode_var = tk.StringVar(value=EMIT_MODE_NAMES[DEFAULT_EMIT_MODE])
        for field, default in zip(EMIT_TARGET_FIELDS,
                                  (DEFAULT_EMIT_TARGET_R, DEFAULT_EMIT_TARGET_THETA,
                                   DEFAULT_EMIT_TARGET_CX, DEFAULT_EMIT_TARGET_CY)):
            setattr(self, f"emit_target_{field}_var", tk.StringVar(value=str(default)))

        ttk.Label(em, text="Direction:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=(6, 2))
        mode_combo = ttk.Combobox(em, textvariable=self.emit_mode_var, values=EMIT_MODE_NAMES,
                                  state="readonly", width=14)
        mode_combo.grid(row=0, column=1, padx=(0, 8), pady=(6, 2))
        mode_combo.bind("<<ComboboxSelected>>", lambda _e: mode_combo.selection_clear())
        self.emit_mode_combo = mode_combo

        self.emit_target_entries = []
        for r, (field, text) in enumerate(zip(EMIT_TARGET_FIELDS, EMIT_TARGET_LABELS),
                                          start=1):
            ttk.Label(em, text="Target " + text[0].lower() + text[1:]).grid(
                row=r, column=0, sticky="w", padx=(8, 6), pady=2)
            entry = ttk.Entry(em, textvariable=getattr(self, f"emit_target_{field}_var"),
                              width=14, justify="right", validate="key",
                              validatecommand=coord_vcmd)
            entry.grid(row=r, column=1, padx=(0, 8), pady=2)
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, self.on_limit_commit)
            self.emit_target_entries.append(entry)
            setattr(self, f"emit_target_{field}_entry", entry)

        offset_frame = ttk.Frame(em)
        offset_frame.grid(row=5, column=0, columnspan=2, sticky="w")
        self.emit_offset_scale = self.add_angle_entry(
            offset_frame, 0, "Angle offset (\u00b0, + = counterclockwise)", DEFAULT_EMIT_OFFSET,
            on_change=self.on_emission_changed, fmt="{:.2f}")

        self.emit_msg_var = tk.StringVar()
        ttk.Label(em, textvariable=self.emit_msg_var, foreground=ERROR_RED,
                  wraplength=RIGHT_W - 70).grid(row=6, column=0, columnspan=2,
                                                sticky="w", padx=8)
        ttk.Label(em, foreground="#666", wraplength=RIGHT_W - 70,
                  text="Each ray starts out pointing straight up, or from its own "
                       "start pixel towards the target point (which can be anywhere, "
                       "positioned like an obstacle: a distance r at angle \u03b8 from "
                       "the origin, plus an offset X and Y; a pixel exactly on it points "
                       "up). The angle offset is then "
                       "added, positive meaning more counterclockwise: with Up, the "
                       "default -45\u00b0 gives the up-and-to-the-right direction "
                       "the program has always used. Fuzzing's emission jitter is "
                       "added on top."
                  ).grid(row=7, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        self.update_emission_state()

    def update_emission_state(self):
        """The target point only matters when rays are fired towards it."""
        state = ["!disabled"] if self.emit_mode_var.get() == EMIT_MODE_NAMES[EMIT_TOWARDS] \
            else ["disabled"]
        for entry in self.emit_target_entries:
            entry.state(state)

    def on_emit_mode_changed(self, *_args):
        self.update_emission_state()
        self.on_limit_var_changed()

    def on_emission_changed(self):
        """The angle offset entry committed a new value (already parsed and
        clamped into range by its FloatEntryHandle)."""
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_state_text()
        self.refresh_timeline_diffs()

    def update_confine_threshold_label(self, *_args):
        """The threshold means a box size or a path length, depending on the method."""
        box = self.confine_method_var.get() == CONFINE_METHOD_NAMES[CONFINE_BOX]
        self.confine_threshold_label_var.set(
            "Box size (world units):" if box else "Path length (world units):")

    def on_confine_method_changed(self, *_args):
        self.update_confine_threshold_label()
        self.on_limit_var_changed()

    def add_slider_row(self, parent, row, label, lo, hi, resolution, default, fmt,
                       on_change=None):
        """One compact row: label | slider | current value. Returns the slider."""
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(8, 8), pady=1)
        scale = tk.Scale(parent, from_=lo, to=hi, resolution=resolution,
                         orient="horizontal", showvalue=False,
                         highlightthickness=0, width=14, sliderlength=18)
        scale.set(default)
        scale.grid(row=row, column=1, sticky="ew", pady=1)
        value_label = ttk.Label(parent, text=fmt.format(default), width=6,
                                anchor="e")
        value_label.grid(row=row, column=2, sticky="e", padx=(6, 8))

        callback = on_change or self.on_shape_param_changed

        def changed(value, value_label=value_label, fmt=fmt, callback=callback):
            value_label.configure(text=fmt.format(float(value)))
            callback()

        scale.configure(command=changed)           # only after the set() above
        return scale

    def add_slider_pair(self, parent, row, spec_a, spec_b, on_change=None):
        """Two compact sliders side by side on one row, to save space. Each
        spec is (label, lo, hi, resolution, default, fmt). Returns (scale_a,
        scale_b)."""
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=0, columnspan=3, sticky="ew")
        row_frame.columnconfigure((0, 1), weight=1)
        fa = ttk.Frame(row_frame)
        fa.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        fa.columnconfigure(1, weight=1)
        fb = ttk.Frame(row_frame)
        fb.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        fb.columnconfigure(1, weight=1)
        scale_a = self.add_slider_row(fa, 0, *spec_a, on_change=on_change)
        scale_b = self.add_slider_row(fb, 0, *spec_b, on_change=on_change)
        return scale_a, scale_b

    def add_angle_entry(self, parent, row, label, default, on_change=None,
                        lo=None, hi=None, fmt="{:.2f}"):
        """A label and a text entry on one snug row - a drop-in .get()/.set()
        replacement for a tk.Scale. With no bounds (the default), accepts any
        float, so many-revolution angles can be typed or animated; pass lo/hi
        to clamp into a range instead - used for settings that keep a
        slider-like range but as a text field (jitter/multiplier). Returns a
        FloatEntryHandle."""
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=1)
        ttk.Label(row_frame, text=label).pack(side="left", padx=(8, 6))
        var = tk.StringVar(value=fmt.format(default))
        handle = FloatEntryHandle(var, fmt, default, lo=lo, hi=hi)
        vcmd = (self.root.register(self.validate_float_free), "%P")
        entry = ttk.Entry(row_frame, textvariable=var, width=9, justify="right",
                          validate="key", validatecommand=vcmd)
        entry.pack(side="left")

        def commit(_event=None):
            handle.commit()
            if on_change is not None:
                on_change()

        entry.bind("<FocusOut>", commit)
        entry.bind("<Return>", commit)
        entry.bind("<KP_Enter>", commit)
        handle.widget = entry
        return handle

    def build_fuzzing_controls(self, parent):
        """Angle controls for ray emission, bounces off the boundary, and
        bounces off the ray itself (laid out in a 2x2 grid). Everything except
        the Enable checkbox is greyed out while fuzzing is off."""
        fz = ttk.LabelFrame(parent, text="Fuzzing")
        fz.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 6))
        fz.columnconfigure((0, 1), weight=1, uniform="fuzzcols")

        self.fuzz_enabled_var = tk.BooleanVar(value=DEFAULT_FUZZ.enabled)
        self.fuzz_enable_check = ttk.Checkbutton(
            fz, text="Enable fuzzing", variable=self.fuzz_enabled_var,
            command=self.on_fuzz_changed)
        self.fuzz_enable_check.grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 4))
        ttk.Label(fz, foreground="#666", wraplength=RIGHT_W - 70,
                  text="Perturbs the ray's angles. Emission jitter is added to the "
                       "ray's starting direction (set in the Simulation tab). Each "
                       "bounce group perturbs the "
                       "deflection angle at that kind of bounce: Offset always "
                       "added, Jitter a fresh random amount within \u00b1 the range "
                       "each time, Multiplier scales the angle before those are "
                       "added (\u00d71 = unchanged)."
                  ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        emit_group = ttk.LabelFrame(fz, text="Ray emission")
        emit_group.columnconfigure(1, weight=1)
        self.emit_jitter_scale = self.add_angle_entry(
            emit_group, 0, "Jitter (\u00b1\u00b0)", 0.0, on_change=self.on_fuzz_changed,
            lo=FUZZ_JITTER_MIN, hi=FUZZ_JITTER_MAX, fmt="{:.1f}")
        emit_group.grid(row=2, column=0, sticky="nsew", padx=(6, 3), pady=(0, 6))
        self.fuzz_ttk_widgets += [self.emit_jitter_scale.widget]

        def bounce_group(title, prefix):
            g = ttk.LabelFrame(fz, text=title)
            g.columnconfigure(1, weight=1)
            off = self.add_angle_entry(g, 0, "Offset (\u00b0)", 0.0,
                                       on_change=self.on_fuzz_changed)
            jit = self.add_angle_entry(g, 1, "Jitter (\u00b1\u00b0)", 0.0,
                                       on_change=self.on_fuzz_changed,
                                       lo=FUZZ_JITTER_MIN, hi=FUZZ_JITTER_MAX, fmt="{:.1f}")
            mult = self.add_angle_entry(g, 2, "Multiplier", 1.0,
                                        on_change=self.on_fuzz_changed,
                                        lo=FUZZ_MULT_MIN, hi=FUZZ_MULT_MAX, fmt="{:.2f}")
            setattr(self, f"{prefix}_offset_scale", off)
            setattr(self, f"{prefix}_jitter_scale", jit)
            setattr(self, f"{prefix}_mult_scale", mult)
            self.fuzz_ttk_widgets += [off.widget, jit.widget, mult.widget]
            return g

        bounce_group("Boundary bounces", "bnd").grid(
            row=2, column=1, sticky="nsew", padx=(3, 6), pady=(0, 6))
        bounce_group("Self bounces", "self").grid(
            row=3, column=0, sticky="nsew", padx=(6, 3), pady=(0, 6))

        seed = ttk.LabelFrame(fz, text="Random seed")
        seed.grid(row=3, column=1, sticky="nsew", padx=(3, 6), pady=(0, 6))
        seed.columnconfigure(0, weight=1)
        row = ttk.Frame(seed)
        row.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        self.seed_var = tk.StringVar(value=str(DEFAULT_FUZZ_SEED))
        vcmd = (self.root.register(self.validate_seed_digits), "%P")
        self.seed_entry = ttk.Entry(row, textvariable=self.seed_var, width=11,
                                    justify="right", validate="key",
                                    validatecommand=vcmd)
        self.seed_entry.pack(side="left", padx=(0, 8))
        self.seed_entry.bind("<FocusOut>", self.on_seed_commit)
        self.seed_entry.bind("<Return>", self.on_seed_commit)
        self.seed_entry.bind("<KP_Enter>", self.on_seed_commit)
        randomize_btn = ttk.Button(row, text="Randomize", command=self.randomize_seed)
        randomize_btn.pack(side="left")
        self.fuzz_ttk_widgets += [self.seed_entry, randomize_btn]
        ttk.Label(seed, foreground="#666", wraplength=RIGHT_W // 2 - 60,
                  text="Same seed and settings give the same image every time."
                  ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 6))

    # ---- views (Coloration tab) -------------------------------------------- #
    def view_by_id(self, view_id):
        """The live ViewConfig for one view id, or None if it doesn't exist."""
        for v in self.views:
            if v.id == view_id:
                return v
        return None

    def live_view(self):
        """The view the top-left preview shows: whichever view is currently
        selected in the Coloration tab. Selecting a view always shows it
        here - there's no separate "focus" step."""
        return self.selected_view_obj()

    def refresh_live_preview_state(self):
        """Recompute everything that depends on which view is currently live
        (self.mapping, the running Buddha counts, the displayed image) -
        called whenever self.live_view() changes (selecting a different
        view, or creating/deleting one changes the selection)."""
        live = self.live_view()
        self.mapping = make_mapping(live.kinds, live.point_metric, live.other_metric,
                                    live.chan_src, live.constant_choice)
        self.rebuild_buddha_counts()
        self.update_stat_tags()
        self.recolor(0, self.rows_done)
        self.draw()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)

    def selected_view_obj(self):
        return self.view_by_id(self.selected_view_id)

    def all_view_ids(self):
        """Every view's id, in display order (self.views' own list order)."""
        return [v.id for v in self.views]

    def new_view(self):
        """Add a new (default-settings) view right after the selected one,
        and select it - which also makes it the one shown in the preview."""
        v = ViewConfig(self._view_counter)
        self._view_counter += 1
        idx = self.all_view_ids().index(self.selected_view_id)
        self.views.insert(idx + 1, v)
        self.selected_view_id = v.id
        self.view_thumb_cache.pop(v.id, None)
        self.refresh_live_preview_state()
        self.rebuild_view_list()
        self.refresh_color_controls_for_selected_view()

    def duplicate_view(self):
        """A copy of the selected view, inserted right after it, selected,
        and shown in the preview."""
        src = self.selected_view_obj()
        if src is None:
            return
        new_v = ViewConfig(self._view_counter)
        self._view_counter += 1
        for field in ViewConfig.__slots__:
            if field == "id":
                continue
            val = getattr(src, field)
            setattr(new_v, field, list(val) if isinstance(val, list) else val)
        idx = self.all_view_ids().index(self.selected_view_id)
        self.views.insert(idx + 1, new_v)
        self.selected_view_id = new_v.id
        self.view_thumb_cache.pop(new_v.id, None)
        self.refresh_live_preview_state()
        self.rebuild_view_list()
        self.refresh_color_controls_for_selected_view()

    def delete_view(self):
        """Removes the selected view, unless it's the only one left. The
        deleted view was always the one shown in the preview (selection and
        preview are the same thing), so self.views[0] becomes the new
        selection, and the preview is refreshed to match it right away."""
        if len(self.views) <= 1:
            return
        v = self.view_by_id(self.selected_view_id)
        if v is None:
            return
        self.views.remove(v)
        self.view_thumb_cache.pop(v.id, None)
        self.purge_keyframe_settings(f"view:{v.id}:")
        self.selected_view_id = self.views[0].id
        self.view_thumb_cache.pop(self.selected_view_id, None)
        self.refresh_live_preview_state()
        self.rebuild_view_list()
        self.refresh_color_controls_for_selected_view()

    def select_view(self, view_id):
        """Selects a view for editing - which also makes it the one the
        top-left panel shows."""
        if view_id == self.selected_view_id:
            return
        self.selected_view_id = view_id
        self.view_thumb_cache.pop(view_id, None)   # about to become live (and possibly
                                                    # edited) - any cached thumbnail from
                                                    # before must not be reused once it's
                                                    # non-live again
        self.refresh_live_preview_state()
        self.rebuild_view_list()
        self.refresh_color_controls_for_selected_view()

    def refresh_color_controls_for_selected_view(self):
        """Populate every Color mapping control from the selected view (or
        leave them as-is if the Coloration tab hasn't been built yet, or no
        view is selected). Mirrors refresh_obstacle_editor."""
        if not hasattr(self, "mode_var"):
            return                                  # Coloration tab not built yet
        view = self.selected_view_obj()
        if view is None:
            return
        self._color_editor_loading = True
        try:
            self.mode_var.set(view.mode)
            for c, (name, color) in enumerate(MODES[view.mode]):
                self.comp_labels[c].configure(text=name, foreground=color)
            self.map_help_var.set(default_map_help(view.mode))
            for c in range(3):
                self.chan_src_vars[c].set(CHAN_SRC_LABELS[view.chan_src[c]])
                self.point1_vars[c].set(POINT_LABELS[view.point1_choice[c]])
                self.point2_vars[c].set(POINT_LABELS[view.point2_choice[c]])
                for i, bvar in enumerate(self.buddha_point_vars[c]):
                    bvar.set(view.buddha_points[c][i])
                self.buddha_compression_vars[c].set(
                    BUDDHA_COMPRESSION_LABELS[view.buddha_compression[c]])
                self.buddha_brightness_vars[c].set(
                    BUDDHA_BRIGHTNESS_OPTIONS[view.buddha_brightness[c]])
                self.refresh_row(c)
        finally:
            self._color_editor_loading = False

    def build_views_list(self, parent):
        """New/Dupe/Del stacked to the left of a horizontally-scrolling strip
        of view thumbnails - visually the Obstacles tab's list turned
        sideways, since (unlike obstacles) views are always shown in a fixed
        left-to-right order in an exported image, so browsing them side by
        side mirrors how they'll actually be laid out."""
        top_row = ttk.Frame(parent)
        top_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))
        top_row.columnconfigure(1, weight=1)

        btn_col = ttk.Frame(top_row)
        btn_col.grid(row=0, column=0, sticky="n", padx=(0, 6))
        ttk.Button(btn_col, text="New", width=6, command=self.new_view).pack(
            fill="x", pady=(0, 3))
        ttk.Button(btn_col, text="Dupe", width=6, command=self.duplicate_view).pack(
            fill="x", pady=(0, 3))
        self.view_del_btn = ttk.Button(btn_col, text="Del", width=6, command=self.delete_view)
        self.view_del_btn.pack(fill="x")

        list_container = ttk.Frame(top_row)
        list_container.grid(row=0, column=1, sticky="ew")
        list_container.columnconfigure(0, weight=1)
        view_canvas = tk.Canvas(list_container, height=THUMB_SIZE + 34, highlightthickness=0)
        view_hsb = ttk.Scrollbar(list_container, orient="horizontal",
                                 command=view_canvas.xview)
        view_canvas.configure(xscrollcommand=view_hsb.set)
        view_canvas.grid(row=0, column=0, sticky="ew")
        view_hsb.grid(row=1, column=0, sticky="ew")
        self.view_list_frame = ttk.Frame(view_canvas)
        view_canvas.create_window((0, 0), window=self.view_list_frame, anchor="nw")
        self.view_list_frame.bind(
            "<Configure>",
            lambda _e: view_canvas.configure(scrollregion=view_canvas.bbox("all")))
        # (unlike the vertical obstacle list, the inner frame's WIDTH is never
        # pinned to the canvas's - it needs to grow past it, since scrolling
        # sideways to reach it is the whole point)

        self.rebuild_view_list()

    def rebuild_view_list(self):
        for w in self.view_list_frame.winfo_children():
            w.destroy()
        for idx, view_id in enumerate(self.all_view_ids()):
            selected = view_id == self.selected_view_id
            cell = tk.Frame(self.view_list_frame,
                            highlightthickness=SELECTED_OUTLINE_PX if selected else 0,
                            highlightbackground=SELECTED_OUTLINE,
                            highlightcolor=SELECTED_OUTLINE)
            cell.grid(row=0, column=idx, sticky="n", padx=2, pady=1)

            view_obj = self.view_by_id(view_id)
            thumb = tk.PhotoImage(
                data=encode_png_bytes(self.render_view_thumbnail(view_obj, THUMB_SIZE)),
                format="png")
            thumb_lbl = tk.Label(cell, image=thumb, bd=0, cursor="hand2")
            thumb_lbl.image = thumb              # keep a reference alive
            thumb_lbl.grid(row=0, column=0, padx=2, pady=(2, 0))
            thumb_lbl.bind("<Button-1>", lambda _e, vid=view_id: self.select_view(vid))

            lbl = tk.Label(cell, text=f"view {idx}", cursor="hand2", font=self.small_font)
            lbl.grid(row=1, column=0, pady=(0, 2))
            lbl.bind("<Button-1>", lambda _e, vid=view_id: self.select_view(vid))

        if hasattr(self, "view_del_btn"):
            self.view_del_btn.state(
                ["disabled"] if len(self.views) <= 1 else ["!disabled"])

    def build_color_controls(self, parent):
        """The views list (New/Dupe/Del + a horizontal strip of view
        thumbnails - unlabeled, like the Obstacles tab's list), then the
        "Edit selected view" panel: a scrollable list of the color model row
        followed by one block per channel: a Source (Query/Buddha) row, then
        that source's own controls on the row(s) below it. Shows/edits
        whichever view is selected in the list above
        (self.selected_view_obj()) - see
        refresh_color_controls_for_selected_view()."""
        self._color_editor_loading = False
        self.build_views_list(parent)

        view = self.selected_view_obj()
        mp = ttk.LabelFrame(parent, text="Edit selected view")
        mp.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))
        mp.columnconfigure(0, weight=1)

        # Everything scrolls together - the color model row plus the three
        # channel blocks - since Buddha's controls make a channel several
        # rows tall and it can all overflow the panel.
        list_container = ttk.Frame(mp)
        list_container.grid(row=0, column=0, sticky="nsew", padx=(8, 2), pady=(6, 4))
        mp.rowconfigure(0, weight=1)
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)
        chan_canvas = tk.Canvas(list_container, height=300, highlightthickness=0)
        chan_vsb = ttk.Scrollbar(list_container, orient="vertical",
                                 command=chan_canvas.yview)
        chan_canvas.configure(yscrollcommand=chan_vsb.set)
        chan_canvas.grid(row=0, column=0, sticky="nsew")
        chan_vsb.grid(row=0, column=1, sticky="ns")
        chan_list_frame = ttk.Frame(chan_canvas)
        chan_window = chan_canvas.create_window((0, 0), window=chan_list_frame, anchor="nw")
        chan_list_frame.bind(
            "<Configure>",
            lambda _e: chan_canvas.configure(scrollregion=chan_canvas.bbox("all")))
        chan_canvas.bind(
            "<Configure>", lambda e: chan_canvas.itemconfigure(chan_window, width=e.width))
        chan_list_frame.columnconfigure(0, weight=1)

        top = ttk.Frame(chan_list_frame)
        top.grid(row=0, column=0, sticky="w", padx=(0, 4), pady=(0, 6))
        ttk.Label(top, text="Color model:").pack(side="left", padx=(0, 10))
        self.mode_var = tk.StringVar(value=view.mode)
        self.mode_combo = ttk.Combobox(top, textvariable=self.mode_var,
                                       values=MODE_NAMES, state="readonly", width=8)
        self.mode_combo.pack(side="left")
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)

        self.comp_labels = []
        self.chan_src_vars = []
        self.chan_src_combos = []
        self.query_frames = []
        self.kind_vars = []
        self.kind_combos = []
        self.map_vars = []
        self.map_combos = []
        self.point1_vars = []
        self.point1_combos = []
        self.point2_vars = []
        self.point2_combos = []
        self.buddha_frames = []
        self.buddha_point_vars = []
        self.buddha_point_checks = []
        self.buddha_compression_vars = []
        self.buddha_compression_combos = []
        self.buddha_brightness_vars = []
        self.buddha_brightness_combos = []
        self.constant_frames = []
        self.constant_vars = []
        self.constant_combos = []
        self.chan_blocks = []

        for c, (name, color) in enumerate(MODES[view.mode]):
            block = ttk.Frame(chan_list_frame, relief="groove", borderwidth=1)
            block.grid(row=c + 1, column=0, sticky="ew", padx=(0, 4), pady=4)
            block.columnconfigure(2, weight=1)

            lbl = ttk.Label(block, text=name, foreground=color)
            lbl.grid(row=0, column=0, sticky="w", padx=(6, 10), pady=(4, 2))
            ttk.Label(block, text="Source:").grid(row=0, column=1, sticky="w", pady=(4, 2))
            svar = tk.StringVar(value=CHAN_SRC_LABELS[view.chan_src[c]])
            scombo = ttk.Combobox(block, textvariable=svar, values=CHAN_SRC_LABELS,
                                  state="readonly", width=9)
            scombo.grid(row=0, column=2, sticky="w", padx=(6, 6), pady=(4, 2))
            scombo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_chan_src_changed(c))

            # ---- Query sub-controls: Type, Metric, Point 1, Point 2 ----
            qf = ttk.Frame(block)
            qf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(6, 6), pady=(0, 4))
            qf.columnconfigure(1, weight=1)
            qf.columnconfigure(3, weight=1)

            kvar = tk.StringVar(value=KIND_LABELS[view.kinds[c]])
            kcombo = ttk.Combobox(qf, textvariable=kvar, values=KIND_LABELS,
                                  state="readonly", width=11)
            kcombo.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
            kcombo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_kind_changed(c))

            mvar = tk.StringVar()
            mcombo = ttk.Combobox(qf, textvariable=mvar, state="readonly", width=18)
            mcombo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(0, 6), pady=2)
            mcombo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_metric_selected(c))

            ttk.Label(qf, text="Point 1:").grid(row=1, column=0, sticky="w", pady=2)
            p1var = tk.StringVar(value=POINT_LABELS[view.point1_choice[c]])
            p1combo = ttk.Combobox(qf, textvariable=p1var, values=POINT_MENU_LABELS,
                                   state="readonly", width=14)
            p1combo.grid(row=1, column=1, sticky="ew", padx=(4, 6), pady=2)
            p1combo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_point_selected(c))

            ttk.Label(qf, text="Point 2:").grid(row=1, column=2, sticky="w", pady=2)
            p2var = tk.StringVar(value=POINT_LABELS[view.point2_choice[c]])
            p2combo = ttk.Combobox(qf, textvariable=p2var, values=POINT_MENU_LABELS,
                                   state="readonly", width=14)
            p2combo.grid(row=1, column=3, sticky="ew", padx=(4, 0), pady=2)
            p2combo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_point_selected(c))

            # ---- Buddha sub-controls: Points (multi-select), Compression,
            # brightness multiplier ----
            bf = ttk.Frame(block)
            bf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(6, 6), pady=(0, 4))
            ttk.Label(bf, text="Points:").grid(row=0, column=0, sticky="nw", pady=2)
            pts_grid = ttk.Frame(bf)
            pts_grid.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=2)
            bvars = []
            bchecks = []
            for i, label in enumerate(BUDDHA_POINT_LABELS):
                bvar = tk.BooleanVar(value=view.buddha_points[c][i])
                chk = ttk.Checkbutton(pts_grid, text=label, variable=bvar,
                                      command=lambda c=c: self.on_buddha_points_changed(c))
                chk.grid(row=i, column=0, sticky="w", padx=(0, 10))
                bvars.append(bvar)
                bchecks.append(chk)
            settings_row = ttk.Frame(bf)
            settings_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
            ttk.Label(settings_row, text="Compression:").pack(side="left", padx=(0, 4))
            cvar = tk.StringVar(value=BUDDHA_COMPRESSION_LABELS[view.buddha_compression[c]])
            ccombo = ttk.Combobox(settings_row, textvariable=cvar,
                                  values=BUDDHA_COMPRESSION_LABELS, state="readonly", width=11)
            ccombo.pack(side="left", padx=(0, 12))
            ccombo.bind("<<ComboboxSelected>>",
                        lambda _e, c=c: self.on_buddha_compression_changed(c))
            ttk.Label(settings_row, text="Brightness:").pack(side="left", padx=(0, 4))
            brvar = tk.StringVar(value=BUDDHA_BRIGHTNESS_OPTIONS[view.buddha_brightness[c]])
            brcombo = ttk.Combobox(settings_row, textvariable=brvar,
                                   values=BUDDHA_BRIGHTNESS_OPTIONS, state="readonly", width=5)
            brcombo.pack(side="left")
            brcombo.bind("<<ComboboxSelected>>",
                        lambda _e, c=c: self.on_buddha_brightness_changed(c))

            # ---- Constant sub-controls: a single dropdown, no simulation
            # data involved at all ----
            cnf = ttk.Frame(block)
            cnf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(6, 6), pady=(0, 4))
            ttk.Label(cnf, text="Value:").pack(side="left", padx=(0, 4))
            const_var = tk.StringVar(
                value=constant_stat_to_label(view.constant_choice[c]))
            const_combo = ttk.Combobox(cnf, textvariable=const_var,
                                       values=CHAN_CONSTANT_LABELS, state="readonly", width=9)
            const_combo.pack(side="left")
            const_combo.bind("<<ComboboxSelected>>",
                             lambda _e, c=c: self.on_constant_changed(c))

            self.chan_blocks.append(block)
            self.comp_labels.append(lbl)
            self.chan_src_vars.append(svar)
            self.chan_src_combos.append(scombo)
            self.query_frames.append(qf)
            self.kind_vars.append(kvar)
            self.kind_combos.append(kcombo)
            self.map_vars.append(mvar)
            self.map_combos.append(mcombo)
            self.point1_vars.append(p1var)
            self.point1_combos.append(p1combo)
            self.point2_vars.append(p2var)
            self.point2_combos.append(p2combo)
            self.buddha_frames.append(bf)
            self.buddha_point_vars.append(bvars)
            self.buddha_point_checks.append(bchecks)
            self.buddha_compression_vars.append(cvar)
            self.buddha_compression_combos.append(ccombo)
            self.buddha_brightness_vars.append(brvar)
            self.buddha_brightness_combos.append(brcombo)
            self.constant_frames.append(cnf)
            self.constant_vars.append(const_var)
            self.constant_combos.append(const_combo)
            self.refresh_row(c)

        self.map_help_var = tk.StringVar(value=default_map_help(view.mode))
        ttk.Label(mp, textvariable=self.map_help_var, foreground="#666",
                  wraplength=RIGHT_W - 70, justify="left").grid(
            row=1, column=0, sticky="w", padx=8, pady=(4, 6))

    def build_buddha_blur_controls(self, parent):
        """Radius slider and blur-type dropdown for Buddha coloration's
        point-scattering blur - a simulation-wide setting (like the shape or
        the iteration limit), not a per-view/per-channel one, so it lives in
        its own tab rather than inside the Coloration tab's per-channel
        Buddha controls. Only takes effect on the next Start, exactly like
        every other Simulation-tab setting."""
        rf = ttk.LabelFrame(parent, text="Radius")
        rf.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 6))
        rf.columnconfigure(1, weight=1)
        self.buddha_blur_radius_scale = self.add_angle_entry(
            rf, 0, "Radius", DEFAULT_BUDDHA_BLUR_RADIUS, on_change=self.on_buddha_blur_changed,
            lo=BUDDHA_BLUR_RADIUS_MIN, hi=BUDDHA_BLUR_RADIUS_MAX, fmt="{:.2f}")
        ttk.Label(rf, foreground="#666", wraplength=RIGHT_W - 70,
                  text="A multiple of the view's smallest dimension (2 units): 0.1 "
                       "means a blur radius of 0.2 units. 0 means no blur at all - "
                       "each point still lands on exactly one pixel, as before."
                  ).grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 6))

        tf = ttk.LabelFrame(parent, text="Blur type")
        tf.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        tf.columnconfigure(0, weight=1)
        self.buddha_blur_type_var = tk.StringVar(value=BUDDHA_BLUR_LABELS[
            DEFAULT_BUDDHA_BLUR_TYPE])
        self.buddha_blur_type_combo = ttk.Combobox(
            tf, textvariable=self.buddha_blur_type_var, values=BUDDHA_BLUR_LABELS,
            state="readonly", width=22)
        self.buddha_blur_type_combo.grid(row=0, column=0, sticky="w", padx=8, pady=(6, 4))
        self.buddha_blur_type_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self.on_buddha_blur_changed())
        ttk.Label(tf, foreground="#666", wraplength=RIGHT_W - 70,
                  text="All four share the radius above. Gaussian treats it as a "
                       "standard deviation, so it can affect pixels somewhat beyond "
                       "that distance; the other three stop exactly at it. Bokeh is "
                       "a uniform disc; catadioptric bokeh is a ring, like a mirror "
                       "lens's characteristic donut-shaped out-of-focus highlights."
                  ).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 6))

    def on_buddha_blur_changed(self):
        """The Buddha blur radius or type changed - a Simulation-wide
        setting, so (like the iteration limit or the shape) this only marks
        Start as available; it doesn't touch the live preview until Start
        is actually pressed."""
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_state_text()
        self.refresh_timeline_diffs()

    def build_decoration_controls(self, parent):
        """Checkboxes for the things drawn on / over the image, in two groups."""
        groups = [("Affect the output image", DECO_IMAGE,
                   "Drawn into the image itself, so they also appear in saved "
                   "files and keyframes."),
                  ("Preview only (never saved)", DECO_PREVIEW,
                   "Drawn over the image on screen while you hover or edit "
                   "settings; saved files and keyframes never contain them.")]
        for g, (title, items, note) in enumerate(groups):
            frame = ttk.LabelFrame(parent, text=title)
            frame.grid(row=g, column=0, sticky="ew", padx=6,
                       pady=(8 if g == 0 else 0, 6))
            frame.columnconfigure(0, weight=1)
            for r, (key, label, default) in enumerate(items):
                var = tk.BooleanVar(value=default)
                check = ttk.Checkbutton(
                    frame, text=label, variable=var,
                    command=lambda k=key: self.on_decoration_changed(k))
                check.grid(row=r, column=0, sticky="w", padx=10, pady=2)
                setattr(self, f"deco_{key}_check", check)
                self.deco[key] = var
            ttk.Label(frame, text=note, foreground="#666",
                      wraplength=RIGHT_W - 70).grid(
                row=len(items), column=0, sticky="w", padx=10, pady=(2, 6))

    def build_obstacles_controls(self, parent):
        """Obstacle list (scrollable, with "new..." - a button opening a menu of the shape kinds -
        Dupe and Del stacked to its left) plus an editor for whichever obstacle is
        selected: its shape's name (fixed when it was made), a rotation, the
        controls of its own shape kind (one group per kind, built from that
        kind's PARAMS table; only the selected obstacle's is shown), a polar
        position (r, theta) with an x/y offset, incoming and outgoing ray-effect
        dropdowns, and a "fire rays inside" checkbox. Every one of an obstacle's
        settings can be animated between keyframes, just like the main
        Simulation/Fuzzing/Coloration settings. `self.obs_widgets` maps
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

    def purge_keyframe_settings(self, prefix):
        """Removes a just-deleted obstacle's or view's settings from every
        keyframe right away (see remove_keyframe_keys), so no keyframe is ever
        left holding settings for something that no longer exists - they
        would otherwise be invisible in the Timeline's settings list, and so
        impossible for the user to remove."""
        remove_keyframe_keys(self.keyframes, prefix)
        self.refresh_timeline_diffs()

    def build_info_quadrant(self):
        """Bottom left: the previewed ray, in Numbers and Graphics tabs."""
        info = ttk.LabelFrame(self.root, text="Ray preview", width=SIZE, height=BOTTOM_H)
        info.grid(row=1, column=0, sticky="nsew",
                  padx=(PAD, PAD // 2), pady=(PAD // 2, PAD))
        info.grid_propagate(False)
        info.columnconfigure(0, weight=1)
        info.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(info)
        notebook.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        numbers_tab = ttk.Frame(notebook)
        graphics_tab = ttk.Frame(notebook)
        notebook.add(numbers_tab, text="Numbers")
        notebook.add(graphics_tab, text="Graphics")
        numbers_tab.columnconfigure(0, weight=1)
        graphics_tab.columnconfigure(0, weight=1)

        self.build_numbers_tab(numbers_tab)
        self.build_graphics_tab(graphics_tab)

    def build_numbers_tab(self, tab):
        """Every detail and metric of the previewed ray."""
        self.info_placeholder = ttk.Label(
            tab, foreground="#666",
            text="Hover the mouse over a pixel inside the shape to preview its "
                 "ray. Click to pin it there and free up the mouse.")
        self.info_placeholder.grid(row=0, column=0, sticky="nw", padx=6, pady=8)

        rows = ttk.Frame(tab)
        rows.grid(row=0, column=0, sticky="nw", padx=2, pady=2)
        self.info_rows = rows
        right = ttk.Frame(rows)
        right.grid(row=0, column=0, sticky="nw", padx=(4, 8))
        left = ttk.Frame(rows)
        left.grid(row=0, column=1, sticky="nw")

        fields = [("Start", "start"), ("Outcome", "outcome"),
                  ("Bounces", "bounces"), ("Final", "final"),
                  ("Length", "length"), ("Last seg", "lastseg"),
                  ("Self-loop", "loop"), ("Components", "components"),
                  ("Color", "color")]
        self.info_vars = {}
        for r, (label, key) in enumerate(fields):
            ttk.Label(left, text=label + ":", foreground="#666").grid(
                row=r, column=0, sticky="nw", padx=(0, 8))
            var = tk.StringVar()
            self.info_vars[key] = var
            if key == "color":
                holder = ttk.Frame(left)
                holder.grid(row=r, column=1, sticky="w")
                self.swatch = tk.Label(holder, width=4, bd=1, relief="solid")
                self.swatch.pack(side="left")
                ttk.Label(holder, textvariable=var, wraplength=130).pack(
                    side="left", padx=8)
            else:
                ttk.Label(left, textvariable=var, wraplength=205).grid(
                    row=r, column=1, sticky="w")

        # Metric list: name | value | "-> R" tag for channels currently using it
        self.stat_vars = []
        self.stat_tags = []
        for s in range(N_STATS):
            ttk.Label(right, text=STAT_SHORT[s], font=self.small_font).grid(
                row=s, column=0, sticky="w", padx=(0, 10))
            var = tk.StringVar()
            self.stat_vars.append(var)
            ttk.Label(right, textvariable=var, font=self.small_font).grid(
                row=s, column=1, sticky="e")
            tag = ttk.Label(right, text="", font=self.small_font, width=8)
            tag.grid(row=s, column=2, sticky="w", padx=(6, 0))
            self.stat_tags.append(tag)
        rows.grid_remove()
        self.update_stat_tags()

    def build_graphics_tab(self, tab):
        """A plot of the previewed ray's segment lengths, earliest on the left."""
        legend = ttk.Frame(tab)
        legend.grid(row=0, column=0, sticky="w", padx=6, pady=(8, 2))
        ttk.Label(legend, text="Every segment, earliest to latest, as a bar "
                              "(one unit tall; longer segments are truncated). "
                              "Color marks each end's kind: "
                  ).pack(side="left")
        for rgb, text in [(KIND_COLOR[KIND_START], "start/distance-cut"),
                          (KIND_COLOR[KIND_BOUNDARY], "boundary bounce"),
                          (KIND_COLOR[KIND_SELF], "self bounce"),
                          (KIND_COLOR[KIND_OBSTACLE_TERM], "edge termination"),
                          (KIND_COLOR[KIND_ESCAPED], "escape")]:
            sw = tk.Label(legend, text="  ", bg=lerp_color(rgb, rgb, 0.0))
            sw.pack(side="left", padx=(4, 2))
            ttk.Label(legend, text=text).pack(side="left", padx=(0, 8))
        self.seg_canvas = tk.Canvas(tab, width=SEG_CANVAS_W, height=SEG_CANVAS_H,
                                    bd=0, highlightthickness=1,
                                    highlightbackground="#444", bg="black")
        self.seg_canvas.grid(row=1, column=0, sticky="w", padx=6, pady=(2, 8))

    def draw_segments(self, pts, kinds, n):
        """Redraw the segment-lengths graphic for the ray pts[0..n-1], bars
        growing downward from the top, gradient-colored by each end's kind."""
        cv = self.seg_canvas
        cv.delete("all")
        nseg = n - 1
        if nseg <= 0:
            return
        px_per_unit = SEG_CANVAS_H              # canvas height = 1 world unit
        colw = SEG_CANVAS_W / nseg
        halfw = max(colw / 2.0, 0.75)
        for k in range(nseg):
            length = float(np.hypot(pts[k + 1, 0] - pts[k, 0], pts[k + 1, 1] - pts[k, 1]))
            h = min(length * px_per_unit, SEG_CANVAS_H)
            if h <= 0:
                continue
            x = (k + 0.5) * colw
            top_rgb = KIND_COLOR[float(kinds[k])]
            bot_rgb = KIND_COLOR[float(kinds[k + 1])]
            if top_rgb == bot_rgb:
                cv.create_rectangle(x - halfw, 0, x + halfw, h,
                                    fill=lerp_color(top_rgb, top_rgb, 0.0), width=0)
                continue
            steps = max(2, min(64, int(h)))
            for s in range(steps):
                t0 = s / steps
                t1 = (s + 1) / steps
                color = lerp_color(top_rgb, bot_rgb, (t0 + t1) / 2.0)
                cv.create_rectangle(x - halfw, t0 * h, x + halfw, t1 * h,
                                    fill=color, width=0)

    def build_status_quadrant(self):
        """Bottom middle: the Export designer (whether to include one view
        or all of them in exported output), then the Export controls (what
        to export, its format, and the Export / Export As buttons)."""
        container = ttk.Frame(self.root, width=RIGHT_W, height=BOTTOM_H)
        container.grid(row=1, column=1, sticky="nsew",
                       padx=(PAD // 2, PAD // 2), pady=(PAD // 2, PAD))
        container.grid_propagate(False)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        vcmd_wide = (self.root.register(self.validate_bitrate_digits), "%P")

        # ---- Export designer: which view(s) end up in the output ----
        designer = ttk.LabelFrame(container, text="Export designer")
        designer.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.include_all_views_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(designer, text="Include all views",
                        variable=self.include_all_views_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))

        color_info_row = ttk.Frame(designer)
        color_info_row.grid(row=1, column=0, sticky="w", padx=8, pady=2)
        ttk.Label(color_info_row, text="Color info location:").pack(side="left", padx=(0, 6))
        self.color_info_location_var = tk.StringVar(value=COLOR_INFO_LOCATIONS[0])
        ttk.Combobox(color_info_row, textvariable=self.color_info_location_var,
                    values=COLOR_INFO_LOCATIONS, state="readonly", width=13).pack(side="left")

        figure_info_row = ttk.Frame(designer)
        figure_info_row.grid(row=2, column=0, sticky="w", padx=8, pady=(2, 6))
        ttk.Label(figure_info_row, text="Figure settings location:").pack(
            side="left", padx=(0, 6))
        self.figure_info_location_var = tk.StringVar(value=FIGURE_INFO_LOCATIONS[0])
        ttk.Combobox(figure_info_row, textvariable=self.figure_info_location_var,
                    values=FIGURE_INFO_LOCATIONS, state="readonly", width=13).pack(side="left")

        # ---- Export: what to export, its format, and the buttons ----
        exp = ttk.LabelFrame(container, text="Export")
        exp.grid(row=1, column=0, sticky="nsew")
        exp.columnconfigure(0, weight=1)

        # Export buttons are packed at the bottom FIRST, so they always get
        # their space no matter how tall everything else below ends up
        # being (a fixed-height panel plus content that can grow otherwise
        # risks the classic "everything after the tall content gets
        # squeezed out" grid/pack failure mode). Disabled (with a tooltip
        # explaining why) until a figure has actually finished generating -
        # see update_buttons().
        self.save_frame = ttk.Frame(exp)
        self.save_frame.columnconfigure((0, 1), weight=1)
        self.save_btn = ttk.Button(self.save_frame, text="Export", command=self.save)
        self.save_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.save_as_btn = ttk.Button(self.save_frame, text="Export As...",
                                      command=self.save_as)
        self.save_as_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.save_frame.pack(side="bottom", fill="x", padx=10, pady=(6, 8))
        self.save_btn_tooltip = Tooltip(self.save_btn)
        self.save_as_btn_tooltip = Tooltip(self.save_as_btn)

        # Everything else stacks above it, top to bottom
        rest = ttk.Frame(exp)
        rest.pack(side="top", fill="both", expand=True)
        rest.columnconfigure(0, weight=1)

        # ---- what to export and in what format (both apply whichever
        # source is chosen) ----
        source_row = ttk.Frame(rest)
        source_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 1))
        ttk.Label(source_row, text="Image to export:").pack(side="left", padx=(0, 6))
        self.save_source_var = tk.StringVar(value=SAVE_SOURCE_OPTIONS[0])
        save_source_combo = ttk.Combobox(
            source_row, textvariable=self.save_source_var, values=SAVE_SOURCE_OPTIONS,
            state="readonly", width=19)
        save_source_combo.pack(side="left")
        save_source_combo.bind("<<ComboboxSelected>>", self.on_save_source_changed)

        fmt_row = ttk.Frame(rest)
        fmt_row.grid(row=1, column=0, sticky="ew", padx=10, pady=1)
        ttk.Label(fmt_row, text="Format:").pack(side="left", padx=(0, 4))
        self.save_format_var = tk.StringVar(value=SAVE_NEW_FORMATS[0])
        format_combo = ttk.Combobox(fmt_row, textvariable=self.save_format_var,
                                    values=SAVE_NEW_FORMATS, state="readonly", width=6)
        format_combo.pack(side="left", padx=(0, 10))
        format_combo.bind("<<ComboboxSelected>>", self.on_save_format_changed)
        self.save_jxl_controls = JxlQualityControls(fmt_row)
        self.save_jxl_mode_var = self.save_jxl_controls.mode_var
        self.save_jxl_quality_scale = self.save_jxl_controls.scale
        self.on_save_format_changed()          # (the default format is jxl)

        # "New image" only: its own resolution (an existing preview already
        # has one, so this only makes sense when generating a fresh image) -
        # this is per view: with "Include all views" on, each view is
        # rendered at this resolution and they're placed side by side, so
        # the final output is wider than this.
        self.save_new_frame = ttk.Frame(rest)
        self.save_new_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(1, 8))
        dims_row = ttk.Frame(self.save_new_frame)
        dims_row.grid(row=0, column=0, sticky="w")
        self.save_new_w_var = tk.StringVar(value="1080")
        self.save_new_h_var = tk.StringVar(value="1080")
        for label, var in [("Width (px):", self.save_new_w_var),
                           ("Height (px):", self.save_new_h_var)]:
            ttk.Label(dims_row, text=label).pack(side="left", padx=(0, 4))
            entry = ttk.Entry(dims_row, textvariable=var, width=6, justify="right",
                              validate="key", validatecommand=vcmd_wide)
            entry.pack(side="left", padx=(0, 10))
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, lambda _e, var=var: self.commit_resolution_var(var))
        self.save_new_frame.grid_remove()

    def build_timeline_panel(self):
        """Top right: the keyframe list, and beside it, a
        list of every setting, showing how each relates to the selected
        keyframe, grouped by tab."""
        tl = ttk.LabelFrame(self.root, text="Timeline", width=TIMELINE_W, height=SIZE)
        tl.grid(row=0, column=2, sticky="nsew", padx=(PAD // 2, PAD), pady=(PAD, PAD // 2))
        tl.grid_propagate(False)
        tl.columnconfigure(0, weight=1)
        tl.columnconfigure(1, weight=1)
        tl.rowconfigure(2, weight=1)

        new_row = ttk.Frame(tl)
        new_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        new_row.columnconfigure((0, 1), weight=1)
        ttk.Button(new_row, text="New keyframe (from current settings)",
                  command=self.new_keyframe).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(new_row, text="New keyframe (empty)",
                  command=self.new_keyframe_empty).grid(row=0, column=1, sticky="ew",
                                                        padx=(2, 0))

        ttk.Label(tl, text="Keyframes:", foreground="#666").grid(
            row=1, column=0, sticky="w", padx=8)

        diff_header = ttk.Frame(tl)
        diff_header.grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Label(diff_header, text="Settings:", foreground="#666").pack(side="left")
        self.diff_delete_all_btn = ttk.Button(diff_header, text="Delete all",
                                              command=self.delete_all_diffs)
        self.diff_delete_all_btn.pack(side="right")
        self.diff_save_all_btn = ttk.Button(diff_header, text="Save all modified",
                                            command=self.save_modified_settings)
        self.diff_save_all_btn.pack(side="right", padx=(0, 4))

        # Keyframe list (left column): one label per keyframe, click to select
        list_container = ttk.Frame(tl)
        list_container.grid(row=2, column=0, sticky="nsew", padx=8, pady=(2, 4))
        list_container.columnconfigure(0, weight=1)
        list_container.rowconfigure(0, weight=1)
        kf_canvas = tk.Canvas(list_container, highlightthickness=0)
        vsb = ttk.Scrollbar(list_container, orient="vertical", command=kf_canvas.yview)
        kf_canvas.configure(yscrollcommand=vsb.set)
        kf_canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.kf_list_frame = ttk.Frame(kf_canvas)
        kf_window = kf_canvas.create_window((0, 0), window=self.kf_list_frame, anchor="nw")
        self.kf_list_frame.bind(
            "<Configure>", lambda _e: kf_canvas.configure(scrollregion=kf_canvas.bbox("all")))
        kf_canvas.bind(
            "<Configure>", lambda e: kf_canvas.itemconfigure(kf_window, width=e.width))
        for background in (kf_canvas, self.kf_list_frame):      # empty background deselects
            background.bind("<Button-1>", lambda _e: self.deselect_keyframe())

        # Settings list (right column): every setting, grouped by tab
        diff_container = ttk.Frame(tl)
        diff_container.grid(row=2, column=1, sticky="nsew", padx=8, pady=(2, 4))
        diff_container.columnconfigure(0, weight=1)
        diff_container.rowconfigure(1, weight=1)
        self.diff_presence_var = tk.StringVar(value=DIFF_PRESENCE_CHOICES[-1])
        filters = ttk.Frame(diff_container)
        filters.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))
        ttk.Label(filters, text="Show", font=self.small_font).pack(side="left")
        combo = ttk.Combobox(filters, textvariable=self.diff_presence_var,
                             values=DIFF_PRESENCE_CHOICES, state="readonly", width=17)
        combo.pack(side="left", padx=(4, 0))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_timeline_diffs())
        diff_canvas = self.diff_canvas = tk.Canvas(diff_container, highlightthickness=0)
        dvsb = ttk.Scrollbar(diff_container, orient="vertical", command=diff_canvas.yview)
        diff_canvas.configure(yscrollcommand=dvsb.set)
        diff_canvas.grid(row=1, column=0, sticky="nsew")
        dvsb.grid(row=1, column=1, sticky="ns")
        self.diff_list_frame = ttk.Frame(diff_canvas)
        diff_window = diff_canvas.create_window((0, 0), window=self.diff_list_frame, anchor="nw")
        self.diff_list_frame.bind(
            "<Configure>", lambda _e: diff_canvas.configure(scrollregion=diff_canvas.bbox("all")))
        diff_canvas.bind(
            "<Configure>", lambda e: diff_canvas.itemconfigure(diff_window, width=e.width))

        det = ttk.LabelFrame(tl, text="Selected keyframe")
        det.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 8))
        det.columnconfigure(1, weight=1)
        self.kf_detail_label_var = tk.StringVar(value="(none selected)")
        ttk.Label(det, textvariable=self.kf_detail_label_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 4))

        ttk.Label(det, text="Frame index:").grid(
            row=1, column=0, sticky="w", padx=8, pady=2)
        self.kf_frame_var = tk.StringVar(value="")
        vcmd = (self.root.register(self.validate_digits), "%P")
        self.kf_frame_entry = ttk.Entry(det, textvariable=self.kf_frame_var, width=8,
                                        justify="right", validate="key",
                                        validatecommand=vcmd)
        self.kf_frame_entry.grid(row=1, column=1, sticky="w", padx=(0, 8), pady=2)
        self.kf_frame_entry.bind("<FocusOut>", self.on_kf_frame_commit)
        self.kf_frame_entry.bind("<Return>", self.on_kf_frame_commit)
        self.kf_frame_entry.bind("<KP_Enter>", self.on_kf_frame_commit)

        btnrow = ttk.Frame(det)
        btnrow.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 8))
        self.kf_split_btn = ttk.Button(btnrow, text="Split",
                                       command=self.split_selected_keyframe)
        self.kf_split_btn.pack(side="left", padx=(0, 4))
        self.kf_delete_btn = ttk.Button(btnrow, text="Delete",
                                        command=self.delete_selected_keyframe)
        self.kf_delete_btn.pack(side="left")

        ttk.Label(tl, foreground="#666", wraplength=TIMELINE_W - 30,
                  text="Selecting a keyframe loads its settings (and the settings "
                       "its neighbors animate through it) into the controls; a setting "
                       "you then change gets a blue outline until you save it or "
                       "deselect (click the empty part of the keyframe list). Only float "
                       "values are interpolated (linearly) between keyframes. Other "
                       "values are held until the next keyframe."
                 ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))

        self.update_kf_detail_panel()
        self.rebuild_keyframe_list()
        self.root.after(MODIFIED_POLL_MS, self.poll_modified)
        self.refresh_timeline_diffs()

    def build_video_panel(self):
        """Bottom right: resolution/framerate/quality/bitrate settings, a
        preview of the latest rendered frame, and the render button."""
        vp = ttk.LabelFrame(self.root, text="Video", width=TIMELINE_W, height=BOTTOM_H)
        vp.grid(row=1, column=2, sticky="nsew", padx=(PAD // 2, PAD), pady=(PAD // 2, PAD))
        vp.grid_propagate(False)
        vp.columnconfigure(0, weight=1)
        vp.columnconfigure(1, weight=1)
        vcmd = (self.root.register(self.validate_digits), "%P")
        vcmd_wide = (self.root.register(self.validate_bitrate_digits), "%P")

        # ---- Render settings (left): resolution and which frames to render ----
        render_settings_frame = ttk.LabelFrame(vp, text="Render settings")
        render_settings_frame.grid(row=0, column=0, sticky="new", padx=(8, 4), pady=(8, 6))
        render_settings_frame.columnconfigure(0, weight=1)

        dims_row = ttk.Frame(render_settings_frame)
        dims_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
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

        self.skip_first_frame_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(render_settings_frame, text="Skip first frame",
                       variable=self.skip_first_frame_var).grid(
            row=1, column=0, sticky="w", padx=8, pady=(2, 6))

        # ---- Encoding (right): format, bitrate, framerate, downsampling ----
        encoding_frame = ttk.LabelFrame(vp, text="Encoding")
        encoding_frame.grid(row=0, column=1, sticky="new", padx=(4, 8), pady=(8, 6))
        encoding_frame.columnconfigure(0, weight=1)

        fmt_row = ttk.Frame(encoding_frame)
        fmt_row.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        self.fps_var = tk.StringVar(value="30")
        ttk.Label(fmt_row, text="FPS:").pack(side="left", padx=(0, 4))
        ttk.Entry(fmt_row, textvariable=self.fps_var, width=5, justify="right",
                 validate="key", validatecommand=vcmd_wide).pack(side="left", padx=(0, 10))
        ttk.Label(fmt_row, text="Format:").pack(side="left", padx=(0, 4))
        self.export_format_var = tk.StringVar(value="mp4 (h264)")
        format_combo = ttk.Combobox(fmt_row, textvariable=self.export_format_var,
                                    values=VIDEO_FORMATS,
                                    state="readonly", width=6)
        format_combo.pack(side="left")
        format_combo.bind("<<ComboboxSelected>>", self.on_export_format_changed)

        mode_line = ttk.Frame(encoding_frame)
        mode_line.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
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

        downsample_row = ttk.Frame(encoding_frame)
        downsample_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 6))
        ttk.Label(downsample_row, text="Downsample:").pack(side="left", padx=(0, 4))
        self.downsample_factor_var = tk.StringVar(value=DOWNSAMPLE_FACTORS[0])
        downsample_combo = ttk.Combobox(
            downsample_row, textvariable=self.downsample_factor_var,
            values=DOWNSAMPLE_FACTORS, state="readonly", width=9)
        downsample_combo.pack(side="left", padx=(0, 10))
        downsample_combo.bind("<<ComboboxSelected>>", self.on_downsample_factor_changed)
        self.downsample_method_label = ttk.Label(downsample_row, text="Method:")
        self.downsample_method_label.pack(side="left", padx=(0, 4))
        self.downsample_method_var = tk.StringVar(value=DOWNSAMPLE_METHODS[0])
        self.downsample_method_combo = ttk.Combobox(
            downsample_row, textvariable=self.downsample_method_var,
            values=DOWNSAMPLE_METHODS, state="readonly", width=7)
        self.downsample_method_combo.pack(side="left")

        self.on_bitrate_mode_changed()
        self.on_export_format_changed()
        self.on_downsample_factor_changed()

        note = "" if external.ffmpeg is not None else ("\nffmpeg-python was not found - install "
                                              "it with:  pip install ffmpeg-python")
        ttk.Label(vp, foreground="#666", wraplength=TIMELINE_W - 30,
                  text=f"Frames are written as PNGs to exports/{os.path.basename(paths.TEMP_DIR)}/ "
                       "next to this program at the resolution above (same world "
                       "area as the viewport), then combined with ffmpeg into an "
                       ".mp4, .gif or .apng saved in exports/. Downsampling "
                       "shrinks only that combined video, not the PNG sequence. "
                       "Re-encode combines whatever frames are already in TEMP with "
                       "the current encoding settings (ignoring width/height - it "
                       "uses the existing frames' own resolution). Nothing is ever "
                       f"deleted automatically - clear TEMP yourself between "
                       f"renders.{note}"
                 ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        # Preview of the latest rendered frame, beside the render controls
        lower = ttk.Frame(vp)
        lower.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=8, pady=(2, 8))
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)
        vp.rowconfigure(2, weight=1)

        self.preview_label = tk.Label(lower, bd=1, relief="solid")
        self.preview_label.grid(row=0, column=0, sticky="n", padx=(0, 8))
        self.set_preview_image(None)

        right_col = ttk.Frame(lower)
        right_col.grid(row=0, column=1, sticky="nsew")
        right_col.columnconfigure(0, weight=1)
        right_col.rowconfigure(1, weight=1)

        render_row = ttk.Frame(right_col)
        render_row.grid(row=0, column=0, sticky="ew", pady=(0, 6))
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
        self.video_progress = ttk.Progressbar(render_row, maximum=1, mode="determinate")
        self.video_progress.grid(row=0, column=3, sticky="ew", padx=(3, 0))

        status_frame = ttk.Frame(right_col)
        status_frame.grid(row=1, column=0, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        self.video_status_text = tk.Text(status_frame, height=7, wrap="word",
                                         font=self.small_font, state="disabled",
                                         bd=1, relief="solid")
        status_vsb = ttk.Scrollbar(status_frame, orient="vertical",
                                   command=self.video_status_text.yview)
        self.video_status_text.configure(yscrollcommand=status_vsb.set)
        self.video_status_text.grid(row=0, column=0, sticky="nsew")
        status_vsb.grid(row=0, column=1, sticky="ns")
        self.set_video_status("")

    def on_downsample_factor_changed(self, _event=None):
        """The Method dropdown only matters once downsampling is actually on."""
        if self.downsample_factor_var.get() == DOWNSAMPLE_FACTORS[0]:
            self.downsample_method_label.state(["disabled"])
            self.downsample_method_combo.state(["disabled"])
        else:
            self.downsample_method_label.state(["!disabled"])
            self.downsample_method_combo.state(["!disabled"])

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

    # ---- decorations -------------------------------------------------------- #
    def on_decoration_changed(self, key):
        """A decoration checkbox was toggled."""
        self.hover_px = None                      # redraw any overlay on next hover
        if key == "outline":                      # part of the image itself
            self.draw()
            self.refresh_timeline_diffs()
        elif key == "pending":
            self.update_pending_outline()
        if self.pinned_px is not None:
            self.update_preview(*self.pinned_px)

    # ---- limits: number entries with validation ---------------------------- #
    @staticmethod
    def validate_digits(proposed):
        """Key-by-key check: only plain digits, at most four of them. (Whether
        the value is in range is judged by parse_iter_limit.)"""
        return re.fullmatch(r"[0-9]{0,4}", proposed) is not None

    @staticmethod
    def validate_dist_digits(proposed):
        """Key-by-key check for the distance limit: digits, at most four
        before the point, and an optional decimal point with at most two
        digits after it - so it accepts non-integer distances while typing,
        without ever accepting a character that could never be part of a
        valid one. (Whether the final value is in range is judged by
        parse_dist_limit.)"""
        return re.fullmatch(r"[0-9]{0,4}(\.[0-9]{0,2})?", proposed) is not None

    @staticmethod
    def validate_signed_float(proposed):
        """Key-by-key check for a coordinate: a partially typed number with an
        optional leading '-' and optional exponent, nothing that could never
        become a valid one. Whether it is finite is judged by
        parse_target_coordinate."""
        return re.fullmatch(r"-?[0-9]*\.?[0-9]*(?:[eE][-+]?[0-9]*)?", proposed) is not None

    @staticmethod
    def validate_window_digits(proposed):
        """Key-by-key check for the confinement window: up to three digits."""
        return re.fullmatch(r"[0-9]{0,3}", proposed) is not None

    @staticmethod
    def validate_positive_float(proposed):
        """Key-by-key check for the confinement threshold: a partially typed
        plain or scientific-notation number (no sign at the front - it must be
        above zero - but an exponent may be negative), nothing that could never
        become a valid one. Whether it is actually above zero is judged by
        parse_confine_threshold."""
        return re.fullmatch(r"[0-9]*\.?[0-9]*(?:[eE][-+]?[0-9]*)?", proposed) is not None

    @staticmethod
    def validate_seed_digits(proposed):
        return re.fullmatch(r"[0-9]{0,9}", proposed) is not None

    @staticmethod
    def validate_bitrate_digits(proposed):
        return re.fullmatch(r"[0-9]{0,6}", proposed) is not None

    @staticmethod
    def validate_float_free(proposed):
        """Key-by-key check for an unrestricted float entry (angles): allows
        an optional leading '-' and a partially-typed number, but nothing
        that could never become a valid float."""
        return re.fullmatch(r"-?[0-9]*\.?[0-9]*", proposed) is not None

    def commit_resolution_var(self, var):
        """A resolution field (New image or video width/height) lost focus
        or had Enter pressed: rewrite its text to the rounded value it will
        actually be read as (see parse_resolution_dimension), so what's
        displayed always matches what an export will actually use. Leaves
        invalid text alone - the existing error messaging when the value is
        actually read covers that case."""
        rounded = parse_resolution_dimension(var.get())
        if rounded is not None:
            var.set(str(rounded))

    def on_limit_var_changed(self, *_args):
        """Either entry's text changed: flag bad values, track the last good ones."""
        it = parse_iter_limit(self.iter_var.get())
        ds = parse_dist_limit(self.dist_var.get())
        cw = parse_confine_window(self.confine_window_var.get())
        ct = parse_confine_threshold(self.confine_threshold_var.get())
        if it is not None:
            self.last_iter = it
        if ds is not None:
            self.last_dist = ds
        if cw is not None:
            self.last_confine_window = cw
        if ct is not None:
            self.last_confine_threshold = ct
        self.iter_entry.configure(style="TEntry" if it is not None else "Invalid.TEntry")
        self.dist_entry.configure(style="TEntry" if ds is not None else "Invalid.TEntry")
        self.confine_window_entry.configure(
            style="TEntry" if cw is not None else "Invalid.TEntry")
        self.confine_threshold_entry.configure(
            style="TEntry" if ct is not None else "Invalid.TEntry")
        emit_msgs = []
        for field in EMIT_TARGET_FIELDS:
            value = parse_target_coordinate(getattr(self, f"emit_target_{field}_var").get())
            if value is not None:
                setattr(self, f"last_emit_target_{field}", value)
            getattr(self, f"emit_target_{field}_entry").configure(
                style="TEntry" if value is not None else "Invalid.TEntry")
            if value is None and not emit_msgs:
                emit_msgs.append("Target position values must be numbers (any size, "
                                 "positive or negative, e.g. -0.5 or 3e8).")
        self.emit_msg_var.set(" ".join(emit_msgs))
        confine_msgs = []
        if cw is None:
            confine_msgs.append(f"Window size must be a whole number from "
                                f"{CONFINE_WINDOW_MIN} to {CONFINE_WINDOW_MAX}.")
        if ct is None:
            confine_msgs.append("Threshold must be a number greater than 0 "
                                "(any size, e.g. 0.004 or 1e-9).")
        self.confine_msg_var.set(" ".join(confine_msgs))
        msgs = []
        if it is None:
            msgs.append(f"Iteration limit must be a whole number from "
                        f"{ITER_LIMIT_MIN} to {ITER_LIMIT_MAX}.")
        if ds is None:
            msgs.append(f"Distance limit must be a whole number from "
                        f"{DIST_LIMIT_MIN} to {DIST_LIMIT_MAX} (0 = no limit).")
        self.limit_msg_var.set(" ".join(msgs))
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_state_text()
        self.refresh_timeline_diffs()

    def on_limit_commit(self, _event=None):
        """Enter pressed or focus left an entry: tidy the text, and put back the
        last valid value if the field holds something invalid."""
        for var, last, parser in (
                (self.iter_var, self.last_iter, parse_iter_limit),
                (self.dist_var, self.last_dist, parse_dist_limit),
                (self.confine_window_var, self.last_confine_window, parse_confine_window),
                (self.confine_threshold_var, self.last_confine_threshold,
                 parse_confine_threshold),
                *[(getattr(self, f"emit_target_{field}_var"),
                   getattr(self, f"last_emit_target_{field}"), parse_target_coordinate)
                  for field in EMIT_TARGET_FIELDS]):
            value = parser(var.get())
            var.set(str(last if value is None else value))

    def on_seg_mult_changed(self):
        """The segment length multiplier entry committed a new value (already
        parsed and clamped into range by its FloatEntryHandle)."""
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_state_text()
        self.refresh_timeline_diffs()

    def on_seed_var_changed(self, *_args):
        seed = parse_seed(self.seed_var.get())
        if seed is not None:
            self.last_seed = seed
        self.seed_entry.configure(style="TEntry" if seed is not None else "Invalid.TEntry")
        self.on_fuzz_changed()

    def on_seed_commit(self, _event=None):
        seed = parse_seed(self.seed_var.get())
        self.seed_var.set(str(self.last_seed if seed is None else seed))

    def randomize_seed(self):
        self.seed_var.set(str(random.randint(0, 999999999)))
        self.on_seed_commit()

    # ---- change handling (obstacle shape parameters and the like) -------------- #
    def on_shape_param_changed(self, _value=None):
        self.invalidate_preview()
        self.schedule_outline_refresh()           # live outlines before first Start
        self.update_pending_outline()             # dashed outlines afterwards
        self.update_buttons()
        self.refresh_state_text()
        self.refresh_timeline_diffs()

    # ---- fuzzing controls ---------------------------------------------------- #
    def read_fuzz_controls(self):
        """FuzzParams from the current widget values."""
        seed = parse_seed(self.seed_var.get())
        return FuzzParams(
            enabled=bool(self.fuzz_enabled_var.get()),
            emit_jitter=round(float(self.emit_jitter_scale.get()), 2),
            bnd_offset=round(float(self.bnd_offset_scale.get()), 2),
            bnd_jitter=round(float(self.bnd_jitter_scale.get()), 2),
            bnd_mult=round(float(self.bnd_mult_scale.get()), 3),
            self_offset=round(float(self.self_offset_scale.get()), 2),
            self_jitter=round(float(self.self_jitter_scale.get()), 2),
            self_mult=round(float(self.self_mult_scale.get()), 3),
            seed=self.last_seed if seed is None else seed)

    def update_fuzz_enabled_state(self):
        """Grey out every Fuzzing-tab control except the Enable checkbox itself
        whenever fuzzing is disabled."""
        enabled = self.fuzz_enabled_var.get()
        tk_state = tk.NORMAL if enabled else tk.DISABLED
        for w in self.fuzz_tk_widgets:
            w.configure(state=tk_state)
        ttk_state = ["!disabled"] if enabled else ["disabled"]
        for w in self.fuzz_ttk_widgets:
            w.state(ttk_state)

    def on_fuzz_changed(self, *_args):
        self.update_fuzz_enabled_state()
        self.invalidate_preview()
        self.update_buttons()
        self.refresh_state_text()
        self.refresh_timeline_diffs()

    def schedule_outline_refresh(self):
        """Coalesce rapid slider events into one outline redraw."""
        if self.outline_job is None:
            self.outline_job = self.root.after_idle(self.refresh_outline)

    def refresh_outline(self):
        self.outline_job = None
        if self.generated_settings is None:       # nothing generated yet: live preview
            self.draw()

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
                self.canvas.create_line(*o.outline_coords(),
                                        fill=PENDING_OUTLINE, dash=(5, 4), tags="pending")

    # ---- settings, Start and Revert ---------------------------------------- #
    def current_ray_settings(self):
        """Every simulation control's value, or None if a limit entry is invalid."""
        it = parse_iter_limit(self.iter_var.get())
        ds = parse_dist_limit(self.dist_var.get())
        cw = parse_confine_window(self.confine_window_var.get())
        ct = parse_confine_threshold(self.confine_threshold_var.get())
        target = [parse_target_coordinate(getattr(self, f"emit_target_{field}_var").get())
                  for field in EMIT_TARGET_FIELDS]
        if it is None or ds is None or cw is None or ct is None or None in target:
            return None
        sm = self.seg_mult_scale.get()
        blur = BuddhaBlurSettings(radius=self.buddha_blur_radius_scale.get(),
                                  type=BUDDHA_BLUR_LABELS.index(self.buddha_blur_type_var.get()))
        return RaySettings(max_iter=it, max_dist=ds, seg_mult=sm, fuzz=self.read_fuzz_controls(),
                        buddha_blur=blur, confine_window=cw,
                        confine_method=CONFINE_METHOD_NAMES.index(
                            self.confine_method_var.get()),
                        confine_threshold=ct,
                        emit_mode=EMIT_MODE_NAMES.index(self.emit_mode_var.get()),
                        emit_target_r=target[0], emit_target_theta=target[1],
                        emit_target_cx=target[2], emit_target_cy=target[3],
                        emit_offset=round(float(self.emit_offset_scale.get()), 2))

    def update_buttons(self):
        """Start: enabled if the entries are valid and either nothing has been
        generated yet or the settings (or obstacles) differ from those the
        image was started with. Revert: enabled whenever any simulation
        control differs from the baseline (obstacles are scene furniture, not
        part of Revert). Export/Export As: enabled once a figure has actually
        finished generating (started, not currently running, all rows done) -
        exporting before that would just export the blank background, which
        is never useful, so the button explains what to do instead."""
        cur = self.current_ray_settings()
        start_ok = (self.ready and cur is not None
                    and (self.generated_settings is None
                         or cur != self.generated_settings
                         or obstacles_signature(self.obstacles) != self.gen_obstacles_sig))
        revert_ok = cur is None or cur != self.baseline
        self.start_btn.state(["!disabled"] if start_ok else ["disabled"])
        self.revert_btn.state(["!disabled"] if revert_ok else ["disabled"])
        if hasattr(self, "save_btn"):
            has_image = (self.generated_settings is not None and not self.running
                         and self.rows_done >= SIZE)
            self.save_btn.state(["!disabled"] if has_image else ["disabled"])
            self.save_as_btn.state(["!disabled"] if has_image else ["disabled"])
            tip = ("" if has_image else
                  "Press Start and wait for generation to finish before exporting.")
            self.save_btn_tooltip.update(tip)
            self.save_as_btn_tooltip.update(tip)

    def revert_settings(self):
        """Put every simulation control back to the baseline: its value when the
        program started, or when the last generation was started."""
        for key, value in ray_settings_to_dict(self.baseline).items():
            self.set_setting_control(key, value)
        self.on_limit_var_changed()
        self.on_shape_param_changed()
        self.on_fuzz_changed()

    def refresh_state_text(self):
        """No-op: the status panel no longer shows any status text. Kept
        (rather than removing its several call sites) so nothing else has
        to change."""
        pass

    def update_status(self):
        """No-op: the status panel no longer shows generation progress.
        Kept (rather than removing its several call sites) so nothing else
        has to change."""
        pass

    # ---- color mapping ---------------------------------------------------- #
    def on_mode_changed(self, _event=None):
        """RGB <-> HSV: relabel the table rows and, if the edited view is the
        live one, recolor right away."""
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        view.mode = self.mode_var.get()
        for c, (name, color) in enumerate(MODES[view.mode]):
            self.comp_labels[c].configure(text=name, foreground=color)
        self.map_help_var.set(default_map_help(view.mode))
        self.mode_combo.selection_clear()
        if view is self.live_view():
            self.update_stat_tags()
            self.recolor(0, self.rows_done)
            self.draw()
            if self.pinned_px is not None:
                self.update_preview(*self.pinned_px)
        self.rebuild_view_list()
        self.refresh_timeline_diffs()

    def refresh_row(self, c):
        """Show channel c's Query, Buddha, or Constant controls (whichever
        its Source is set to), and within Query, set up the Metric and Point
        dropdowns to match its Type: point comparison -> the comparison
        metrics plus Point 1/2 dropdowns; other -> the remaining metrics and
        no Point dropdowns. Reflects the selected view (whichever one the
        Color mapping panel is currently showing/editing)."""
        view = self.selected_view_obj()
        src = view.chan_src[c]
        self.query_frames[c].grid_remove()
        self.buddha_frames[c].grid_remove()
        self.constant_frames[c].grid_remove()
        if src == CHAN_SRC_BUDDHA:
            self.buddha_frames[c].grid()
            return
        if src == CHAN_SRC_CONSTANT:
            self.constant_frames[c].grid()
            self.constant_vars[c].set(constant_stat_to_label(view.constant_choice[c]))
            return
        self.query_frames[c].grid()
        self.kind_vars[c].set(KIND_LABELS[view.kinds[c]])
        if view.kinds[c] == KIND_POINT:
            self.map_combos[c].configure(values=POINT_METRIC_LABELS)
            self.map_vars[c].set(STAT_LABELS[view.point_metric[c]])
            self.point1_combos[c].grid()
            self.point2_combos[c].grid()
        else:
            self.map_combos[c].configure(values=OTHER_LABELS)
            self.map_vars[c].set(other_stat_to_label(view.other_metric[c]))
            self.point1_combos[c].grid_remove()
            self.point2_combos[c].grid_remove()

    def update_stat_tags(self):
        """Tag each metric in the preview list, after its value, with the
        channel letter(s) it currently drives, e.g. '-> B'. Always about the
        live view specifically - this describes the top-left preview image,
        not whichever view is being edited."""
        letters = [name[0] for name, _color in MODES[self.live_view().mode]]
        for s in range(N_STATS):
            used = [letters[c] for c in range(3) if self.mapping[c] == s]
            self.stat_tags[s].configure(text="-> " + ", ".join(used) if used else "")

    def on_chan_src_changed(self, c):
        """Source dropdown changed (Query <-> Buddha <-> Constant)."""
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        view.chan_src[c] = CHAN_SRC_LABELS.index(self.chan_src_vars[c].get())
        self.chan_src_combos[c].selection_clear()
        self.refresh_row(c)
        self.apply_mapping_change(c)

    def on_constant_changed(self, c):
        """The "Constant" source's own dropdown changed (Always 0 <-> Always 1)."""
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        view.constant_choice[c] = constant_label_to_stat(self.constant_vars[c].get())
        self.constant_combos[c].selection_clear()
        self.apply_mapping_change(c)

    def on_kind_changed(self, c):
        """Type dropdown changed (point comparison <-> other)."""
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        view.kinds[c] = KIND_LABELS.index(self.kind_vars[c].get())
        self.kind_combos[c].selection_clear()
        self.refresh_row(c)
        self.apply_mapping_change(c)

    def on_metric_selected(self, c):
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        label = self.map_vars[c].get()
        if view.kinds[c] == KIND_POINT:
            view.point_metric[c] = POINT_STATS[POINT_METRIC_LABELS.index(label)]
        else:
            view.other_metric[c] = other_label_to_stat(label)
        self.apply_mapping_change(c)

    def on_point_selected(self, c):
        if self._color_editor_loading:
            return
        self.apply_mapping_change(c)

    def on_buddha_points_changed(self, c):
        """One of channel c's 8 Buddha point checkboxes was toggled."""
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        view.buddha_points[c] = tuple(v.get() for v in self.buddha_point_vars[c])
        self.apply_mapping_change(c)

    def on_buddha_compression_changed(self, c):
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        view.buddha_compression[c] = BUDDHA_COMPRESSION_LABELS.index(
            self.buddha_compression_vars[c].get())
        self.buddha_compression_combos[c].selection_clear()
        self.apply_mapping_change(c)

    def on_buddha_brightness_changed(self, c):
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        view.buddha_brightness[c] = BUDDHA_BRIGHTNESS_OPTIONS.index(
            self.buddha_brightness_vars[c].get())
        self.buddha_brightness_combos[c].selection_clear()
        self.apply_mapping_change(c)

    def apply_mapping_change(self, c):
        """Channel c's Source, Type, Metric, Point 1/2, Buddha points,
        Buddha compression or Buddha brightness changed, for whichever view
        is selected: update its own data and the panel's help text always;
        if the edited view is also the live one, additionally update
        everything the live preview depends on (self.mapping is always the
        LIVE view's mapping specifically, so it's only touched in that
        case - the help text below is computed fresh from `view` instead,
        so it's correct regardless of which view is being edited)."""
        if self._color_editor_loading:
            return
        view = self.selected_view_obj()
        if view is None:
            return
        view.point1_choice = [POINT_LABELS.index(v.get()) for v in self.point1_vars]
        view.point2_choice = [POINT_LABELS.index(v.get()) for v in self.point2_vars]
        view_mapping = make_mapping(view.kinds, view.point_metric, view.other_metric,
                                    view.chan_src, view.constant_choice)

        name = MODES[view.mode][c][0]
        if view.chan_src[c] == CHAN_SRC_BUDDHA:
            chosen = [BUDDHA_POINT_LABELS[i] for i, v in enumerate(view.buddha_points[c]) if v]
            comp = BUDDHA_COMPRESSION_LABELS[view.buddha_compression[c]]
            bright = BUDDHA_BRIGHTNESS_OPTIONS[view.buddha_brightness[c]]
            if chosen:
                text = (f"{name}: Buddha coloration - a visit count builds up at "
                       f"whichever pixel each ray's {', '.join(chosen).lower()} lands "
                       f"in, compressed to 0..255 with {comp.lower()}"
                       + (f", boosted {bright}." if bright != "1x" else "."))
            else:
                text = f"{name}: Buddha coloration, but no points are selected - always 0."
        else:
            stat = view_mapping[c]
            if stat == NONE_ZERO:
                text = f"{name}: always 0."
            elif stat == NONE_ONE:
                text = f"{name}: always 1."
            else:
                text = f"{name}: {STAT_DESCRIPTIONS[stat]}"
                if stat in POINT_STATS:
                    text += (f" Point 1: {POINT_DESCRIPTIONS[view.point1_choice[c]]} "
                            f"Point 2: {POINT_DESCRIPTIONS[view.point2_choice[c]]}"
                            + POINT_NOTE)
        self.map_help_var.set(text)
        self.map_combos[c].selection_clear()
        self.point1_combos[c].selection_clear()
        self.point2_combos[c].selection_clear()

        if view is self.live_view():
            self.mapping = view_mapping
            self.update_stat_tags()
            if view.chan_src[c] == CHAN_SRC_BUDDHA:
                self.rebuild_buddha_counts()   # only the edited channel's Buddha
                                               # state could possibly need this -
                                               # editing an unrelated (Query/
                                               # Constant) channel never does, and
                                               # skipping it there is what keeps a
                                               # color change fast even when some
                                               # OTHER channel uses an expensive
                                               # "multi" Buddha point option
            self.recolor(0, self.rows_done)
            self.draw()
            if self.pinned_px is not None:
                self.update_preview(*self.pinned_px)
        self.rebuild_view_list()
        self.refresh_timeline_diffs()

    def scatter_buddha_rows(self, r0, r1):
        """Add rows [r0, r1)'s contribution to every Buddha-mode channel's
        RAW (unblurred) visit-count buffer. For each such channel's selected
        points: a "simple" point increments the count at whichever pixel
        its (already computed) sample point lands in, for every ray that
        fired (and, if the option is restricted to a termination reason,
        only for rays that ended that way); a "multi" point re-traces the
        row's rays to reach parts of the path the per-pixel summary doesn't
        keep (see scatter_multi_buddha). Blur (if any) is never applied
        here - it's applied once, lazily, to the whole accumulated buffer
        when a colorized value is actually needed (see
        blurred_buddha_counts) - so this stays cheap regardless of the
        blur radius; each channel whose raw counts changed here has its
        cached blurred version invalidated."""
        view = self.live_view()
        if r1 <= r0 or not any(s == CHAN_SRC_BUDDHA for s in view.chan_src):
            return
        h, w = SIZE, SIZE
        fired = self.status[r0:r1] == 1
        if not fired.any():
            return
        term_stat = self.stats[r0:r1, :, ST_TERMINATION]
        for c in range(3):
            if view.chan_src[c] != CHAN_SRC_BUDDHA or self.buddha_counts[c] is None:
                continue
            want_multi = {BUDDHA_MULTI_ALL_SELF: False, BUDDHA_MULTI_ALL_MID: False,
                         BUDDHA_MULTI_PROG_BOUNCE: False, BUDDHA_MULTI_PROG_MID: False}
            touched = False
            for i, on in enumerate(view.buddha_points[c]):
                if not on:
                    continue
                _label, style, ref, term_filter = BUDDHA_POINT_CATALOG[i]
                if style == BUDDHA_MULTI:
                    want_multi[ref] = True
                    continue
                mask = fired
                if term_filter is not None:
                    target = TERM_STAT_VALUE[term_filter]
                    mask = mask & (np.abs(term_stat - target) < 0.01)
                if not mask.any():
                    continue
                xs = self.points[r0:r1, :, ref, 0][mask]
                ys = self.points[r0:r1, :, ref, 1][mask]
                col = np.floor((xs + 1.0) * (w / 2.0)).astype(np.int64)
                row = np.floor((1.0 - ys) * (h / 2.0)).astype(np.int64)
                valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)
                scatter_points_raw(self.buddha_counts[c], row, col, valid)
                touched = True
            if any(want_multi.values()):
                g = self.settings_of_current_image()
                scatter_multi_buddha(
                    r0, r1, g.max_iter, effective_dist(g.max_dist), g.seg_mult, self.gen_fuzz,
                    self.gen_confine, self.gen_emission, self.gen_obstacles, g.fuzz.seed,
                    want_multi[BUDDHA_MULTI_ALL_SELF], want_multi[BUDDHA_MULTI_ALL_MID],
                    want_multi[BUDDHA_MULTI_PROG_BOUNCE], want_multi[BUDDHA_MULTI_PROG_MID],
                    self.buddha_counts[c])
                touched = True
            if touched:
                self.buddha_counts_blurred[c] = None

    def blurred_buddha_counts(self, c):
        """Channel c's blurred Buddha counts, computed once (via a single
        whole-image FFT pass, independent of how many points were
        scattered or how large the blur radius is) and cached until the
        next scatter_buddha_rows call invalidates it. None if the channel
        isn't in Buddha mode at all."""
        if self.buddha_counts[c] is None:
            return None
        if self.buddha_counts_blurred[c] is None:
            blur = self.settings_of_current_image().buddha_blur
            radius_px = world_radius_to_pixels(blur.radius, SIZE, SIZE)
            kernel = build_buddha_blur_kernel(radius_px, blur.type)
            self.buddha_counts_blurred[c] = blur_counts_image(self.buddha_counts[c], kernel)
        return self.buddha_counts_blurred[c]

    def rebuild_buddha_counts(self):
        """Full reset + rescatter of every Buddha-mode channel's visit-count
        buffer, from every row computed so far. Used when the Buddha
        configuration itself changes (which points, or which channels are in
        Buddha mode at all) - the existing counts would otherwise be for the
        wrong selection - and when starting a fresh render."""
        self.buddha_counts = [
            np.zeros((SIZE, SIZE), dtype=np.float64) if s == CHAN_SRC_BUDDHA else None
            for s in self.live_view().chan_src]
        self.buddha_counts_blurred = [None] * 3
        self.scatter_buddha_rows(0, self.rows_done)

    def recolor(self, r0, r1):
        """Turn stored metrics and sample points into pixel colors, rows [r0, r1)."""
        if r1 <= r0:
            return
        view = self.live_view()
        st = self.stats[r0:r1]
        sp = self.points[r0:r1]
        computed = self.status[r0:r1] == 1
        out = self.img[r0:r1]
        out[:] = BACKGROUND
        n = int(computed.sum())
        if n:
            vals = []
            for c in range(3):
                s = self.mapping[c]
                if s == CHAN_BUDDHA:
                    counts = self.blurred_buddha_counts(c)
                    if counts is None:
                        vals.append(np.zeros(n))
                    else:
                        sub = counts[r0:r1][computed].astype(np.float64)
                        mult = BUDDHA_BRIGHTNESS_VALUES[view.buddha_brightness[c]]
                        vals.append(buddha_compress(sub, view.buddha_compression[c], mult))
                elif s == NONE_ZERO:
                    vals.append(np.zeros(n))
                elif s == NONE_ONE:
                    vals.append(np.ones(n))
                elif s in POINT_STATS:
                    x1, y1 = extract_point_xy(sp, view.point1_choice[c], computed, n)
                    x2, y2 = extract_point_xy(sp, view.point2_choice[c], computed, n)
                    vals.append(point_comparison_metric(s, x1, y1, x2, y2))
                else:
                    vals.append(np.clip(st[..., s][computed].astype(np.float64),
                                        0.0, 1.0))
            r, g, b = channels_to_rgb(view.mode, *vals)
            rgb = np.stack((r, g, b), axis=-1)
            out[computed] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    def component_values(self, vec, sp, x0, y0):
        """The three channel values (0..1) for one ray under the current
        mapping. vec: its metrics, sp: its sample points, (x0, y0): the
        pixel it was emitted from (used to look up a Buddha channel's
        already-accumulated count at that same pixel)."""
        view = self.live_view()
        vals = []
        for c in range(3):
            s = self.mapping[c]
            if s == CHAN_BUDDHA:
                counts = self.blurred_buddha_counts(c)
                if counts is None:
                    v = 0.0
                else:
                    col = int(np.floor((x0 + 1.0) * (SIZE / 2.0)))
                    row = int(np.floor((1.0 - y0) * (SIZE / 2.0)))
                    if 0 <= row < SIZE and 0 <= col < SIZE:
                        mult = BUDDHA_BRIGHTNESS_VALUES[view.buddha_brightness[c]]
                        v = float(buddha_compress(counts[row, col], view.buddha_compression[c],
                                                  mult))
                    else:
                        v = 0.0
            elif s == NONE_ZERO:
                v = 0.0
            elif s == NONE_ONE:
                v = 1.0
            elif s in POINT_STATS:
                x1, y1 = resolve_point_xy(sp, view.point1_choice[c])
                x2, y2 = resolve_point_xy(sp, view.point2_choice[c])
                v = float(point_comparison_metric(s, x1, y1, x2, y2))
            else:
                v = min(max(float(vec[s]), 0.0), 1.0)
            vals.append(v)
        return vals

    def map_color(self, vals):
        """Color (r, g, b) from three channel values under the current model."""
        r, g, b = channels_to_rgb(self.live_view().mode, *[np.array([v]) for v in vals])
        return tuple(int(np.clip(np.rint(ch[0]), 0, 255)) for ch in (r, g, b))

    # ---- generation ------------------------------------------------------- #
    def warm_up(self):
        """Trigger numba compilation (slow only the first time; then cached)."""
        big = float(DIST_LIMIT_MAX)
        bound = ellipse_bound(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_ROTATION)
        fz = fuzz_rule(DEFAULT_FUZZ)
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_bounds = np.empty((0, OBST_BOUND_LEN), dtype=np.float64)
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=np.bool_)
        inside_boundary(0.0, 0.0, SHAPE_ELLIPSE, bound)
        no_obstacles = ObstacleArrays(
            shapes=empty_i64, bounds=empty_bounds, cx=empty_f64, cy=empty_f64,
            effects=np.empty((0, 2), dtype=np.int64), fire_inside=empty_bool)
        emits_ray(0.0, 0.0, no_obstacles)
        trace_path(0.0, 0.0, self.hover_pts, self.hover_kinds, WARM_ITER, big,
                   DEFAULT_SEGMENT_MULT, fz, CONFINE_DEFAULT, EMISSION_DEFAULT, no_obstacles,
                   self.hover_raw)
        compute_stats(self.hover_pts, 4, self.hover_raw, WARM_ITER,
                      DEFAULT_SEGMENT_MULT, self.hover_vec)
        compute_points(self.hover_pts, 4, self.hover_raw, self.hover_kinds, self.hover_sp)
        render_rows(self.stats, self.points, self.status, 0, 1, WARM_ITER, big,
                    DEFAULT_SEGMENT_MULT, fz, CONFINE_DEFAULT, EMISSION_DEFAULT, no_obstacles, 0)
        self.stats[:] = 0
        self.points[:] = 0
        self.status[:] = 0
        self.ready = True
        self.update_buttons()
        self.update_status()
        self.root.after(500, self.take_startup_captures)   # let the UI settle first

    def build_menu_bar(self):
        """The menu bar: File (projects) and Debug (the dev aids; they do nothing
        unless the dev/ folder exists)."""
        bar = tk.Menu(self.root, tearoff=False)
        file_menu = tk.Menu(bar, tearoff=False)
        file_menu.add_command(label="New project", accelerator="Ctrl+N",
                              command=self.new_project)
        file_menu.add_command(label="Open project...", accelerator="Ctrl+O",
                              command=self.open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Save project", accelerator="Ctrl+S",
                              command=self.save_project)
        file_menu.add_command(label="Save project as...", accelerator="Ctrl+Shift+S",
                              command=self.save_project_as)
        bar.add_cascade(label="File", menu=file_menu)
        debug = tk.Menu(bar, tearoff=False)
        debug.add_command(label="Take screenshot", command=self.manual_screenshot)
        debug.add_command(label="Dump (without large arrays)",
                          command=lambda: self.manual_dump(False))
        debug.add_command(label="Dump (with large arrays)",
                          command=lambda: self.manual_dump(True))
        bar.add_cascade(label="Debug", menu=debug)
        self.root.config(menu=bar)

    def report_debug_result(self, what, path):
        if path is None:
            messagebox.showinfo("Debug", f"No {what} was saved (it needs the dev folder "
                                         "next to the program" +
                                (", and Pillow" if what == "screenshot" else "") + ").")
        else:
            messagebox.showinfo("Debug", f"Saved {what}:\n{path}")

    def manual_screenshot(self):
        try:
            path = take_ui_screenshot(self.root, "manual")
        except Exception as exc:                   # noqa: BLE001
            print(f"Screenshot failed: {exc}", file=sys.stderr)
            path = None
        self.report_debug_result("screenshot", path)

    def manual_dump(self, include_large_arrays):
        self.report_debug_result("dump", dump_app_state(self, "manual", include_large_arrays))

    def on_callback_exception(self, exc_type, exc_value, exc_tb):
        """Tk calls this for an exception escaping any callback (replacing its
        default, which only prints it). Prints it the same way, then leaves a crash
        dump (without large arrays; never a screenshot), at most MAX_CRASH_DUMPS
        per launch."""
        print("Exception in Tkinter callback", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.dump_crash((exc_type, exc_value, exc_tb))

    def dump_crash(self, exc_info):
        if self._crash_dumps >= MAX_CRASH_DUMPS:
            return
        self._crash_dumps += 1
        dump_app_state(self, "crash", False, exc_info)

    def install_crash_hooks(self):
        """Also dump on an exception that ends the program (sys.excepthook) or a
        background thread (run later on the main thread, since Tk variables
        must not be read from another one). The previous hooks still run."""
        previous_hook, previous_thread_hook = sys.excepthook, threading.excepthook

        def hook(exc_type, exc_value, exc_tb):
            self.dump_crash((exc_type, exc_value, exc_tb))
            previous_hook(exc_type, exc_value, exc_tb)

        def thread_hook(args):
            info = (args.exc_type, args.exc_value, args.exc_traceback)
            try:
                self.root.after(0, lambda: self.dump_crash(info))
            except Exception:                      # noqa: BLE001
                pass
            previous_thread_hook(args)

        sys.excepthook = hook
        threading.excepthook = thread_hook

    def keep_window_on_screen(self):
        """Move the window (never resize it) so all of it is on the screen it is
        on, e.g. when the operating system placed it with its bottom edge below the
        screen. A window bigger than the screen ends up with its top-left corner at
        the screen's. Startup only; never raises."""
        try:
            self.root.update_idletasks()
            rect, area = window_rect_and_work_area(self.root)
            dx, dy = shift_into_area(rect, area)
            position = parse_geometry_position(self.root.geometry())
            if (dx or dy) and position is not None:
                self.root.geometry(f"+{position[0] + dx}+{position[1] + dy}")
                self.root.update_idletasks()
        except Exception as exc:                    # noqa: BLE001
            print(f"Could not reposition the window: {exc}", file=sys.stderr)

    def take_startup_captures(self):
        """Once per launch, shortly after the window settles: a screenshot, then a
        dump without large arrays."""
        self.keep_window_on_screen()                # so the screenshot shows all of it
        self.take_startup_screenshot()
        if getattr(self, "_startup_dump_taken", False):
            return
        self._startup_dump_taken = True
        dump_app_state(self, "startup", False)

    def take_startup_screenshot(self):
        """At most once per launch (see take_ui_screenshot for the conditions).
        A development aid, so any failure is reported and otherwise ignored."""
        if getattr(self, "_startup_screenshot_taken", False):
            return
        self._startup_screenshot_taken = True
        try:
            take_ui_screenshot(self.root, "startup")
        except Exception as exc:                   # noqa: BLE001
            print(f"Startup screenshot failed: {exc}", file=sys.stderr)

    def start(self):
        """(Re)start generation using the current settings."""
        snap = self.current_ray_settings()
        if snap is None:
            return
        if self.tick_id is not None:
            self.root.after_cancel(self.tick_id)
            self.tick_id = None

        settings = snap
        self.generated_settings = settings
        self.baseline = settings                   # Revert now returns to these values
        self.gen_fuzz = fuzz_rule(settings.fuzz)
        self.gen_confine = settings_confine(settings)
        self.gen_emission = settings_emission(settings)
        self.gen_obstacles = build_obstacle_arrays(self.obstacles)
        self.gen_obstacles_sig = obstacles_signature(self.obstacles)
        self.gen_obstacles_by_id = {o.id: o.state() for o in self.obstacles}
        self.gen_obstacle_snapshots = [o.snapshot() for o in self.obstacles]
        self.canvas.delete("pending")
        self.stats[:] = 0
        self.points[:] = 0
        self.status[:] = 0
        self.img[:] = BACKGROUND
        self.rows_done = 0
        self.batch_rows = 1
        self.rebuild_buddha_counts()
        self.view_thumb_cache = {}   # every non-live view's cached thumbnail is for the
                                     # previous generation's data - none of it applies here
        self.done = False
        self.running = True
        if self.pinned_px is None:
            self.clear_overlay()
        self.start_time = time.perf_counter()
        self.elapsed = 0.0
        self.next_tick = self.start_time
        self.draw()
        self.update_status()
        self.update_buttons()
        self.tick()

    def tick(self):
        """Compute for a slice of the refresh period, redraw, reschedule."""
        t_start = time.perf_counter()
        budget = self.PERIOD * COMPUTE_FRACTION
        target = budget / 3.0                      # aim for ~3 batches per tick
        rows_before = self.rows_done
        g = self.settings_of_current_image()
        max_dist = effective_dist(g.max_dist)

        while self.rows_done < SIZE:
            t0 = time.perf_counter()
            row1 = min(self.rows_done + self.batch_rows, SIZE)
            render_rows(self.stats, self.points, self.status, self.rows_done, row1,
                        g.max_iter, max_dist, g.seg_mult, self.gen_fuzz,
                        self.gen_confine, self.gen_emission, self.gen_obstacles,
                        g.fuzz.seed)
            self.rows_done = row1
            now = time.perf_counter()

            # Adapt batch size so each batch takes about `target` seconds
            dt = max(now - t0, 1e-6)
            scale = min(max(target / dt, 0.5), 2.0)
            self.batch_rows = max(1, int(self.batch_rows * scale))

            if now - t_start >= budget:
                break

        self.scatter_buddha_rows(rows_before, self.rows_done)
        if any(s == CHAN_SRC_BUDDHA for s in self.live_view().chan_src):
            self.recolor(0, self.rows_done)         # a Buddha channel's counts can
                                                     # affect any already-drawn pixel
        else:
            self.recolor(rows_before, self.rows_done)
        self.elapsed = time.perf_counter() - self.start_time
        self.draw()

        if self.rows_done >= SIZE:
            self.on_finished()
            return
        self.update_status()

        # Keep a steady 10 Hz cadence; idle time lets Tk handle mouse events
        now = time.perf_counter()
        self.next_tick += self.PERIOD
        if self.next_tick < now:
            self.next_tick = now
        delay_ms = max(1, int((self.next_tick - now) * 1000))
        self.tick_id = self.root.after(delay_ms, self.tick)

    def obstacle_ring_specs(self):
        """[ObstacleRing, ...] for obstacles to draw a solid two-layer outline for:
        the frozen set if an image has been generated (matching what was actually
        simulated/saved), else the live (current) set."""
        if self.generated_settings is not None:
            g = self.gen_obstacles
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

    def on_finished(self):
        self.tick_id = None
        self.running = False
        self.done = True
        self.update_status()
        self.update_buttons()
        if self.pinned_px is not None:             # refresh the pinned preview
            self.hover_px = None                   # with the settings just used
            self.update_preview(*self.pinned_px)
        self.rebuild_view_list()                   # every view's thumbnail reflects the
                                                    # simulation that just finished, even
                                                    # if the Coloration tab isn't showing

    # ---- hover demonstration ---------------------------------------------- #
    def preview_settings(self):
        """Settings for previews: those of the image being generated, or the
        current entries/controls (last valid values) if nothing has been started."""
        if self.generated_settings is not None:
            return self.generated_settings
        return RaySettings(max_iter=self.last_iter, max_dist=self.last_dist,
                        seg_mult=self.seg_mult_scale.get(), fuzz=self.read_fuzz_controls(),
                        buddha_blur=BuddhaBlurSettings(
                            radius=self.buddha_blur_radius_scale.get(),
                            type=BUDDHA_BLUR_LABELS.index(self.buddha_blur_type_var.get())),
                        confine_window=self.last_confine_window,
                        confine_method=CONFINE_METHOD_NAMES.index(
                            self.confine_method_var.get()),
                        confine_threshold=self.last_confine_threshold,
                        emit_mode=EMIT_MODE_NAMES.index(self.emit_mode_var.get()),
                        emit_target_r=self.last_emit_target_r,
                        emit_target_theta=self.last_emit_target_theta,
                        emit_target_cx=self.last_emit_target_cx,
                        emit_target_cy=self.last_emit_target_cy,
                        emit_offset=round(float(self.emit_offset_scale.get()), 2))

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
        iters, dist = ps.max_iter, effective_dist(ps.max_dist)
        fz = fuzz_rule(ps.fuzz)
        if ps.fuzz.enabled:
            np.random.seed(ps.fuzz.seed + i * 1000003 + j * 97)
        pts, kinds = self.hover_pts, self.hover_kinds
        raw, vec, sp = self.hover_raw, self.hover_vec, self.hover_sp
        term, n = trace_path(x, y, pts, kinds, iters, dist, ps.seg_mult, fz, settings_confine(ps),
                             settings_emission(ps), obstacles, raw)
        compute_stats(pts, n, raw, iters, ps.seg_mult, vec)
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
                self.generated_settings, fmt, self.save_jxl_mode_var.get() == "Lossless",
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
                    self.generated_settings, fmt, self.save_jxl_mode_var.get() == "Lossless",
                    int(round(float(self.save_jxl_quality_scale.get())))),
                defaultextension="." + fmt, filetypes=[(label, f"*.{fmt}")])
            if path:
                self._write(path)

    def settings_of_current_image(self):
        """The RaySettings the current image was generated with (the defaults, for
        the placeholder image before the first Start). Coloring a view from the
        stored data needs all of them, not just the limits: Buddha coloration
        re-traces rays with the generated confinement and emission rules."""
        if self.generated_settings is not None:
            return self.generated_settings
        return DEFAULT_RAY_SETTINGS

    def views_to_export(self):
        """Which views to include in an export, per the Export designer's
        "Include all views" checkbox: all of them, in list/composition
        order, or just the one currently selected/shown in the preview."""
        if self.include_all_views_var.get():
            return list(self.views)
        return [self.live_view()]

    def render_current_image(self, views):
        """The CURRENT (already fully-traced) simulation, colorized by each
        of `views` and composed side by side at the live preview's own
        resolution (SIZE x SIZE per view) - reusing self.stats/self.points/
        self.status as already computed, no re-simulation. Each view gets
        the outline decoration drawn in if enabled, exactly like
        self.frame/draw(). Only called for more than one view - the
        single-view case is just self.frame, already rendered."""
        mini_settings = self.settings_of_current_image()
        obst_arrays = self.gen_obstacles
        images = []
        for v in views:
            images.append(self.colorize_view(v.settings(), self.stats, self.points, self.status,
                                             SIZE, SIZE, mini_settings, self.gen_fuzz,
                                             obst_arrays))
        outline = self.deco.get("outline")
        if outline is None or outline.get():
            for img in images:
                for ring in self.obstacle_ring_specs():
                    draw_obstacle_outline(img, ring.shape, ring.bound, self.gx - ring.cx,
                                          self.gy - ring.cy, ring.in_effect, ring.out_effect)
        return images[0] if len(images) == 1 else np.concatenate(images, axis=1)

    def _write(self, path):
        fmt = self.save_format_var.get()
        jxl_lossless = self.save_jxl_mode_var.get() == "Lossless"
        jxl_quality = int(round(float(self.save_jxl_quality_scale.get())))
        views = self.views_to_export()
        # self.frame already holds exactly the single-view case (with the
        # outline if that decoration is on, no preview overlays) - only
        # recompute when more than one view needs to be composed in.
        rgb = self.frame if len(views) == 1 else self.render_current_image(views)
        rgb = apply_info_locations(
            rgb, views, rgb.shape[1] // len(views), self.generated_settings,
            self.gen_obstacle_snapshots, self.color_info_location_var.get(),
            self.figure_info_location_var.get(), mutate_ok=len(views) > 1)   # len(views) > 1 means rgb is already a fresh
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
            self.generated_settings, fmt, self.save_jxl_mode_var.get() == "Lossless",
            int(round(float(self.save_jxl_quality_scale.get()))))
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save new image as",
            initialdir=ensure_export_dir(), initialfile=initial,
            defaultextension="." + fmt, filetypes=[(label, f"*.{fmt}")])
        if path:
            self.save_new_image(path)

    def save_new_image(self, path_or_none):
        """Generate a fresh image at the New-figure controls' own resolution
        (per view - with "Include all views" on, each view is rendered at
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
        view_snapshots = [snapshot_view(v) for v in self.views_to_export()]
        views = resolve_views(view_snapshots, [fake_kf], 0)
        color_location = self.color_info_location_var.get()
        figure_location = self.figure_info_location_var.get()
        if path_or_none is None:
            settings_for_name = data.settings
            path = os.path.join(ensure_export_dir(),
                                image_filename(settings_for_name, fmt, jxl_lossless, jxl_quality))
        else:
            path = path_or_none
        self.save_btn.state(["disabled"])
        self.save_as_btn.state(["disabled"])
        thread = threading.Thread(
            target=self._save_new_image_worker,
            args=(data, w, h, obst_arrays, obst_values, views, fmt, jxl_lossless,
                 jxl_quality, path, color_location, figure_location),
            daemon=True)
        thread.start()

    def _save_new_image_worker(self, data, w, h, obst_arrays, obst_values, views, fmt,
                               jxl_lossless, jxl_quality, path, color_location,
                               figure_location):
        """Runs on a background thread: never touches a Tk widget directly,
        only via self.root.after(0, ...)."""
        try:
            rgb = self.render_full_image(data, w, h, obst_arrays, views)
            rgb = apply_info_locations(rgb, views, w, data.settings,
                                       obst_values, color_location, figure_location,
                                       mutate_ok=True)
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

    # ---- timeline: keyframes ------------------------------------------------ #
    def capture_all_settings(self, show_error=True):
        """dict of every canonical setting's current value (plus every current
        obstacle's fields and every current view's fields, as dynamic
        "obst:<id>:<field>" / "view:<id>:<field>" keys), or None (with an
        error dialog, unless show_error is False - for passive/reactive
        callers like refresh_timeline_diffs, which already show their own
        gentle inline message and run on every keystroke, where a popup
        would be disruptive) if the limit fields are currently invalid."""
        snap = self.current_ray_settings()
        if snap is None:
            if show_error:
                messagebox.showerror("Invalid settings",
                                     "Fix the iteration/distance limit fields first.")
            return None
        d = ray_settings_to_dict(snap)
        d["outline"] = self.deco["outline"].get()
        for o in self.obstacles:
            for field in o.FIELD_NAMES:
                d[f"obst:{o.id}:{field}"] = getattr(o, field)
        for v in self.views:
            d[f"view:{v.id}:mode"] = v.mode
            for c in range(3):
                d[f"view:{v.id}:chan{c}"] = view_channel_value(v, c)
        return d

    def new_keyframe(self):
        """A new keyframe that defines every setting, from the controls' current
        values."""
        data = self.capture_all_settings()
        if data is None:
            return
        frame_index = (0 if not self.keyframes
                      else max(kf.frame_index for kf in self.keyframes) + 1)
        kf = Keyframe(self._kf_counter, frame_index, data)
        self._kf_counter += 1
        self.keyframes.append(kf)
        self.select_keyframe(kf, apply=False)      # it holds exactly the current values

    def new_keyframe_empty(self):
        """A new keyframe that defines no settings at all; selecting it loads
        whatever its neighbors animate through it."""
        frame_index = (0 if not self.keyframes
                      else max(kf.frame_index for kf in self.keyframes) + 1)
        kf = Keyframe(self._kf_counter, frame_index, {})
        self._kf_counter += 1
        self.keyframes.append(kf)
        self.select_keyframe(kf)

    def load_value(self, key, val):
        """Apply one resolved value to whichever control(s) hold that setting.
        Does NOT itself refresh dependent state - callers batch that after."""
        if key.startswith("obst:"):
            _prefix, obst_id, field = key.split(":", 2)
            obst_id = int(obst_id)
            o = next((o for o in self.obstacles if o.id == obst_id), None)
            if o is not None and field in o.FIELD_NAMES:
                want = o.FIELD_TYPES[field]
                setattr(o, field, want(val) if want is float else
                        int(round(val)) if want is int else bool(val))
                if o is self.selected_obstacle:
                    self.refresh_obstacle_editor()
                self.rebuild_obstacle_list()
            return
        if key.startswith("view:"):
            _prefix, view_id, field = key.split(":", 2)
            v = self.view_by_id(int(view_id))
            if v is None:
                return
            if field == "mode":
                v.mode = val
            else:
                c = int(field[-1])
                v.set_channel(c, val)
            if v is self.selected_view_obj():
                self.refresh_color_controls_for_selected_view()
            if v is self.live_view():
                self.mapping = make_mapping(v.kinds, v.point_metric, v.other_metric,
                                            v.chan_src, v.constant_choice)
                if field != "mode" and v.chan_src[int(field[-1])] == CHAN_SRC_BUDDHA:
                    self.rebuild_buddha_counts()   # mode never affects Buddha scattering
                                                   # at all, and an unrelated channel's
                                                   # own edit can't either - see
                                                   # apply_mapping_change's identical fix
                self.update_stat_tags()
                self.recolor(0, self.rows_done)
                self.draw()
                if self.pinned_px is not None:
                    self.update_preview(*self.pinned_px)
            self.rebuild_view_list()
            return
        if key == "outline":
            self.deco["outline"].set(val)
            self.on_decoration_changed("outline")
        else:
            self.set_setting_control(key, val)

    def set_setting_control(self, key, val):
        """Write a simulation setting (a RAY_SETTING_SPECS key) into its control, as the
        control's kind needs it. Does NOT refresh what depends on it."""
        spec = RAY_SETTING_SPECS[key]
        control = getattr(self, spec.control)
        if spec.kind == "text":
            control.set(str(val))
        elif spec.kind == "int_text":
            control.set(str(int(round(val))))
        elif spec.kind == "float_text":
            control.set(str(float(val)))
        elif spec.kind == "choice":
            control.set(spec.names[int(round(val))])
        else:                                       # "number" (a slider or number entry), "bool"
            control.set(val)

    def setting_fallback(self, key):
        """The value to fall back to when no keyframe defines `key` at all:
        the static default for a normal setting, that obstacle's own current
        value (None if it no longer exists) for an obst:<id>:<field> key, or
        that view's own current value (None if it no longer exists) for a
        view:<id>:<field> key."""
        if key.startswith("obst:"):
            _prefix, obst_id, field = key.split(":", 2)
            o = next((o for o in self.obstacles if o.id == int(obst_id)), None)
            return getattr(o, field) if o is not None else None
        if key.startswith("view:"):
            _prefix, view_id, field = key.split(":", 2)
            v = self.view_by_id(int(view_id))
            return view_field_value(v, field) if v is not None else None
        return DEFAULT_SETTING_VALUES[key]

    def setting_is_float(self, key):
        if key.startswith("obst:"):
            return OBST_FIELD_IS_FLOAT[key.split(":", 2)[2]]
        if key.startswith("view:"):
            field = key.split(":", 2)[2]
            return next(is_f for f, _label, is_f in VIEW_FIELD_INFO if f == field)
        return key in SETTING_FLOAT_KEYS

    def load_animated_setting(self, kf, key, refresh=True):
        """Load the value `key` resolves to at kf's own frame index, blended
        from whichever surrounding keyframes actually define it (kf itself
        doesn't) - without changing kf. `refresh=False` skips the refresh
        side effects (each of which re-walks every current setting, so
        they're too expensive to repeat once per key) - callers loading
        several settings at once should pass False here and do one shared
        refresh afterwards (see apply_keyframe_settings)."""
        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame_index)
        val = resolve_value(key, sorted_kfs, kf.frame_index,
                            self.setting_fallback(key), self.setting_is_float(key))
        self.load_value(key, val)
        if refresh:
            self.on_limit_var_changed()
            self.on_shape_param_changed()
            self.on_fuzz_changed()
            self.refresh_timeline_diffs()

    def save_settings(self, kf, keys):
        """Save several of the controls' current settings into `kf` (adding
        any it didn't already define)."""
        current = self.capture_all_settings()
        if current is None:
            return
        for key in keys:
            if key in current:
                kf.data[key] = current[key]
        self.kf_modified.difference_update(keys)
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()
        self.refresh_modified_outlines()

    def delete_setting(self, kf, key):
        """Remove `key` from `kf`, so it goes back to being either animated by
        surrounding keyframes or undefined."""
        if key in kf.data:
            del kf.data[key]
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

    def setting_status(self, kf, key, sorted_kfs, idx_kf):
        """How `key` relates to `kf`: 'present' (kf defines it), 'animated'
        (kf doesn't, but some other keyframe does - on either side or both, so
        its value at kf comes from the neighbors: blended between the nearest
        one on each side, or, with one side only, held from the first/last
        keyframe defining it), or 'none' (no keyframe defines it)."""
        if key in kf.data:
            return "present"
        others = sorted_kfs[:idx_kf] + sorted_kfs[idx_kf + 1:]
        return "animated" if any(key in k2.data for k2 in others) else "none"

    def select_keyframe(self, kf, apply=True):
        """Select `kf`: equivalent to deselecting the current keyframe first (its
        unsaved modifications stop counting as modifications; the values stay in
        the controls until overwritten below), then, with `apply`, loading into
        the controls every setting `kf` defines and every setting its
        neighbors animate through it. Selecting the selected keyframe again
        does nothing."""
        if kf is self.selected_keyframe:
            return
        self.selected_keyframe = kf
        self.kf_modified = set()
        if apply:
            self.apply_keyframe_settings(kf)
        self.kf_last_seen = self.capture_all_settings(show_error=False)
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()
        self.refresh_modified_outlines()

    def deselect_keyframe(self):
        """No keyframe selected. The controls keep their values, but none of them
        counts as modified any more, so selecting another keyframe overwrites
        them freely."""
        if self.selected_keyframe is None:
            return
        self.selected_keyframe = None
        self.kf_modified = set()
        self.kf_last_seen = None
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()
        self.refresh_modified_outlines()

    def apply_keyframe_settings(self, kf):
        """Load into the controls every setting `kf` defines, and every one it
        lacks but its neighbors animate through its frame index, then refresh
        what depends on them once."""
        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame_index)
        idx_kf = sorted_kfs.index(kf)
        self.kf_applying = True
        try:
            for entry in self.all_setting_entries():
                key = entry.key
                status = self.setting_status(kf, key, sorted_kfs, idx_kf)
                if status == "present":
                    self.load_value(key, kf.data[key])
                elif status == "animated":
                    self.load_animated_setting(kf, key, refresh=False)
            self.on_limit_var_changed()
            self.on_shape_param_changed()
            self.on_fuzz_changed()
        finally:
            self.kf_applying = False

    def poll_modified(self):
        """Runs every MODIFIED_POLL_MS for the life of the program."""
        try:
            self.check_modified()
            self.refresh_modified_outlines()
        finally:
            self.root.after(MODIFIED_POLL_MS, self.poll_modified)

    def check_modified(self):
        """With a keyframe selected, mark every setting whose value changed since
        the previous check as modified (see detect_modified_keys). Polling the
        settings, rather than hooking every control, catches every way a setting
        can be edited (entries, sliders, dropdowns, the obstacle and view
        editors) and costs well under a millisecond. Returns whether the set of
        modified settings changed."""
        if self.selected_keyframe is None or self.kf_applying:
            return False
        current = self.capture_all_settings(show_error=False)
        if current is None:                       # a limit field is mid-edit
            return False
        last = self.kf_last_seen if self.kf_last_seen is not None else current
        new = detect_modified_keys(last, current, self.kf_modified)
        self.kf_last_seen = current
        if new == self.kf_modified:
            return False
        self.kf_modified = new
        self.refresh_timeline_diffs()
        return True

    def widgets_for_setting(self, key):
        """The main control(s) currently showing setting `key`: [] when it has
        none visible now (an obstacle or view other than the one being edited)."""
        def attr(name):
            obj = getattr(self, name, None)
            return getattr(obj, "widget", obj) if obj is not None else None

        if key.startswith("obst:"):
            _prefix, obst_id, field = key.split(":", 2)
            o = self.selected_obstacle
            if o is None or o.id != int(obst_id):
                return []
            widget = self.obstacle_widget(type(o), field)
            widget = getattr(widget, "widget", widget)
            return [widget] if widget is not None else []
        if key.startswith("view:"):
            _prefix, view_id, field = key.split(":", 2)
            v = self.selected_view_obj()
            if v is None or v.id != int(view_id):
                return []
            if field == "mode":
                widget = attr("mode_combo")
            else:
                blocks = getattr(self, "chan_blocks", [])
                widget = blocks[int(field[-1])] if int(field[-1]) < len(blocks) else None
            return [widget] if widget is not None else []
        widget = attr(SETTING_WIDGET_ATTRS[key])
        return [widget] if widget is not None else []

    def refresh_modified_outlines(self):
        """Show a blue outline around the control of every modified setting, and
        remove the outline from every other."""
        want = set()
        for key in self.kf_modified:
            want.update(self.widgets_for_setting(key))
        for widget in want:
            if widget not in self.outlines:
                self.outlines[widget] = ModifiedOutline(widget)
        for widget, outline in self.outlines.items():
            outline.show(widget in want)

    def rebuild_keyframe_list(self):
        for w in self.kf_list_frame.winfo_children():
            w.destroy()
        for idx, kf in enumerate(self.keyframes):
            selected = kf is self.selected_keyframe
            row = tk.Frame(self.kf_list_frame,
                          highlightthickness=SELECTED_OUTLINE_PX if selected else 0,
                          highlightbackground=SELECTED_OUTLINE,
                          highlightcolor=SELECTED_OUTLINE)
            row.grid(row=idx, column=0, sticky="ew", pady=2, padx=1)
            row.columnconfigure(0, weight=1)
            row.bind("<Button-1>", lambda _e, kf=kf: self.select_keyframe(kf))

            lbl = tk.Label(row, text=f"#{idx + 1}  frame {kf.frame_index}",
                          anchor="w", cursor="hand2", font=self.small_font)
            lbl.grid(row=0, column=0, sticky="ew", padx=4, pady=6)
            lbl.bind("<Button-1>", lambda _e, kf=kf: self.select_keyframe(kf))
        self.update_kf_detail_panel()

    def all_setting_entries(self):
        """[SettingEntry, ...] for every static setting plus every current
        obstacle's and view's fields, in display order."""
        entries = [SettingEntry(key=key, tab=tab, label=label, is_float=key in SETTING_FLOAT_KEYS,
                                fallback=DEFAULT_SETTING_VALUES[key])
                   for key, tab, label in SETTING_INFO]
        for idx, o in enumerate(self.obstacles):
            tab = obstacle_tab_title(idx, o)
            for field, label, is_float in o.KEYFRAME_FIELDS:
                entries.append(SettingEntry(key=f"obst:{o.id}:{field}", tab=tab, label=label,
                                            is_float=is_float, fallback=getattr(o, field)))
        for idx, v in enumerate(self.views):
            tab = f"View {idx}"
            for field, label, is_float in VIEW_FIELD_INFO:
                entries.append(SettingEntry(key=f"view:{v.id}:{field}", tab=tab, label=label,
                                            is_float=is_float,
                                            fallback=view_field_value(v, field)))
        return entries

    def all_setting_tabs(self):
        """Tab names in display order: the fixed ones, then one per
        obstacle, then one per view."""
        return (SETTING_TABS_ORDER + [obstacle_tab_title(i, o) for i, o in enumerate(self.obstacles)]
               + [f"View {i}" for i in range(len(self.views))])

    def refresh_timeline_diffs(self):
        """Rebuild the settings list (see _rebuild_timeline_diffs), then refit its
        scrolling to the new content."""
        self._rebuild_timeline_diffs()
        self.fit_diff_scroll()

    def fit_diff_scroll(self):
        """Make the settings list's scroll area match its content right now, and keep
        the view inside it. Waiting for the list frame's <Configure> event is not
        enough: when a rebuild leaves the list shorter than the current scroll
        position (a filter or a collapsed section shrinking it), the canvas shows
        none of the frame, never resizes it, and so never sends that event - the
        list stayed scrolled past its end, blank."""
        canvas = getattr(self, "diff_canvas", None)
        if canvas is None:
            return
        self.diff_list_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.yview_moveto(canvas.yview()[0])      # clamps into the new scroll area

    def _rebuild_timeline_diffs(self):
        """Rebuild the settings list for the selected keyframe: every setting,
        grouped into collapsible sections (by the tab it lives on, or the obstacle
        or view it belongs to; all collapsed until expanded, see
        toggle_diff_section), each setting showing whether it's defined here,
        animated by surrounding keyframes, or neither."""
        if not hasattr(self, "diff_list_frame"):
            return                                 # Timeline panel not built yet
        for w in self.diff_list_frame.winfo_children():
            w.destroy()
        kf = self.selected_keyframe

        if kf is None:
            self.diff_save_all_btn.state(["disabled"])
            self.diff_delete_all_btn.state(["disabled"])
            ttk.Label(self.diff_list_frame, text="(select a keyframe)",
                     foreground="#666", font=self.small_font).grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return
        current = self.capture_all_settings(show_error=False)
        if current is None:
            self.diff_save_all_btn.state(["disabled"])
            self.diff_delete_all_btn.state(["disabled"])
            ttk.Label(self.diff_list_frame, text="(fix the limit fields first)",
                     foreground="#666", font=self.small_font).grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return

        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame_index)
        idx_kf = sorted_kfs.index(kf)
        self.diff_save_all_btn.state(["!disabled"] if self.kf_modified else ["disabled"])
        self.diff_delete_all_btn.state(["!disabled"] if kf.data else ["disabled"])

        row = 0
        entries = self.all_setting_entries()
        modified = self.kf_modified
        presence = self.diff_presence_var.get()

        def passes(key):
            return setting_passes_presence_filter(key in kf.data, presence)

        for tab in self.all_setting_tabs():
            all_tab_entries = [e for e in entries if e.tab == tab]
            tab_entries = [e for e in all_tab_entries if passes(e.key)]
            if not tab_entries:
                continue                       # nothing here matches the filters
            expanded = tab in self.diff_expanded
            n_set = sum(1 for e in all_tab_entries if e.key in kf.data)
            n_modified = sum(1 for e in all_tab_entries if e.key in modified)
            header = ttk.Label(self.diff_list_frame, cursor="hand2",
                               text=diff_section_header_text(tab, expanded, n_set,
                                                             len(all_tab_entries),
                                                             n_modified),
                               foreground=MODIFIED_OUTLINE if n_modified else "",
                               font=self.small_bold_font)
            header.grid(row=row, column=0, sticky="w", padx=4, pady=(6, 1))
            header.bind("<Button-1>", lambda _e, t=tab: self.toggle_diff_section(t))
            row += 1
            if not expanded:
                continue
            for entry in tab_entries:
                self.add_setting_row(row, kf, entry.key, entry.label, entry.is_float,
                                     entry.fallback, current, sorted_kfs, idx_kf)
                row += 1
        if row == 0:
            ttk.Label(self.diff_list_frame, text="(no settings match the filters)",
                     foreground="#666", font=self.small_font).grid(
                row=0, column=0, sticky="w", padx=4, pady=4)

    def toggle_diff_section(self, tab):
        """Expand or collapse one section of the keyframe settings list. Sections
        are remembered by title, and all start collapsed."""
        if tab in self.diff_expanded:
            self.diff_expanded.discard(tab)
        else:
            self.diff_expanded.add(tab)
        self.refresh_timeline_diffs()

    def add_setting_row(self, row, kf, key, label, is_float, fallback, current, sorted_kfs,
                        idx_kf):
        """One setting's row in the keyframe settings list: its label, and Save /
        Delete buttons as they apply, outlined in blue while the setting is
        modified."""
        status = self.setting_status(kf, key, sorted_kfs, idx_kf)
        r = tk.Frame(self.diff_list_frame,
                     highlightthickness=MODIFIED_OUTLINE_PX if key in self.kf_modified else 0,
                     highlightbackground=MODIFIED_OUTLINE, highlightcolor=MODIFIED_OUTLINE)
        r.grid(row=row, column=0, sticky="ew", padx=4, pady=(1, 3))
        ttk.Label(r, text=label, font=self.small_font).grid(row=0, column=0, sticky="w")
        btns = ttk.Frame(r)
        btns.grid(row=1, column=0, sticky="w")

        save_text = f"Save {short_value_text(key, current[key])}"
        ttk.Button(btns, text=save_text,
                  command=lambda kf=kf, k=key: self.save_settings(
                      kf, [k])).pack(side="left", padx=2)

        if status == "present":
            ttk.Button(btns, text="Delete",
                      command=lambda kf=kf, k=key: self.delete_setting(
                          kf, k)).pack(side="left", padx=(2, 0))
        elif status == "animated":
            ttk.Label(btns, text="(animated by neighbors)",
                     foreground="#666", font=self.small_font).pack(
                side="left", padx=(4, 0))

    def save_modified_settings(self):
        """Save into the selected keyframe every setting marked modified."""
        kf = self.selected_keyframe
        if kf is None or not self.kf_modified:
            return
        self.save_settings(kf, sorted(self.kf_modified))

    def delete_all_diffs(self):
        """Delete every setting the selected keyframe defines, leaving it
        empty (every setting then falls back to being animated by neighbors,
        or undefined, just like a freshly emptied single setting would)."""
        kf = self.selected_keyframe
        if kf is None:
            return
        kf.data.clear()
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

    def update_kf_detail_panel(self):
        kf = self.selected_keyframe
        if kf is None:
            self.kf_detail_label_var.set("(none selected)")
            self.kf_frame_var.set("")
            self.kf_frame_entry.state(["disabled"])
            self.kf_split_btn.state(["disabled"])
            self.kf_delete_btn.state(["disabled"])
            return
        idx = self.keyframes.index(kf)
        self.kf_detail_label_var.set(f"Keyframe #{idx + 1}")
        self.kf_frame_var.set(str(kf.frame_index))
        self.kf_frame_entry.state(["!disabled"])
        self.kf_split_btn.state(["!disabled"])
        self.kf_delete_btn.state(["!disabled"])

    def on_kf_frame_commit(self, _event=None):
        kf = self.selected_keyframe
        if kf is None:
            return
        text = self.kf_frame_var.get()
        if text == "":
            self.kf_frame_var.set(str(kf.frame_index))  # every keyframe needs an index
            return
        kf.frame_index = int(text)
        self.reorder_keyframes()
        self.rebuild_keyframe_list()

    def reorder_keyframes(self):
        """Sort keyframes ascending by frame index (stable, so ties keep their
        relative order)."""
        self.keyframes.sort(key=lambda kf: kf.frame_index)

    def split_selected_keyframe(self):
        kf = self.selected_keyframe
        if kf is None:
            return
        new_kf = Keyframe(self._kf_counter, kf.frame_index, dict(kf.data))
        self._kf_counter += 1
        idx = self.keyframes.index(kf)
        self.keyframes.insert(idx + 1, new_kf)
        self.rebuild_keyframe_list()

    def delete_selected_keyframe(self):
        kf = self.selected_keyframe
        if kf is None:
            return
        self.keyframes.remove(kf)
        self.deselect_keyframe()

    # ---- video rendering ------------------------------------------------------ #
    def set_video_status(self, text):
        """Replace the scrollable render-status text (so long messages, like a
        full error, are never cut off - just scroll)."""
        self.video_status_text.configure(state="normal")
        self.video_status_text.delete("1.0", "end")
        self.video_status_text.insert("1.0", text)
        self.video_status_text.configure(state="disabled")

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
        settings = data.settings
        fz = fuzz_rule(settings.fuzz)
        stats = np.zeros((out_h, out_w, N_STATS), dtype=np.float32)
        points = np.zeros((out_h, out_w, N_POINTS, 2), dtype=np.float32)
        status = np.zeros((out_h, out_w), dtype=np.uint8)
        render_rows_wh(stats, points, status, 0, out_h, out_w, out_h, settings.max_iter,
                       effective_dist(settings.max_dist), settings.seg_mult, fz,
                       settings_confine(settings), settings_emission(settings), obst_arrays,
                       settings.fuzz.seed)

        images = [self.colorize_view(v, stats, points, status, out_w, out_h, settings, fz,
                                     obst_arrays)
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

    def colorize_view(self, v, stats, points, status, out_w, out_h, settings, fz,
                      obst_arrays):
        """One view's (out_h, out_w, 3) uint8 image (background color where no
        ray was computed, no outline - that's applied uniformly afterwards)
        from already-traced stats/points/status. `v` is a ViewSettings: this
        is the same coloring computation every view goes through, whichever
        one it is."""
        mapping = make_mapping(v.kinds, v.point_metric, v.other_metric, v.chan_src,
                               v.constant_choice)
        img = np.empty((out_h, out_w, 3), dtype=np.uint8)
        img[:] = BACKGROUND
        computed = status == 1
        n = int(computed.sum())
        if n:
            radius_px = world_radius_to_pixels(settings.buddha_blur.radius, out_w, out_h)
            buddha_kernel = build_buddha_blur_kernel(radius_px, settings.buddha_blur.type)
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
                            0, out_h, settings.max_iter, effective_dist(settings.max_dist),
                            settings.seg_mult, fz, settings_confine(settings),
                            settings_emission(settings), obst_arrays, settings.fuzz.seed,
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
        mini_settings = self.settings_of_current_image()
        obst_arrays = build_obstacle_arrays(self.obstacles)
        full = self.colorize_view(v, self.stats, self.points, self.status, SIZE, SIZE,
                                  mini_settings, self.gen_fuzz, obst_arrays)
        ys = (np.arange(size) * SIZE / size).astype(np.int64)
        xs = (np.arange(size) * SIZE / size).astype(np.int64)
        thumb = full[np.ix_(ys, xs)]
        self.view_thumb_cache[view_obj.id] = ((self.rows_done, size), thumb)
        return thumb

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
            self.set_video_status("Cancelling (finishing the current frame)...")
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

        frame_lo, frame_hi = min(indices), max(indices)
        if self.skip_first_frame_var.get():
            frame_lo += 1
            if frame_lo > frame_hi:
                messagebox.showerror("Nothing to render",
                                     "Skipping the first frame would leave no "
                                     "frames to render - add another keyframe "
                                     "or uncheck 'Skip first frame'.")
                return
        kf_snapshot = snapshot_keyframes(self.keyframes)   # independent copies: edits made
                                                   # while rendering must not leak in
        obst_snapshot = [o.snapshot() for o in self.obstacles]
        view_snapshot = [snapshot_view(v) for v in self.views_to_export()]
        color_location = self.color_info_location_var.get()
        figure_location = self.figure_info_location_var.get()

        self.render_cancel = threading.Event()
        self.render_btn.configure(text="Cancel Render")
        self.reencode_btn.state(["disabled"])
        self.clear_btn.state(["disabled"])
        self.set_video_status("Starting...")
        total = frame_hi - frame_lo + 1
        self.video_progress.configure(maximum=total)
        self.video_progress["value"] = 0
        self.render_thread = threading.Thread(
            target=self._render_worker,
            args=(kf_snapshot, frame_lo, frame_hi, rs, obst_snapshot, view_snapshot,
                 color_location, figure_location),
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
            self.root.after(0, lambda: self.set_video_status("Converting to JXL with cjxl..."))
            try:
                convert_apng_to_jxl(jxl_apng_temp, out_path, rs["jxl_lossless"],
                                    rs["jxl_quality"])
            finally:
                if os.path.exists(jxl_apng_temp):
                    os.remove(jxl_apng_temp)

    def _render_worker(self, keyframes, frame_lo, frame_hi, rs, obst_snapshot, view_snapshot,
                       color_location, figure_location):
        """Runs on a background thread: never touches a Tk widget directly, only
        via self.root.after(0, ...)."""
        ext, suffix = self.output_extension_and_suffix(rs)
        out_name = f"{paths.PROGRAM_NAME}-video-{time.strftime('%Y%m%d-%H%M%S')}{suffix}.{ext}"
        out_path = os.path.join(ensure_export_dir(), out_name)
        pattern = os.path.join(paths.TEMP_DIR, "frame_%05d.png")
        scratch_dir = None
        try:
            assigned_sorted = sorted(keyframes, key=lambda k: k.frame_index)
            total = frame_hi - frame_lo + 1
            for n, fn in enumerate(range(frame_lo, frame_hi + 1)):
                if self.render_cancel.is_set():
                    self.root.after(0, lambda: self.set_video_status("Cancelled."))
                    return
                data = resolve_frame_settings(assigned_sorted, fn)
                obst_values = resolve_obstacles(obst_snapshot, assigned_sorted, fn)
                obst_arrays = build_obstacle_arrays(obst_values)
                views = resolve_views(view_snapshot, assigned_sorted, fn)
                rgb = self.render_full_image(data, rs["video_w"], rs["video_h"], obst_arrays,
                                             views)
                # The figure's settings can be keyframe-animated, so this is
                # resolved per frame - but the margin's own size depends only on
                # the fixed video_w/video_h and the number of obstacles, so every
                # frame still comes out the same size.
                rgb = apply_info_locations(rgb, views, rs["video_w"],
                                           data.settings, obst_values,
                                           color_location, figure_location, mutate_ok=True)
                path = os.path.join(paths.TEMP_DIR, f"frame_{fn - frame_lo:05d}.png")
                write_png(path, rgb)
                done = n + 1
                self.root.after(0, lambda d=done, t=total, im=rgb:
                                self._update_render_progress(d, t, im))

            if rs["downsample_factor"] > 1 and rs["downsample_method"] != "ffmpeg":
                self.root.after(0, lambda: self.set_video_status("Downsampling frames..."))
                pattern, start_number, scratch_dir = self.prepare_ffmpeg_source(0, total, rs)
            else:
                start_number = 0
            self.root.after(0, lambda: self.set_video_status("Encoding video with ffmpeg..."))
            self.encode_video_output(pattern, start_number, rs, out_path)
            info = self.describe_output_file(out_path, total, rs["fps"])
            self.root.after(0, lambda: self.set_video_status(f"Done: {out_path}\n{info}"))
        except Exception as exc:                    # noqa: BLE001
            detail = self.ffmpeg_error_detail(exc, out_path, pattern)
            self.root.after(0, lambda m=detail: self.set_video_status(f"Render failed: {m}"))
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
        self.set_video_status(f"Re-encoding {len(frame_files)} existing frame(s)...")
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
                self.root.after(0, lambda: self.set_video_status("Downsampling frames..."))
                pattern, start_number, scratch_dir = self.prepare_ffmpeg_source(
                    start_number, num_frames, rs)
            self.root.after(0, lambda: self.set_video_status("Encoding video with ffmpeg..."))
            self.encode_video_output(pattern, start_number, rs, out_path)
            info = self.describe_output_file(out_path, num_frames, rs["fps"])
            self.root.after(0, lambda: self.set_video_status(f"Done: {out_path}\n{info}"))
        except Exception as exc:                    # noqa: BLE001
            detail = self.ffmpeg_error_detail(exc, out_path, pattern)
            self.root.after(0, lambda m=detail: self.set_video_status(f"Render failed: {m}"))
        finally:
            self.cleanup_scratch_dir(scratch_dir)
            self.root.after(0, self._render_finished)

    def _update_render_progress(self, done, total, latest_frame):
        self.video_progress.configure(maximum=total)
        self.video_progress["value"] = done
        self.set_video_status(f"Rendering frames: {done}/{total}")
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
            self.set_video_status(f"Cleared {deleted} frame file(s) from TEMP.")

    def _render_finished(self):
        self.render_thread = None
        self.render_btn.configure(text="Start Render")
        self.render_btn.state(["!disabled"])
        self.reencode_btn.state(["!disabled"])
        self.clear_btn.state(["!disabled"])
        self.set_preview_image(None)              # back to the placeholder


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
