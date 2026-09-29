#!/usr/bin/env python3
"""
Light-bounce image / video generator.

A bounding shape centred on the origin sits in a 512x512 preview image that
spans [-1, 1] in both directions (the simulation itself always runs in that
fixed square, whatever resolution is later exported). The shape is one of
eight, each with its own size/angle parameters and rotation:

  * an ellipse - the defaults (2.0 x 2.0, no rotation) give the unit circle,
    which fills the image,
  * a regular polygon (number of sides), inscribed in the unit circle,
  * a Moss's egg, a circular sector, a circular segment, a two-circle lens,
    a Reuleaux polygon (odd number of sides), or a capsule.

Ray effects (bounding shape and obstacles alike)
  What happens to a ray as it reaches the edge of a shape depends on which
  way it is crossing: coming in from outside uses the shape's "incoming ray
  effect", going out from inside its "outgoing ray effect". Each is one of

    Reflect       a mirror (using the boundary's fuzzing settings),
    Terminate     the ray stops the instant it reaches the edge,
    Pass through  the ray carries on across the edge. This is neither a bounce
                  nor a recorded path point, and it doesn't cut the segment
                  short: the ray simply becomes inside (or outside) that shape.

  Reflect both ways makes a container that rays never leave; Pass through
  both ways makes the shape invisible to rays; incoming Pass through with
  outgoing Reflect lets rays in but never out. The bounding shape's controls
  default to Reflect / Reflect (a ray inside it is bounced back, as it always
  was); a new obstacle defaults to incoming Reflect, outgoing Pass through.

Obstacles
  Any number of extra shapes (the same eight kinds), each placed by a polar
  position (r, theta) from the origin plus an x/y offset, each with its own
  incoming and outgoing ray effects and its own "Fire rays inside" box.

Rays
  A ray starts at the centre of every pixel that is inside ANY shape - the
  bounding shape or an obstacle - whose "Fire rays inside" box is ticked (the
  bounding shape's is ticked by default). Being inside a shape whose box is
  clear makes no difference. The ray travels up and to the right (45 degrees,
  unless fuzzing perturbs its emission angle - see below), starting out inside
  every shape that contains its start point. It reflects, terminates or passes
  through at each shape's edge as described above, and it reflects off any
  earlier segment of its own path. A ray stops when

  * it becomes confined to a region smaller than one pixel,
  * it has bounced as many times as the iteration limit allows (1 to 9999),
  * its total path length reaches the distance limit (in units, where one unit
    is half the image width; 0, the default, means no distance limit). If the
    limit is reached part-way along a segment, the segment is cut off there,
  * it reaches an edge whose effect for that direction is Terminate,
  * it strikes a reflective surface at exactly perpendicular incidence (which
    side it would retrace along is undefined, so it stops there), or
  * it escapes: nothing is left ahead for it to hit, so its last point is
    where it leaves the [-1, 1] image square.

  A segment length multiplier (0.01 to 1; 1 means no effect) shortens every
  ordinary bounce's segment to that fraction of its true length, after which
  the ray continues in the newly bounced direction from where it stopped. A
  segment includes any edges the ray passed through on the way.

Fuzzing
  Optional randomness/offsets applied to three kinds of angle: the ray's
  emission angle (once, when it leaves its start pixel - a plain angle slider,
  defaulting to 45 degrees, the unfuzzed direction, plus jitter), the
  deflection angle of each bounce off any reflecting edge (of the bounding
  shape or an obstacle), and the deflection angle of each bounce off the
  ray's own earlier
  path (each of the latter two with an offset, a jitter range and a
  multiplier). A seed makes the randomness reproducible. The whole tab is
  greyed out while fuzzing is off.

Coloration
  For every ray, thirteen metrics are measured (each scaled to 0..1): four
  that compare two chosen sample points (horizontal, vertical, Manhattan and
  Euclidean distance) and nine others (bounce counts and fractions, loop area,
  bounce angles, path length, last-segment direction and length, and why the
  ray stopped - one of six reasons, spaced evenly from 0 to 1). Fifteen sample points can be chosen: thirteen picked out of the
  ray's path (its start and end, the means of its bounce points, its
  uppermost/lowermost/rightmost/leftmost bounces overall and among only its
  self-bounces) plus two fixed ones (the origin and the top-left corner).

  A view is a color model (RGB or HSV) plus three channels. Each channel has a
  source:
    Query     a metric, evaluated once per ray for the pixel it was emitted
              from - either a point comparison (with its two sample points)
              or one of the other metrics,
    Buddha    named after the Buddhabrot: characteristic points of every ray
              are scattered into a visit-count histogram, and a pixel's
              brightness comes from how often its own location was landed on
              by any ray. Any number of point kinds can be scattered at once
              (means, termination points by reason, every self-bounce, every
              segment midpoint, progressive means). Counts are compressed to
              0..1 (arctangent or modulus) with an optional brightness boost,
              and the Buddha blur tab spreads each scattered point over a
              neighborhood (Gaussian, triangular, bokeh or catadioptric
              bokeh; radius 0 means no blur),
    Constant  always 0 or always 1, taking no part in the simulation.
  Per-pixel results are stored, so changing a view's coloration recolors the
  image without regenerating it (Buddha channels rebuild their histograms
  first; that never restarts the generation either).

  Several views can exist at once, listed in the Coloration tab. Whichever
  one is selected is the one shown in the preview (with its name overlaid,
  in the preview only); exports and videos can include all of them, side by
  side, so more than three channels' worth of information can be shown at once.

Decorations
  The outline of the bounding shape (white) and of each obstacle is drawn into
  the image and so appears in exports. An obstacle's outline has two layers:
  the outer one shows its incoming ray effect and the inner one its outgoing
  effect (pale blue for Reflect, brown for Terminate, black for Pass through).
  Everything else (the dashed outline of not-yet-generated shape
  changes, the previewed ray's path, markers and guide lines, the color
  sample points) is drawn over the preview only and never saved.

Timeline and video
  A keyframe has a frame index, an interpolation method (Linear or Step), a
  small thumbnail, and a set of settings - any subset of every setting in the
  Simulation, Fuzzing, Buddha blur and Decorations tabs plus every
  obstacle's and every view's own fields. "New keyframe" defines every
  setting from the controls' current values; afterwards, each setting can be
  loaded into the controls, saved from them, or deleted from that keyframe
  individually (or all at once), so different settings can start and stop
  animating at different points on the timeline. Keyframes are managed
  independently of the live controls: selecting one doesn't touch the
  controls, and changing the controls doesn't touch any keyframe. "Split"
  makes a duplicate right after the original, with the same frame index;
  editing the selected keyframe's frame-index field re-sorts the list. Every
  keyframe always has a frame index (0 for the first one created, one more
  than the highest existing index for each new one after that). A keyframe's
  thumbnail is a small bilinearly-resized copy of whatever the image panel
  showed at the moment it was saved, or a black "no img" placeholder if that
  image didn't match the current settings.

  Each setting is resolved independently, using only the keyframes that
  define it. Between two of them, float-slider settings (shape dimensions and
  rotations, fuzzing angles/jitters/multipliers, segment multiplier, blur
  radius, obstacle geometry and position, and so on) are interpolated using
  the method chosen on the later keyframe; everything else (which bounding
  shape, which metric, colors, and so on) holds the earlier keyframe's value
  until the later one is reached, since it isn't a number that could be
  blended.

  A render walks every whole frame from the lowest to the highest frame index
  (optionally skipping the first), draws every frame at the chosen width and
  height (the same world area as the preview), and writes each as a PNG into
  exports/TEMP, next to this program. It then combines them with ffmpeg into
  an .mp4 (h264), .gif, .apng or .mkv (ffv1), or, through the external cjxl
  tool, an animated .jxl, saved in the exports folder; the bitrate mode
  (constant quality, target or variable bitrate) and a downsampling factor
  apply to that combined video only, never to the PNG frames. "Re-encode"
  combines whatever frames are already in TEMP using the current encoding
  settings. A render refuses to start if two keyframes share a frame index or
  if the TEMP folder already has any files in it. This program never deletes a
  file on its own; the one exception is the Video panel's Clear button, which,
  after a confirmation, deletes only frame_NNNNN.png files from TEMP.

Exporting a still image
  Export designer: whether the export includes all views or only the selected
  one, where to put the coloration description (none, overlaid on the image,
  or in a bottom margin) and where to put the simulation description (none,
  or a left margin); these apply to videos as well. Export: either the
  Current figure (the preview as it was generated) or a New figure generated
  afresh at its own width and height (per view), saved as png, qoi or jxl
  (lossless, or lossy with a quality setting). "Export" saves into the
  exports folder (next to the program) under a default name; "Export As..." asks for one. Both are
  disabled until an image has finished generating.

Window layout (six panels, in three columns)
  top left       the generated image (512x512). Hover to preview a ray; click
                 to pin the preview to that pixel, freeing the mouse.
  top middle     Controls, in six tabs: Simulation (which scrolls), Fuzzing,
                 Coloration, Buddha blur, Decorations and Obstacles; Start
                 and Revert settings are always visible
  top right      Timeline: the keyframe list (with thumbnails), each
                 keyframe's settings, and the selected keyframe's controls
  bottom left    Ray preview: the previewed ray, in Numbers and Graphics tabs
  bottom middle  the Export designer, and Export
  bottom right   Video: render settings (size, skip first frame), encoding
                 settings (fps, format, bitrate, downsampling), a preview of
                 the latest frame, and Start Render / Re-encode / Clear

Saved image files (all written to the exports folder) are named
  <program name>-YYYYMMDD-hhmmss-<simulation settings>.<png|qoi|jxl>
and rendered videos
  <program name>-video-YYYYMMDD-hhmmss.<mp4|gif|apng|jxl>  (.ffv1.mkv for mkv)
where the program name is taken from this file's own name (so it follows any
renaming).

Requirements:  pip install numpy numba ffmpeg-python   (tkinter ships with
Python). The ffmpeg command-line tool must also be installed and on PATH for
video rendering and for saving a still as JPEG XL; the cjxl command-line tool
(part of libjxl) is additionally needed for animated JXL video. All three are
otherwise unnecessary: PNG and QOI stills need only the packages above.
Pillow is optional too, and used only for the development startup screenshot.

Development startup screenshot
  If a dev folder exists next to this program, then once per launch (shortly
  after the numba warm-up finishes) the program grabs a picture of its own
  window into dev/screenshots/<file version>-YYYYMMDD-hhmmss.png. There is no
  button for it, and nothing happens if the dev folder or Pillow is missing.
"""

# FILE_VERSION increments by exactly 1 each time work on this file is
# handed back to the user as finished, counting from 1 - once per
# completed task/turn that changed the file, not once per individual edit
# within it. It is a delivery counter, not a semantic/feature version - it
# says nothing about what changed.
FILE_VERSION = 78

import os
import random
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import zlib
from collections import namedtuple
from math import acos, atan, atan2, ceil, cos, floor, pi, radians, sin, sqrt, tan
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # works, but far too slowly to be practical
    print("WARNING: numba not found - falling back to pure Python (very slow).")
    print("         Install it with:  pip install numba")

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    prange = range

try:
    import ffmpeg               # the ffmpeg-python package
except ImportError:
    ffmpeg = None                # video rendering is disabled; stills still work

try:
    from PIL import ImageGrab   # Pillow: only used for the development startup screenshot
except ImportError:
    ImageGrab = None             # no startup screenshot; nothing else needs Pillow

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
SIZE = 512                 # image size in pixels
PIXEL = 2.0 / SIZE         # width of one pixel in world units (image spans [-1, 1])
ITER_LIMIT_MIN = 1         # allowed range (whole numbers) for the iteration limit
ITER_LIMIT_MAX = 9999
DIST_LIMIT_MIN = 0         # allowed range for the distance limit; 0 = no limit
DIST_LIMIT_MAX = 9999
DEFAULT_ITER = 32          # default iteration limit (bounces)
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


# Why a ray stopped (returned by trace_path as `term`)
TERM_CONFINED = 0   # confined to less than one pixel
TERM_ITER = 1       # iteration limit reached
TERM_DIST = 2       # distance limit reached
TERM_OBSTACLE = 3   # hit an obstacle whose ray effect is "terminate"
TERM_PERPENDICULAR = 4   # hit a reflective surface at exactly perpendicular
                         # incidence (which side it would retrace along is
                         # undefined - see trace_path)
TERM_ESCAPED = 5         # nothing left ahead to hit: the ray leaves for good (its
                         # last point is where it leaves the [-1, 1] image square)

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

# Bounding shapes
SHAPE_ELLIPSE = 0
SHAPE_POLYGON = 1
SHAPE_MOSS_EGG = 2
SHAPE_SECTOR = 3
SHAPE_SEGMENT = 4
SHAPE_LENS = 5
SHAPE_REULEAUX = 6
SHAPE_CAPSULE = 7
SHAPE_NAMES = ["Ellipse", "Regular polygon", "Moss's egg", "Circular sector",
              "Circular segment", "Two-circle lens", "Reuleaux polygon", "Capsule"]
DEFAULT_SHAPE = SHAPE_ELLIPSE

# Ellipse controls: full width / height (0..2), rotation in degrees
ELLIPSE_MIN = 0.0
ELLIPSE_MAX = 2.0
DEFAULT_WIDTH = 2.0        # 2.0 x 2.0, no rotation = the unit circle
DEFAULT_HEIGHT = 2.0
DEFAULT_ELL_ROTATION = 0.0
AXIS_MIN = 1e-3            # smallest semi-axis used in the maths (avoids 1/0)

# Regular-polygon controls (circumradius is fixed at 1: vertices on the unit circle)
POLY_MIN_SIDES = 3
POLY_MAX_SIDES = 64
DEFAULT_SIDES = 5
DEFAULT_POLY_ROTATION = 0.0

ROTATION_MIN = -90.0       # covers every orientation: an ellipse repeats every
ROTATION_MAX = 90.0        # 180 degrees, a regular polygon at least every 120

# Moss's egg: apex angle ACB (the isosceles triangle's apex angle), size
MOSS_ANGLE_MIN = 60.0      # below this the construction degenerates further
MOSS_ANGLE_MAX = 180.0
DEFAULT_MOSS_ANGLE = 90.0  # the classic right-isosceles-triangle egg
MOSS_SIZE_MIN = 0.0
MOSS_SIZE_MAX = 2.0
DEFAULT_MOSS_SIZE = 1.0

# Circular sector / segment: radius, angular span (degrees), rotation
ARC_RADIUS_MIN = 0.0
ARC_RADIUS_MAX = 2.0
DEFAULT_ARC_RADIUS = 1.0
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

# Fuzzing controls
FUZZ_ANGLE_MIN = -180.0    # general angle / offset range, degrees
FUZZ_ANGLE_MAX = 180.0
FUZZ_JITTER_MIN = 0.0      # jitter range (+/-), degrees
FUZZ_JITTER_MAX = 180.0
FUZZ_MULT_MIN = 0.0        # bounce angle multiplier range
FUZZ_MULT_MAX = 3.0
DEFAULT_EMIT_ANGLE = 45.0  # the unfuzzed emission direction (up and to the right)
DEFAULT_FUZZ_SEED = 0
FuzzParams = namedtuple("FuzzParams", [
    "enabled",
    "emit_angle", "emit_jitter",                     # ray emission
    "bnd_offset", "bnd_jitter", "bnd_mult",          # bounces off the boundary
    "self_offset", "self_jitter", "self_mult",       # bounces off the ray itself
    "seed"])
DEFAULT_FUZZ = FuzzParams(False, DEFAULT_EMIT_ANGLE, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                          DEFAULT_FUZZ_SEED)

CONFINE_WINDOW = 4         # a ray is "confined" if its last N bounce points ...
                           # ... all fit inside a box smaller than one pixel
REFRESH_HZ = 10            # redraws per second during generation
COMPUTE_FRACTION = 0.55    # share of each refresh period spent computing
EPS = 1e-9
SELF_TIE_EPS = 1e-7        # a self-segment collision must beat a boundary/obstacle
                          # collision by more than this to override it (avoids
                          # floating-point noise letting a self-intersection
                          # "win" a near-tie against a solid wall - see trace_path)
RETRO_EPS = 1e-9          # how close to EXACTLY perpendicular (a ray reflecting
                          # straight back the way it came) counts as "perfectly
                          # perpendicular" - see trace_path
INV_SQRT2 = 1.0 / sqrt(2.0)   # direction (1, 1) / sqrt(2): up and to the right (45 deg)

BACKGROUND = (12, 12, 16)
RING_HALF_WIDTH_PX = 0.7   # thickness of the white shape outline
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
KIND_START = 0.0           # a point's "kind", used only for the segment graphic
KIND_BOUNDARY = 1.0
KIND_SELF = 2.0
KIND_DISTCUT = 3.0
KIND_OBSTACLE_TERM = 4.0   # the point where a "terminate" obstacle ended the ray
KIND_ESCAPED = 5.0         # the point where an escaping ray leaves the image square
KIND_COLOR = {KIND_START: (67, 160, 71), KIND_DISTCUT: (67, 160, 71),   # green
             KIND_BOUNDARY: (229, 57, 53),                              # red
             KIND_SELF: (30, 136, 229),                                 # blue
             KIND_OBSTACLE_TERM: (255, 235, 59),                        # yellow
             KIND_ESCAPED: (0, 229, 255)}                               # cyan

# Solid outline colors for obstacles (part of the saved image/video, drawn with
# the same "outline" decoration flag as the main boundary), by ray effect
OBST_TERMINATE_RGB = (121, 85, 72)     # brown
OBST_REFLECT_RGB = (179, 229, 252)     # pale light blue
OBST_PASS_RGB = (0, 0, 0)              # black
OBST_TERMINATE_HEX = "#795548"
OBST_REFLECT_HEX = "#b3e5fc"
OBST_PASS_HEX = "#000000"
# An obstacle's outline is two layers, each this many pixels thick: the outer
# one (just outside its edge) shows its incoming effect, the inner one (just
# inside) its outgoing effect.
OBST_RING_LAYER_PX = 1.0

# What happens to a ray as it crosses an obstacle's edge - chosen separately for
# a ray coming in from outside ("incoming") and one going out from inside
# ("outgoing"). Passing through in both directions is as if the obstacle
# weren't there at all.
OBSTACLE_EFFECT_REFLECT = 0
OBSTACLE_EFFECT_TERMINATE = 1
OBSTACLE_EFFECT_PASS = 2
OBSTACLE_EFFECT_NAMES = ["Reflect", "Terminate", "Pass through"]
OBST_EFFECT_RGB = {OBSTACLE_EFFECT_REFLECT: OBST_REFLECT_RGB,
                   OBSTACLE_EFFECT_TERMINATE: OBST_TERMINATE_RGB,
                   OBSTACLE_EFFECT_PASS: OBST_PASS_RGB}

# Keyframe thumbnails
THUMB_SIZE = 48            # small resolution, per pixel
THUMB_BACKGROUND = (110, 110, 110)   # obstacle thumbnails: background and shape fill
THUMB_FILL = (150, 150, 150)
PREVIEW_SIZE = 150         # side length of the video-panel frame preview
INTERP_METHODS = ["Linear", "Step"]
DEFAULT_INTERP = "Linear"
SELECTED_OUTLINE = "#ffeb3b"       # yellow
SELECTED_OUTLINE_PX = 4

# A tiny 5x7 bitmap font, used for the "no img"/"preview will go here"
# placeholders and for the view-name overlay on the live preview
FONT_5X7 = {
    "n": ["00000", "11010", "10110", "10010", "10010", "10010", "00000"],
    "o": ["00000", "01110", "10001", "10001", "10001", "01110", "00000"],
    "i": ["00000", "00100", "00000", "00100", "00100", "00100", "00000"],
    "m": ["00000", "11011", "10101", "10101", "10101", "10101", "00000"],
    "g": ["00000", "01111", "10001", "10001", "01111", "00001", "01110"],
    "p": ["00000", "11110", "10001", "11110", "10000", "10000", "00000"],
    "r": ["00000", "11110", "10001", "11110", "10100", "10010", "00000"],
    "e": ["00000", "01111", "10000", "11110", "10000", "01111", "00000"],
    "v": ["00000", "10001", "10001", "10001", "01010", "00100", "00000"],
    "w": ["00000", "10001", "10001", "10101", "10101", "01010", "00000"],
    "l": ["00000", "10000", "10000", "10000", "10000", "01111", "00000"],
    "h": ["00000", "10001", "10001", "11111", "10001", "10001", "00000"],
    "a": ["00000", "01110", "10001", "01111", "10001", "01111", "00000"],
    "b": ["10000", "10000", "11110", "10001", "10001", "11110", "00000"],
    "c": ["00000", "00000", "01110", "10000", "10000", "01110", "00000"],
    "d": ["00001", "00001", "01111", "10001", "10001", "01111", "00000"],
    "f": ["00110", "01000", "11110", "01000", "01000", "01000", "00000"],
    "s": ["00000", "01111", "10000", "01110", "00001", "11110", "00000"],
    "t": ["01000", "01000", "11110", "01000", "01000", "00110", "00000"],
    "u": ["00000", "10001", "10001", "10001", "10001", "01111", "00000"],
    "x": ["00000", "10001", "01010", "00100", "01010", "10001", "00000"],
    "y": ["00000", "10001", "10001", "01111", "00001", "00001", "01110"],
    "z": ["00000", "11111", "00010", "00100", "01000", "11111", "00000"],
    "0": ["00000", "01110", "10001", "10001", "10001", "01110", "00000"],
    "1": ["00000", "00100", "01100", "00100", "00100", "01110", "00000"],
    "2": ["00000", "01110", "10001", "00110", "01000", "11111", "00000"],
    "3": ["00000", "11110", "00001", "00110", "00001", "11110", "00000"],
    "4": ["00000", "00110", "01010", "10010", "11111", "00010", "00000"],
    "5": ["00000", "11111", "10000", "11110", "00001", "01110", "00000"],
    "6": ["00000", "01110", "10000", "11110", "10001", "01110", "00000"],
    "7": ["00000", "11111", "00001", "00010", "00100", "00100", "00000"],
    "8": ["00000", "01110", "10001", "01110", "10001", "01110", "00000"],
    "9": ["00000", "01110", "10001", "01111", "00001", "01110", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    # Uppercase: given the fuller row-0..5 height (6 rows), reading visually
    # taller than the lowercase letters above (which use only rows 1..5).
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "00000"],
    "B": ["11110", "10001", "11110", "10001", "10001", "11110", "00000"],
    "C": ["01110", "10001", "10000", "10000", "10001", "01110", "00000"],
    "D": ["11110", "10001", "10001", "10001", "10001", "11110", "00000"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "00000"],
    "I": ["11111", "00100", "00100", "00100", "00100", "11111", "00000"],
    "O": ["01110", "10001", "10001", "10001", "10001", "01110", "00000"],
    "E": ["11111", "10000", "11110", "10000", "10000", "11111", "00000"],
    "G": ["01110", "10000", "10000", "10111", "10001", "01110", "00000"],
    "H": ["10001", "10001", "11111", "10001", "10001", "10001", "00000"],
    "L": ["10000", "10000", "10000", "10000", "10000", "11111", "00000"],
    "M": ["10001", "11011", "10101", "10001", "10001", "10001", "00000"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "00000"],
    "Q": ["01110", "10001", "10001", "10101", "10010", "01101", "00000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "00000"],
    "S": ["01111", "10000", "01110", "00001", "00001", "11110", "00000"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00000"],
    "V": ["10001", "10001", "10001", "10001", "01010", "00100", "00000"],
    # Punctuation
    "(": ["00100", "01000", "10000", "10000", "10000", "01000", "00100"],
    ")": ["00100", "00010", "00001", "00001", "00001", "00010", "00100"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "00100", "01000", "10000", "00000"],
    ":": ["00000", "00100", "00000", "00000", "00100", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00000"],
    ",": ["00000", "00000", "00000", "00000", "00100", "00100", "01000"],
    "'": ["01000", "01000", "00000", "00000", "00000", "00000", "00000"],
    "\u00b7": ["00000", "00000", "00000", "00100", "00000", "00000", "00000"],
    "\u00b0": ["01100", "10010", "01100", "00000", "00000", "00000", "00000"],
    "\u00d7": ["00000", "10001", "01010", "00100", "01010", "10001", "00000"],
    "\u00b1": ["00000", "00100", "01110", "00100", "00000", "11111", "00000"],
}

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

# Shown under both the bounding shape's ray controls and an obstacle's
RAY_HELP_TEXT = ("Incoming: what happens to a ray reaching this shape's edge from "
                 "outside; outgoing: from inside. Pass through both ways and rays "
                 "ignore the shape. A ray is fired from every pixel that is inside "
                 "any shape (this one, or an obstacle) with 'Fire rays inside' "
                 "ticked; a ray with nothing left to hit escapes.")

# Names / paths (based on the file actually being run, even if renamed)
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PROGRAM_NAME = os.path.splitext(os.path.basename(SCRIPT_PATH))[0]
# Everything the program writes (exported images and videos) goes in EXPORT_DIR;
# the video frames' TEMP folder lives inside it.
EXPORT_DIR = os.path.join(SCRIPT_DIR, "exports")
# Development aid: if this folder exists, the app screenshots its own window once
# per launch into DEV_DIR/screenshots (see take_ui_screenshot).
DEV_DIR = os.path.join(SCRIPT_DIR, "dev")
TEMP_DIR = os.path.join(EXPORT_DIR, "TEMP")

# Snapshot: every value of every simulation control, including the parameters of
# every shape that is not currently selected. This is what "Revert" restores, and
# what a keyframe's `snapshot` field holds.
Snapshot = namedtuple("Snapshot", [
    "max_iter", "max_dist", "seg_mult", "shape", "width", "height", "ell_rotation",
    "sides", "poly_rotation", "moss_angle", "moss_size", "moss_rotation", "sector_radius",
    "sector_span", "sector_rotation", "segment_radius", "segment_span",
    "segment_rotation", "lens_radius1", "lens_radius2", "lens_sep", "lens_rotation",
    "reuleaux_sides", "reuleaux_rotation", "capsule_length", "capsule_radius",
    "capsule_rotation", "fuzz", "buddha_blur_radius", "buddha_blur_type",
    "main_fire", "main_in_effect", "main_out_effect"],
    defaults=[True, 0, 0])   # main boundary: fires rays inside, reflects in and out (0 = Reflect)

# Settings: only what actually determines the generated image. `params` holds the
# selected shape's own parameters (a different-length tuple per shape - see
# effective_settings). Start compares these.
Settings = namedtuple("Settings", ["max_iter", "max_dist", "seg_mult", "shape", "params",
                                   "fuzz", "buddha_blur_radius", "buddha_blur_type",
                                   "main_fire", "main_in_effect", "main_out_effect"],
                      defaults=[True, 0, 0])

DEFAULT_SNAPSHOT = Snapshot(
    max_iter=DEFAULT_ITER, max_dist=DEFAULT_DIST, seg_mult=DEFAULT_SEGMENT_MULT,
    shape=DEFAULT_SHAPE, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
    ell_rotation=DEFAULT_ELL_ROTATION, sides=DEFAULT_SIDES,
    poly_rotation=DEFAULT_POLY_ROTATION, moss_angle=DEFAULT_MOSS_ANGLE,
    moss_size=DEFAULT_MOSS_SIZE, moss_rotation=0.0, sector_radius=DEFAULT_ARC_RADIUS,
    sector_span=DEFAULT_SECTOR_SPAN, sector_rotation=0.0,
    segment_radius=DEFAULT_ARC_RADIUS, segment_span=DEFAULT_SEGMENT_SPAN,
    segment_rotation=0.0, lens_radius1=DEFAULT_LENS_RADIUS,
    lens_radius2=DEFAULT_LENS_RADIUS, lens_sep=DEFAULT_LENS_SEP, lens_rotation=0.0,
    reuleaux_sides=DEFAULT_REULEAUX_SIDES, reuleaux_rotation=0.0,
    capsule_length=DEFAULT_CAPSULE_LENGTH, capsule_radius=DEFAULT_CAPSULE_RADIUS,
    capsule_rotation=0.0, fuzz=DEFAULT_FUZZ,
    buddha_blur_radius=DEFAULT_BUDDHA_BLUR_RADIUS, buddha_blur_type=DEFAULT_BUDDHA_BLUR_TYPE)

# A resolved frame's complete simulation settings (what render_full_image
# needs, aside from the views and obstacles - each resolved separately, the
# same way: a fixed set established when the render starts, each one's own
# fields independently held/animated from the keyframes). Built by
# resolve_frame_settings() from whichever keyframes define each setting.
KeyframeData = namedtuple("KeyframeData", ["snapshot", "outline"])


def make_scrollable_tab(tab):
    """Makes a notebook tab's contents vertically scrollable: returns
    (canvas, inner_frame), where widgets that would have gone straight into
    `tab` should go into `inner_frame` instead. Contents keep their natural
    height rather than being squeezed to fit the tab, and the width follows
    the tab's."""
    tab.rowconfigure(0, weight=1)
    tab.columnconfigure(0, weight=1)
    canvas = tk.Canvas(tab, highlightthickness=0)
    vsb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    inner = ttk.Frame(canvas)
    inner.columnconfigure(0, weight=1)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
    return canvas, inner


class Tooltip:
    """A small hover tooltip attached to one widget: shows `text` in a
    borderless popup near the cursor on <Enter>, hides it on <Leave>.
    update(text) changes what it shows later (e.g. as a button's enabled
    state changes) without rebinding anything; an empty text shows nothing."""
    def __init__(self, widget, text=""):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def update(self, text):
        self.text = text
        if self.tip is not None and not text:
            self._hide()

    def _show(self, _event=None):
        if not self.text or self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        ttk.Label(self.tip, text=self.text, background="#ffffe0",
                 relief="solid", borderwidth=1, padx=4, pady=2).pack()

    def _hide(self, _event=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class FloatEntryHandle:
    """A .get()/.set() wrapper around a float Entry, making it a drop-in
    replacement for a tk.Scale everywhere else in the code that calls
    .get()/.set() on one of these controls. With no bounds (the default),
    accepts any float, so many-revolution angles can be typed or animated;
    with lo/hi given, the value is clamped into range - used for settings
    that keep a slider-like range but as a text field (so typing isn't
    interrupted mid-edit the way a live-validating popup would be)."""

    def __init__(self, var, fmt, default, lo=None, hi=None):
        self.var = var
        self.fmt = fmt
        self.lo = lo
        self.hi = hi
        self.last_valid = self._clamp(float(default))
        self.widget = None          # set by add_angle_entry to the ttk.Entry itself

    def _clamp(self, value):
        if self.lo is not None:
            value = max(self.lo, value)
        if self.hi is not None:
            value = min(self.hi, value)
        return value

    def get(self):
        try:
            self.last_valid = self._clamp(float(self.var.get()))
        except (ValueError, tk.TclError):
            pass
        return self.last_valid

    def set(self, val):
        self.last_valid = self._clamp(float(val))
        self.var.set(self.fmt.format(self.last_valid))

    def commit(self):
        """Called on FocusOut/Return: normalize the text, or revert it if
        it's not currently a valid number."""
        try:
            self.last_valid = self._clamp(float(self.var.get()))
        except ValueError:
            pass
        self.var.set(self.fmt.format(self.last_valid))


class Keyframe:
    """One entry in the timeline. `data` is a dict of setting-key -> value -
    only for the settings this keyframe actually defines (see SETTING_INFO);
    a keyframe may define any subset, which is how different settings can
    start/stop animating at different points on the timeline. `interp` is the
    method used for the interval ENDING at this keyframe (i.e. the one
    leading up to it), applied per setting only where that setting is a float
    slider - see SETTING_FLOAT_KEYS."""
    __slots__ = ("id", "frame_index", "data", "interp", "thumb")

    def __init__(self, id_, frame_index, data, interp=DEFAULT_INTERP, thumb=None):
        self.id = id_
        self.frame_index = frame_index
        self.data = data
        self.interp = interp
        self.thumb = thumb


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
    """Independent copies of `keyframes` (own data dicts; no thumbnails, which
    a render never uses), so a running render is unaffected by anything done
    to the real keyframes meanwhile - including the cleanup of a deleted
    obstacle's or view's settings."""
    return [Keyframe(kf.id, kf.frame_index, dict(kf.data), kf.interp) for kf in keyframes]


class Obstacle:
    """One obstacle: another shape that rays can reflect off, be terminated
    by, or pass through - separately for rays crossing its edge from outside
    (in_effect) and from inside (out_effect). Its position is given in polar
    coordinates (r, theta) from the origin, with (cx, cy) as an additional
    plain x/y offset from that polar position."""
    __slots__ = ("id", "shape", "width", "height", "ell_rotation", "sides",
                "poly_rotation", "poly_radius", "moss_angle", "moss_size", "moss_rotation",
                "sector_radius", "sector_span", "sector_rotation", "segment_radius",
                "segment_span", "segment_rotation", "lens_radius1", "lens_radius2",
                "lens_sep", "lens_rotation", "reuleaux_sides", "reuleaux_rotation",
                "reuleaux_radius", "capsule_length", "capsule_radius",
                "capsule_rotation", "r", "theta", "cx", "cy", "in_effect", "out_effect",
                "fire_inside")

    def __init__(self, id_):
        self.id = id_
        self.shape = SHAPE_ELLIPSE
        self.width = 1.0
        self.height = 1.0
        self.ell_rotation = 0.0
        self.sides = DEFAULT_SIDES
        self.poly_rotation = 0.0
        self.poly_radius = 0.5
        self.moss_angle = DEFAULT_MOSS_ANGLE
        self.moss_size = DEFAULT_MOSS_SIZE
        self.moss_rotation = 0.0
        self.sector_radius = DEFAULT_ARC_RADIUS
        self.sector_span = DEFAULT_SECTOR_SPAN
        self.sector_rotation = 0.0
        self.segment_radius = DEFAULT_ARC_RADIUS
        self.segment_span = DEFAULT_SEGMENT_SPAN
        self.segment_rotation = 0.0
        self.lens_radius1 = DEFAULT_LENS_RADIUS
        self.lens_radius2 = DEFAULT_LENS_RADIUS
        self.lens_sep = DEFAULT_LENS_SEP
        self.lens_rotation = 0.0
        self.reuleaux_sides = DEFAULT_REULEAUX_SIDES
        self.reuleaux_rotation = 0.0
        self.reuleaux_radius = 0.5
        self.capsule_length = DEFAULT_CAPSULE_LENGTH
        self.capsule_radius = DEFAULT_CAPSULE_RADIUS
        self.capsule_rotation = 0.0
        self.r = 0.0
        self.theta = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.in_effect = OBSTACLE_EFFECT_REFLECT
        self.out_effect = OBSTACLE_EFFECT_PASS
        self.fire_inside = False

    def params(self):
        """The selected shape's own parameter tuple - see shape_params_tuple,
        shared with the bounding shape's equivalent (snapshot_shape_params).
        include_obstacle_only=True since an obstacle, unlike the bounding
        shape, isn't constrained to the unit circle and so does carry its
        own radius for polygon and Reuleaux shapes."""
        return shape_params_tuple(self.shape, self, include_obstacle_only=True)

    def center(self):
        """World (x, y) of the obstacle's centre: the polar position (r,
        theta) from the origin, plus the plain (cx, cy) offset."""
        th = radians(self.theta)
        return self.r * cos(th) + self.cx, self.r * sin(th) + self.cy


def obstacle_center_from(r, theta, cx, cy):
    """Same as Obstacle.center(), for already-resolved field values (used
    when animating obstacles between keyframes)."""
    th = radians(theta)
    return r * cos(th) + cx, r * sin(th) + cy


def describe_obstacle(o):
    if o.shape == SHAPE_ELLIPSE:
        shape_txt = f"Ellipse {o.width:.2f}\u00d7{o.height:.2f}, rot {o.ell_rotation:.1f}\u00b0"
    elif o.shape == SHAPE_POLYGON:
        shape_txt = f"{o.sides}-gon r{o.poly_radius:.2f}, rot {o.poly_rotation:.1f}\u00b0"
    elif o.shape == SHAPE_MOSS_EGG:
        shape_txt = (f"Moss's egg {o.moss_angle:.0f}\u00b0, size {o.moss_size:.2f}, "
                    f"rot {o.moss_rotation:.1f}\u00b0")
    elif o.shape == SHAPE_SECTOR:
        shape_txt = f"Sector r{o.sector_radius:.2f} {o.sector_span:.0f}\u00b0"
    elif o.shape == SHAPE_SEGMENT:
        shape_txt = f"Segment r{o.segment_radius:.2f} {o.segment_span:.0f}\u00b0"
    elif o.shape == SHAPE_LENS:
        shape_txt = f"Lens r{o.lens_radius1:.2f}/{o.lens_radius2:.2f} d{o.lens_sep:.2f}"
    elif o.shape == SHAPE_REULEAUX:
        shape_txt = f"Reuleaux-{o.reuleaux_sides} r{o.reuleaux_radius:.2f}"
    else:
        shape_txt = f"Capsule L{o.capsule_length:.2f} r{o.capsule_radius:.2f}"
    pos_txt = f"r{o.r:.2f} @ {o.theta:.1f}\u00b0"
    if o.cx or o.cy:
        pos_txt += f" +({o.cx:+.2f}, {o.cy:+.2f})"
    return (f"{shape_txt}, {pos_txt}, in: {OBSTACLE_EFFECT_NAMES[o.in_effect]}, "
            f"out: {OBSTACLE_EFFECT_NAMES[o.out_effect]}")


def obstacle_extent(o):
    """A reasonable half-width (in world units) to zoom an obstacle's
    thumbnail to fit, per shape."""
    if o.shape == SHAPE_ELLIPSE:
        return max(o.width, o.height) / 2.0
    if o.shape == SHAPE_POLYGON:
        return o.poly_radius
    if o.shape == SHAPE_MOSS_EGG:
        return 2.2 * o.moss_size
    if o.shape == SHAPE_SECTOR:
        return o.sector_radius
    if o.shape == SHAPE_SEGMENT:
        return o.segment_radius
    if o.shape == SHAPE_LENS:
        return max(o.lens_radius1, o.lens_radius2) + o.lens_sep / 2.0
    if o.shape == SHAPE_REULEAUX:
        return o.reuleaux_radius * 2.0
    return o.capsule_length / 2.0 + o.capsule_radius   # CAPSULE


def render_obstacle_thumbnail(o, size):
    """A size x size RGB thumbnail of just obstacle `o`'s own shape (not a
    piece of the simulated image), zoomed so it fills the thumbnail
    regardless of its actual world size, drawn as the same two-layer outline
    it gets in the image (outer layer = incoming effect, inner layer =
    outgoing effect) around a plain grey fill, on a mid-grey background so
    all three effect colors (including black) show up."""
    extent = max(obstacle_extent(o), 1e-6)
    scale = (size / 2.0) * 0.86 / extent            # zoom to fill, with a small margin
    ys, xs = np.mgrid[0:size, 0:size]
    wx = (xs + 0.5 - size / 2.0) / scale
    wy = -(ys + 0.5 - size / 2.0) / scale
    bound = make_bound(o.shape, o.params())
    mask = inside_boundary_grid(wx, wy, o.shape, bound)
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


def ring_layer_masks(shape, bound, x, y, layer_px=OBST_RING_LAYER_PX):
    """(outer, inner) boolean masks on the world-coordinate grid (x, y) for a
    shape's two-layer outline: `outer` is the band just outside its edge and
    `inner` the band just inside it, each `layer_px` pixels thick."""
    w = layer_px * PIXEL
    if shape == SHAPE_ELLIPSE:
        m11, m12, m22 = bound[1], bound[2], bound[3]
        q = m11 * x * x + 2.0 * m12 * x * y + m22 * y * y
        grad = 2.0 * np.sqrt((m11 * x + m12 * y) ** 2 + (m12 * x + m22 * y) ** 2)
        sd = (q - 1.0) / np.maximum(grad, 1e-9)    # first-order signed distance (+ outside)
        return (sd >= 0.0) & (sd < w), (sd < 0.0) & (sd >= -w)
    if shape == SHAPE_POLYGON:
        h = bound[1]
        n = int(bound[2])
        f = np.full(x.shape, -1e9)
        for k in range(n):
            np.maximum(f, bound[3 + 2 * k] * x + bound[4 + 2 * k] * y, out=f)
        sd = f - h
        return (sd >= 0.0) & (sd < w), (sd < 0.0) & (sd >= -w)
    inside = inside_boundary_grid(x, y, shape, bound)
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


def build_obstacle_arrays(obstacles):
    """[Obstacle, ...] -> (shapes, bounds, cx, cy, effects, fire_inside)
    parallel numpy arrays, one entry per obstacle (an obstacle that passes
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
        shapes[idx] = o.shape
        b = make_bound(o.shape, o.params())
        bounds[idx, :len(b)] = b
        cx[idx], cy[idx] = o.center()
        effects[idx, 0] = o.in_effect
        effects[idx, 1] = o.out_effect
        fire_inside[idx] = o.fire_inside
    return shapes, bounds, cx, cy, effects, fire_inside


def obstacles_signature(obstacles):
    """A plain-data snapshot of every obstacle's settings, for detecting
    whether the obstacle list has changed since the last Start."""
    fields = [f for f in Obstacle.__slots__ if f != "id"]
    return tuple(tuple(getattr(o, f) for f in fields) for o in obstacles)


# Every field of an obstacle that can be saved into a keyframe and animated:
# (field name, short label, whether it's a float slider - only float fields
# are ever smoothly interpolated, exactly like the fields in SETTING_INFO).
# Every bounding-shape parameter's full identity, in one place. This exists
# because one parameter's "identity" was previously scattered by hand across
# SETTING_INFO, SETTING_FLOAT_KEYS, DEFAULT_SETTING_VALUES, OBST_FIELD_INFO,
# short_value_text's formatting rules, read_shape_controls, load_value, and
# apply_snapshot_to_controls - roughly a dozen places for a single new
# parameter, discovered the hard way (twice) when a couple of those places
# got missed. All of those are now generated from, or read directly out of,
# this one table instead.
#
# Three different names for the same parameter had to be reconciled to build
# this table, and are recorded explicitly rather than assumed to match:
#   field   the Snapshot field name - also the Obstacle attribute name, since
#           an obstacle can be any of the same shapes with the same params
#   key     the string key used in SETTING_INFO / DEFAULT_SETTING_VALUES /
#           a keyframe's own data dict - equal to `field` except for
#           ell_rotation/poly_rotation, shortened long ago to ell_rot/poly_rot
#   widget  the "self.<widget>" control that holds this parameter's live
#           value on the main Simulation tab (always a FloatEntryHandle- or
#           tk.Scale-like object with .get()/.set()) - usually "<key>_scale"
#           but inconsistently abbreviated further for several fields
#           (moss_rotation -> moss_rot_scale, lens_radius1 -> lens_r1_scale,
#           etc.) by whoever wrote each one at the time
#
# `animate` says whether this parameter smoothly interpolates between
# keyframes (SETTING_FLOAT_KEYS) rather than holding the nearest earlier
# keyframe's value - true for every genuinely continuous parameter; false
# only for the two whole-number "how many sides/points" counts. `fmt` says
# how short_value_text displays a value: "size" (N.NN), "angle" (N.N followed
# by a degree sign), or "int" (plain).
ShapeParamInfo = namedtuple("ShapeParamInfo",
                            ["field", "key", "widget", "label", "default", "animate", "fmt"])

SHAPE_PARAM_INFO = [
    ShapeParamInfo("width", "width", "width_scale", "Width", DEFAULT_WIDTH, True, "size"),
    ShapeParamInfo("height", "height", "height_scale", "Height", DEFAULT_HEIGHT, True, "size"),
    ShapeParamInfo("ell_rotation", "ell_rot", "ell_rot_scale", "Ellipse rotation",
                   DEFAULT_ELL_ROTATION, True, "angle"),
    ShapeParamInfo("sides", "sides", "sides_scale", "Sides", DEFAULT_SIDES, False, "int"),
    ShapeParamInfo("poly_rotation", "poly_rot", "poly_rot_scale", "Polygon rotation",
                   DEFAULT_POLY_ROTATION, True, "angle"),
    ShapeParamInfo("moss_angle", "moss_angle", "moss_angle_scale", "Moss egg angle ACB",
                   DEFAULT_MOSS_ANGLE, True, "angle"),
    ShapeParamInfo("moss_size", "moss_size", "moss_size_scale", "Moss egg size",
                   DEFAULT_MOSS_SIZE, True, "size"),
    ShapeParamInfo("moss_rotation", "moss_rotation", "moss_rot_scale", "Moss egg rotation",
                   0.0, True, "angle"),
    ShapeParamInfo("sector_radius", "sector_radius", "sector_radius_scale", "Sector radius",
                   DEFAULT_ARC_RADIUS, True, "size"),
    ShapeParamInfo("sector_span", "sector_span", "sector_span_scale", "Sector span",
                   DEFAULT_SECTOR_SPAN, True, "angle"),
    ShapeParamInfo("sector_rotation", "sector_rotation", "sector_rot_scale",
                   "Sector rotation", 0.0, True, "angle"),
    ShapeParamInfo("segment_radius", "segment_radius", "segment_radius_scale",
                   "Segment radius", DEFAULT_ARC_RADIUS, True, "size"),
    ShapeParamInfo("segment_span", "segment_span", "segment_span_scale", "Segment span",
                   DEFAULT_SEGMENT_SPAN, True, "angle"),
    ShapeParamInfo("segment_rotation", "segment_rotation", "segment_rot_scale",
                   "Segment rotation", 0.0, True, "angle"),
    ShapeParamInfo("lens_radius1", "lens_radius1", "lens_r1_scale", "Lens radius 1",
                   DEFAULT_LENS_RADIUS, True, "size"),
    ShapeParamInfo("lens_radius2", "lens_radius2", "lens_r2_scale", "Lens radius 2",
                   DEFAULT_LENS_RADIUS, True, "size"),
    ShapeParamInfo("lens_sep", "lens_sep", "lens_sep_scale", "Lens separation",
                   DEFAULT_LENS_SEP, True, "size"),
    ShapeParamInfo("lens_rotation", "lens_rotation", "lens_rot_scale", "Lens rotation",
                   0.0, True, "angle"),
    ShapeParamInfo("reuleaux_sides", "reuleaux_sides", "reuleaux_sides_scale",
                   "Reuleaux sides", DEFAULT_REULEAUX_SIDES, False, "int"),
    ShapeParamInfo("reuleaux_rotation", "reuleaux_rotation", "reuleaux_rot_scale",
                   "Reuleaux rotation", 0.0, True, "angle"),
    ShapeParamInfo("capsule_length", "capsule_length", "capsule_length_scale",
                   "Capsule length", DEFAULT_CAPSULE_LENGTH, True, "size"),
    ShapeParamInfo("capsule_radius", "capsule_radius", "capsule_radius_scale",
                   "Capsule radius", DEFAULT_CAPSULE_RADIUS, True, "size"),
    ShapeParamInfo("capsule_rotation", "capsule_rotation", "capsule_rot_scale",
                   "Capsule rotation", 0.0, True, "angle"),
]
SHAPE_PARAM_BY_KEY = {p.key: p for p in SHAPE_PARAM_INFO}
SHAPE_PARAM_BY_FIELD = {p.field: p for p in SHAPE_PARAM_INFO}

# Display format for the handful of Obstacle fields that AREN'T shared with
# the main bounding shape (its own position, and the extra radius a polygon
# or Reuleaux-polygon obstacle needs that the main shape doesn't) - everything
# else's format comes from SHAPE_PARAM_BY_FIELD above.
OBST_ONLY_FMT = {"poly_radius": "size", "reuleaux_radius": "size", "r": "size",
                 "theta": "angle"}

def _shared_obst_field(field):
    """An OBST_FIELD_INFO row for a parameter Obstacle shares with the main
    bounding shape, read from SHAPE_PARAM_INFO instead of repeated here."""
    p = SHAPE_PARAM_BY_FIELD[field]
    return (p.field, p.label, p.animate)


OBST_FIELD_INFO = [
    ("shape", "Shape", False),
    _shared_obst_field("width"),
    _shared_obst_field("height"),
    _shared_obst_field("ell_rotation"),
    _shared_obst_field("sides"),
    _shared_obst_field("poly_rotation"),
    ("poly_radius", "Radius", True),
    _shared_obst_field("moss_angle"),
    _shared_obst_field("moss_size"),
    _shared_obst_field("moss_rotation"),
    _shared_obst_field("sector_radius"),
    _shared_obst_field("sector_span"),
    _shared_obst_field("sector_rotation"),
    _shared_obst_field("segment_radius"),
    _shared_obst_field("segment_span"),
    _shared_obst_field("segment_rotation"),
    _shared_obst_field("lens_radius1"),
    _shared_obst_field("lens_radius2"),
    _shared_obst_field("lens_sep"),
    _shared_obst_field("lens_rotation"),
    _shared_obst_field("reuleaux_sides"),
    _shared_obst_field("reuleaux_rotation"),
    ("reuleaux_radius", "Reuleaux radius", True),
    _shared_obst_field("capsule_length"),
    _shared_obst_field("capsule_radius"),
    _shared_obst_field("capsule_rotation"),
    ("r", "Distance (r)", True),
    ("theta", "Angle (\u03b8)", True),
    ("cx", "Offset X", True),
    ("cy", "Offset Y", True),
    ("in_effect", "Incoming ray effect", False),
    ("out_effect", "Outgoing ray effect", False),
    ("fire_inside", "Fire rays inside", False),
]

ObstacleSnapshot = namedtuple("ObstacleSnapshot", list(Obstacle.__slots__))


def snapshot_obstacle(o):
    """A plain, immutable copy of one Obstacle's fields - safe to read from a
    background render thread while the GUI thread might still be editing the
    live Obstacle object."""
    return ObstacleSnapshot(*(getattr(o, f) for f in Obstacle.__slots__))


def obstacle_params_from_vals(shape, vals):
    """The shape-specific params() tuple, built from a dict of resolved field
    values (used when animating an obstacle's fields between keyframes)."""
    if shape == SHAPE_ELLIPSE:
        return (vals["width"], vals["height"], vals["ell_rotation"])
    if shape == SHAPE_POLYGON:
        return (vals["sides"], vals["poly_rotation"], vals["poly_radius"])
    if shape == SHAPE_MOSS_EGG:
        return (vals["moss_angle"], vals["moss_size"], vals["moss_rotation"])
    if shape == SHAPE_SECTOR:
        return (vals["sector_radius"], vals["sector_span"], vals["sector_rotation"])
    if shape == SHAPE_SEGMENT:
        return (vals["segment_radius"], vals["segment_span"], vals["segment_rotation"])
    if shape == SHAPE_LENS:
        return (vals["lens_radius1"], vals["lens_radius2"], vals["lens_sep"],
               vals["lens_rotation"])
    if shape == SHAPE_REULEAUX:
        return (vals["reuleaux_sides"], vals["reuleaux_rotation"], vals["reuleaux_radius"])
    return (vals["capsule_length"], vals["capsule_radius"], vals["capsule_rotation"])


def resolve_obstacle_arrays(obstacle_snapshots, keyframes_sorted, frame_number):
    """(shapes, bounds, cx, cy, effects, fire_inside) parallel arrays for one
    video frame: the set of obstacles is fixed (whichever existed when the
    render started), but every field of every one of them is independently
    resolved (animated or held) from the keyframes, exactly like the other
    settings - falling back to that obstacle's own value (at render-start
    time) wherever no keyframe defines a given field."""
    active = []
    for snap in obstacle_snapshots:
        vals = {}
        for field, _label, is_float in OBST_FIELD_INFO:
            key = f"obst:{snap.id}:{field}"
            vals[field] = resolve_value(key, keyframes_sorted, frame_number,
                                        getattr(snap, field), is_float)
        shape = vals["shape"]
        params = obstacle_params_from_vals(shape, vals)
        bound = make_bound(shape, params)
        cx, cy = obstacle_center_from(vals["r"], vals["theta"], vals["cx"], vals["cy"])
        active.append((shape, bound, cx, cy, vals["in_effect"], vals["out_effect"],
                       vals["fire_inside"]))

    n = len(active)
    shapes = np.empty(n, dtype=np.int64)
    bounds = np.zeros((n, OBST_BOUND_LEN), dtype=np.float64)
    cxs = np.empty(n, dtype=np.float64)
    cys = np.empty(n, dtype=np.float64)
    effects = np.empty((n, 2), dtype=np.int64)
    fire_inside = np.empty(n, dtype=np.bool_)
    for i, (shape, bound, cx, cy, in_effect, out_effect, fire) in enumerate(active):
        shapes[i] = shape
        bounds[i, :len(bound)] = bound
        cxs[i] = cx
        cys[i] = cy
        effects[i, 0] = in_effect
        effects[i, 1] = out_effect
        fire_inside[i] = fire
    return shapes, bounds, cxs, cys, effects, fire_inside



# Every setting a keyframe can define: (key, tab it lives on, short label for
# the settings list). Order here is the order shown within each tab group.
SETTING_INFO = [
    ("iter", "Simulation", "Iteration limit"),
    ("dist", "Simulation", "Distance limit"),
    ("seg_mult", "Simulation", "Segment length multiplier"),
    ("shape", "Simulation", "Shape"),
    *((p.key, "Simulation", p.label) for p in SHAPE_PARAM_INFO),
    ("fuzz_on", "Fuzzing", "Enabled"),
    ("emit_angle", "Fuzzing", "Emission angle"),
    ("emit_jitter", "Fuzzing", "Emission jitter"),
    ("bnd_offset", "Fuzzing", "Boundary offset"),
    ("bnd_jitter", "Fuzzing", "Boundary jitter"),
    ("bnd_mult", "Fuzzing", "Boundary multiplier"),
    ("self_offset", "Fuzzing", "Self offset"),
    ("self_jitter", "Fuzzing", "Self jitter"),
    ("self_mult", "Fuzzing", "Self multiplier"),
    ("seed", "Fuzzing", "Seed"),
    ("buddha_blur_radius", "Buddha blur", "Radius"),
    ("buddha_blur_type", "Buddha blur", "Blur type"),
    ("main_fire", "Simulation", "Bounding shape: fire rays inside"),
    ("main_in_effect", "Simulation", "Bounding shape: incoming ray effect"),
    ("main_out_effect", "Simulation", "Bounding shape: outgoing ray effect"),
    ("outline", "Decorations", "Outline"),
]
SETTING_TABS_ORDER = ["Simulation", "Fuzzing", "Buddha blur", "Decorations"]
SETTING_LABEL = {key: label for key, _tab, label in SETTING_INFO}
SETTING_TAB = {key: tab for key, tab, _label in SETTING_INFO}

# Only these are float sliders, so only these are ever smoothly interpolated;
# everything else (shape choice, colors, metrics, sides, flags, seed) always
# holds the value of the nearest earlier keyframe that defines it.
SETTING_FLOAT_KEYS = {
    "dist", "seg_mult",
    *(p.key for p in SHAPE_PARAM_INFO if p.animate),
    "emit_angle", "emit_jitter", "bnd_offset", "bnd_jitter",
    "bnd_mult", "self_offset", "self_jitter", "self_mult",
    "buddha_blur_radius",
}


def shape_params_tuple(shape, obj, include_obstacle_only=False):
    """The selected shape's own parameter tuple, read directly off `obj`
    (a Snapshot or an Obstacle - both expose the same field names for every
    shared parameter, so plain attribute access works for either). Shared
    by snapshot_shape_params, Obstacle.params(), effective_settings(), and
    the keyframe-resolution path - the single place that maps a shape to
    its own parameters, instead of two separate copies of that mapping.

    include_obstacle_only=True adds polygon's and Reuleaux's extra radius
    field (poly_radius / reuleaux_radius) at the end of their tuples - only
    an obstacle needs it, since the bounding shape is always inscribed in
    the unit circle for those two shapes and has no radius parameter of its
    own to read."""
    if shape == SHAPE_ELLIPSE:
        return (obj.width, obj.height, obj.ell_rotation)
    if shape == SHAPE_POLYGON:
        base = (obj.sides, obj.poly_rotation)
        return base + (obj.poly_radius,) if include_obstacle_only else base
    if shape == SHAPE_MOSS_EGG:
        return (obj.moss_angle, obj.moss_size, obj.moss_rotation)
    if shape == SHAPE_SECTOR:
        return (obj.sector_radius, obj.sector_span, obj.sector_rotation)
    if shape == SHAPE_SEGMENT:
        return (obj.segment_radius, obj.segment_span, obj.segment_rotation)
    if shape == SHAPE_LENS:
        return (obj.lens_radius1, obj.lens_radius2, obj.lens_sep, obj.lens_rotation)
    if shape == SHAPE_REULEAUX:
        base = (obj.reuleaux_sides, obj.reuleaux_rotation)
        return base + (obj.reuleaux_radius,) if include_obstacle_only else base
    return (obj.capsule_length, obj.capsule_radius, obj.capsule_rotation)  # CAPSULE


def snapshot_shape_params(snap):
    """The selected shape's own parameter tuple, extracted from a Snapshot -
    shared by effective_settings() and the keyframe-resolution path."""
    return shape_params_tuple(snap.shape, snap)


def main_ctl_array(fire, in_effect, out_effect):
    """The main bounding shape's ray behavior, packed for the kernels:
    [fires rays inside (0/1), incoming effect, outgoing effect]."""
    return np.array([1 if fire else 0, in_effect, out_effect], dtype=np.int64)


# The behavior the bounding shape always had: fires rays inside, reflects them
# whichever way they cross (in practice only ever from inside).
MAIN_CTL_DEFAULT = main_ctl_array(True, OBSTACLE_EFFECT_REFLECT, OBSTACLE_EFFECT_REFLECT)


def settings_main_ctl(settings):
    """main_ctl_array() for a Settings."""
    return main_ctl_array(settings.main_fire, settings.main_in_effect,
                          settings.main_out_effect)


def effective_settings(snap):
    """The Settings (selected shape's parameters only) implied by a Snapshot."""
    return Settings(max_iter=snap.max_iter, max_dist=snap.max_dist, seg_mult=snap.seg_mult,
                    shape=snap.shape, params=snapshot_shape_params(snap), fuzz=snap.fuzz,
                    buddha_blur_radius=snap.buddha_blur_radius,
                    buddha_blur_type=snap.buddha_blur_type, main_fire=snap.main_fire,
                    main_in_effect=snap.main_in_effect, main_out_effect=snap.main_out_effect)


def blend_value(a, b, t, method):
    """One interpolated float, using the method chosen on the *later* keyframe."""
    if method == "Step":
        return a
    return a + (b - a) * t


def resolve_value(key, keyframes_sorted, frame_number, fallback, is_float):
    """The value of one setting at `frame_number`, using only the keyframes
    (already sorted by frame_index) that actually define `key` - keyframes
    lacking it are skipped entirely, both for holding and for interpolation,
    so different settings can animate over different stretches of the
    timeline. `fallback` is used when no keyframe defines it at all;
    `is_float` says whether it's ever smoothly interpolated (True) or always
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

    if not is_float or active.frame_index == frame_number or active_i + 1 >= len(have):
        return active.data[key]
    nxt = have[active_i + 1]
    if nxt.frame_index == active.frame_index:
        return active.data[key]
    t = (frame_number - active.frame_index) / (nxt.frame_index - active.frame_index)
    return blend_value(active.data[key], nxt.data[key], t, nxt.interp)


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
    fuzz = FuzzParams(bool(v["fuzz_on"]), v["emit_angle"], v["emit_jitter"],
                      v["bnd_offset"], v["bnd_jitter"], v["bnd_mult"],
                      v["self_offset"], v["self_jitter"], v["self_mult"],
                      int(round(v["seed"])))
    snapshot = Snapshot(
        max_iter=int(round(v["iter"])), max_dist=v["dist"], seg_mult=v["seg_mult"],
        shape=int(round(v["shape"])), width=v["width"], height=v["height"],
        ell_rotation=v["ell_rot"], sides=int(round(v["sides"])),
        poly_rotation=v["poly_rot"], moss_angle=v["moss_angle"], moss_size=v["moss_size"],
        moss_rotation=v["moss_rotation"], sector_radius=v["sector_radius"],
        sector_span=v["sector_span"], sector_rotation=v["sector_rotation"],
        segment_radius=v["segment_radius"], segment_span=v["segment_span"],
        segment_rotation=v["segment_rotation"], lens_radius1=v["lens_radius1"],
        lens_radius2=v["lens_radius2"], lens_sep=v["lens_sep"],
        lens_rotation=v["lens_rotation"], reuleaux_sides=int(round(v["reuleaux_sides"])),
        reuleaux_rotation=v["reuleaux_rotation"], capsule_length=v["capsule_length"],
        capsule_radius=v["capsule_radius"], capsule_rotation=v["capsule_rotation"],
        fuzz=fuzz, buddha_blur_radius=v["buddha_blur_radius"],
        buddha_blur_type=int(round(v["buddha_blur_type"])),
        main_fire=bool(v["main_fire"]), main_in_effect=int(round(v["main_in_effect"])),
        main_out_effect=int(round(v["main_out_effect"])))
    return KeyframeData(snapshot, bool(v["outline"]))


def describe_shape(shape, params):
    if shape == SHAPE_ELLIPSE:
        w, h, rot = params
        return f"Ellipse {w:.2f} \u00d7 {h:.2f}, rotated {rot:.1f}\u00b0"
    if shape == SHAPE_POLYGON:
        sides, rot = params
        return f"Regular {sides}-gon, rotated {rot:.1f}\u00b0"
    if shape == SHAPE_MOSS_EGG:
        angle, size, rot = params
        return (f"Moss's egg, angle ACB {angle:.1f}\u00b0, size {size:.2f}, "
               f"rotated {rot:.1f}\u00b0")
    if shape == SHAPE_SECTOR:
        r, span, rot = params
        return f"Circular sector, radius {r:.2f}, span {span:.1f}\u00b0, rotated {rot:.1f}\u00b0"
    if shape == SHAPE_SEGMENT:
        r, span, rot = params
        return f"Circular segment, radius {r:.2f}, span {span:.1f}\u00b0, rotated {rot:.1f}\u00b0"
    if shape == SHAPE_LENS:
        r1, r2, sep, rot = params
        return (f"Two-circle lens, radii {r1:.2f}/{r2:.2f}, separation {sep:.2f}, "
               f"rotated {rot:.1f}\u00b0")
    if shape == SHAPE_REULEAUX:
        sides, rot = params
        return f"Reuleaux {sides}-gon, rotated {rot:.1f}\u00b0"
    length, r, rot = params                       # SHAPE_CAPSULE
    return f"Capsule, length {length:.2f}, radius {r:.2f}, rotated {rot:.1f}\u00b0"


def describe_fuzz(f):
    if not f.enabled:
        return "off"

    def part(name, off, jit, mult):
        return f"{name} {off:+.1f}\u00b0 \u00b1{jit:.1f}\u00b0 \u00d7{mult:.2f}"

    return "; ".join([f"emission {f.emit_angle:.1f}\u00b0 \u00b1{f.emit_jitter:.1f}\u00b0",
                      part("boundary bounces", f.bnd_offset, f.bnd_jitter, f.bnd_mult),
                      part("self bounces", f.self_offset, f.self_jitter, f.self_mult),
                      f"seed {f.seed}"])


def effective_dist(max_dist):
    """The distance limit passed to the simulation: infinite when 0 (no limit)."""
    return float("inf") if max_dist == 0 else float(max_dist)


def fuzz_array(f):
    """FuzzParams -> the float array the simulation kernels expect (the seed
    is handled separately, by seeding the RNG before each ray is traced)."""
    return np.array([1.0 if f.enabled else 0.0,
                     f.emit_angle, f.emit_jitter,
                     f.bnd_offset, f.bnd_jitter, f.bnd_mult,
                     f.self_offset, f.self_jitter, f.self_mult], dtype=np.float64)


def settings_suffix(settings):
    """Simulation settings as a filename fragment: plain characters, no spaces,
    e.g. 'iter32-distNONE-ellipse-w2.00-h2.00-rot0.0'."""
    parts = [f"iter{settings.max_iter}",
             "distNONE" if settings.max_dist == 0 else f"dist{settings.max_dist}"]
    if settings.shape == SHAPE_ELLIPSE:
        w, h, rot = settings.params
        parts += ["ellipse", f"w{w:.2f}", f"h{h:.2f}", f"rot{rot:.1f}"]
    elif settings.shape == SHAPE_POLYGON:
        sides, rot = settings.params
        parts += ["polygon", f"n{sides}", f"rot{rot:.1f}"]
    elif settings.shape == SHAPE_MOSS_EGG:
        angle, size, rot = settings.params
        parts += ["mossegg", f"a{angle:.1f}", f"sz{size:.2f}", f"rot{rot:.1f}"]
    elif settings.shape == SHAPE_SECTOR:
        r, span, rot = settings.params
        parts += ["sector", f"r{r:.2f}", f"span{span:.1f}", f"rot{rot:.1f}"]
    elif settings.shape == SHAPE_SEGMENT:
        r, span, rot = settings.params
        parts += ["segment", f"r{r:.2f}", f"span{span:.1f}", f"rot{rot:.1f}"]
    elif settings.shape == SHAPE_LENS:
        r1, r2, sep, rot = settings.params
        parts += ["lens", f"r1{r1:.2f}", f"r2{r2:.2f}", f"sep{sep:.2f}", f"rot{rot:.1f}"]
    elif settings.shape == SHAPE_REULEAUX:
        sides, rot = settings.params
        parts += ["reuleaux", f"n{sides}", f"rot{rot:.1f}"]
    else:                                          # SHAPE_CAPSULE
        length, r, rot = settings.params
        parts += ["capsule", f"len{length:.2f}", f"r{r:.2f}", f"rot{rot:.1f}"]
    f = settings.fuzz
    if f.enabled:
        parts += ["fuzz",
                  f"ea{f.emit_angle:.1f}", f"ej{f.emit_jitter:.1f}",
                  f"bo{f.bnd_offset:.1f}", f"bj{f.bnd_jitter:.1f}",
                  f"bm{f.bnd_mult:.2f}",
                  f"so{f.self_offset:.1f}", f"sj{f.self_jitter:.1f}",
                  f"sm{f.self_mult:.2f}", f"seed{f.seed}"]
    return "-".join(parts)


def ensure_export_dir():
    """EXPORT_DIR, created first if it doesn't exist yet."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    return EXPORT_DIR


def take_ui_screenshot(root):
    """Development aid: grabs the app's own window from the screen and saves
    it as DEV_DIR/screenshots/<FILE_VERSION>-YYYYMMDD-hhmmss.png, returning the
    path - but only if the DEV_DIR folder already exists (this never creates
    it) and Pillow is installed; otherwise does nothing and returns None. It
    grabs the actual screen pixels inside the window's rectangle, so anything
    covering the window at that moment appears in the shot instead."""
    if ImageGrab is None or not os.path.isdir(DEV_DIR):
        return None
    root.update()                                  # finish drawing, settle geometry
    x, y = root.winfo_rootx(), root.winfo_rooty()
    img = ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height()))
    shots_dir = os.path.join(DEV_DIR, "screenshots")
    os.makedirs(shots_dir, exist_ok=True)
    path = os.path.join(shots_dir, f"{FILE_VERSION}-{time.strftime('%Y%m%d-%H%M%S')}.png")
    img.save(path)
    return path


def default_filename(settings=None, ext="png"):
    """programname-YYYYMMDD-hhmmss-<settings>.<ext>, stamped with the current time."""
    name = f"{PROGRAM_NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
    if settings is not None:
        name += "-" + settings_suffix(settings)
    return f"{name}.{ext}"


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
    export control (New figure, video) reads its value through this, so an
    exported frame's dimensions are always a multiple of 4, regardless of
    exactly what was typed."""
    value = parse_positive_int(text, max_digits=max_digits)
    if value is None:
        return None
    return round_to_multiple_of_4(value)


def png_rgba(width, height, pixel):
    """Minimal PNG encoder: a width x height image where every pixel is the
    4-byte RGBA value `pixel`. (Lets us make alpha-blended images with only the
    standard library.)"""
    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    row = b"\x00" + bytes(pixel) * width          # filter byte 0 + pixels
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * height))
            + chunk(b"IEND", b""))


def encode_png_bytes(rgb):
    """Encode an (H, W, 3) uint8 array as plain (no-alpha) PNG bytes."""
    height, width = rgb.shape[0], rgb.shape[1]

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    rows = bytearray()
    for y in range(height):
        rows.append(0)                             # filter type 0 (none)
        rows.extend(rgb[y].tobytes())
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
            + chunk(b"IEND", b""))


def write_png(path, rgb):
    """Write an (H, W, 3) uint8 array as a plain (no-alpha) PNG file."""
    with open(path, "wb") as fh:
        fh.write(encode_png_bytes(rgb))


def decode_png_bytes(data):
    """Read the pixels of a PNG written by this program's own
    encode_png_bytes (8-bit RGB, filter type 0/none throughout) back into an
    (H, W, 3) uint8 array."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    width = height = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", body[:10])
            if depth != 8 or color_type != 2:
                raise ValueError("expected an 8-bit RGB PNG (as written by this program)")
        elif tag == b"IDAT":
            idat.extend(body)
        elif tag == b"IEND":
            break
        pos += 12 + length                          # length(4) + tag(4) + data + crc(4)
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    for y in range(height):
        row_start = y * (stride + 1)
        filt = raw[row_start]
        if filt != 0:
            raise ValueError(f"unsupported PNG filter type {filt} (expected 0)")
        rgb[y] = np.frombuffer(raw, dtype=np.uint8, count=stride,
                               offset=row_start + 1).reshape(-1, 3)
    return rgb


def read_png(path):
    """The inverse of write_png."""
    with open(path, "rb") as fh:
        return decode_png_bytes(fh.read())


@njit(cache=True)
def qoi_encode_body(rgb, out):
    """Encode an (H, W, 3) uint8 image into QOI's chunk stream (the part
    between the 14-byte header and the 8-byte end marker), written into the
    pre-allocated `out` uint8 array (must be at least 5*H*W bytes - the
    worst case, one uncompressed QOI_OP_RGB chunk per pixel). Returns the
    number of bytes actually written."""
    h, w = rgb.shape[0], rgb.shape[1]
    n = h * w
    flat = rgb.reshape(n, 3)
    index_r = np.zeros(64, dtype=np.uint8)
    index_g = np.zeros(64, dtype=np.uint8)
    index_b = np.zeros(64, dtype=np.uint8)
    index_seen = np.zeros(64, dtype=np.bool_)
    prev_r = 0
    prev_g = 0
    prev_b = 0
    pos = 0
    run = 0
    for i in range(n):
        r = int(flat[i, 0])
        g = int(flat[i, 1])
        b = int(flat[i, 2])
        if r == prev_r and g == prev_g and b == prev_b:
            run += 1
            if run == 62:
                out[pos] = 0xC0 | (run - 1)
                pos += 1
                run = 0
            continue
        if run > 0:
            out[pos] = 0xC0 | (run - 1)
            pos += 1
            run = 0

        idx = (r * 3 + g * 5 + b * 7 + 255 * 11) % 64
        if index_seen[idx] and index_r[idx] == r and index_g[idx] == g and index_b[idx] == b:
            out[pos] = idx                              # QOI_OP_INDEX (tag 00 is idx<64)
            pos += 1
        else:
            dr = (r - prev_r + 128) % 256 - 128          # wrapped signed difference
            dg = (g - prev_g + 128) % 256 - 128
            db = (b - prev_b + 128) % 256 - 128
            if -2 <= dr <= 1 and -2 <= dg <= 1 and -2 <= db <= 1:
                out[pos] = 0x40 | ((dr + 2) << 4) | ((dg + 2) << 2) | (db + 2)
                pos += 1
            else:
                dr_dg = dr - dg
                db_dg = db - dg
                if -32 <= dg <= 31 and -8 <= dr_dg <= 7 and -8 <= db_dg <= 7:
                    out[pos] = 0x80 | (dg + 32)
                    out[pos + 1] = ((dr_dg + 8) << 4) | (db_dg + 8)
                    pos += 2
                else:
                    out[pos] = 0xFE
                    out[pos + 1] = r
                    out[pos + 2] = g
                    out[pos + 3] = b
                    pos += 4
        index_r[idx] = r
        index_g[idx] = g
        index_b[idx] = b
        index_seen[idx] = True
        prev_r = r
        prev_g = g
        prev_b = b
    if run > 0:
        out[pos] = 0xC0 | (run - 1)
        pos += 1
    return pos


def encode_qoi_bytes(rgb):
    """An (H, W, 3) uint8 array -> the bytes of a complete .qoi file."""
    h, w = rgb.shape[0], rgb.shape[1]
    out = np.empty(5 * h * w, dtype=np.uint8)
    length = qoi_encode_body(np.ascontiguousarray(rgb), out)
    header = b"qoif" + struct.pack(">IIBB", w, h, 3, 0)
    return header + out[:length].tobytes() + b"\x00\x00\x00\x00\x00\x00\x00\x01"


def write_qoi(path, rgb):
    with open(path, "wb") as fh:
        fh.write(encode_qoi_bytes(rgb))


def jxl_quality_to_distance(quality):
    """A 0..100 "quality" (higher = better, matching the usual convention)
    -> libjxl's own "distance" scale (0 = lossless, higher = worse; sane
    values rarely exceed 5-10). quality=90 lands on distance=1.0, ffmpeg's
    own default and roughly comparable to libjpeg quality 90."""
    return max(0.1, (100 - quality) / 10.0)


def write_jxl(path, rgb, lossless, quality=90):
    """Encode an (H, W, 3) uint8 image to a .jxl file via ffmpeg's libjxl
    wrapper. Unlike PNG/QOI, JPEG XL has no remotely hand-rollable encoder -
    this genuinely needs the real codec, the same one the video export
    already depends on. Writes a temporary PNG (this program's own writer)
    as ffmpeg's input, and removes it again once the JXL file is written."""
    if ffmpeg is None:
        raise RuntimeError("ffmpeg-python is required to save JPEG XL images")
    os.makedirs(TEMP_DIR, exist_ok=True)
    tmp_path = os.path.join(TEMP_DIR, "_jxl_source.png")
    write_png(tmp_path, rgb)
    try:
        distance = 0.0 if lossless else jxl_quality_to_distance(quality)
        stream = ffmpeg.input(tmp_path)
        stream = ffmpeg.output(stream, path, **{"c:v": "libjxl", "distance": distance,
                                                "update": 1, "frames:v": 1})
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def cjxl_available():
    """Whether the external `cjxl` command-line tool (part of libjxl-tools)
    can be found on PATH - needed for an animated JXL video, since ffmpeg's
    own libjxl wrapper (write_jxl, above) only supports a single still
    frame, not animation."""
    return shutil.which("cjxl") is not None


def convert_apng_to_jxl(apng_path, jxl_path):
    """Converts an animated PNG to an animated JPEG XL file via the external
    `cjxl` tool. This is the only available path to an animated JXL file:
    ffmpeg cannot produce one directly (its libjxl wrapper is still-image
    only), but it CAN produce an APNG (natively, losslessly), and cjxl CAN
    convert an existing APNG's full animation to JXL - so the video export
    pipeline renders to APNG first and hands the result to this function.
    Lossless (-d 0): the source APNG is itself lossless, so there would be
    nothing to gain from a lossy pass, and no separate quality control is
    exposed for this in the video panel."""
    # Run the path shutil.which resolved, not the bare name "cjxl": on
    # Windows, a bare name only launches .exe files, whereas which() (and
    # so cjxl_available()) also finds .cmd/.bat shims.
    cjxl_path = shutil.which("cjxl")
    if cjxl_path is None:
        raise RuntimeError("cjxl failed: the cjxl tool was not found on PATH")
    result = subprocess.run([cjxl_path, apng_path, jxl_path, "-d", "0"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output captured"
        raise RuntimeError(f"cjxl failed (exit code {result.returncode}): {detail}")


# Saving a freshly generated image (as opposed to the live preview): what to
# render, and which format to encode it in
SAVE_SOURCE_OPTIONS = ["Current figure", "New figure"]
SAVE_NEW_FORMATS = ["png", "qoi", "jxl"]
JXL_MODES = ["Lossless", "Lossy"]

# Export designer: where (if anywhere) each kind of info text goes
COLOR_INFO_LOCATIONS = ["None", "Overlay", "Bottom margin"]
SIM_INFO_LOCATIONS = ["None", "Left margin"]


def encode_and_write(rgb, path, fmt, jxl_lossless=True, jxl_quality=90):
    """An (H, W, 3) uint8 image -> a file at `path`, in whichever of the
    three save formats is selected. Shared by saving the current preview
    (instant) and generating+saving a "New image" (backgrounded)."""
    if fmt == "png":
        write_png(path, rgb)
    elif fmt == "qoi":
        write_qoi(path, rgb)
    else:
        write_jxl(path, rgb, jxl_lossless, jxl_quality)


# Downsampling (video-encoding only - never applied to the PNG sequence, since
# it's an encoding setting, not a rendering one)
DOWNSAMPLE_FACTORS = ["No change", "2x2", "4x4", "8x8"]
DOWNSAMPLE_FACTOR_VALUES = {"No change": 1, "2x2": 2, "4x4": 4, "8x8": 8}
DOWNSAMPLE_METHODS = ["Mean", "Median", "Sum", "Max", "ffmpeg"]


def downsample_image(rgb, factor, method):
    """Block-reduce an (H, W, 3) uint8 image by `factor` in each dimension,
    using the mean, the (per-channel) median, the (per-channel, clamped to
    255) sum, or the (per-channel) maximum of each factor x factor block.
    Any leftover rows/columns that don't fill a whole block are trimmed off
    first."""
    if factor <= 1:
        return rgb
    h, w = rgb.shape[0], rgb.shape[1]
    factor = max(1, min(factor, h, w))               # never larger than the image itself
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    trimmed = rgb[:h2, :w2]
    blocks = trimmed.reshape(h2 // factor, factor, w2 // factor, factor, 3)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(h2 // factor, w2 // factor,
                                                     factor * factor, 3)
    if method == "Mean":
        out = blocks.mean(axis=2)
    elif method == "Median":
        out = np.median(blocks, axis=2)
    elif method == "Sum":
        out = blocks.astype(np.int64).sum(axis=2)    # avoid uint8 wraparound before clamping
    else:                                            # "Max"
        out = blocks.max(axis=2)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def make_grid(w, h):
    """World-coordinate (x, y) grids for a w x h array of pixel centres,
    spanning [-1, 1] on both axes (the same world extent regardless of w, h)."""
    cx = (np.arange(w) + 0.5) * (2.0 / w) - 1.0
    cy = (np.arange(h) + 0.5) * (2.0 / h) - 1.0
    return np.meshgrid(cx, -cy)


def lerp_color(c1, c2, t):
    """Linear interpolation between two (r, g, b) tuples -> a '#rrggbb' string."""
    r = int(round(c1[0] + (c2[0] - c1[0]) * t))
    g = int(round(c1[1] + (c2[1] - c1[1]) * t))
    b = int(round(c1[2] + (c2[2] - c1[2]) * t))
    return f"#{r:02x}{g:02x}{b:02x}"


def bilinear_resize(img, out_h, out_w):
    """Bilinearly resize an (H, W, 3) uint8 array to (out_h, out_w, 3) uint8."""
    in_h, in_w = img.shape[0], img.shape[1]
    ys = np.clip((np.arange(out_h) + 0.5) * in_h / out_h - 0.5, 0, in_h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * in_w / out_w - 0.5, 0, in_w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    rows0 = img[y0].astype(np.float64)             # (out_h, in_w, 3)
    rows1 = img[y1].astype(np.float64)
    top = rows0[:, x0] * (1.0 - wx) + rows0[:, x1] * wx
    bot = rows1[:, x0] * (1.0 - wx) + rows1[:, x1] * wx
    out = top * (1.0 - wy) + bot * wy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def render_text_line(text, scale):
    """A boolean (H, W) mask of `text` (lowercase letters in FONT_5X7 only)
    drawn in a 5x7 bitmap font, `scale`x, one column of gap between letters."""
    cell_w = 6                                     # 5 pixels + 1 gap column
    height = 7 * scale
    width = (cell_w * len(text) - 1) * scale
    mask = np.zeros((height, width), dtype=bool)
    for ci, ch in enumerate(text):
        ox = ci * cell_w * scale
        for ry, row in enumerate(FONT_5X7[ch]):
            for rx, px in enumerate(row):
                if px == "1":
                    mask[ry * scale:(ry + 1) * scale, ox + rx * scale:ox + (rx + 1) * scale] = True
    return mask


def render_word_lines(size_w, size_h, words, scale=1):
    """A black `size_w` x `size_h` RGB image with each string in `words` on
    its own centred magenta line (the whole block centred vertically)."""
    canvas = np.zeros((size_h, size_w, 3), dtype=np.uint8)
    lines = [render_text_line(w, scale) for w in words]
    gap = 2 * scale
    total_h = sum(line.shape[0] for line in lines) + gap * (len(lines) - 1)
    y = max(0, (size_h - total_h) // 2)
    for line in lines:
        h, w = line.shape
        x = max(0, (size_w - w) // 2)
        if y + h <= size_h and x + w <= size_w:
            region = canvas[y:y + h, x:x + w]
            region[line] = (255, 0, 255)
        y += h + gap
    return canvas


def no_img_thumbnail(size):
    """A black `size` x `size` RGB image with 'no' / 'img' centred in magenta."""
    return render_word_lines(size, size, ["no", "img"], scale=2)


def preview_placeholder(width, height):
    """A black width x height RGB image reading 'preview will go here' in
    magenta, shown in the Video panel whenever nothing is being rendered."""
    return render_word_lines(width, height, ["preview", "will", "go", "here"], scale=2)


# --------------------------------------------------------------------------- #
# Bounding shapes
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
    vertices lie on a circle of the given `radius` (1.0 = the unit circle, the
    main bounding shape's only option). With rotation 0 one vertex points
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


def scale_bound(shape, bound, factor, center=(0.0, 0.0)):
    """A copy of `bound` uniformly scaled by `factor` around `center` (every
    length grows/shrinks; angles and counts don't change) - used to draw a
    thin ring around a shape without a simple analytic gradient, by taking
    the boundary between a slightly bigger and a slightly smaller copy of it.
    `center` MUST be strictly inside the shape (see find_interior_point) for
    the bigger copy to actually contain the smaller one - scaling around a
    point outside the shape, or a corner of it, doesn't produce nested
    shapes and silently gives an empty or wrong ring instead."""
    cx0, cy0 = center
    b = bound.copy()
    b[0] *= factor * factor                     # area scales with factor^2
    if shape in (SHAPE_LENS, SHAPE_REULEAUX):
        n = int(b[1])
        idx = 2
        for _ in range(n):
            b[idx] = cx0 + factor * (b[idx] - cx0)
            b[idx + 1] = cy0 + factor * (b[idx + 1] - cy0)
            b[idx + 2] *= factor                # r
            idx += 3
    elif shape in (SHAPE_MOSS_EGG, SHAPE_SECTOR, SHAPE_SEGMENT, SHAPE_CAPSULE):
        n_a = int(b[1])
        n_e = int(b[2])
        idx = 3
        for _ in range(n_a):
            b[idx] = cx0 + factor * (b[idx] - cx0)
            b[idx + 1] = cy0 + factor * (b[idx + 1] - cy0)
            b[idx + 2] *= factor                # r (ang_lo/span untouched)
            idx += 5
        for _ in range(n_e):
            b[idx] = cx0 + factor * (b[idx] - cx0)         # ax
            b[idx + 1] = cy0 + factor * (b[idx + 1] - cy0)  # ay
            b[idx + 2] = cx0 + factor * (b[idx + 2] - cx0)  # bx
            b[idx + 3] = cy0 + factor * (b[idx + 3] - cy0)  # by
            idx += 4
    return b


def offset_bound(shape, bound, thickness):
    """A copy of `bound` moved outward by a fixed absolute `thickness`
    (negative moves it inward): every arc's radius changes by `thickness`,
    and every edge shifts along its own outward normal by `thickness`. Unlike
    scale_bound, this gives a ring of CONSTANT width everywhere - scaling by a
    multiplicative factor instead shifts each point by an amount proportional
    to its distance from the scaling centre, which for a point close to that
    centre (e.g. capsule's straight edges, close to its own middle) can be a
    tiny fraction of a pixel and effectively invisible."""
    b = bound.copy()
    if shape in (SHAPE_LENS, SHAPE_REULEAUX):
        n = int(b[1])
        idx = 2
        for _ in range(n):
            b[idx + 2] = max(b[idx + 2] + thickness, 1e-9)   # r
            idx += 3
    elif shape in (SHAPE_MOSS_EGG, SHAPE_SECTOR, SHAPE_SEGMENT, SHAPE_CAPSULE):
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



def make_bound(shape, params):
    if shape == SHAPE_ELLIPSE:
        return ellipse_bound(*params)
    if shape == SHAPE_POLYGON:
        return polygon_bound(*params)
    if shape == SHAPE_MOSS_EGG:
        return moss_egg_bound(*params)
    if shape == SHAPE_SECTOR:
        return sector_bound(*params)
    if shape == SHAPE_SEGMENT:
        return segment_bound(*params)
    if shape == SHAPE_LENS:
        return lens_bound(*params)
    if shape == SHAPE_REULEAUX:
        return reuleaux_bound(*params)
    return capsule_bound(*params)         # SHAPE_CAPSULE


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
    needing a bespoke formula per family. Used both as a ray-casting origin
    (sample_boundary_xy) and as scale_bound's scaling centre, both of which
    need genuine breathing room to work correctly."""
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


def outline_coords(shape, params):
    """Canvas coordinates (flat list) of a polygon tracing the bounding shape."""
    if shape == SHAPE_ELLIPSE:
        width, height, rotation = params
        t = np.linspace(0.0, 2.0 * pi, 181)
        ex, ey = 0.5 * width * np.cos(t), 0.5 * height * np.sin(t)
        th = radians(rotation)
        c, s = cos(th), sin(th)
        wx, wy = c * ex - s * ey, s * ex + c * ey
    elif shape == SHAPE_POLYGON:
        sides, rotation = params
        ang = pi / 2.0 + radians(rotation) + 2.0 * pi * np.arange(sides + 1) / sides
        wx, wy = np.cos(ang), np.sin(ang)          # first vertex repeated to close
    else:
        wx, wy = sample_boundary_xy(shape, make_bound(shape, params))
    xs = (wx + 1.0) * (SIZE / 2.0)
    ys = (1.0 - wy) * (SIZE / 2.0)
    return np.column_stack((xs, ys)).ravel().tolist()


def obstacle_outline_coords(o):
    """Canvas coordinates (flat list) of a polygon tracing obstacle `o`'s
    shape - like outline_coords, but also handles its centre (polar position
    plus x/y offset)."""
    if o.shape == SHAPE_ELLIPSE:
        t = np.linspace(0.0, 2.0 * pi, 181)
        ex, ey = 0.5 * o.width * np.cos(t), 0.5 * o.height * np.sin(t)
        th = radians(o.ell_rotation)
        c, s = cos(th), sin(th)
        wx, wy = c * ex - s * ey, s * ex + c * ey
    elif o.shape == SHAPE_POLYGON:
        ang = (pi / 2.0 + radians(o.poly_rotation)
              + 2.0 * pi * np.arange(o.sides + 1) / o.sides)
        wx, wy = o.poly_radius * np.cos(ang), o.poly_radius * np.sin(ang)
    else:
        wx, wy = sample_boundary_xy(o.shape, make_bound(o.shape, o.params()))
    ocx, ocy = o.center()
    wx = wx + ocx
    wy = wy + ocy
    xs = (wx + 1.0) * (SIZE / 2.0)
    ys = (1.0 - wy) * (SIZE / 2.0)
    return np.column_stack((xs, ys)).ravel().tolist()


# --------------------------------------------------------------------------- #
# Ray metrics (each is scaled to 0..1 so it can drive a color component)
# --------------------------------------------------------------------------- #
ST_HDIST = 0        # horizontal distance |x1-x2| between two points  [comparison]
ST_VDIST = 1        # vertical distance |y1-y2| between two points    [comparison]
ST_MANHATTAN = 2    # |x1-x2| + |y1-y2| between two points            [comparison]
ST_EUCLIDEAN = 3    # straight-line distance between two points       [comparison]
ST_BOUNCES = 4      # bounces / iteration limit               (linear)
ST_RAY_FRAC = 5     # bounces against ray / bounces           (linear)
ST_BOUNDARY_FRAC = 6  # bounces against the shape / bounces   (linear)
ST_LOOP_AREA = 7    # first self-loop area / same-perimeter circle (linear)
ST_ANGLES = 8       # sum of bounce angles / (180 * bounces)  (linear)
ST_LENGTH = 9       # total ray length                        (arctangent)
ST_LAST_DIR = 10    # direction of the last line segment      (linear angle)
ST_TERMINATION = 11  # why the ray stopped (see TERM_STAT_VALUE)
ST_LAST_RATIO = 12  # last segment length / its uncut length
N_STATS = 13

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
    "bounces off the bounding shape, divided by total bounces "
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


# Sample points that can be picked out of a ray's path
POINT_START = 0          # the ray's starting point (the pixel it was emitted from)
POINT_END = 1            # the last point of the ray (a cut point if the distance
                         # limit ended it part-way along a segment)
POINT_ALL_MEAN = 2       # mean of every bounce point, of any kind
POINT_BOUNDARY_MEAN = 3  # mean of bounce points off the boundary (or a reflecting obstacle)
POINT_SELF_MEAN = 4      # mean of bounce points off the ray's own earlier path
POINT_TOP = 5            # bounce point with the largest y (uppermost), any kind
POINT_BOTTOM = 6         # bounce point with the smallest y (lowermost), any kind
POINT_RIGHT = 7          # bounce point with the largest x (rightmost), any kind
POINT_LEFT = 8           # bounce point with the smallest x (leftmost), any kind
POINT_SELF_TOP = 9       # uppermost bounce off the ray's own earlier path
POINT_SELF_BOTTOM = 10   # lowermost bounce off the ray's own earlier path
POINT_SELF_RIGHT = 11    # rightmost bounce off the ray's own earlier path
POINT_SELF_LEFT = 12     # leftmost bounce off the ray's own earlier path
N_POINTS = 13             # the simulation's own per-ray sample point count - NOT the
                         # total number of selectable points; see N_POINT_CHOICES

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
    "the average of the points where the ray bounced off the bounding "
    "shape or off a reflecting obstacle.",
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


def short_value_text(key, val):
    """A compact display of one setting's value, for the Load/Save button
    text in the keyframe settings list."""
    if key.startswith("obst:"):
        field = key.split(":", 2)[2]
        if field == "shape":
            return SHAPE_NAMES[val]
        if field in ("in_effect", "out_effect"):
            return OBSTACLE_EFFECT_NAMES[val]
        if field == "fire_inside":
            return "On" if val else "Off"
        if field in ("cx", "cy"):
            return f"{val:+.2f}"
        p = SHAPE_PARAM_BY_FIELD.get(field)
        fmt = p.fmt if p is not None else OBST_ONLY_FMT.get(field)
        if fmt == "size":
            return f"{val:.2f}"
        if fmt == "angle":
            return f"{val:.1f}\u00b0"
        return str(val)                          # "int" fields, and anything else
    if key.startswith("view:"):
        field = key.split(":", 2)[2]
        if field == "mode":
            return val
        # field is "chan0"/"chan1"/"chan2"
        chan_src, kind, stat, point1, point2, buddha_pts, buddha_comp, buddha_bright, const = val
        if chan_src == CHAN_SRC_BUDDHA:
            chosen = [BUDDHA_POINT_LABELS[i].split()[0] for i, on in enumerate(buddha_pts) if on]
            comp = BUDDHA_COMPRESSION_LABELS[buddha_comp]
            bright = BUDDHA_BRIGHTNESS_OPTIONS[buddha_bright]
            return f"Buddha:{'+'.join(chosen) if chosen else 'none'}/{comp}/{bright}"
        if chan_src == CHAN_SRC_CONSTANT:
            return f"Const:{'0' if const == NONE_ZERO else '1'}"
        if kind == KIND_POINT:
            return f"{STAT_SHORT[stat]}@{POINT_SHORT[point1]}/{POINT_SHORT[point2]}"
        if stat in (NONE_ZERO, NONE_ONE):
            raise ValueError(
                f"channel value has stat={stat} (a Constant-only sentinel) but "
                f"chan_src is {chan_src}, not CHAN_SRC_CONSTANT")
        return STAT_SHORT[stat]
    if key in ("iter", "dist", "sides", "seed", "reuleaux_sides"):
        return str(val)
    if key == "shape":
        return SHAPE_NAMES[val]
    shape_param = SHAPE_PARAM_BY_KEY.get(key)
    if shape_param is not None:
        if shape_param.fmt == "size":
            return f"{val:.2f}"
        if shape_param.fmt == "angle":
            return f"{val:.1f}\u00b0"
        return str(val)                          # "int" fields
    if key == "emit_angle":
        return f"{val:.1f}\u00b0"
    if key in ("emit_jitter", "bnd_offset", "bnd_jitter", "self_offset", "self_jitter"):
        return f"\u00b1{val:.1f}\u00b0" if "jitter" in key else f"{val:+.1f}\u00b0"
    if key in ("bnd_mult", "self_mult", "seg_mult"):
        return f"\u00d7{val:.2f}"
    if key == "fuzz_on":
        return "On" if val else "Off"
    if key == "outline" or key == "main_fire":
        return "On" if val else "Off"
    if key in ("main_in_effect", "main_out_effect"):
        return OBSTACLE_EFFECT_NAMES[val]
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
# Obstacle/self.obstacles and "obst:<id>:<field>" exactly, since the two
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


ViewSnapshot = namedtuple("ViewSnapshot", list(ViewConfig.__slots__))


def snapshot_view(v):
    """A plain, immutable copy of one ViewConfig's fields - safe to read from
    a background render thread while the GUI thread might still be editing
    the live ViewConfig object (mirrors snapshot_obstacle). ViewConfig's
    per-channel fields are mutable lists (the UI mutates them in place, e.g.
    self.chan_src[c] = ...), so each one is copied here rather than shared -
    a shared reference would let a background thread see a live edit
    mid-render, exactly what a "snapshot" is meant to prevent."""
    vals = []
    for f in ViewConfig.__slots__:
        val = getattr(v, f)
        vals.append(list(val) if isinstance(val, list) else val)
    return ViewSnapshot(*vals)


def view_channel_value(v, c):
    """Channel c's current setting for a ViewConfig or ViewSnapshot `v`, as
    stored in a keyframe. constant_choice is included as its own element
    even though it's only meaningful when chan_src is CHAN_SRC_CONSTANT -
    keeping it a dedicated slot (never packed into point_metric/other_metric,
    whose meaning already depends on kind) means reading it never requires
    guessing which of several interpretations currently applies."""
    stat = v.point_metric[c] if v.kinds[c] == KIND_POINT else v.other_metric[c]
    return (v.chan_src[c], v.kinds[c], stat, v.point1_choice[c], v.point2_choice[c],
            v.buddha_points[c], v.buddha_compression[c], v.buddha_brightness[c],
            v.constant_choice[c])


# Every field a view can define in a keyframe: (field, label, is_float) -
# mirrors OBST_FIELD_INFO. None of these are float (a color model or a whole
# channel's configuration is held between keyframes, never interpolated).
VIEW_FIELD_INFO = [("mode", "Color model", False), ("chan0", "Channel 1", False),
                   ("chan1", "Channel 2", False), ("chan2", "Channel 3", False)]


def view_field_value(v, field):
    """The current value of one of a view's VIEW_FIELD_INFO fields, in the
    same shape a keyframe stores it in."""
    if field == "mode":
        return v.mode
    return view_channel_value(v, int(field[-1]))


ViewSettings = namedtuple("ViewSettings", ["mode", "kinds", "point_metric", "other_metric",
                                           "point1_choice", "point2_choice", "chan_src",
                                           "buddha_points", "buddha_compression",
                                           "buddha_brightness", "constant_choice"])


def decompose_chans(mode, chans):
    """A color model plus 3 channels' worth of raw 9-tuples (as stored in a
    keyframe, or built by view_channel_value) -> a ViewSettings, decomposed
    into the separate per-channel fields render_full_image's coloring step
    actually reads. Used by resolve_views() for every view alike."""
    chan_src = tuple(c[0] for c in chans)
    kinds = tuple(c[1] for c in chans)
    point_metric = tuple(c[2] if c[1] == KIND_POINT else DEFAULT_POINT_METRICS[i]
                         for i, c in enumerate(chans))
    other_metric = tuple(c[2] if c[1] == KIND_OTHER else DEFAULT_OTHER_METRICS[i]
                         for i, c in enumerate(chans))
    point1_choice = tuple(c[3] for c in chans)
    point2_choice = tuple(c[4] for c in chans)
    buddha_points = tuple(c[5] for c in chans)
    buddha_compression = tuple(c[6] for c in chans)
    buddha_brightness = tuple(c[7] for c in chans)
    constant_choice = tuple(c[8] for c in chans)
    return ViewSettings(mode, kinds, point_metric, other_metric, point1_choice, point2_choice,
                        chan_src, buddha_points, buddha_compression, buddha_brightness,
                        constant_choice)


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


DEFAULT_SETTING_VALUES = {
    "iter": DEFAULT_ITER, "dist": DEFAULT_DIST, "seg_mult": DEFAULT_SEGMENT_MULT,
    "shape": DEFAULT_SHAPE,
    **{p.key: p.default for p in SHAPE_PARAM_INFO},
    "fuzz_on": DEFAULT_FUZZ.enabled, "emit_angle": DEFAULT_FUZZ.emit_angle,
    "emit_jitter": DEFAULT_FUZZ.emit_jitter, "bnd_offset": DEFAULT_FUZZ.bnd_offset,
    "bnd_jitter": DEFAULT_FUZZ.bnd_jitter, "bnd_mult": DEFAULT_FUZZ.bnd_mult,
    "self_offset": DEFAULT_FUZZ.self_offset, "self_jitter": DEFAULT_FUZZ.self_jitter,
    "self_mult": DEFAULT_FUZZ.self_mult, "seed": DEFAULT_FUZZ.seed,
    "buddha_blur_radius": DEFAULT_BUDDHA_BLUR_RADIUS,
    "buddha_blur_type": DEFAULT_BUDDHA_BLUR_TYPE,
    "main_fire": True,
    "main_in_effect": OBSTACLE_EFFECT_REFLECT,
    "main_out_effect": OBSTACLE_EFFECT_REFLECT,
    "outline": True,
}


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

    slot_fields = set(Obstacle.__slots__) - {"id"}
    obst_info_fields = {field for field, _label, _is_float in OBST_FIELD_INFO}
    if slot_fields != obst_info_fields:
        raise AssertionError(
            "Obstacle.__slots__ and OBST_FIELD_INFO disagree - "
            f"only in __slots__: {slot_fields - obst_info_fields or None}; "
            f"only in OBST_FIELD_INFO: {obst_info_fields - slot_fields or None}")

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

    for attr in ("field", "key", "widget"):
        names = [getattr(p, attr) for p in SHAPE_PARAM_INFO]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise AssertionError(f"SHAPE_PARAM_INFO has duplicate {attr} value(s): {dupes}")
    bad_fmt = [p.field for p in SHAPE_PARAM_INFO if p.fmt not in ("size", "angle", "int")]
    if bad_fmt:
        raise AssertionError(f"SHAPE_PARAM_INFO has invalid fmt for: {bad_fmt}")
    non_shape_snapshot_fields = {"max_iter", "max_dist", "seg_mult", "shape", "fuzz",
                                 "buddha_blur_radius", "buddha_blur_type", "main_fire",
                                 "main_in_effect", "main_out_effect"}
    table_fields = {p.field for p in SHAPE_PARAM_INFO}
    snapshot_fields = set(Snapshot._fields) - non_shape_snapshot_fields
    if table_fields != snapshot_fields:
        raise AssertionError(
            "SHAPE_PARAM_INFO and Snapshot's fields disagree - "
            f"only in SHAPE_PARAM_INFO: {table_fields - snapshot_fields or None}; "
            f"only in Snapshot: {snapshot_fields - table_fields or None}")


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


# Color models: channel names (and label colors) for the three table rows
MODES = {
    "RGB": [("Red", "#c62828"), ("Green", "#2e7d32"), ("Blue", "#1565c0")],
    "HSV": [("Hue", "#6a1b9a"), ("Saturation", "#00838f"), ("Value", "#616161")],
}
MODE_NAMES = list(MODES)
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


def simulation_info_lines(settings):
    """The current simulation's settings (shared across every view, since
    coloration is the only thing that varies by view) as a list of text
    lines, mirroring the Simulation and Fuzzing tabs' own organization: the
    iteration and distance limits, the segment length multiplier (if not
    the default), the bounding shape and its own parameters (one clause per
    line), its ray behavior (only if not the default), and the fuzzing
    settings."""
    lines = [f"Iter limit:{settings.max_iter}"]
    dist_text = "none" if settings.max_dist == 0 else f"{settings.max_dist:.2f}"
    lines.append(f"Dist limit:{dist_text}")
    if settings.seg_mult < 1.0:
        lines.append(f"Segment:x{settings.seg_mult:.2f}")
    lines.append("Shape:")
    for clause in describe_shape(settings.shape, settings.params).split(", "):
        lines.append(f"  {clause}")
    if (not settings.main_fire or settings.main_in_effect != OBSTACLE_EFFECT_REFLECT
            or settings.main_out_effect != OBSTACLE_EFFECT_REFLECT):
        lines.append("Bound rays:")
        lines.append(f"  Fire:{'On' if settings.main_fire else 'Off'}")
        lines.append(f"  In:{OBSTACLE_EFFECT_NAMES[settings.main_in_effect]}")
        lines.append(f"  Out:{OBSTACLE_EFFECT_NAMES[settings.main_out_effect]}")
    f = settings.fuzz
    if not f.enabled:
        lines.append("Fuzz:Off")
    else:
        lines.append("Fuzz:On")
        lines.append(f"  Emit {f.emit_angle:.1f}\u00b0 \u00b1{f.emit_jitter:.1f}\u00b0")
        lines.append(f"  Bound {f.bnd_offset:+.1f}\u00b0 \u00b1{f.bnd_jitter:.1f}\u00b0 "
                     f"\u00d7{f.bnd_mult:.2f}")
        lines.append(f"  Self {f.self_offset:+.1f}\u00b0 \u00b1{f.self_jitter:.1f}\u00b0 "
                     f"\u00d7{f.self_mult:.2f}")
        lines.append(f"  Seed {f.seed}")
    return lines


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


def add_left_simulation_margin(img, settings):
    """Returns a new, wider copy of `img`: a solid black strip added to its
    left edge, as tall as `img` itself and exactly half as wide as `img`
    (the figure) itself, holding simulation_info_lines. Never mutates
    `img`."""
    h, w, _ = img.shape
    margin_w = w // 2
    out = np.empty((h, margin_w + w, 3), dtype=img.dtype)
    out[:, margin_w:] = img
    out[:, :margin_w] = COLORATION_INFO_BG
    scale = resolution_scale(h)   # the view's own size, not the margin's (see
                                  # add_bottom_color_margin's identical fix)
    m = COLORATION_INFO_MARGIN * scale
    _draw_text_block(out, simulation_info_lines(settings), m, m,
                     margin_w - 2 * m, h - 2 * m, scale, draw_bg=False)
    return out


def apply_info_locations(rgb, views, view_width, settings, color_location, sim_location,
                         mutate_ok):
    """Applies the Export designer's Color info location and Simulation
    info location choices to `rgb` (already composed, before any margin
    expansion), returning the (possibly larger) final image - a plain
    module-level function, safe to call from a background thread, since it
    never reads a Tk variable directly (callers capture color_location/
    sim_location on the main thread first). mutate_ok=False never draws
    into `rgb` in place for the "Overlay" case (a defensive copy is made
    first instead) - needed when `rgb` might be a shared, reused buffer
    like self.frame."""
    if color_location == "Overlay":
        if not mutate_ok:
            rgb = rgb.copy()
        draw_coloration_info(rgb, views, view_width)
    elif color_location == "Bottom margin":
        rgb = add_bottom_color_margin(rgb, views, view_width)
    if sim_location == "Left margin":
        rgb = add_left_simulation_margin(rgb, settings)
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
    ("outline", "Bounding shape outline (white)", True),
]
DECO_PREVIEW = [    # drawn over the image on screen only: never saved, never in a keyframe
    ("pending", "Pending shape outline (dashed)", True),
    ("ray_path", "Ray path", True),
    ("ray_start", "Ray start marker", True),
    ("ray_end", "Ray endpoint marker", True),
    ("guides", "Endpoint guide lines", True),
    ("samples", "Color sample points (one ring per point-based channel)", False),
]

# Hints shown under the shape parameter controls
SHAPE_HINTS = {
    SHAPE_ELLIPSE: "Full extents; 2.0 \u00d7 2.0 is the unit circle. Rotation is "
                   "counterclockwise. Changes need Start; a dashed outline "
                   "previews them.",
    SHAPE_POLYGON: "Vertices lie on the unit circle; at rotation 0 one points up. "
                   "Rotation is counterclockwise. Changes need Start; a dashed "
                   "outline previews them.",
    SHAPE_MOSS_EGG: "Built from an isosceles triangle with apex angle ACB; the "
                    "round end is at the bottom, the pointed/capped end at the "
                    "top, before rotation. Changes need Start.",
    SHAPE_SECTOR: "A pie slice: Span is the wedge's total angle (up to 180\u00b0), "
                  "Rotation orients its bisector. Changes need Start.",
    SHAPE_SEGMENT: "A circle's cap beyond a chord: Span is the angle the chord "
                   "subtends at the centre, Rotation orients the cap. Changes "
                   "need Start.",
    SHAPE_LENS: "The overlap of two circles, Separation apart along Rotation's "
               "direction. Changes need Start.",
    SHAPE_REULEAUX: "A constant-width shape built on a regular polygon with an "
                    "odd number of sides. Changes need Start.",
    SHAPE_CAPSULE: "A rectangle capped with a semicircle at each end (a "
                   "\"stadium\"): Length is the straight part, Radius the caps. "
                   "Changes need Start.",
}

# Raw measurements filled in by trace_path / compute_stats
RAW_BOUNDARY = 0    # number of bounces off the bounding shape
RAW_SELF_K = 1      # index of the segment hit at the first self-touch (-1: never)
RAW_SELF_M = 2      # index of the hit point at the first self-touch
RAW_DEFLECT = 3     # sum of deflection angles, degrees
RAW_LENGTH = 4      # total path length
RAW_LAST = 5        # length of the last segment (as drawn)
RAW_AREA = 6        # area of the first self-loop
RAW_LAST_DIR = 7    # direction of the last segment, degrees (-180..180)
RAW_BOUNCES = 8     # number of bounces
RAW_TERM = 9        # why the ray stopped (TERM_*)
RAW_LAST_FULL = 10  # uncut length of a distance-cut last segment (0 if not cut)
RAW_DISP = 11       # straight-line distance from start point to end point
N_RAW = 12

# Obstacles: a fixed-width row per obstacle (padded with zeros) so their bounds
# can live in one ordinary numpy array for the njit kernels.
OBST_BOUND_LEN = 3 + 2 * POLY_MAX_SIDES
OBST_OBJ_BASE = -100        # best_obj <= this encodes "obstacle index -100 - best_obj"
OBST_POS_MIN = -1.0         # obstacle centre (cx, cy) slider range
OBST_POS_MAX = 1.0
OBST_RADIUS_MIN = 0.0       # polygon obstacle circumradius slider range
OBST_RADIUS_MAX = 2.0


# --------------------------------------------------------------------------- #
# Simulation kernels
@njit(cache=True)
def _circle_roots(px, py, dx, dy, cx, cy, r):
    """Both t roots of the ray hitting circle (cx, cy, r) (near, far); a
    sentinel (1e30, -1e30) if the ray's line misses the circle entirely."""
    lx = px - cx
    ly = py - cy
    qb = lx * dx + ly * dy
    qc = lx * lx + ly * ly - r * r
    disc = qb * qb - qc
    if disc < 0.0:
        return 1e30, -1e30
    sq = sqrt(disc)
    return -qb - sq, -qb + sq


@njit(cache=True)
def _angle_in_range(ang, ang_lo, span):
    d = ang - ang_lo
    while d < 0.0:
        d += 2.0 * pi
    while d >= 2.0 * pi:
        d -= 2.0 * pi
    return d <= span


@njit(cache=True)
def arc_exit_hit(px, py, dx, dy, cx, cy, r, ang_lo, span):
    """Smallest t > EPS where the ray crosses the bounded arc (cx, cy, r,
    ang_lo, span); 1e30 if it never does."""
    t_near, t_far = _circle_roots(px, py, dx, dy, cx, cy, r)
    best = 1e30
    if t_near > EPS:
        hx = px + t_near * dx
        hy = py + t_near * dy
        if _angle_in_range(atan2(hy - cy, hx - cx), ang_lo, span):
            best = t_near
    if t_far > EPS and t_far < best:
        hx = px + t_far * dx
        hy = py + t_far * dy
        if _angle_in_range(atan2(hy - cy, hx - cx), ang_lo, span):
            best = t_far
    return best


@njit(cache=True)
def edge_exit_hit(px, py, dx, dy, ax, ay, bx, by):
    """Smallest t > EPS where the ray crosses the bounded segment a->b; 1e30
    if it never does."""
    ex = bx - ax
    ey = by - ay
    denom = dx * ey - dy * ex
    if -1e-14 < denom < 1e-14:
        return 1e30
    wx = ax - px
    wy = ay - py
    t = (wx * ey - wy * ex) / denom
    if t <= EPS:
        return 1e30
    u = (wx * dy - wy * dx) / denom
    if u < 0.0 or u > 1.0:
        return 1e30
    return t


@njit(cache=True)
def arcs_edges_hit(px, py, dx, dy, bound):
    """The nearest point (t > EPS) where a ray touches the actual boundary of
    an "arcs + edges" shape (Moss's egg / circular sector / circular segment /
    capsule) - works identically whether the ray starts inside (this is then
    its exit) or outside (this is then its entry), since a convex shape's
    boundary is touched at exactly the near point either way."""
    n_a = int(bound[1])
    n_e = int(bound[2])
    best_t = 1e30
    best_nx = 0.0
    best_ny = 0.0
    idx = 3
    for _i in range(n_a):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        ang_lo = bound[idx + 3]
        span = bound[idx + 4]
        idx += 5
        t = arc_exit_hit(px, py, dx, dy, cx, cy, r, ang_lo, span)
        if t < best_t:
            best_t = t
            best_nx = (px + t * dx) - cx
            best_ny = (py + t * dy) - cy
    for _i in range(n_e):
        ax = bound[idx]
        ay = bound[idx + 1]
        bx = bound[idx + 2]
        by = bound[idx + 3]
        idx += 4
        t = edge_exit_hit(px, py, dx, dy, ax, ay, bx, by)
        if t < best_t:
            best_t = t
            best_nx = by - ay
            best_ny = -(bx - ax)
    return best_t, best_nx, best_ny


@njit(cache=True)
def inside_arcs_edges_and(x, y, bound):
    """Is (x, y) inside an "arcs + edges" shape that IS exactly the
    intersection of its arcs' full circles and its edges' half-planes
    (circular sector, circular segment)? True iff within range of every arc's
    circle and on the interior side of every edge."""
    n_a = int(bound[1])
    n_e = int(bound[2])
    idx = 3
    for _i in range(n_a):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        idx += 5
        if (x - cx) ** 2 + (y - cy) ** 2 > r * r:
            return False
    for _i in range(n_e):
        ax = bound[idx]
        ay = bound[idx + 1]
        bx = bound[idx + 2]
        by = bound[idx + 3]
        idx += 4
        if (bx - ax) * (y - ay) - (by - ay) * (x - ax) < 0.0:
            return False
    return True


@njit(cache=True)
def inside_arcs_edges_star(x, y, bound):
    """Is (x, y) inside an "arcs + edges" shape whose local origin is always
    strictly interior (Moss's egg, capsule) but whose arcs only cover part of
    their own circles, so the AND-of-full-primitives test above doesn't apply?
    Cast a ray from the origin through (x, y) and compare distances - valid
    because the shape is convex and genuinely star-shaped from the origin."""
    dist = sqrt(x * x + y * y)
    if dist < 1e-9:
        return True
    ux = x / dist
    uy = y / dist
    t_exit, _nx, _ny = arcs_edges_hit(0.0, 0.0, ux, uy, bound)
    return dist <= t_exit


@njit(cache=True)
def inside_circles(x, y, bound):
    """Is (x, y) inside every one of a "circle intersection" shape's circles
    (two-circle lens / Reuleaux polygon)?"""
    n = int(bound[1])
    idx = 2
    for _i in range(n):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        idx += 3
        if (x - cx) ** 2 + (y - cy) ** 2 > r * r:
            return False
    return True


@njit(cache=True)
def circles_exit_hit(px, py, dx, dy, bound):
    """boundary_hit for the "circle intersection" family, from inside: the
    exit is the smallest of every circle's own far root."""
    n = int(bound[1])
    idx = 2
    best_t = 1e30
    best_nx = 0.0
    best_ny = 0.0
    for _i in range(n):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        idx += 3
        _t_near, t_far = _circle_roots(px, py, dx, dy, cx, cy, r)
        if t_far < best_t:
            best_t = t_far
            best_nx = (px + t_far * dx) - cx
            best_ny = (py + t_far * dy) - cy
    return best_t, best_nx, best_ny


@njit(cache=True)
def circles_entry_hit(px, py, dx, dy, bound):
    """obstacle_hit for the "circle intersection" family, from outside: the
    entry/exit of the intersection are the latest near-root / earliest
    far-root across all circles (Cyrus-Beck-style interval intersection)."""
    n = int(bound[1])
    idx = 2
    t_enter = -1e30
    t_exit = 1e30
    enter_nx = 0.0
    enter_ny = 0.0
    for _i in range(n):
        cx = bound[idx]
        cy = bound[idx + 1]
        r = bound[idx + 2]
        idx += 3
        t_near, t_far = _circle_roots(px, py, dx, dy, cx, cy, r)
        if t_near > 1e29:                       # this circle's line misses entirely
            return -1.0, 0.0, 0.0
        if t_near > t_enter:
            t_enter = t_near
            enter_nx = (px + t_near * dx) - cx
            enter_ny = (py + t_near * dy) - cy
        if t_far < t_exit:
            t_exit = t_far
    if t_enter <= EPS or t_enter > t_exit:
        return -1.0, 0.0, 0.0
    return t_enter, enter_nx, enter_ny


# --------------------------------------------------------------------------- #
@njit(cache=True)
def inside_boundary(x, y, shape, bound):
    """Is the point (x, y) strictly inside the bounding shape?"""
    if shape == SHAPE_ELLIPSE:
        return bound[1] * x * x + 2.0 * bound[2] * x * y + bound[3] * y * y < 1.0
    if shape == SHAPE_POLYGON:
        h = bound[1]
        n = int(bound[2])
        for k in range(n):
            if bound[3 + 2 * k] * x + bound[4 + 2 * k] * y >= h:
                return False
        return True
    if shape == SHAPE_LENS or shape == SHAPE_REULEAUX:
        return inside_circles(x, y, bound)
    if shape == SHAPE_SECTOR or shape == SHAPE_SEGMENT:
        return inside_arcs_edges_and(x, y, bound)
    return inside_arcs_edges_star(x, y, bound)   # MOSS_EGG, CAPSULE


@njit(cache=True)
def inside_boundary_grid(gx, gy, shape, bound):
    """inside_boundary(), over a whole (H, W) grid at once - compiled, so this
    stays fast even though it's a plain loop; used for drawing the outline of
    shapes without a simple analytic gradient (see compute_ring)."""
    h, w = gx.shape
    out = np.empty((h, w), dtype=np.bool_)
    for j in range(h):
        for i in range(w):
            out[j, i] = inside_boundary(gx[j, i], gy[j, i], shape, bound)
    return out



@njit(cache=True)
def emits_ray(x, y, shape, bound, main_ctl, obst_shapes, obst_bounds, obst_cx, obst_cy,
              obst_fire_inside):
    """Does a ray actually start at (x, y)? True if (x, y) is inside any shape
    whose "fire rays inside" is on - the main bounding shape (main_ctl[0]) or
    an obstacle. Being inside shapes that have it off makes no difference."""
    if main_ctl[0] != 0 and inside_boundary(x, y, shape, bound):
        return True
    for oi in range(len(obst_shapes)):
        if obst_fire_inside[oi] and inside_boundary(x - obst_cx[oi], y - obst_cy[oi],
                                                     obst_shapes[oi], obst_bounds[oi]):
            return True
    return False


@njit(cache=True)
def boundary_hit(px, py, dx, dy, shape, bound):
    """
    For a ray starting at (px, py), inside the shape, heading along the unit
    vector (dx, dy): the distance t to where it meets the boundary, and the
    normal there (not necessarily unit length).
    """
    if shape == SHAPE_ELLIPSE:
        m11 = bound[1]
        m12 = bound[2]
        m22 = bound[3]
        # Solve  A t^2 + 2 B t + C = 0  for the point p + t d on  p^T M p = 1.
        # The ray starts inside, so take the far root.
        qa = m11 * dx * dx + 2.0 * m12 * dx * dy + m22 * dy * dy
        qb = m11 * px * dx + m12 * (px * dy + py * dx) + m22 * py * dy
        qc = m11 * px * px + 2.0 * m12 * px * py + m22 * py * py - 1.0
        disc = qb * qb - qa * qc
        if disc < 0.0:
            disc = 0.0
        t = (-qb + sqrt(disc)) / qa
        hx = px + t * dx
        hy = py + t * dy
        return t, m11 * hx + m12 * hy, m12 * hx + m22 * hy
    if shape == SHAPE_POLYGON:
        # Convex polygon: the ray leaves through the nearest edge it is heading out of
        h = bound[1]
        n = int(bound[2])
        best = 1e30
        bnx = 0.0
        bny = 0.0
        for k in range(n):
            nx = bound[3 + 2 * k]
            ny = bound[4 + 2 * k]
            nd = nx * dx + ny * dy
            if nd > 1e-12:
                t = (h - (nx * px + ny * py)) / nd
                if t < 0.0:
                    t = 0.0
                if t < best:
                    best = t
                    bnx = nx
                    bny = ny
        return best, bnx, bny
    if shape == SHAPE_LENS or shape == SHAPE_REULEAUX:
        return circles_exit_hit(px, py, dx, dy, bound)
    return arcs_edges_hit(px, py, dx, dy, bound)   # MOSS_EGG, SECTOR, SEGMENT, CAPSULE


@njit(cache=True)
def obstacle_hit(px, py, dx, dy, shape, bound):
    """
    For a ray at (px, py), OUTSIDE this solid obstacle, heading along the unit
    vector (dx, dy): the distance t to where it first enters the obstacle (and
    the normal there), or t < 0 if this ray's forward path (t > EPS) never
    reaches it. Ellipses use the near root of the quadratic; convex polygons
    use a Cyrus-Beck-style half-plane clip.
    """
    if shape == SHAPE_ELLIPSE:
        m11 = bound[1]
        m12 = bound[2]
        m22 = bound[3]
        qa = m11 * dx * dx + 2.0 * m12 * dx * dy + m22 * dy * dy
        qb = m11 * px * dx + m12 * (px * dy + py * dx) + m22 * py * dy
        qc = m11 * px * px + 2.0 * m12 * px * py + m22 * py * py - 1.0
        disc = qb * qb - qa * qc
        if disc < 0.0 or qa <= 0.0:
            return -1.0, 0.0, 0.0
        t = (-qb - sqrt(disc)) / qa                # near root: the entry point
        if t <= EPS:
            return -1.0, 0.0, 0.0
        hx = px + t * dx
        hy = py + t * dy
        return t, m11 * hx + m12 * hy, m12 * hx + m22 * hy
    if shape == SHAPE_POLYGON:
        # Convex polygon, as a solid obstacle: clip the ray against every edge's
        # half-plane; the entry time is the latest of the "becomes satisfied"
        # crossings, the exit time the earliest of the "stops being satisfied" ones.
        h = bound[1]
        n = int(bound[2])
        t_enter = -1e30
        t_exit = 1e30
        enter_nx = 0.0
        enter_ny = 0.0
        for k in range(n):
            nx = bound[3 + 2 * k]
            ny = bound[4 + 2 * k]
            nd = nx * dx + ny * dy
            num = h - (nx * px + ny * py)
            if nd > 1e-14:
                if num < 0.0:
                    return -1.0, 0.0, 0.0            # never satisfied for t >= 0
                t = num / nd
                if t < t_exit:
                    t_exit = t
            elif nd < -1e-14:
                if num < 0.0:
                    t = num / nd
                    if t > t_enter:
                        t_enter = t
                        enter_nx = nx
                        enter_ny = ny
            else:
                if num < 0.0:
                    return -1.0, 0.0, 0.0
        if t_enter <= EPS or t_enter > t_exit:
            return -1.0, 0.0, 0.0
        return t_enter, enter_nx, enter_ny
    if shape == SHAPE_LENS or shape == SHAPE_REULEAUX:
        return circles_entry_hit(px, py, dx, dy, bound)
    t, nx, ny = arcs_edges_hit(px, py, dx, dy, bound)   # MOSS_EGG, SECTOR, SEGMENT, CAPSULE
    if t > 1e29:
        return -1.0, 0.0, 0.0
    return t, nx, ny


@njit(cache=True)
def square_exit_t(px, py, dx, dy):
    """Distance along the unit direction (dx, dy) from (px, py) to where the
    ray leaves the [-1, 1] image square (0 if it is already outside it and
    heading away)."""
    t = 1e30
    if dx > 1e-15:
        t = min(t, (1.0 - px) / dx)
    elif dx < -1e-15:
        t = min(t, (-1.0 - px) / dx)
    if dy > 1e-15:
        t = min(t, (1.0 - py) / dy)
    elif dy < -1e-15:
        t = min(t, (-1.0 - py) / dy)
    return max(t, 0.0)


ESCAPE_OBJ = -3      # trace_path's "what did the ray hit": nothing, it escaped


@njit(nogil=True, cache=True)
def trace_path(x0, y0, pts, kinds, max_iter, max_dist, shape, bound, main_ctl, seg_mult,
               fuzz, obst_shapes, obst_bounds, obst_cx, obst_cy, obst_effects, raw):
    """
    Trace one ray, storing the start point and every
    bounce point in pts (needs at least max_iter + 2 rows, 2 columns), the
    "kind" of each of those points in kinds (KIND_START / KIND_BOUNDARY /
    KIND_SELF / KIND_DISTCUT / KIND_OBSTACLE_TERM / KIND_ESCAPED, same length
    as pts), and
    some raw measurements in raw (N_RAW entries). If the distance limit is
    reached part-way along a segment, the ray's end point (on that segment) is
    appended to pts (kind KIND_DISTCUT) too.

    `seg_mult` (0 < seg_mult <= 1) shortens every ordinary bounce's recorded
    segment to that fraction of its true length - the endpoint moves back
    towards the segment's start - while the ray still continues on from there
    in the direction the bounce actually calculated (from the true, full-length
    hit point's surface normal), so the ray visibly changes direction in
    mid-air rather than exactly at the boundary/obstacle/its own path. 1 means
    no effect. This only applies to ordinary bounces - a ray that terminates
    (distance limit, a "terminate" obstacle, perpendicular incidence) always
    still reaches its true, unshortened endpoint, since there's no next
    segment to continue into.

    `main_ctl` is [fires rays inside, incoming effect, outgoing effect] for the
    main bounding shape (see main_ctl_array).

    `obst_shapes`/`obst_bounds`/`obst_cx`/`obst_cy`/`obst_effects` describe
    every obstacle (parallel arrays; each obstacle's own bound is centred on
    (obst_cx[i], obst_cy[i]); `obst_effects` is an (n, 2) array of
    OBSTACLE_EFFECT_* codes: column 0 for rays coming in from outside, column
    1 for rays going out from inside). The main bounding shape and every
    obstacle are treated alike: what happens to a ray as it reaches an edge
    depends only on the effect chosen for that direction. Reflect acts like a
    mirror (using the boundary's fuzzing settings and counting toward the same
    bounce total); Terminate ends the ray the instant the edge is reached;
    Pass through lets the ray cross (flipping whether it is inside that
    shape) without a bounce or a recorded point. A ray starts inside every
    shape that contains its start point. A shape that passes rays through in
    both directions is ignored. If a ray has nothing left ahead to hit, it
    escapes (TERM_ESCAPED): its last point is where it leaves the [-1, 1]
    image square.

    `fuzz` is [enabled,
               emit_angle, emit_jitter,
               boundary_offset, boundary_jitter, boundary_mult,
               self_offset, self_jitter, self_mult]
    (angles in degrees; emit_angle is the emission direction itself, not an
    offset). When enabled, the RNG is used for the jitter terms, so the caller
    must have seeded it beforehand for reproducibility.

    Returns (term, npts):
      term  why the ray stopped: TERM_CONFINED, TERM_ITER, TERM_DIST,
            TERM_OBSTACLE, TERM_PERPENDICULAR or TERM_ESCAPED
      npts  number of valid points in pts (the last is the final location)
    """
    fuzzing = fuzz[0] > 0.5
    n_obst = len(obst_shapes)
    n_shapes = n_obst + 1                    # shape 0 = the main bounding shape, then obstacles

    pts[0, 0] = x0
    pts[0, 1] = y0
    kinds[0] = KIND_START
    npts = 1
    n_bounces = 0
    n_boundary = 0
    self_k = -1
    self_m = -1
    deflect = 0.0
    dist = 0.0
    last_full = 0.0
    term = TERM_ITER

    # Which shapes the ray is inside right now. It changes only when the ray
    # passes through an edge (see below), and is always decided by this flag
    # rather than re-derived from the geometry, so a ray grazing an edge can't
    # be judged inside by one test and outside by the next.
    inside = np.zeros(n_shapes, dtype=np.bool_)
    inside[0] = inside_boundary(x0, y0, shape, bound)
    for oi in range(n_obst):
        inside[oi + 1] = inside_boundary(x0 - obst_cx[oi], y0 - obst_cy[oi],
                                         obst_shapes[oi], obst_bounds[oi])
    max_cross = 2 * n_shapes + 2             # a straight line crosses a convex shape at most twice

    px = x0
    py = y0
    if fuzzing:
        ej = fuzz[2]
        jitter = np.random.uniform(-ej, ej) if ej > 0.0 else 0.0
        ang0 = (fuzz[1] + jitter) * pi / 180.0
        dx = cos(ang0)
        dy = sin(ang0)
    else:
        dx = INV_SQRT2
        dy = INV_SQRT2
    last_obj = -2                            # >= 0: a segment; <= OBST_OBJ_BASE: a shape's edge

    for _ in range(max_iter):
        # Find where this straight segment ends: at a reflecting edge, a
        # terminating edge, the ray's own earlier path, or (if there is nothing
        # ahead) the edge of the image. Passing through an edge is not an end:
        # the ray just flips its inside/outside state for that shape and carries
        # on, all as part of the same segment, so a crossing is never a
        # bounce, never a recorded point, and never shortens the segment.
        t_off = 0.0                          # distance already advanced along this segment
        n_cross = 0
        stop = False
        best_t = 1e30
        best_obj = -1
        hit_eff = OBSTACLE_EFFECT_PASS
        wnx = 0.0
        wny = 0.0
        while True:
            qx = px + t_off * dx
            qy = py + t_off * dy
            best_t = 1e30
            best_obj = -1                    # nothing hit (yet)
            hit_eff = OBSTACLE_EFFECT_PASS

            # Distance to every shape's edge: from inside, the way out; from
            # outside, the way in. A shape that passes rays through in both
            # directions is not there at all.
            for si in range(n_shapes):
                if si == 0:
                    e_in = main_ctl[1]
                    e_out = main_ctl[2]
                    ocx = 0.0
                    ocy = 0.0
                else:
                    e_in = obst_effects[si - 1, 0]
                    e_out = obst_effects[si - 1, 1]
                    ocx = obst_cx[si - 1]
                    ocy = obst_cy[si - 1]
                if e_in == OBSTACLE_EFFECT_PASS and e_out == OBSTACLE_EFFECT_PASS:
                    continue
                if si == 0:
                    if inside[0]:
                        t, hnx, hny = boundary_hit(qx, qy, dx, dy, shape, bound)
                    else:
                        t, hnx, hny = obstacle_hit(qx, qy, dx, dy, shape, bound)
                elif inside[si]:
                    t, hnx, hny = boundary_hit(qx - ocx, qy - ocy, dx, dy,
                                               obst_shapes[si - 1], obst_bounds[si - 1])
                else:
                    t, hnx, hny = obstacle_hit(qx - ocx, qy - ocy, dx, dy,
                                               obst_shapes[si - 1], obst_bounds[si - 1])
                if t >= 0.0 and t < best_t:
                    best_t = t
                    best_obj = OBST_OBJ_BASE - si
                    wnx = hnx
                    wny = hny
                    hit_eff = e_out if inside[si] else e_in

            # Distance to earlier path segments: pts[k] -> pts[k+1]. A
            # self-segment collision only overrides an edge collision already
            # found above if it's MEANINGFULLY closer (not just closer by
            # floating-point noise) - otherwise, when retracing an old path
            # back through a point that's also on an edge (e.g. after a
            # self-bounce at normal incidence sends the ray straight back the
            # way it came), a near-tie could let the self-segment "win" and
            # the ray would reflect as if the solid wall weren't there,
            # sneaking through it.
            for k in range(npts - 1):
                if k == last_obj or k == npts - 2:   # skip segment we're leaving
                    continue
                ax = pts[k, 0]
                ay = pts[k, 1]
                ex = pts[k + 1, 0] - ax
                ey = pts[k + 1, 1] - ay
                denom = dx * ey - dy * ex
                if -1e-14 < denom < 1e-14:
                    continue
                wx = ax - qx
                wy = ay - qy
                t = (wx * ey - wy * ex) / denom
                if t <= EPS or t >= best_t - SELF_TIE_EPS:
                    continue
                u = (wx * dy - wy * dx) / denom
                if u < 0.0 or u > 1.0:
                    continue
                best_t = t
                best_obj = k

            if best_t >= 1e29:
                # Nothing at all ahead: the ray escapes. Its segment ends where
                # it leaves the image square (never before the last edge it
                # passed through).
                best_obj = ESCAPE_OBJ
                best_t = max(square_exit_t(px, py, dx, dy), t_off) - t_off

            if best_obj <= OBST_OBJ_BASE and hit_eff == OBSTACLE_EFFECT_PASS:
                # Through an edge: flip inside/outside for that shape, carry on
                inside[OBST_OBJ_BASE - best_obj] = not inside[OBST_OBJ_BASE - best_obj]
                t_off += best_t
                n_cross += 1
                if n_cross > max_cross:      # coincident edges etc.: cannot happen normally
                    term = TERM_CONFINED
                    stop = True
                    break
                continue
            break                            # a real end to this segment
        if stop:
            break
        total_t = t_off + best_t             # the segment's full length, crossings included

        # Distance limit: cut this segment off where the limit is reached
        if dist + total_t > max_dist:
            rem = max_dist - dist
            if rem > 1e-12:
                pts[npts, 0] = px + rem * dx
                pts[npts, 1] = py + rem * dy
                kinds[npts] = KIND_DISTCUT
                npts += 1
                last_full = total_t
            term = TERM_DIST
            break

        # Nothing ahead: the ray leaves for good
        if best_obj == ESCAPE_OBJ:
            pts[npts, 0] = px + total_t * dx
            pts[npts, 1] = py + total_t * dy
            kinds[npts] = KIND_ESCAPED
            npts += 1
            term = TERM_ESCAPED
            break

        # A "terminate" edge ends the ray outright, with no reflection
        if best_obj <= OBST_OBJ_BASE and hit_eff == OBSTACLE_EFFECT_TERMINATE:
            pts[npts, 0] = px + total_t * dx
            pts[npts, 1] = py + total_t * dy
            kinds[npts] = KIND_OBSTACLE_TERM
            npts += 1
            term = TERM_OBSTACLE
            break

        # Move to the hit point
        hx = px + total_t * dx
        hy = py + total_t * dy

        # Surface normal at the hit point
        if best_obj <= OBST_OBJ_BASE:
            nx = wnx
            ny = wny
            n_boundary += 1                      # any reflecting edge counts as boundary
        else:
            if self_k < 0:                       # first time the ray touches itself
                self_k = best_obj
                self_m = npts                    # index the hit point will occupy
            nx = -(pts[best_obj + 1, 1] - pts[best_obj, 1])
            ny = pts[best_obj + 1, 0] - pts[best_obj, 0]
        nlen = sqrt(nx * nx + ny * ny)
        if nlen < 1e-12:
            # A degenerate (zero-length) normal means the ray has become
            # trapped at a single point (e.g. a sharp corner like a sector's
            # apex) rather than crossing a real surface - treat it the same
            # as becoming confined, rather than dividing by zero.
            term = TERM_CONFINED
            break
        nx /= nlen
        ny /= nlen

        # Perfectly perpendicular incidence: the ray would reflect straight
        # back the way it came, retracing its own path - and which side of
        # itself it ends up on from there is undefined, so the ray ends here
        # instead (a distinct termination reason, though it behaves like
        # confinement in that it's a sudden stop rather than a hit on a
        # boundary/obstacle/distance/iteration limit).
        dot = dx * nx + dy * ny
        if dot > 1.0 - RETRO_EPS or dot < -1.0 + RETRO_EPS:
            pts[npts, 0] = hx
            pts[npts, 1] = hy
            kinds[npts] = KIND_SELF if best_obj >= 0 else KIND_BOUNDARY
            npts += 1
            term = TERM_PERPENDICULAR
            break

        # Perfect-mirror reflection of the incoming direction
        odx = dx
        ody = dy
        rdx = dx - 2.0 * dot * nx
        rdy = dy - 2.0 * dot * ny
        rlen = sqrt(rdx * rdx + rdy * rdy)
        rdx /= rlen
        rdy /= rlen

        # Fuzzing: perturb the deflection angle (incoming -> outgoing) itself,
        # so an unfuzzed bounce (offset 0, jitter 0, multiplier 1) is identical
        # to a plain mirror reflection. Bounces off any reflecting edge (the
        # bounding shape or an obstacle) and off the ray's own path have
        # separate settings.
        if fuzzing:
            if best_obj <= OBST_OBJ_BASE:
                f_off = fuzz[3]
                f_jit = fuzz[4]
                f_mult = fuzz[5]
            else:
                f_off = fuzz[6]
                f_jit = fuzz[7]
                f_mult = fuzz[8]
            cross = odx * rdy - ody * rdx
            dotr = odx * rdx + ody * rdy
            base_ang = atan2(cross, dotr)
            jitter_b = np.random.uniform(-f_jit, f_jit) if f_jit > 0.0 else 0.0
            new_ang = base_ang * f_mult + f_off * pi / 180.0 + jitter_b * pi / 180.0
            ca = cos(new_ang)
            sa = sin(new_ang)
            dx = odx * ca - ody * sa
            dy = odx * sa + ody * ca
            dlen = sqrt(dx * dx + dy * dy)
            dx /= dlen
            dy /= dlen
        else:
            dx = rdx
            dy = rdy

        c = odx * dx + ody * dy
        if c > 1.0:
            c = 1.0
        elif c < -1.0:
            c = -1.0
        deflect += acos(c) * (180.0 / pi)

        # The travelled segment now becomes part of the "mirror" set - shortened
        # to seg_mult of its true length if that's less than 1, though the ray
        # still continues on in the direction just calculated above (from the
        # true, full-length hit point's normal), not recalculated from here.
        if seg_mult < 1.0:
            ex = px + seg_mult * total_t * odx
            ey = py + seg_mult * total_t * ody
        else:
            ex = hx
            ey = hy
        pts[npts, 0] = ex
        pts[npts, 1] = ey
        kinds[npts] = KIND_SELF if best_obj >= 0 else KIND_BOUNDARY
        npts += 1
        n_bounces += 1
        dist += seg_mult * total_t
        px = ex
        py = ey
        last_obj = best_obj

        # Pulled back along a segment that passed through edges: the ray is
        # now wherever that is, which may be on the other side of some of
        # them, so re-derive which shapes it is inside from where it actually is.
        if seg_mult < 1.0 and n_cross > 0:
            inside[0] = inside_boundary(ex, ey, shape, bound)
            for oi in range(n_obst):
                inside[oi + 1] = inside_boundary(ex - obst_cx[oi], ey - obst_cy[oi],
                                                 obst_shapes[oi], obst_bounds[oi])

        # Confined to less than one pixel?
        if npts >= CONFINE_WINDOW:
            minx = 1e30
            maxx = -1e30
            miny = 1e30
            maxy = -1e30
            for k in range(npts - CONFINE_WINDOW, npts):
                minx = min(minx, pts[k, 0])
                maxx = max(maxx, pts[k, 0])
                miny = min(miny, pts[k, 1])
                maxy = max(maxy, pts[k, 1])
            if (maxx - minx) < PIXEL and (maxy - miny) < PIXEL:
                term = TERM_CONFINED
                break

    raw[RAW_BOUNDARY] = n_boundary
    raw[RAW_SELF_K] = self_k
    raw[RAW_SELF_M] = self_m
    raw[RAW_DEFLECT] = deflect
    raw[RAW_BOUNCES] = n_bounces
    raw[RAW_TERM] = term
    raw[RAW_LAST_FULL] = last_full
    return term, npts


@njit(cache=True)
def compute_stats(pts, npts, raw, max_iter, seg_mult, out):
    """
    Turn a traced path into the N_STATS metrics (each in 0..1, written to
    `out`). Also fills RAW_LENGTH, RAW_LAST, RAW_AREA, RAW_LAST_DIR and
    RAW_DISP in `raw`. The loop-area metric is the loop's area divided by the
    area of a circle with the same perimeter as the loop, so it is
    dimensionless and needs no reference shape (1 is a perfect circle). The point-comparison metrics are written here using
    start->end as a placeholder pair; the real values (using whichever two
    points a channel is actually set to compare) are derived later, once the
    sample points have been computed.
    """
    bounces = int(raw[RAW_BOUNCES])
    n_boundary = raw[RAW_BOUNDARY]

    # Path lengths
    total = 0.0
    last = 0.0
    for k in range(npts - 1):
        last = sqrt((pts[k + 1, 0] - pts[k, 0]) ** 2 + (pts[k + 1, 1] - pts[k, 1]) ** 2)
        total += last

    # Direction of the last segment (counterclockwise from pointing right)
    ldx = pts[npts - 1, 0] - pts[npts - 2, 0]
    ldy = pts[npts - 1, 1] - pts[npts - 2, 1]
    ang = atan2(ldy, ldx)

    # Straight-line distance from the start point to the end point
    sdx = pts[npts - 1, 0] - pts[0, 0]
    sdy = pts[npts - 1, 1] - pts[0, 1]
    disp = sqrt(sdx * sdx + sdy * sdy)

    # Area of the loop formed when the ray first touches itself: the polygon
    # pts[k+1] ... pts[m], where pts[m] is the hit point on segment k. With
    # the segment length multiplier active, bounce points are systematically
    # pulled back from where the ray's own path actually is, so this polygon
    # no longer has the meaning its name implies - report 0 rather than a
    # number that looks like a real self-intersection area but isn't one.
    area = 0.0
    perimeter = 0.0
    sk = int(raw[RAW_SELF_K])
    sm = int(raw[RAW_SELF_M])
    if sk >= 0 and seg_mult >= 1.0:
        first = sk + 1
        s = 0.0
        for a in range(first, sm + 1):
            b2 = a + 1 if a < sm else first
            s += pts[a, 0] * pts[b2, 1] - pts[b2, 0] * pts[a, 1]
            perimeter += sqrt((pts[b2, 0] - pts[a, 0]) ** 2 + (pts[b2, 1] - pts[a, 1]) ** 2)
        area = abs(s) * 0.5

    raw[RAW_LENGTH] = total
    raw[RAW_LAST] = last
    raw[RAW_AREA] = area
    raw[RAW_LAST_DIR] = ang * (180.0 / pi)
    raw[RAW_DISP] = disp

    # Point-comparison metrics: filled here using start->end (this function has
    # no sample-point array to work from); recolor()/component_values() use the
    # actually-chosen point pair instead. Written so the stats array is never
    # left uninitialized, and so the preview list shows a sensible value even
    # for a channel not currently using one of these metrics.
    out[ST_HDIST] = min(abs(sdx) * 0.5, 1.0)
    out[ST_VDIST] = min(abs(sdy) * 0.5, 1.0)
    out[ST_MANHATTAN] = min((abs(sdx) + abs(sdy)) * 0.5, 1.0)
    out[ST_EUCLIDEAN] = min(disp * 0.5, 1.0)

    out[ST_BOUNCES] = min(bounces / max_iter, 1.0)
    if bounces > 0:
        out[ST_RAY_FRAC] = (bounces - n_boundary) / bounces
        out[ST_BOUNDARY_FRAC] = n_boundary / bounces
        out[ST_ANGLES] = min(raw[RAW_DEFLECT] / (180.0 * bounces), 1.0)
    else:                                          # cut off before the first bounce
        out[ST_RAY_FRAC] = 0.0
        out[ST_BOUNDARY_FRAC] = 0.0
        out[ST_ANGLES] = 0.0
    if perimeter > 1e-12:
        # the area of a circle with circumference `perimeter` is perimeter^2 / (4 pi)
        out[ST_LOOP_AREA] = min(area / (perimeter * perimeter / (4.0 * pi)), 1.0)
    else:
        out[ST_LOOP_AREA] = 0.0
    out[ST_LENGTH] = atan(total / 5.0) * 2.0 / pi
    out[ST_LAST_DIR] = min(max((ang + pi) / (2.0 * pi), 0.0), 1.0)

    term = int(raw[RAW_TERM])
    if term == TERM_ITER:
        out[ST_TERMINATION] = 0.0
    elif term == TERM_CONFINED:
        out[ST_TERMINATION] = 0.2
    elif term == TERM_DIST:
        out[ST_TERMINATION] = 0.4
    elif term == TERM_OBSTACLE:
        out[ST_TERMINATION] = 0.6
    elif term == TERM_PERPENDICULAR:
        out[ST_TERMINATION] = 0.8
    else:                                          # TERM_ESCAPED
        out[ST_TERMINATION] = 1.0

    full = raw[RAW_LAST_FULL]
    if full > 0.0:
        out[ST_LAST_RATIO] = min(last / full, 1.0)
    else:
        out[ST_LAST_RATIO] = 1.0


@njit(cache=True)
def compute_points(pts, npts, raw, kinds, sp):
    """
    Pick the N_POINTS candidate sample points out of a traced path and write
    their (x, y) coordinates to sp (shape (N_POINTS, 2)). The bounce points are
    pts[1] .. pts[n_bounces]; the start point and a distance-cut end point are
    not bounce points. A ray with no bounces of the relevant kind uses its end
    point for that mean/extremum (all/boundary/self); a ray with no bounces at
    all uses its end point for every bounce-based choice.
    """
    nb = int(raw[RAW_BOUNCES])
    ex = pts[npts - 1, 0]
    ey = pts[npts - 1, 1]

    sp[POINT_START, 0] = pts[0, 0]
    sp[POINT_START, 1] = pts[0, 1]
    sp[POINT_END, 0] = ex
    sp[POINT_END, 1] = ey

    if nb > 0:
        ax = 0.0
        ay = 0.0
        bx = 0.0
        by = 0.0
        bn = 0
        sxm = 0.0
        sym = 0.0
        sn = 0
        i_r = 1                                    # max x, any kind
        i_l = 1                                    # min x, any kind
        i_t = 1                                    # max y, any kind
        i_b = 1                                    # min y, any kind
        i_rs = -1                                   # max x, self bounces only
        i_ls = -1                                   # min x, self bounces only
        i_ts = -1                                   # max y, self bounces only
        i_bs = -1                                   # min y, self bounces only
        for k in range(1, nb + 1):
            x = pts[k, 0]
            y = pts[k, 1]
            ax += x
            ay += y
            if kinds[k] == KIND_BOUNDARY:
                bx += x
                by += y
                bn += 1
            elif kinds[k] == KIND_SELF:
                sxm += x
                sym += y
                sn += 1
                if i_rs < 0 or x > pts[i_rs, 0]:
                    i_rs = k
                if i_ls < 0 or x < pts[i_ls, 0]:
                    i_ls = k
                if i_ts < 0 or y > pts[i_ts, 1]:
                    i_ts = k
                if i_bs < 0 or y < pts[i_bs, 1]:
                    i_bs = k
            if x > pts[i_r, 0]:
                i_r = k
            if x < pts[i_l, 0]:
                i_l = k
            if y > pts[i_t, 1]:
                i_t = k
            if y < pts[i_b, 1]:
                i_b = k
        sp[POINT_ALL_MEAN, 0] = ax / nb
        sp[POINT_ALL_MEAN, 1] = ay / nb
        if bn > 0:
            sp[POINT_BOUNDARY_MEAN, 0] = bx / bn
            sp[POINT_BOUNDARY_MEAN, 1] = by / bn
        else:
            sp[POINT_BOUNDARY_MEAN, 0] = ex
            sp[POINT_BOUNDARY_MEAN, 1] = ey
        if sn > 0:
            sp[POINT_SELF_MEAN, 0] = sxm / sn
            sp[POINT_SELF_MEAN, 1] = sym / sn
            sp[POINT_SELF_RIGHT, 0] = pts[i_rs, 0]
            sp[POINT_SELF_RIGHT, 1] = pts[i_rs, 1]
            sp[POINT_SELF_LEFT, 0] = pts[i_ls, 0]
            sp[POINT_SELF_LEFT, 1] = pts[i_ls, 1]
            sp[POINT_SELF_TOP, 0] = pts[i_ts, 0]
            sp[POINT_SELF_TOP, 1] = pts[i_ts, 1]
            sp[POINT_SELF_BOTTOM, 0] = pts[i_bs, 0]
            sp[POINT_SELF_BOTTOM, 1] = pts[i_bs, 1]
        else:
            sp[POINT_SELF_MEAN, 0] = ex
            sp[POINT_SELF_MEAN, 1] = ey
            sp[POINT_SELF_RIGHT, 0] = ex
            sp[POINT_SELF_RIGHT, 1] = ey
            sp[POINT_SELF_LEFT, 0] = ex
            sp[POINT_SELF_LEFT, 1] = ey
            sp[POINT_SELF_TOP, 0] = ex
            sp[POINT_SELF_TOP, 1] = ey
            sp[POINT_SELF_BOTTOM, 0] = ex
            sp[POINT_SELF_BOTTOM, 1] = ey
        sp[POINT_RIGHT, 0] = pts[i_r, 0]
        sp[POINT_RIGHT, 1] = pts[i_r, 1]
        sp[POINT_LEFT, 0] = pts[i_l, 0]
        sp[POINT_LEFT, 1] = pts[i_l, 1]
        sp[POINT_TOP, 0] = pts[i_t, 0]
        sp[POINT_TOP, 1] = pts[i_t, 1]
        sp[POINT_BOTTOM, 0] = pts[i_b, 0]
        sp[POINT_BOTTOM, 1] = pts[i_b, 1]
    else:
        for p in range(N_POINTS):
            if p != POINT_START and p != POINT_END:
                sp[p, 0] = ex
                sp[p, 1] = ey


@njit(nogil=True, parallel=True, cache=True)
def render_rows(stats, points, status, row0, row1, max_iter, max_dist, shape, bound,
                main_ctl, seg_mult, fuzz, obst_shapes, obst_bounds, obst_cx, obst_cy, obst_effects,
                obst_fire_inside, seed_base):
    """
    Simulate rows [row0, row1). For each pixel that actually fires a ray, fill
    in stats[j, i, :] and points[j, i, :, :] and set status[j, i] = 1. When
    fuzzing is enabled, each pixel's ray gets its own reproducible seed.
    """
    fuzzing = fuzz[0] > 0.5
    n = (row1 - row0) * SIZE
    for idx in prange(n):
        j = row0 + idx // SIZE
        i = idx % SIZE
        x = (i + 0.5) * PIXEL - 1.0
        y = 1.0 - (j + 0.5) * PIXEL
        if not emits_ray(x, y, shape, bound, main_ctl, obst_shapes, obst_bounds, obst_cx,
                         obst_cy, obst_fire_inside):
            continue
        if fuzzing:
            np.random.seed(seed_base + i * 1000003 + j * 97)
        pts = np.empty((max_iter + 2, 2))
        kinds = np.empty(max_iter + 2)
        raw = np.empty(N_RAW)
        out = np.empty(N_STATS)
        sp = np.empty((N_POINTS, 2))
        term, npts = trace_path(x, y, pts, kinds, max_iter, max_dist, shape, bound,
                                main_ctl, seg_mult, fuzz, obst_shapes, obst_bounds, obst_cx,
                                obst_cy, obst_effects, raw)
        compute_stats(pts, npts, raw, max_iter, seg_mult, out)
        compute_points(pts, npts, raw, kinds, sp)
        for s in range(N_STATS):
            stats[j, i, s] = out[s]
        for p in range(N_POINTS):
            points[j, i, p, 0] = sp[p, 0]
            points[j, i, p, 1] = sp[p, 1]
        status[j, i] = 1


@njit(cache=True)
def scatter_one_buddha_point(counts, x, y, w, h):
    """Scatters one point's weight of 1 into `counts` at its own pixel -
    cheap, no blur kernel applied here at all. Blur (if any) is applied
    once, afterward, to the whole accumulated counts array - see
    blur_counts_image - rather than per point, which is both mathematically
    identical (convolution distributes over sums) and vastly cheaper."""
    col = int(floor((x + 1.0) * (w / 2.0)))
    row = int(floor((1.0 - y) * (h / 2.0)))
    if 0 <= row < h and 0 <= col < w:
        counts[row, col] += 1


@njit(cache=True)
def scatter_multi_buddha(row0, row1, max_iter, max_dist, shape, bound, main_ctl, seg_mult,
                         fuzz, obst_shapes, obst_bounds, obst_cx, obst_cy, obst_effects,
                         obst_fire_inside, seed_base, want_all_self, want_all_mid,
                         want_prog_bounce, want_prog_mid, counts):
    """
    For rows [row0, row1), re-trace every ray that actually fires (using the
    exact same seed formula as render_rows, so this reproduces the identical
    rays already shown) and, for whichever of the four "multi-point" Buddha
    options are requested, scatter every one of that option's points into
    `counts` (in place) - cheap, no blur kernel applied here; blur (if any)
    is applied once, afterward, to the whole accumulated counts array.
    Deliberately single-threaded (no prange): a scattered point can land on
    any pixel, not just ones in this row range, so a parallel version would
    risk concurrent writes to the same pixel.
    """
    fuzzing = fuzz[0] > 0.5
    h, w = counts.shape
    pts = np.empty((max_iter + 2, 2))
    kinds = np.empty(max_iter + 2)
    raw = np.empty(N_RAW)
    for j in range(row0, row1):
        for i in range(w):
            x = (i + 0.5) * PIXEL - 1.0
            y = 1.0 - (j + 0.5) * PIXEL
            if not emits_ray(x, y, shape, bound, main_ctl, obst_shapes, obst_bounds,
                             obst_cx, obst_cy, obst_fire_inside):
                continue
            if fuzzing:
                np.random.seed(seed_base + i * 1000003 + j * 97)
            _term, npts = trace_path(x, y, pts, kinds, max_iter, max_dist, shape, bound,
                                     main_ctl, seg_mult, fuzz, obst_shapes, obst_bounds,
                                     obst_cx, obst_cy, obst_effects, raw)
            nb = int(raw[RAW_BOUNCES])

            if want_all_self:
                for k in range(1, nb + 1):
                    if kinds[k] == KIND_SELF:
                        scatter_one_buddha_point(counts, pts[k, 0], pts[k, 1], w, h)

            if want_all_mid:
                for k in range(npts - 1):
                    mx = 0.5 * (pts[k, 0] + pts[k + 1, 0])
                    my = 0.5 * (pts[k, 1] + pts[k + 1, 1])
                    scatter_one_buddha_point(counts, mx, my, w, h)

            if want_prog_bounce and nb >= 2:
                sx = pts[1, 0] + pts[2, 0]
                sy = pts[1, 1] + pts[2, 1]
                scatter_one_buddha_point(counts, sx / 2.0, sy / 2.0, w, h)
                for c in range(3, nb + 1):
                    sx += pts[c, 0]
                    sy += pts[c, 1]
                    scatter_one_buddha_point(counts, sx / c, sy / c, w, h)

            if want_prog_mid and npts - 1 >= 2:
                sx = 0.5 * (pts[0, 0] + pts[1, 0]) + 0.5 * (pts[1, 0] + pts[2, 0])
                sy = 0.5 * (pts[0, 1] + pts[1, 1]) + 0.5 * (pts[1, 1] + pts[2, 1])
                scatter_one_buddha_point(counts, sx / 2.0, sy / 2.0, w, h)
                cnt = 2
                for k in range(2, npts - 1):
                    sx += 0.5 * (pts[k, 0] + pts[k + 1, 0])
                    sy += 0.5 * (pts[k, 1] + pts[k + 1, 1])
                    cnt += 1
                    scatter_one_buddha_point(counts, sx / cnt, sy / cnt, w, h)


@njit(nogil=True, parallel=True, cache=True)
def render_rows_wh(stats, points, status, row0, row1, out_w, out_h, max_iter, max_dist,
                   shape, bound, main_ctl, seg_mult, fuzz, obst_shapes, obst_bounds, obst_cx,
                   obst_cy, obst_effects, obst_fire_inside, seed_base):
    """
    Like render_rows, but for an arbitrary out_w x out_h output grid (used for
    video frames) instead of the fixed SIZE x SIZE main image. The grid still
    covers the same [-1, 1] world extent on both axes, so a non-square output
    stretches the shape to fit; the "confined to less than a pixel" physics
    stays tied to the main image's fixed PIXEL size, not to out_w/out_h, so a
    ray behaves identically regardless of the resolution it's sampled at.
    """
    fuzzing = fuzz[0] > 0.5
    step_x = 2.0 / out_w
    step_y = 2.0 / out_h
    n = (row1 - row0) * out_w
    for idx in prange(n):
        j = row0 + idx // out_w
        i = idx % out_w
        x = (i + 0.5) * step_x - 1.0
        y = 1.0 - (j + 0.5) * step_y
        if not emits_ray(x, y, shape, bound, main_ctl, obst_shapes, obst_bounds, obst_cx,
                         obst_cy, obst_fire_inside):
            continue
        if fuzzing:
            np.random.seed(seed_base + i * 1000003 + j * 97)
        pts = np.empty((max_iter + 2, 2))
        kinds = np.empty(max_iter + 2)
        raw = np.empty(N_RAW)
        out = np.empty(N_STATS)
        sp = np.empty((N_POINTS, 2))
        term, npts = trace_path(x, y, pts, kinds, max_iter, max_dist, shape, bound,
                                main_ctl, seg_mult, fuzz, obst_shapes, obst_bounds, obst_cx,
                                obst_cy, obst_effects, raw)
        compute_stats(pts, npts, raw, max_iter, seg_mult, out)
        compute_points(pts, npts, raw, kinds, sp)
        for s in range(N_STATS):
            stats[j, i, s] = out[s]
        for p in range(N_POINTS):
            points[j, i, p, 0] = sp[p, 0]
            points[j, i, p, 1] = sp[p, 1]
        status[j, i] = 1


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
        self.baseline = DEFAULT_SNAPSHOT  # what "Revert" restores
        self.max_iter = DEFAULT_ITER      # values used by the current image
        self.max_dist = effective_dist(DEFAULT_DIST)
        self.gen_seg_mult = DEFAULT_SEGMENT_MULT
        self.gen_shape = DEFAULT_SHAPE
        self.gen_bound = ellipse_bound(DEFAULT_WIDTH, DEFAULT_HEIGHT,
                                       DEFAULT_ELL_ROTATION)
        self.gen_fuzz = fuzz_array(DEFAULT_FUZZ)
        self.gen_main_ctl = MAIN_CTL_DEFAULT
        self.gen_seed_base = DEFAULT_FUZZ.seed
        self.gen_buddha_blur_radius = DEFAULT_BUDDHA_BLUR_RADIUS
        self.gen_buddha_blur_type = DEFAULT_BUDDHA_BLUR_TYPE
        self.last_iter = DEFAULT_ITER     # most recent valid entry values
        self.last_dist = DEFAULT_DIST
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
        self.render_thread = None
        self.render_cancel = None

        # ---- obstacles ---- (scene furniture: not part of Settings/keyframes)
        self.obstacles = []                # list[Obstacle], in list order
        self._obst_counter = 0
        self.selected_obstacle = None
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_bounds = np.empty((0, OBST_BOUND_LEN), dtype=np.float64)
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=np.bool_)
        self.gen_obst_shapes = empty_i64
        self.gen_obst_bounds = empty_bounds
        self.gen_obst_cx = empty_f64
        self.gen_obst_cy = empty_f64
        self.gen_obst_effects = np.empty((0, 2), dtype=np.int64)
        self.gen_obst_fire_inside = empty_bool
        self.gen_obstacles_sig = ()
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
        self.ring = None
        self.set_ring(self.gen_shape, self.gen_bound)

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
        self.seed_var.trace_add("write", self.on_seed_var_changed)

        self.draw()
        self.update_status()
        self.update_buttons()
        root.update()                              # show the window before compiling
        root.after(20, self.warm_up)

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
        self.build_shape_controls(sim_inner)
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
                  text="A ray also stops when it is confined to less than a pixel, "
                       "or when it escapes (has nothing left to hit). "
                       "One unit is half the image width; a segment that would pass "
                       "the distance limit is cut off where it is reached. A "
                       "distance limit of 0 means no distance limit. Large limits "
                       "(or fuzzing) can take a very long time."
                  ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

    def build_main_ray_controls(self, sh):
        """The bounding shape's own ray behavior, at the bottom of its frame:
        whether it fires rays inside it, and what happens to a ray crossing
        its edge in each direction - the same controls an obstacle has."""
        box = ttk.Frame(sh)
        box.grid(row=30, column=0, sticky="ew", padx=8, pady=(4, 6))
        ttk.Label(box, foreground="#666", wraplength=RIGHT_W - 90,
                  text=RAY_HELP_TEXT).grid(row=3, column=0, columnspan=2, sticky="w",
                                           pady=(4, 0))
        self.main_fire_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Fire rays inside", variable=self.main_fire_var,
                        command=self.on_shape_param_changed).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))
        rows = [("Incoming ray effect:", "main_in_effect_var"),
                ("Outgoing ray effect:", "main_out_effect_var")]
        for r, (text, attr) in enumerate(rows, start=1):
            ttk.Label(box, text=text, width=20).grid(row=r, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=OBSTACLE_EFFECT_NAMES[OBSTACLE_EFFECT_REFLECT])
            setattr(self, attr, var)
            combo = ttk.Combobox(box, textvariable=var, values=OBSTACLE_EFFECT_NAMES,
                                 state="readonly", width=12)
            combo.grid(row=r, column=1, sticky="w", padx=(8, 0), pady=1)
            combo.bind("<<ComboboxSelected>>",
                       lambda _e, c=combo: (c.selection_clear(), self.on_shape_param_changed()))

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

    def build_shape_controls(self, parent):
        """Shape dropdown, plus the parameter controls of whichever shape is chosen."""
        sh = ttk.LabelFrame(parent, text="Bounding shape")
        sh.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        sh.columnconfigure(0, weight=1)
        self.build_main_ray_controls(sh)

        top = ttk.Frame(sh)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))
        ttk.Label(top, text="Shape:").pack(side="left")
        self.shape_var = tk.StringVar(value=SHAPE_NAMES[DEFAULT_SHAPE])
        self.shape_combo = ttk.Combobox(top, textvariable=self.shape_var,
                                        values=SHAPE_NAMES, state="readonly",
                                        width=18)
        self.shape_combo.pack(side="left", padx=(8, 0))
        self.shape_combo.bind("<<ComboboxSelected>>", self.on_shape_changed)

        # Ellipse controls (everything that was available for circles/ellipses)
        eg = ttk.Frame(sh)
        eg.grid(row=1, column=0, sticky="ew")
        eg.columnconfigure(1, weight=1)
        self.width_scale = self.add_slider_row(
            eg, 0, "Width", ELLIPSE_MIN, ELLIPSE_MAX, 0.01, DEFAULT_WIDTH, "{:.2f}")
        self.height_scale = self.add_slider_row(
            eg, 1, "Height", ELLIPSE_MIN, ELLIPSE_MAX, 0.01, DEFAULT_HEIGHT, "{:.2f}")
        self.ell_rot_scale = self.add_angle_entry(
            eg, 2, "Rotation (\u00b0)", DEFAULT_ELL_ROTATION,
            on_change=self.on_shape_param_changed)
        ttk.Button(eg, text="Reset to unit circle", command=self.reset_ellipse).grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=(4, 2))
        self.ellipse_group = eg

        # Regular-polygon controls
        pg = ttk.Frame(sh)
        pg.grid(row=1, column=0, sticky="ew")
        pg.columnconfigure(1, weight=1)
        self.sides_scale = self.add_slider_row(
            pg, 0, "Sides", POLY_MIN_SIDES, POLY_MAX_SIDES, 1, DEFAULT_SIDES, "{:.0f}")
        self.poly_rot_scale = self.add_angle_entry(
            pg, 1, "Rotation (\u00b0)", DEFAULT_POLY_ROTATION,
            on_change=self.on_shape_param_changed)
        self.polygon_group = pg

        # Moss's egg controls
        mg = ttk.Frame(sh)
        mg.grid(row=1, column=0, sticky="ew")
        mg.columnconfigure(1, weight=1)
        self.moss_angle_scale = self.add_slider_row(
            mg, 0, "Angle ACB (\u00b0)", MOSS_ANGLE_MIN, MOSS_ANGLE_MAX, 0.5,
            DEFAULT_MOSS_ANGLE, "{:.1f}")
        self.moss_size_scale = self.add_slider_row(
            mg, 1, "Size", MOSS_SIZE_MIN, MOSS_SIZE_MAX, 0.01,
            DEFAULT_MOSS_SIZE, "{:.2f}")
        self.moss_rot_scale = self.add_angle_entry(
            mg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_shape_param_changed)
        self.moss_group = mg

        # Circular sector controls
        secg = ttk.Frame(sh)
        secg.grid(row=1, column=0, sticky="ew")
        secg.columnconfigure(1, weight=1)
        self.sector_radius_scale = self.add_slider_row(
            secg, 0, "Radius", ARC_RADIUS_MIN, ARC_RADIUS_MAX, 0.01,
            DEFAULT_ARC_RADIUS, "{:.2f}")
        self.sector_span_scale = self.add_slider_row(
            secg, 1, "Span (\u00b0)", ARC_SPAN_MIN, SECTOR_SPAN_MAX, 0.5,
            DEFAULT_SECTOR_SPAN, "{:.1f}")
        self.sector_rot_scale = self.add_angle_entry(
            secg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_shape_param_changed)
        self.sector_group = secg

        # Circular segment controls
        segg = ttk.Frame(sh)
        segg.grid(row=1, column=0, sticky="ew")
        segg.columnconfigure(1, weight=1)
        self.segment_radius_scale = self.add_slider_row(
            segg, 0, "Radius", ARC_RADIUS_MIN, ARC_RADIUS_MAX, 0.01,
            DEFAULT_ARC_RADIUS, "{:.2f}")
        self.segment_span_scale = self.add_slider_row(
            segg, 1, "Span (\u00b0)", ARC_SPAN_MIN, ARC_SPAN_MAX, 0.5,
            DEFAULT_SEGMENT_SPAN, "{:.1f}")
        self.segment_rot_scale = self.add_angle_entry(
            segg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_shape_param_changed)
        self.segment_group = segg

        # Two-circle lens controls
        lg = ttk.Frame(sh)
        lg.grid(row=1, column=0, sticky="ew")
        lg.columnconfigure((0, 1), weight=1)
        self.lens_r1_scale, self.lens_r2_scale = self.add_slider_pair(
            lg, 0,
            ("Radius 1", LENS_RADIUS_MIN, LENS_RADIUS_MAX, 0.01, DEFAULT_LENS_RADIUS, "{:.2f}"),
            ("Radius 2", LENS_RADIUS_MIN, LENS_RADIUS_MAX, 0.01, DEFAULT_LENS_RADIUS, "{:.2f}"))
        self.lens_sep_scale = self.add_slider_row(
            lg, 1, "Separation", LENS_SEP_MIN, LENS_SEP_MAX, 0.01, DEFAULT_LENS_SEP, "{:.2f}")
        self.lens_rot_scale = self.add_angle_entry(
            lg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_shape_param_changed)
        self.lens_group = lg

        # Reuleaux polygon controls (sides kept odd: resolution 2, starting at 3)
        rg = ttk.Frame(sh)
        rg.grid(row=1, column=0, sticky="ew")
        rg.columnconfigure(1, weight=1)
        self.reuleaux_sides_scale = self.add_slider_row(
            rg, 0, "Sides (odd)", REULEAUX_MIN_SIDES, REULEAUX_MAX_SIDES, 2,
            DEFAULT_REULEAUX_SIDES, "{:.0f}")
        self.reuleaux_rot_scale = self.add_angle_entry(
            rg, 1, "Rotation (\u00b0)", 0.0, on_change=self.on_shape_param_changed)
        self.reuleaux_group = rg

        # Capsule (stadium) controls
        cg = ttk.Frame(sh)
        cg.grid(row=1, column=0, sticky="ew")
        cg.columnconfigure(1, weight=1)
        self.capsule_length_scale = self.add_slider_row(
            cg, 0, "Length", CAPSULE_LENGTH_MIN, CAPSULE_LENGTH_MAX, 0.01,
            DEFAULT_CAPSULE_LENGTH, "{:.2f}")
        self.capsule_radius_scale = self.add_slider_row(
            cg, 1, "Radius", CAPSULE_RADIUS_MIN, CAPSULE_RADIUS_MAX, 0.01,
            DEFAULT_CAPSULE_RADIUS, "{:.2f}")
        self.capsule_rot_scale = self.add_angle_entry(
            cg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_shape_param_changed)
        self.capsule_group = cg

        self.shape_hint_var = tk.StringVar()
        ttk.Label(sh, textvariable=self.shape_hint_var, foreground="#666",
                  wraplength=RIGHT_W - 70).grid(
            row=2, column=0, sticky="w", padx=8, pady=(4, 6))
        self.show_shape_controls()

    def build_fuzzing_controls(self, parent):
        """Angle controls for ray emission, bounces off the boundary, and
        bounces off the ray itself (laid out in a 2x2 grid). Everything except
        the Enable checkbox is greyed out while fuzzing is off."""
        fz = ttk.LabelFrame(parent, text="Fuzzing")
        fz.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 6))
        fz.columnconfigure((0, 1), weight=1, uniform="fuzzcols")

        self.fuzz_enabled_var = tk.BooleanVar(value=DEFAULT_FUZZ.enabled)
        ttk.Checkbutton(fz, text="Enable fuzzing", variable=self.fuzz_enabled_var,
                        command=self.on_fuzz_changed).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 4))
        ttk.Label(fz, foreground="#666", wraplength=RIGHT_W - 70,
                  text="Perturbs the ray's angles. Emission sets the ray's starting "
                       "direction directly (45\u00b0 is the unfuzzed up-and-right "
                       "direction), plus jitter. Each bounce group perturbs the "
                       "deflection angle at that kind of bounce: Offset always "
                       "added, Jitter a fresh random amount within \u00b1 the range "
                       "each time, Multiplier scales the angle before those are "
                       "added (\u00d71 = unchanged)."
                  ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        emit_group = ttk.LabelFrame(fz, text="Ray emission")
        emit_group.columnconfigure(1, weight=1)
        self.emit_angle_scale = self.add_angle_entry(
            emit_group, 0, "Angle (\u00b0)", DEFAULT_EMIT_ANGLE, on_change=self.on_fuzz_changed)
        self.emit_jitter_scale = self.add_angle_entry(
            emit_group, 1, "Jitter (\u00b1\u00b0)", 0.0, on_change=self.on_fuzz_changed,
            lo=FUZZ_JITTER_MIN, hi=FUZZ_JITTER_MAX, fmt="{:.1f}")
        emit_group.grid(row=2, column=0, sticky="nsew", padx=(6, 3), pady=(0, 6))
        self.fuzz_ttk_widgets += [self.emit_angle_scale.widget, self.emit_jitter_scale.widget]

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
            p1combo = ttk.Combobox(qf, textvariable=p1var, values=POINT_LABELS,
                                   state="readonly", width=14)
            p1combo.grid(row=1, column=1, sticky="ew", padx=(4, 6), pady=2)
            p1combo.bind("<<ComboboxSelected>>", lambda _e, c=c: self.on_point_selected(c))

            ttk.Label(qf, text="Point 2:").grid(row=1, column=2, sticky="w", pady=2)
            p2var = tk.StringVar(value=POINT_LABELS[view.point2_choice[c]])
            p2combo = ttk.Combobox(qf, textvariable=p2var, values=POINT_LABELS,
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
                ttk.Checkbutton(
                    frame, text=label, variable=var,
                    command=lambda k=key: self.on_decoration_changed(k)).grid(
                    row=r, column=0, sticky="w", padx=10, pady=2)
                self.deco[key] = var
            ttk.Label(frame, text=note, foreground="#666",
                      wraplength=RIGHT_W - 70).grid(
                row=len(items), column=0, sticky="w", padx=10, pady=(2, 6))

    def build_obstacles_controls(self, parent):
        """Obstacle list (scrollable, with New/Dupe/Del stacked to its left)
        plus a shared editor for whichever obstacle is selected: the same
        shape controls as the main bounding shape (rotation full-width above
        the others, which are paired up to save space), a polar position (r,
        theta) with an x/y offset, incoming and outgoing ray-effect dropdowns,
        and a "fire rays inside" checkbox. Every one of an obstacle's settings can be animated
        between keyframes, just like the main Simulation/Fuzzing/Coloration
        settings."""
        self._obst_editor_loading = False
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        top_row = ttk.Frame(parent)
        top_row.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        top_row.columnconfigure(1, weight=1)
        top_row.rowconfigure(0, weight=1)

        btn_col = ttk.Frame(top_row)
        btn_col.grid(row=0, column=0, sticky="n", padx=(0, 6))
        ttk.Button(btn_col, text="New", width=6, command=self.new_obstacle).pack(
            fill="x", pady=(0, 3))
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
        self.obs_shape_var = tk.StringVar(value=SHAPE_NAMES[SHAPE_ELLIPSE])
        self.obs_shape_combo = ttk.Combobox(top, textvariable=self.obs_shape_var,
                                            values=SHAPE_NAMES, state="readonly",
                                            width=16)
        self.obs_shape_combo.pack(side="left", padx=(8, 0))
        self.obs_shape_combo.bind("<<ComboboxSelected>>", self.on_obstacle_shape_changed)

        oeg = ttk.Frame(editor)
        oeg.grid(row=1, column=0, sticky="ew")
        oeg.columnconfigure((0, 1), weight=1)
        self.obs_ell_rot_scale = self.add_angle_entry(
            oeg, 0, "Rotation (\u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_width_scale, self.obs_height_scale = self.add_slider_pair(
            oeg, 1,
            ("Width", ELLIPSE_MIN, ELLIPSE_MAX, 0.01, 1.0, "{:.2f}"),
            ("Height", ELLIPSE_MIN, ELLIPSE_MAX, 0.01, 1.0, "{:.2f}"),
            on_change=self.on_obstacle_param_changed)
        self.obs_ellipse_group = oeg

        opg = ttk.Frame(editor)
        opg.grid(row=1, column=0, sticky="ew")
        opg.columnconfigure((0, 1), weight=1)
        self.obs_poly_rot_scale = self.add_angle_entry(
            opg, 0, "Rotation (\u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_sides_scale, self.obs_radius_scale = self.add_slider_pair(
            opg, 1,
            ("Sides", POLY_MIN_SIDES, POLY_MAX_SIDES, 1, DEFAULT_SIDES, "{:.0f}"),
            ("Radius", OBST_RADIUS_MIN, OBST_RADIUS_MAX, 0.01, 0.5, "{:.2f}"),
            on_change=self.on_obstacle_param_changed)
        self.obs_polygon_group = opg

        # Moss's egg
        omg = ttk.Frame(editor)
        omg.grid(row=1, column=0, sticky="ew")
        omg.columnconfigure(1, weight=1)
        self.obs_moss_angle_scale = self.add_slider_row(
            omg, 0, "Angle ACB (\u00b0)", MOSS_ANGLE_MIN, MOSS_ANGLE_MAX, 0.5,
            DEFAULT_MOSS_ANGLE, "{:.1f}", on_change=self.on_obstacle_param_changed)
        self.obs_moss_size_scale = self.add_slider_row(
            omg, 1, "Size", MOSS_SIZE_MIN, MOSS_SIZE_MAX, 0.01,
            DEFAULT_MOSS_SIZE, "{:.2f}", on_change=self.on_obstacle_param_changed)
        self.obs_moss_rot_scale = self.add_angle_entry(
            omg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_moss_group = omg

        # Circular sector
        osecg = ttk.Frame(editor)
        osecg.grid(row=1, column=0, sticky="ew")
        osecg.columnconfigure(1, weight=1)
        self.obs_sector_radius_scale = self.add_slider_row(
            osecg, 0, "Radius", ARC_RADIUS_MIN, ARC_RADIUS_MAX, 0.01,
            DEFAULT_ARC_RADIUS, "{:.2f}", on_change=self.on_obstacle_param_changed)
        self.obs_sector_span_scale = self.add_slider_row(
            osecg, 1, "Span (\u00b0)", ARC_SPAN_MIN, SECTOR_SPAN_MAX, 0.5,
            DEFAULT_SECTOR_SPAN, "{:.1f}", on_change=self.on_obstacle_param_changed)
        self.obs_sector_rot_scale = self.add_angle_entry(
            osecg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_sector_group = osecg

        # Circular segment
        osegg = ttk.Frame(editor)
        osegg.grid(row=1, column=0, sticky="ew")
        osegg.columnconfigure(1, weight=1)
        self.obs_segment_radius_scale = self.add_slider_row(
            osegg, 0, "Radius", ARC_RADIUS_MIN, ARC_RADIUS_MAX, 0.01,
            DEFAULT_ARC_RADIUS, "{:.2f}", on_change=self.on_obstacle_param_changed)
        self.obs_segment_span_scale = self.add_slider_row(
            osegg, 1, "Span (\u00b0)", ARC_SPAN_MIN, ARC_SPAN_MAX, 0.5,
            DEFAULT_SEGMENT_SPAN, "{:.1f}", on_change=self.on_obstacle_param_changed)
        self.obs_segment_rot_scale = self.add_angle_entry(
            osegg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_segment_group = osegg

        # Two-circle lens
        olg = ttk.Frame(editor)
        olg.grid(row=1, column=0, sticky="ew")
        olg.columnconfigure((0, 1), weight=1)
        self.obs_lens_r1_scale, self.obs_lens_r2_scale = self.add_slider_pair(
            olg, 0,
            ("Radius 1", LENS_RADIUS_MIN, LENS_RADIUS_MAX, 0.01, DEFAULT_LENS_RADIUS, "{:.2f}"),
            ("Radius 2", LENS_RADIUS_MIN, LENS_RADIUS_MAX, 0.01, DEFAULT_LENS_RADIUS, "{:.2f}"),
            on_change=self.on_obstacle_param_changed)
        self.obs_lens_sep_scale = self.add_slider_row(
            olg, 1, "Separation", LENS_SEP_MIN, LENS_SEP_MAX, 0.01, DEFAULT_LENS_SEP,
            "{:.2f}", on_change=self.on_obstacle_param_changed)
        self.obs_lens_rot_scale = self.add_angle_entry(
            olg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_lens_group = olg

        # Reuleaux polygon
        org = ttk.Frame(editor)
        org.grid(row=1, column=0, sticky="ew")
        org.columnconfigure((0, 1), weight=1)
        self.obs_reuleaux_rot_scale = self.add_angle_entry(
            org, 0, "Rotation (\u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_reuleaux_sides_scale, self.obs_reuleaux_radius_scale = self.add_slider_pair(
            org, 1,
            ("Sides (odd)", REULEAUX_MIN_SIDES, REULEAUX_MAX_SIDES, 2,
             DEFAULT_REULEAUX_SIDES, "{:.0f}"),
            ("Radius", OBST_RADIUS_MIN, OBST_RADIUS_MAX, 0.01, 0.5, "{:.2f}"),
            on_change=self.on_obstacle_param_changed)
        self.obs_reuleaux_group = org

        # Capsule (stadium)
        ocg = ttk.Frame(editor)
        ocg.grid(row=1, column=0, sticky="ew")
        ocg.columnconfigure(1, weight=1)
        self.obs_capsule_length_scale = self.add_slider_row(
            ocg, 0, "Length", CAPSULE_LENGTH_MIN, CAPSULE_LENGTH_MAX, 0.01,
            DEFAULT_CAPSULE_LENGTH, "{:.2f}", on_change=self.on_obstacle_param_changed)
        self.obs_capsule_radius_scale = self.add_slider_row(
            ocg, 1, "Radius", CAPSULE_RADIUS_MIN, CAPSULE_RADIUS_MAX, 0.01,
            DEFAULT_CAPSULE_RADIUS, "{:.2f}", on_change=self.on_obstacle_param_changed)
        self.obs_capsule_rot_scale = self.add_angle_entry(
            ocg, 2, "Rotation (\u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_capsule_group = ocg

        pos_group = ttk.Frame(editor)
        pos_group.grid(row=2, column=0, sticky="ew")
        pos_group.columnconfigure((0, 1), weight=1)
        self.obs_r_scale = self.add_slider_row(
            pos_group, 0, "Distance (r)", 0.0, OBST_POS_MAX, 0.01, 0.0, "{:.2f}",
            on_change=self.on_obstacle_param_changed)
        self.obs_theta_scale = self.add_angle_entry(
            pos_group, 1, "Angle (\u03b8, \u00b0)", 0.0, on_change=self.on_obstacle_param_changed)
        self.obs_cx_scale, self.obs_cy_scale = self.add_slider_pair(
            pos_group, 2,
            ("Offset X", OBST_POS_MIN, OBST_POS_MAX, 0.01, 0.0, "{:+.2f}"),
            ("Offset Y", OBST_POS_MIN, OBST_POS_MAX, 0.01, 0.0, "{:+.2f}"),
            on_change=self.on_obstacle_param_changed)

        in_row = ttk.Frame(editor)
        in_row.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 2))
        ttk.Label(in_row, text="Incoming ray effect:", width=20).pack(side="left")
        self.obs_in_effect_var = tk.StringVar(
            value=OBSTACLE_EFFECT_NAMES[OBSTACLE_EFFECT_REFLECT])
        self.obs_in_effect_combo = ttk.Combobox(in_row, textvariable=self.obs_in_effect_var,
                                                values=OBSTACLE_EFFECT_NAMES,
                                                state="readonly", width=12)
        self.obs_in_effect_combo.pack(side="left", padx=(8, 0))
        self.obs_in_effect_combo.bind("<<ComboboxSelected>>", self.on_obstacle_effect_changed)

        out_row = ttk.Frame(editor)
        out_row.grid(row=4, column=0, sticky="ew", padx=8, pady=(2, 2))
        ttk.Label(out_row, text="Outgoing ray effect:", width=20).pack(side="left")
        self.obs_out_effect_var = tk.StringVar(
            value=OBSTACLE_EFFECT_NAMES[OBSTACLE_EFFECT_PASS])
        self.obs_out_effect_combo = ttk.Combobox(out_row, textvariable=self.obs_out_effect_var,
                                                 values=OBSTACLE_EFFECT_NAMES,
                                                 state="readonly", width=12)
        self.obs_out_effect_combo.pack(side="left", padx=(8, 0))
        self.obs_out_effect_combo.bind("<<ComboboxSelected>>", self.on_obstacle_effect_changed)

        self.obs_fire_inside_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(editor, text="Fire rays inside", variable=self.obs_fire_inside_var,
                        command=self.on_obstacle_fire_inside_changed).grid(
            row=5, column=0, sticky="w", padx=8, pady=2)

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

    def show_obstacle_shape_controls(self):
        shape = SHAPE_NAMES.index(self.obs_shape_var.get())
        groups = {
            SHAPE_ELLIPSE: self.obs_ellipse_group, SHAPE_POLYGON: self.obs_polygon_group,
            SHAPE_MOSS_EGG: self.obs_moss_group, SHAPE_SECTOR: self.obs_sector_group,
            SHAPE_SEGMENT: self.obs_segment_group, SHAPE_LENS: self.obs_lens_group,
            SHAPE_REULEAUX: self.obs_reuleaux_group, SHAPE_CAPSULE: self.obs_capsule_group,
        }
        for s, g in groups.items():
            if s == shape:
                g.grid()
            else:
                g.grid_remove()

    def refresh_obstacle_editor(self):
        """Populate the editor controls from the selected obstacle (or leave
        them as-is, disabled-by-convention, if none is selected). Every
        field shared with the bounding shape is set the same way
        apply_snapshot_to_controls sets its own widgets - table-driven off
        SHAPE_PARAM_INFO, rather than one hand-written line per field."""
        o = self.selected_obstacle
        if o is None:
            return
        self._obst_editor_loading = True
        try:
            self.obs_shape_var.set(SHAPE_NAMES[o.shape])
            for p in SHAPE_PARAM_INFO:
                getattr(self, "obs_" + p.widget).set(getattr(o, p.field))
            self.obs_radius_scale.set(o.poly_radius)
            self.obs_reuleaux_radius_scale.set(o.reuleaux_radius)
            self.obs_r_scale.set(o.r)
            self.obs_theta_scale.set(o.theta)
            self.obs_cx_scale.set(o.cx)
            self.obs_cy_scale.set(o.cy)
            self.obs_in_effect_var.set(OBSTACLE_EFFECT_NAMES[o.in_effect])
            self.obs_out_effect_var.set(OBSTACLE_EFFECT_NAMES[o.out_effect])
            self.obs_fire_inside_var.set(o.fire_inside)
            self.show_obstacle_shape_controls()
        finally:
            self._obst_editor_loading = False

    def new_obstacle(self):
        o = Obstacle(self._obst_counter)
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

            lbl = tk.Label(row, text=f"#{idx + 1}  {describe_obstacle(o)}",
                          anchor="w", cursor="hand2", font=self.small_font)
            lbl.grid(row=0, column=1, sticky="ew")
            lbl.bind("<Button-1>", lambda _e, o=o: self.select_obstacle(o))

    def on_obstacle_shape_changed(self, _event=None):
        if self._obst_editor_loading:
            return
        o = self.selected_obstacle
        if o is None:
            return
        o.shape = SHAPE_NAMES.index(self.obs_shape_var.get())
        self.obs_shape_combo.selection_clear()
        self.show_obstacle_shape_controls()
        self.rebuild_obstacle_list()
        self.on_shape_param_changed()

    def on_obstacle_param_changed(self):
        if self._obst_editor_loading:
            return
        o = self.selected_obstacle
        if o is None:
            return
        for p in SHAPE_PARAM_INFO:
            raw = getattr(self, "obs_" + p.widget).get()
            setattr(o, p.field,
                   int(round(float(raw))) if p.fmt == "int" else round(float(raw), 2))
        o.poly_radius = round(float(self.obs_radius_scale.get()), 2)
        o.reuleaux_radius = round(float(self.obs_reuleaux_radius_scale.get()), 2)
        o.r = round(float(self.obs_r_scale.get()), 2)
        o.theta = round(float(self.obs_theta_scale.get()), 2)
        o.cx = round(float(self.obs_cx_scale.get()), 2)
        o.cy = round(float(self.obs_cy_scale.get()), 2)
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
        new_o = Obstacle(self._obst_counter)
        self._obst_counter += 1
        for field in Obstacle.__slots__:
            if field != "id":
                setattr(new_o, field, getattr(o, field))
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

        sim_info_row = ttk.Frame(designer)
        sim_info_row.grid(row=2, column=0, sticky="w", padx=8, pady=(2, 6))
        ttk.Label(sim_info_row, text="Simulation info location:").pack(side="left", padx=(0, 6))
        self.sim_info_location_var = tk.StringVar(value=SIM_INFO_LOCATIONS[0])
        ttk.Combobox(sim_info_row, textvariable=self.sim_info_location_var,
                    values=SIM_INFO_LOCATIONS, state="readonly", width=13).pack(side="left")

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
        self.save_jxl_label = ttk.Label(fmt_row, text="JPEG XL mode:")
        self.save_jxl_label.pack(side="left", padx=(0, 4))
        self.save_jxl_mode_var = tk.StringVar(value=JXL_MODES[0])
        self.save_jxl_combo = ttk.Combobox(
            fmt_row, textvariable=self.save_jxl_mode_var, values=JXL_MODES,
            state="readonly", width=8)
        self.save_jxl_combo.pack(side="left")
        self.save_jxl_combo.bind("<<ComboboxSelected>>", self.on_save_jxl_mode_changed)
        self.save_jxl_label.pack_forget()
        self.save_jxl_combo.pack_forget()

        # JXL + Lossy only: a quality slider, on the same line as the rest
        self.save_jxl_quality_frame = ttk.Frame(fmt_row)
        ttk.Label(self.save_jxl_quality_frame, text="Quality:").pack(
            side="left", padx=(8, 4))
        self.save_jxl_quality_scale = tk.Scale(
            self.save_jxl_quality_frame, from_=5, to=100, resolution=5,
            orient="horizontal", showvalue=False, highlightthickness=0,
            width=14, sliderlength=18, length=80)
        self.save_jxl_quality_scale.set(90)
        self.save_jxl_quality_scale.pack(side="left")
        self.save_jxl_quality_value = ttk.Label(self.save_jxl_quality_frame, text="90",
                                                width=3, anchor="e")
        self.save_jxl_quality_value.pack(side="left", padx=(4, 0))
        self.save_jxl_quality_scale.configure(
            command=lambda v: self.save_jxl_quality_value.configure(text=f"{float(v):.0f}"))
        self.save_jxl_quality_frame.pack(side="left")
        self.save_jxl_quality_frame.pack_forget()

        # "New figure" only: its own resolution (an existing preview already
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
        """Top right: the keyframe list (with thumbnails), and beside it, a
        list of every setting, showing how each relates to the selected
        keyframe, grouped by tab."""
        tl = ttk.LabelFrame(self.root, text="Timeline", width=TIMELINE_W, height=SIZE)
        tl.grid(row=0, column=2, sticky="nsew", padx=(PAD // 2, PAD), pady=(PAD, PAD // 2))
        tl.grid_propagate(False)
        tl.columnconfigure(0, weight=1)
        tl.columnconfigure(1, weight=1)
        tl.rowconfigure(2, weight=1)

        ttk.Button(tl, text="New keyframe (from current settings)",
                  command=self.new_keyframe).grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))

        ttk.Label(tl, text="Keyframes:", foreground="#666").grid(
            row=1, column=0, sticky="w", padx=8)

        diff_header = ttk.Frame(tl)
        diff_header.grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Label(diff_header, text="Settings:", foreground="#666").pack(side="left")
        self.diff_delete_all_btn = ttk.Button(diff_header, text="Delete all",
                                              command=self.delete_all_diffs)
        self.diff_delete_all_btn.pack(side="right")
        self.diff_save_all_btn = ttk.Button(diff_header, text="Save all",
                                            command=self.save_all_diffs)
        self.diff_save_all_btn.pack(side="right", padx=(4, 4))
        self.diff_load_all_btn = ttk.Button(diff_header, text="Load all",
                                            command=self.load_all_diffs)
        self.diff_load_all_btn.pack(side="right")

        # Keyframe list (left column): thumbnail + label, click either to select
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

        # Settings list (right column): every setting, grouped by tab
        diff_container = ttk.Frame(tl)
        diff_container.grid(row=2, column=1, sticky="nsew", padx=8, pady=(2, 4))
        diff_container.columnconfigure(0, weight=1)
        diff_container.rowconfigure(0, weight=1)
        diff_canvas = tk.Canvas(diff_container, highlightthickness=0)
        dvsb = ttk.Scrollbar(diff_container, orient="vertical", command=diff_canvas.yview)
        diff_canvas.configure(yscrollcommand=dvsb.set)
        diff_canvas.grid(row=0, column=0, sticky="nsew")
        dvsb.grid(row=0, column=1, sticky="ns")
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

        ttk.Label(det, text="Interpolation:").grid(
            row=2, column=0, sticky="w", padx=8, pady=2)
        self.kf_interp_var = tk.StringVar(value=DEFAULT_INTERP)
        self.kf_interp_combo = ttk.Combobox(det, textvariable=self.kf_interp_var,
                                            values=INTERP_METHODS, state="readonly",
                                            width=10)
        self.kf_interp_combo.grid(row=2, column=1, sticky="w", padx=(0, 8), pady=2)
        self.kf_interp_combo.bind("<<ComboboxSelected>>", self.on_kf_interp_changed)

        btnrow = ttk.Frame(det)
        btnrow.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 8))
        self.kf_split_btn = ttk.Button(btnrow, text="Split",
                                       command=self.split_selected_keyframe)
        self.kf_split_btn.pack(side="left", padx=(0, 4))
        self.kf_delete_btn = ttk.Button(btnrow, text="Delete",
                                        command=self.delete_selected_keyframe)
        self.kf_delete_btn.pack(side="left")

        ttk.Label(tl, foreground="#666", wraplength=TIMELINE_W - 30,
                  text="Only float values will be interpolated according to the "
                       "next keyframe's interpolation values. Other values will "
                       "be held until the next keyframe."
                 ).grid(row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))

        self.update_kf_detail_panel()
        self.rebuild_keyframe_list()
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
                                    values=["mp4 (h264)", "GIF", "APNG", "mkv (ffv1)", "JXL"],
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
            values=["Constant quality", "Target bitrate", "Variable bitrate"],
            state="readonly", width=15)
        mode_combo.pack(side="left", padx=(0, 10))
        mode_combo.bind("<<ComboboxSelected>>", self.on_bitrate_mode_changed)

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

        note = "" if ffmpeg is not None else ("\nffmpeg-python was not found - install "
                                              "it with:  pip install ffmpeg-python")
        ttk.Label(vp, foreground="#666", wraplength=TIMELINE_W - 30,
                  text=f"Frames are written as PNGs to exports/{os.path.basename(TEMP_DIR)}/ "
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
        """A resolution field (New figure or video width/height) lost focus
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
        if it is not None:
            self.last_iter = it
        if ds is not None:
            self.last_dist = ds
        self.iter_entry.configure(style="TEntry" if it is not None else "Invalid.TEntry")
        self.dist_entry.configure(style="TEntry" if ds is not None else "Invalid.TEntry")
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
        for var, last, parser in ((self.iter_var, self.last_iter, parse_iter_limit),
                                  (self.dist_var, self.last_dist, parse_dist_limit)):
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

    # ---- bounding shape controls -------------------------------------------- #
    def read_shape_controls(self):
        """Every shape's own parameter values, as a dict keyed by their exact
        Snapshot field names (shape, width, height, ..., capsule_rotation) -
        regardless of which shape is currently selected, so Revert/keyframes
        can restore them all. Returning a dict (for Snapshot(**...)) rather
        than a plain positional tuple means a field added here in the wrong
        place, or left out, fails loudly (TypeError) instead of silently
        landing in the wrong Snapshot slot."""
        vals = {"shape": SHAPE_NAMES.index(self.shape_var.get())}
        for p in SHAPE_PARAM_INFO:
            raw = getattr(self, p.widget).get()
            vals[p.field] = int(round(float(raw))) if p.fmt == "int" else round(float(raw), 2)
        return vals

    def control_shape(self):
        """(shape, params) of the currently selected shape, from the controls."""
        mini = Snapshot(max_iter=0, max_dist=0, seg_mult=1.0, fuzz=DEFAULT_FUZZ,
                        buddha_blur_radius=DEFAULT_BUDDHA_BLUR_RADIUS,
                        buddha_blur_type=DEFAULT_BUDDHA_BLUR_TYPE,
                        **self.read_shape_controls())
        return mini.shape, snapshot_shape_params(mini)

    def show_shape_controls(self):
        """Show only the parameter controls of the selected shape."""
        shape = SHAPE_NAMES.index(self.shape_var.get())
        groups = {
            SHAPE_ELLIPSE: self.ellipse_group, SHAPE_POLYGON: self.polygon_group,
            SHAPE_MOSS_EGG: self.moss_group, SHAPE_SECTOR: self.sector_group,
            SHAPE_SEGMENT: self.segment_group, SHAPE_LENS: self.lens_group,
            SHAPE_REULEAUX: self.reuleaux_group, SHAPE_CAPSULE: self.capsule_group,
        }
        for s, g in groups.items():
            if s == shape:
                g.grid()
            else:
                g.grid_remove()
        self.shape_hint_var.set(SHAPE_HINTS[shape])

    def on_shape_changed(self, _event=None):
        self.shape_combo.selection_clear()
        self.show_shape_controls()
        self.on_shape_param_changed()

    def reset_ellipse(self):
        self.width_scale.set(DEFAULT_WIDTH)
        self.height_scale.set(DEFAULT_HEIGHT)
        self.ell_rot_scale.set(DEFAULT_ELL_ROTATION)
        self.on_shape_param_changed()

    def on_shape_param_changed(self, _value=None):
        self.invalidate_preview()
        self.schedule_outline_refresh()           # live white outline before first Start
        self.update_pending_outline()             # dashed outline afterwards
        self.update_buttons()
        self.refresh_state_text()
        self.refresh_timeline_diffs()

    # ---- fuzzing controls ---------------------------------------------------- #
    def read_fuzz_controls(self):
        """FuzzParams from the current widget values."""
        seed = parse_seed(self.seed_var.get())
        return FuzzParams(
            bool(self.fuzz_enabled_var.get()),
            round(float(self.emit_angle_scale.get()), 2),
            round(float(self.emit_jitter_scale.get()), 2),
            round(float(self.bnd_offset_scale.get()), 2),
            round(float(self.bnd_jitter_scale.get()), 2),
            round(float(self.bnd_mult_scale.get()), 3),
            round(float(self.self_offset_scale.get()), 2),
            round(float(self.self_jitter_scale.get()), 2),
            round(float(self.self_mult_scale.get()), 3),
            self.last_seed if seed is None else seed)

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

    def compute_ring(self, shape, bound, gx=None, gy=None):
        """Boolean mask of the pixels forming the white outline of a shape, on
        the given world-coordinate grid (the main image's by default)."""
        x = self.gx if gx is None else gx
        y = self.gy if gy is None else gy
        if shape == SHAPE_ELLIPSE:
            m11, m12, m22 = bound[1], bound[2], bound[3]
            q = m11 * x * x + 2.0 * m12 * x * y + m22 * y * y
            grad = 2.0 * np.sqrt((m11 * x + m12 * y) ** 2 + (m12 * x + m22 * y) ** 2)
            # first-order distance from each pixel centre to the boundary
            dist = np.abs(q - 1.0) / np.maximum(grad, 1e-9)
            return dist < RING_HALF_WIDTH_PX * PIXEL
        if shape == SHAPE_POLYGON:
            h = bound[1]
            n = int(bound[2])
            f = np.full(x.shape, -1e9)
            for k in range(n):
                np.maximum(f, bound[3 + 2 * k] * x + bound[4 + 2 * k] * y, out=f)
            dist = np.abs(f - h)                  # 0 exactly on the boundary
            return dist < RING_HALF_WIDTH_PX * PIXEL
        # The other six shapes don't have as simple an analytic gradient, so
        # take the boundary between a copy of the shape nudged slightly
        # bigger and one nudged slightly smaller - by a fixed absolute
        # distance (offset_bound), not a multiplicative scale, so the ring
        # comes out a constant width everywhere. (A multiplicative scale's
        # shift is proportional to distance from the scaling centre, which
        # for points close to it - e.g. capsule's straight edges, close to
        # its own middle - could be a tiny fraction of a pixel and invisible.)
        eps = RING_HALF_WIDTH_PX * PIXEL
        bigger = inside_boundary_grid(x, y, shape, offset_bound(shape, bound, eps))
        smaller = inside_boundary_grid(x, y, shape, offset_bound(shape, bound, -eps))
        return bigger & ~smaller

    def set_ring(self, shape, bound):
        self.ring = self.compute_ring(shape, bound)

    def schedule_outline_refresh(self):
        """Coalesce rapid slider events into one outline redraw."""
        if self.outline_job is None:
            self.outline_job = self.root.after_idle(self.refresh_outline)

    def refresh_outline(self):
        self.outline_job = None
        if self.generated_settings is None:       # nothing generated yet: live preview
            shape, params = self.control_shape()
            self.set_ring(shape, make_bound(shape, params))
            self.draw()

    def obstacle_state(self, o):
        """A plain-data snapshot of one obstacle's settings, for the "changed
        since the last generation" comparison."""
        return (o.shape, o.params(), o.r, o.theta, o.cx, o.cy, o.in_effect, o.out_effect,
                o.fire_inside)

    def obstacle_is_pending(self, o):
        """Should `o` get the dashed editing guide? Yes if it does nothing to
        rays (passes them through both ways - regardless of whether it
        changed), is new since the last Start, or its settings differ from
        what was last generated."""
        if o.in_effect == OBSTACLE_EFFECT_PASS and o.out_effect == OBSTACLE_EFFECT_PASS:
            return True
        frozen = self.gen_obstacles_by_id.get(o.id)
        return frozen is None or self.obstacle_state(o) != frozen

    def update_pending_outline(self):
        """Dashed outline of the selected shape (and any pending/disabled
        obstacle) when it differs from what's actually been generated."""
        self.canvas.delete("pending")
        if not self.deco["pending"].get():
            return
        g = self.generated_settings
        if g is None:
            return                                 # nothing generated yet: nothing "pending"
        shape, params = self.control_shape()
        if (shape, params) != (g.shape, g.params):
            self.canvas.create_line(*outline_coords(shape, params),
                                    fill=PENDING_OUTLINE, dash=(5, 4), tags="pending")
        for o in self.obstacles:
            if self.obstacle_is_pending(o):
                self.canvas.create_line(*obstacle_outline_coords(o),
                                        fill=PENDING_OUTLINE, dash=(5, 4), tags="pending")

    # ---- settings, Start and Revert ---------------------------------------- #
    def current_snapshot(self):
        """Every simulation control's value, or None if a limit entry is invalid."""
        it = parse_iter_limit(self.iter_var.get())
        ds = parse_dist_limit(self.dist_var.get())
        if it is None or ds is None:
            return None
        sm = self.seg_mult_scale.get()
        br = self.buddha_blur_radius_scale.get()
        bt = BUDDHA_BLUR_LABELS.index(self.buddha_blur_type_var.get())
        return Snapshot(max_iter=it, max_dist=ds, seg_mult=sm, fuzz=self.read_fuzz_controls(),
                        buddha_blur_radius=br, buddha_blur_type=bt,
                        main_fire=bool(self.main_fire_var.get()),
                        main_in_effect=OBSTACLE_EFFECT_NAMES.index(
                            self.main_in_effect_var.get()),
                        main_out_effect=OBSTACLE_EFFECT_NAMES.index(
                            self.main_out_effect_var.get()),
                        **self.read_shape_controls())

    def is_image_current(self):
        """Does the image currently shown in the image panel actually match the
        current control settings (used to decide what a keyframe's thumbnail
        should show)?"""
        if self.generated_settings is None or self.running or self.rows_done < SIZE:
            return False
        return not self.settings_changed()

    def settings_changed(self):
        """Do the controls now describe a different image than the one generated?"""
        cur = self.current_snapshot()
        return (self.generated_settings is not None and cur is not None
                and effective_settings(cur) != self.generated_settings)

    def update_buttons(self):
        """Start: enabled if the entries are valid and either nothing has been
        generated yet or the settings (or obstacles) differ from those the
        image was started with. Revert: enabled whenever any simulation
        control differs from the baseline (obstacles are scene furniture, not
        part of Revert). Export/Export As: enabled once a figure has actually
        finished generating (started, not currently running, all rows done) -
        exporting before that would just export the blank background, which
        is never useful, so the button explains what to do instead."""
        cur = self.current_snapshot()
        start_ok = (self.ready and cur is not None
                    and (self.generated_settings is None
                         or effective_settings(cur) != self.generated_settings
                         or obstacles_signature(self.obstacles) != self.gen_obstacles_sig))
        revert_ok = cur is None or cur != self.baseline
        self.start_btn.state(["!disabled"] if start_ok else ["disabled"])
        self.revert_btn.state(["!disabled"] if revert_ok else ["disabled"])
        if hasattr(self, "save_btn"):
            has_figure = (self.generated_settings is not None and not self.running
                         and self.rows_done >= SIZE)
            self.save_btn.state(["!disabled"] if has_figure else ["disabled"])
            self.save_as_btn.state(["!disabled"] if has_figure else ["disabled"])
            tip = ("" if has_figure else
                  "Press Start and wait for generation to finish before exporting.")
            self.save_btn_tooltip.update(tip)
            self.save_as_btn_tooltip.update(tip)

    def apply_snapshot_to_controls(self, s):
        """Set every Simulation/Fuzzing control from a Snapshot `s`. Used by
        both Revert and Load."""
        self.iter_var.set(str(s.max_iter))
        self.dist_var.set(str(s.max_dist))
        self.seg_mult_scale.set(s.seg_mult)
        self.shape_var.set(SHAPE_NAMES[s.shape])
        for p in SHAPE_PARAM_INFO:
            getattr(self, p.widget).set(getattr(s, p.field))
        self.show_shape_controls()
        f = s.fuzz
        self.fuzz_enabled_var.set(f.enabled)
        self.emit_angle_scale.set(f.emit_angle)
        self.emit_jitter_scale.set(f.emit_jitter)
        self.bnd_offset_scale.set(f.bnd_offset)
        self.bnd_jitter_scale.set(f.bnd_jitter)
        self.bnd_mult_scale.set(f.bnd_mult)
        self.self_offset_scale.set(f.self_offset)
        self.self_jitter_scale.set(f.self_jitter)
        self.self_mult_scale.set(f.self_mult)
        self.seed_var.set(str(f.seed))
        self.buddha_blur_radius_scale.set(s.buddha_blur_radius)
        self.buddha_blur_type_var.set(BUDDHA_BLUR_LABELS[s.buddha_blur_type])
        self.main_fire_var.set(s.main_fire)
        self.main_in_effect_var.set(OBSTACLE_EFFECT_NAMES[s.main_in_effect])
        self.main_out_effect_var.set(OBSTACLE_EFFECT_NAMES[s.main_out_effect])

    def revert_settings(self):
        """Put every simulation control back to the baseline: its value when the
        program started, or when the last generation was started."""
        self.apply_snapshot_to_controls(self.baseline)
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
                scatter_multi_buddha(
                    r0, r1, self.max_iter, self.max_dist, self.gen_shape, self.gen_bound,
                    self.gen_main_ctl, self.gen_seg_mult, self.gen_fuzz, self.gen_obst_shapes,
                    self.gen_obst_bounds, self.gen_obst_cx, self.gen_obst_cy,
                    self.gen_obst_effects, self.gen_obst_fire_inside, self.gen_seed_base,
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
            radius_px = world_radius_to_pixels(self.gen_buddha_blur_radius, SIZE, SIZE)
            kernel = build_buddha_blur_kernel(radius_px, self.gen_buddha_blur_type)
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
        bound = ellipse_bound(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_ELL_ROTATION)
        fz = fuzz_array(DEFAULT_FUZZ)
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_bounds = np.empty((0, OBST_BOUND_LEN), dtype=np.float64)
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=np.bool_)
        inside_boundary(0.0, 0.0, SHAPE_ELLIPSE, bound)
        empty_effects = np.empty((0, 2), dtype=np.int64)
        emits_ray(0.0, 0.0, SHAPE_ELLIPSE, bound, MAIN_CTL_DEFAULT, empty_i64, empty_bounds,
                 empty_f64, empty_f64, empty_bool)
        trace_path(0.0, 0.0, self.hover_pts, self.hover_kinds, WARM_ITER, big,
                   SHAPE_ELLIPSE, bound, MAIN_CTL_DEFAULT, DEFAULT_SEGMENT_MULT, fz,
                   empty_i64, empty_bounds, empty_f64, empty_f64, empty_effects,
                   self.hover_raw)
        compute_stats(self.hover_pts, 4, self.hover_raw, WARM_ITER,
                      DEFAULT_SEGMENT_MULT, self.hover_vec)
        compute_points(self.hover_pts, 4, self.hover_raw, self.hover_kinds, self.hover_sp)
        render_rows(self.stats, self.points, self.status, 0, 1, WARM_ITER, big,
                    SHAPE_ELLIPSE, bound, MAIN_CTL_DEFAULT, DEFAULT_SEGMENT_MULT, fz,
                    empty_i64, empty_bounds, empty_f64, empty_f64, empty_effects, empty_bool, 0)
        self.stats[:] = 0
        self.points[:] = 0
        self.status[:] = 0
        self.ready = True
        self.update_buttons()
        self.update_status()
        self.root.after(500, self.take_startup_screenshot)   # let the UI settle first

    def take_startup_screenshot(self):
        """At most once per launch (see take_ui_screenshot for the conditions).
        A development aid, so any failure is reported and otherwise ignored."""
        if getattr(self, "_startup_screenshot_taken", False):
            return
        self._startup_screenshot_taken = True
        try:
            take_ui_screenshot(self.root)
        except Exception as exc:                   # noqa: BLE001
            print(f"Startup screenshot failed: {exc}", file=sys.stderr)

    def start(self):
        """(Re)start generation using the current settings."""
        snap = self.current_snapshot()
        if snap is None:
            return
        if self.tick_id is not None:
            self.root.after_cancel(self.tick_id)
            self.tick_id = None

        settings = effective_settings(snap)
        self.generated_settings = settings
        self.baseline = snap                       # Revert now returns to these values
        self.max_iter = settings.max_iter
        self.max_dist = effective_dist(settings.max_dist)
        self.gen_seg_mult = settings.seg_mult
        self.gen_shape = settings.shape
        self.gen_bound = make_bound(settings.shape, settings.params)
        self.gen_main_ctl = settings_main_ctl(settings)
        self.gen_fuzz = fuzz_array(settings.fuzz)
        self.gen_seed_base = settings.fuzz.seed
        self.gen_buddha_blur_radius = settings.buddha_blur_radius
        self.gen_buddha_blur_type = settings.buddha_blur_type
        (self.gen_obst_shapes, self.gen_obst_bounds, self.gen_obst_cx, self.gen_obst_cy,
         self.gen_obst_effects, self.gen_obst_fire_inside) = build_obstacle_arrays(self.obstacles)
        self.gen_obstacles_sig = obstacles_signature(self.obstacles)
        self.gen_obstacles_by_id = {o.id: self.obstacle_state(o) for o in self.obstacles}
        self.set_ring(self.gen_shape, self.gen_bound)   # white outline = generated shape
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

        while self.rows_done < SIZE:
            t0 = time.perf_counter()
            row1 = min(self.rows_done + self.batch_rows, SIZE)
            render_rows(self.stats, self.points, self.status, self.rows_done, row1,
                        self.max_iter, self.max_dist, self.gen_shape, self.gen_bound,
                        self.gen_main_ctl, self.gen_seg_mult, self.gen_fuzz,
                        self.gen_obst_shapes,
                        self.gen_obst_bounds, self.gen_obst_cx, self.gen_obst_cy,
                        self.gen_obst_effects, self.gen_obst_fire_inside,
                        self.gen_seed_base)
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
        """[(shape, bound, cx, cy, in_effect, out_effect), ...] for obstacles
        to draw a solid two-layer outline for: the frozen set if an image has
        been generated (matching what was actually simulated/saved), else the
        live (current) set."""
        if self.generated_settings is not None:
            return [(self.gen_obst_shapes[i], self.gen_obst_bounds[i],
                     self.gen_obst_cx[i], self.gen_obst_cy[i],
                     self.gen_obst_effects[i, 0], self.gen_obst_effects[i, 1])
                    for i in range(len(self.gen_obst_shapes))]
        return [(o.shape, make_bound(o.shape, o.params()), *o.center(), o.in_effect,
                 o.out_effect) for o in self.obstacles]

    def draw(self):
        """Push the current image to the screen, with the white shape outline
        and any obstacle outlines drawn into it if that decoration is enabled
        (two layers: outer = incoming effect, inner = outgoing effect; brown for
        Terminate, pale light blue for Reflect, black for Pass through),
        then the live view's name in the bottom-left corner - a preview-only
        decoration drawn into self.display_frame, a separate buffer from
        self.frame (which is what gets exported/saved, and so must never
        carry it)."""
        np.copyto(self.frame, self.img)
        outline = self.deco.get("outline")
        if outline is None or outline.get():
            self.frame[self.ring] = 255
            for shape, bound, cx, cy, in_e, out_e in self.obstacle_ring_specs():
                draw_obstacle_outline(self.frame, shape, bound, self.gx - cx, self.gy - cy,
                                      in_e, out_e)
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
        shape, params = self.control_shape()
        return Settings(max_iter=self.last_iter, max_dist=self.last_dist,
                        seg_mult=self.seg_mult_scale.get(), shape=shape, params=params,
                        fuzz=self.read_fuzz_controls(),
                        buddha_blur_radius=self.buddha_blur_radius_scale.get(),
                        buddha_blur_type=BUDDHA_BLUR_LABELS.index(
                            self.buddha_blur_type_var.get()),
                        main_fire=bool(self.main_fire_var.get()),
                        main_in_effect=OBSTACLE_EFFECT_NAMES.index(
                            self.main_in_effect_var.get()),
                        main_out_effect=OBSTACLE_EFFECT_NAMES.index(
                            self.main_out_effect_var.get()))

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
        bound = make_bound(ps.shape, ps.params)
        o_shapes, o_bounds, o_cx, o_cy, _o_eff, o_fire = build_obstacle_arrays(self.obstacles)
        if not emits_ray(x, y, ps.shape, bound, settings_main_ctl(ps), o_shapes, o_bounds,
                         o_cx, o_cy, o_fire):
            return                                 # clicking outside the shape does nothing
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
        bound = make_bound(ps.shape, ps.params)
        o_shapes, o_bounds, o_cx, o_cy, o_effects, o_fire = build_obstacle_arrays(self.obstacles)
        main_ctl = settings_main_ctl(ps)
        if not emits_ray(x, y, ps.shape, bound, main_ctl, o_shapes, o_bounds, o_cx, o_cy,
                         o_fire):
            self.clear_overlay()                   # not a ray-emitting pixel
            return
        if (i, j) == self.hover_px:
            return

        self.hover_px = (i, j)
        iters, dist = ps.max_iter, effective_dist(ps.max_dist)
        fz = fuzz_array(ps.fuzz)
        if ps.fuzz.enabled:
            np.random.seed(ps.fuzz.seed + i * 1000003 + j * 97)
        pts, kinds = self.hover_pts, self.hover_kinds
        raw, vec, sp = self.hover_raw, self.hover_vec, self.hover_sp
        term, n = trace_path(x, y, pts, kinds, iters, dist, ps.shape, bound, main_ctl,
                             ps.seg_mult, fz, o_shapes, o_bounds, o_cx, o_cy, o_effects, raw)
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
            self.save_jxl_label.pack(side="left", padx=(0, 4))
            self.save_jxl_combo.pack(side="left")
        else:
            self.save_jxl_label.pack_forget()
            self.save_jxl_combo.pack_forget()
        self.update_save_jxl_quality_visibility()

    def on_save_jxl_mode_changed(self, _event=None):
        self.update_save_jxl_quality_visibility()

    def update_save_jxl_quality_visibility(self):
        """The quality slider only matters for JPEG XL in Lossy mode."""
        if self.save_format_var.get() == "jxl" and self.save_jxl_mode_var.get() == "Lossy":
            self.save_jxl_quality_frame.pack(side="left")
        else:
            self.save_jxl_quality_frame.pack_forget()

    def save(self):
        if self.save_source_var.get() == SAVE_SOURCE_OPTIONS[1]:
            self.save_new_image(None)
        else:
            fmt = self.save_format_var.get()
            self._write(os.path.join(ensure_export_dir(),
                                     default_filename(self.generated_settings, fmt)))

    def save_as(self):
        if self.save_source_var.get() == SAVE_SOURCE_OPTIONS[1]:
            self.save_new_image_as()
        else:
            fmt = self.save_format_var.get()
            label = {"png": "PNG image", "qoi": "QOI image", "jxl": "JPEG XL image"}[fmt]
            path = filedialog.asksaveasfilename(
                parent=self.root, title="Save image as",
                initialdir=ensure_export_dir(),
                initialfile=default_filename(self.generated_settings, fmt),
                defaultextension="." + fmt, filetypes=[(label, f"*.{fmt}")])
            if path:
                self._write(path)

    def views_to_export(self):
        """Which views to include in an export, per the Export designer's
        "Include all views" checkbox: all of them, in list/composition
        order, or just the one currently selected/shown in the preview."""
        if self.include_all_views_var.get():
            return list(self.views)
        return [self.live_view()]

    def render_current_figure(self, views):
        """The CURRENT (already fully-traced) simulation, colorized by each
        of `views` and composed side by side at the live preview's own
        resolution (SIZE x SIZE per view) - reusing self.stats/self.points/
        self.status as already computed, no re-simulation. Each view gets
        the outline decoration drawn in if enabled, exactly like
        self.frame/draw(). Only called for more than one view - the
        single-view case is just self.frame, already rendered."""
        mini_settings = Settings(self.max_iter, self.max_dist, self.gen_seg_mult,
                                 self.gen_shape, (),
                                 FuzzParams(False, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                           self.gen_seed_base),
                                 self.gen_buddha_blur_radius, self.gen_buddha_blur_type)
        obst_arrays = (self.gen_obst_shapes, self.gen_obst_bounds, self.gen_obst_cx,
                      self.gen_obst_cy, self.gen_obst_effects, self.gen_obst_fire_inside)
        images = []
        for v in views:
            vs = ViewSettings(v.mode, v.kinds, v.point_metric, v.other_metric,
                              v.point1_choice, v.point2_choice, v.chan_src, v.buddha_points,
                              v.buddha_compression, v.buddha_brightness, v.constant_choice)
            images.append(self.colorize_view(vs, self.stats, self.points, self.status,
                                             SIZE, SIZE, mini_settings, self.gen_bound,
                                             self.gen_fuzz, obst_arrays))
        outline = self.deco.get("outline")
        if outline is None or outline.get():
            for img in images:
                img[self.ring] = 255
                for shape, bound, cx, cy, in_e, out_e in self.obstacle_ring_specs():
                    draw_obstacle_outline(img, shape, bound, self.gx - cx, self.gy - cy,
                                          in_e, out_e)
        return images[0] if len(images) == 1 else np.concatenate(images, axis=1)

    def _write(self, path):
        fmt = self.save_format_var.get()
        jxl_lossless = self.save_jxl_mode_var.get() == "Lossless"
        jxl_quality = int(round(float(self.save_jxl_quality_scale.get())))
        views = self.views_to_export()
        # self.frame already holds exactly the single-view case (with the
        # outline if that decoration is on, no preview overlays) - only
        # recompute when more than one view needs to be composed in.
        rgb = self.frame if len(views) == 1 else self.render_current_figure(views)
        rgb = apply_info_locations(
            rgb, views, rgb.shape[1] // len(views), self.generated_settings,
            self.color_info_location_var.get(), self.sim_info_location_var.get(),
            mutate_ok=len(views) > 1)   # len(views) > 1 means rgb is already a fresh
                                       # array from render_current_figure, not self.frame
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
        initial = default_filename(self.generated_settings, fmt)
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
        fake_kf = Keyframe(-1, 0, settings_dict, DEFAULT_INTERP, None)
        data = resolve_frame_settings([fake_kf], 0)
        obst_snapshots = [snapshot_obstacle(o) for o in self.obstacles]
        obst_arrays = resolve_obstacle_arrays(obst_snapshots, [fake_kf], 0)
        view_snapshots = [snapshot_view(v) for v in self.views_to_export()]
        views = resolve_views(view_snapshots, [fake_kf], 0)
        color_location = self.color_info_location_var.get()
        sim_location = self.sim_info_location_var.get()
        if path_or_none is None:
            settings_for_name = effective_settings(data.snapshot)
            path = os.path.join(ensure_export_dir(),
                                default_filename(settings_for_name, fmt))
        else:
            path = path_or_none
        self.save_btn.state(["disabled"])
        self.save_as_btn.state(["disabled"])
        thread = threading.Thread(
            target=self._save_new_image_worker,
            args=(data, w, h, obst_arrays, views, fmt, jxl_lossless, jxl_quality, path,
                 color_location, sim_location),
            daemon=True)
        thread.start()

    def _save_new_image_worker(self, data, w, h, obst_arrays, views, fmt, jxl_lossless,
                               jxl_quality, path, color_location, sim_location):
        """Runs on a background thread: never touches a Tk widget directly,
        only via self.root.after(0, ...)."""
        try:
            rgb = self.render_full_image(data, w, h, obst_arrays, views)
            rgb = apply_info_locations(rgb, views, w, effective_settings(data.snapshot),
                                       color_location, sim_location, mutate_ok=True)
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
        snap = self.current_snapshot()
        if snap is None:
            if show_error:
                messagebox.showerror("Invalid settings",
                                     "Fix the iteration/distance limit fields first.")
            return None
        f = snap.fuzz
        d = {
            "iter": snap.max_iter, "dist": snap.max_dist, "seg_mult": snap.seg_mult,
            "shape": snap.shape,
            "width": snap.width, "height": snap.height, "ell_rot": snap.ell_rotation,
            "sides": snap.sides, "poly_rot": snap.poly_rotation,
            "moss_angle": snap.moss_angle, "moss_size": snap.moss_size,
            "moss_rotation": snap.moss_rotation,
            "sector_radius": snap.sector_radius, "sector_span": snap.sector_span,
            "sector_rotation": snap.sector_rotation,
            "segment_radius": snap.segment_radius, "segment_span": snap.segment_span,
            "segment_rotation": snap.segment_rotation,
            "lens_radius1": snap.lens_radius1, "lens_radius2": snap.lens_radius2,
            "lens_sep": snap.lens_sep, "lens_rotation": snap.lens_rotation,
            "reuleaux_sides": snap.reuleaux_sides,
            "reuleaux_rotation": snap.reuleaux_rotation,
            "capsule_length": snap.capsule_length, "capsule_radius": snap.capsule_radius,
            "capsule_rotation": snap.capsule_rotation,
            "fuzz_on": f.enabled, "emit_angle": f.emit_angle, "emit_jitter": f.emit_jitter,
            "bnd_offset": f.bnd_offset, "bnd_jitter": f.bnd_jitter, "bnd_mult": f.bnd_mult,
            "self_offset": f.self_offset, "self_jitter": f.self_jitter,
            "self_mult": f.self_mult, "seed": f.seed,
            "buddha_blur_radius": snap.buddha_blur_radius,
            "buddha_blur_type": snap.buddha_blur_type,
            "main_fire": snap.main_fire, "main_in_effect": snap.main_in_effect,
            "main_out_effect": snap.main_out_effect,
            "outline": self.deco["outline"].get(),
        }
        for o in self.obstacles:
            for field, _label, _is_float in OBST_FIELD_INFO:
                d[f"obst:{o.id}:{field}"] = getattr(o, field)
        for v in self.views:
            d[f"view:{v.id}:mode"] = v.mode
            for c in range(3):
                d[f"view:{v.id}:chan{c}"] = view_channel_value(v, c)
        return d

    def capture_thumbnail(self):
        """A THUMB_SIZE x THUMB_SIZE PhotoImage: a bilinearly-resized copy of the
        image panel if it currently matches the controls, else a 'no img'
        placeholder."""
        if self.is_image_current():
            small = bilinear_resize(self.frame, THUMB_SIZE, THUMB_SIZE)
        else:
            small = no_img_thumbnail(THUMB_SIZE)
        return tk.PhotoImage(data=encode_png_bytes(small), format="png")

    def new_keyframe(self):
        """A new keyframe that defines every setting, from the controls' current
        values."""
        data = self.capture_all_settings()
        if data is None:
            return
        frame_index = (0 if not self.keyframes
                      else max(kf.frame_index for kf in self.keyframes) + 1)
        kf = Keyframe(self._kf_counter, frame_index, data, DEFAULT_INTERP,
                     self.capture_thumbnail())
        self._kf_counter += 1
        self.keyframes.append(kf)
        self.selected_keyframe = kf
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

    def load_value(self, key, val):
        """Apply one resolved value to whichever control(s) hold that setting.
        Does NOT itself refresh dependent state - callers batch that after."""
        if key.startswith("obst:"):
            _prefix, obst_id, field = key.split(":", 2)
            obst_id = int(obst_id)
            o = next((o for o in self.obstacles if o.id == obst_id), None)
            if o is not None:
                setattr(o, field, val)
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
                chan_src, kind, stat, point1, point2, buddha_pts, comp, bright, const = val
                v.chan_src[c] = chan_src
                v.kinds[c] = kind
                if kind == KIND_POINT:
                    v.point_metric[c] = stat
                else:
                    v.other_metric[c] = stat
                v.point1_choice[c] = point1
                v.point2_choice[c] = point2
                v.buddha_points[c] = buddha_pts
                v.buddha_compression[c] = comp
                v.buddha_brightness[c] = bright
                v.constant_choice[c] = const
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
        if key == "iter":
            self.iter_var.set(str(val))
        elif key == "dist":
            self.dist_var.set(str(val))
        elif key == "seg_mult":
            self.seg_mult_scale.set(val)
        elif key == "shape":
            self.shape_var.set(SHAPE_NAMES[val])
            self.show_shape_controls()
        elif key in SHAPE_PARAM_BY_KEY:
            getattr(self, SHAPE_PARAM_BY_KEY[key].widget).set(val)
        elif key == "fuzz_on":
            self.fuzz_enabled_var.set(val)
        elif key == "emit_angle":
            self.emit_angle_scale.set(val)
        elif key == "emit_jitter":
            self.emit_jitter_scale.set(val)
        elif key == "bnd_offset":
            self.bnd_offset_scale.set(val)
        elif key == "bnd_jitter":
            self.bnd_jitter_scale.set(val)
        elif key == "bnd_mult":
            self.bnd_mult_scale.set(val)
        elif key == "self_offset":
            self.self_offset_scale.set(val)
        elif key == "self_jitter":
            self.self_jitter_scale.set(val)
        elif key == "self_mult":
            self.self_mult_scale.set(val)
        elif key == "seed":
            self.seed_var.set(str(val))
        elif key == "buddha_blur_radius":
            self.buddha_blur_radius_scale.set(val)
        elif key == "buddha_blur_type":
            self.buddha_blur_type_var.set(BUDDHA_BLUR_LABELS[val])
        elif key == "main_fire":
            self.main_fire_var.set(bool(val))
        elif key == "main_in_effect":
            self.main_in_effect_var.set(OBSTACLE_EFFECT_NAMES[int(val)])
        elif key == "main_out_effect":
            self.main_out_effect_var.set(OBSTACLE_EFFECT_NAMES[int(val)])
        elif key == "outline":
            self.deco["outline"].set(val)
            self.on_decoration_changed("outline")

    def load_setting(self, kf, key):
        """Apply kf.data[key] to the controls, if kf actually defines it."""
        if key in kf.data:
            self.load_value(key, kf.data[key])

    def load_settings(self, kf, keys):
        """Load several settings from `kf` into the controls, then refresh
        everything that depends on them once."""
        for key in keys:
            self.load_setting(kf, key)
        self.on_limit_var_changed()
        self.on_shape_param_changed()
        self.on_fuzz_changed()
        self.refresh_timeline_diffs()

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
            field = key.split(":", 2)[2]
            return next(is_f for f, _label, is_f in OBST_FIELD_INFO if f == field)
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
        refresh afterwards, the same way load_settings() already does for
        "present" keys."""
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
            kf.data[key] = current[key]
        kf.thumb = self.capture_thumbnail()
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

    def delete_setting(self, kf, key):
        """Remove `key` from `kf`, so it goes back to being either animated by
        surrounding keyframes or undefined."""
        if key in kf.data:
            del kf.data[key]
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

    def setting_status(self, kf, key, sorted_kfs, idx_kf):
        """How `key` relates to `kf`: 'present' (kf defines it), 'animated'
        (kf doesn't, but some keyframe before AND some keyframe after it in
        timeline order do, so it can still be loaded/saved here), or 'none'
        (nobody nearby defines it)."""
        if key in kf.data:
            return "present"
        before = any(key in k2.data for k2 in sorted_kfs[:idx_kf])
        after = any(key in k2.data for k2 in sorted_kfs[idx_kf + 1:])
        return "animated" if before and after else "none"

    def select_keyframe(self, kf):
        self.selected_keyframe = kf
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

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
            row.columnconfigure(1, weight=1)

            thumb_lbl = tk.Label(row, image=kf.thumb, bd=0, cursor="hand2")
            thumb_lbl.grid(row=0, column=0, padx=(2, 6), pady=2)
            thumb_lbl.bind("<Button-1>", lambda _e, kf=kf: self.select_keyframe(kf))

            lbl = tk.Label(row, text=f"#{idx + 1}  frame {kf.frame_index}",
                          anchor="w", cursor="hand2", font=self.small_font)
            lbl.grid(row=0, column=1, sticky="ew")
            lbl.bind("<Button-1>", lambda _e, kf=kf: self.select_keyframe(kf))
        self.update_kf_detail_panel()

    def all_setting_entries(self):
        """[(key, tab, label, is_float, fallback), ...] for every static
        setting plus every current obstacle's and view's fields, in display
        order."""
        entries = [(key, tab, label, key in SETTING_FLOAT_KEYS, DEFAULT_SETTING_VALUES[key])
                  for key, tab, label in SETTING_INFO]
        for idx, o in enumerate(self.obstacles):
            tab = f"Obstacle #{idx + 1}"
            for field, label, is_float in OBST_FIELD_INFO:
                entries.append((f"obst:{o.id}:{field}", tab, label, is_float,
                               getattr(o, field)))
        for idx, v in enumerate(self.views):
            tab = f"View {idx}"
            for field, label, is_float in VIEW_FIELD_INFO:
                entries.append((f"view:{v.id}:{field}", tab, label, is_float,
                               view_field_value(v, field)))
        return entries

    def all_setting_tabs(self):
        """Tab names in display order: the fixed ones, then one per
        obstacle, then one per view."""
        return (SETTING_TABS_ORDER + [f"Obstacle #{i + 1}" for i in range(len(self.obstacles))]
               + [f"View {i}" for i in range(len(self.views))])

    def refresh_timeline_diffs(self):
        """Rebuild the settings list for the selected keyframe: every setting,
        grouped by the tab it lives on, each showing whether it's defined here,
        animated by surrounding keyframes, or neither."""
        if not hasattr(self, "diff_list_frame"):
            return                                 # Timeline panel not built yet
        for w in self.diff_list_frame.winfo_children():
            w.destroy()
        kf = self.selected_keyframe

        if kf is None:
            self.diff_load_all_btn.state(["disabled"])
            self.diff_save_all_btn.state(["disabled"])
            self.diff_delete_all_btn.state(["disabled"])
            ttk.Label(self.diff_list_frame, text="(select a keyframe)",
                     foreground="#666", font=self.small_font).grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return
        current = self.capture_all_settings(show_error=False)
        if current is None:
            self.diff_load_all_btn.state(["disabled"])
            self.diff_save_all_btn.state(["disabled"])
            self.diff_delete_all_btn.state(["disabled"])
            ttk.Label(self.diff_list_frame, text="(fix the limit fields first)",
                     foreground="#666", font=self.small_font).grid(
                row=0, column=0, sticky="w", padx=4, pady=4)
            return

        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame_index)
        idx_kf = sorted_kfs.index(kf)
        self.diff_load_all_btn.state(["!disabled"])
        self.diff_save_all_btn.state(["!disabled"])
        self.diff_delete_all_btn.state(["!disabled"] if kf.data else ["disabled"])

        row = 0
        entries = self.all_setting_entries()
        for tab in self.all_setting_tabs():
            tab_entries = [e for e in entries if e[1] == tab]
            if not tab_entries:
                continue
            ttk.Label(self.diff_list_frame, text=tab, font=self.small_bold_font).grid(
                row=row, column=0, sticky="w", padx=4, pady=(6, 1))
            row += 1
            for key, _tab, label, is_float, fallback in tab_entries:
                status = self.setting_status(kf, key, sorted_kfs, idx_kf)
                r = ttk.Frame(self.diff_list_frame)
                r.grid(row=row, column=0, sticky="ew", padx=4, pady=(1, 3))
                ttk.Label(r, text=label, font=self.small_font).grid(
                    row=0, column=0, sticky="w")
                btns = ttk.Frame(r)
                btns.grid(row=1, column=0, sticky="w")

                if status == "present":
                    load_text = f"Load {short_value_text(key, kf.data[key])}"
                    ttk.Button(btns, text=load_text,
                              command=lambda kf=kf, k=key: self.load_settings(
                                  kf, [k])).pack(side="left", padx=(0, 2))
                elif status == "animated":
                    val = resolve_value(key, sorted_kfs, kf.frame_index, fallback, is_float)
                    load_text = f"Load {short_value_text(key, val)}"
                    ttk.Button(btns, text=load_text,
                              command=lambda kf=kf, k=key: self.load_animated_setting(
                                  kf, k)).pack(side="left", padx=(0, 2))

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
                row += 1

    def load_all_diffs(self):
        kf = self.selected_keyframe
        if kf is None:
            return
        sorted_kfs = sorted(self.keyframes, key=lambda k: k.frame_index)
        idx_kf = sorted_kfs.index(kf)
        present_keys = []
        animated_keys = []
        for key, _tab, _label, _is_float, _fallback in self.all_setting_entries():
            status = self.setting_status(kf, key, sorted_kfs, idx_kf)
            if status == "present":
                present_keys.append(key)
            elif status == "animated":
                animated_keys.append(key)
        for key in animated_keys:
            self.load_animated_setting(kf, key, refresh=False)   # one shared refresh below
        if present_keys:
            self.load_settings(kf, present_keys)   # refreshes once, covering the
                                                   # animated keys loaded above too
        elif animated_keys:
            self.on_limit_var_changed()
            self.on_shape_param_changed()
            self.on_fuzz_changed()
            self.refresh_timeline_diffs()
        else:
            self.refresh_timeline_diffs()

    def save_all_diffs(self):
        kf = self.selected_keyframe
        if kf is None:
            return
        self.save_settings(kf, [key for key, _t, _l, _f, _fb in self.all_setting_entries()])

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
            self.kf_interp_var.set(DEFAULT_INTERP)
            self.kf_frame_entry.state(["disabled"])
            self.kf_interp_combo.state(["disabled"])
            self.kf_split_btn.state(["disabled"])
            self.kf_delete_btn.state(["disabled"])
            return
        idx = self.keyframes.index(kf)
        self.kf_detail_label_var.set(f"Keyframe #{idx + 1}")
        self.kf_frame_var.set(str(kf.frame_index))
        self.kf_interp_var.set(kf.interp)
        self.kf_frame_entry.state(["!disabled"])
        self.kf_interp_combo.state(["!disabled"])
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

    def on_kf_interp_changed(self, _event=None):
        kf = self.selected_keyframe
        if kf is None:
            return
        kf.interp = self.kf_interp_var.get()
        self.kf_interp_combo.selection_clear()

    def reorder_keyframes(self):
        """Sort keyframes ascending by frame index (stable, so ties keep their
        relative order)."""
        self.keyframes.sort(key=lambda kf: kf.frame_index)

    def split_selected_keyframe(self):
        kf = self.selected_keyframe
        if kf is None:
            return
        new_kf = Keyframe(self._kf_counter, kf.frame_index, dict(kf.data), kf.interp, kf.thumb)
        self._kf_counter += 1
        idx = self.keyframes.index(kf)
        self.keyframes.insert(idx + 1, new_kf)
        self.rebuild_keyframe_list()

    def delete_selected_keyframe(self):
        kf = self.selected_keyframe
        if kf is None:
            return
        self.keyframes.remove(kf)
        self.selected_keyframe = None
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()

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
        if fmt == "mp4 (h264)":
            self.bitrate_mode_row.pack(side="left")

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
        limits, fuzzing); `obst_arrays` is a (shapes, bounds, cx, cy,
        effects, fire_inside) tuple built on the main thread beforehand.

        `views` is every view to render (see resolve_views()), in order -
        each one rendered as its own full (out_h, out_w, 3) image from the
        SAME traced rays (ray-tracing depends only on the simulation, never
        on how it's colored), then placed side by side: the returned image
        is (out_h, out_w * len(views), 3). With a single view (the ordinary
        case before this feature existed), this is exactly the one image
        it's always been."""
        o_shapes, o_bounds, o_cx, o_cy, o_effects, o_fire = obst_arrays
        settings = effective_settings(data.snapshot)
        bound = make_bound(settings.shape, settings.params)
        fz = fuzz_array(settings.fuzz)
        stats = np.zeros((out_h, out_w, N_STATS), dtype=np.float32)
        points = np.zeros((out_h, out_w, N_POINTS, 2), dtype=np.float32)
        status = np.zeros((out_h, out_w), dtype=np.uint8)
        render_rows_wh(stats, points, status, 0, out_h, out_w, out_h, settings.max_iter,
                       effective_dist(settings.max_dist), settings.shape, bound,
                       settings_main_ctl(settings), settings.seg_mult, fz, o_shapes,
                       o_bounds, o_cx, o_cy, o_effects, o_fire, settings.fuzz.seed)

        images = [self.colorize_view(v, stats, points, status, out_w, out_h, settings, bound,
                                     fz, obst_arrays)
                 for v in views]

        if data.outline:
            gx, gy = make_grid(out_w, out_h)
            ring = self.compute_ring(settings.shape, bound, gx, gy)
            obst_layers = [ring_layer_masks(o_shapes[oi], o_bounds[oi],
                                            gx - o_cx[oi], gy - o_cy[oi])
                           + (o_effects[oi, 0], o_effects[oi, 1]) for oi in range(len(o_shapes))]
            for img in images:
                img[ring] = 255
                for outer, inner, in_e, out_e in obst_layers:
                    img[outer] = OBST_EFFECT_RGB[int(in_e)]
                    img[inner] = OBST_EFFECT_RGB[int(out_e)]

        return images[0] if len(images) == 1 else np.concatenate(images, axis=1)

    def colorize_view(self, v, stats, points, status, out_w, out_h, settings, bound, fz,
                      obst_arrays):
        """One view's (out_h, out_w, 3) uint8 image (background color where no
        ray was computed, no outline - that's applied uniformly afterwards)
        from already-traced stats/points/status. `v` is a ViewSettings: this
        is the same coloring computation every view goes through, whichever
        one it is."""
        o_shapes, o_bounds, o_cx, o_cy, o_effects, o_fire = obst_arrays
        mapping = make_mapping(v.kinds, v.point_metric, v.other_metric, v.chan_src,
                               v.constant_choice)
        img = np.empty((out_h, out_w, 3), dtype=np.uint8)
        img[:] = BACKGROUND
        computed = status == 1
        n = int(computed.sum())
        if n:
            radius_px = world_radius_to_pixels(settings.buddha_blur_radius, out_w, out_h)
            buddha_kernel = build_buddha_blur_kernel(radius_px, settings.buddha_blur_type)
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
                            settings.shape, bound, settings_main_ctl(settings),
                            settings.seg_mult, fz, o_shapes, o_bounds,
                            o_cx, o_cy, o_effects, o_fire, settings.fuzz.seed,
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
        v = ViewSettings(view_obj.mode, view_obj.kinds, view_obj.point_metric,
                         view_obj.other_metric, view_obj.point1_choice, view_obj.point2_choice,
                         view_obj.chan_src, view_obj.buddha_points, view_obj.buddha_compression,
                         view_obj.buddha_brightness, view_obj.constant_choice)
        mini_settings = Settings(self.max_iter, self.max_dist, self.gen_seg_mult,
                                 self.gen_shape, (),
                                 FuzzParams(False, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                           self.gen_seed_base),
                                 self.gen_buddha_blur_radius, self.gen_buddha_blur_type)
        obst_arrays = build_obstacle_arrays(self.obstacles)
        full = self.colorize_view(v, self.stats, self.points, self.status, SIZE, SIZE,
                                  mini_settings, self.gen_bound, self.gen_fuzz, obst_arrays)
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
        if ffmpeg is None:
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

        os.makedirs(TEMP_DIR, exist_ok=True)
        existing = os.listdir(TEMP_DIR)
        if existing:
            messagebox.showerror("TEMP folder is not empty",
                                 f"{TEMP_DIR}\nalready has {len(existing)} file(s) "
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
        obst_snapshot = [snapshot_obstacle(o) for o in self.obstacles]
        view_snapshot = [snapshot_view(v) for v in self.views_to_export()]
        color_location = self.color_info_location_var.get()
        sim_location = self.sim_info_location_var.get()

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
                 color_location, sim_location),
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
            return "jxl", ""
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
            return os.path.join(TEMP_DIR, "frame_%05d.png"), start_number, None
        scratch_dir = os.path.join(TEMP_DIR, "_downsample_scratch")
        if os.path.isdir(scratch_dir):
            shutil.rmtree(scratch_dir)
        os.makedirs(scratch_dir, exist_ok=True)
        for i in range(num_frames):
            src = os.path.join(TEMP_DIR, f"frame_{start_number + i:05d}.png")
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
            stream = ffmpeg.filter(stream, "scale", f"iw/{factor}", f"ih/{factor}",
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
        if isinstance(exc, ffmpeg.Error) and getattr(exc, "stderr", None):
            decoded = exc.stderr.decode("utf-8", errors="replace").strip()
            if decoded:
                stderr_text = "\nffmpeg said:\n" + decoded[-4000:]
        return (f"{exc}{stderr_text}\n"
               f"Frames folder: {TEMP_DIR}\n"
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
        support - the temporary APNG is removed either way."""
        kwargs = self.ffmpeg_output_kwargs(rs["fmt"], rs["mode"], rs["quality"],
                                           rs["bitrate"], rs["vbr_target"],
                                           rs["vbr_max"])
        ffmpeg_out_path = out_path
        jxl_apng_temp = None
        if rs["fmt"] == "JXL":
            jxl_apng_temp = os.path.join(TEMP_DIR, "_jxl_video_source.apng")
            ffmpeg_out_path = jxl_apng_temp
        stream = ffmpeg.input(pattern, framerate=rs["fps"], start_number=start_number)
        stream = self.apply_downsample_filter(stream, rs)
        stream = ffmpeg.output(stream, ffmpeg_out_path, **kwargs)
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        if jxl_apng_temp is not None:
            self.root.after(0, lambda: self.set_video_status("Converting to JXL with cjxl..."))
            try:
                convert_apng_to_jxl(jxl_apng_temp, out_path)
            finally:
                if os.path.exists(jxl_apng_temp):
                    os.remove(jxl_apng_temp)

    def _render_worker(self, keyframes, frame_lo, frame_hi, rs, obst_snapshot, view_snapshot,
                       color_location, sim_location):
        """Runs on a background thread: never touches a Tk widget directly, only
        via self.root.after(0, ...)."""
        ext, suffix = self.output_extension_and_suffix(rs)
        out_name = f"{PROGRAM_NAME}-video-{time.strftime('%Y%m%d-%H%M%S')}{suffix}.{ext}"
        out_path = os.path.join(ensure_export_dir(), out_name)
        pattern = os.path.join(TEMP_DIR, "frame_%05d.png")
        scratch_dir = None
        try:
            assigned_sorted = sorted(keyframes, key=lambda k: k.frame_index)
            total = frame_hi - frame_lo + 1
            for n, fn in enumerate(range(frame_lo, frame_hi + 1)):
                if self.render_cancel.is_set():
                    self.root.after(0, lambda: self.set_video_status("Cancelled."))
                    return
                data = resolve_frame_settings(assigned_sorted, fn)
                obst_arrays = resolve_obstacle_arrays(obst_snapshot, assigned_sorted, fn)
                views = resolve_views(view_snapshot, assigned_sorted, fn)
                rgb = self.render_full_image(data, rs["video_w"], rs["video_h"], obst_arrays,
                                             views)
                # Simulation settings can be keyframe-animated, so this is resolved
                # per frame - but the margin's own size depends only on the fixed
                # video_w/video_h, so every frame still comes out the same size.
                rgb = apply_info_locations(rgb, views, rs["video_w"],
                                           effective_settings(data.snapshot),
                                           color_location, sim_location, mutate_ok=True)
                path = os.path.join(TEMP_DIR, f"frame_{fn - frame_lo:05d}.png")
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
        if not os.path.isdir(TEMP_DIR):
            messagebox.showerror("No frames",
                                 f"{TEMP_DIR}\ndoesn't exist yet - render a video "
                                 "first, or place frame PNGs there yourself.")
            return
        frame_files = sorted(f for f in os.listdir(TEMP_DIR)
                             if re.fullmatch(r"frame_\d+\.png", f))
        if not frame_files:
            messagebox.showerror("No frames",
                                 f"{TEMP_DIR}\nhas no frame_NNNNN.png files to "
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
        out_name = f"{PROGRAM_NAME}-video-{time.strftime('%Y%m%d-%H%M%S')}{suffix}.{ext}"
        out_path = os.path.join(ensure_export_dir(), out_name)
        pattern = os.path.join(TEMP_DIR, "frame_%05d.png")
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
        if not os.path.isdir(TEMP_DIR):
            messagebox.showinfo("Nothing to clear", f"{TEMP_DIR}\ndoesn't exist yet.")
            return
        frame_files = sorted(f for f in os.listdir(TEMP_DIR)
                             if re.fullmatch(r"frame_\d+\.png", f))
        if not frame_files:
            messagebox.showinfo("Nothing to clear",
                                f"{TEMP_DIR}\nhas no frame_NNNNN.png files to delete.")
            return
        if not messagebox.askyesno(
                "Delete frame files?",
                f"Delete {len(frame_files)} frame_NNNNN.png file(s) from\n{TEMP_DIR}?\n\n"
                "Only files matching that exact name pattern are touched - anything "
                "else in the folder is left alone. This can't be undone."):
            return
        errors = []
        deleted = 0
        for f in frame_files:
            try:
                os.remove(os.path.join(TEMP_DIR, f))
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
