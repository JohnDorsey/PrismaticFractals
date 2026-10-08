"""The program's fixed numbers: allowed ranges, defaults, colours and sizes of the user interface, and the lookup
tables built from them. None of these is read by a numba kernel (those are in kernels.py, see its docstring); a
value that is both needed here and by a kernel lives in kernels.py and is imported from there."""
import math
from .kernels import (
    CONFINE_BOX, EMIT_UP, TAG_BOUNDARY, TAG_DISTCUT, TAG_ESCAPED, TAG_OBSTACLE_TERM,
    TAG_PERPENDICULAR, TAG_SELF, TAG_START, OBSTACLE_EFFECT_PASS, OBSTACLE_EFFECT_REFLECT,
    OBSTACLE_EFFECT_TERMINATE, PIXEL, SIZE, TERM_CONFINED, TERM_DIST, TERM_ESCAPED, TERM_ITER,
    TERM_OBSTACLE, TERM_PERPENDICULAR
)


ITER_LIMIT_MIN = 1         # allowed range (whole numbers) for the iteration limit
ITER_LIMIT_MAX = 9999
DIST_LIMIT_MIN = 0         # allowed range for the distance limit; 0 = no limit
DIST_LIMIT_MAX = 9999
DEFAULT_ITER = 256         # default iteration limit (bounces)
DEFAULT_DIST = 0           # default distance limit (units): 0 = no limit
DEFAULT_ESCAPE_RADIUS = 2.0     # an escaping ray terminates this far from the origin (any number above 0)
SEGMENT_MULT_MIN = 0.01    # how much of each bounce's true segment length is
SEGMENT_MULT_MAX = 1.0     # actually travelled before continuing in the newly
DEFAULT_SEGMENT_MULT = 1.0 # -calculated (bounced) direction; 1 = no effect
WARM_ITER = 8              # iteration limit used for the compile warm-up

# Buddha blur: spreads each scattered point's weight across a neighborhood of
# pixels instead of a single one, applied only at the moment a point is
# recorded into a Buddha data channel - never touches the ray simulation
# itself. The radius control is a multiple of the output image's smaller side
# in PIXELS (world_radius_to_pixels): control value 0.1 -> 0.1 * min(width,
# height) pixels, so the blur looks the same at any export size and for any
# framing (a zoomed-in framing blurs over fewer world units).
BUDDHA_BLUR_GAUSSIAN = 0
BUDDHA_BLUR_TRIANGULAR = 1
BUDDHA_BLUR_BOKEH = 2
BUDDHA_BLUR_CATADIOPTRIC = 3
BUDDHA_BLUR_LABELS = ["Gaussian distribution", "Triangular distribution", "Bokeh",
                     "Catadioptric bokeh"]
BUDDHA_BLUR_RADIUS_MIN = 0.0
BUDDHA_BLUR_RADIUS_MAX = 0.5
# Buddha seeding: where the rays that Buddha channels scatter are emitted (see BuddhaSeedSettings). The zone
# defaults to the whole world, which is exactly what the camera pixels cover.
DEFAULT_BUDDHA_SEED_CAMERA = True
DEFAULT_BUDDHA_SEED_X_START = -1.0
DEFAULT_BUDDHA_SEED_X_STOP = 1.0
DEFAULT_BUDDHA_SEED_Y_START = -1.0
DEFAULT_BUDDHA_SEED_Y_STOP = 1.0
SEED_ZONE_OUTLINE = "#a040e0"      # the seeding zone's dashed outline on the preview (purple)
DEFAULT_FRAMING_X = 0.0            # a framing: where the camera is (the middle of what it sees) ...
DEFAULT_FRAMING_Y = 0.0
DEFAULT_FRAMING_DIAGONAL = 2.0 * math.sqrt(2.0)   # ... how far across it sees (world units): the whole world at aspect 1
DEFAULT_FRAMING_ASPECT = 1.0       # ... and the shape of what it sees (width / height)
DEFAULT_BUDDHA_BLUR_RADIUS = 0.0     # no blur - exactly the pre-blur single-pixel behavior
DEFAULT_BUDDHA_BLUR_TYPE = BUDDHA_BLUR_GAUSSIAN
BUDDHA_BLUR_GAUSSIAN_TRUNCATE = 3.0   # gaussian is truncated past this many standard
                                     # deviations - negligible weight beyond it, but
                                     # per the spec it genuinely does extend past the
                                     # nominal radius, unlike the other three kernels
BUDDHA_BLUR_CATADIOPTRIC_INNER_RATIO = 0.5   # the "donut" hole's inner edge, as a
                                             # fraction of the outer radius

# The ST_TERMINATION metric's 0..1 value for each termination reason (see
# compute_run_measurements) - also used outside njit context to filter by termination
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
LAYOUT_SQUIRCLE = 5         # a convex curve given by an implicit function (the two squircles), found by root search
LAYOUT_SPIRAL = 6           # the Archimedean spiral ribbon, found by root search
LAYOUT_SNAIL = 7            # the snail of Pythagoras: the corners of a union of triangles
LAYOUT_EGG = 8              # the Hugelschaffer egg: a cubic, found by root search
LAYOUT_FLOWER = 9           # the sine flower: a polar curve, found by root search
LAYOUTS = (LAYOUT_ELLIPSE, LAYOUT_POLYGON, LAYOUT_CIRCLES, LAYOUT_ARCS_EDGES, LAYOUT_ARC_LOOPS,
           LAYOUT_SQUIRCLE, LAYOUT_SPIRAL, LAYOUT_SNAIL, LAYOUT_EGG, LAYOUT_FLOWER)

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

# Hugelschaffer egg: half its length a, half its width b, and the shift d of its second circle's centre as a
# fraction of a (the egg is closed for |d| < a; near that it comes to a needle point)
EGG_AXIS_MAX = 2.0
DEFAULT_EGG_A = 1.0
DEFAULT_EGG_B = 0.7
EGG_SHIFT_MIN = -0.9
EGG_SHIFT_MAX = 0.9
DEFAULT_EGG_SHIFT = 0.4

# Sine flower rho = R (1 + A sin(k phi)): the number of petals k and the amplitude A (below 1, or the petals would
# pinch off at the centre); the radius is the shared one
FLOWER_MIN_PETALS = 1
FLOWER_MAX_PETALS = 30
DEFAULT_FLOWER_PETALS = 6
FLOWER_AMPLITUDE_MIN = 0.0
FLOWER_AMPLITUDE_MAX = 0.95
DEFAULT_FLOWER_AMPLITUDE = 0.3

# Snail of Pythagoras: how many right triangles it is made of
SNAIL_MIN_SEGMENTS = 1
SNAIL_MAX_SEGMENTS = 100
DEFAULT_SNAIL_SEGMENTS = 17     # the first count whose last triangle overlaps the first

# Obstacles: a fixed-width row per obstacle (padded with zeros) so their bounds
# can live in one ordinary numpy array for the njit kernels (the longest is the snail's: its n + 1 corners).
OBST_BOUND_LEN = max(3 + 2 * POLY_MAX_SIDES, 3 + 2 * (SNAIL_MAX_SEGMENTS + 1))

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

# Superelliptical squircle |x/a|^n + |y/b|^n = 1: the exponent n (2 is an ellipse, 1 a diamond, large n a
# rectangle; below 1 the shape would stop being convex, which its root finder relies on)
SUPER_EXPONENT_MIN = 1.0
SUPER_EXPONENT_MAX = 16.0
DEFAULT_SUPER_EXPONENT = 4.0

# Fernández-Guasti squircle x^2 + y^2 - (s^2/r^2) x^2 y^2 = r^2: the squareness s (0 is a circle of radius r, 1 the
# square of side 2r; the radius is the shared one)
SQUARENESS_MIN = 0.0
SQUARENESS_MAX = 1.0
DEFAULT_SQUARENESS = 0.7

# Archimedean spiral ribbon: the radial distance between successive turns (the pitch), and how much of it
# the ribbon fills (the shared radius is the outer edge's greatest distance from the centre; 1 would fill the
# whole disc and leave no edge between turns)
SPIRAL_PITCH_MIN = 0.04
SPIRAL_PITCH_MAX = 1.0
DEFAULT_SPIRAL_PITCH = 0.25
SPIRAL_FILL_MIN = 0.05
SPIRAL_FILL_MAX = 0.95
DEFAULT_SPIRAL_FILL = 0.5

# Fuzzing controls
FUZZ_JITTER_MIN = 0.0      # jitter range (+/-), degrees
FUZZ_JITTER_MAX = 180.0
FUZZ_MULT_MIN = 0.0        # bounce angle multiplier range
FUZZ_MULT_MAX = 3.0
DEFAULT_RNG_SEED = 0

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
CONFINE_LOOKBACK_MIN = 2
CONFINE_LOOKBACK_MAX = 999
DEFAULT_CONFINE_LOOKBACK = 4
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
# length rather than of position). Each bar is colored by the tag of the point
# at its top (earlier) and bottom (later) end.
SEG_CANVAS_W = SIZE
SEG_CANVAS_H = SIZE // 2   # 1 world unit, at the main image's pixels-per-unit
# The corkscrew graphic: the sequence's segments all the same length, laid end to end from the centre of a
# square canvas, each pointing the way its own segment does. The length is chosen so that the farthest point
# is CORKSCREW_MARGIN of the canvas's side (at least CORKSCREW_MARGIN_MIN pixels) short of the edge.
CORKSCREW_MARGIN = 0.04
CORKSCREW_MARGIN_MIN = 6
CORKSCREW_LINE_WIDTH = 2
CORKSCREW_TEXT_W = 130             # width of the description to the left of the canvas (narrow enough that the canvas is
                                   # limited by the tab's height, not its width, at the window's size)
CORKSCREW_SIDE_MIN = 60
TAG_COLOR = {TAG_START: (67, 160, 71), TAG_DISTCUT: (67, 160, 71),   # green
             TAG_BOUNDARY: (229, 57, 53),                              # red
             TAG_SELF: (30, 136, 229),                                 # blue
             TAG_OBSTACLE_TERM: (255, 235, 59),                        # yellow
             TAG_ESCAPED: (0, 229, 255),                               # cyan
             TAG_PERPENDICULAR: (124, 77, 255)}                        # purple (as TERM_END_COLOR's)

# Solid outline colors for obstacles (part of the saved image/video, drawn with
# the same "outline" decoration flag), by ray effect
OBST_TERMINATE_RGB = (121, 85, 72)     # brown
OBST_REFLECT_RGB = (179, 229, 252)     # pale light blue
OBST_PASS_RGB = (0, 0, 0)              # black
# An obstacle's outline is two layers, each this many pixels thick: the outer
# one (just outside its edge) shows its incoming effect, the inner one (just
# inside) its outgoing effect.
OBST_RING_LAYER_PX = 1.5

OBSTACLE_EFFECT_NAMES = ["Reflect", "Terminate", "Pass through"]
OBST_EFFECT_RGB = {OBSTACLE_EFFECT_REFLECT: OBST_REFLECT_RGB,
                   OBSTACLE_EFFECT_TERMINATE: OBST_TERMINATE_RGB,
                   OBSTACLE_EFFECT_PASS: OBST_PASS_RGB}

# Obstacle and coloration thumbnails
THUMB_SIZE = 48            # small resolution, per pixel
THUMB_BACKGROUND = (110, 110, 110)   # obstacle thumbnails: background and shape fill
THUMB_FILL = (150, 150, 150)
PREVIEW_SIZE = 150         # side length of the video-panel frame preview
SELECTED_OUTLINE = "#ffeb3b"       # yellow
SELECTED_OUTLINE_PX = 4

# The live preview's "coloration N" name overlay (bottom-left corner, drawn with
# FONT_5X7 - see draw_coloration_name_overlay). Preview-only: never part of an
# exported image.
COLORATION_NAME_OVERLAY_SCALE = 2
COLORATION_NAME_OVERLAY_PAD = 3          # blank border between the text and its background box
COLORATION_NAME_OVERLAY_MARGIN = 6       # gap between the background box and the image edge
COLORATION_NAME_OVERLAY_BG = (0, 0, 0)
COLORATION_NAME_OVERLAY_FG = (255, 255, 255)

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

MODIFIED_POLL_MS = 100
PROJECT_POLL_MS = 1000     # how often the window title's unsaved-changes mark is refreshed


# Brighter colors for the on-image sample-point markers (dark image behind them)
SAMPLE_COLORS = {"RGB": ["#ff5252", "#4cff6a", "#5aa0ff"],
                  "HSV": ["#c77dff", "#3de0e8", "#d8d8d8"]}

# Decorations: (key, label, default). Shown in two groups on the Decorations tab.
DECO_IMAGE = [      # drawn into the image: they appear in saved files (and keyframes)
    ("outline", "Obstacle outlines", True),
]
DECO_PREVIEW = [    # drawn over the image on screen only: never saved, never in a keyframe
    ("pending", "Pending shape outline (dashed)", True),
    ("ray_path", "Ray path", True),
    ("ray_start", "Ray start marker", True),
    ("ray_end", "Ray endpoint marker", True),
    ("guides", "Termination point guide lines", True),
    ("samples", "Color sample points (one ring per point-based channel)", False),
]
