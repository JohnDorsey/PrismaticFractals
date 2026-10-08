"""The obstacles. Every obstacle is an instance of one subclass of Shape and stays that kind for its whole life (the kind is
chosen when it is created); a subclass has only the parameters its own geometry needs. This module holds the Shape classes, the
functions that turn shapes into what the kernels take (make_bound, build_obstacle_arrays), and the drawing helpers that show them
(thumbnails, outlines, the ring layers). The packed bound arrays themselves are built by geometry.py, which this module imports."""
from collections import namedtuple
from math import cos, pi, radians, sin, sqrt
import numpy as np
from .constants import (
    ARC_SPAN_MAX, ARC_SPAN_MIN, CAPSULE_LENGTH_MAX, CAPSULE_LENGTH_MIN, CAPSULE_RADIUS_MAX,
    CAPSULE_RADIUS_MIN, CUT_OFFSET_MAX, CUT_OFFSET_MIN, CUT_RADIUS_MAX, CUT_RADIUS_MIN,
    DEFAULT_CAPSULE_LENGTH, DEFAULT_CAPSULE_RADIUS, DEFAULT_CUT_OFFSET, DEFAULT_CUT_RADIUS,
    DEFAULT_LENS_RADIUS, DEFAULT_LENS_SEP, DEFAULT_LOBE_SIZE, DEFAULT_MOSS_ANGLE,
    DEFAULT_MOSS_SIZE, DEFAULT_RADIUS, DEFAULT_REULEAUX_SIDES, DEFAULT_ROTATION,
    DEFAULT_EGG_A, DEFAULT_EGG_B, DEFAULT_EGG_SHIFT, DEFAULT_FLOWER_AMPLITUDE, DEFAULT_FLOWER_PETALS,
    DEFAULT_SECTOR_SPAN, DEFAULT_SEGMENT_SPAN,
    DEFAULT_SIDES, DEFAULT_SNAIL_SEGMENTS, DEFAULT_SPIRAL_FILL,
    DEFAULT_SPIRAL_PITCH, DEFAULT_SPLIT, DEFAULT_SQUARENESS, DEFAULT_SUPER_EXPONENT, DEFAULT_WIDTH,
    EGG_AXIS_MAX, EGG_SHIFT_MAX, EGG_SHIFT_MIN, ELLIPSE_MAX, ELLIPSE_MIN, FLOWER_AMPLITUDE_MAX,
    FLOWER_AMPLITUDE_MIN, FLOWER_MAX_PETALS, FLOWER_MIN_PETALS, LAYOUT_ARCS_EDGES, LAYOUT_ARC_LOOPS, LAYOUT_CIRCLES,
    LAYOUT_EGG, LAYOUT_ELLIPSE, LAYOUT_FLOWER, LAYOUT_POLYGON, LAYOUT_SNAIL, LAYOUT_SPIRAL, LAYOUT_SQUIRCLE, LENS_RADIUS_MAX,
    LENS_RADIUS_MIN,
    LENS_SEP_MAX, LENS_SEP_MIN,
    LOBE_SIZE_MAX, LOBE_SIZE_MIN, MOSS_ANGLE_MAX, MOSS_ANGLE_MIN, MOSS_SIZE_MAX, MOSS_SIZE_MIN,
    OBSTACLE_EFFECT_NAMES, OBST_BOUND_LEN, OBST_EFFECT_RGB, OBST_POS_MAX, OBST_POS_MIN,
    OBST_RADIUS_MAX, OBST_RADIUS_MIN, OBST_RING_LAYER_PX, POLY_MAX_SIDES, POLY_MIN_SIDES,
    REULEAUX_MAX_SIDES, REULEAUX_MIN_SIDES, SECTOR_SPAN_MAX, SNAIL_MAX_SEGMENTS, SNAIL_MIN_SEGMENTS,
    SPIRAL_FILL_MAX, SPIRAL_FILL_MIN,
    SPIRAL_PITCH_MAX, SPIRAL_PITCH_MIN, SPLIT_MAX, SPLIT_MIN, SQUARENESS_MAX, SQUARENESS_MIN,
    SUPER_EXPONENT_MAX, SUPER_EXPONENT_MIN, THUMB_BACKGROUND, THUMB_FILL
)
from .geometry import (
    ARC_FIELDS, arbelos_bound, capsule_bound, chain_arcs_xy, egg_bound, egg_outline_xy, ellipse_bound,
    fg_squircle_bound, flower_bound, flower_outline_xy,
    lens_bound, lobed_bound, lune_bound, moss_egg_bound, polygon_bound, reuleaux_bound,
    sample_boundary_xy, sector_bound, segment_bound, snail_bound, snail_outline_xy, spiral_bound,
    spiral_outline_xy,
    squircle_outline_xy, superellipse_bound
)
from .kernels import (
    OBSTACLE_EFFECT_REFLECT, ObstacleArrays, PIXEL, SHAPE_ARBELOS, SHAPE_CAPSULE, SHAPE_ELLIPSE,
    SHAPE_EGG, SHAPE_FG_SQUIRCLE, SHAPE_FLOWER, SHAPE_LENS, SHAPE_LUNE, SHAPE_MOSS_EGG, SHAPE_POLYGON, SHAPE_QUATREFOIL,
    SHAPE_REULEAUX, SHAPE_SECTOR, SHAPE_SEGMENT, SHAPE_SNAIL, SHAPE_SPIRAL, SHAPE_SUPERELLIPSE,
    SHAPE_TREFOIL, SIZE, WORLD_RECT_DEFAULT, egg_distance_grid, flower_distance_grid, inside_boundary_grid, snail_near_grid,
    spiral_distance_grid, squircle_distance_grid
)
from .settings import resolve_obstacles, strict_record


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
    SHORT_NAME = None       # what the figure-settings margin calls it, where NAME would not fit a column
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
        return [f"Obstacle #{index}: {self.SHORT_NAME or self.NAME}", self.info_param_line(),
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

    def outline_coords(self, world_rect=WORLD_RECT_DEFAULT, w=SIZE, h=SIZE):
        """Canvas coordinates (flat list) of a polygon tracing the shape, at its
        centre (polar position plus x/y offset), on a canvas w x h pixels that shows the world rectangle `world_rect`."""
        wx, wy = self.boundary_xy()
        ocx, ocy = self.center()
        xs = (wx + ocx - world_rect.x_start) * (w / (world_rect.x_stop - world_rect.x_start))
        ys = (world_rect.y_stop - (wy + ocy)) * (h / (world_rect.y_stop - world_rect.y_start))
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


class SuperellipseShape(Shape):
    KIND = SHAPE_SUPERELLIPSE
    LAYOUT = LAYOUT_SQUIRCLE
    KEY = "superelliptical_squircle"
    NAME = "Superelliptical squircle"
    SHORT_NAME = "Superellipse"
    PARAMS = (_slider("width", "Width", 1.0, ELLIPSE_MIN, ELLIPSE_MAX),
              _slider("height", "Height", 1.0, ELLIPSE_MIN, ELLIPSE_MAX),
              _slider("exponent", "Exponent", DEFAULT_SUPER_EXPONENT, SUPER_EXPONENT_MIN,
                      SUPER_EXPONENT_MAX, 0.05))
    __slots__ = ("width", "height", "exponent")

    def params(self):
        return (self.width, self.height, self.exponent, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return superellipse_bound(*params)

    def shape_text(self):
        return (f"Squircle {self.width:.2f}\u00d7{self.height:.2f} n{self.exponent:.2f}, "
                f"rot {self.rotation:.1f}\u00b0")

    def extent(self):
        a, b = self.width / 2.0, self.height / 2.0
        return max(a, b) if self.exponent < 2.0 else sqrt(a * a + b * b)

    def info_param_line(self):
        return f"  w{self.width:.2f} h{self.height:.2f} n{self.exponent:.2f}"

    def boundary_xy(self):
        return squircle_outline_xy(self.KIND, self.bound())


class FernandezGuastiShape(Shape):
    KIND = SHAPE_FG_SQUIRCLE
    LAYOUT = LAYOUT_SQUIRCLE
    KEY = "fernandez_guasti_squircle"
    NAME = "Fern\u00e1ndez-Guasti squircle"
    SHORT_NAME = "FG squircle"
    PARAMS = (RADIUS_PARAM,
              _slider("squareness", "Squareness", DEFAULT_SQUARENESS, SQUARENESS_MIN,
                      SQUARENESS_MAX))
    __slots__ = ("radius", "squareness")

    def params(self):
        return (self.radius, self.squareness, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return fg_squircle_bound(*params)

    def shape_text(self):
        return (f"Fern\u00e1ndez-Guasti r{self.radius:.2f} s{self.squareness:.2f}, "
                f"rot {self.rotation:.1f}\u00b0")

    def extent(self):
        # its corners are the farthest points, on the diagonals
        return self.radius * sqrt(2.0 / (1.0 + sqrt(1.0 - self.squareness ** 2)))

    def info_param_line(self):
        return f"  r{self.radius:.2f} sq {self.squareness:.2f}"

    def boundary_xy(self):
        return squircle_outline_xy(self.KIND, self.bound())


class ArchimedeanSpiralShape(Shape):
    """A ribbon winding out from the centre as an Archimedean spiral; `radius` is how far out the ribbon's
    outer edge reaches, `spiral_pitch` the gap from one turn to the next, `spiral_fill` how much of that gap
    the ribbon takes up (see spiral_bound)."""
    KIND = SHAPE_SPIRAL
    LAYOUT = LAYOUT_SPIRAL
    KEY = "archimedean_spiral"
    NAME = "Archimedean spiral"
    SHORT_NAME = "Spiral"
    PARAMS = (RADIUS_PARAM,
              _slider("spiral_pitch", "Pitch", DEFAULT_SPIRAL_PITCH, SPIRAL_PITCH_MIN,
                      SPIRAL_PITCH_MAX),
              _slider("spiral_fill", "Ribbon fill", DEFAULT_SPIRAL_FILL, SPIRAL_FILL_MIN,
                      SPIRAL_FILL_MAX))
    __slots__ = ("radius", "spiral_pitch", "spiral_fill")

    def params(self):
        return (self.radius, self.spiral_pitch, self.spiral_fill, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return spiral_bound(*params)

    def shape_text(self):
        return (f"Spiral r{self.radius:.2f} pitch {self.spiral_pitch:.2f} "
                f"fill {self.spiral_fill:.2f}, rot {self.rotation:.1f}\u00b0")

    def extent(self):
        return self.radius

    def info_param_line(self):
        return f"  r{self.radius:.2f} p{self.spiral_pitch:.2f} fill {self.spiral_fill:.2f}"

    def boundary_xy(self):
        return spiral_outline_xy(self.bound())


class PythagorasSnailShape(Shape):
    """The snail of Pythagoras (the spiral of Theodorus): right triangles with legs 1 and sqrt(k), laid round a common
    vertex, `snail_segments` of them (see snail_bound). Scaled so that its farthest corner is 1 from the centre."""
    KIND = SHAPE_SNAIL
    LAYOUT = LAYOUT_SNAIL
    KEY = "snail_of_pythagoras"
    NAME = "Snail of Pythagoras"
    SHORT_NAME = "Snail"
    PARAMS = (_count("snail_segments", "Segments", DEFAULT_SNAIL_SEGMENTS, SNAIL_MIN_SEGMENTS,
                     SNAIL_MAX_SEGMENTS),)
    __slots__ = ("snail_segments",)

    def params(self):
        return (self.snail_segments, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return snail_bound(*params)

    def shape_text(self):
        return f"Snail {self.snail_segments} segments, rot {self.rotation:.1f}\u00b0"

    def extent(self):
        return 1.0

    def info_param_line(self):
        return f"  segments {int(round(self.snail_segments))}"

    def boundary_xy(self):
        return snail_outline_xy(self.bound())


class HugelschafferEggShape(Shape):
    """A Hugelschaffer egg (the Newton transform of two circles): `egg_a` half its length, `egg_b` its half-width,
    `egg_shift` how far off the middle the second circle's centre is, as a fraction of `egg_a` (0 is an ellipse; the
    pointed end of the egg is at +x for a positive shift at rotation 0). See egg_bound."""
    KIND = SHAPE_EGG
    LAYOUT = LAYOUT_EGG
    KEY = "hugelschaffer_egg"
    NAME = "H\u00fcgelsch\u00e4ffer egg"
    SHORT_NAME = "Egg"
    PARAMS = (_slider("egg_a", "Half-length (a)", DEFAULT_EGG_A, 0.0, EGG_AXIS_MAX),
              _slider("egg_b", "Half-width (b)", DEFAULT_EGG_B, 0.0, EGG_AXIS_MAX),
              _slider("egg_shift", "Shift (d / a)", DEFAULT_EGG_SHIFT, EGG_SHIFT_MIN, EGG_SHIFT_MAX))
    __slots__ = ("egg_a", "egg_b", "egg_shift")

    def params(self):
        return (self.egg_a, self.egg_b, self.egg_shift, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return egg_bound(*params)

    def shape_text(self):
        return (f"H\u00fcgelsch\u00e4ffer egg a{self.egg_a:.2f} b{self.egg_b:.2f} "
                f"d/a {self.egg_shift:+.2f}, rot {self.rotation:.1f}\u00b0")

    def extent(self):
        return sqrt(self.egg_a ** 2 + self.egg_b ** 2)

    def info_param_line(self):
        return f"  a{self.egg_a:.2f} b{self.egg_b:.2f} d/a {self.egg_shift:+.2f}"

    def boundary_xy(self):
        return egg_outline_xy(self.bound())


class SineFlowerShape(Shape):
    """A circle whose radius is raised by a sine of the angle round its centre: rho = R (1 + A sin(k phi)), with
    `radius` R, `flower_petals` k (a whole number: the number of petals) and `flower_amplitude` A. See flower_bound."""
    KIND = SHAPE_FLOWER
    LAYOUT = LAYOUT_FLOWER
    KEY = "sine_flower"
    NAME = "Sine flower"
    PARAMS = (RADIUS_PARAM,
              _count("flower_petals", "Petals", DEFAULT_FLOWER_PETALS, FLOWER_MIN_PETALS,
                     FLOWER_MAX_PETALS),
              _slider("flower_amplitude", "Amplitude", DEFAULT_FLOWER_AMPLITUDE, FLOWER_AMPLITUDE_MIN,
                      FLOWER_AMPLITUDE_MAX))
    __slots__ = ("radius", "flower_petals", "flower_amplitude")

    def params(self):
        return (self.radius, self.flower_petals, self.flower_amplitude, self.rotation)

    @classmethod
    def bound_from_params(cls, params):
        return flower_bound(*params)

    def shape_text(self):
        return (f"Flower r{self.radius:.2f} {self.flower_petals} petals amp {self.flower_amplitude:.2f}, "
                f"rot {self.rotation:.1f}\u00b0")

    def extent(self):
        return self.radius * (1.0 + self.flower_amplitude)

    def info_param_line(self):
        return f"  r{self.radius:.2f} petals {int(round(self.flower_petals))} amp {self.flower_amplitude:.2f}"

    def boundary_xy(self):
        return flower_outline_xy(self.bound())


# Every kind, indexed by its SHAPE_* code (the order of the "new..." menu).
SHAPE_CLASSES = [EllipseShape, RegularPolygonShape, MossEggShape, CircularSectorShape,
                 CircularSegmentShape, LensShape, ReuleauxPolygonShape, CapsuleShape,
                 TrefoilShape, QuatrefoilShape, LuneShape, ArbelosShape,
                 SuperellipseShape, FernandezGuastiShape, ArchimedeanSpiralShape, PythagorasSnailShape,
                 HugelschafferEggShape, SineFlowerShape]
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


def ring_layer_masks(shape, bound, x, y, layer_px=OBST_RING_LAYER_PX, pixel=PIXEL):
    """(outer, inner) boolean masks on the world-coordinate grid (x, y) for a
    shape's two-layer outline: `outer` is the band just outside its edge and
    `inner` the band just inside it, each `layer_px` pixels of `pixel` world units thick (the world width of a pixel
    of the 512-pixel preview: PIXEL for the whole world, less for a framing that zooms in)."""
    w = layer_px * pixel
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
    if layout == LAYOUT_SQUIRCLE:
        near = squircle_distance_grid(x, y, shape, bound) < w
        return near & ~inside, near & inside
    if layout == LAYOUT_FLOWER:
        near = flower_distance_grid(x, y, bound) < w
        return near & ~inside, near & inside
    if layout == LAYOUT_EGG:
        near = egg_distance_grid(x, y, bound, w) < w
        return near & ~inside, near & inside
    if layout == LAYOUT_SNAIL:                          # (its edge is only part of its triangles' edges)
        near = snail_near_grid(x, y, bound, w)
        return near & ~inside, near & inside
    if layout == LAYOUT_SPIRAL:                         # (offsetting a ribbon's edges would run them into its neighbours')
        near = spiral_distance_grid(x, y, bound) < w
        return near & ~inside, near & inside
    bigger = inside_boundary_grid(x, y, shape, offset_bound(shape, bound, w))
    smaller = inside_boundary_grid(x, y, shape, offset_bound(shape, bound, -w))
    return bigger & ~inside, inside & ~smaller


def draw_obstacle_outline(img, shape, bound, x, y, in_effect, out_effect, pixel=PIXEL):
    """Draws an obstacle's two-layer outline into `img` (in place), on the
    world-coordinate grid (x, y) already shifted to be relative to the
    obstacle's centre: the outer layer in its incoming effect's color, the
    inner layer in its outgoing effect's (each ring_layer_masks' layer thick, in `pixel` world units)."""
    outer, inner = ring_layer_masks(shape, bound, x, y, pixel=pixel)
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
