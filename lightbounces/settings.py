"""The simulation settings and their keyframes: the strict record types (`strict_record`), `RaySettings` and what it holds,
the one table with a row per setting (`RAY_SETTING_SPECS`) and everything derived from it, the kernel rule
tuples built from the settings (`fuzz_rule`, `confine_rule`, `emission_rule`), and the interpolation between
keyframes (`resolve_value`, `resolve_frame_settings`)."""
import dataclasses
from math import cos, radians, sin
from .constants import (
    BUDDHA_BLUR_LABELS, CONFINE_METHOD_NAMES, DEFAULT_BUDDHA_BLUR_RADIUS,
    DEFAULT_BUDDHA_BLUR_TYPE, DEFAULT_CONFINE_METHOD, DEFAULT_CONFINE_THRESHOLD,
    DEFAULT_CONFINE_WINDOW, DEFAULT_DIST, DEFAULT_EMIT_MODE, DEFAULT_EMIT_OFFSET,
    DEFAULT_EMIT_TARGET_CX, DEFAULT_EMIT_TARGET_CY, DEFAULT_EMIT_TARGET_R,
    DEFAULT_EMIT_TARGET_THETA, DEFAULT_FUZZ_SEED, DEFAULT_ITER, DEFAULT_SEGMENT_MULT,
    EMIT_MODE_NAMES
)
from .kernels import ConfineRule, EmissionRule, FuzzRule


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
