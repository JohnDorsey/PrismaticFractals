"""The settings and their keyframes: the strict record types (`strict_record`); the records of settings - `RaySettings`
(the ray-bouncing engine's), `NaturalTerminationSettings` (the limits that end any sequence of points, whichever engine makes
it) and `BuddhaBlurSettings` (coloration's), side by side, none inside another; one table with a row per setting for
each record (`RAY_SETTING_SPECS`, `NATURAL_TERMINATION_SETTING_SPECS`, `BLUR_SETTING_SPECS`, listed in `SETTING_TABLES`) and
everything derived from them; the kernel argument tuples built from them (`fuzz_rule`, `confine_rule`, `emission_rule`,
`natural_termination_rule`, `ray_args`); and the interpolation between keyframes (`resolve_value`, `resolve_frame_settings`)."""
import dataclasses
from math import cos, radians, sin
from .constants import (
    BUDDHA_BLUR_LABELS, CONFINE_METHOD_NAMES, DEFAULT_BUDDHA_BLUR_RADIUS, DEFAULT_BUDDHA_SEED_CAMERA, DEFAULT_ENGINE,
    ENGINE_NAMES,
    DEFAULT_BUDDHA_SEED_X_START, DEFAULT_BUDDHA_SEED_X_STOP, DEFAULT_BUDDHA_SEED_Y_START,
    DEFAULT_BUDDHA_SEED_Y_STOP, DEFAULT_BUDDHA_BLUR_TYPE, DEFAULT_CONFINE_METHOD, DEFAULT_CONFINE_THRESHOLD,
    DEFAULT_CONFINE_LOOKBACK, DEFAULT_DIST, DEFAULT_ESCAPE_RADIUS, DEFAULT_EMIT_MODE, DEFAULT_EMIT_OFFSET,
    DEFAULT_EMIT_TARGET_CX, DEFAULT_EMIT_TARGET_CY, DEFAULT_EMIT_TARGET_R,
    DEFAULT_EMIT_TARGET_THETA, DEFAULT_RNG_SEED, DEFAULT_ITER, DEFAULT_SEGMENT_MULT,
    EMIT_MODE_NAMES
)
from .kernels import ConfineRule, EmissionRule, FuzzRule, NaturalTerminationRule, RayArgs


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
    rng_seed: int


DEFAULT_FUZZ = FuzzParams(enabled=False, emit_jitter=0.0, bnd_offset=0.0, bnd_jitter=0.0,
                          bnd_mult=1.0, self_offset=0.0, self_jitter=0.0, self_mult=1.0,
                          rng_seed=DEFAULT_RNG_SEED)


@strict_record
class BuddhaBlurSettings:
    """The blur applied to Buddha coloration's scattered points. Part of coloration, not of
    the simulation: it is its own record, held beside a RaySettings (never inside one) by
    everything that holds both - the controls, a keyframe's data, a project."""
    radius: float                                    # a multiple of the view's smaller dimension
    type: int                                        # index into BUDDHA_BLUR_LABELS


DEFAULT_BUDDHA_BLUR = BuddhaBlurSettings(radius=DEFAULT_BUDDHA_BLUR_RADIUS,
                                         type=DEFAULT_BUDDHA_BLUR_TYPE)


@strict_record
class EngineSettings:
    """Which path engine makes the figure (the Simulation tab's dropdown): a record of its own, held beside the ray
    engine's settings and the natural termination settings. Figure-defining: a different engine is a different figure
    (it needs Start). The engines' own settings are records of their own."""
    engine: int                                      # an ENGINE_* code, an index into ENGINE_NAMES


DEFAULT_ENGINE_SETTINGS = EngineSettings(engine=DEFAULT_ENGINE)


@strict_record
class BuddhaSeedSettings:
    """Where the rays that Buddha channels scatter start (their seed points). With `camera_pixels` they start at the
    centres of the pixels of the image being made, in world space - for the live image from rays it has already
    traced, so nothing is traced twice for the single-point options. Otherwise they start on a grid laid over the
    zone x_start..x_stop by y_start..y_stop (world units) with as many rows and columns as the image has pixels: so
    shrinking the zone makes the seeds closer together, never fewer. A record of its own, beside the others, and like
    the blur not part of what a generated image is made of."""
    camera_pixels: bool
    x_start: float
    x_stop: float
    y_start: float
    y_stop: float


DEFAULT_BUDDHA_SEED = BuddhaSeedSettings(
    camera_pixels=DEFAULT_BUDDHA_SEED_CAMERA, x_start=DEFAULT_BUDDHA_SEED_X_START,
    x_stop=DEFAULT_BUDDHA_SEED_X_STOP, y_start=DEFAULT_BUDDHA_SEED_Y_START,
    y_stop=DEFAULT_BUDDHA_SEED_Y_STOP)


@strict_record
class NaturalTerminationSettings:
    """What brings a sequence of points to its natural end, whichever engine makes the sequence (the ray-bouncing
    engine today; an escape-time iteration could take the same limits): how many points (`max_iter`), how long a path
    (`max_dist`), how far from the origin (`escape_radius`), and when it has stopped getting anywhere (the three
    confinement settings). It is a record of its own, held beside a RaySettings and a BuddhaBlurSettings (never inside
    either) by everything that holds them - the controls, a keyframe's data, a project - and what the Termination
    tab's "Termination limits" and "Confinement" sections show."""
    max_iter: int
    max_dist: float                                  # 0 = no limit; see effective_dist
    escape_radius: float                             # a sequence ends this far from the origin
    confine_lookback: int
    confine_method: int
    confine_threshold: float


DEFAULT_NATURAL_TERMINATION_SETTINGS = NaturalTerminationSettings(
    max_iter=DEFAULT_ITER, max_dist=DEFAULT_DIST, escape_radius=DEFAULT_ESCAPE_RADIUS,
    confine_lookback=DEFAULT_CONFINE_LOOKBACK, confine_method=DEFAULT_CONFINE_METHOD,
    confine_threshold=DEFAULT_CONFINE_THRESHOLD)


@strict_record
class RaySettings:
    """Every value of every control of the ray-bouncing engine itself (not the natural termination limits, which are a
    NaturalTerminationSettings of their own) - what determines the generated image, together with the natural termination settings and
    the obstacles (the scene, the obstacles and the colorations, is tracked separately). Start compares these, "Revert"
    restores them, and a keyframe resolves to one for each video frame. The Buddha blur is NOT in here either: it is a
    BuddhaBlurSettings of its own, used side by side with this."""
    seg_mult: float
    fuzz: FuzzParams
    emit_mode: int
    emit_target_r: float
    emit_target_theta: float
    emit_target_cx: float
    emit_target_cy: float
    emit_offset: float


DEFAULT_RAY_SETTINGS = RaySettings(
    seg_mult=DEFAULT_SEGMENT_MULT, fuzz=DEFAULT_FUZZ,
    emit_mode=DEFAULT_EMIT_MODE, emit_target_r=DEFAULT_EMIT_TARGET_R,
    emit_target_theta=DEFAULT_EMIT_TARGET_THETA, emit_target_cx=DEFAULT_EMIT_TARGET_CX,
    emit_target_cy=DEFAULT_EMIT_TARGET_CY, emit_offset=DEFAULT_EMIT_OFFSET)


# A resolved frame's complete settings (what render_full_image needs, aside from the
# colorations and obstacles - each resolved separately, the same way: a fixed set
# established when the render starts, each one's own fields independently
# held/animated from the keyframes). Built by resolve_frame_settings() from whichever
# keyframes define each setting. The ray engine's settings, the natural termination limits and the
# Buddha blur sit side by side, each its own record.
@strict_record
class KeyframeData:
    ray: RaySettings
    natural_termination: NaturalTerminationSettings
    buddha_blur: BuddhaBlurSettings
    buddha_seed: BuddhaSeedSettings
    engine: EngineSettings
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
    or coloration is deleted, with a prefix like "obst:3:" - the trailing colon
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
    obstacle's or coloration's settings."""
    return [Keyframe(kf.id, kf.frame_index, dict(kf.data)) for kf in keyframes]


@strict_record
class SettingEntry:
    """One row of the keyframe settings list: a setting (a static one, or one field of
    an obstacle or a coloration), where it is listed and what it falls back to."""
    key: str
    tab: str
    label: str
    is_float: bool
    fallback: object             # its value when no keyframe defines it


@strict_record
class SettingSpec:
    """What the program knows about one setting - one row per setting, so that adding a
    field to a settings record (RaySettings, BuddhaBlurSettings...) means adding one row
    to that record's table rather than touching a chain of places."""
    path: tuple            # where the value lives in its record, e.g. ("fuzz", "seed")
    control: str           # the App attribute holding the control that shows the value
    widget: str            # the App attribute of the widget its modified-outline surrounds
    kind: str              # how set_setting_control writes the control: "text", "int_text",
                           # "float_text", "number", "bool" or "choice"
    names: tuple           # for "choice": the names the value indexes into
    tab: str               # the keyframe settings list section it is listed under
    label: str             # its name there


def _spec(path, control, widget, kind, tab, label, names=()):
    return SettingSpec(path=path, control=control, widget=widget, kind=kind, names=names,
                       tab=tab, label=label)


# Every natural-termination setting (a NaturalTerminationSettings), in the order the keyframe settings list shows
# them within each tab. run_self_checks() makes sure these rows and the record's fields are the same set.
NATURAL_TERMINATION_SETTING_SPECS = {
    "iter": _spec(("max_iter",), "iter_var", "iter_entry", "text",
                  "Termination", "Iteration limit"),
    "dist": _spec(("max_dist",), "dist_var", "dist_entry", "text",
                  "Termination", "Distance limit"),
    "escape_radius": _spec(("escape_radius",), "escape_radius_var", "escape_radius_entry",
                           "float_text", "Termination", "Escape radius"),
    "confine_lookback": _spec(("confine_lookback",), "confine_lookback_var", "confine_lookback_entry",
                            "int_text", "Termination", "Confinement lookback"),
    "confine_method": _spec(("confine_method",), "confine_method_var", "confine_method_combo",
                            "choice", "Termination", "Confinement method", CONFINE_METHOD_NAMES),
    "confine_threshold": _spec(("confine_threshold",), "confine_threshold_var",
                               "confine_threshold_entry", "float_text", "Termination",
                               "Confinement threshold"),
}

# Every setting of the ray-bouncing engine itself (a RaySettings), the same way.
RAY_SETTING_SPECS = {
    "seg_mult": _spec(("seg_mult",), "seg_mult_scale", "seg_mult_scale", "number",
                      "Mutations", "Segment length multiplier"),
    "emit_mode": _spec(("emit_mode",), "emit_mode_var", "emit_mode_combo", "choice",
                       "Emissions", "Emission direction", EMIT_MODE_NAMES),
    "emit_target_r": _spec(("emit_target_r",), "emit_target_r_var", "emit_target_r_entry",
                           "float_text", "Emissions", "Emission target distance (r)"),
    "emit_target_theta": _spec(("emit_target_theta",), "emit_target_theta_var",
                               "emit_target_theta_entry", "float_text", "Emissions",
                               "Emission target angle (\u03b8)"),
    "emit_target_cx": _spec(("emit_target_cx",), "emit_target_cx_var", "emit_target_cx_entry",
                            "float_text", "Emissions", "Emission target offset X"),
    "emit_target_cy": _spec(("emit_target_cy",), "emit_target_cy_var", "emit_target_cy_entry",
                            "float_text", "Emissions", "Emission target offset Y"),
    "emit_offset": _spec(("emit_offset",), "emit_offset_scale", "emit_offset_scale", "number",
                         "Emissions", "Emission angle offset"),
    "fuzz_on": _spec(("fuzz", "enabled"), "fuzz_enabled_var", "fuzz_enable_check", "bool",
                     "Mutations", "Enabled"),
    "emit_jitter": _spec(("fuzz", "emit_jitter"), "emit_jitter_scale", "emit_jitter_scale",
                         "number", "Mutations", "Emission jitter"),
    "bnd_offset": _spec(("fuzz", "bnd_offset"), "bnd_offset_scale", "bnd_offset_scale",
                        "number", "Mutations", "Boundary offset"),
    "bnd_jitter": _spec(("fuzz", "bnd_jitter"), "bnd_jitter_scale", "bnd_jitter_scale",
                        "number", "Mutations", "Boundary jitter"),
    "bnd_mult": _spec(("fuzz", "bnd_mult"), "bnd_mult_scale", "bnd_mult_scale", "number",
                      "Mutations", "Boundary multiplier"),
    "self_offset": _spec(("fuzz", "self_offset"), "self_offset_scale", "self_offset_scale",
                         "number", "Mutations", "Self offset"),
    "self_jitter": _spec(("fuzz", "self_jitter"), "self_jitter_scale", "self_jitter_scale",
                         "number", "Mutations", "Self jitter"),
    "self_mult": _spec(("fuzz", "self_mult"), "self_mult_scale", "self_mult_scale", "number",
                       "Mutations", "Self multiplier"),
    "seed": _spec(("fuzz", "rng_seed"), "rng_seed_var", "rng_seed_entry", "text", "Mutations", "Seed"),
}

# The same for the Buddha blur (a BuddhaBlurSettings), which is a record of its own.
BLUR_SETTING_SPECS = {
    "buddha_blur_radius": _spec(("radius",), "buddha_blur_radius_scale",
                                "buddha_blur_radius_scale", "number", "Buddha blur", "Radius"),
    "buddha_blur_type": _spec(("type",), "buddha_blur_type_var",
                              "buddha_blur_type_combo", "choice", "Buddha blur", "Blur type",
                              BUDDHA_BLUR_LABELS),
}

# The same for the Buddha seeding (a BuddhaSeedSettings).
BUDDHA_SEED_SETTING_SPECS = {
    "buddha_seed_camera": _spec(("camera_pixels",), "buddha_seed_camera_var", "buddha_seed_camera_check",
                                "bool", "Buddha seeding", "Use camera pixels"),
    "buddha_seed_x_start": _spec(("x_start",), "buddha_seed_x_start_var", "buddha_seed_x_start_entry",
                                 "float_text", "Buddha seeding", "X start"),
    "buddha_seed_x_stop": _spec(("x_stop",), "buddha_seed_x_stop_var", "buddha_seed_x_stop_entry",
                                "float_text", "Buddha seeding", "X stop"),
    "buddha_seed_y_start": _spec(("y_start",), "buddha_seed_y_start_var", "buddha_seed_y_start_entry",
                                 "float_text", "Buddha seeding", "Y start"),
    "buddha_seed_y_stop": _spec(("y_stop",), "buddha_seed_y_stop_var", "buddha_seed_y_stop_entry",
                                "float_text", "Buddha seeding", "Y stop"),
}

# The same for the choice of path engine (an EngineSettings).
ENGINE_SETTING_SPECS = {
    "engine": _spec(("engine",), "engine_var", "engine_combo", "choice", "Simulation", "Engine", ENGINE_NAMES),
}

# Every record of settings and the table of its rows. This is the one place that says
# which records there are: a new kind of settings (say, those of another way to generate
# bounces) is a record class, a table like those above, and an entry here. Setting keys
# are unique across all the tables (they are what keyframes and project files call a
# setting), and run_self_checks() checks each table against its record's fields.
SETTING_TABLES = {NaturalTerminationSettings: NATURAL_TERMINATION_SETTING_SPECS, RaySettings: RAY_SETTING_SPECS,
                  BuddhaBlurSettings: BLUR_SETTING_SPECS, BuddhaSeedSettings: BUDDHA_SEED_SETTING_SPECS,
                  EngineSettings: ENGINE_SETTING_SPECS}
DEFAULT_SETTINGS = {NaturalTerminationSettings: DEFAULT_NATURAL_TERMINATION_SETTINGS, RaySettings: DEFAULT_RAY_SETTINGS,
                    BuddhaBlurSettings: DEFAULT_BUDDHA_BLUR, BuddhaSeedSettings: DEFAULT_BUDDHA_SEED,
                    EngineSettings: DEFAULT_ENGINE_SETTINGS}

# Every static setting a keyframe can define, from all the tables, in table order.
SETTING_SPECS = {key: spec for table in SETTING_TABLES.values() for key, spec in table.items()}

# Every setting a keyframe can define: (key, tab it lives on, short label for the
# settings list), in the order shown within each tab group - the SETTING_SPECS rows,
# then the one decoration that is not part of any record of settings.
SETTING_INFO = ([(key, spec.tab, spec.label) for key, spec in SETTING_SPECS.items()]
                + [("outline", "Decorations", "Outline")])
SETTING_TABS_ORDER = ["Emissions", "Termination", "Mutations", "Buddha seeding", "Buddha blur", "Decorations", "Simulation"]
SETTING_LABEL = {key: label for key, _tab, label in SETTING_INFO}
SETTING_TAB = {key: tab for key, tab, _label in SETTING_INFO}


def _field_types(record_class):
    return {f.name: f.type for f in dataclasses.fields(record_class)}


def record_leaf_paths(record_class, prefix=()):
    """Every value a settings record holds, as a path: for RaySettings ("seg_mult",),
    ("fuzz", "seed"), ...; for BuddhaBlurSettings ("radius",), ("type",)."""
    paths = []
    for f in dataclasses.fields(record_class):
        if dataclasses.is_dataclass(f.type):
            paths += record_leaf_paths(f.type, prefix + (f.name,))
        else:
            paths.append(prefix + (f.name,))
    return paths


def settings_to_dict(*records):
    """{setting key: value} for every row of the table of each record given (a RaySettings,
    a NaturalTerminationSettings, a BuddhaBlurSettings, ...), read from the records themselves."""
    values = {}
    for record in records:
        for key, spec in SETTING_TABLES[type(record)].items():
            value = record
            for name in spec.path:
                value = getattr(value, name)
            values[key] = value
    return values


def settings_from_dict(record_class, values):
    """A record of `record_class` (RaySettings, NaturalTerminationSettings...) from {setting key:
    value} (for example the values resolved from keyframes, where an int setting may arrive
    as a float; keys of other records are ignored): each value is converted to its field's
    type - int fields rounded, bool fields made bool, anything else kept as it is (the
    record then checks it)."""
    top, nested = {}, {}
    for key, spec in SETTING_TABLES[record_class].items():
        *outer, leaf = spec.path
        assert len(outer) <= 1, spec.path
        owner = record_class
        for name in outer:
            owner = _field_types(owner)[name]
        want = _field_types(owner)[leaf]
        value = values[key]
        if want is int:
            value = int(round(value))
        elif want is bool:
            value = bool(value)
        (nested.setdefault(outer[0], {}) if outer else top)[leaf] = value
    for name, fields in nested.items():
        top[name] = _field_types(record_class)[name](**fields)
    return record_class(**top)


# Only the float settings are ever smoothly interpolated; everything else (which
# shape, colors, metrics, sides, flags, seed) always holds the value of the nearest
# earlier keyframe that defines it. Derived: a float field of a settings record is a
# float setting.
def _is_float_setting(record_class, spec):
    owner = record_class
    for name in spec.path[:-1]:
        owner = _field_types(owner)[name]
    return _field_types(owner)[spec.path[-1]] is float


SETTING_FLOAT_KEYS = {key for record_class, table in SETTING_TABLES.items()
                      for key, spec in table.items() if _is_float_setting(record_class, spec)}

# Which attribute of App holds the main control of each SETTING_INFO setting (a
# widget, or a handle whose `.widget` is the entry) - where its blue "modified"
# outline is drawn. run_self_checks() keeps this in step with SETTING_INFO.
SETTING_WIDGET_ATTRS = {**{key: spec.widget for key, spec in SETTING_SPECS.items()},
                        "outline": "deco_outline_check"}

DEFAULT_SETTING_VALUES = {**settings_to_dict(*DEFAULT_SETTINGS.values()), "outline": True}


def confine_rule(lookback, method, threshold):
    """The confinement rule in the form the simulation kernels expect."""
    return ConfineRule(lookback=int(lookback), method=int(method), threshold=float(threshold))


# What was always hard-wired (a 4-point lookback, the Box method, one pixel).
CONFINE_DEFAULT = confine_rule(DEFAULT_CONFINE_LOOKBACK, DEFAULT_CONFINE_METHOD,
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
    """confine_rule() for a NaturalTerminationSettings."""
    return confine_rule(settings.confine_lookback, settings.confine_method,
                        settings.confine_threshold)


def effective_dist(max_dist):
    """The distance limit passed to the simulation: infinite when 0 (no limit)."""
    return float("inf") if max_dist == 0 else float(max_dist)


def natural_termination_rule(max_iter, max_dist, escape_radius, confine):
    """The natural termination rule in the form the simulation kernels expect: `max_dist` as the kernels read it
    (infinite for no limit - see effective_dist) and `confine` a confine_rule."""
    return NaturalTerminationRule(max_iter=int(max_iter), max_dist=float(max_dist),
                           escape_radius=float(escape_radius), confine=confine)


def settings_natural_termination(settings):
    """natural_stop_rule() for a NaturalTerminationSettings."""
    return natural_termination_rule(settings.max_iter, effective_dist(settings.max_dist),
                             settings.escape_radius, settings_confine(settings))


# What was always hard-wired, plus the defaults of the other limits.
NATURAL_TERMINATION_DEFAULT = settings_natural_termination(DEFAULT_NATURAL_TERMINATION_SETTINGS)


def ray_args(seg_mult, fuzz, emission, obstacles):
    """The ray-bouncing engine's own arguments in the form the simulation kernels expect: `fuzz` a fuzz_rule,
    `emission` an emission_rule and `obstacles` an ObstacleArrays (see build_obstacle_arrays)."""
    return RayArgs(seg_mult=float(seg_mult), fuzz=fuzz, emission=emission, obstacles=obstacles)


def settings_ray_args(settings, obstacles):
    """ray_args() for a RaySettings and the obstacles' ObstacleArrays. (The fuzzing seed is not part of
    it: callers pass `settings.fuzz.rng_seed` to the kernels themselves.)"""
    return ray_args(settings.seg_mult, fuzz_rule(settings.fuzz), settings_emission(settings), obstacles)


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


def resolve_setting(key, keyframes_sorted, frame_number, base):
    """resolve_value() for one of the static settings in SETTING_INFO. `base` ({setting key: value}, every static
    setting) holds the value of a setting that no keyframe defines: what the controls held when the export began (never
    the program's defaults - a missing key is a KeyError, not a silent default)."""
    return resolve_value(key, keyframes_sorted, frame_number, base[key], key in SETTING_FLOAT_KEYS)


def resolve_frame_settings(keyframes_sorted, frame_number, base):
    """A full KeyframeData for one video frame - the ray engine's settings, the natural termination
    limits and the Buddha blur, side by side - with every setting resolved independently via
    resolve_setting(). `base` is the value of every static setting as the controls held it when the export began
    (App.capture_all_settings; its dynamic keys are ignored): a setting no keyframe defines keeps that value at every
    frame, as an obstacle, coloration or framing no keyframe animates keeps its own value (resolve_obstacles,
    resolve_colorations, resolve_framing). The colorations are not part of this - see resolve_colorations(),
    called separately (mirroring resolve_obstacle_arrays), since colorations are a dynamic list
    like obstacles, not a fixed set of settings."""
    v = {key: resolve_setting(key, keyframes_sorted, frame_number, base)
        for key, _tab, _label in SETTING_INFO}
    return KeyframeData(ray=settings_from_dict(RaySettings, v),
                        natural_termination=settings_from_dict(NaturalTerminationSettings, v),
                        buddha_blur=settings_from_dict(BuddhaBlurSettings, v),
                        buddha_seed=settings_from_dict(BuddhaSeedSettings, v),
                        engine=settings_from_dict(EngineSettings, v),
                        outline=bool(v["outline"]))


def fuzz_rule(f):
    """FuzzParams -> the FuzzRule the simulation kernels expect (the seed is handled
    separately, by seeding the RNG before each ray is traced)."""
    return FuzzRule(enabled=bool(f.enabled), emit_jitter=float(f.emit_jitter),
                    bnd_offset=float(f.bnd_offset), bnd_jitter=float(f.bnd_jitter),
                    bnd_mult=float(f.bnd_mult), self_offset=float(f.self_offset),
                    self_jitter=float(f.self_jitter), self_mult=float(f.self_mult))
