"""Projects: saving and loading everything the user has set up.

A project file is pretty-printed, versioned JSON (PROJECT_EXTENSION). It holds the controls' simulation settings (the ray engine's and the natural termination limits, each its own section), the Buddha blur, the
decorations, every obstacle, coloration and keyframe, and the Export designer and video panel choices - not the generated image, which Start
makes again (fuzzing is seeded, so the same settings give the same image).

Every choice in the file is a readable string KEY ("euclidean_distance", "reflect", "lune"), never a number. A key is permanent: once a
project file can contain it, it is never renamed or reused, and a Catalog translates it to the program's own code (an int the kernels
index arrays with, which may be renumbered freely). So the program is free to reorder, insert and renumber its codes;
test_the_keys_in_project_files_never_change pins the keys. Object ids (obstacles, colorations, keyframes) stay the whole numbers they are.
Older files are converted one format version at a time by PROJECT_MIGRATIONS."""
import json
import math
import re
from .constants import (
    BUDDHA_BLUR_BOKEH, BUDDHA_BLUR_CATADIOPTRIC, BUDDHA_BLUR_GAUSSIAN, BUDDHA_BLUR_TRIANGULAR, ENGINE_NAMES, OFFERED_ENGINES,
    ORBIT_ESCAPE_BEYOND, ORBIT_ESCAPE_CLIP, ORBIT_MAPPING_JULIA, ORBIT_MAPPING_MANDELBROT, ORBIT_Z0_C, ORBIT_Z0_ZERO
)
from .export_text import (
    COLOR_INFO_LOCATIONS, FIGURE_INFO_LOCATIONS, FRAMING_INFO_LOCATIONS, SAVE_SOURCE_OPTIONS, VIDEO_BITRATE_MODES,
    VIDEO_FORMATS
)
from .image_io import DOWNSAMPLE_FACTORS, DOWNSAMPLE_METHODS, JXL_MODES, SAVE_NEW_FORMATS
from .kernels import (
    CONFINE_BOX, CONFINE_PATH, EMIT_TOWARDS, EMIT_UP, ENGINE_MANDELBROT, ENGINE_RAY_BOUNCING, N_PRED, N_SEQS, N_SRED, OBSTACLE_EFFECT_PASS,
    OBSTACLE_EFFECT_REFLECT, OBSTACLE_EFFECT_TERMINATE, PRED_BOTTOM, PRED_FIRST, PRED_LAST, PRED_LEFT, PRED_MEAN,
    PRED_RIGHT, PRED_TOP, SEQ_BOUNCES, SEQ_BOUNDARY, SEQ_PATH, SEQ_SELF, SRED_COUNT, SRED_LAST_DIR, SRED_LENGTH,
    SRED_SHARE, ST_ANGLES, ST_EUCLIDEAN, ST_HDIST, ST_LAST_RATIO, ST_LOOP_AREA, ST_MANHATTAN, ST_TERMINATION,
    ST_VDIST, TERM_CONFINED, TERM_DIST, TERM_PERPENDICULAR
)
from .metrics import (
    BUDDHA_COMPRESS_ARCTAN, BUDDHA_COMPRESS_MODULUS, BUDDHA_MULTI, BUDDHA_MULTI_ALL_MID, BUDDHA_MULTI_ALL_PATH,
    BUDDHA_MULTI_ALL_SELF, BUDDHA_MULTI_PROG_BOUNCE, BUDDHA_MULTI_PROG_MID, BUDDHA_POINT_CATALOG, CHAN_SRC_BUDDHA,
    CHAN_SRC_CONSTANT, CHAN_SRC_QUERY, METRIC_TYPE_OTHER, METRIC_TYPE_POINT_PAIR, NONE_ONE, NONE_ZERO, N_BUDDHA_POINTS,
    OTHER_METRICS, CHOICE_ORIGIN, PATH_KIND_ESCAPING, PATH_KIND_LABELS, PATH_KIND_NON_ESCAPING,
    PATH_KIND_TERMINATED_BY_OBSTACLE, POINT_METRICS, CHOICE_POINT_NEG1_POS1, CHOICE_TOP_LEFT_POINT, point_slot, stat_slot
)
from .settings import (
    BuddhaBlurSettings, BuddhaSeedSettings, DEFAULT_SETTING_VALUES, EngineSettings, Keyframe, MandelbrotSettings, NaturalTerminationSettings,
    RaySettings, SETTING_TABLES,
    settings_from_dict, settings_to_dict, strict_record
)
from .shapes import SHAPE_CLASSES
from .version import FILE_VERSION
from .colorations import ChannelSettings, COLOR_MODELS, ColorationConfig, snapshot_coloration, coloration_channel_value
from .framings import (
    DEFAULT_FRAMING, FRAMING_FIELDS, FramingConfig, framing_is_usable, snapshot_framing
)


PROJECT_EXTENSION = ".lightbounce"
PROJECT_FORMAT_NAME = "light-bounces-project"
PROJECT_FORMAT_VERSION = 9
PROJECT_FILE_TYPES = [("Prismatic Fractals project", f"*{PROJECT_EXTENSION}")]
# A key is lower-case words joined by underscores, or the coordinates of a fixed point, "(-1,1)": a point that is named
# by where it is can never change what it means (a different point is a different key).
PROJECT_KEY_PATTERN = re.compile(r"[a-z0-9_]+|\(-?\d+,-?\d+\)")


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
CAT_METRIC_TYPE = Catalog("channel kind", [("point_metric", METRIC_TYPE_POINT_PAIR), ("other_metric", METRIC_TYPE_OTHER)])
# The named sequences of points, and the reductions of them a channel can use (what the engine stores for
# each ray). A point, a metric of a sequence and a Buddha point are each named by the sequence's key and the
# reduction's: "self_bounces_uppermost", "path_length", "path_last_confined".
CAT_SEQUENCE = Catalog("sequence", [("path", SEQ_PATH), ("bounces", SEQ_BOUNCES),
                                    ("boundary_bounces", SEQ_BOUNDARY), ("self_bounces", SEQ_SELF)])
CAT_POINT_REDUCTION = Catalog("point reduction", [
    ("first", PRED_FIRST), ("last", PRED_LAST), ("mean", PRED_MEAN), ("uppermost", PRED_TOP),
    ("lowermost", PRED_BOTTOM), ("rightmost", PRED_RIGHT), ("leftmost", PRED_LEFT)])
CAT_STAT_REDUCTION = Catalog("scalar reduction", [
    ("count", SRED_COUNT), ("length", SRED_LENGTH), ("last_direction", SRED_LAST_DIR), ("share", SRED_SHARE)])
CAT_METRIC = Catalog("metric", [          # the measurements of a ray's own run, then each sequence's reductions
    ("horizontal_distance", ST_HDIST), ("vertical_distance", ST_VDIST),
    ("manhattan_distance", ST_MANHATTAN), ("euclidean_distance", ST_EUCLIDEAN),
    ("loop_area", ST_LOOP_AREA), ("mean_bounce_angle", ST_ANGLES), ("termination_reason", ST_TERMINATION),
    ("last_segment_ratio", ST_LAST_RATIO)]
    + [(f"{CAT_SEQUENCE.key(seq)}_{CAT_STAT_REDUCTION.key(red)}", stat_slot(seq, red))
       for seq in range(N_SEQS) for red in range(N_SRED)])
CAT_POINT = Catalog("point", [
    (f"{CAT_SEQUENCE.key(seq)}_{CAT_POINT_REDUCTION.key(red)}", point_slot(seq, red))
    for seq in range(N_SEQS) for red in range(N_PRED)]
    + [("origin", CHOICE_ORIGIN), ("(-1,1)", CHOICE_POINT_NEG1_POS1), ("(-2,2)", CHOICE_TOP_LEFT_POINT)])
_BUDDHA_TERM_KEYS = {TERM_DIST: "distance_limit", TERM_PERPENDICULAR: "perpendicular_collision",
                     TERM_CONFINED: "confined"}
_BUDDHA_MULTI_KEYS = {BUDDHA_MULTI_ALL_SELF: "all_self_bounces", BUDDHA_MULTI_ALL_MID: "all_segment_midpoints",
                      BUDDHA_MULTI_PROG_BOUNCE: "progressive_means_of_bounce_points",
                      BUDDHA_MULTI_PROG_MID: "progressive_means_of_segment_midpoints",
                      BUDDHA_MULTI_ALL_PATH: "all_path_points"}


def _buddha_point_key(style, ref, term_filter):
    """The key of one entry of BUDDHA_POINT_CATALOG: a stored point's own key, with the limit that ended the
    rays it is limited to; or the multi option's."""
    if style == BUDDHA_MULTI:
        return _BUDDHA_MULTI_KEYS[ref]
    key = CAT_POINT.key(ref)
    return key if term_filter is None else f"{key}_{_BUDDHA_TERM_KEYS[term_filter]}"


CAT_BUDDHA_POINT = Catalog("Buddha point", [          # the code is the place in BUDDHA_POINT_CATALOG
    (_buddha_point_key(style, ref, term_filter), index)
    for index, (_label, style, ref, term_filter) in enumerate(BUDDHA_POINT_CATALOG)])
CAT_PATH_KIND = Catalog("kind of path", [          # which kinds of path a Buddha channel scatters (a list of these keys)
    ("escaping", PATH_KIND_ESCAPING), ("non_escaping", PATH_KIND_NON_ESCAPING),
    ("terminated_by_obstacle", PATH_KIND_TERMINATED_BY_OBSTACLE)])
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
CAT_ENGINE = Catalog("path engine", [("ray_bouncing", ENGINE_RAY_BOUNCING), ("mandelbrot", ENGINE_MANDELBROT)])
CAT_ORBIT_Z0 = Catalog("orbit start", [("zero", ORBIT_Z0_ZERO), ("c", ORBIT_Z0_C)])
CAT_ORBIT_MAPPING = Catalog("orbit pixel mapping", [("mandelbrot", ORBIT_MAPPING_MANDELBROT), ("julia", ORBIT_MAPPING_JULIA)])
CAT_ORBIT_ESCAPE = Catalog("orbit escape behaviour", [("clip", ORBIT_ESCAPE_CLIP), ("beyond", ORBIT_ESCAPE_BEYOND)])
CAT_EFFECT = Catalog("ray effect", [
    ("reflect", OBSTACLE_EFFECT_REFLECT), ("terminate", OBSTACLE_EFFECT_TERMINATE),
    ("pass_through", OBSTACLE_EFFECT_PASS)])
CAT_SHAPE_KIND = Catalog("obstacle kind", [(cls.KEY, cls) for cls in SHAPE_CLASSES])
# The settings (of any settings record) that are choices, by setting key
SETTING_CHOICE_CATALOGS = {"emit_mode": CAT_EMIT_MODE, "confine_method": CAT_CONFINE_METHOD,
                           "buddha_blur_type": CAT_BLUR_TYPE, "engine": CAT_ENGINE, "orbit_z0": CAT_ORBIT_Z0,
                           "orbit_mapping": CAT_ORBIT_MAPPING, "orbit_escape": CAT_ORBIT_ESCAPE}
# Choices made in the Export designer and video panel
CAT_COLOR_INFO = _string_catalog("colour info location", ["none", "overlay", "bottom_margin"],
                                 COLOR_INFO_LOCATIONS)
CAT_FIGURE_INFO = _string_catalog("figure info location", ["none", "left_margin"],
                                  FIGURE_INFO_LOCATIONS)
CAT_FRAMING_INFO = _string_catalog("framing info location", ["none", "left_margin"],
                                   FRAMING_INFO_LOCATIONS)
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
    ("include_all_colorations", "include_all_colorations_var", "bool", None),
    ("color_info_location", "color_info_location_var", "choice", CAT_COLOR_INFO),
    ("figure_info_location", "figure_info_location_var", "choice", CAT_FIGURE_INFO),
    ("framing_info_location", "framing_info_location_var", "choice", CAT_FRAMING_INFO),
    ("image_source", "save_source_var", "choice", CAT_IMAGE_SOURCE),
    ("image_format", "save_format_var", "choice", CAT_IMAGE_FORMAT),
    ("image_jxl_mode", "save_jxl_mode_var", "choice", CAT_JXL_MODE),
    ("image_jxl_quality", "save_jxl_quality_scale", "number", None),
    ("image_width", "save_new_w_var", "text", None),
    ("image_height", "save_new_h_var", "text", None),
    ("video_fps", "fps_var", "text", None),
    ("video_width", "video_w_var", "text", None),
    ("video_height", "video_h_var", "text", None),
    ("video_start_frame", "start_frame_var", "text", None),
    ("video_number_files_from_start", "number_files_from_start_var", "bool", None),
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
    """Everything a project file holds, as plain values (the obstacles, colorations and keyframes
    are independent frozen copies). Made by App.capture_project or project_from_dict."""
    ray: RaySettings                    # the ray-bouncing engine's controls
    natural_termination: NaturalTerminationSettings   # the limits that end a sequence of points, beside them
    buddha_blur: BuddhaBlurSettings     # the Buddha blur's, beside them (not inside)
    buddha_seed: BuddhaSeedSettings     # where the rays Buddha channels scatter start, beside them
    engine: EngineSettings              # which path engine makes the figure, beside them
    mandelbrot: MandelbrotSettings      # the Mandelbrot engine's own settings, beside them (kept whichever engine is selected)
    decorations: dict                   # decoration key -> bool
    obstacles: tuple                    # frozen Shapes, in list order
    colorations: tuple                  # ColorationSnapshots, in list order
    framings: tuple                     # FramingSnapshots, in list order
    keyframes: tuple                    # Keyframes, in list order
    export: dict                        # EXPORT_OPTIONS key -> the control's value
    obstacle_counter: int               # the next new obstacle's / coloration's / framing's / keyframe's id
    coloration_counter: int
    framing_counter: int
    keyframe_counter: int
    selected_obstacle_id: int           # -1 for none
    selected_coloration_id: int
    selected_framing_id: int


def coloration_config_from_snapshot(snapshot):
    """A live ColorationConfig with the contents of a ColorationSnapshot."""
    coloration = ColorationConfig(snapshot.id)
    coloration.color_model = snapshot.color_model
    for c in range(3):
        coloration.set_channel(c, coloration_channel_value(snapshot, c))
    return coloration


def project_signature(project):
    """What a project's content amounts to, comparable with == : exactly what its file would say,
    apart from the program version, the id counters and which obstacle or coloration is selected. Used
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
        out["kind"] = CAT_METRIC_TYPE.key(channel.metric_type)
        out["metric"] = CAT_METRIC.key(channel.metric_slot)
        if channel.metric_type == METRIC_TYPE_POINT_PAIR:
            out["point1"] = CAT_POINT.key(channel.point1)
            out["point2"] = CAT_POINT.key(channel.point2)
    elif channel.chan_src == CHAN_SRC_BUDDHA:
        out["buddha_points"] = [CAT_BUDDHA_POINT.key(i)
                                for i, chosen in enumerate(channel.buddha_points) if chosen]
        out["buddha_path_kinds"] = [CAT_PATH_KIND.key(k) for k, on in enumerate(channel.buddha_path_kinds) if on]
        out["buddha_compression"] = CAT_BUDDHA_COMPRESSION.key(channel.buddha_compression)
        out["buddha_brightness"] = CAT_BUDDHA_BRIGHTNESS.key(channel.buddha_brightness)
    else:
        out["constant"] = CAT_CONSTANT.key(channel.constant)
    return out


def _setting_catalog(key):
    """The Catalog for the values of a simulation or keyframe setting `key` that is a choice."""
    if key in SETTING_CHOICE_CATALOGS:
        return SETTING_CHOICE_CATALOGS[key]
    if key.startswith("obst:") and key.rsplit(":", 1)[-1] in ("in_effect", "out_effect"):
        return CAT_EFFECT
    return None


def _setting_to_json(key, value):
    if isinstance(value, ChannelSettings):
        return _channel_to_json(value)
    catalog = _setting_catalog(key)
    return catalog.key(value) if catalog is not None else value


def _settings_to_json(record):
    """A settings record (a RaySettings, a NaturalTerminationSettings...) as the file's section for it:
    {setting key: value}, choices as their keys."""
    return {key: _setting_to_json(key, value) for key, value in settings_to_dict(record).items()}


def _settings_from_section(data, section, record_class, where):
    """The record of `record_class` from the file's section `section`: the settings the file
    has (those the program does not know are ignored), the program's defaults for the rest."""
    table = SETTING_TABLES[record_class]
    in_file = {key: value for key, value in (data.get(section) or {}).items() if key in table}
    values = {key: DEFAULT_SETTING_VALUES[key] for key in table}
    values.update({key: _setting_from_json(key, value, where) for key, value in in_file.items()})
    return settings_from_dict(record_class, values)


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
        "simulation": _settings_to_json(project.ray),
        "natural_termination": _settings_to_json(project.natural_termination),
        "buddha_blur": _settings_to_json(project.buddha_blur),
        "buddha_seeding": _settings_to_json(project.buddha_seed),
        "engine": _settings_to_json(project.engine),
        "mandelbrot": _settings_to_json(project.mandelbrot),
        "decorations": dict(project.decorations),
        "obstacles": [{"kind": type(o).KEY, "id": o.id,
                       "fields": {name: _setting_to_json(f"obst:{o.id}:{name}", getattr(o, name))
                                  for name in o.FIELD_NAMES}}
                      for o in project.obstacles],
        "colorations": [{"id": v.id, "color_model": v.color_model,
                         "channels": [_channel_to_json(coloration_channel_value(v, c)) for c in range(3)]}
                        for v in project.colorations],
        "framings": [{"id": f.id, **{field: getattr(f, field) for field in FRAMING_FIELDS}}
                     for f in project.framings],
        "keyframes": [{"id": kf.id, "frame_index": kf.frame_index,
                       "settings": {key: _setting_to_json(key, value)
                                    for key, value in kf.data.items()}}
                      for kf in project.keyframes],
        "export": export,
        "counters": {"obstacle": project.obstacle_counter, "coloration": project.coloration_counter,
                     "framing": project.framing_counter, "keyframe": project.keyframe_counter},
        "selected": {"obstacle": (project.selected_obstacle_id
                                  if project.selected_obstacle_id >= 0 else None),
                     "coloration": project.selected_coloration_id, "framing": project.selected_framing_id},
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
    default = coloration_channel_value(ColorationConfig(0), c)
    source = CAT_CHAN_SOURCE.code(data["source"], where)
    values = {"chan_src": source, "metric_type": default.metric_type, "metric_slot": default.metric_slot,
              "point1": default.point1, "point2": default.point2,
              "buddha_points": default.buddha_points, "buddha_path_kinds": default.buddha_path_kinds,
              "buddha_compression": default.buddha_compression,
              "buddha_brightness": default.buddha_brightness, "constant": default.constant}
    for name, field, catalog in (("kind", "metric_type", CAT_METRIC_TYPE), ("metric", "metric_slot", CAT_METRIC),
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
    if "buddha_path_kinds" in data:                  # (a file without it: every kind, as before channels could be limited)
        ticked = data["buddha_path_kinds"]
        _expect(isinstance(ticked, list), f"{where}: buddha_path_kinds must be a list")
        kinds = {CAT_PATH_KIND.code(key, where) for key in ticked}
        values["buddha_path_kinds"] = tuple(kind in kinds for kind in range(len(PATH_KIND_LABELS)))
    needed = {CHAN_SRC_QUERY: ["kind", "metric"] + (["point1", "point2"]
                                                    if values["metric_type"] == METRIC_TYPE_POINT_PAIR else []),
              CHAN_SRC_BUDDHA: ["buddha_points", "buddha_compression", "buddha_brightness"],
              CHAN_SRC_CONSTANT: ["constant"]}[source]
    for name in needed:
        _expect(name in data, f"{where}: the channel has no {name!r}")
    if source == CHAN_SRC_QUERY:
        _expect(values["metric_slot"] in (POINT_METRICS if values["metric_type"] == METRIC_TYPE_POINT_PAIR else OTHER_METRICS),
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


def _coloration_from_json(data, seen_ids):
    _expect(isinstance(data, dict), "a coloration must be an object")
    coloration_id, color_model, channels = data.get("id"), data.get("color_model"), data.get("channels")
    _expect(isinstance(coloration_id, int) and not isinstance(coloration_id, bool) and coloration_id >= 0,
            "a coloration's id must be a whole number")
    _expect(coloration_id not in seen_ids, f"two colorations have the id {coloration_id}")
    seen_ids.add(coloration_id)
    _expect(color_model in COLOR_MODELS, f"coloration {coloration_id}: unknown colour model {color_model!r}")
    _expect(isinstance(channels, list) and len(channels) == 3,
            f"coloration {coloration_id}: it needs exactly 3 channels")
    coloration = ColorationConfig(coloration_id)
    coloration.color_model = color_model
    for c, channel in enumerate(channels):
        coloration.set_channel(c, _channel_from_json(channel, f"coloration {coloration_id} channel {c + 1}", c))
    return snapshot_coloration(coloration)


def _framing_from_json(data, seen_ids):
    _expect(isinstance(data, dict), "a framing must be an object")
    framing_id = data.get("id")
    _expect(isinstance(framing_id, int) and not isinstance(framing_id, bool) and framing_id >= 0,
            "a framing's id must be a whole number")
    _expect(framing_id not in seen_ids, f"two framings have the id {framing_id}")
    seen_ids.add(framing_id)
    framing = FramingConfig(framing_id)
    for field in FRAMING_FIELDS:
        value = data.get(field, getattr(framing, field))
        _expect(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                f"framing {framing_id}: {field} must be a number")
        setattr(framing, field, float(value))
    _expect(framing.diagonal > 0.0 and framing.aspect > 0.0 and framing_is_usable(framing),
            f"framing {framing_id}: its diagonal and aspect ratio must be above 0 and leave it a rectangle of the world to see")
    return snapshot_framing(framing)


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
        if key.startswith("coloration:") and ":chan" in key:
            channel_number = key.rsplit(":chan", 1)[1]
            _expect(channel_number in ("0", "1", "2"), f"keyframe {kf_id}: {key!r} is not a channel")
            values[key] = _channel_from_json(value, f"keyframe {kf_id} {key}", int(channel_number))
        elif key.startswith("framing:"):
            parts = key.split(":")
            _expect(len(parts) == 3 and parts[1].isdigit() and parts[2] in FRAMING_FIELDS,
                    f"keyframe {kf_id}: {key!r} is not a framing setting")
            _expect(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                    and (parts[2] in ("x", "y") or value > 0.0),
                    f"keyframe {kf_id}: {key!r} must be a number (above 0 for a diagonal or an aspect ratio)")
            values[key] = float(value)
        else:
            values[key] = _setting_from_json(key, value, f"keyframe {kf_id}")
    return Keyframe(kf_id, frame_index, values)


# What format 1 and 2 files call the metrics, sample points and Buddha points, and the numbers format 1 used for them
# (the program's own numbers of that time). Frozen history: the program's tables have since been rebuilt from named
# sequences, so these are spelt out; _migrate_project_4_to_5 renames them to today's keys.
_FORMAT_1_METRICS = {0: "horizontal_distance", 1: "vertical_distance", 2: "manhattan_distance",
                     3: "euclidean_distance", 4: "bounces", 5: "ray_bounce_fraction",
                     6: "boundary_bounce_fraction", 7: "loop_area", 8: "mean_bounce_angle", 9: "ray_length",
                     10: "last_segment_direction", 11: "termination_reason", 12: "last_segment_ratio"}
_FORMAT_1_POINTS = {0: "start_point", 1: "end_point", 2: "all_bounces_mean", 3: "boundary_bounces_mean",
                    4: "self_bounces_mean", 5: "uppermost_bounce", 6: "lowermost_bounce", 7: "rightmost_bounce",
                    8: "leftmost_bounce", 9: "uppermost_self_bounce", 10: "lowermost_self_bounce",
                    11: "rightmost_self_bounce", 12: "leftmost_self_bounce", 13: "origin",
                    14: "top_left_corner"}
_FORMAT_1_BUDDHA_POINTS = ["all_bounces_mean", "self_bounces_mean", "boundary_bounces_mean", "termination_point",
                           "termination_point_distance_limit", "termination_point_perpendicular_collision",
                           "termination_point_confined", "all_self_bounces", "all_segment_midpoints",
                           "progressive_means_of_bounce_points", "progressive_means_of_segment_midpoints"]


def _migrate_project_1_to_2(data):
    """Format 1 stored choices as the program's numbers (and obstacle kinds as class names); format 2
    stores their keys. Converts a parsed format-1 file."""
    import copy
    data = copy.deepcopy(data)
    kind_keys = {cls.__name__: cls.KEY for cls in SHAPE_CLASSES}
    choice = {key: catalog for key, catalog in SETTING_CHOICE_CATALOGS.items()}

    def channel(old, where):
        _expect(isinstance(old, dict), f"{where}: a colour channel must be an object")
        points = old.get("buddha_points")
        _expect(isinstance(points, list), f"{where}: buddha_points must be a list")
        out = {"source": CAT_CHAN_SOURCE.key(old["chan_src"])}
        if old["chan_src"] == CHAN_SRC_QUERY:
            out["kind"] = CAT_METRIC_TYPE.key(old["kind"])
            out["metric"] = _FORMAT_1_METRICS[old["stat"]]
            if old["kind"] == METRIC_TYPE_POINT_PAIR:
                out["point1"] = _FORMAT_1_POINTS[old["point1"]]
                out["point2"] = _FORMAT_1_POINTS[old["point2"]]
        elif old["chan_src"] == CHAN_SRC_BUDDHA:
            out["buddha_points"] = [key for key, chosen in zip(_FORMAT_1_BUDDHA_POINTS, points) if chosen]
            out["buddha_compression"] = CAT_BUDDHA_COMPRESSION.key(old["buddha_compression"])
            out["buddha_brightness"] = CAT_BUDDHA_BRIGHTNESS.key(old["buddha_brightness"])
        else:
            out["constant"] = CAT_CONSTANT.key(old["constant"])
        return out

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
    for coloration in data.get("views", []):                  # (what formats 1-5 call a coloration)
        coloration["channels"] = [channel(ch, f"coloration {coloration.get('id')} channel {c + 1}")
                                  for c, ch in enumerate(coloration.get("channels", []))]
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


def _migrate_project_2_to_3(data):
    """Format 2 kept the Buddha blur's two settings in the "simulation" section; format 3 has a
    section of its own for them (the blur is not part of the ray simulation). Converts a parsed
    format-2 file: the two settings move, unchanged, to the new "buddha_blur" section."""
    import copy
    data = copy.deepcopy(data)
    simulation = data.get("simulation")
    moved = {}
    if isinstance(simulation, dict):
        for key in ("buddha_blur_radius", "buddha_blur_type"):         # (the format-2 keys, spelt
            if key in simulation:                                       # out: history, not the
                moved[key] = simulation.pop(key)                        # program's current tables)
    data["buddha_blur"] = moved
    data["format_version"] = 3
    return data


def _migrate_project_3_to_4(data):
    """Format 3 kept the natural termination settings (the iteration, distance and escape-radius limits and the
    three confinement settings) in the "simulation" section; format 4 has a section of their own for them (they are
    not specific to the ray-bouncing engine). Converts a parsed format-3 file: the settings that are there move,
    unchanged, to the new "natural_stop" section."""
    import copy
    data = copy.deepcopy(data)
    simulation = data.get("simulation")
    moved = {}
    if isinstance(simulation, dict):
        for key in ("iter", "dist", "escape_radius", "confine_window", "confine_method",      # (the format-3
                    "confine_threshold"):                                                     # keys, spelt out)
            if key in simulation:
                moved[key] = simulation.pop(key)
    data["natural_stop"] = moved
    data["format_version"] = 4
    return data


# Format 4 named the metrics, sample points and Buddha points one by one; format 5 names each by its sequence
# and reduction (the other keys of format 4 are still the same). Only the renamed ones are listed.
_FORMAT_4_METRIC_RENAMES = {
    "bounces": "bounces_count", "ray_bounce_fraction": "self_bounces_share",
    "boundary_bounce_fraction": "boundary_bounces_share", "ray_length": "path_length",
    "last_segment_direction": "path_last_direction"}
_FORMAT_4_POINT_RENAMES = {
    "start_point": "path_first", "end_point": "path_last", "all_bounces_mean": "bounces_mean",
    "uppermost_bounce": "bounces_uppermost", "lowermost_bounce": "bounces_lowermost",
    "rightmost_bounce": "bounces_rightmost", "leftmost_bounce": "bounces_leftmost",
    "uppermost_self_bounce": "self_bounces_uppermost", "lowermost_self_bounce": "self_bounces_lowermost",
    "rightmost_self_bounce": "self_bounces_rightmost", "leftmost_self_bounce": "self_bounces_leftmost"}
_FORMAT_4_BUDDHA_RENAMES = {
    **_FORMAT_4_POINT_RENAMES, "termination_point": "path_last",
    "termination_point_distance_limit": "path_last_distance_limit",
    "termination_point_perpendicular_collision": "path_last_perpendicular_collision",
    "termination_point_confined": "path_last_confined"}


def _migrate_project_4_to_5(data):
    """Format 4 named each metric, sample point and Buddha point of a colour channel on its own (a few for what is
    now a reduction of a named sequence); format 5 names them by sequence and reduction. Converts a parsed
    format-4 file: the channels of every coloration (a "view" in formats 1-5) and of every keyframe have those names
    replaced (see the tables above); all else is unchanged."""
    import copy
    data = copy.deepcopy(data)

    def channel(ch):
        if not isinstance(ch, dict):
            return ch
        if "metric" in ch:
            ch["metric"] = _FORMAT_4_METRIC_RENAMES.get(ch["metric"], ch["metric"])
        for name in ("point1", "point2"):
            if name in ch:
                ch[name] = _FORMAT_4_POINT_RENAMES.get(ch[name], ch[name])
        if isinstance(ch.get("buddha_points"), list):
            ch["buddha_points"] = [_FORMAT_4_BUDDHA_RENAMES.get(key, key) for key in ch["buddha_points"]]
        return ch

    for coloration in data.get("views", []):
        if isinstance(coloration, dict) and isinstance(coloration.get("channels"), list):
            coloration["channels"] = [channel(ch) for ch in coloration["channels"]]
    for kf in data.get("keyframes", []):
        settings = kf.get("settings") if isinstance(kf, dict) else None
        if isinstance(settings, dict):
            for key in settings:
                if key.startswith("view:") and ":chan" in key:
                    settings[key] = channel(settings[key])
    data["format_version"] = 5
    return data


def _migrate_project_5_to_6(data):
    """Format 5 called what the program now calls a coloration a "view": the "views" section, the "view" id counter
    and selection, the "include_all_views" export choice and the "view:<id>:..." keyframe keys. Format 6 renames them
    all ("colorations", "coloration", "include_all_colorations", "coloration:<id>:..."); the values are unchanged.
    Converts a parsed format-5 file."""
    import copy
    data = copy.deepcopy(data)
    if "views" in data:
        data["colorations"] = data.pop("views")
    for section, old, new in (("counters", "view", "coloration"), ("selected", "view", "coloration"),
                              ("export", "include_all_views", "include_all_colorations")):
        if isinstance(data.get(section), dict) and old in data[section]:
            data[section][new] = data[section].pop(old)
    for kf in data.get("keyframes", []):
        settings = kf.get("settings") if isinstance(kf, dict) else None
        if isinstance(settings, dict):
            kf["settings"] = {("coloration:" + key[len("view:"):] if key.startswith("view:") else key): value
                              for key, value in settings.items()}
    data["format_version"] = 6
    return data


def _migrate_project_6_to_7(data):
    """Format 6 called how many of the most recent path points the confinement rule judges "confine_window" (the
    natural termination setting of that name and the keyframe setting of that name); format 7 calls it
    "confine_lookback" (a "window" is a rectangle of the world, in this program). The values are unchanged.
    Converts a parsed format-6 file."""
    import copy
    data = copy.deepcopy(data)
    natural_termination = data.get("natural_stop")
    if isinstance(natural_termination, dict) and "confine_window" in natural_termination:
        natural_termination["confine_lookback"] = natural_termination.pop("confine_window")
    for kf in data.get("keyframes", []):
        settings = kf.get("settings") if isinstance(kf, dict) else None
        if isinstance(settings, dict) and "confine_window" in settings:
            kf["settings"] = {("confine_lookback" if key == "confine_window" else key): value
                              for key, value in settings.items()}
    data["format_version"] = 7
    return data


def _migrate_project_7_to_8(data):
    """Format 7 called the section of the natural termination settings (the iteration, distance and escape-radius
    limits and the three confinement settings) "natural_stop", and the color model of a coloration (RGB, HSV...)
    its "mode" (in the coloration's entry and in the keyframe settings "coloration:<id>:mode"). Format 8 calls them
    "natural_termination" (a ray ends by termination, never by "stop") and "color_model" ("mode" is also the
    emission's and the video bitrate's). Their values are unchanged. Converts a parsed format-7 file."""
    import copy
    data = copy.deepcopy(data)
    if "natural_stop" in data:
        data["natural_termination"] = data.pop("natural_stop")
    for coloration in data.get("colorations", []):
        if isinstance(coloration, dict) and "mode" in coloration:
            coloration["color_model"] = coloration.pop("mode")
    for kf in data.get("keyframes", []):
        settings = kf.get("settings") if isinstance(kf, dict) else None
        if isinstance(settings, dict):
            kf["settings"] = {(key[:-len(":mode")] + ":color_model" if key.startswith("coloration:") and key.endswith(":mode")
                               else key): value for key, value in settings.items()}
    data["format_version"] = 8
    return data


# What a format-8 file meant by a value it does not hold: the defaults of that time (the program's defaults have been
# changed since - the world is twice as large - and a file must keep its picture). Frozen history, spelt out.
_FORMAT_8_NATURAL_TERMINATION = {"escape_radius": 2.0, "confine_threshold": 2.0 / 512}
_FORMAT_8_BUDDHA_SEEDING = {"buddha_seed_camera": True, "buddha_seed_x_start": -1.0, "buddha_seed_x_stop": 1.0,
                            "buddha_seed_y_start": -1.0, "buddha_seed_y_stop": 1.0}
_FORMAT_8_FRAMING = {"x": 0.0, "y": 0.0, "diagonal": 2.0 * math.sqrt(2.0), "aspect": 1.0}      # the whole world [-1, 1]
_FORMAT_8_OBSTACLE_DEFAULTS = {            # kind key -> field -> the default of a field that is a distance
    "ellipse": {"width": 1.0, "height": 1.0},
    "regular_polygon": {"radius": 1.0},
    "moss_egg": {"moss_size": 0.9},
    "circular_sector": {"radius": 1.0},
    "circular_segment": {"radius": 1.0},
    "two_circle_lens": {"lens_radius1": 1.0, "lens_radius2": 1.0, "lens_sep": 1.0},
    "reuleaux_polygon": {"radius": 1.0},
    "capsule": {"capsule_length": 1.0, "capsule_radius": 0.5},
    "trefoil": {"radius": 1.0},
    "quatrefoil": {"radius": 1.0},
    "lune": {"radius": 1.0, "cut_radius": 0.8, "cut_offset": 0.6},
    "arbelos": {"radius": 1.0},
    "superelliptical_squircle": {"width": 1.0, "height": 1.0},
    "fernandez_guasti_squircle": {"radius": 1.0},
    "archimedean_spiral": {"radius": 1.0, "spiral_pitch": 0.25},
    "snail_of_pythagoras": {"radius": 1.0},          # (the snail had no size: its farthest corner was 1 from the centre)
    "hugelschaffer_egg": {"egg_a": 1.0, "egg_b": 0.7},
    "sine_flower": {"radius": 1.0}}
_FORMAT_8_POINT_RENAMES = {"top_left_corner": "(-1,1)"}


def _migrate_project_8_to_9(data):
    """Format 9 is written with the world twice as large by default, so a file now says explicitly what a format-8 file
    left to the defaults of its time: the escape radius and confinement threshold, the Buddha seeding zone, the framing
    (a file with none had the whole world [-1, 1]), the size of every obstacle (a field it does not hold). It also renames
    the point choice "top_left_corner" (the fixed point (-1, 1)) to its key "(-1,1)", in the colorations and in the
    keyframes' channels. The keyframes are not otherwise touched: a setting no keyframe defines is taken, by an export,
    from what the controls held when it began (resolve_frame_settings), which is the file's section, now explicit.
    The picture the live image makes does not change. Converts a parsed format-8 file."""
    import copy
    data = copy.deepcopy(data)
    for section, defaults in (("natural_termination", _FORMAT_8_NATURAL_TERMINATION),
                              ("buddha_seeding", _FORMAT_8_BUDDHA_SEEDING)):
        if not isinstance(data.get(section), dict):
            data[section] = {}
        for key, value in defaults.items():
            data[section].setdefault(key, value)
    framings = data.get("framings")
    if not framings:
        data["framings"] = [{"id": 0, **_FORMAT_8_FRAMING}]
    else:
        for framing in framings:
            if isinstance(framing, dict):
                for field, value in _FORMAT_8_FRAMING.items():
                    framing.setdefault(field, value)
    for obstacle in data.get("obstacles", []):
        fields = obstacle.get("fields") if isinstance(obstacle, dict) else None
        if isinstance(fields, dict):
            for name, value in _FORMAT_8_OBSTACLE_DEFAULTS.get(obstacle.get("kind"), {}).items():
                fields.setdefault(name, value)

    def rename_points(channel):
        if isinstance(channel, dict):
            for name in ("point1", "point2"):
                if channel.get(name) in _FORMAT_8_POINT_RENAMES:
                    channel[name] = _FORMAT_8_POINT_RENAMES[channel[name]]
    for coloration in data.get("colorations", []):
        for channel in (coloration.get("channels") or []) if isinstance(coloration, dict) else []:
            rename_points(channel)
    keyframes = [kf for kf in data.get("keyframes", []) if isinstance(kf, dict) and isinstance(kf.get("settings"), dict)]
    for kf in keyframes:
        for key, value in kf["settings"].items():
            if key.startswith("coloration:") and ":chan" in key:
                rename_points(value)
    data["format_version"] = 9
    return data


PROJECT_MIGRATIONS = {1: _migrate_project_1_to_2,      # format n -> format n + 1
                      2: _migrate_project_2_to_3,
                      3: _migrate_project_3_to_4,
                      4: _migrate_project_4_to_5,
                      5: _migrate_project_5_to_6,
                      6: _migrate_project_6_to_7,
                      7: _migrate_project_7_to_8,
                      8: _migrate_project_8_to_9}


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
            "This is not a Prismatic Fractals project file.")
    version = data.get("format_version")
    _expect(isinstance(version, int) and not isinstance(version, bool) and version >= 1,
            "The project file has no valid format version.")
    _expect(version <= PROJECT_FORMAT_VERSION,
            f"This project was saved by a newer version of the program (project format "
            f"{version}; this version reads up to {PROJECT_FORMAT_VERSION}).")
    while version < PROJECT_FORMAT_VERSION:
        data = PROJECT_MIGRATIONS[version](data)
        version += 1

    ray = _settings_from_section(data, "simulation", RaySettings, "the simulation settings")
    natural_termination = _settings_from_section(data, "natural_termination", NaturalTerminationSettings,
                                          "the natural termination settings")
    buddha_blur = _settings_from_section(data, "buddha_blur", BuddhaBlurSettings,
                                         "the Buddha blur settings")
    buddha_seed = _settings_from_section(data, "buddha_seeding", BuddhaSeedSettings,
                                         "the Buddha seeding settings")
    engine = _settings_from_section(data, "engine", EngineSettings, "the path engine")
    mandelbrot = _settings_from_section(data, "mandelbrot", MandelbrotSettings, "the Mandelbrot engine's settings")
    _expect(engine.engine in OFFERED_ENGINES, f"This project uses the {ENGINE_NAMES[engine.engine]} path engine, which this "
            "version of the program cannot run yet.")

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

    obstacle_ids, coloration_ids, framing_ids, keyframe_ids = set(), set(), set(), set()
    obstacles = tuple(_shape_from_json(o, obstacle_ids) for o in data.get("obstacles", []))
    colorations = tuple(_coloration_from_json(v, coloration_ids) for v in data.get("colorations", []))
    _expect(colorations, "The project has no colorations.")
    framings = tuple(_framing_from_json(f, framing_ids) for f in data.get("framings") or [])
    if not framings:                                # (a file from before framings existed: the whole world)
        framings = (DEFAULT_FRAMING,)
        framing_ids.add(DEFAULT_FRAMING.id)
    keyframes = tuple(_keyframe_from_json(kf, keyframe_ids) for kf in data.get("keyframes", []))

    counters = data.get("counters") or {}
    selected = data.get("selected") or {}
    selected_obstacle = selected.get("obstacle")
    selected_coloration = selected.get("coloration")
    selected_framing = selected.get("framing")
    return Project(
        ray=ray, natural_termination=natural_termination, buddha_blur=buddha_blur, buddha_seed=buddha_seed, engine=engine,
        mandelbrot=mandelbrot, decorations=decorations, obstacles=obstacles, colorations=colorations, framings=framings,
        keyframes=keyframes, export=export,
        obstacle_counter=max(counters.get("obstacle", 0), max(obstacle_ids, default=-1) + 1),
        coloration_counter=max(counters.get("coloration", 0), max(coloration_ids) + 1),
        framing_counter=max(counters.get("framing", 0), max(framing_ids) + 1),
        keyframe_counter=max(counters.get("keyframe", 0), max(keyframe_ids, default=-1) + 1),
        selected_obstacle_id=selected_obstacle if selected_obstacle in obstacle_ids else -1,
        selected_coloration_id=selected_coloration if selected_coloration in coloration_ids else colorations[0].id,
        selected_framing_id=selected_framing if selected_framing in framing_ids else framings[0].id)
