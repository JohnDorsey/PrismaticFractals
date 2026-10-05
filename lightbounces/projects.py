"""Projects: saving and loading everything the user has set up.

A project file is pretty-printed, versioned JSON (PROJECT_EXTENSION). It holds the controls' simulation settings, the Buddha blur, the
decorations, every obstacle, view and keyframe, and the Export designer and video panel choices - not the generated image, which Start
makes again (fuzzing is seeded, so the same settings give the same image).

Every choice in the file is a readable string KEY ("euclidean_distance", "reflect", "lune"), never a number. A key is permanent: once a
project file can contain it, it is never renamed or reused, and a Catalog translates it to the program's own code (an int the kernels
index arrays with, which may be renumbered freely). So the program is free to reorder, insert and renumber its codes;
test_the_keys_in_project_files_never_change pins the keys. Object ids (obstacles, views, keyframes) stay the whole numbers they are.
Older files are converted one format version at a time by PROJECT_MIGRATIONS."""
import json
import re
from .constants import (
    BUDDHA_BLUR_BOKEH, BUDDHA_BLUR_CATADIOPTRIC, BUDDHA_BLUR_GAUSSIAN, BUDDHA_BLUR_TRIANGULAR
)
from .export_text import (
    COLOR_INFO_LOCATIONS, FIGURE_INFO_LOCATIONS, SAVE_SOURCE_OPTIONS, VIDEO_BITRATE_MODES,
    VIDEO_FORMATS
)
from .image_io import DOWNSAMPLE_FACTORS, DOWNSAMPLE_METHODS, JXL_MODES, SAVE_NEW_FORMATS
from .kernels import (
    CONFINE_BOX, CONFINE_PATH, EMIT_TOWARDS, EMIT_UP, OBSTACLE_EFFECT_PASS,
    OBSTACLE_EFFECT_REFLECT, OBSTACLE_EFFECT_TERMINATE, POINT_ALL_MEAN, POINT_BOTTOM,
    POINT_BOUNDARY_MEAN, POINT_END, POINT_LEFT, POINT_RIGHT, POINT_SELF_BOTTOM, POINT_SELF_LEFT,
    POINT_SELF_MEAN, POINT_SELF_RIGHT, POINT_SELF_TOP, POINT_START, POINT_TOP, ST_ANGLES,
    ST_BOUNCES, ST_BOUNDARY_FRAC, ST_EUCLIDEAN, ST_HDIST, ST_LAST_DIR, ST_LAST_RATIO, ST_LENGTH,
    ST_LOOP_AREA, ST_MANHATTAN, ST_RAY_FRAC, ST_TERMINATION, ST_VDIST
)
from .metrics import (
    BUDDHA_COMPRESS_ARCTAN, BUDDHA_COMPRESS_MODULUS, CHAN_SRC_BUDDHA, CHAN_SRC_CONSTANT,
    CHAN_SRC_QUERY, KIND_OTHER, KIND_POINT, NONE_ONE, NONE_ZERO, N_BUDDHA_POINTS, OTHER_STATS,
    POINT_ORIGIN, POINT_STATS, POINT_TOP_LEFT_CORNER
)
from .settings import (
    BuddhaBlurSettings, DEFAULT_SETTING_VALUES, Keyframe, RaySettings, SETTING_TABLES,
    settings_from_dict, settings_to_dict, strict_record
)
from .shapes import SHAPE_CLASSES
from .version import FILE_VERSION
from .views import ChannelSettings, MODES, ViewConfig, snapshot_view, view_channel_value


PROJECT_EXTENSION = ".lightbounce"
PROJECT_FORMAT_NAME = "light-bounces-project"
PROJECT_FORMAT_VERSION = 3
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
# The settings (of any settings record) that are choices, by setting key
SETTING_CHOICE_CATALOGS = {"emit_mode": CAT_EMIT_MODE, "confine_method": CAT_CONFINE_METHOD,
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
    """Everything a project file holds, as plain values (the obstacles, views and keyframes
    are independent frozen copies). Made by App.capture_project or project_from_dict."""
    ray: RaySettings                    # the ray simulation's controls
    buddha_blur: BuddhaBlurSettings     # the Buddha blur's, beside them (not inside)
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
    """A settings record (a RaySettings, a BuddhaBlurSettings...) as the file's section for it:
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
        "buddha_blur": _settings_to_json(project.buddha_blur),
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
    choice = {key: catalog for key, catalog in SETTING_CHOICE_CATALOGS.items()}

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


PROJECT_MIGRATIONS = {1: _migrate_project_1_to_2,      # format n -> format n + 1
                      2: _migrate_project_2_to_3}


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

    ray = _settings_from_section(data, "simulation", RaySettings, "the simulation settings")
    buddha_blur = _settings_from_section(data, "buddha_blur", BuddhaBlurSettings,
                                         "the Buddha blur settings")

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
        ray=ray, buddha_blur=buddha_blur, decorations=decorations, obstacles=obstacles, views=views,
        keyframes=keyframes, export=export,
        obstacle_counter=max(counters.get("obstacle", 0), max(obstacle_ids, default=-1) + 1),
        view_counter=max(counters.get("view", 0), max(view_ids) + 1),
        keyframe_counter=max(counters.get("keyframe", 0), max(keyframe_ids, default=-1) + 1),
        selected_obstacle_id=selected_obstacle if selected_obstacle in obstacle_ids else -1,
        selected_view_id=selected_view if selected_view in view_ids else views[0].id)
