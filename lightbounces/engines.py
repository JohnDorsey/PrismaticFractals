"""What each path engine offers and accepts, and the rules that keep a set of settings valid for the engine in use (the engine
refactor, `Engine_Refactor_Design.md` D7, D11, D17, D18, D34, D35, D39). All of it is pure: functions of plain dicts and records, with no window.

* An `EngineDescription` says which choices an engine offers (the ids of the metrics, points and Buddha options are GLOBAL, shared by
  every engine - D34 - so an engine offers a subset of them and nothing is renumbered), which values of other choice settings it
  accepts (an obstacle's ray effects, say), the range of a number it accepts, the defaults it uses where the usual default is not
  offered (D39), and what it ignores (the tabs and settings that are greyed out, D13). `ENGINE_DESCRIPTIONS` holds the real ones.
* `correct_settings(settings, engine)` is THE correction (D17, D35): given a dict of settings (the keys of a keyframe: all of them
  or any subset) it returns the dict with every value the engine does not accept replaced - a choice by its default, a number
  clamped - and the list of `Correction`s made. Every place a render starts calls it.
* `switch_engine` is what happens when the user changes engine (D7, D18): the settings the previous engine reset come back where
  nothing has changed them since, then the new engine's corrections are made, and what they reset is remembered (`EngineMemory`) for
  the way back; `go_back` undoes the switch.
* `correction_message` words a list of corrections as the pop-ups of D7, D20 and D36 show them."""
import math
from .colorations import (
    DEFAULT_OTHER_METRICS, DEFAULT_POINT1S, DEFAULT_POINT2S, DEFAULT_POINT_METRICS, ChannelSettings, coloration_channel_value,
    decompose_chans
)
from .constants import ENGINE_NAMES, OBSTACLE_EFFECT_NAMES
from .kernels import (
    ENGINE_MANDELBROT, ENGINE_RAY_BOUNCING, N_POINTS, N_PRED, N_STATS, OBSTACLE_EFFECT_PASS, OBSTACLE_EFFECT_REFLECT,
    OBSTACLE_EFFECT_TERMINATE, POINT_ALL_MEAN, POINT_START, PRED_MEAN, SEQ_PATH, SRED_COUNT, SRED_LAST_DIR, SRED_LENGTH, ST_BOUNCE_COUNT,
    ST_HDIST, ST_TERMINATION
)
from .metrics import (
    BUDDHA_POINT_LABELS, CHAN_SRC_BUDDHA, CHAN_SRC_QUERY, CHOICE_ORIGIN, CHOICE_POINT_NEG1_POS1, CHOICE_TOP_LEFT_POINT, METRIC_LABELS,
    METRIC_TYPE_POINT_PAIR, N_BUDDHA_POINTS, N_POINT_CHOICES, OTHER_METRICS, POINT_LABELS, POINT_MENU_ORDER, POINT_METRICS,
    point_slot, stat_slot
)
from .settings import (
    DEFAULT_SETTING_VALUES, SETTING_FLOAT_KEYS, SETTING_SPECS, SETTING_TAB, SETTING_TABS_ORDER, SETTING_LABEL, BuddhaBlurSettings,
    BuddhaSeedSettings, EngineSettings, MandelbrotSettings, NaturalTerminationSettings, RaySettings, settings_from_dict, settings_to_dict,
    strict_record
)
from .shapes import EFFECT_FIELDS
from .stored_layout import StoredLayout, sequence_tables


@strict_record
class EngineDescription:
    """What one path engine offers and accepts. The sets hold the GLOBAL ids of the catalogs (a metric's id is its stored slot, a
    point's its slot or one of the three fixed points, a Buddha option's its place in BUDDHA_POINT_CATALOG), so a coloration means
    the same choice whichever engine is in use."""
    engine: int
    name: str
    metrics: frozenset                  # the metric ids offered (the point-comparison metrics among them)
    points: frozenset                   # the point ids offered
    buddha_options: frozenset           # the Buddha options offered (indices into BUDDHA_POINT_CATALOG)
    default_point_metric: int           # used where a channel's own default point-comparison metric is not offered
    default_other_metric: int           # ... its default "other" metric
    default_point: int                  # ... its default point 1 / point 2
    default_buddha_option: int          # ... the Buddha option a channel is left with when none of its own is offered
    choices: dict                       # {key pattern: frozenset of the values offered} for the other choice settings
    fallbacks: dict                     # {key pattern: the engine's own default}, for a choice whose usual default is not offered
    ranges: dict                        # {key pattern: (lo, hi)} the numbers it accepts (None for no limit on that side)
    ignored_tabs: tuple                 # the tabs whose settings it makes no use of (greyed out, never reset)
    ignored_keys: frozenset             # ... and other settings it makes no use of

    def ignores(self, key):
        """Does this engine make no use of the setting `key` (so that its control is greyed out)? It still has a value, which is
        saved, animated and shown (D13): nothing here ever resets it."""
        return key in self.ignored_keys or SETTING_TAB.get(key) in self.ignored_tabs


# The ray-bouncing engine offers everything there is, so its ids are the catalogs' own ids (D34).
RAY_BOUNCING_ENGINE = EngineDescription(
    engine=ENGINE_RAY_BOUNCING, name=ENGINE_NAMES[ENGINE_RAY_BOUNCING],
    metrics=frozenset(range(N_STATS)), points=frozenset(range(N_POINT_CHOICES)),
    buddha_options=frozenset(range(N_BUDDHA_POINTS)),
    default_point_metric=ST_HDIST, default_other_metric=ST_BOUNCE_COUNT, default_point=POINT_START,
    default_buddha_option=POINT_ALL_MEAN, choices={}, fallbacks={}, ranges={}, ignored_tabs=(), ignored_keys=frozenset())

def _mandelbrot_description():
    """The Mandelbrot engine (Engine_Refactor_Design.md D31, D38, D39, D11, D35, D5): one sequence, "Path", with its seven point
    reductions; the metrics iteration count (the path's point count), path length, last direction, termination reason and the four
    point distances; the Buddha options of the path (its stored points, the last point for a distance limit or confinement - not
    for a perpendicular collision, which it has none of -, every point of the path and the two segment-midpoint options (D38), none
    of the bounce-based ones; no Reflect
    (pass-through in its place); its own defaults (Path count, Path: Mean point); and the Emissions and Mutations tabs unused."""
    path_points = {point_slot(SEQ_PATH, red) for red in range(N_PRED)}
    metrics = set(POINT_METRICS) | {stat_slot(SEQ_PATH, SRED_COUNT), stat_slot(SEQ_PATH, SRED_LENGTH),
                                    stat_slot(SEQ_PATH, SRED_LAST_DIR), ST_TERMINATION}
    options = set(path_points) | {BUDDHA_POINT_LABELS.index(label) for label in
                                  ("Path: Last point (distance limit)", "Path: Last point (confined)", "All path points",
                                   "All segment midpoints", "Progressive means of segment midpoints")}
    effects = frozenset({OBSTACLE_EFFECT_TERMINATE, OBSTACLE_EFFECT_PASS})
    return EngineDescription(
        engine=ENGINE_MANDELBROT, name=ENGINE_NAMES[ENGINE_MANDELBROT], metrics=frozenset(metrics),
        points=frozenset(path_points | {CHOICE_ORIGIN, CHOICE_POINT_NEG1_POS1, CHOICE_TOP_LEFT_POINT}),
        buddha_options=frozenset(options), default_point_metric=ST_HDIST, default_other_metric=stat_slot(SEQ_PATH, SRED_COUNT),
        default_point=POINT_START, default_buddha_option=point_slot(SEQ_PATH, PRED_MEAN),
        choices={"obst:*:in_effect": effects, "obst:*:out_effect": effects},
        fallbacks={"obst:*:in_effect": OBSTACLE_EFFECT_PASS, "obst:*:out_effect": OBSTACLE_EFFECT_PASS}, ranges={},
        ignored_tabs=("Emissions", "Mutations"), ignored_keys=frozenset())


MANDELBROT_ENGINE = _mandelbrot_description()

ENGINE_DESCRIPTIONS = {ENGINE_RAY_BOUNCING: RAY_BOUNCING_ENGINE, ENGINE_MANDELBROT: MANDELBROT_ENGINE}


def engine_description(engine):
    """The description of the engine with code `engine`."""
    try:
        return ENGINE_DESCRIPTIONS[engine]
    except KeyError:
        raise KeyError(f"the path engine {engine!r} ({ENGINE_NAMES[engine] if 0 <= engine < len(ENGINE_NAMES) else '?'}) has no "
                       f"description yet") from None


def engine_layout(engine):
    """The StoredLayout of everything the engine described by `engine` (an EngineDescription) stores per path: the metrics and the
    characteristic points it offers (the fixed points have no slot), in the order of their ids. Its positions are where the engine's
    per-path arrays keep them; for the ray engine, which offers every id, they are the ids."""
    return StoredLayout.of(sorted(engine.metrics), sorted(point for point in engine.points if point < N_POINTS))


_LAYOUTS = {}


def layout_of_engine(code):
    """engine_layout of the engine with code `code`, kept."""
    if code not in _LAYOUTS:
        _LAYOUTS[code] = engine_layout(engine_description(code))
    return _LAYOUTS[code]


_TABLES = {}


def tables_of_engine(code):
    """The SequenceTables of the engine with code `code` (see layout_of_engine), kept."""
    if code not in _TABLES:
        _TABLES[code] = sequence_tables(layout_of_engine(code))
    return _TABLES[code]


# --------------------------------------------------------------------------- #
# Keys, usual defaults, labels
# --------------------------------------------------------------------------- #
# The dynamic keys ("obst:<id>:<field>", "coloration:<id>:<field>", "framing:<id>:<field>") have a pattern with the id replaced by *:
# a description's `choices`, `fallbacks` and `ranges` are keyed by pattern (a static setting's pattern is its key).
DYNAMIC_KEY_PREFIXES = ("obst", "coloration", "framing")


def key_pattern(key):
    """`obst:3:in_effect` -> `obst:*:in_effect`; a static setting's key is its own pattern."""
    parts = key.split(":")
    if len(parts) == 3 and parts[0] in DYNAMIC_KEY_PREFIXES:
        return f"{parts[0]}:*:{parts[2]}"
    return key


# The value a choice setting has by default, where it is not a static setting (whose default is in DEFAULT_SETTING_VALUES).
USUAL_DEFAULTS = {"obst:*:in_effect": OBSTACLE_EFFECT_REFLECT, "obst:*:out_effect": OBSTACLE_EFFECT_REFLECT}
EFFECT_LABELS = {field: label for field, label, _type in EFFECT_FIELDS}

# The tabs corrections are grouped under: the settings tabs, then the lists.
TAB_OBSTACLES = "Obstacles"
TAB_COLORATIONS = "Colorations"
CORRECTION_TAB_ORDER = SETTING_TABS_ORDER + [TAB_OBSTACLES, TAB_COLORATIONS]


def point_metric_labels(engine):
    """The labels of the point-comparison metrics `engine` offers, in the order of the dropdown."""
    return [METRIC_LABELS[s] for s in POINT_METRICS if s in engine.metrics]


def other_metric_labels(engine):
    """The labels of the other metrics `engine` offers, in the order of the dropdown."""
    return [METRIC_LABELS[s] for s in OTHER_METRICS if s in engine.metrics]


def point_menu_labels(engine):
    """The labels of the points `engine` offers, in the order of the two point dropdowns."""
    return [POINT_LABELS[i] for i in POINT_MENU_ORDER if i in engine.points]


def effect_names(engine, pattern="obst:*:in_effect"):
    """The names of the obstacle effects `engine` offers for the setting with key pattern `pattern` (all of them where it
    restricts nothing)."""
    allowed = engine.choices.get(pattern)
    return [name for effect, name in enumerate(OBSTACLE_EFFECT_NAMES) if allowed is None or effect in allowed]


def default_value(engine, pattern):
    """The value the setting with this key pattern is reset to under `engine`: its usual default if the engine accepts it,
    otherwise the engine's own (D39). Also the value a NEW obstacle's effects start with (D35)."""
    usual = USUAL_DEFAULTS.get(pattern, DEFAULT_SETTING_VALUES.get(pattern))
    allowed = engine.choices.get(pattern)
    if usual is not None and (allowed is None or usual in allowed):
        return usual
    if pattern in engine.fallbacks:
        return engine.fallbacks[pattern]
    raise ValueError(f"the engine {engine.name!r} accepts no default for {pattern!r} (usual default {usual!r}, "
                     f"accepted {sorted(allowed) if allowed is not None else 'anything'}) and gives no fallback")


def new_obstacle_effects(engine):
    """(incoming, outgoing) effects of an obstacle made while `engine` is selected (D35)."""
    return default_value(engine, "obst:*:in_effect"), default_value(engine, "obst:*:out_effect")


def _number_text(value):
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"


def _ticked_text(flags, labels):
    return ", ".join(label for label, on in zip(labels, flags) if on) or "none"


# --------------------------------------------------------------------------- #
# The correction
# --------------------------------------------------------------------------- #
@strict_record
class Correction:
    """One value `correct_settings` changed. `key` is the setting's key in the dict; `field` names the part of the value for a
    colour channel ("metric_slot", "point1"...) and is "" for a value changed whole. `tab` and `label` say where the user finds it,
    `old_text` and `new_text` how the two values are worded."""
    key: str
    field: str
    tab: str
    label: str
    old: object
    new: object
    old_text: str
    new_text: str


CHANNEL_FIELD_LABELS = {"metric_slot": "Metric", "point1": "Point 1", "point2": "Point 2", "buddha_points": "Buddha points"}


def channel_index(key):
    """The channel (0..2) a `coloration:<id>:chan<c>` key names."""
    return int(key.rsplit("chan", 1)[1])


def correct_channel(channel, engine, index):
    """(the ChannelSettings that `engine` accepts, [(field, old, new, old_text, new_text), ...]) for colour channel `index`
    (0..2) as the record `channel` has it. Only the fields the channel USES are looked at: a Query channel's metric (and its two
    points when the metric compares points), a Buddha channel's options; a field it does not use may hold anything, and
    is corrected when the channel is switched to use it. A choice the engine does not offer is replaced by the channel's usual
    default for that choice, or by the engine's own where the usual one is not offered either (D39). A Buddha channel loses the
    options the engine does not offer, and is given the engine's default option only if that leaves it with none."""
    changes = {}
    if channel.chan_src == CHAN_SRC_QUERY:
        if channel.metric_type == METRIC_TYPE_POINT_PAIR:
            usual, own, labels = DEFAULT_POINT_METRICS[index], engine.default_point_metric, METRIC_LABELS
            if channel.metric_slot not in engine.metrics:
                new = usual if usual in engine.metrics else own
                changes["metric_slot"] = (channel.metric_slot, new, labels[channel.metric_slot], labels[new])
            for field, usual_point in (("point1", DEFAULT_POINT1S[index]), ("point2", DEFAULT_POINT2S[index])):
                old = getattr(channel, field)
                if old not in engine.points:
                    new = usual_point if usual_point in engine.points else engine.default_point
                    changes[field] = (old, new, POINT_LABELS[old], POINT_LABELS[new])
        else:
            usual = DEFAULT_OTHER_METRICS[index]
            if channel.metric_slot not in engine.metrics:
                new = usual if usual in engine.metrics else engine.default_other_metric
                changes["metric_slot"] = (channel.metric_slot, new, METRIC_LABELS[channel.metric_slot], METRIC_LABELS[new])
    elif channel.chan_src == CHAN_SRC_BUDDHA:
        old = channel.buddha_points
        kept = tuple(bool(on) and i in engine.buddha_options for i, on in enumerate(old))
        if kept != tuple(bool(on) for on in old):
            if not any(kept):
                kept = tuple(i == engine.default_buddha_option for i in range(len(old)))
            changes["buddha_points"] = (old, kept, _ticked_text(old, BUDDHA_POINT_LABELS), _ticked_text(kept, BUDDHA_POINT_LABELS))
    corrected = channel.replace(**{field: change[1] for field, change in changes.items()}) if changes else channel
    return corrected, [(field, *change) for field, change in changes.items()]


def _clamp(value, lo, hi):
    """`value` limited to lo..hi (None = open). A whole number stays whole: the limit is taken in towards the range."""
    if isinstance(value, int):
        if lo is not None:
            value = max(value, math.ceil(lo))
        if hi is not None:
            value = min(value, math.floor(hi))
        return value
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return value


def _label_of(key, titles):
    """(tab, label) of a setting that is not part of a colour channel."""
    parts = key.split(":")
    if len(parts) == 3 and parts[0] == "obst":
        owner = titles.get(f"obst:{parts[1]}", f"Obstacle {parts[1]}")
        return TAB_OBSTACLES, f"{owner}: {EFFECT_LABELS.get(parts[2], parts[2].replace('_', ' '))}"
    return SETTING_TAB.get(key, "Settings"), SETTING_LABEL.get(key, key)


def _value_text(key, value):
    pattern = key_pattern(key)
    if pattern in ("obst:*:in_effect", "obst:*:out_effect"):
        return OBSTACLE_EFFECT_NAMES[value]
    spec = SETTING_SPECS.get(key)
    if spec is not None and spec.kind == "choice" and not isinstance(value, bool):
        return spec.names[value]
    return _number_text(value)


def correct_settings(settings, engine, titles=None):
    """(corrected, corrections): `settings` (a dict of setting key -> value: every setting, or the few a keyframe defines) with
    every value that `engine` (an EngineDescription) does not accept replaced - a choice not offered by its default (D39), a number
    outside the engine's range clamped (D17) - and the list of `Correction`s, in the order of the dict. The input is not changed;
    the key "engine" itself is never touched. `titles` maps "obst:<id>" / "coloration:<id>" / "framing:<id>" to the title to word
    that item with (the keyframe list's "Obstacle #2 (Lens)"), else its id is used."""
    titles = titles or {}
    corrected = dict(settings)
    corrections = []
    for key, value in settings.items():
        if key == "engine":
            continue
        if isinstance(value, ChannelSettings):
            new, changes = correct_channel(value, engine, channel_index(key))
            if changes:
                corrected[key] = new
                owner = titles.get(":".join(key.split(":")[:2]), f"Coloration {key.split(':')[1]}")
                for field, old_value, new_value, old_text, new_text in changes:
                    corrections.append(Correction(
                        key=key, field=field, tab=TAB_COLORATIONS,
                        label=f"{owner}, channel {channel_index(key) + 1}: {CHANNEL_FIELD_LABELS[field]}",
                        old=old_value, new=new_value, old_text=old_text, new_text=new_text))
            continue
        pattern = key_pattern(key)
        new = value
        allowed = engine.choices.get(pattern)
        if allowed is not None and value not in allowed:
            new = default_value(engine, pattern)
        limits = engine.ranges.get(pattern)
        if limits is not None and isinstance(new, (int, float)) and not isinstance(new, bool):
            new = _clamp(new, *limits)
        if new != value:
            corrected[key] = new
            tab, label = _label_of(key, titles)
            corrections.append(Correction(key=key, field="", tab=tab, label=label, old=value, new=new,
                                          old_text=_value_text(key, value), new_text=_value_text(key, new)))
    return corrected, corrections


# --------------------------------------------------------------------------- #
# Wording
# --------------------------------------------------------------------------- #
SWITCH_HEADER = "Some settings that weren't supported by this engine have been reset:"            # D7
KEYFRAME_HEADER = "Some values from this keyframe have been clamped or defaulted:"               # D36
PROJECT_HEADER = "Some values from this project have been clamped or defaulted:"                 # D20
FRAME_ERROR_HEADER = "Some values of this frame have been clamped or defaulted:"                  # D20 (a video frame)


def group_corrections(corrections):
    """[(tab, [Correction, ...]), ...]: the corrections grouped by tab, in the order of the tabs, each group in the order given."""
    groups = {}
    for correction in corrections:
        groups.setdefault(correction.tab, []).append(correction)
    order = {tab: i for i, tab in enumerate(CORRECTION_TAB_ORDER)}
    return sorted(groups.items(), key=lambda item: order.get(item[0], len(order)))


def correction_message(corrections, header=SWITCH_HEADER):
    """The text of the pop-up of D7 (and, with another header, D20 and D36): the header, then one group per tab - `in the <tab> tab:` -
    and under it one `<setting>: <old> -> <new>` line for each correction."""
    lines = [header]
    for tab, items in group_corrections(corrections):
        lines.append(f"  in the {tab} tab:")
        lines += [f"    {c.label}: {c.old_text} -> {c.new_text}" for c in items]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Switching engine, and the engines' memory
# --------------------------------------------------------------------------- #
@strict_record
class EngineMemory:
    """What the engines remember of their settings, for the session (D18; never saved in a project). `kept[engine]` is {(key,
    field): (the value the engine had, the value it was reset to)} for what switching AWAY from that engine reset; coming back
    restores a value if the setting still holds the value it was reset to - nothing (the user, or a selected keyframe) has changed
    it since."""
    kept: dict


EMPTY_ENGINE_MEMORY = EngineMemory(kept={})


@strict_record
class EngineSwitch:
    """The result of `switch_engine`: the new `settings`, the `corrections` to tell the user about (what the new engine reset),
    the `memory` to keep, and everything `go_back` needs to undo it."""
    settings: dict
    corrections: tuple
    restored: tuple                     # the (key, field) pairs given back from the new engine's memory
    memory: EngineMemory
    before_settings: dict
    before_memory: EngineMemory
    touched: tuple                      # every key the switch changed


def _field_of(value, field):
    return getattr(value, field) if field else value


def _with_field(value, field, new):
    return value.replace(**{field: new}) if field else new


def switch_engine(settings, old, new, memory=EMPTY_ENGINE_MEMORY, titles=None):
    """Change the engine from `old` to `new` (EngineDescriptions) in `settings` (every setting, a dict): the key "engine" becomes the
    new engine; what the new engine remembered from the last time it was left comes back (D18) wherever the setting still holds
    the value that was put there; then everything the new engine does not accept is corrected (D7, D17) and what that reset is
    remembered for the old engine. Returns an `EngineSwitch`. Switching to the engine already in use changes nothing."""
    before = dict(settings)
    work = dict(settings)
    touched = []
    kept = {engine: dict(entries) for engine, entries in memory.kept.items()}
    restored = []
    if old.engine == new.engine:
        return EngineSwitch(settings=work, corrections=(), restored=(), memory=memory, before_settings=before,
                            before_memory=memory, touched=())
    for (key, field), (value, reset_to) in kept.pop(new.engine, {}).items():
        if key in work and _field_of(work[key], field) == reset_to:
            work[key] = _with_field(work[key], field, value)
            restored.append((key, field))
            touched.append(key)
    if "engine" in work:
        work["engine"] = new.engine
        touched.append("engine")
    corrected, corrections = correct_settings(work, new, titles)
    remember = {}
    for c in corrections:
        # a channel's fields are remembered one by one: the first correction of the key holds the value before any of them
        remember[(c.key, c.field)] = (c.old, c.new)
        touched.append(c.key)
    if remember:
        kept[old.engine] = remember
    else:
        kept.pop(old.engine, None)
    return EngineSwitch(settings=corrected, corrections=tuple(corrections), restored=tuple(restored),
                        memory=EngineMemory(kept=kept), before_settings=before, before_memory=memory,
                        touched=tuple(dict.fromkeys(touched)))


def go_back(switch, settings=None):
    """(settings, memory) after undoing `switch` (D7's "go back"): every setting the switch touched has the value it had before and
    the memory is as it was. `settings` is the present dict (default: the one the switch made); anything the switch did not touch
    is left as it is in it."""
    undone = dict(switch.settings if settings is None else settings)
    for key in switch.touched:
        if key in switch.before_settings:
            undone[key] = switch.before_settings[key]
    return undone, switch.before_memory


# --------------------------------------------------------------------------- #
# The correction applied to the things a render is made of (D20, D35)
# --------------------------------------------------------------------------- #
STATIC_RECORDS = (("ray", RaySettings), ("natural_termination", NaturalTerminationSettings), ("buddha_blur", BuddhaBlurSettings),
                  ("buddha_seed", BuddhaSeedSettings), ("engine", EngineSettings), ("mandelbrot", MandelbrotSettings))


def _correct_records(owner, engine, titles):
    """(owner with its static settings records corrected for `engine`, corrections): `owner` is a KeyframeData or a Project."""
    values = settings_to_dict(*[getattr(owner, name) for name, _cls in STATIC_RECORDS])
    corrected, corrections = correct_settings(values, engine, titles)
    if not corrections:
        return owner, []
    return owner.replace(**{name: settings_from_dict(cls, corrected) for name, cls in STATIC_RECORDS}), corrections


def _correct_obstacles(obstacles, engine, titles):
    """([obstacles with the effects `engine` accepts], corrections) for frozen shapes."""
    values = {f"obst:{o.id}:{field}": getattr(o, field) for o in obstacles for field in ("in_effect", "out_effect")}
    corrected, corrections = correct_settings(values, engine, titles)
    out = [o.with_values(in_effect=corrected[f"obst:{o.id}:in_effect"], out_effect=corrected[f"obst:{o.id}:out_effect"])
           if any(f"obst:{o.id}:" in c.key for c in corrections) else o for o in obstacles]
    return out, corrections


def _channel_snapshot(coloration, channel, c):
    """`coloration` (a ColorationSnapshot) with channel c replaced by the ChannelSettings `channel`."""
    def put(values, value):
        values = list(values)
        values[c] = value
        return tuple(values)
    pair = channel.metric_type == METRIC_TYPE_POINT_PAIR
    return coloration.replace(
        chan_src=put(coloration.chan_src, channel.chan_src), metric_types=put(coloration.metric_types, channel.metric_type),
        point_metric=put(coloration.point_metric, channel.metric_slot) if pair else coloration.point_metric,
        other_metric=coloration.other_metric if pair else put(coloration.other_metric, channel.metric_slot),
        point1_choice=put(coloration.point1_choice, channel.point1), point2_choice=put(coloration.point2_choice, channel.point2),
        buddha_points=put(coloration.buddha_points, channel.buddha_points),
        buddha_path_kinds=put(coloration.buddha_path_kinds, channel.buddha_path_kinds),
        buddha_compression=put(coloration.buddha_compression, channel.buddha_compression),
        buddha_brightness=put(coloration.buddha_brightness, channel.buddha_brightness),
        constant_choice=put(coloration.constant_choice, channel.constant))


def correct_frame(data, obstacles, colorations, titles=None):
    """(data, obstacles, colorations, corrections) of one frame - a KeyframeData, the frozen shapes and the tuple of
    ColorationSettings that `resolve_*` made for it - corrected for the engine `data` says (D20, D35): each is returned as it came
    when nothing in it needed correcting. This is what a render does after the keyframes have applied their values and before the
    frame is drawn or its margin text made. `titles` as in correct_settings (colorations are keyed `coloration:<position>`)."""
    engine = engine_description(data.engine.engine)
    data, corrections = _correct_records(data, engine, titles)
    obstacles, more = _correct_obstacles(obstacles, engine, titles)
    corrections += more
    fixed = []
    for i, coloration in enumerate(colorations):
        values = {f"coloration:{i}:chan{c}": coloration_channel_value(coloration, c) for c in range(3)}
        corrected, found = correct_settings(values, engine, {f"coloration:{i}": f"Coloration {i + 1}", **(titles or {})})
        corrections += found
        fixed.append(decompose_chans(coloration.color_model, [corrected[f"coloration:{i}:chan{c}"] for c in range(3)])
                     if found else coloration)
    return data, obstacles, tuple(fixed), corrections


def correct_project(project, titles=None):
    """(project, corrections): a loaded `Project` with everything its engine does not accept corrected (D20) - its static settings,
    the effects of its obstacles and the channels of its colorations. Its keyframes are left as they are: they may hold
    values that are invalid for an engine (D19), which are corrected when they are used."""
    engine = engine_description(project.engine.engine)
    project, corrections = _correct_records(project, engine, titles)
    obstacles, more = _correct_obstacles(project.obstacles, engine, titles)
    corrections += more
    colorations = []
    for coloration in project.colorations:
        values = {f"coloration:{coloration.id}:chan{c}": coloration_channel_value(coloration, c) for c in range(3)}
        corrected, found = correct_settings(values, engine, titles)
        corrections += found
        for c in range(3):
            coloration = _channel_snapshot(coloration, corrected[f"coloration:{coloration.id}:chan{c}"], c)
        colorations.append(coloration)
    return project.replace(obstacles=tuple(obstacles), colorations=tuple(colorations)), corrections


# --------------------------------------------------------------------------- #
# Keyframes that cannot all be right at once (D19)
# --------------------------------------------------------------------------- #
@strict_record
class KeyframeFix:
    """One change the keyframes need to be valid: keyframe `kf_id`'s value of `key` becomes `new`. `correction` words it."""
    kf_id: int
    key: str
    new: object
    correction: Correction


def _engine_timeline(work, live_engine):
    """(engine_at, reigns) for `work` ([[id, frame, data], ...] in the order of the frames): `reigns` is [(start, end, engine code)] for
    the keyframes that define the engine - the first reigns from the beginning (start None), each until the next (the last for ever,
    end None) - and engine_at(frame) the engine in effect at a frame (`live_engine` if no keyframe defines it)."""
    definers = [(frame, data["engine"], kf_id) for kf_id, frame, data in work if "engine" in data]
    reigns = [(None if k == 0 else definers[k][0], definers[k + 1][0] if k + 1 < len(definers) else None, definers[k][1], definers[k][2])
              for k in range(len(definers))]

    def engine_at(frame):
        for start, end, code, _kf_id in reigns:
            if (start is None or start <= frame) and (end is None or frame < end):
                return code
        return live_engine
    return engine_at, reigns


def _overlapping(reigns, lo, hi, live_engine):
    """[(engine code, keyframe id or None)] of the engines in effect somewhere on the frames from `lo` (None: from the start) up to,
    not including, `hi` (None: for ever)."""
    if not reigns:
        return [(live_engine, None)]
    return [(code, kf_id) for start, end, code, kf_id in reigns
            if (hi is None or start is None or start < hi) and (lo is None or end is None or end > lo)]


def keyframe_fixes(keyframes, live_engine, saved=frozenset(), is_float=None, titles=None, descriptions=None):
    """The changes that make `keyframes` (Keyframe records, as they would be after a save) valid for the engines they put in
    effect (D19): [KeyframeFix, ...], empty if they already are. A keyframe's value of a setting is in effect from its frame until
    the next keyframe that defines the setting (a number also from the one before, being blended with it; the first and last
    reach to the ends of time), and every engine in effect then must accept it. A value some engine does not accept is corrected
    for it - unless it is in `saved` ({(keyframe id, key)}: what the user is saving now, which the fixes leave alone if they can):
    then the keyframe that puts that engine in effect gets the engine in effect at the saved value's own frame instead - unless that
    engine is being saved too, or does not accept the value either, when the value is corrected after all. `is_float(key)` says
    whether a setting is blended (default: the static ones in SETTING_FLOAT_KEYS); `live_engine` is the engine when no keyframe
    defines it."""
    descriptions = ENGINE_DESCRIPTIONS if descriptions is None else descriptions
    is_float = (lambda key: key in SETTING_FLOAT_KEYS) if is_float is None else is_float
    titles = titles or {}
    work = [[kf.id, kf.frame_index, dict(kf.data)] for kf in sorted(keyframes, key=lambda k: k.frame_index)]
    first = {(kf.id, key): value for kf in keyframes for key, value in kf.data.items()}
    fixes = {}

    def record(kf_id, frame, key, new, old_text, new_text, label, old):
        kept = fixes.get((kf_id, key))
        correction = Correction(key=key, field="", tab=f"Keyframe at frame {frame}", label=label, old=first[(kf_id, key)] if kept is None else kept.correction.old,
                                new=new, old_text=old_text if kept is None else kept.correction.old_text, new_text=new_text)
        fixes[(kf_id, key)] = KeyframeFix(kf_id=kf_id, key=key, new=new, correction=correction)

    for _round in range(8):
        engine_at, reigns = _engine_timeline(work, live_engine)
        changed = False
        for entry in work:
            kf_id, frame, data = entry
            for key in [k for k in data if k != "engine"]:
                definers = [e for e in work if key in e[2]]
                at = definers.index(entry)
                lo = None if at == 0 else (definers[at - 1][1] if is_float(key) else frame)
                hi = definers[at + 1][1] if at + 1 < len(definers) else None
                for code, culprit in _overlapping(reigns, lo, hi, live_engine):
                    corrected, found = correct_settings({key: data[key]}, descriptions[code], titles)
                    if not found:
                        continue
                    instead = next((c for c in [engine_at(frame)] + sorted(descriptions)
                                    if c != code and not correct_settings({key: data[key]}, descriptions[c], titles)[1]), None)
                    if (kf_id, key) in saved and culprit is not None and (culprit, "engine") not in saved and instead is not None:
                        owner = next(e for e in work if e[0] == culprit)
                        owner[2]["engine"] = instead
                        record(culprit, owner[1], "engine", instead, ENGINE_NAMES[code], ENGINE_NAMES[instead], "Engine", code)
                    else:
                        data[key] = corrected[key]
                        record(kf_id, frame, key, corrected[key], found[0].old_text, found[0].new_text, found[0].label, found[0].old)
                    changed = True
                    break
        if not changed:
            break
    return [fix for fix in fixes.values() if first[(fix.kf_id, fix.key)] != fix.new]
