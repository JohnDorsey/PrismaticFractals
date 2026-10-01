"""Consistency checks between tables that must independently agree. Each setting, field, termination reason, shape kind, catalog... has to be
listed correctly in several separate places, and nothing else enforces that agreement. run_self_checks() runs once when the package is
imported (see lightbounces/__init__.py, after every module is loaded), so a table that is missing an entry fails loudly then, rather than
silently producing a KeyError or a wrong value the first time that path is hit. Tests break a fresh copy of the package on purpose to
confirm that each check fires (they rebind the table on THIS module, which is where the checks read it)."""
from .constants import (
    LAYOUTS, OBSTACLE_EFFECT_NAMES, TERM_CODES, TERM_END_COLOR, TERM_OUTCOME_TEXT,
    TERM_STAT_VALUE
)
from .kernels import N_STATS
from .metrics import (
    BUDDHA_BRIGHTNESS_OPTIONS, BUDDHA_COMPRESSION_LABELS, BUDDHA_POINT_SHORT,
    CHAN_CONSTANT_LABELS, CHAN_SRC_BUDDHA, CHAN_SRC_CONSTANT, CHAN_SRC_LABELS, CHAN_SRC_QUERY,
    CONSTANT_POINT_COORDS, DEFAULT_BUDDHA_POINTS, KIND_OTHER, KIND_POINT, NONE_ONE, NONE_ZERO,
    N_BUDDHA_POINTS, N_POINT_CHOICES, POINT_DESCRIPTIONS, POINT_LABELS, POINT_MENU_ORDER,
    POINT_ORIGIN, POINT_SHORT, POINT_TOP_LEFT_CORNER
)
from .projects import (
    CAT_BUDDHA_BRIGHTNESS, CAT_BUDDHA_COMPRESSION, CAT_BUDDHA_POINT, CAT_CHAN_KIND,
    CAT_CHAN_SOURCE, CAT_CONSTANT, CAT_EFFECT, CAT_METRIC, CAT_POINT, PROJECT_KEY_PATTERN,
    SETTING_CHOICE_CATALOGS
)
from .settings import (
    DEFAULT_SETTINGS, DEFAULT_SETTING_VALUES, SETTING_FLOAT_KEYS, SETTING_INFO, SETTING_SPECS,
    SETTING_TABLES, SETTING_WIDGET_ATTRS, record_leaf_paths
)
from .shapes import SHAPE_CLASSES
from .views import DEFAULT_BUDDHA_POINTS_PER_CHAN, ViewConfig


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
    choice_settings = {key for table in SETTING_TABLES.values()
                       for key, spec in table.items() if spec.kind == "choice"}
    if set(SETTING_CHOICE_CATALOGS) != choice_settings:
        raise AssertionError("SETTING_CHOICE_CATALOGS and the choice settings in SETTING_TABLES "
                             f"disagree: {set(SETTING_CHOICE_CATALOGS) ^ choice_settings}")
    for key, catalog in SETTING_CHOICE_CATALOGS.items():
        if set(catalog.codes) != set(range(len(SETTING_SPECS[key].names))):
            raise AssertionError(f"the catalog of the {key} setting does not cover its choices")
    if set(SETTING_WIDGET_ATTRS) != info_keys:
        raise AssertionError(
            "SETTING_WIDGET_ATTRS and SETTING_INFO disagree - "
            f"only in SETTING_WIDGET_ATTRS: {set(SETTING_WIDGET_ATTRS) - info_keys or None}; "
            f"only in SETTING_INFO: {info_keys - set(SETTING_WIDGET_ATTRS) or None}")
    table_keys = {key for table in SETTING_TABLES.values() for key in table}
    if table_keys != info_keys - {"outline"}:
        raise AssertionError(
            "SETTING_TABLES and SETTING_INFO disagree - "
            f"only in SETTING_TABLES: {table_keys - info_keys or None}; "
            f"only in SETTING_INFO: {info_keys - {'outline'} - table_keys or None}")
    if sum(len(table) for table in SETTING_TABLES.values()) != len(table_keys):
        raise AssertionError("a setting key is in more than one table of SETTING_TABLES: "
                             "keys name one setting each, in keyframes and project files")
    if set(SETTING_TABLES) != set(DEFAULT_SETTINGS):
        raise AssertionError("SETTING_TABLES and DEFAULT_SETTINGS are not about the same records")
    for record_class, table in SETTING_TABLES.items():
        spec_paths = sorted(spec.path for spec in table.values())
        if spec_paths != sorted(record_leaf_paths(record_class)):
            leaves = set(record_leaf_paths(record_class))
            raise AssertionError(
                f"the {record_class.__name__} fields and the rows of its settings table disagree - "
                f"fields without a row: {leaves - set(spec_paths) or None}; "
                f"rows without a field: {set(spec_paths) - leaves or None}")
        bad_kinds = [k for k, spec in table.items()
                     if spec.kind not in ("text", "int_text", "float_text", "number", "bool",
                                          "choice")
                     or bool(spec.names) != (spec.kind == "choice")]
        if bad_kinds:
            raise AssertionError(f"settings rows of {record_class.__name__} with a bad kind or "
                                 f"names: {bad_kinds}")
