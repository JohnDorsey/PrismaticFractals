"""The settings model: the values of the static settings, held apart from any control that shows them (see dev/plans/Settings_Model_Design.md).

A view (the Tk window today; another front end, a script, or an audio node tomorrow) reports what the user typed with `submit` and
writes values with `set`; it reads them back with `value`, `error` and `record`. The model holds only VALUES: for each setting its last
valid value and the error message of the text last submitted (None when that text was valid). The text being typed is the view's.

Only the settings that have a row in a parser table are managed so far (the natural termination limits and confinement); the rest are
still held by their controls, and this module changes nothing for them.

Pure: imports no Tk."""
from .constants import (
    CONFINE_LOOKBACK_MAX, CONFINE_LOOKBACK_MIN, CONFINE_METHOD_NAMES, DIST_LIMIT_MAX, DIST_LIMIT_MIN, ITER_LIMIT_MAX, ITER_LIMIT_MIN
)
from .parsing import (
    parse_confine_lookback, parse_confine_threshold, parse_dist_limit, parse_escape_radius, parse_iter_limit
)
from .settings import DEFAULT_SETTING_VALUES, SETTING_SPECS, SETTING_TABLES, strict_record, settings_from_dict


@strict_record
class SettingParser:
    """How the text of one setting becomes its value: `parse` returns the value, or None for text that is not acceptable, which is then
    reported as `message`."""
    parse: object          # text -> value, or None
    message: str


def choice_parser(names):
    """A parser for a setting chosen from `names` (the text is a name; the value is its index)."""
    return lambda text: names.index(text) if text in names else None


# Every natural-termination setting (NaturalTerminationSettings), by key, with the sentence shown while its entry holds nonsense.
NATURAL_TERMINATION_PARSERS = {
    "iter": SettingParser(parse=parse_iter_limit,
                          message=f"Iteration limit must be a whole number from {ITER_LIMIT_MIN} to {ITER_LIMIT_MAX}."),
    "dist": SettingParser(parse=parse_dist_limit,
                          message=f"Distance limit must be a whole number from {DIST_LIMIT_MIN} to {DIST_LIMIT_MAX} (0 = no limit)."),
    "escape_radius": SettingParser(parse=parse_escape_radius,
                                   message="Escape radius must be a number greater than 0 (e.g. 2 or 1.5e3)."),
    "confine_lookback": SettingParser(parse=parse_confine_lookback,
                                      message=f"Lookback must be a whole number from {CONFINE_LOOKBACK_MIN} to {CONFINE_LOOKBACK_MAX}."),
    "confine_method": SettingParser(parse=choice_parser(CONFINE_METHOD_NAMES), message="Choose one of the detection methods."),
    "confine_threshold": SettingParser(parse=parse_confine_threshold,
                                       message="Threshold must be a number greater than 0 (any size, e.g. 0.004 or 1e-9)."),
}


def format_value(key, value):
    """The text that shows `value` of setting `key` in a control (what a view writes after `set`, and what committing an entry tidies
    its text to)."""
    spec = SETTING_SPECS[key]
    if spec.kind == "text":
        return str(value)
    if spec.kind == "int_text":
        return str(int(round(value)))
    if spec.kind == "float_text":
        return str(float(value))
    if spec.kind == "choice":
        return spec.names[int(round(value))]
    raise ValueError(f"setting {key!r} is shown as {spec.kind!r}, which is not text")


class SettingsModel:
    """The values of the settings that have a parser row. A listener (`subscribe`) is called with {key: origin} for each setting whose
    value or error changed, origin being "submit" (the user typed it) or "set" (a program wrote it: the view must then show it)."""

    def __init__(self, parsers):
        self._parsers = dict(parsers)
        self._values = {key: DEFAULT_SETTING_VALUES[key] for key in self._parsers}
        self._errors = {key: None for key in self._parsers}
        self._listeners = []
        self._batch_depth = 0
        self._pending = {}

    def keys(self):
        return list(self._parsers)

    def value(self, key):
        """The last valid value of `key`."""
        return self._values[key]

    def error(self, key):
        """The message for the text last submitted for `key`, or None if it was acceptable."""
        return self._errors[key]

    def record(self, record_class):
        """A `record_class` of the current values, or None while any of its settings has an error (the text being typed is not usable).
        Every setting of the record must be managed (KeyError otherwise: no partial records)."""
        keys = list(SETTING_TABLES[record_class])
        missing = [key for key in keys if key not in self._parsers]
        if missing:
            raise KeyError(f"{record_class.__name__} has settings the model does not manage: {missing}")
        if any(self._errors[key] is not None for key in keys):
            return None
        return settings_from_dict(record_class, self._values)

    def submit(self, key, text):
        """The user typed `text` for `key`: a valid text changes the value and clears the error; an invalid one leaves the value and sets
        the error. Returns the error (None when valid)."""
        return self._submit(key, text, "submit")

    def set(self, key, value):
        """A program writes `value` for `key` (a keyframe loads, Revert, a project opens): it goes through the same text and parser as
        anything typed, so it is rounded and checked exactly as an entry's text would be. Returns the error (None when valid)."""
        return self._submit(key, format_value(key, value), "set")

    def _submit(self, key, text, origin):
        parser = self._parsers[key]
        value = parser.parse(text)
        error = parser.message if value is None else None
        old = (self._values[key], self._errors[key])
        if value is not None:
            self._values[key] = value
        self._errors[key] = error
        if (self._values[key], error) != old or origin == "set":
            self._changed(key, origin)
        return error

    def subscribe(self, listener):
        self._listeners.append(listener)

    def batch(self):
        """A `with` block: the listeners are told once, at its end, of everything that changed inside it."""
        return _Batch(self)

    def _changed(self, key, origin):
        self._pending[key] = origin
        if self._batch_depth == 0:
            self._notify()

    def _notify(self):
        changes, self._pending = self._pending, {}
        if changes:
            for listener in list(self._listeners):
                listener(changes)


class _Batch:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model._batch_depth += 1
        return self.model

    def __exit__(self, *_exc):
        self.model._batch_depth -= 1
        if self.model._batch_depth == 0:
            self.model._notify()
        return False
