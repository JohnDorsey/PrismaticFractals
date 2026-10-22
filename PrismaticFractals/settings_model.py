"""The settings model: the values of the static settings, held apart from any control that shows them (see dev/plans/Settings_Model_Design.md).

A view (the Tk window today; another front end, a script, or an audio node tomorrow) reports what the user typed with `submit` and
writes values with `set`; it reads them back with `value`, `error` and `record`. The model holds only VALUES: for each setting its last
valid value and the error message of the text last submitted (None when that text was valid). The text being typed is the view's.

The model holds every static setting of `SETTING_SPECS`: each spec row carries the parser of its text (`parse`, `message`, `lenient`, `fmt`),
and a self-check refuses a row without one. The window's controls are views of it (`widgets.ModelBinding`); the keyframe, project and
export code read the records it makes.

Pure: imports no Tk."""
from .settings import DEFAULT_SETTING_VALUES, SETTING_SPECS, SETTING_TABLES, settings_from_dict


def format_value(key, value):
    """What a control shows for `value` of setting `key`: the text of an entry or dropdown, a checkbox's bool (what a view writes after
    `set`, and what committing an entry tidies its text to)."""
    spec = SETTING_SPECS[key]
    if spec.kind == "bool":
        return bool(value)
    if spec.kind == "text":
        return str(value)
    if spec.kind == "int_text":
        return str(int(round(value)))
    if spec.kind == "float_text":
        return spec.fmt.format(float(value)) if spec.fmt else str(float(value))
    if spec.kind == "choice":
        return spec.names[int(round(value))]
    raise ValueError(f"setting {key!r} is shown as {spec.kind!r}, which is not text")


class SettingsModel:
    """The values of the settings of `specs` ({key: SettingSpec with a parser}). A listener (`subscribe`) is called with {key: origin} for each setting whose
    value or error changed, origin being "submit" (the user typed it) or "set" (a program wrote it: the view must then show it)."""

    def __init__(self, specs):
        self._parsers = dict(specs)
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
        """A `record_class` of the current values, or None while any of its settings has an error (the text being typed is not usable;
        a lenient setting never counts: its last valid value stands). Every setting of the record must be managed (KeyError otherwise:
        no partial records)."""
        keys = self._record_keys(record_class)
        if any(self._errors[key] is not None and not self._parsers[key].lenient for key in keys):
            return None
        return settings_from_dict(record_class, self._values)

    def last_valid_record(self, record_class):
        """A `record_class` of the last valid values, whatever is being typed (never None): what a preview uses."""
        self._record_keys(record_class)
        return settings_from_dict(record_class, self._values)

    def _record_keys(self, record_class):
        keys = list(SETTING_TABLES[record_class])
        missing = [key for key in keys if key not in self._parsers]
        if missing:
            raise KeyError(f"{record_class.__name__} has settings the model does not manage: {missing}")
        return keys

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
        error = parser.message if value is None and (parser.message or not parser.lenient) else None
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
