"""
Debug dump (development aid): pickles everything the running application
holds into one file, for inspecting later.

This file is deliberately standalone. It knows nothing about what the
application's attributes mean, only about kinds of Python values, so a new
attribute or setting added to the application is captured without touching
this file. PrismaticFractals.py loads it only if the dev/ folder exists (see
dump_app_state there), exactly like the startup screenshot.

What goes in: every attribute of the application object, and the module's own
non-code globals. A value that pickles is stored exactly as it is. A value that
cannot be pickled is replaced by a "stand-in", a plain dict marked with the key
"__standin__" that says what the value was and keeps whatever can be read from
it:
  * Tk variables (StringVar and friends): their type, name and current value
  * widgets: their type, Tk path name and configured options
  * functions, lambdas, bound methods: their qualified name
  * other objects that cannot be pickled: their type and their attributes,
    each encoded by the same rules
  * a container holding any of the above is rebuilt with only those items
    replaced
Stand-ins are plain dicts, so the dump loads without this file or the
application importing anything special.

Large arrays: unless `include_large_arrays` is true, every numpy array of at least
`large_array_bytes` (default LARGE_ARRAY_BYTES) found anywhere - in an attribute,
inside a container or object, in the module's data - is replaced by a stand-in
recording its shape, dtype and size. The rule is purely by size, so it needs no
list of which attributes hold the simulation's output or other big buffers.

Loading: use load_dump(path). A dump written while the program ran as a script
names its classes as belonging to "__main__"; load_dump maps that to the real
module name (default "PrismaticFractals", the package, which must be importable; the program's own classes
are recorded under their real module names, e.g. "PrismaticFractals.core", and need no mapping).
"""
import os
import pickle
import platform
import sys
import threading
import time
import traceback

FORMAT = 2
LARGE_ARRAY_BYTES = 64 * 1024
MAX_DEPTH = 40
REPR_LIMIT = 300


def standin(kind, **info):
    """A marked plain-dict replacement for a value that cannot be pickled."""
    result = {"__standin__": kind}
    result.update(info)
    return result


def _short_repr(value):
    try:
        text = repr(value)
    except Exception as exc:                        # noqa: BLE001
        text = f"<repr failed: {exc!r}>"
    return text if len(text) <= REPR_LIMIT else text[:REPR_LIMIT] + "..."


def _type_name(value):
    t = type(value)
    return f"{t.__module__}.{t.__qualname__}"


def _is_ndarray(value):
    t = type(value)
    return t.__name__ == "ndarray" and t.__module__ == "numpy"


def _is_tk_variable(value):
    t = type(value)
    return (callable(getattr(t, "get", None)) and callable(getattr(t, "set", None))
            and callable(getattr(t, "trace_add", None)) and hasattr(value, "_name"))


def _is_widget(value):
    t = type(value)
    return callable(getattr(t, "winfo_class", None)) and callable(getattr(t, "winfo_children", None))


def _widget_options(widget):
    try:
        return {key: str(spec[-1]) for key, spec in widget.configure().items()}
    except Exception:                               # noqa: BLE001
        return None


class Encoder:
    """Turns a value into something picklable, changing as little as possible."""

    def __init__(self, omit_bytes=None):
        self.omit_bytes = omit_bytes                # arrays at least this big are omitted
        self.active = set()                         # ids being descended into (cycles)

    def encode(self, value, depth=0):
        try:
            return self._encode(value, depth)
        except Exception as exc:                    # noqa: BLE001
            return standin("unreadable", type=_type_name(value), repr=_short_repr(value),
                           error=repr(exc))

    def _encode(self, value, depth):
        if self._is_large_array(value):
            return standin("large array", shape=tuple(value.shape),
                           dtype=str(value.dtype), nbytes=int(value.nbytes))
        if _is_tk_variable(value):
            try:
                current = value.get()
            except Exception as exc:                # noqa: BLE001
                current = f"<get failed: {exc!r}>"
            return standin("tk variable", type=_type_name(value), name=str(value._name),
                           value=current)
        if _is_widget(value):
            try:
                path = str(value)
            except Exception:                       # noqa: BLE001
                path = None
            return standin("widget", type=_type_name(value), path=path,
                           options=_widget_options(value))
        container = isinstance(value, (list, tuple, dict, set, frozenset)) or hasattr(value, "__dict__")
        if not (container and self._holds_large_array(value)):
            try:
                pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                return value                        # stored exactly as it is
            except Exception:                       # noqa: BLE001
                pass
        if id(value) in self.active:
            return standin("cycle", type=_type_name(value))
        if depth > MAX_DEPTH:
            return standin("too deep", type=_type_name(value), repr=_short_repr(value))
        self.active.add(id(value))
        try:
            return self._descend(value, depth + 1)
        finally:
            self.active.discard(id(value))

    def _descend(self, value, depth):
        if isinstance(value, dict):
            return {self._key(k): self.encode(v, depth) for k, v in value.items()}
        if isinstance(value, list):
            return [self.encode(v, depth) for v in value]
        if isinstance(value, tuple):
            items = [self.encode(v, depth) for v in value]
            if hasattr(value, "_fields"):
                try:
                    pickle.dumps(type(value))       # only rebuild a class that can be pickled
                    return type(value)(*items)
                except Exception:                   # noqa: BLE001
                    return standin("object", type=_type_name(value),
                                   state=dict(zip(value._fields, items)))
            return tuple(items)
        if isinstance(value, (set, frozenset)):
            return standin("set", items=[self.encode(v, depth) for v in value])
        if isinstance(value, type):
            return standin("class", name=_type_name(value))
        state = self._object_state(value)
        if state is not None:
            return standin("object", type=_type_name(value),
                           state={k: self.encode(v, depth) for k, v in state.items()})
        if callable(value):
            return standin("callable", type=_type_name(value),
                           name=getattr(value, "__qualname__", None) or _short_repr(value),
                           module=getattr(value, "__module__", None))
        return standin("unpicklable", type=_type_name(value), repr=_short_repr(value))

    def _is_large_array(self, value):
        return (self.omit_bytes is not None and _is_ndarray(value)
                and value.nbytes >= self.omit_bytes)

    def _holds_large_array(self, value, seen=None, depth=0):
        """Does an array to omit sit anywhere inside this container or object?
        (Decides whether the value can be stored exactly as it is.)"""
        if self.omit_bytes is None or depth > MAX_DEPTH:
            return False
        seen = set() if seen is None else seen
        if id(value) in seen:
            return False
        seen.add(id(value))
        if self._is_large_array(value):
            return True
        if isinstance(value, dict):
            children = list(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = list(value)
        elif isinstance(value, type) or _is_ndarray(value):
            return False
        else:
            state = self._object_state(value)
            children = list(state.values()) if state else []
        return any(self._holds_large_array(c, seen, depth + 1) for c in children)

    def encode_checked(self, value):
        """encode(), then make sure the result really pickles: whatever still does
        not (say, an instance of a class that cannot be found by name) becomes a
        stand-in, so one odd value can never cost the whole dump."""
        encoded = self.encode(value)
        try:
            pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
            return encoded
        except Exception as exc:                    # noqa: BLE001
            return standin("unpicklable", type=_type_name(value), repr=_short_repr(value),
                           error=repr(exc))

    @staticmethod
    def _key(key):
        try:
            pickle.dumps(key)
            return key
        except Exception:                           # noqa: BLE001
            return _short_repr(key)

    @staticmethod
    def _object_state(value):
        state = {}
        if hasattr(value, "__dict__"):
            state.update(vars(value))
        for cls in type(value).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                if isinstance(slot, str) and hasattr(value, slot):
                    state[slot] = getattr(value, slot)
        return state or None


def _module_data(module_globals):
    """The module's own data: no dunders, modules, classes or callables (code)."""
    import types
    out = {}
    for name, value in module_globals.items():
        if name.startswith("__") or isinstance(value, (types.ModuleType, type)):
            continue
        if callable(value):
            continue
        out[name] = value
    return out


def make_dump(app, module_globals, cause, include_large_arrays=True, exc_info=None,
              file_version=None, large_array_bytes=LARGE_ARRAY_BYTES):
    """The dump as a dict of plain, picklable data (see the module docstring)."""
    omit = None if include_large_arrays else large_array_bytes
    app_state = {name: Encoder(omit).encode_checked(value)
                 for name, value in list(getattr(app, "__dict__", {}).items())}
    module_state = {name: Encoder(omit).encode_checked(value)
                    for name, value in _module_data(module_globals).items()}
    threads = [{"name": t.name, "alive": t.is_alive(), "daemon": t.daemon}
               for t in threading.enumerate()]
    meta = {"format": FORMAT, "cause": cause, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "file_version": file_version, "large_arrays_included": bool(include_large_arrays),
            "large_array_bytes": large_array_bytes, "python": sys.version,
            "platform": platform.platform(), "app_type": _type_name(app)}
    if exc_info is not None and exc_info[0] is not None:
        meta["exception"] = "".join(traceback.format_exception(*exc_info))
    return {"meta": meta, "app": app_state, "module": module_state, "threads": threads}


def dump_everything(app, module_globals, path, cause, include_large_arrays=True,
                    exc_info=None, file_version=None, large_array_bytes=LARGE_ARRAY_BYTES):
    """Write make_dump(...) to `path` (its folder must exist); returns the path."""
    dump = make_dump(app, module_globals, cause, include_large_arrays, exc_info,
                     file_version, large_array_bytes)
    temporary = path + ".tmp"
    with open(temporary, "wb") as f:
        pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return path


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, module_name):
        super().__init__(f)
        self.module_name = module_name

    def find_class(self, module, name):
        if module == "__main__":
            module = self.module_name
        return super().find_class(module, name)


def load_dump(path, module_name="PrismaticFractals"):
    """Read a dump back. Classes recorded as belonging to "__main__" (the program
    run as a script) are looked up in `module_name` instead."""
    with open(path, "rb") as f:
        return _Unpickler(f, module_name).load()
