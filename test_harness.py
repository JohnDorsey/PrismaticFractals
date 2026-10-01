"""
Shared test harness for the program (the `lightbounces` package).

The program is a Tkinter app, so importing it directly requires a
display. This module stubs out just enough of tkinter/ttk for the import to
succeed headlessly - every widget class becomes a no-op that accepts any
constructor arguments and swallows any method call - so test scripts can
import the real module and call its plain functions (trace_path,
resolve_frame_settings, the PNG/QOI encoders, etc.) directly, with no
display, no mainloop, and no real widgets involved.

Usage, from a test script in the same directory as light_bounces.py (and the lightbounces package):

    from test_harness import load_module
    m = load_module()
    print(m.parse_dist_limit("5.5"))

Only the module's plain functions and classes are safe to exercise this way.
Anything that actually needs a live widget (constructing an App and clicking
things) is out of scope for this harness - those parts need a real Tk
mainloop and are exercised by hand instead.
"""
import importlib.util
import os
import sys
import types

_STUBBED = False


def _install_stubs():
    """Install no-op tkinter/ttk/font/filedialog/messagebox modules into
    sys.modules, once per process. Safe to call more than once."""
    global _STUBBED
    if _STUBBED:
        return
    _STUBBED = True

    class _Var:
        """Stands in for StringVar/IntVar/BooleanVar/DoubleVar: a plain
        boxed value with .get()/.set(), no Tk variable machinery."""
        def __init__(self, value=None, *_a, **_k):
            self._v = value

        def get(self):
            return self._v

        def set(self, v):
            self._v = v

    class _Widget:
        """Stands in for any widget or other Tk object. Accepts any
        constructor args, and returns itself from any attribute access or
        call, so arbitrarily deep chains like `x.grid(...).configure(...)`
        never raise - the important thing is that constructing and wiring
        up widgets never fails, not that it does anything."""
        def __init__(self, *_a, **_k):
            pass

        def __getattr__(self, _name):
            return self

        def __call__(self, *_a, **_k):
            return self

        def __setitem__(self, _key, _value):
            pass

        def __getitem__(self, _key):
            return self

        def state(self, *_a, **_k):
            pass

    tk = types.ModuleType("tkinter")
    for name in ["Tk", "Frame", "Canvas", "Label", "Scale", "PhotoImage", "Text",
                "Scrollbar", "Menu", "Toplevel", "LabelFrame", "Button", "Entry",
                "Checkbutton", "Radiobutton", "Listbox", "Spinbox", "PanedWindow"]:
        setattr(tk, name, _Widget)
    tk.StringVar = _Var
    tk.IntVar = _Var
    tk.BooleanVar = _Var
    tk.DoubleVar = _Var
    tk.TclError = Exception
    tk.NORMAL = "normal"
    tk.DISABLED = "disabled"
    tk.END = "end"

    ttk = types.ModuleType("tkinter.ttk")
    for name in ["Frame", "LabelFrame", "Label", "Button", "Entry", "Combobox",
                "Checkbutton", "Radiobutton", "Notebook", "Scrollbar", "Progressbar",
                "Style", "Separator", "Treeview", "Spinbox", "PanedWindow"]:
        setattr(ttk, name, _Widget)

    font_mod = types.ModuleType("tkinter.font")
    font_mod.nametofont = lambda *_a, **_k: _Widget()
    font_mod.Font = _Widget

    filedialog = types.ModuleType("tkinter.filedialog")
    filedialog.asksaveasfilename = lambda *_a, **_k: ""
    filedialog.askopenfilename = lambda *_a, **_k: ""

    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showerror = lambda *_a, **_k: None
    messagebox.showwarning = lambda *_a, **_k: None
    messagebox.showinfo = lambda *_a, **_k: None
    messagebox.askyesno = lambda *_a, **_k: True
    messagebox.askyesnocancel = lambda *_a, **_k: False

    tk.ttk = ttk
    tk.font = font_mod
    tk.filedialog = filedialog
    tk.messagebox = messagebox

    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = ttk
    sys.modules["tkinter.font"] = font_mod
    sys.modules["tkinter.filedialog"] = filedialog
    sys.modules["tkinter.messagebox"] = messagebox


PACKAGE = "lightbounces"


def _package_modules():
    return {name: module for name, module in sys.modules.items()
            if name == PACKAGE or name.startswith(PACKAGE + ".")}


def load_module(path=None):
    """Import the program headlessly and return it as a module-like object: the `lightbounces` package, whose
    attributes include every name of every module of it (public and underscore). Safe to call multiple times
    (e.g. once per test function that needs a clean copy to break on purpose, like the deliberate-breakage
    checks in test_light_bounces.py); each call imports the WHOLE package afresh, so the copies are independent
    and tests that mutate module-level state don't affect each other.

    To PATCH a global of the program, patch the module that owns it (m.paths.EXPORT_DIR, m.external.ffmpeg):
    reading any name through the package works, but an assignment on the package would not reach the functions
    that read it. The copy's modules are kept in m.__lb_modules__ and are NOT left in sys.modules (so the next
    copy starts clean); tests that need them there (pickling) use _ModuleRegistered.

    `path` (for development scripts) loads one old single-file version of the program from that file."""
    _install_stubs()
    if path is not None:
        spec = importlib.util.spec_from_file_location("light_bounces", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    for name in _package_modules():
        del sys.modules[name]
    package = importlib.import_module(PACKAGE)
    package.__lb_modules__ = _package_modules()
    for name in package.__lb_modules__:
        del sys.modules[name]
    return package
