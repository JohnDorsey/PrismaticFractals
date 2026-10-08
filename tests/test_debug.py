"""Tests: debug."""
import os
import shutil
import sys
import tempfile
import threading
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# Startup screenshot (development aid): once per launch, and only if a dev/
# folder exists next to the program, the app grabs a picture of its own
# window into dev/screenshots/<FILE_VERSION>-YYYYMMDD-hhmmss.png
# --------------------------------------------------------------------------- #
class _FakeGrabbedImage:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = path
        with open(path, "wb") as f:
            f.write(b"fake png bytes")


class _FakeImageGrab:
    def __init__(self):
        self.bboxes = []
        self.images = []

    def grab(self, bbox=None):
        self.bboxes.append(bbox)
        img = _FakeGrabbedImage()
        self.images.append(img)
        return img


class _FakeGeometryRoot:
    """winfo_* values a real root would report: window at (100, 50), 1500x900."""
    updates = 0

    def update(self):
        self.updates += 1

    def winfo_rootx(self):
        return 100

    def winfo_rooty(self):
        return 50

    def winfo_width(self):
        return 1500

    def winfo_height(self):
        return 900


class _ScreenshotEnv:
    """Points m.paths.DEV_DIR at a scratch directory (created only if
    `with_dev_dir`) and m.external.ImageGrab at a fake (or None), restoring both."""
    def __init__(self, with_dev_dir=True, grab="fake"):
        self.with_dev_dir = with_dev_dir
        self.grab = _FakeImageGrab() if grab == "fake" else grab

    def __enter__(self):
        self.parent = tempfile.mkdtemp()
        self.dev_dir = os.path.join(self.parent, "dev")
        if self.with_dev_dir:
            os.makedirs(self.dev_dir)
        self.orig_dev, self.orig_grab = m.paths.DEV_DIR, m.external.ImageGrab
        m.paths.DEV_DIR, m.external.ImageGrab = self.dev_dir, self.grab
        return self

    def __exit__(self, *exc):
        m.paths.DEV_DIR, m.external.ImageGrab = self.orig_dev, self.orig_grab


def test_dev_dir_is_dev_next_to_the_program():
    assert m.paths.DEV_DIR == os.path.join(m.SCRIPT_DIR, "dev")


def test_screenshot_is_skipped_without_a_dev_dir():
    with _ScreenshotEnv(with_dev_dir=False) as env:
        assert m.take_ui_screenshot(_FakeGeometryRoot(), "startup") is None
        assert env.grab.bboxes == [], "must not even attempt a grab"
        assert not os.path.exists(env.dev_dir), "must not create dev/ itself"


def test_screenshot_is_skipped_without_pillow():
    with _ScreenshotEnv(grab=None) as env:
        assert m.take_ui_screenshot(_FakeGeometryRoot(), "manual") is None
        assert not os.path.exists(os.path.join(env.dev_dir, "screenshots"))


def test_screenshot_grabs_the_windows_own_rectangle_and_saves_it_named_by_version_and_cause():
    import re
    root = _FakeGeometryRoot()
    with _ScreenshotEnv() as env:
        path = m.take_ui_screenshot(root, "startup")
        shots_dir = os.path.join(env.dev_dir, "screenshots")
        assert os.path.dirname(path) == shots_dir
        assert re.fullmatch(rf"{m.FILE_VERSION}-\d{{8}}-\d{{6}}-startup\.png",
                            os.path.basename(path)), os.path.basename(path)
        assert os.path.isfile(path)
        assert env.grab.bboxes == [(100, 50, 1600, 950)]
        assert env.grab.images[0].saved_to == path
    assert root.updates >= 1, "must let Tk finish drawing before grabbing"


def test_startup_screenshot_happens_once_per_launch():
    class FakeApp:
        pass
    app = FakeApp()
    app.root = _FakeGeometryRoot()
    app.take_startup_screenshot = m.App.take_startup_screenshot.__get__(app)
    with _ScreenshotEnv() as env:
        app.take_startup_screenshot()
        app.take_startup_screenshot()
        assert len(env.grab.bboxes) == 1


def test_startup_screenshot_failure_never_breaks_the_app():
    class BoomGrab:
        def grab(self, bbox=None):
            raise OSError("screen unavailable")
    class FakeApp:
        pass
    app = FakeApp()
    app.root = _FakeGeometryRoot()
    app.take_startup_screenshot = m.App.take_startup_screenshot.__get__(app)
    with _ScreenshotEnv(grab=BoomGrab()):
        app.take_startup_screenshot()      # must not raise


class _Plain:
    """Picklable, with attributes that are not."""
    def __init__(self):
        self.count = 3
        self.var = _FakeVar()
        self.helper = lambda: None


def _load_test_dump(path):
    """Test doubles live in the test runner (__main__), so map __main__ to itself."""
    with _ModuleRegistered():
        return _load_dumper().load_dump(path, module_name="__main__")


def _dump_roundtrip(app, cause="manual", include_large_arrays=True, exc_info=None,
                    module_globals=None, large_array_bytes=None):
    import tempfile
    dumper = _load_dumper()
    path = os.path.join(tempfile.mkdtemp(), "x.pkl")
    with _ModuleRegistered():
        extra = {} if large_array_bytes is None else {"large_array_bytes": large_array_bytes}
        dumper.dump_everything(app, module_globals or {}, path, cause,
                               include_large_arrays=include_large_arrays, exc_info=exc_info,
                               file_version=7, **extra)
    return _load_test_dump(path)


def test_dump_keeps_picklable_values_exactly_as_they_are():
    class App:
        pass
    app = App()
    app.arr = np.arange(12.0).reshape(3, 4)
    app.nested = {"a": [1, (2, 3)], "b": m.EllipseShape(4), "c": m.DEFAULT_RAY_SETTINGS}
    app.obstacles = [m.EllipseShape(1), m.EllipseShape(2)]
    app.obstacles[1].r = 0.75
    data = _dump_roundtrip(app)["app"]
    assert np.array_equal(data["arr"], app.arr)
    assert data["nested"]["a"] == [1, (2, 3)]
    assert data["nested"]["b"].id == 4 and data["nested"]["c"] == m.DEFAULT_RAY_SETTINGS
    assert [o.r for o in data["obstacles"]] == [0.0, 0.75]
    assert isinstance(data["obstacles"][0], m.EllipseShape)       # still the real class


def test_dump_replaces_what_cannot_be_pickled_with_readable_stand_ins():
    class App:
        pass
    app = App()
    app.var = _FakeVar()
    app.widget = _FakeWidget()
    app.callback = lambda: None
    app.plain = _Plain()
    app.mixed = {"ok": 1, "bad": [_FakeVar(), 5, lambda: None]}
    app.handle = m.FloatEntryHandle(_FakeVar(), "{:.2f}", 1.0)
    data = _dump_roundtrip(app)["app"]
    assert data["var"] == {"__standin__": "tk variable", "type": data["var"]["type"],
                           "name": "PY_VAR7", "value": "256"}
    assert data["widget"]["__standin__"] == "widget" and data["widget"]["path"] == ".frame.entry"
    assert data["widget"]["options"] == {"width": "8", "text": "hi"}
    assert data["callback"]["__standin__"] == "callable"
    assert "lambda" in data["callback"]["name"]
    obj = data["plain"]
    assert obj["__standin__"] == "object" and obj["state"]["count"] == 3
    assert obj["state"]["var"]["value"] == "256"
    assert data["mixed"]["ok"] == 1
    assert data["mixed"]["bad"][0]["__standin__"] == "tk variable" and data["mixed"]["bad"][1] == 5
    assert data["mixed"]["bad"][2]["__standin__"] == "callable"
    assert data["handle"]["__standin__"] == "object"            # structure kept, not flattened
    assert data["handle"]["state"]["fmt"] == "{:.2f}"


def test_dump_survives_cycles_and_objects_that_fail_when_read():
    class App:
        pass

    class Hostile:
        def __init__(self):
            self.fn = lambda: None
            self.me = self

        def __repr__(self):
            raise RuntimeError("no repr")
    app = App()
    app.loop = {"self": None, "fn": lambda: None}
    app.loop["self"] = app.loop
    app.hostile = Hostile()
    data = _dump_roundtrip(app)["app"]
    assert data["loop"]["self"]["__standin__"] == "cycle"
    assert data["hostile"]["state"]["me"]["__standin__"] == "cycle"


class _Holder:
    """Picklable, with a large array inside."""
    def __init__(self, big):
        self.big = big
        self.label = "held"


def test_dump_without_large_arrays_replaces_big_arrays_wherever_they_are():
    class App:
        pass
    app = App()
    big = np.zeros((200, 200), dtype=np.float64)              # 320 KB
    app.top = big
    app.in_list = [big, None, np.arange(5.0)]
    app.in_dict = {"k": {"deep": big}}
    app.in_object = _Holder(big)
    app.small = np.arange(5.0)
    app.obstacles = [m.EllipseShape(1), m.EllipseShape(2)]
    data = _dump_roundtrip(app, include_large_arrays=False, module_globals={"GRID": big, "SMALL": np.ones(3)})
    want = {"__standin__": "large array", "shape": (200, 200), "dtype": "float64",
            "nbytes": 320000}
    assert data["app"]["top"] == want
    assert data["app"]["in_list"][0] == want and data["app"]["in_list"][1] is None
    assert np.array_equal(data["app"]["in_list"][2], np.arange(5.0))     # small: kept
    assert data["app"]["in_dict"]["k"]["deep"] == want
    held = data["app"]["in_object"]
    assert held["__standin__"] == "object" and held["state"]["big"] == want
    assert held["state"]["label"] == "held"
    assert np.array_equal(data["app"]["small"], np.arange(5.0))
    assert data["module"]["GRID"] == want and np.array_equal(data["module"]["SMALL"], np.ones(3))
    # values with no large array inside are still stored exactly, real classes and all
    assert [type(o).__name__ for o in data["app"]["obstacles"]] == ["EllipseShape", "EllipseShape"]
    assert data["meta"]["large_arrays_included"] is False
    assert data["meta"]["large_array_bytes"] == 64 * 1024


def test_dump_with_large_arrays_keeps_them_and_the_size_limit_can_be_changed():
    class App:
        pass
    app = App()
    app.top = np.zeros((200, 200))
    app.mid = np.zeros(1000)                                  # 8 KB
    full = _dump_roundtrip(app, include_large_arrays=True)
    assert np.array_equal(full["app"]["top"], app.top) and full["meta"]["large_arrays_included"]
    tiny = _dump_roundtrip(app, include_large_arrays=False, large_array_bytes=1000)
    assert tiny["app"]["mid"]["__standin__"] == "large array"


def test_dump_records_the_cause_the_exception_the_threads_and_module_data():
    class App:
        pass
    try:
        raise KeyError("boom")
    except KeyError:
        exc_info = sys.exc_info()
    data = _dump_roundtrip(App(), cause="crash", exc_info=exc_info,
                           module_globals={"LIMIT": 5, "helper": len, "__name__": "x", "np": np,
                                           "TABLE": {"a": _FakeVar()}})
    assert data["meta"]["cause"] == "crash" and "KeyError: 'boom'" in data["meta"]["exception"]
    assert data["meta"]["file_version"] == 7
    assert any(t["name"] == threading.current_thread().name for t in data["threads"])
    assert set(data["module"]) == {"LIMIT", "TABLE"}           # no dunders, modules, callables
    assert data["module"]["TABLE"]["a"]["__standin__"] == "tk variable"


def test_a_dump_written_by_a_script_loads_by_mapping___main___to_the_module():
    import tempfile, types
    main = sys.modules["__main__"]

    class Zed:
        def __init__(self):
            self.n = 9
    Zed.__module__ = "__main__"
    Zed.__qualname__ = "Zed"
    main.Zed = Zed
    try:
        class App:
            pass
        app = App()
        app.zed = Zed()
        dumper = _load_dumper()
        path = os.path.join(tempfile.mkdtemp(), "x.pkl")
        dumper.dump_everything(app, {}, path, "manual")
    finally:
        del main.Zed
    fake_module = types.ModuleType("zed_module")
    fake_module.Zed = Zed
    sys.modules["zed_module"] = fake_module
    try:
        assert dumper.load_dump(path, module_name="zed_module")["app"]["zed"].n == 9
    finally:
        del sys.modules["zed_module"]


def test_one_unpicklable_value_never_costs_the_whole_dump():
    """Regression guard found while writing the dumper: a namedtuple of a class
    pickle cannot find by name was rebuilt as that same class, and the final
    write then failed for the entire dump. Such values become stand-ins."""
    import collections
    Lost = collections.namedtuple("Lost", ["a", "b"])
    Lost.__module__ = "no_such_module_anywhere"

    class App:
        pass
    app = App()
    app.lost = Lost(1, _FakeVar())
    app.fine = 5
    data = _dump_roundtrip(app)["app"]
    assert data["fine"] == 5
    assert data["lost"]["__standin__"] == "object" and data["lost"]["state"]["a"] == 1
    assert data["lost"]["state"]["b"]["__standin__"] == "tk variable"


def _fake_debug_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.root = _FakeGeometryRoot()
    app._crash_dumps = 0
    app.keep_window_on_screen = lambda: None
    app.characteristic_points = np.zeros((200, 200, 2), dtype=np.float32)   # 320 KB: a large array
    app.iter_var = _FakeVar()
    for name in ("take_startup_screenshot", "take_startup_captures", "manual_screenshot",
                 "manual_dump", "report_debug_result", "on_callback_exception", "dump_crash"):
        setattr(app, name, getattr(m.App, name).__get__(app))
    return app


class _DumpEnv(_ScreenshotEnv):
    """A scratch dev/ folder containing the real dumper."""
    def __enter__(self):
        import shutil
        super().__enter__()
        shutil.copy(os.path.join(ROOT, "dev",
                                 "debug_dump.py"), self.dev_dir)
        return self

    def dumps(self):
        folder = os.path.join(self.dev_dir, "dumps")
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def test_captures_in_the_same_second_do_not_overwrite_each_other():
    with tempfile.TemporaryDirectory() as d:
        first = m.unused_path(os.path.join(d, "a.pkl"))
        assert first == os.path.join(d, "a.pkl")
        open(first, "w").close()
        second = m.unused_path(os.path.join(d, "a.pkl"))
        assert second == os.path.join(d, "a-2.pkl")
        open(second, "w").close()
        assert m.unused_path(os.path.join(d, "a.pkl")) == os.path.join(d, "a-3.pkl")


def test_dump_file_names_carry_the_version_time_and_cause():
    import re
    app = _fake_debug_app()
    with _DumpEnv() as env:
        for cause in ("startup", "crash", "manual"):
            path = m.debug.dump_app_state(app, cause, False)
            assert re.fullmatch(rf"{m.FILE_VERSION}-\d{{8}}-\d{{6}}-{cause}\.pkl",
                                os.path.basename(path)), path
            assert os.path.dirname(path) == os.path.join(env.dev_dir, "dumps")
        try:
            m.debug.dump_app_state(app, "other", False)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_dumps_are_skipped_without_a_dev_dir_or_the_dumper_and_never_raise():
    app = _fake_debug_app()
    with _ScreenshotEnv(with_dev_dir=False) as env:
        assert m.debug.dump_app_state(app, "manual", True) is None
        assert not os.path.exists(env.dev_dir)
    with _ScreenshotEnv() as env:                      # dev/ but no dumper in it
        assert m.debug.dump_app_state(app, "manual", True) is None
    with _DumpEnv() as env:                            # a dumper that blows up
        with open(os.path.join(env.dev_dir, "debug_dump.py"), "w") as f:
            f.write("raise RuntimeError('broken dumper')\n")
        assert m.debug.dump_app_state(app, "manual", True) is None


def test_startup_captures_take_a_screenshot_then_a_dump_without_large_arrays_once():
    app = _fake_debug_app()
    with _DumpEnv() as env:
        app.take_startup_captures()
        app.take_startup_captures()
        assert len(env.grab.bboxes) == 1
        (name,) = env.dumps()
        assert name.endswith("-startup.pkl")
        shots = os.listdir(os.path.join(env.dev_dir, "screenshots"))
        assert len(shots) == 1 and shots[0].endswith("-startup.png")
        data = _load_test_dump(os.path.join(env.dev_dir, "dumps", name))
        assert data["meta"]["cause"] == "startup" and data["meta"]["large_arrays_included"] is False
        assert data["app"]["characteristic_points"]["__standin__"] == "large array"
        assert data["app"]["iter_var"]["value"] == "256"


def test_a_failing_startup_screenshot_still_leaves_the_startup_dump():
    class BoomGrab:
        def grab(self, bbox=None):
            raise OSError("screen unavailable")
    app = _fake_debug_app()
    with _DumpEnv(grab=BoomGrab()) as env:
        app.take_startup_captures()
        assert len(env.dumps()) == 1


def test_manual_actions_save_with_the_manual_suffix_and_report_the_result():
    shown = []
    saved = m.messagebox.showinfo
    m.messagebox.showinfo = lambda title, text: shown.append(text)
    try:
        app = _fake_debug_app()
        with _DumpEnv() as env:
            app.manual_screenshot()
            app.manual_dump(False)
            app.manual_dump(True)
            dumps = env.dumps()
            import re
            assert len(dumps) == 2, dumps                   # never overwritten, even in one second
            assert all(re.fullmatch(r"\d+-\d{8}-\d{6}-manual(-\d+)?\.pkl", d) for d in dumps)
            assert os.listdir(os.path.join(env.dev_dir, "screenshots"))[0].endswith("-manual.png")
            assert len(shown) == 3 and all("Saved" in s for s in shown)
            included = {_load_test_dump(os.path.join(env.dev_dir, "dumps", d))
                        ["meta"]["large_arrays_included"] for d in dumps}
            assert included == {False, True}
        shown.clear()
        with _ScreenshotEnv(with_dev_dir=False):
            app.manual_screenshot()
            app.manual_dump(True)
        assert len(shown) == 2 and all("No " in s for s in shown)
    finally:
        m.messagebox.showinfo = saved


def test_a_crash_leaves_a_dump_without_large_arrays_and_no_screenshot():
    app = _fake_debug_app()
    try:
        raise ValueError("kaboom")
    except ValueError:
        exc_info = sys.exc_info()
    import io, contextlib
    with _DumpEnv() as env, contextlib.redirect_stderr(io.StringIO()) as err:
        app.on_callback_exception(*exc_info)
        assert env.grab.bboxes == [], "a crash must not trigger a screenshot"
        assert not os.path.exists(os.path.join(env.dev_dir, "screenshots"))
        (name,) = env.dumps()
        assert name.endswith("-crash.pkl")
        data = _load_test_dump(os.path.join(env.dev_dir, "dumps", name))
        assert data["meta"]["large_arrays_included"] is False
        assert data["app"]["characteristic_points"]["__standin__"] == "large array"
        assert "ValueError: kaboom" in data["meta"]["exception"]
    assert "Exception in Tkinter callback" in err.getvalue() and "kaboom" in err.getvalue()


def test_crash_dumps_are_capped_per_launch():
    app = _fake_debug_app()
    try:
        raise ValueError("again")
    except ValueError:
        exc_info = sys.exc_info()
    import io, contextlib, time as _time
    calls = []
    saved = m.debug.dump_app_state
    m.debug.dump_app_state = lambda *a, **k: calls.append(a[1])
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            for _ in range(m.MAX_CRASH_DUMPS + 4):
                app.on_callback_exception(*exc_info)
    finally:
        m.debug.dump_app_state = saved
    assert calls == ["crash"] * m.MAX_CRASH_DUMPS


def test_the_menu_bar_has_a_file_menu_and_a_debug_menu_with_the_three_actions():
    class Menu:
        made = []

        def __init__(self, parent=None, **kw):
            self.commands, self.cascades = [], []
            Menu.made.append(self)

        def add_command(self, label, command, accelerator=None):
            self.commands.append((label, command))

        def add_separator(self):
            pass

        def add_cascade(self, label, menu):
            self.cascades.append((label, menu))

    class Root:
        menu = None

        def config(self, **kw):
            Root.menu = kw.get("menu")

    class FakeApp:
        root = Root()
        new_project = staticmethod(lambda: "new")
        open_project = staticmethod(lambda: "open")
        save_project = staticmethod(lambda: "save")
        save_project_as = staticmethod(lambda: "save as")
        manual_screenshot = staticmethod(lambda: "shot")
        manual_dump = staticmethod(lambda with_arrays: with_arrays)
    saved = m.tk.Menu
    m.tk.Menu = Menu
    try:
        m.App.build_menu_bar(FakeApp())
    finally:
        m.tk.Menu = saved
    bar = Root.menu
    assert [label for label, _ in bar.cascades] == ["File", "Debug"]
    file_menu = bar.cascades[0][1]
    assert [(label, command()) for label, command in file_menu.commands] == [
        ("New project", "new"), ("Open project...", "open"), ("Save project", "save"),
        ("Save project as...", "save as")]
    debug = bar.cascades[1][1]
    assert [label for label, _ in debug.commands] == [
        "Take screenshot", "Dump (without large arrays)", "Dump (with large arrays)"]
    assert debug.commands[0][1]() == "shot"
    assert debug.commands[1][1]() is False and debug.commands[2][1]() is True


# --------------------------------------------------------------------------- #
# Keeping the window on the screen at startup
# --------------------------------------------------------------------------- #
def test_shift_into_area_moves_a_window_only_as_far_as_needed():
    area = (0, 0, 1920, 1080)
    assert m.shift_into_area((0, 104, 1920, 1099), area) == (0, -19)       # the real case
    assert m.shift_into_area((100, 50, 900, 600), area) == (0, 0)          # already inside
    assert m.shift_into_area((1500, 200, 2100, 700), area) == (-180, 0)
    assert m.shift_into_area((-30, -40, 500, 400), area) == (30, 40)
    assert m.shift_into_area((1800, 1000, 2000, 1200), area) == (-80, -120)
    assert m.shift_into_area((0, 0, 1920, 1080), area) == (0, 0)           # exact fit


def test_a_window_bigger_than_the_area_lines_up_by_its_top_left_corner():
    area = (0, 0, 1000, 800)
    dx, dy = m.shift_into_area((50, 60, 1250, 1060), area)                 # 1200 x 1000
    assert (50 + dx, 60 + dy) == (0, 0)
    dx, dy = m.shift_into_area((-20, 500, 1180, 1500), (100, 40, 1100, 840))
    assert (-20 + dx, 500 + dy) == (100, 40)


def test_a_work_area_that_does_not_start_at_the_origin_is_respected():
    area = (1920, 0, 3840, 1040)                                           # a second monitor
    assert m.shift_into_area((1900, 100, 3000, 1100), area) == (20, -60)


def test_geometry_strings_give_the_window_position():
    assert m.parse_geometry_position("1904x936+0+104") == (0, 104)
    assert m.parse_geometry_position("1904x936+-8+-3") == (-8, -3)
    assert m.parse_geometry_position("1904x936-8+10") == (-8, 10)
    assert m.parse_geometry_position("1904x936") is None
    assert m.parse_geometry_position("") is None


class _FakePlacedRoot:
    def __init__(self, geometry, rect, screen=(1920, 1080)):
        self.geometry_text = geometry
        x, y, r, b = rect
        self.x, self.y, self.w, self.h = x, y, r - x, b - y
        self.screen = screen
        self.set_to = []

    def update_idletasks(self):
        pass

    def winfo_rootx(self):
        return self.x

    def winfo_rooty(self):
        return self.y

    def winfo_width(self):
        return self.w

    def winfo_height(self):
        return self.h

    def winfo_screenwidth(self):
        return self.screen[0]

    def winfo_screenheight(self):
        return self.screen[1]

    def geometry(self, text=None):
        if text is None:
            return self.geometry_text
        self.set_to.append(text)


def _placed_app(root):
    class FakeApp:
        pass
    app = FakeApp()
    app.root = root
    app.keep_window_on_screen = m.App.keep_window_on_screen.__get__(app)
    return app


def test_the_window_is_moved_up_when_its_bottom_hangs_off_the_screen():
    saved = sys.platform
    sys.platform = "linux"                      # the portable path: client rect, whole screen
    try:
        root = _FakePlacedRoot("1904x936+0+104", (0, 104, 1904, 1099))
        _placed_app(root).keep_window_on_screen()
    finally:
        sys.platform = saved
    assert root.set_to == ["+0+85"]


def test_a_window_already_on_screen_is_left_alone_and_never_resized():
    saved = sys.platform
    sys.platform = "linux"
    try:
        root = _FakePlacedRoot("1904x936+10+20", (10, 20, 1914, 956))
        _placed_app(root).keep_window_on_screen()
    finally:
        sys.platform = saved
    assert root.set_to == []


def test_repositioning_failures_are_reported_and_never_raise():
    import io, contextlib

    class BrokenRoot:
        def update_idletasks(self):
            raise RuntimeError("no window")
    with contextlib.redirect_stderr(io.StringIO()) as err:
        _placed_app(BrokenRoot()).keep_window_on_screen()
    assert "Could not reposition" in err.getvalue()


def test_startup_captures_reposition_the_window_before_the_screenshot():
    order = []

    class FakeApp:
        pass
    app = FakeApp()
    app.keep_window_on_screen = lambda: order.append("move")
    app.take_startup_screenshot = lambda: order.append("shot")
    app.take_startup_captures = m.App.take_startup_captures.__get__(app)
    saved = m.debug.dump_app_state
    m.debug.dump_app_state = lambda *a, **k: order.append("dump")
    try:
        app.take_startup_captures()
    finally:
        m.debug.dump_app_state = saved
    assert order == ["move", "shot", "dump"]
