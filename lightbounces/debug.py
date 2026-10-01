"""Debug aids that need the dev/ folder (screenshots and state dumps): paths for the files, the screen grab, and dump_app_state, which hands the app's state to dev/debug_dump.py. Nothing here may raise into the program."""
import os
import sys
import time
from .version import FILE_VERSION
from . import external, paths


def unused_path(path):
    """`path`, or with "-2", "-3", ... before its extension if it already exists
    (two captures in the same second must not overwrite each other)."""
    base, ext = os.path.splitext(path)
    candidate, n = path, 1
    while os.path.exists(candidate):
        n += 1
        candidate = f"{base}-{n}{ext}"
    return candidate


def take_ui_screenshot(root, cause):
    """Development aid: grabs the app's own window from the screen and saves
    it as DEV_DIR/screenshots/<FILE_VERSION>-YYYYMMDD-hhmmss-<cause>.png
    (`cause` is "startup" or "manual"), returning the
    path - but only if the DEV_DIR folder already exists (this never creates
    it) and Pillow is installed; otherwise does nothing and returns None. It
    grabs the actual screen pixels inside the window's rectangle, so anything
    covering the window at that moment appears in the shot instead."""
    if cause not in ("startup", "manual"):
        raise ValueError(f"screenshot cause must be startup or manual, not {cause!r}")
    if external.ImageGrab is None or not os.path.isdir(paths.DEV_DIR):
        return None
    root.update()                                  # finish drawing, settle geometry
    x, y = root.winfo_rootx(), root.winfo_rooty()
    img = external.ImageGrab.grab(bbox=(x, y, x + root.winfo_width(), y + root.winfo_height()))
    shots_dir = os.path.join(paths.DEV_DIR, "screenshots")
    os.makedirs(shots_dir, exist_ok=True)
    path = unused_path(os.path.join(
        shots_dir, f"{FILE_VERSION}-{time.strftime('%Y%m%d-%H%M%S')}-{cause}.png"))
    img.save(path)
    return path


def dump_app_state(app, cause, include_large_arrays, exc_info=None):
    """Development aid: pickles everything the application holds into
    DEV_DIR/dumps/<FILE_VERSION>-YYYYMMDD-hhmmss-<cause>.pkl (`cause` is "startup",
    "crash" or "manual") using dev/debug_dump.py, and returns the path; without
    `include_large_arrays` every big array in it is replaced by a stand-in
    (dev/debug_dump.py has the size rule). Does
    nothing and returns None unless the DEV_DIR folder and that file exist.
    Never raises: a failure is reported on stderr and otherwise ignored, so a
    debugging aid can never add a crash (or hide the one being dumped)."""
    if cause not in ("startup", "crash", "manual"):
        raise ValueError(f"dump cause must be startup, crash or manual, not {cause!r}")
    try:
        dumper_path = os.path.join(paths.DEV_DIR, "debug_dump.py")
        if not os.path.isfile(dumper_path):
            return None
        import importlib.util
        spec = importlib.util.spec_from_file_location("debug_dump", dumper_path)
        dumper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dumper)
        dumps_dir = os.path.join(paths.DEV_DIR, "dumps")
        os.makedirs(dumps_dir, exist_ok=True)
        path = unused_path(os.path.join(
            dumps_dir, f"{FILE_VERSION}-{time.strftime('%Y%m%d-%H%M%S')}-{cause}.pkl"))
        return dumper.dump_everything(
            app, getattr(type(app).__init__, "__globals__", globals()), path, cause,   # the app module's globals
            include_large_arrays=include_large_arrays, exc_info=exc_info,
            file_version=FILE_VERSION)
    except Exception as exc:                        # noqa: BLE001
        print(f"Debug dump failed: {exc}", file=sys.stderr)
        return None
