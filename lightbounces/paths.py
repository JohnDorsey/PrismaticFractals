"""Where the program reads and writes: the folders next to the launcher, and the program's name.

These are globals that tests patch (EXPORT_DIR, TEMP_DIR, PROJECT_DIR, DEV_DIR, DEBUG_DIR, PROGRAM_NAME), so every other module
reads them as `paths.EXPORT_DIR` at call time, never with `from .paths import EXPORT_DIR` (that would copy the value
at import and make a later patch do nothing).
"""
import os

# The folder with the launcher (light_bounces.py): the one above this package.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# What the files the program writes are named after. This default is the launcher's usual file name; the launcher
# sets it to its own actual file name, so renaming the launcher renames the output files.
PROGRAM_NAME = "light_bounces"
# What the program reads and writes (exports, projects, the debug captures) lives in the data folder next to the launcher.
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
# Everything the program writes (exported images and videos) goes in EXPORT_DIR;
# the video frames' TEMP folder lives inside it.
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
PROJECT_DIR = os.path.join(DATA_DIR, "projects")       # where the project dialogs start
# Development aid: if this folder exists (and holds debug_dump.py, for dumps), the app screenshots its own window once
# per launch and dumps its state (see take_ui_screenshot, dump_app_state); the captures themselves go in DEBUG_DIR.
DEV_DIR = os.path.join(SCRIPT_DIR, "dev")
DEBUG_DIR = os.path.join(DATA_DIR, "debug")            # DEBUG_DIR/screenshots and DEBUG_DIR/dumps
TEMP_DIR = os.path.join(EXPORT_DIR, "TEMP")


def ensure_project_dir():
    """PROJECT_DIR, created first if it doesn't exist yet."""
    os.makedirs(PROJECT_DIR, exist_ok=True)
    return PROJECT_DIR


def ensure_export_dir():
    """EXPORT_DIR, created first if it doesn't exist yet."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    return EXPORT_DIR
