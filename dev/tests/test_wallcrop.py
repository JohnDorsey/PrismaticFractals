"""Tests: wallcrop (dev/tools/wallcrop.nim, the crop-to-wallpaper tool written in Nim)."""
import os
import shutil
import subprocess
import sys
import tempfile

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers

TOOL = os.path.join(ROOT, "dev", "tools", "wallcrop.nim")


def _nim():
    nim = shutil.which("nim")
    if not nim:
        raise SkipTest("Nim is not installed")
    return nim


def test_the_wallcrop_tool_passes_its_own_checks():
    """The tool carries its own checks (`--selftest`): the exact mean / median of n x n blocks against a slow
    reference, the camera and outline, the keys and menus, the pyramid, and (when vips is installed) a real
    PNG -> cache -> export round trip.  This compiles it and runs them."""
    nim = _nim()
    with tempfile.TemporaryDirectory() as folder:
        exe = os.path.join(folder, "wallcrop" + (".exe" if sys.platform == "win32" else ""))
        built = subprocess.run([nim, "c", "-d:release", "--hints:off", f"--nimcache:{os.path.join(folder, 'nimcache')}",
                                f"-o:{exe}", TOOL], capture_output=True, text=True, timeout=600)
        if "invocation of external compiler program failed" in built.stdout + built.stderr:
            raise SkipTest("Nim has no C compiler here (set `cc = clang` or install gcc)")
        assert built.returncode == 0, built.stdout + built.stderr
        ran = subprocess.run([exe, "--selftest"], capture_output=True, text=True, timeout=600)
        assert ran.returncode == 0, ran.stdout + ran.stderr
        assert "checks passed" in ran.stdout, ran.stdout


def test_the_wallcrop_tool_type_checks_for_linux():
    """The Windows-only code (file dialogs, DPI) and the Linux-only code (zenity / kdialog, library names) sit in
    `when` branches, so Windows alone would never compile the Linux ones; the compiler can check them
    without building anything."""
    nim = _nim()
    checked = subprocess.run([nim, "check", "--os:linux", "--cpu:amd64", "--hints:off", TOOL],
                             capture_output=True, text=True, timeout=300)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "Error" not in checked.stdout + checked.stderr, checked.stdout + checked.stderr
