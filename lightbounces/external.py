"""Optional outside dependencies.

ffmpeg (ffmpeg-python) and ImageGrab (Pillow) are None when the package is not installed. Tests patch them
(external.ffmpeg = ...), so every other module reads them as attributes (`external.ffmpeg`) and none imports them by value.

njit and prange are numba's decorator and parallel range, or - when numba is not installed - stand-ins that run the code as
plain Python (very slow). They are decorators, not values a compiled function freezes in, so modules do import them by value.
"""
import shutil


try:
    import ffmpeg               # the ffmpeg-python package
except ImportError:
    ffmpeg = None                # video rendering is disabled; stills still work

try:
    from PIL import ImageGrab   # Pillow: only used for the development startup screenshot
except ImportError:
    ImageGrab = None             # no startup screenshot; nothing else needs Pillow


def cjxl_available():
    """Whether the external `cjxl` command-line tool (part of libjxl-tools)
    can be found on PATH - needed for an animated JXL video, since ffmpeg's
    own libjxl wrapper (write_jxl, above) only supports a single still
    frame, not animation."""
    return shutil.which("cjxl") is not None


try:
    from numba import njit, prange
except ImportError:  # works, but far too slowly to be practical
    print("WARNING: numba not found - falling back to pure Python (very slow).")
    print("         Install it with:  pip install numba")

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    prange = range
