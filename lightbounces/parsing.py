"""Parsing and validating what the user types into the number entries and size fields: each parse_* returns the value, or None when the text is not acceptable (the entry then shows it as invalid)."""
import math
import re
from .constants import (
    CONFINE_WINDOW_MAX, CONFINE_WINDOW_MIN, DIST_LIMIT_MAX, DIST_LIMIT_MIN, ITER_LIMIT_MAX,
    ITER_LIMIT_MIN
)


def parse_iter_limit(text):
    """Whole number in [ITER_LIMIT_MIN, ITER_LIMIT_MAX] from entry text, else None."""
    if re.fullmatch(r"[0-9]{1,4}", text):
        value = int(text)
        if ITER_LIMIT_MIN <= value <= ITER_LIMIT_MAX:
            return value
    return None


def parse_dist_limit(text):
    """Non-negative number in [DIST_LIMIT_MIN, DIST_LIMIT_MAX] from entry
    text (0 = no limit), else None. A whole number is returned as an int (so
    it displays cleanly, e.g. "5" not "5.0"); anything with a fractional
    part stays a float, rounded to 2 decimal places."""
    try:
        value = round(float(text), 2)
    except ValueError:
        return None
    if DIST_LIMIT_MIN <= value <= DIST_LIMIT_MAX:
        return int(value) if value == int(value) else value
    return None


def parse_confine_window(text):
    """Whole number of path points in [CONFINE_WINDOW_MIN, CONFINE_WINDOW_MAX]
    from entry text, else None."""
    if re.fullmatch(r"[0-9]{1,3}", text):
        value = int(text)
        if CONFINE_WINDOW_MIN <= value <= CONFINE_WINDOW_MAX:
            return value
    return None


def parse_confine_threshold(text):
    """The confinement threshold from entry text: any finite number strictly
    greater than zero, however small (plain or scientific notation, e.g.
    0.004 or 1e-300), else None. Deliberately not rounded or clamped."""
    if not re.fullmatch(r"(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?", text):
        return None
    value = float(text)
    return value if math.isfinite(value) and value > 0.0 else None


def parse_escape_radius(text):
    """The escape radius from entry text: any finite number strictly greater than zero (plain or
    scientific notation), else None. Not rounded or clamped."""
    if not re.fullmatch(r"(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?", text):
        return None
    value = float(text)
    return value if math.isfinite(value) and value > 0.0 else None


def parse_target_coordinate(text):
    """A coordinate of the emission target point from entry text: any finite
    number, negative or positive, however large or small (plain or scientific
    notation, e.g. -0.5 or 3e8), else None. Not rounded or clamped."""
    if not re.fullmatch(r"-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?", text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def parse_seed(text):
    """Whole number seed from entry text (blank counts as 0), else None."""
    if text == "":
        return 0
    if re.fullmatch(r"[0-9]{1,9}", text):
        return int(text)
    return None


def parse_frame_number(text, max_digits=6):
    """A frame number - a whole number from 0 (up to max_digits digits) - else None."""
    if re.fullmatch(rf"[0-9]{{1,{max_digits}}}", text):
        return int(text)
    return None


def parse_positive_int(text, max_digits=6):
    """A whole number of at least 1 (up to max_digits digits), else None."""
    if re.fullmatch(rf"[0-9]{{1,{max_digits}}}", text):
        value = int(text)
        if value >= 1:
            return value
    return None


def round_to_multiple_of_4(n):
    """The nearest positive multiple of 4 to n (minimum 4, never 0) - ffmpeg
    needs even width/height to encode most video formats; rounding to a
    multiple of 4 (rather than just 2) gives some safety margin over that
    bare minimum for codecs pickier still (some require a multiple of 4 or
    8, e.g. under chroma subsampling with certain block sizes)."""
    return max(4, round(n / 4.0) * 4)


def parse_resolution_dimension(text, max_digits=6):
    """A resolution field's value: parsed the same way as any other
    positive whole number, then rounded to the nearest multiple of 4 - or
    None if the text isn't a valid whole number at all. Every width/height
    export control (New image, video) reads its value through this, so an
    exported frame's dimensions are always a multiple of 4, regardless of
    exactly what was typed."""
    value = parse_positive_int(text, max_digits=max_digits)
    if value is None:
        return None
    return round_to_multiple_of_4(value)
