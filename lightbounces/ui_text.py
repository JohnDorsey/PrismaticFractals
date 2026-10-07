"""Short texts the interface shows about settings: the keyframe settings list (section headers, a value as text, the presence filter) and the Coloration tab's help line."""
from .constants import CONFINE_METHOD_NAMES, EMIT_MODE_NAMES, OBSTACLE_EFFECT_NAMES
from .metrics import (
    BUDDHA_BRIGHTNESS_OPTIONS, BUDDHA_COMPRESSION_LABELS, BUDDHA_POINT_SHORT, CHAN_SRC_BUDDHA,
    CHAN_SRC_CONSTANT, KIND_POINT, NONE_ONE, NONE_ZERO, POINT_SHORT, STAT_SHORT
)
from .shapes import OBST_FIELD_FMT


DIFF_PRESENCE_CHOICES = ("present", "absent", "present or absent")


def setting_passes_presence_filter(in_keyframe, presence):
    """Whether a setting shows in the keyframe settings list under the "Show"
    filter. `in_keyframe`: the selected keyframe itself defines it; `presence` is
    a value from DIFF_PRESENCE_CHOICES."""
    if presence == "present":
        return in_keyframe
    if presence == "absent":
        return not in_keyframe
    return True


def diff_section_header_text(title, expanded, n_set, n_total, n_modified=0):
    """A section header of the keyframe settings list: an arrow showing whether
    it is expanded, its title, and how many of its settings the selected
    keyframe itself defines (so a collapsed section still says something)."""
    arrow = "\u25bc" if expanded else "\u25b6"
    modified = f", {n_modified} modified" if n_modified else ""
    return f"{arrow} {title}  ({n_set} of {n_total} set{modified})"


def short_value_text(key, val):
    """A compact display of one setting's value, for the Load/Save button
    text in the keyframe settings list."""
    if key.startswith("obst:"):
        field = key.split(":", 2)[2]
        if field in ("in_effect", "out_effect"):
            return OBSTACLE_EFFECT_NAMES[val]
        if field == "fire_inside":
            return "On" if val else "Off"
        fmt = OBST_FIELD_FMT.get(field)
        if fmt == "size":
            return f"{val:.2f}"
        if fmt == "offset":
            return f"{val:+.2f}"
        if fmt == "angle":
            return f"{val:.1f}\u00b0"
        return str(val)                          # "int" fields, and anything else
    if key.startswith("coloration:"):
        field = key.split(":", 2)[2]
        if field == "mode":
            return val
        # field is "chan0"/"chan1"/"chan2"
        if val.chan_src == CHAN_SRC_BUDDHA:
            chosen = [BUDDHA_POINT_SHORT[i] for i, on in enumerate(val.buddha_points) if on]
            comp = BUDDHA_COMPRESSION_LABELS[val.buddha_compression]
            bright = BUDDHA_BRIGHTNESS_OPTIONS[val.buddha_brightness]
            return f"Buddha:{'+'.join(chosen) if chosen else 'none'}/{comp}/{bright}"
        if val.chan_src == CHAN_SRC_CONSTANT:
            return f"Const:{'0' if val.constant == NONE_ZERO else '1'}"
        if val.kind == KIND_POINT:
            return f"{STAT_SHORT[val.stat]}@{POINT_SHORT[val.point1]}/{POINT_SHORT[val.point2]}"
        if val.stat in (NONE_ZERO, NONE_ONE):
            raise ValueError(
                f"channel value has stat={val.stat} (a Constant-only sentinel) but "
                f"chan_src is {val.chan_src}, not CHAN_SRC_CONSTANT")
        return STAT_SHORT[val.stat]
    if key in ("iter", "dist", "seed", "confine_window"):
        return str(val)
    if key == "confine_method":
        return CONFINE_METHOD_NAMES[val]
    if key in ("confine_threshold", "escape_radius"):
        return f"{val:.6g}"
    if key == "emit_offset":
        return f"{val:+.1f}\u00b0"
    if key == "emit_mode":
        return EMIT_MODE_NAMES[val]
    if key == "emit_target_theta":
        return f"{val:.6g}\u00b0"
    if key in ("emit_target_r", "emit_target_cx", "emit_target_cy"):
        return f"{val:.6g}"
    if key in ("emit_jitter", "bnd_offset", "bnd_jitter", "self_offset", "self_jitter"):
        return f"\u00b1{val:.1f}\u00b0" if "jitter" in key else f"{val:+.1f}\u00b0"
    if key in ("bnd_mult", "self_mult", "seg_mult"):
        return f"\u00d7{val:.2f}"
    if key == "fuzz_on":
        return "On" if val else "Off"
    if key == "outline":
        return "On" if val else "Off"
    return str(val)


def default_map_help(mode):
    text = ("For each channel, choose the Type of metric: point comparison "
            "metrics measure the distance between two sample points of the "
            "ray (Point 1 and Point 2); other metrics don't need a point. "
            "Changes recolor the image immediately.")
    if mode == "HSV":
        text += " Hue 0 and hue 1 are both red."
    return text
