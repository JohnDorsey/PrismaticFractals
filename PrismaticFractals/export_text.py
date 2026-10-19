"""What is written around and into exported images: the Export designer's and video panel's option lists, default file names, the text of the
coloration info and of the figure settings (the left margin that lists every figure-defining setting), the layout of that text, and the
functions that draw it into an image (overlay, bottom margin, left margin). Pure functions of settings, colorations and obstacles; no UI."""
import time
import numpy as np
from .constants import (
    CONFINE_METHOD_NAMES, EMIT_MODE_NAMES, ENGINE_NAMES, ORBIT_ESCAPE_NAMES, ORBIT_MAPPING_NAMES, ORBIT_Z0_NAMES
)
from .engines import engine_description
from .image_io import GLYPH_ROWS, jxl_quality_tag, render_text_line
from .kernels import CONFINE_BOX, EMIT_TOWARDS, ENGINE_MANDELBROT, PREVIEW_BOX
from .metrics import (
    BUDDHA_BRIGHTNESS_OPTIONS, BUDDHA_COMPRESSION_LABELS, BUDDHA_POINT_LABELS, CHAN_SRC_BUDDHA,
    CHAN_SRC_CONSTANT, METRIC_TYPE_POINT_PAIR, NONE_ONE, NONE_ZERO, PATH_KIND_LABELS, POINT_SHORT, POINT_METRICS,
    METRIC_LABELS, constant_metric_to_label
)
from .colorations import COLOR_MODELS
from .framings import framing_info_block
from . import paths


def settings_suffix(ray, natural_termination):
    """Simulation settings (a RaySettings and a NaturalTerminationSettings) as a filename fragment: plain
    characters, no spaces, e.g. 'iter32-distNONE'."""
    parts = [f"iter{natural_termination.max_iter}",
             "distNONE" if natural_termination.max_dist == 0 else f"dist{natural_termination.max_dist}"]
    f = ray.fuzz
    if f.enabled:
        parts += ["fuzz",
                  f"ej{f.emit_jitter:.1f}",
                  f"bo{f.bnd_offset:.1f}", f"bj{f.bnd_jitter:.1f}",
                  f"bm{f.bnd_mult:.2f}",
                  f"so{f.self_offset:.1f}", f"sj{f.self_jitter:.1f}",
                  f"sm{f.self_mult:.2f}", f"seed{f.rng_seed}"]
    return "-".join(parts)


# Saving a freshly generated image (as opposed to the live preview): what to
# render, and which format to encode it in
SAVE_SOURCE_OPTIONS = ["Current image", "New image"]
VIDEO_FORMATS = ["mp4 (h264)", "GIF", "APNG", "mkv (ffv1)", "JXL"]
VIDEO_BITRATE_MODES = ["Constant quality", "Target bitrate", "Variable bitrate"]

# Export designer: where (if anywhere) each kind of info text goes
COLOR_INFO_LOCATIONS = ["None", "Overlay", "Bottom margin"]
FIGURE_INFO_LOCATIONS = ["None", "Left margin"]
FRAMING_INFO_LOCATIONS = ["None", "Left margin"]


def default_filename(ray=None, natural_termination=None, ext="png", quality_tag=None):
    """programname-YYYYMMDD-hhmmss-<settings>[.<quality_tag>].<ext>, stamped with the
    current time (no <settings> part without both records)."""
    name = f"{paths.PROGRAM_NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
    if ray is not None and natural_termination is not None:
        name += "-" + settings_suffix(ray, natural_termination)
    if quality_tag:
        name += "." + quality_tag
    return f"{name}.{ext}"


def image_filename(ray, natural_termination, fmt, jxl_lossless, jxl_quality):
    """The default name of a saved still image: a JPEG XL file also says its quality
    right before the extension (name.q90.jxl; q100 for lossless)."""
    tag = jxl_quality_tag(jxl_lossless, jxl_quality) if fmt == "jxl" else None
    return default_filename(ray, natural_termination, fmt, tag)


# The "Include coloration info" export overlay: pure black background (per
# request), and a light gray - never pure white - for the text.
COLORATION_INFO_BG = (0, 0, 0)
COLORATION_INFO_FG = (210, 210, 210)
COLORATION_INFO_IGNORED_FG = (120, 120, 120)         # medium grey: a setting the engine in use makes no use of (D13)


class GreyLine(str):
    """A line of margin text for a setting the engine in use ignores: it reads like any string and is drawn in medium grey."""
COLORATION_INFO_SCALE = 1
COLORATION_INFO_MARGIN = 4
COLORATION_INFO_LINE_GAP = 2


def coloration_info_lines(coloration):
    """One coloration's settings as a list of text lines, mirroring how
    the Colorations tab itself is organized: the color model, then each
    channel's Source and either its Metric (+ Point 1/2, if it's a point-
    comparison metric) or its Buddha points (one per line, as in the
    checkbox list itself) and compression/brightness. Uses the same long
    names shown in the Colorations tab's own dropdowns - the metric's Type
    (point comparison/other) is never shown on its own, though; it's
    implied by which metric is named."""
    lines = [f"Mode:{coloration.color_model}"]
    for c in range(3):
        letter = COLOR_MODELS[coloration.color_model][c][0][0]
        if coloration.chan_src[c] == CHAN_SRC_BUDDHA:
            lines.append(f"{letter}:Buddha")
            chosen = [BUDDHA_POINT_LABELS[i] for i, on in enumerate(coloration.buddha_points[c]) if on]
            if chosen:
                lines.extend(f"  {name}" for name in chosen)
            else:
                lines.append("  (none)")
            if not all(coloration.buddha_path_kinds[c]):
                lines.append("  paths: " + (", ".join(PATH_KIND_LABELS[k] for k, on in enumerate(coloration.buddha_path_kinds[c]) if on)
                                            or "none"))
            comp = BUDDHA_COMPRESSION_LABELS[coloration.buddha_compression[c]]
            bright = BUDDHA_BRIGHTNESS_OPTIONS[coloration.buddha_brightness[c]]
            lines.append(f"  {comp} {bright}")
        elif coloration.chan_src[c] == CHAN_SRC_CONSTANT:
            lines.append(f"{letter}:Constant")
            lines.append(f"  {constant_metric_to_label(coloration.constant_choice[c])}")
        else:
            metric_slot = coloration.point_metric[c] if coloration.metric_types[c] == METRIC_TYPE_POINT_PAIR else coloration.other_metric[c]
            if metric_slot in (NONE_ZERO, NONE_ONE):
                raise ValueError(
                    f"channel {c} has metric_slot={metric_slot} (a Constant-only sentinel) but "
                    f"chan_src is {coloration.chan_src[c]}, not CHAN_SRC_CONSTANT")
            name = METRIC_LABELS[metric_slot]
            lines.append(f"{letter}:{name}")
            if metric_slot in POINT_METRICS:
                p1 = POINT_SHORT[coloration.point1_choice[c]]
                p2 = POINT_SHORT[coloration.point2_choice[c]]
                lines.append(f"  P1:{p1} P2:{p2}")
    return lines


FIGURE_INFO_COLUMN_CHARS = 32        # a column of figure settings holds lines up to this long
FIGURE_INFO_SEPARATOR = "-" * FIGURE_INFO_COLUMN_CHARS   # the line between the figure's settings and the framing's


def figure_info_blocks(ray, natural_termination, obstacles, engine=None, mandelbrot=None):
    """Every setting that defines the figure - the mathematical result of
    perfectly simulating these rules, of which each generated image is only a
    coloration - as a list of blocks of text lines: the termination limits and segment
    length, the emission direction, the confinement rule, the fuzzing, and every
    obstacle. (Coloration, decorations and the
    Buddha blur only process or decorate the generated data, so they aren't
    part of it.) The blocks' sizes depend only on how many obstacles there
    are, never on the settings' values - so a video's frames, whose settings
    may be animated, all lay out identically. `engine` (an EngineSettings, or None to leave the line out) adds the
    first block, the path engine's name, and, with `mandelbrot` too (a MandelbrotSettings), a block of the Mandelbrot engine's own settings.
    The lines of a setting the engine ignores (its description's `ignores`) are GreyLines: drawn in medium grey, still there (D13)."""
    description = engine_description(engine.engine) if engine is not None else None
    limits = [f"Iter limit:{natural_termination.max_iter}",
              "Dist limit:" + ("none" if natural_termination.max_dist == 0
                               else f"{natural_termination.max_dist:.2f}"),
              f"Escape r:{natural_termination.escape_radius:.9g}",
              f"Segment:x{ray.seg_mult:.2f}"]
    toward = ray.emit_mode == EMIT_TOWARDS
    emission = [f"Emit:{EMIT_MODE_NAMES[ray.emit_mode]}",
                f"  target r {ray.emit_target_r:.9g}" if toward else " ",
                f"  target angle {ray.emit_target_theta:.9g}" if toward else " ",
                f"  offset x {ray.emit_target_cx:.9g}" if toward else " ",
                f"  offset y {ray.emit_target_cy:.9g}" if toward else " ",
                f"  angle offset {ray.emit_offset:+.2f}\u00b0"]
    confinement = [f"Confine:{CONFINE_METHOD_NAMES[natural_termination.confine_method]}",
                   f"  lookback {natural_termination.confine_lookback}",
                   ("  box size " if natural_termination.confine_method == CONFINE_BOX
                    else "  path length ") + f"{natural_termination.confine_threshold:.9g}"]
    f = ray.fuzz
    if not f.enabled:
        fuzz = ["Fuzz:Off"] + [" "] * 4
    else:
        fuzz = ["Fuzz:On",
                f"  Emit jitter \u00b1{f.emit_jitter:.1f}\u00b0",
                f"  Bound {f.bnd_offset:+.1f}\u00b0 \u00b1{f.bnd_jitter:.1f}\u00b0 "
                f"\u00d7{f.bnd_mult:.2f}",
                f"  Self {f.self_offset:+.1f}\u00b0 \u00b1{f.self_jitter:.1f}\u00b0 "
                f"\u00d7{f.self_mult:.2f}",
                f"  Seed {f.rng_seed}"]
    def dim(lines, key):
        return [GreyLine(line) for line in lines] if description is not None and description.ignores(key) else lines
    limits = limits[:3] + dim(limits[3:], "seg_mult")
    emission, fuzz = dim(emission, "emit_mode"), dim(fuzz, "fuzz_on")
    blocks = ([[f"Engine:{ENGINE_NAMES[engine.engine]}"]] if engine is not None else []) + [limits, emission, confinement, fuzz]
    if engine is not None and mandelbrot is not None:
        orbit = [f"Orbit:z^{mandelbrot.exponent:.9g}+c", f"  z0 {ORBIT_Z0_NAMES[mandelbrot.z0]}",
                 f"  {ORBIT_MAPPING_NAMES[mandelbrot.mapping].split(' (')[0]}",
                 f"  Julia {mandelbrot.julia_re:.9g}{mandelbrot.julia_im:+.9g}i", f"  escape {ORBIT_ESCAPE_NAMES[mandelbrot.escape].split(' ')[0]}",
                 f"  step x{mandelbrot.pre_seg_mult:.2f} (before)"]
        blocks.append(orbit if engine.engine == ENGINE_MANDELBROT else [GreyLine(line) for line in orbit])
    if obstacles:
        blocks += [o.info_lines(i + 1) for i, o in enumerate(obstacles)]
    else:
        blocks.append(["no obstacles"])
    return blocks


def figure_info_lines(ray, natural_termination, obstacles, engine=None, mandelbrot=None):
    """figure_info_blocks(), flattened into one list of lines."""
    return [line for block in figure_info_blocks(ray, natural_termination, obstacles, engine, mandelbrot) for line in block]


def layout_info_columns(blocks, lines_per_column):
    """Packs blocks of text lines into columns of at most `lines_per_column`
    lines, top to bottom then left to right, keeping each block together
    where it fits (a blank line separates blocks within a column) and
    splitting a block only if it is taller than a whole column."""
    columns = [[]]
    for block in blocks:
        col = columns[-1]
        if col and len(col) + 1 + len(block) > lines_per_column and len(block) <= lines_per_column:
            columns.append([])
            col = columns[-1]
        if col:
            col.append(" ")                        # a blank line between blocks
        for line in block:
            if len(col) >= lines_per_column:
                columns.append([])
                col = columns[-1]
            col.append(line)
    return columns


def resolution_scale(size):
    """The pixel-font scale factor for info text relating to something
    `size` pixels across: never smaller than the PREVIEW_BOX=512 baseline scale,
    growing in whole steps above it, so text stays about the same size
    relative to whatever it's describing rather than shrinking to
    illegibility at high export resolutions."""
    return COLORATION_INFO_SCALE * max(1, round(size / PREVIEW_BOX))


def _draw_text_block(img, lines, x0, y0, avail_w, avail_h, scale, draw_bg):
    """Draws `lines` top to bottom into `img` starting at (x0, y0), each in
    the tiny pixel font at `scale`. draw_bg=True paints an individual solid
    background box behind each line first (for drawing over existing image
    content); draw_bg=False skips that (for a margin area that's already a
    solid color). Clips safely to (avail_w, avail_h) and to img's own
    bounds - never raises even if the text would overflow."""
    line_h = GLYPH_ROWS * scale + COLORATION_INFO_LINE_GAP * scale
    y = y0
    y_limit = min(y0 + avail_h, img.shape[0])
    x_limit = min(x0 + avail_w, img.shape[1])
    for line in lines:
        if y >= y_limit:
            break
        mask = render_text_line(line, scale)
        h, w = mask.shape
        draw_h = min(h, y_limit - y)
        draw_w = min(w, x_limit - x0)
        if draw_w <= 0:
            y += line_h
            continue
        if draw_bg:
            bg_y1 = min(y + h + 2 * scale, y_limit)
            bg_x1 = min(x0 + w + 2 * scale, x_limit)
            img[max(y0, y - scale):bg_y1, max(x0, x0 - scale):bg_x1] = COLORATION_INFO_BG
        region = img[y:y + draw_h, x0:x0 + draw_w]
        region[mask[:draw_h, :draw_w]] = COLORATION_INFO_IGNORED_FG if isinstance(line, GreyLine) else COLORATION_INFO_FG
        y += line_h


def draw_coloration_info(img, colorations, coloration_width):
    """Draws each of `colorations`' coloration_info_lines onto its own
    coloration_width-wide portion of the already-composed `img` (side by side, in
    the same order as `colorations`), top-left, in the tiny pixel font on a solid
    background box drawn over the existing image content. The text scales
    up with the per-coloration resolution (the smaller of coloration_width and the
    image height, so an unusually elongated export doesn't get oversized
    text) - see resolution_scale."""
    scale = resolution_scale(min(coloration_width, img.shape[0]))
    m = COLORATION_INFO_MARGIN * scale
    for i, v in enumerate(colorations):
        x_lo = i * coloration_width
        _draw_text_block(img, coloration_info_lines(v), x_lo + m, m,
                         coloration_width - 2 * m, img.shape[0] - 2 * m, scale, draw_bg=True)


def add_bottom_color_margin(img, colorations, coloration_width):
    """Returns a new, taller copy of `img`: a solid black strip appended
    below it, coloration_width wide per coloration (so each coloration's own coloration info
    sits directly beneath that coloration, matching its own side-by-side
    position) and a quarter that tall, holding that coloration's
    coloration_info_lines. Never mutates `img`."""
    h, w, _ = img.shape
    margin_h = coloration_width // 4
    out = np.empty((h + margin_h, w, 3), dtype=img.dtype)
    out[:h] = img
    out[h:] = COLORATION_INFO_BG
    scale = resolution_scale(coloration_width)   # the coloration's own size, not the margin's
                                           # (always smaller, one quarter of it) -
                                           # otherwise the text would barely ever
                                           # grow even at large export resolutions
    m = COLORATION_INFO_MARGIN * scale
    for i, v in enumerate(colorations):
        x_lo = i * coloration_width
        _draw_text_block(out, coloration_info_lines(v), x_lo + m, h + m,
                         coloration_width - 2 * m, margin_h - 2 * m, scale, draw_bg=False)
    return out


def left_margin_blocks(ray, natural_termination, obstacles, framing, figure_location, framing_location, engine=None, mandelbrot=None):
    """The blocks of text lines of the left margin, as the Export designer's two "settings location" choices have
    it: the figure's settings (figure_info_blocks) if that location is "Left margin", then the framing's
    (framing_info_block) if its location is, below them with a line of dashes between (just the framing's, with
    no line, if the figure's are elsewhere). No blocks at all when neither location is "Left margin". Their sizes
    depend only on how many obstacles there are."""
    blocks = []
    if figure_location == "Left margin":
        blocks += figure_info_blocks(ray, natural_termination, obstacles, engine, mandelbrot)
    if framing_location == "Left margin":
        block = framing_info_block(framing)
        blocks.append([FIGURE_INFO_SEPARATOR] + block if blocks else block)
    return blocks


def add_left_info_margin(img, blocks):
    """Returns a new, wider copy of `img`: a solid black strip added to its
    left edge, as tall as `img` itself, holding `blocks` of text lines (see left_margin_blocks: the settings that
    define the figure, and the framing's). The strip is half as wide as `img`
    - or wider, with more columns of text, if the text needs more room
    than that - and its size depends only on `img`'s size and the blocks' sizes (which depend only on the number of
    obstacles), so every frame of a video gets the same one. Never mutates
    `img`."""
    h, w, _ = img.shape
    scale = resolution_scale(h)   # the coloration's own size, not the margin's (see
                                  # add_bottom_color_margin's identical fix)
    m = COLORATION_INFO_MARGIN * scale
    line_h = GLYPH_ROWS * scale + COLORATION_INFO_LINE_GAP * scale
    columns = layout_info_columns(blocks, max(1, (h - 2 * m) // line_h))
    col_w = 6 * FIGURE_INFO_COLUMN_CHARS * scale
    col_gap = 12 * scale
    margin_w = max(w // 2, 2 * m + len(columns) * col_w + (len(columns) - 1) * col_gap)
    out = np.empty((h, margin_w + w, 3), dtype=img.dtype)
    out[:, margin_w:] = img
    out[:, :margin_w] = COLORATION_INFO_BG
    for k, lines in enumerate(columns):
        _draw_text_block(out, lines, m + k * (col_w + col_gap), m, col_w, h - 2 * m, scale,
                         draw_bg=False)
    return out


def apply_info_locations(rgb, colorations, coloration_width, ray, natural_termination, obstacles, color_location,
                         figure_location, framing, framing_location, mutate_ok, engine=None, mandelbrot=None):
    """Applies the Export designer's Color info location, Figure settings
    location and Framing settings location choices to `rgb` (already composed, before any margin
    expansion), returning the (possibly larger) final image - a plain
    module-level function, safe to call from a background thread, since it
    never reads a Tk variable directly (callers capture the locations
    on the main thread first). `ray`, `natural_termination` and `obstacles` are
    what defines the figure being exported (see figure_info_blocks), `framing` the framing it is an image
    of (see framing_info_block); `engine` (an EngineSettings, or None) adds the path engine's name to the figure's
    settings. mutate_ok=False never draws
    into `rgb` in place for the "Overlay" case (a defensive copy is made
    first instead) - needed when `rgb` might be a shared, reused buffer
    like self.image_for_export."""
    if color_location == "Overlay":
        if not mutate_ok:
            rgb = rgb.copy()
        draw_coloration_info(rgb, colorations, coloration_width)
    elif color_location == "Bottom margin":
        rgb = add_bottom_color_margin(rgb, colorations, coloration_width)
    blocks = left_margin_blocks(ray, natural_termination, obstacles, framing, figure_location, framing_location, engine, mandelbrot)
    if blocks:
        rgb = add_left_info_margin(rgb, blocks)
    return rgb
