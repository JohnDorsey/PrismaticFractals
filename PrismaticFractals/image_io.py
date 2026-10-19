"""PNG, QOI and JPEG XL files, downsampling, and the 5x8 pixel font that draws text into images.

A leaf module: it depends only on paths, external and numpy/numba (the QOI encoder is an @njit function with its own
constants, all defined here)."""
import os
import shutil
import struct
import subprocess
import zlib

import numpy as np

from . import external, paths
from .external import njit


# The pixel font: 5 columns x 8 rows (`GLYPH_ROWS`), each glyph a list of eight strings of five 0/1 cells, row 0 on top.
# The glyphs of printable ASCII, and of the degree sign, plus-minus sign, middle dot and a-acute (their positions in the
# code page 437 part of the table), are from the "classic" font of Adafruit's Adafruit-GFX-Library (glcdfont.c, BSD
# license, Copyright (c) 2012 Adafruit Industries; see THIRD_PARTY_NOTICES.md), 5 columns of 8 bits each: the 8th row holds
# the tails of , g p q y. The two glyphs that font lacks, theta and the multiplication sign, were drawn for this program.
# Used for the "preview will go here" placeholders, the coloration-name overlay on the live preview and the text written
# into exported images.
GLYPH_ROWS = 8
FONT_5X8 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100", "00000"],
    "\"": ["01010", "01010", "01010", "00000", "00000", "00000", "00000", "00000"],
    "#": ["01010", "01010", "11111", "01010", "11111", "01010", "01010", "00000"],
    "$": ["00100", "01111", "10100", "01110", "00101", "11110", "00100", "00000"],
    "%": ["11000", "11001", "00010", "00100", "01000", "10011", "00011", "00000"],
    "&": ["01000", "10100", "10100", "01000", "10101", "10010", "01101", "00000"],
    "'": ["00110", "00110", "00100", "01000", "00000", "00000", "00000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010", "00000"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000", "00000"],
    "*": ["00100", "10101", "01110", "11111", "01110", "10101", "00100", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000", "00000"],
    ",": ["00000", "00000", "00000", "00000", "00110", "00110", "00100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110", "00000"],
    "/": ["00000", "00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110", "00000"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110", "00000"],
    "2": ["01110", "10001", "00001", "01110", "10000", "10000", "11111", "00000"],
    "3": ["11111", "00001", "00010", "00110", "00001", "10001", "01110", "00000"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010", "00000"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110", "00000"],
    "6": ["00111", "01000", "10000", "11110", "10001", "10001", "01110", "00000"],
    "7": ["11111", "00001", "00001", "00010", "00100", "01000", "10000", "00000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110", "00000"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100", "00000"],
    ":": ["00000", "00000", "00100", "00000", "00100", "00000", "00000", "00000"],
    ";": ["00000", "00000", "00100", "00000", "00100", "00100", "01000", "00000"],
    "<": ["00001", "00010", "00100", "01000", "00100", "00010", "00001", "00000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000", "00000"],
    "?": ["01110", "10001", "00001", "00110", "00100", "00000", "00100", "00000"],
    "@": ["01110", "10001", "10101", "10111", "10110", "10000", "01111", "00000"],
    "A": ["00100", "01010", "10001", "10001", "11111", "10001", "10001", "00000"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110", "00000"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110", "00000"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110", "00000"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111", "00000"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000", "00000"],
    "G": ["01111", "10001", "10000", "10000", "10011", "10001", "01111", "00000"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001", "00000"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110", "00000"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100", "00000"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001", "00000"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111", "00000"],
    "M": ["10001", "11011", "10101", "10101", "10101", "10001", "10001", "00000"],
    "N": ["10001", "10001", "11001", "10101", "10011", "10001", "10001", "00000"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110", "00000"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000", "00000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101", "00000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001", "00000"],
    "S": ["01110", "10001", "10000", "01110", "00001", "10001", "01110", "00000"],
    "T": ["11111", "10101", "00100", "00100", "00100", "00100", "00100", "00000"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110", "00000"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100", "00000"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010", "00000"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001", "00000"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100", "00000"],
    "Z": ["11111", "00001", "00010", "01110", "01000", "10000", "11111", "00000"],
    "[": ["01111", "01000", "01000", "01000", "01000", "01000", "01111", "00000"],
    "\\": ["00000", "10000", "01000", "00100", "00010", "00001", "00000", "00000"],
    "]": ["01111", "00001", "00001", "00001", "00001", "00001", "01111", "00000"],
    "^": ["00100", "01010", "10001", "00000", "00000", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111", "00000"],
    "`": ["01100", "01100", "00100", "00010", "00000", "00000", "00000", "00000"],
    "a": ["00000", "00000", "01100", "00010", "01110", "10010", "01111", "00000"],
    "b": ["10000", "10000", "10110", "11001", "10001", "11001", "10110", "00000"],
    "c": ["00000", "00000", "01110", "10001", "10000", "10001", "01110", "00000"],
    "d": ["00001", "00001", "01101", "10011", "10001", "10011", "01101", "00000"],
    "e": ["00000", "00000", "01110", "10001", "11111", "10000", "01110", "00000"],
    "f": ["00010", "00101", "00100", "01110", "00100", "00100", "00100", "00000"],
    "g": ["00000", "00000", "01110", "10011", "10011", "01101", "00001", "01110"],
    "h": ["10000", "10000", "10110", "11001", "10001", "10001", "10001", "00000"],
    "i": ["00100", "00000", "01100", "00100", "00100", "00100", "01110", "00000"],
    "j": ["00010", "00000", "00010", "00010", "00010", "10010", "01100", "00000"],
    "k": ["10000", "10000", "10010", "10100", "11000", "10100", "10010", "00000"],
    "l": ["01100", "00100", "00100", "00100", "00100", "00100", "01110", "00000"],
    "m": ["00000", "00000", "11010", "10101", "10101", "10101", "10101", "00000"],
    "n": ["00000", "00000", "10110", "11001", "10001", "10001", "10001", "00000"],
    "o": ["00000", "00000", "01110", "10001", "10001", "10001", "01110", "00000"],
    "p": ["00000", "00000", "10110", "11001", "11001", "10110", "10000", "10000"],
    "q": ["00000", "00000", "01101", "10011", "10011", "01101", "00001", "00001"],
    "r": ["00000", "00000", "10110", "11001", "10000", "10000", "10000", "00000"],
    "s": ["00000", "00000", "01111", "10000", "01110", "00001", "11110", "00000"],
    "t": ["00100", "00100", "11111", "00100", "00100", "00101", "00010", "00000"],
    "u": ["00000", "00000", "10001", "10001", "10001", "10011", "01101", "00000"],
    "v": ["00000", "00000", "10001", "10001", "10001", "01010", "00100", "00000"],
    "w": ["00000", "00000", "10001", "10001", "10101", "10101", "01010", "00000"],
    "x": ["00000", "00000", "10001", "01010", "00100", "01010", "10001", "00000"],
    "y": ["00000", "00000", "10001", "10001", "01111", "00001", "10001", "01110"],
    "z": ["00000", "00000", "11111", "00010", "00100", "01000", "11111", "00000"],
    "{": ["00010", "00100", "00100", "01000", "00100", "00100", "00010", "00000"],
    "|": ["00100", "00100", "00100", "00000", "00100", "00100", "00100", "00000"],
    "}": ["01000", "00100", "00100", "00010", "00100", "00100", "01000", "00000"],
    "~": ["01000", "10101", "00010", "00000", "00000", "00000", "00000", "00000"],
    "\u00b0": ["01110", "11011", "11011", "01110", "00000", "00000", "00000", "00000"],
    "\u00b1": ["00100", "00100", "11111", "00100", "00100", "00000", "11111", "00000"],
    "\u00b7": ["00000", "00000", "00000", "00000", "00110", "00000", "00000", "00000"],
    "\u00e1": ["00011", "00000", "01100", "00010", "01110", "10010", "01111", "00000"],
    "\u03b8": ["01110", "10001", "10001", "11111", "10001", "10001", "01110", "00000"],
    "\u00d7": ["00000", "00000", "01010", "00100", "01010", "00000", "00000", "00000"],
}


def png_rgba(width, height, pixel):
    """Minimal PNG encoder: a width x height image where every pixel is the
    4-byte RGBA value `pixel`. (Lets us make alpha-blended images with only the
    standard library.)"""
    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    row = b"\x00" + bytes(pixel) * width          # filter byte 0 + pixels
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * height))
            + chunk(b"IEND", b""))


def encode_png_bytes(rgb):
    """Encode an (H, W, 3) uint8 array as plain (no-alpha) PNG bytes."""
    height, width = rgb.shape[0], rgb.shape[1]

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    rows = bytearray()
    for y in range(height):
        rows.append(0)                             # filter type 0 (none)
        rows.extend(rgb[y].tobytes())
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
            + chunk(b"IEND", b""))


def write_png(path, rgb):
    """Write an (H, W, 3) uint8 array as a plain (no-alpha) PNG file."""
    with open(path, "wb") as fh:
        fh.write(encode_png_bytes(rgb))


def decode_png_bytes(data):
    """Read the pixels of a PNG written by this program's own
    encode_png_bytes (8-bit RGB, filter type 0/none throughout) back into an
    (H, W, 3) uint8 array."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    width = height = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", body[:10])
            if depth != 8 or color_type != 2:
                raise ValueError("expected an 8-bit RGB PNG (as written by this program)")
        elif tag == b"IDAT":
            idat.extend(body)
        elif tag == b"IEND":
            break
        pos += 12 + length                          # length(4) + tag(4) + data + crc(4)
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    for y in range(height):
        row_start = y * (stride + 1)
        filt = raw[row_start]
        if filt != 0:
            raise ValueError(f"unsupported PNG filter type {filt} (expected 0)")
        rgb[y] = np.frombuffer(raw, dtype=np.uint8, count=stride,
                               offset=row_start + 1).reshape(-1, 3)
    return rgb


def read_png(path):
    """The inverse of write_png."""
    with open(path, "rb") as fh:
        return decode_png_bytes(fh.read())


@njit(cache=True)
def qoi_encode_body(rgb, out):
    """Encode an (H, W, 3) uint8 image into QOI's chunk stream (the part
    between the 14-byte header and the 8-byte end marker), written into the
    pre-allocated `out` uint8 array (must be at least 5*H*W bytes - the
    worst case, one uncompressed QOI_OP_RGB chunk per pixel). Returns the
    number of bytes actually written."""
    h, w = rgb.shape[0], rgb.shape[1]
    n = h * w
    flat = rgb.reshape(n, 3)
    index_r = np.zeros(64, dtype=np.uint8)
    index_g = np.zeros(64, dtype=np.uint8)
    index_b = np.zeros(64, dtype=np.uint8)
    index_seen = np.zeros(64, dtype=np.bool_)
    prev_r = 0
    prev_g = 0
    prev_b = 0
    pos = 0
    run = 0
    for i in range(n):
        r = int(flat[i, 0])
        g = int(flat[i, 1])
        b = int(flat[i, 2])
        if r == prev_r and g == prev_g and b == prev_b:
            run += 1
            if run == 62:
                out[pos] = 0xC0 | (run - 1)
                pos += 1
                run = 0
            continue
        if run > 0:
            out[pos] = 0xC0 | (run - 1)
            pos += 1
            run = 0

        idx = (r * 3 + g * 5 + b * 7 + 255 * 11) % 64
        if index_seen[idx] and index_r[idx] == r and index_g[idx] == g and index_b[idx] == b:
            out[pos] = idx                              # QOI_OP_INDEX (tag 00 is idx<64)
            pos += 1
        else:
            dr = (r - prev_r + 128) % 256 - 128          # wrapped signed difference
            dg = (g - prev_g + 128) % 256 - 128
            db = (b - prev_b + 128) % 256 - 128
            if -2 <= dr <= 1 and -2 <= dg <= 1 and -2 <= db <= 1:
                out[pos] = 0x40 | ((dr + 2) << 4) | ((dg + 2) << 2) | (db + 2)
                pos += 1
            else:
                dr_dg = dr - dg
                db_dg = db - dg
                if -32 <= dg <= 31 and -8 <= dr_dg <= 7 and -8 <= db_dg <= 7:
                    out[pos] = 0x80 | (dg + 32)
                    out[pos + 1] = ((dr_dg + 8) << 4) | (db_dg + 8)
                    pos += 2
                else:
                    out[pos] = 0xFE
                    out[pos + 1] = r
                    out[pos + 2] = g
                    out[pos + 3] = b
                    pos += 4
        index_r[idx] = r
        index_g[idx] = g
        index_b[idx] = b
        index_seen[idx] = True
        prev_r = r
        prev_g = g
        prev_b = b
    if run > 0:
        out[pos] = 0xC0 | (run - 1)
        pos += 1
    return pos


def encode_qoi_bytes(rgb):
    """An (H, W, 3) uint8 array -> the bytes of a complete .qoi file."""
    h, w = rgb.shape[0], rgb.shape[1]
    out = np.empty(5 * h * w, dtype=np.uint8)
    length = qoi_encode_body(np.ascontiguousarray(rgb), out)
    header = b"qoif" + struct.pack(">IIBB", w, h, 3, 0)
    return header + out[:length].tobytes() + b"\x00\x00\x00\x00\x00\x00\x00\x01"


def write_qoi(path, rgb):
    with open(path, "wb") as fh:
        fh.write(encode_qoi_bytes(rgb))


JXL_LOSSY_MAX_QUALITY = 95   # lossy quality stops here: "q100" in a file name always means lossless


def jxl_quality_tag(lossless, quality):
    """The quality part of a JPEG XL file's name (name.q90.jxl): q100 for lossless, else
    q<quality> (a lossy quality is at most JXL_LOSSY_MAX_QUALITY)."""
    if lossless:
        return "q100"
    return f"q{min(int(round(quality)), JXL_LOSSY_MAX_QUALITY)}"


def jxl_quality_to_distance(quality):
    """A 0..100 "quality" (higher = better, matching the usual convention)
    -> libjxl's own "distance" scale (0 = lossless, higher = worse; sane
    values rarely exceed 5-10). quality=90 lands on distance=1.0, ffmpeg's
    own default and roughly comparable to libjpeg quality 90."""
    return max(0.1, (100 - quality) / 10.0)


# libjxl's default codestream level (5) holds images of up to 2^28 = 268,435,456 pixels, and ffmpeg's libjxl wrapper has no option to
# raise it: a larger image makes ffmpeg fail ("Could not open encoder"). cjxl picks the level the image needs (up to 2^40 pixels).
JXL_FFMPEG_MAX_PIXELS = 1 << 28


def _run_cjxl(source_path, jxl_path, distance):
    """Convert the PNG / APNG `source_path` to `jxl_path` with the external `cjxl` tool at Butteraugli `distance` (0 = lossless).
    Runs the path shutil.which resolved, not the bare name "cjxl": on Windows, a bare name only launches .exe files, whereas
    which() (and so cjxl_available()) also finds .cmd/.bat shims."""
    cjxl_path = shutil.which("cjxl")
    if cjxl_path is None:
        raise RuntimeError("cjxl failed: the cjxl tool was not found on PATH")
    result = subprocess.run([cjxl_path, source_path, jxl_path, "-d", f"{distance:g}"], capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output captured"
        raise RuntimeError(f"cjxl failed (exit code {result.returncode}): {detail}")


def write_jxl(path, rgb, lossless, quality=90):
    """Encode an (H, W, 3) uint8 image to a .jxl file via ffmpeg's libjxl
    wrapper. Unlike PNG/QOI, JPEG XL has no remotely hand-rollable encoder -
    this genuinely needs the real codec, the same one the video export
    already depends on. Writes a temporary PNG (this program's own writer)
    as the encoder's input, and removes it again once the JXL file is written.
    An image of more than JXL_FFMPEG_MAX_PIXELS pixels (12800 x 25600 and up) is beyond what ffmpeg's wrapper can encode, so
    it goes through the external cjxl tool instead (RuntimeError, saying why, when that is not installed); a failure of
    ffmpeg itself is a RuntimeError that carries the end of ffmpeg's own error output."""
    big = rgb.shape[0] * rgb.shape[1] > JXL_FFMPEG_MAX_PIXELS
    if big:
        if shutil.which("cjxl") is None:
            raise RuntimeError(f"an image of {rgb.shape[1]} x {rgb.shape[0]} pixels is too large for ffmpeg's JPEG XL encoder (the limit is "
                               f"{JXL_FFMPEG_MAX_PIXELS:,} pixels) and the cjxl tool, which can encode it, was not found on PATH")
    elif external.ffmpeg is None:
        raise RuntimeError("ffmpeg-python is required to save JPEG XL images")
    os.makedirs(paths.TEMP_DIR, exist_ok=True)
    tmp_path = os.path.join(paths.TEMP_DIR, "_jxl_source.png")
    write_png(tmp_path, rgb)
    try:
        distance = 0.0 if lossless else jxl_quality_to_distance(
            min(quality, JXL_LOSSY_MAX_QUALITY))
        if big:
            _run_cjxl(tmp_path, path, distance)
            return
        stream = external.ffmpeg.input(tmp_path)
        stream = external.ffmpeg.output(stream, path, **{"c:v": "libjxl", "distance": distance,
                                                "update": 1, "frames:v": 1})
        try:
            external.ffmpeg.run(stream, overwrite_output=True, quiet=True)
        except Exception as error:
            stderr = getattr(error, "stderr", None)
            text = stderr.decode("utf-8", "replace") if isinstance(stderr, (bytes, bytearray)) else str(stderr or "")
            lines = [line for line in text.strip().splitlines() if line.strip()]
            if not lines:
                raise
            raise RuntimeError("ffmpeg failed: " + " / ".join(lines[-4:])) from error
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def convert_apng_to_jxl(apng_path, jxl_path, lossless=True, quality=90):
    """Converts an animated PNG to an animated JPEG XL file via the external
    `cjxl` tool. This is the only available path to an animated JXL file:
    ffmpeg cannot produce one directly (its libjxl wrapper is still-image
    only), but it CAN produce an APNG (natively, losslessly), and cjxl CAN
    convert an existing APNG's full animation to JXL - so the video export
    pipeline renders to APNG first and hands the result to this function.
    Lossless (-d 0) by default; with `lossless` False, cjxl makes a lossy animation
    (verified with cjxl 0.12: jxlinfo reports a lossy animation) at the distance
    that `quality` (0-100, as the still images' quality slider) maps to."""
    distance = 0.0 if lossless else jxl_quality_to_distance(min(quality, JXL_LOSSY_MAX_QUALITY))
    _run_cjxl(apng_path, jxl_path, distance)


SAVE_NEW_FORMATS = ["jxl", "png", "qoi"]      # the first is the default

JXL_MODES = ["Lossless", "Lossy"]


def encode_and_write(rgb, path, fmt, jxl_lossless=True, jxl_quality=90):
    """An (H, W, 3) uint8 image -> a file at `path`, in whichever of the
    three save formats is selected. Shared by saving the current preview
    (instant) and generating+saving a "New image" (backgrounded)."""
    if fmt == "png":
        write_png(path, rgb)
    elif fmt == "qoi":
        write_qoi(path, rgb)
    else:
        write_jxl(path, rgb, jxl_lossless, jxl_quality)


# Downsampling (video-encoding only - never applied to the PNG sequence, since
# it's an encoding setting, not a rendering one)
DOWNSAMPLE_FACTORS = ["No change", "2x2", "4x4", "8x8"]

DOWNSAMPLE_FACTOR_VALUES = {"No change": 1, "2x2": 2, "4x4": 4, "8x8": 8}

DOWNSAMPLE_METHODS = ["Mean", "Median", "Sum", "Max", "ffmpeg"]


def downsample_image(rgb, factor, method):
    """Block-reduce an (H, W, 3) uint8 image by `factor` in each dimension,
    using the mean, the (per-channel) median, the (per-channel, clamped to
    255) sum, or the (per-channel) maximum of each factor x factor block.
    Any leftover rows/columns that don't fill a whole block are trimmed off
    first."""
    if factor <= 1:
        return rgb
    h, w = rgb.shape[0], rgb.shape[1]
    factor = max(1, min(factor, h, w))               # never larger than the image itself
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    trimmed = rgb[:h2, :w2]
    blocks = trimmed.reshape(h2 // factor, factor, w2 // factor, factor, 3)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(h2 // factor, w2 // factor,
                                                     factor * factor, 3)
    if method == "Mean":
        out = blocks.mean(axis=2)
    elif method == "Median":
        out = np.median(blocks, axis=2)
    elif method == "Sum":
        out = blocks.astype(np.int64).sum(axis=2)    # avoid uint8 wraparound before clamping
    else:                                            # "Max"
        out = blocks.max(axis=2)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def make_grid(w, h, world_rect=None):
    """World-coordinate (x, y) grids for a w x h array of pixel centres covering `world_rect` (anything with x_start,
    x_stop, y_start and y_stop; the default world [-2, 2] on both axes if None), the same extent regardless of w, h.
    The same arithmetic as the kernels' (trace_rows), so a grid point is exactly where its pixel's ray starts."""
    x_start, x_stop, y_start, y_stop = ((-2.0, 2.0, -2.0, 2.0) if world_rect is None
                                        else (world_rect.x_start, world_rect.x_stop, world_rect.y_start, world_rect.y_stop))
    cx = x_start + (np.arange(w) + 0.5) * ((x_stop - x_start) / w)
    cy = y_stop - (np.arange(h) + 0.5) * ((y_stop - y_start) / h)
    return np.meshgrid(cx, cy)


def lerp_color(c1, c2, t):
    """Linear interpolation between two (r, g, b) tuples -> a '#rrggbb' string."""
    r = int(round(c1[0] + (c2[0] - c1[0]) * t))
    g = int(round(c1[1] + (c2[1] - c1[1]) * t))
    b = int(round(c1[2] + (c2[2] - c1[2]) * t))
    return f"#{r:02x}{g:02x}{b:02x}"


def bilinear_resize(img, out_h, out_w):
    """Bilinearly resize an (H, W, 3) uint8 array to (out_h, out_w, 3) uint8."""
    in_h, in_w = img.shape[0], img.shape[1]
    ys = np.clip((np.arange(out_h) + 0.5) * in_h / out_h - 0.5, 0, in_h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * in_w / out_w - 0.5, 0, in_w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    rows0 = img[y0].astype(np.float64)             # (out_h, in_w, 3)
    rows1 = img[y1].astype(np.float64)
    top = rows0[:, x0] * (1.0 - wx) + rows0[:, x1] * wx
    bot = rows1[:, x0] * (1.0 - wx) + rows1[:, x1] * wx
    out = top * (1.0 - wy) + bot * wy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def render_text_line(text, scale):
    """A boolean (H, W) mask of `text` (only characters FONT_5X8 has) drawn in the
    5x8 bitmap font, `scale`x, one column of gap between letters."""
    cell_w = 6                                     # 5 pixels + 1 gap column
    height = GLYPH_ROWS * scale
    width = (cell_w * len(text) - 1) * scale
    mask = np.zeros((height, width), dtype=bool)
    for ci, ch in enumerate(text):
        ox = ci * cell_w * scale
        for ry, row in enumerate(FONT_5X8[ch]):
            for rx, px in enumerate(row):
                if px == "1":
                    mask[ry * scale:(ry + 1) * scale, ox + rx * scale:ox + (rx + 1) * scale] = True
    return mask


def render_word_lines(size_w, size_h, words, scale=1):
    """A black `size_w` x `size_h` RGB image with each string in `words` on
    its own centred magenta line (the whole block centred vertically)."""
    canvas = np.zeros((size_h, size_w, 3), dtype=np.uint8)
    lines = [render_text_line(w, scale) for w in words]
    gap = 2 * scale
    total_h = sum(line.shape[0] for line in lines) + gap * (len(lines) - 1)
    y = max(0, (size_h - total_h) // 2)
    for line in lines:
        h, w = line.shape
        x = max(0, (size_w - w) // 2)
        if y + h <= size_h and x + w <= size_w:
            region = canvas[y:y + h, x:x + w]
            region[line] = (255, 0, 255)
        y += h + gap
    return canvas


def preview_placeholder(width, height):
    """A black width x height RGB image reading 'preview will go here' in
    magenta, shown in the Video panel whenever nothing is being rendered."""
    return render_word_lines(width, height, ["preview", "will", "go", "here"], scale=2)
