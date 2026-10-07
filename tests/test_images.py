"""Tests: images."""
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# Image encoders
# --------------------------------------------------------------------------- #
def _qoi_reference_decode(data):
    """Independent QOI decoder (not sharing any code with the encoder under
    test), used only to verify round-trips here."""
    import struct
    assert data[:4] == b"qoif"
    width, height, _channels, _cs = struct.unpack(">IIBB", data[4:14])
    pixels = np.zeros((height * width, 4), dtype=np.uint8)
    pixels[:, 3] = 255
    index = np.zeros((64, 4), dtype=np.uint8)
    pos = 14
    px = np.array([0, 0, 0, 255], dtype=np.int32)
    i, n = 0, height * width
    while i < n:
        tag = data[pos]
        if tag == 0xFE:
            px[0], px[1], px[2] = data[pos + 1], data[pos + 2], data[pos + 3]
            pos += 4
            pixels[i] = px
            idx = (px[0] * 3 + px[1] * 5 + px[2] * 7 + px[3] * 11) % 64
            index[idx] = px
            i += 1
        elif tag == 0xFF:
            px[0], px[1], px[2], px[3] = (data[pos + 1], data[pos + 2],
                                          data[pos + 3], data[pos + 4])
            pos += 5
            pixels[i] = px
            idx = (px[0] * 3 + px[1] * 5 + px[2] * 7 + px[3] * 11) % 64
            index[idx] = px
            i += 1
        elif (tag & 0xC0) == 0x00:
            px[:] = index[tag & 0x3F]
            pos += 1
            pixels[i] = px
            i += 1
        elif (tag & 0xC0) == 0x40:
            dr = ((tag >> 4) & 0x03) - 2
            dg = ((tag >> 2) & 0x03) - 2
            db = (tag & 0x03) - 2
            px[0] = (px[0] + dr) & 0xFF
            px[1] = (px[1] + dg) & 0xFF
            px[2] = (px[2] + db) & 0xFF
            pos += 1
            pixels[i] = px
            idx = (px[0] * 3 + px[1] * 5 + px[2] * 7 + px[3] * 11) % 64
            index[idx] = px
            i += 1
        elif (tag & 0xC0) == 0x80:
            b2 = data[pos + 1]
            dg = (tag & 0x3F) - 32
            dr_dg = (b2 >> 4) - 8
            db_dg = (b2 & 0x0F) - 8
            px[1] = (px[1] + dg) & 0xFF
            px[0] = (px[0] + dg + dr_dg) & 0xFF
            px[2] = (px[2] + dg + db_dg) & 0xFF
            pos += 2
            pixels[i] = px
            idx = (px[0] * 3 + px[1] * 5 + px[2] * 7 + px[3] * 11) % 64
            index[idx] = px
            i += 1
        else:  # QOI_OP_RUN
            run = (tag & 0x3F) + 1
            for _ in range(run):
                pixels[i] = px
                i += 1
            pos += 1
    return pixels[:, :3].reshape(height, width, 3)


def test_qoi_round_trip():
    rng = np.random.RandomState(0)
    cases = {
        "random": rng.randint(0, 256, (37, 53, 3), dtype=np.uint8),
        "solid": np.full((10, 100, 3), 123, dtype=np.uint8),
        "palette": np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
                            dtype=np.uint8)[rng.randint(0, 4, (20, 30))],
        "run_boundary_62": np.full((1, 62, 3), 50, dtype=np.uint8),
        "run_boundary_63": np.full((1, 63, 3), 50, dtype=np.uint8),
        "single_pixel": np.array([[[7, 8, 9]]], dtype=np.uint8),
    }
    for name, img in cases.items():
        data = m.encode_qoi_bytes(img)
        decoded = _qoi_reference_decode(data)
        assert np.array_equal(decoded, img), f"QOI round-trip failed for case: {name}"


def test_png_round_trip():
    rng = np.random.RandomState(1)
    img = rng.randint(0, 256, (40, 60, 3), dtype=np.uint8)
    data = m.encode_png_bytes(img)
    decoded = m.decode_png_bytes(data)
    assert np.array_equal(decoded, img)


def test_jxl_quality_to_distance_mapping():
    assert m.jxl_quality_to_distance(90) == 1.0   # ffmpeg/libjxl's own default
    assert m.jxl_quality_to_distance(100) < m.jxl_quality_to_distance(50)
    assert m.jxl_quality_to_distance(0) == 10.0


# --------------------------------------------------------------------------- #
# Downsampling
# --------------------------------------------------------------------------- #
def test_downsample_sum_clamps_on_overflow():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0], img[0, 1], img[1, 0], img[1, 1] = (200, 10, 0), (100, 250, 0), (50, 5, 0), (10, 0, 0)
    out = m.downsample_image(img, 2, "Sum")
    assert tuple(out[0, 0]) == (255, 255, 0)


def test_downsample_max_is_per_channel_independent():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0], img[0, 1], img[1, 0], img[1, 1] = (200, 10, 0), (100, 250, 0), (50, 5, 0), (10, 0, 0)
    out = m.downsample_image(img, 2, "Max")
    assert tuple(out[0, 0]) == (200, 250, 0)


# --------------------------------------------------------------------------- #
# The pixel font is complete: every letter, both cases, digits, "#" and theta
# --------------------------------------------------------------------------- #
def test_the_pixel_font_has_every_letter_in_both_cases_every_digit_and_the_symbols():
    import string
    needed = (string.ascii_uppercase + string.ascii_lowercase + string.digits
              + "#\u03b8 ")
    assert [c for c in needed if c not in m.FONT_5X7] == []


def test_every_pixel_font_glyph_is_seven_rows_of_five_zero_or_one_cells():
    for ch, rows in m.FONT_5X7.items():
        assert len(rows) == 7, ch
        for row in rows:
            assert len(row) == 5 and set(row) <= {"0", "1"}, (ch, row)


def test_no_visible_pixel_font_glyph_is_blank_and_letters_are_told_apart():
    import string
    for ch, rows in m.FONT_5X7.items():
        if ch != " ":
            assert any("1" in row for row in rows), f"{ch!r} is blank"
    letters = {c: tuple(m.FONT_5X7[c]) for c in string.ascii_letters + "#\u03b8"}
    assert len(set(letters.values())) == len(letters), "two letters share a glyph"


def test_the_only_glyphs_sharing_a_picture_are_the_two_long_standing_pairs():
    seen = {}
    for ch, rows in m.FONT_5X7.items():
        seen.setdefault(tuple(rows), []).append(ch)
    shared = sorted(sorted(v) for v in seen.values() if len(v) > 1)
    assert shared == [["0", "o"], ["x", "\u00d7"]]


def test_the_new_glyphs_are_the_right_shape():
    def art(ch):
        return ["".join("#" if c == "1" else "." for c in row) for row in m.FONT_5X7[ch]]
    assert art("#")[2] == "#####" and art("#")[4] == "#####"
    theta = art("\u03b8")
    assert theta[3] == "#####" and theta[0] == ".###." and theta[6] == ".###."
    assert art("W")[3] == "#.#.#" and art("N")[1] == "##..#" and art("Z")[0] == "#####"


def test_render_text_line_can_draw_every_letter_hash_and_theta():
    import string
    text = string.ascii_uppercase + string.ascii_lowercase + "#\u03b8 0123456789"
    mask = m.render_text_line(text, scale=1)
    assert mask.shape == (7, 6 * len(text) - 1) and mask.any()


# --------------------------------------------------------------------------- #
# The coloration-name pixel-font overlay on the live preview
# --------------------------------------------------------------------------- #
def test_font_5x7_covers_every_character_needed_for_coloration_names():
    for ch in "coloration0123456789 ":
        assert ch in m.FONT_5X7, f"FONT_5X7 is missing {ch!r}, needed for 'coloration N' text"


def test_render_text_line_produces_legible_glyphs_for_every_digit():
    """Each digit glyph must actually light up some pixels (not be blank/a
    typo'd all-zero row) and must be exactly 7 rows tall."""
    for ch in "0123456789":
        mask = m.render_text_line(ch, scale=1)
        assert mask.shape[0] == 7
        assert mask.any(), f"digit {ch!r} renders as entirely blank"


def test_coloration_name_overlay_draws_in_the_bottom_left_corner_only():
    app = type("FakeApp", (), {})()
    app.display_frame = np.full((m.SIZE, m.SIZE, 3), 100, dtype=np.uint8)
    app.colorations = [m.ColorationConfig(0), m.ColorationConfig(1)]
    app.selected_coloration_id = 1
    app.coloration_by_id = m.App.coloration_by_id.__get__(app)
    app.selected_coloration_obj = m.App.selected_coloration_obj.__get__(app)
    app.live_coloration = m.App.live_coloration.__get__(app)
    app.all_coloration_ids = m.App.all_coloration_ids.__get__(app)
    app.draw_coloration_name_overlay = m.App.draw_coloration_name_overlay.__get__(app)

    app.draw_coloration_name_overlay()

    # A background-box pixel near the bottom-left corner must be the
    # overlay's background color, and some pixels nearby must be its
    # foreground color (the text itself) - while the rest of the image is
    # untouched.
    assert tuple(app.display_frame[m.SIZE - 10, 10]) == m.COLORATION_NAME_OVERLAY_BG
    region = app.display_frame[m.SIZE - 40:m.SIZE, 0:100]
    assert ((region == m.COLORATION_NAME_OVERLAY_FG).all(axis=-1)).sum() > 0
    assert tuple(app.display_frame[10, m.SIZE - 10]) == (100, 100, 100)   # untouched elsewhere


def test_coloration_name_overlay_reflects_the_live_colorations_position():
    app = type("FakeApp", (), {})()
    app.display_frame = np.zeros((m.SIZE, m.SIZE, 3), dtype=np.uint8)
    app.colorations = [m.ColorationConfig(0), m.ColorationConfig(1), m.ColorationConfig(2)]
    app.selected_coloration_id = 2
    app.coloration_by_id = m.App.coloration_by_id.__get__(app)
    app.selected_coloration_obj = m.App.selected_coloration_obj.__get__(app)
    app.live_coloration = m.App.live_coloration.__get__(app)
    app.all_coloration_ids = m.App.all_coloration_ids.__get__(app)
    app.draw_coloration_name_overlay = m.App.draw_coloration_name_overlay.__get__(app)

    app.draw_coloration_name_overlay()
    text_pixels_with_id_2 = (app.display_frame == m.COLORATION_NAME_OVERLAY_FG).all(axis=-1).sum()

    app2 = type("FakeApp", (), {})()
    app2.display_frame = np.zeros((m.SIZE, m.SIZE, 3), dtype=np.uint8)
    app2.colorations = app.colorations
    app2.selected_coloration_id = 0
    app2.coloration_by_id = m.App.coloration_by_id.__get__(app2)
    app2.selected_coloration_obj = m.App.selected_coloration_obj.__get__(app2)
    app2.live_coloration = m.App.live_coloration.__get__(app2)
    app2.all_coloration_ids = m.App.all_coloration_ids.__get__(app2)
    app2.draw_coloration_name_overlay = m.App.draw_coloration_name_overlay.__get__(app2)
    app2.draw_coloration_name_overlay()
    text_pixels_with_id_0 = (app2.display_frame == m.COLORATION_NAME_OVERLAY_FG).all(axis=-1).sum()

    # "coloration 2" and "coloration 0" are different-length strings of lit pixels in
    # general, but at minimum the two renders should differ - same digit
    # count here, so compare the actual pixel patterns are not identical.
    assert not np.array_equal(app.display_frame, app2.display_frame)


def test_draw_never_lets_the_coloration_name_overlay_leak_into_the_export_buffer():
    """Regression test: draw() used to draw the overlay directly into
    self.frame, which _write() (the "Current preview image" export path)
    saves straight to disk - meaning the preview-only overlay was silently
    ending up in exported files. self.frame must stay untouched by the
    overlay; only self.display_frame (never written to disk) should carry it."""
    app = type("FakeApp", (), {})()
    app.img = np.full((m.SIZE, m.SIZE, 3), 50, dtype=np.uint8)
    app.frame = np.empty_like(app.img)
    app.display_frame = np.empty_like(app.img)
    app.deco = {"outline": type("X", (), {"get": lambda self: False})()}
    app.colorations = [m.ColorationConfig(0)]
    app.selected_coloration_id = 0
    app.coloration_by_id = m.App.coloration_by_id.__get__(app)
    app.selected_coloration_obj = m.App.selected_coloration_obj.__get__(app)
    app.live_coloration = m.App.live_coloration.__get__(app)
    app.all_coloration_ids = m.App.all_coloration_ids.__get__(app)
    app.obstacle_ring_specs = lambda: []
    app.draw_coloration_name_overlay = m.App.draw_coloration_name_overlay.__get__(app)
    app.draw = m.App.draw.__get__(app)
    app.photo = type("P", (), {"configure": lambda self, **kw: None})()

    app.draw()

    assert tuple(app.display_frame[m.SIZE - 10, 10]) == m.COLORATION_NAME_OVERLAY_BG
    assert tuple(app.frame[m.SIZE - 10, 10]) == (50, 50, 50), \
        "the overlay leaked into self.frame, which gets written to disk on export"
