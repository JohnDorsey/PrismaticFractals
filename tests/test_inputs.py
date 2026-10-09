"""Tests: inputs."""
import os
import tempfile

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# Distance limit: non-integer support (regression coverage for two bugs)
# --------------------------------------------------------------------------- #
def test_parse_dist_limit_accepts_non_integers():
    assert m.parse_dist_limit("5.5") == 5.5
    assert m.parse_dist_limit("0.4") == 0.4
    assert m.parse_dist_limit("7") == 7 and isinstance(m.parse_dist_limit("7"), int)
    assert m.parse_dist_limit("-1") is None
    assert m.parse_dist_limit("abc") is None
    assert m.parse_dist_limit("") is None
    assert m.parse_dist_limit(f"{m.DIST_LIMIT_MAX + 1}") is None


def test_dist_is_not_rounded_during_keyframe_resolution():
    """Regression test: resolve_frame_settings() used to force dist through
    int(round(...)), so a keyframe with dist=0.4 resolved to max_dist=0 -
    which means "no limit" in this program's convention. A short-distance
    keyframe would silently become an unlimited-distance one."""
    kf = m.Keyframe(1, 0, {**m.DEFAULT_SETTING_VALUES, "dist": 0.4})
    data = m.resolve_frame_settings([kf], 0, m.DEFAULT_SETTING_VALUES)
    assert data.natural_termination.max_dist == 0.4
    assert m.effective_dist(data.natural_termination.max_dist) == 0.4


def test_dist_animates_smoothly_between_keyframes():
    """Regression test: "dist" wasn't in SETTING_FLOAT_KEYS, so it was held
    (jumping only at the second keyframe's exact frame) instead of
    interpolated - the actual bug report was "Buddha colors don't appear
    until the very last frame" of a video animating the distance limit."""
    kf0 = m.Keyframe(1, 0, {**m.DEFAULT_SETTING_VALUES, "dist": 0.01})
    kf1 = m.Keyframe(2, 59, {**m.DEFAULT_SETTING_VALUES, "dist": 16})
    dists = [m.resolve_frame_settings([kf0, kf1], fn, m.DEFAULT_SETTING_VALUES).natural_termination.max_dist for fn in range(60)]
    assert dists[0] == 0.01 and dists[59] == 16
    assert all(dists[i] < dists[i + 1] for i in range(59)), \
        "dist should increase strictly monotonically across every frame"


# --------------------------------------------------------------------------- #
# Resolution controls round to the nearest multiple of 4 (ffmpeg needs even
# dimensions to encode most video formats)
# --------------------------------------------------------------------------- #
def test_round_to_multiple_of_4_table():
    cases = [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 8), (7, 8), (8, 8),
            (1080, 1080), (1081, 1080), (1082, 1080), (1083, 1084)]
    for n, expected in cases:
        assert m.round_to_multiple_of_4(n) == expected, f"n={n}"


def test_round_to_multiple_of_4_never_returns_zero_or_negative():
    for n in range(1, 20):
        assert m.round_to_multiple_of_4(n) >= 4


def test_parse_resolution_dimension_rounds_a_valid_value():
    assert m.parse_resolution_dimension("1081") == 1080
    assert m.parse_resolution_dimension("1") == 4


def test_parse_resolution_dimension_rejects_invalid_text():
    assert m.parse_resolution_dimension("abc") is None
    assert m.parse_resolution_dimension("0") is None
    assert m.parse_resolution_dimension("") is None


def test_commit_resolution_var_rewrites_displayed_text_to_rounded_value():
    class FakeApp:
        pass
    app = FakeApp()
    app.commit_resolution_var = m.App.commit_resolution_var.__get__(app)
    var = _StubVar("1081")
    app.commit_resolution_var(var)
    assert var.get() == "1080"


def test_commit_resolution_var_leaves_invalid_text_alone():
    class FakeApp:
        pass
    app = FakeApp()
    app.commit_resolution_var = m.App.commit_resolution_var.__get__(app)
    var = _StubVar("not a number")
    app.commit_resolution_var(var)
    assert var.get() == "not a number"


def test_save_new_image_rounds_a_non_multiple_of_4_resolution():
    app = _fake_export_app()
    app.save_new_w_var = _StubVar("21")   # not a multiple of 4 -> rounds to 20
    app.save_new_h_var = _StubVar("22")   # not a multiple of 4 -> rounds to 24
    app.save_format_var = _StubVar("png")
    app.save_jxl_mode_var = _StubVar("Lossless")
    app.save_jxl_quality_scale = _StubVar(90)
    app.save_btn = _StubWidget()
    app.save_as_btn = _StubWidget()
    app.save_new_image = m.App.save_new_image.__get__(app)
    app._save_new_image_worker = m.App._save_new_image_worker.__get__(app)
    app._save_new_image_done = m.App._save_new_image_done.__get__(app)
    app._save_new_image_failed = m.App._save_new_image_failed.__get__(app)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()

    tmpdir = tempfile.mkdtemp()
    orig_export_dir = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = tmpdir
    try:
        _run_save_new_image_synchronously(app)
        files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(files) == 1
        img = m.read_png(os.path.join(tmpdir, files[0]))
        assert img.shape == (24, 20, 3), \
            f"21x22 should round to 20x24, got {img.shape}"
    finally:
        m.paths.EXPORT_DIR = orig_export_dir


def test_read_render_settings_rounds_a_non_multiple_of_4_resolution():
    class FakeApp:
        pass
    app = FakeApp()
    app.fps_var = _StubVar("30")
    app.video_w_var = _StubVar("1081")
    app.video_h_var = _StubVar("1082")
    app.export_format_var = _StubVar("mp4 (h264)")
    app.bitrate_mode_var = _StubVar("Constant quality")
    app.quality_scale = _StubVar(75)
    app.downsample_factor_var = _StubVar(list(m.DOWNSAMPLE_FACTOR_VALUES)[0])
    app.downsample_method_var = _StubVar("Mean")
    app.video_jxl_controls = _StubJxlControls()
    app.read_render_settings = m.App.read_render_settings.__get__(app)
    orig_ffmpeg = m.external.ffmpeg
    m.external.ffmpeg = orig_ffmpeg or object()   # only this sandbox lacks ffmpeg-python; the
                                        # function under test doesn't actually call it
    try:
        rs = app.read_render_settings()
    finally:
        m.external.ffmpeg = orig_ffmpeg
    assert rs is not None
    assert rs["video_w"] == 1080
    assert rs["video_h"] == 1080


def test_parse_escape_radius_accepts_any_number_above_zero():
    assert [m.parse_escape_radius(t) for t in ("2", "2.0", "0.001", "1e-30", "1.5E3", ".5", "7.")] == \
        [2.0, 2.0, 0.001, 1e-30, 1500.0, 0.5, 7.0]
    assert [m.parse_escape_radius(t) for t in ("0", "0.0", "-1", "", "abc", "1e999", "1e", "--1", "1 2")] == [None] * 9
    assert m.DEFAULT_ESCAPE_RADIUS == 2.0
