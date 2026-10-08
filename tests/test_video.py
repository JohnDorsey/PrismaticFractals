"""Tests: video."""
import os
import re
import shutil
import tempfile
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# FFV1 video export (lossless, MKV container)
# --------------------------------------------------------------------------- #
def _fake_ffmpeg_format_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.ffmpeg_output_kwargs = m.App.ffmpeg_output_kwargs.__get__(app)
    app.output_extension_and_suffix = m.App.output_extension_and_suffix.__get__(app)
    return app


def test_ffv1_uses_mkv_extension_with_ffv1_suffix():
    app = _fake_ffmpeg_format_app()
    ext, suffix = app.output_extension_and_suffix({"fmt": "mkv (ffv1)"})
    assert ext == "mkv"
    assert suffix == ".ffv1"
    filename = f"light_bounces-video-20260101-000000{suffix}.{ext}"
    assert filename.endswith(".ffv1.mkv")


def test_ffv1_kwargs_are_lossless_full_rgb():
    app = _fake_ffmpeg_format_app()
    kwargs = app.ffmpeg_output_kwargs("mkv (ffv1)", "Constant quality", 75, 0, 0, 0)
    assert kwargs["vcodec"] == "ffv1"
    assert kwargs["pix_fmt"] == "gbrp", \
        "FFV1 should use planar RGB (no chroma subsampling) to stay truly lossless"
    assert "crf" not in kwargs and "video_bitrate" not in kwargs, \
        "a lossless codec has no bitrate/quality knobs to set"


def test_other_formats_unaffected_by_adding_ffv1():
    app = _fake_ffmpeg_format_app()
    assert app.output_extension_and_suffix({"fmt": "mp4 (h264)"}) == ("mp4", "")
    assert app.output_extension_and_suffix({"fmt": "GIF"}) == ("gif", "")
    assert app.output_extension_and_suffix({"fmt": "APNG"}) == ("apng", "")
    mp4_kwargs = app.ffmpeg_output_kwargs("mp4 (h264)", "Constant quality", 75, 0, 0, 0)
    assert mp4_kwargs["vcodec"] == "libx264" and mp4_kwargs["pix_fmt"] == "yuv420p"
    assert app.ffmpeg_output_kwargs("GIF", "Constant quality", 75, 0, 0, 0) == {}
    assert app.ffmpeg_output_kwargs("APNG", "Constant quality", 75, 0, 0, 0) == {}


# --------------------------------------------------------------------------- #
# JXL video: ffmpeg can't produce an animated JXL file directly (its libjxl
# wrapper is still-image only), so this renders to an intermediate APNG and
# hands that to the external `cjxl` tool, which does support conversion
# --------------------------------------------------------------------------- #
FAKE_CJXL_DIR = os.path.join(ROOT,
                             "dev", "testing")
REAL_CJXL_PATH = shutil.which("cjxl")   # captured before any test alters PATH


def _path_entries_without_cjxl():
    """The current PATH's directories, minus any that contain a cjxl
    executable, so a test's view of PATH can't be affected by whether the
    machine running it happens to have the real tool installed. (Everything
    else on PATH - ffmpeg, the system directory cmd.exe needs - is kept.)"""
    keep = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and any(os.path.exists(os.path.join(entry, name))
                         for name in ("cjxl", "cjxl.exe", "cjxl.cmd", "cjxl.bat")):
            continue
        keep.append(entry)
    return keep


class _PathContext:
    """For the duration of a `with` block, replaces os.environ["PATH"] with
    `dir_path` (if given) followed by every original PATH directory that
    doesn't contain a cjxl executable - so `dir_path` decides on its own
    whether any `cjxl` is findable: None means none at all, and
    FAKE_CJXL_DIR means the fake from dev/testing/. `fail_on` sets
    FAKE_CJXL_FAIL_ON for the fake (see dev/testing/cjxl.cmd). Both are
    restored afterward."""
    def __init__(self, dir_path=None, fail_on=None):
        self.dir_path = dir_path
        self.fail_on = fail_on

    def __enter__(self):
        self.orig_path = os.environ.get("PATH", "")
        self.orig_fail_on = os.environ.get("FAKE_CJXL_FAIL_ON")
        entries = _path_entries_without_cjxl()
        if self.dir_path is not None:
            entries.insert(0, self.dir_path)
        os.environ["PATH"] = os.pathsep.join(entries)
        if self.fail_on is None:
            os.environ.pop("FAKE_CJXL_FAIL_ON", None)
        else:
            os.environ["FAKE_CJXL_FAIL_ON"] = self.fail_on
        return self

    def __exit__(self, *exc):
        os.environ["PATH"] = self.orig_path
        if self.orig_fail_on is None:
            os.environ.pop("FAKE_CJXL_FAIL_ON", None)
        else:
            os.environ["FAKE_CJXL_FAIL_ON"] = self.orig_fail_on


def _fake_cjxl(fail_on=None):
    """A _PathContext in which `cjxl` is the fake from dev/testing/: it
    copies its input file to its output path and exits 0, unless the input
    path contains `fail_on`, in which case it prints to stderr and exits 1 -
    lets tests exercise both convert_apng_to_jxl's success and failure paths
    without depending on a real libjxl installation."""
    if os.name != "nt":   # the POSIX fake needs its execute bit
        os.chmod(os.path.join(FAKE_CJXL_DIR, "cjxl"), 0o755)
    return _PathContext(FAKE_CJXL_DIR, fail_on)


def _no_cjxl():
    """A _PathContext in which no `cjxl` is findable at all."""
    return _PathContext(None)


def test_output_extension_and_suffix_for_jxl():
    app = _fake_ffmpeg_format_app()
    rs = {"fmt": "JXL", "jxl_lossless": True, "jxl_quality": 70}
    assert app.output_extension_and_suffix(rs) == ("jxl", ".q100")          # lossless
    assert app.output_extension_and_suffix({**rs, "jxl_lossless": False}) == ("jxl", ".q70")
    assert app.output_extension_and_suffix({**rs, "jxl_lossless": False, "jxl_quality": 100}) \
        == ("jxl", ".q95")                                                    # q100 is only lossless


def test_ffmpeg_output_kwargs_for_jxl_matches_apng():
    app = _fake_ffmpeg_format_app()
    assert app.ffmpeg_output_kwargs("JXL", "Constant quality", 75, 0, 0, 0) == {}


def test_cjxl_available_false_without_a_real_binary():
    with _no_cjxl():
        assert m.cjxl_available() is False


def test_cjxl_available_true_with_a_fake_binary_on_path():
    with _fake_cjxl():
        assert m.cjxl_available() is True


def test_convert_apng_to_jxl_success_produces_the_output_file():
    work_dir = tempfile.mkdtemp()
    apng_path = os.path.join(work_dir, "source.apng")
    jxl_path = os.path.join(work_dir, "out.jxl")
    with open(apng_path, "wb") as f:
        f.write(b"fake apng bytes")
    with _fake_cjxl():
        m.convert_apng_to_jxl(apng_path, jxl_path)
    assert os.path.exists(jxl_path)
    with open(jxl_path, "rb") as f:
        assert f.read() == b"fake apng bytes"


def test_the_quality_tag_in_a_jxl_file_name_says_lossless_or_the_lossy_quality():
    assert m.jxl_quality_tag(True, 90) == "q100" and m.jxl_quality_tag(True, 5) == "q100"
    assert m.jxl_quality_tag(False, 90) == "q90" and m.jxl_quality_tag(False, 5) == "q5"
    assert m.jxl_quality_tag(False, 100) == "q95"        # lossy never reads q100
    assert m.JXL_LOSSY_MAX_QUALITY == 95


def test_default_filename_puts_the_quality_tag_just_before_the_extension():
    ray, stop = m.DEFAULT_RAY_SETTINGS, m.DEFAULT_NATURAL_STOP_SETTINGS
    name = m.default_filename(ray, stop, "jxl", "q90")
    assert name.endswith(".q90.jxl") and name.count(".q90") == 1
    assert m.default_filename(None, None, "jxl", "q100").endswith(".q100.jxl")
    assert m.default_filename(None, None, "png").endswith(".png")
    assert ".q" not in m.default_filename(None, None, "png")
    assert m.image_filename(ray, stop, "jxl", False, 85).endswith(".q85.jxl")
    assert m.image_filename(ray, stop, "jxl", True, 85).endswith(".q100.jxl")
    import re
    for fmt in ("png", "qoi"):                            # only JPEG XL files carry a quality
        assert not re.search(r"\.q\d+\.", m.image_filename(ray, stop, fmt, False, 85))


def test_still_images_default_to_jxl():
    assert m.SAVE_NEW_FORMATS[0] == "jxl" and set(m.SAVE_NEW_FORMATS) == {"jxl", "png", "qoi"}


def _fake_save_app(fmt, lossless, quality):
    class FakeApp:
        pass
    app = FakeApp()
    _give_image_geometry(app)
    app.save_source_var = _StubVar(m.SAVE_SOURCE_OPTIONS[0])
    app.save_format_var = _StubVar(fmt)
    app.save_jxl_mode_var = _StubVar("Lossless" if lossless else "Lossy")
    app.save_jxl_quality_scale = _StubVar(quality)
    app.generated_settings = m.DEFAULT_RAY_SETTINGS
    app.generated_natural_stop = m.DEFAULT_NATURAL_STOP_SETTINGS
    app.written = []
    app._write = lambda path: app.written.append(os.path.basename(path))
    app.save = m.App.save.__get__(app)
    return app


def test_saving_a_jxl_image_names_its_quality_and_other_formats_do_not():
    orig = m.paths.EXPORT_DIR
    m.paths.EXPORT_DIR = os.path.join(tempfile.mkdtemp(), "exports")
    try:
        for fmt, lossless, quality, tail in (("jxl", True, 90, ".q100.jxl"),
                                             ("jxl", False, 80, ".q80.jxl"),
                                             ("png", False, 80, ".png")):
            app = _fake_save_app(fmt, lossless, quality)
            app.save()
            assert app.written[0].endswith(tail), app.written
            import re
            assert bool(re.search(r"\.q\d+\.jxl$", app.written[0])) == (fmt == "jxl")
    finally:
        m.paths.EXPORT_DIR = orig


def _recorded_cjxl_arguments(**options):
    """The command line convert_apng_to_jxl runs (the fake cjxl copies its input)."""
    work_dir = tempfile.mkdtemp()
    apng_path = os.path.join(work_dir, "source.apng")
    with open(apng_path, "wb") as f:
        f.write(b"fake apng")
    seen = []
    real_run = m.subprocess.run
    m.subprocess.run = lambda cmd, **kw: seen.append(list(cmd)) or real_run(cmd, **kw)
    try:
        with _fake_cjxl():
            m.convert_apng_to_jxl(apng_path, os.path.join(work_dir, "out.jxl"), **options)
    finally:
        m.subprocess.run = real_run
    return seen[0]


def test_convert_apng_to_jxl_is_lossless_by_default_and_lossy_at_the_distance_of_the_quality():
    def distance(args):
        return args[args.index("-d") + 1]
    assert distance(_recorded_cjxl_arguments()) == "0"
    assert distance(_recorded_cjxl_arguments(lossless=True, quality=50)) == "0"
    assert distance(_recorded_cjxl_arguments(lossless=False, quality=90)) == "1"
    assert distance(_recorded_cjxl_arguments(lossless=False, quality=50)) == "5"
    assert distance(_recorded_cjxl_arguments(lossless=False, quality=100)) == "0.5"   # capped at q95


def test_a_lossy_animated_jxl_from_the_real_cjxl_is_a_smaller_lossy_animation():
    """cjxl really does lossy animation: jxlinfo (when installed beside it) calls the file
    a lossy animation, and it is smaller than the lossless one. Skipped without cjxl/ffmpeg."""
    import subprocess
    ffmpeg_exe = shutil.which("ffmpeg")
    if REAL_CJXL_PATH is None:
        raise SkipTest("real cjxl not on PATH")
    if ffmpeg_exe is None:
        raise SkipTest("real ffmpeg not on PATH")
    work_dir = tempfile.mkdtemp()
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:64, 0:64]
    for i in range(4):
        frame = np.zeros((64, 64, 3), np.uint8)
        frame[..., 0] = (xx * 4 + i * 20) % 256
        frame[..., 1] = (yy * 4) % 256
        frame[..., 2] = rng.randint(0, 40, (64, 64))
        m.write_png(os.path.join(work_dir, f"frame_{i:05d}.png"), frame)
    apng_path = os.path.join(work_dir, "source.apng")
    subprocess.run([ffmpeg_exe, "-y", "-loglevel", "error", "-framerate", "10",
                    "-i", os.path.join(work_dir, "frame_%05d.png"), "-plays", "0", apng_path],
                   check=True)
    lossless = os.path.join(work_dir, "lossless.jxl")
    lossy = os.path.join(work_dir, "lossy.jxl")
    m.convert_apng_to_jxl(apng_path, lossless)
    m.convert_apng_to_jxl(apng_path, lossy, lossless=False, quality=80)
    assert os.path.getsize(lossy) < os.path.getsize(lossless) / 2
    for path in (lossless, lossy):
        with open(path, "rb") as f:
            head = f.read(12)
        assert head[:2] == bytes.fromhex("ff0a") or head == bytes.fromhex("0000000c4a584c200d0a870a")
    jxlinfo = os.path.join(os.path.dirname(REAL_CJXL_PATH),
                           "jxlinfo.exe" if os.name == "nt" else "jxlinfo")
    if os.path.exists(jxlinfo):
        info = subprocess.run([jxlinfo, lossy], capture_output=True, text=True).stdout
        assert "animation" in info and "lossy" in info, info


def test_convert_apng_to_jxl_failure_raises_with_stderr_and_makes_no_file():
    work_dir = tempfile.mkdtemp()
    apng_path = os.path.join(work_dir, "shouldfail.apng")
    jxl_path = os.path.join(work_dir, "out.jxl")
    with open(apng_path, "wb") as f:
        f.write(b"fake apng bytes")
    with _fake_cjxl("shouldfail"):
        try:
            m.convert_apng_to_jxl(apng_path, jxl_path)
            assert False, "should have raised"
        except RuntimeError as e:
            assert "simulated encoder failure" in str(e)
    assert not os.path.exists(jxl_path)


def test_convert_apng_to_jxl_raises_clearly_when_cjxl_is_missing():
    """convert_apng_to_jxl resolves cjxl via shutil.which (so .cmd shims work
    on Windows); with nothing on PATH that must be a clear RuntimeError, not
    a bare FileNotFoundError from subprocess."""
    work_dir = tempfile.mkdtemp()
    apng_path = os.path.join(work_dir, "source.apng")
    with open(apng_path, "wb") as f:
        f.write(b"unused")
    with _no_cjxl():
        try:
            m.convert_apng_to_jxl(apng_path, os.path.join(work_dir, "out.jxl"))
            assert False, "should have raised"
        except RuntimeError as e:
            assert "not found" in str(e)


def test_convert_apng_to_jxl_with_the_real_cjxl_produces_a_jxl_file():
    """The one test here that uses the real tools instead of the fake: builds
    a genuine 2-frame APNG (this program's own PNG writer, then the real
    ffmpeg) and converts it with the real cjxl. Catches what the fake can't -
    cjxl rejecting our APNG, or its command-line interface changing. Skipped
    when cjxl or ffmpeg isn't installed."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if REAL_CJXL_PATH is None:
        raise SkipTest("real cjxl not on PATH")
    if ffmpeg_exe is None:
        raise SkipTest("real ffmpeg not on PATH")
    import subprocess
    work_dir = tempfile.mkdtemp()
    for i, value in enumerate((40, 200)):
        frame = np.full((16, 16, 3), value, dtype=np.uint8)
        m.write_png(os.path.join(work_dir, f"frame_{i:05d}.png"), frame)
    apng_path = os.path.join(work_dir, "source.apng")
    jxl_path = os.path.join(work_dir, "out.jxl")
    subprocess.run([ffmpeg_exe, "-y", "-loglevel", "error", "-framerate", "10",
                    "-i", os.path.join(work_dir, "frame_%05d.png"),
                    "-plays", "0", apng_path], check=True)
    m.convert_apng_to_jxl(apng_path, jxl_path)
    with open(jxl_path, "rb") as f:
        head = f.read(12)
    bare_codestream = head[:2] == bytes.fromhex("ff0a")
    iso_container = head == bytes.fromhex("0000000c4a584c200d0a870a")
    assert bare_codestream or iso_container, f"not a JXL file: {head!r}"


def test_read_render_settings_blocks_jxl_without_cjxl():
    class FakeApp:
        pass
    app = FakeApp()
    app.fps_var = _StubVar("30")
    app.video_w_var = _StubVar("40")
    app.video_h_var = _StubVar("40")
    app.export_format_var = _StubVar("JXL")
    app.bitrate_mode_var = _StubVar("Constant quality")
    app.quality_scale = _StubVar(75)
    app.downsample_factor_var = _StubVar(list(m.DOWNSAMPLE_FACTOR_VALUES)[0])
    app.downsample_method_var = _StubVar("Mean")
    app.video_jxl_controls = _StubJxlControls()
    app.read_render_settings = m.App.read_render_settings.__get__(app)
    with _no_cjxl():
        rs = app.read_render_settings()
    assert rs is None


def test_read_render_settings_allows_jxl_with_cjxl_available():
    class FakeApp:
        pass
    app = FakeApp()
    app.fps_var = _StubVar("30")
    app.video_w_var = _StubVar("40")
    app.video_h_var = _StubVar("40")
    app.export_format_var = _StubVar("JXL")
    app.bitrate_mode_var = _StubVar("Constant quality")
    app.quality_scale = _StubVar(75)
    app.downsample_factor_var = _StubVar(list(m.DOWNSAMPLE_FACTOR_VALUES)[0])
    app.downsample_method_var = _StubVar("Mean")
    app.video_jxl_controls = _StubJxlControls()
    app.read_render_settings = m.App.read_render_settings.__get__(app)
    orig_ffmpeg = m.external.ffmpeg
    m.external.ffmpeg = orig_ffmpeg or object()   # only this sandbox lacks ffmpeg-python
    try:
        with _fake_cjxl():
            rs = app.read_render_settings()
    finally:
        m.external.ffmpeg = orig_ffmpeg
    assert rs is not None
    assert rs["fmt"] == "JXL"


class _FakeFfmpegStream:
    def __init__(self, path=None):
        self.path = path


class _FakeFfmpegModule:
    """Stands in for the real ffmpeg-python module (unavailable in this
    sandbox): input()/output() just track the target path through a
    trivial stream object, and run() simulates a successful encode by
    actually writing bytes to that path, so the rest of encode_video_output
    (the cjxl handoff, temp-file cleanup) can be exercised for real."""
    class Error(Exception):
        pass

    def input(self, *a, **k):
        return _FakeFfmpegStream()

    def output(self, stream, path, **kwargs):
        return _FakeFfmpegStream(path)

    def filter(self, stream, *a, **k):
        return stream

    def run(self, stream, **kwargs):
        with open(stream.path, "wb") as f:
            f.write(b"fake ffmpeg output bytes")


def _fake_encode_app():
    class FakeApp:
        pass
    app = FakeApp()
    app.apply_downsample_filter = m.App.apply_downsample_filter.__get__(app)
    app.ffmpeg_output_kwargs = m.App.ffmpeg_output_kwargs.__get__(app)
    app.encode_video_output = m.App.encode_video_output.__get__(app)
    app.status_calls = []
    app.set_video_status = lambda text: app.status_calls.append(text)
    app.root = type("R", (), {"after": lambda self, delay, fn: fn()})()
    return app


def _base_rs(fmt):
    return {"fps": 30, "fmt": fmt, "mode": "Constant quality", "quality": 75,
           "bitrate": 0, "vbr_target": 0, "vbr_max": 0, "jxl_lossless": True, "jxl_quality": 90,
           "downsample_factor": 1, "downsample_method": "Mean"}


def test_encode_video_output_writes_directly_for_non_jxl_formats():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.mp4")
    orig_ffmpeg = m.external.ffmpeg
    m.external.ffmpeg = _FakeFfmpegModule()
    try:
        app.encode_video_output("pattern_%05d.png", 0, _base_rs("mp4 (h264)"), out_path)
    finally:
        m.external.ffmpeg = orig_ffmpeg
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == b"fake ffmpeg output bytes"


def test_encode_video_output_routes_jxl_through_apng_and_cjxl():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.jxl")
    orig_ffmpeg = m.external.ffmpeg
    orig_temp_dir = m.paths.TEMP_DIR
    m.external.ffmpeg = _FakeFfmpegModule()
    m.paths.TEMP_DIR = tempfile.mkdtemp()
    try:
        with _fake_cjxl():
            app.encode_video_output("pattern_%05d.png", 0, _base_rs("JXL"), out_path)
    finally:
        m.external.ffmpeg = orig_ffmpeg
        m.paths.TEMP_DIR = orig_temp_dir
    assert os.path.exists(out_path), "the final .jxl file must exist"
    with open(out_path, "rb") as f:
        assert f.read() == b"fake ffmpeg output bytes", \
            "the fake cjxl just copies its input, so this confirms the chain ran: " \
            "ffmpeg wrote the APNG, cjxl copied it to the final path"
    temp_apng = os.path.join(m.paths.TEMP_DIR, "_jxl_video_source.apng")
    assert not os.path.exists(temp_apng), "the intermediate APNG must be cleaned up"
    assert any("cjxl" in s for s in app.status_calls), \
        "the status should mention the cjxl conversion step"


def test_encode_video_output_cleans_up_temp_apng_even_on_cjxl_failure():
    app = _fake_encode_app()
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "out.jxl")
    orig_ffmpeg = m.external.ffmpeg
    orig_temp_dir = m.paths.TEMP_DIR
    m.external.ffmpeg = _FakeFfmpegModule()
    m.paths.TEMP_DIR = tempfile.mkdtemp()
    try:
        with _fake_cjxl("_jxl_video_source"):
            try:
                app.encode_video_output("pattern_%05d.png", 0, _base_rs("JXL"), out_path)
                assert False, "should have raised"
            except RuntimeError:
                pass
    finally:
        m.external.ffmpeg = orig_ffmpeg
        m.paths.TEMP_DIR = orig_temp_dir
    temp_apng = os.path.join(m.paths.TEMP_DIR, "_jxl_video_source.apng")
    assert not os.path.exists(temp_apng), \
        "the intermediate APNG must still be cleaned up even when cjxl fails"
    assert not os.path.exists(out_path)


# --------------------------------------------------------------------------- #
# clear_temp_frames: deletes only frame_NNNNN.png files from TEMP_DIR - the
# one place in the whole program that deletes anything at all
# --------------------------------------------------------------------------- #
class _ClearTempFramesContext:
    """Patches m.paths.TEMP_DIR to a fresh scratch directory and m.messagebox's
    dialog functions to controllable, call-counting stubs, for the duration
    of a `with` block - and restores both afterward, since the test module
    loads light_bounces once and shares it across every test."""
    def __init__(self, confirm=True):
        self.confirm = confirm
        self.showinfo_calls = []
        self.showerror_calls = []
        self.askyesno_calls = []

    def __enter__(self):
        self.orig_temp_dir = m.paths.TEMP_DIR
        self.orig_showinfo = m.messagebox.showinfo
        self.orig_showerror = m.messagebox.showerror
        self.orig_askyesno = m.messagebox.askyesno
        self.tmpdir = tempfile.mkdtemp()
        m.paths.TEMP_DIR = self.tmpdir
        m.messagebox.showinfo = lambda *a, **k: self.showinfo_calls.append((a, k))
        m.messagebox.showerror = lambda *a, **k: self.showerror_calls.append((a, k))
        m.messagebox.askyesno = lambda *a, **k: (self.askyesno_calls.append((a, k))
                                                 or self.confirm)
        return self

    def __exit__(self, *exc):
        m.paths.TEMP_DIR = self.orig_temp_dir
        m.messagebox.showinfo = self.orig_showinfo
        m.messagebox.showerror = self.orig_showerror
        m.messagebox.askyesno = self.orig_askyesno


def _fake_app_for_clear_temp_frames(busy=False):
    class FakeApp:
        pass
    app = FakeApp()
    app.busy_with_render_or_reencode = lambda: busy
    app.video_status_calls = []
    app.set_video_status = lambda text: app.video_status_calls.append(text)
    app.clear_temp_frames = m.App.clear_temp_frames.__get__(app)
    return app


def test_clear_temp_frames_only_deletes_the_strict_frame_pattern():
    with _ClearTempFramesContext(confirm=True) as ctx:
        names = ["frame_00000.png", "frame_00001.png", "frame_abc.png",
                "other_frame_00000.png", "notes.txt", "frame_00002.PNG"]
        for name in names:
            open(os.path.join(ctx.tmpdir, name), "w").close()
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()
        remaining = set(os.listdir(ctx.tmpdir))
        assert remaining == {"frame_abc.png", "other_frame_00000.png", "notes.txt",
                             "frame_00002.PNG"}, \
            f"only the two strict frame_NNNNN.png files should be gone, got {remaining}"
        assert len(ctx.askyesno_calls) == 1


def test_clear_temp_frames_deletes_nothing_without_confirmation():
    with _ClearTempFramesContext(confirm=False) as ctx:
        open(os.path.join(ctx.tmpdir, "frame_00000.png"), "w").close()
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()
        assert os.listdir(ctx.tmpdir) == ["frame_00000.png"], \
            "declining the confirmation dialog must delete nothing"


def test_clear_temp_frames_refuses_while_busy():
    with _ClearTempFramesContext(confirm=True) as ctx:
        open(os.path.join(ctx.tmpdir, "frame_00000.png"), "w").close()
        app = _fake_app_for_clear_temp_frames(busy=True)
        app.clear_temp_frames()
        assert os.listdir(ctx.tmpdir) == ["frame_00000.png"], \
            "must not delete anything while a render or re-encode is running"
        assert len(ctx.askyesno_calls) == 0, "must not even ask while busy"
        assert len(ctx.showerror_calls) == 1


def test_clear_temp_frames_handles_missing_temp_dir():
    with _ClearTempFramesContext(confirm=True) as ctx:
        shutil.rmtree(ctx.tmpdir)   # TEMP_DIR itself doesn't exist
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()   # must not raise
        assert len(ctx.showinfo_calls) == 1
        assert len(ctx.askyesno_calls) == 0


def test_clear_temp_frames_handles_no_matching_files():
    with _ClearTempFramesContext(confirm=True) as ctx:
        open(os.path.join(ctx.tmpdir, "notes.txt"), "w").close()
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()
        assert os.listdir(ctx.tmpdir) == ["notes.txt"]
        assert len(ctx.showinfo_calls) == 1
        assert len(ctx.askyesno_calls) == 0


def test_clear_temp_frames_reports_the_count_deleted():
    with _ClearTempFramesContext(confirm=True) as ctx:
        for i in range(3):
            open(os.path.join(ctx.tmpdir, f"frame_{i:05d}.png"), "w").close()
        app = _fake_app_for_clear_temp_frames()
        app.clear_temp_frames()
        assert os.listdir(ctx.tmpdir) == []
        assert any("3" in msg for msg in app.video_status_calls)


# --------------------------------------------------------------------------- #
# The starting frame and the numbering of frame files
# --------------------------------------------------------------------------- #
def test_frames_to_render_runs_from_the_starting_frame_but_never_before_the_first_keyframe():
    assert m.frames_to_render([0, 30, 10], 0) == (0, 30)
    assert m.frames_to_render([0, 30, 10], 5) == (5, 30)
    assert m.frames_to_render([0, 30, 10], 30) == (30, 30)
    assert m.frames_to_render([8, 30], 0) == (8, 30)          # nothing before the first keyframe
    assert m.frames_to_render([0, 30], 31) is None            # past the end: nothing to render


def test_frame_files_are_numbered_from_the_starting_frame_or_from_zero():
    assert m.frame_file_offset(12, True) == 12
    assert m.frame_file_offset(12, False) == 0
    assert m.frame_file_offset(0, True) == 0


def test_a_starting_frame_is_a_whole_number_from_zero():
    assert [m.parse_frame_number(t) for t in ("0", "7", "012", "999999")] == [0, 7, 12, 999999]
    assert [m.parse_frame_number(t) for t in ("", "-1", "1.5", "a", "1000000")] == [None] * 5


def test_ffmpeg_reads_a_frame_sequence_that_does_not_start_at_zero():
    """What the "number frame files from the starting frame" choice relies on: -start_number. Frames
    numbered 7..11 are read as one sequence, in order, when told where it starts."""
    import shutil
    import subprocess
    import tempfile
    if shutil.which("ffmpeg") is None:
        raise SkipTest("ffmpeg is not installed")
    folder = tempfile.mkdtemp(prefix="lb_start_")
    try:
        for number in range(7, 12):
            m.write_png(os.path.join(folder, f"frame_{number:05d}.png"),
                        np.full((16, 16, 3), 40 * (number - 7), dtype=np.uint8))
        out = os.path.join(folder, "out.mkv")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-start_number", "7",
                        "-i", os.path.join(folder, "frame_%05d.png"), "-c:v", "ffv1", out],
                       check=True)
        probe = subprocess.run(["ffmpeg", "-i", out, "-f", "null", "-"], capture_output=True, text=True)
        counts = re.findall(r"frame=\s*(\d+)", probe.stderr)
        assert counts and int(counts[-1]) == 5, probe.stderr[-300:]
        blind = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-i",
                                os.path.join(folder, "frame_%05d.png"), os.path.join(folder, "x.mkv")],
                               capture_output=True)
        assert blind.returncode != 0        # without -start_number the same files are not found
    finally:
        shutil.rmtree(folder, ignore_errors=True)
