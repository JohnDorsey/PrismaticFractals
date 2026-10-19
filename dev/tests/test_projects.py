"""Tests: projects."""
import os
import tempfile

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# Projects: the file format
# --------------------------------------------------------------------------- #
OLD_FRAMING = m.FramingSnapshot(id=0, x=0.0, y=0.0, diagonal=2.0 * 2.0 ** 0.5, aspect=1.0)        # the whole world [-1, 1]
OLD_SEEDING = m.DEFAULT_BUDDHA_SEED.replace(x_start=-1.0, x_stop=1.0, y_start=-1.0, y_stop=1.0)


def _sample_project(**changes):
    """A project using every kind of obstacle, two colorations (one with Buddha channels),
    keyframes (including channel records) and non-default simulation and export choices. Everything that
    the program's defaults decide (the sizes, the world, the limits) is written out: the project is the one the reference
    files in dev/testing hold, whatever the defaults are now."""
    obstacles = []
    for i, cls in enumerate(m.SHAPE_CLASSES):
        o = cls(i * 2 + 1)                                  # ids 1, 3, 5...: not 0..n
        for name, value in SAMPLE_SIZES[cls.KEY].items():
            setattr(o, name, value)
        for name in o.FIELD_NAMES:
            if name == "rotation":
                o.rotation = 10.0 + i
            elif name == "r":
                o.r = 0.25
            elif name == "fire_inside":
                o.fire_inside = bool(i % 2)
            elif name == "in_effect":
                o.in_effect = i % 3
            elif name == "out_effect":
                o.out_effect = (i + 1) % 3
        obstacles.append(o.snapshot())
    v0, v1 = m.ColorationConfig(0), m.ColorationConfig(5)
    v1.color_model = "HSV"
    v1.set_channel(1, m.ChannelSettings(
        chan_src=m.CHAN_SRC_BUDDHA, metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_BOUNCE_COUNT, point1=0, point2=1,
        buddha_points=_buddha_choice(m.POINT_SELF_MEAN, m.POINT_BOUNDARY_MEAN), buddha_path_kinds=m.ALL_PATH_KINDS,
        buddha_compression=m.BUDDHA_COMPRESS_MODULUS, buddha_brightness=3, constant=m.NONE_ONE))
    channel = m.coloration_channel_value(v1, 1)
    keyframes = (m.Keyframe(2, 0, {"iter": 40, "dist": 2.5, "outline": False,
                                   "obst:1:rotation": 15.0, "coloration:5:color_model": "HSV",
                                   "coloration:5:chan1": channel, "coloration:0:chan0": m.coloration_channel_value(v0, 0)}),
                 m.Keyframe(7, 30, {"obst:3:r": 0.5, "emit_mode": m.EMIT_TOWARDS}))
    ray, natural_termination = _generation(
        max_iter=77, max_dist=3.5, seg_mult=0.8, emit_mode=m.EMIT_TOWARDS, emit_target_r=0.3,
        emit_offset=12.5, confine_lookback=6, escape_radius=2.0, confine_threshold=2.0 / 512,
        fuzz=m.DEFAULT_FUZZ.replace(enabled=True, emit_jitter=2.0, rng_seed=99))
    blur = m.BuddhaBlurSettings(radius=0.03, type=m.BUDDHA_BLUR_BOKEH)
    export = {"include_all_colorations": False, "color_info_location": m.COLOR_INFO_LOCATIONS[1],
              "image_format": "png", "image_jxl_quality": 85, "image_width": "640",
              "video_fps": "24", "video_format": "JXL", "video_quality": 60.5,
              "video_jxl_mode": "Lossy"}
    fields = dict(ray=ray, natural_termination=natural_termination, buddha_blur=blur, buddha_seed=OLD_SEEDING, engine=m.DEFAULT_ENGINE_SETTINGS, mandelbrot=m.DEFAULT_MANDELBROT_SETTINGS,
                  decorations={"outline": False, "pending": True},
                  obstacles=tuple(obstacles),
                  colorations=(m.snapshot_coloration(v0), m.snapshot_coloration(v1)),
                  framings=(OLD_FRAMING,), keyframes=keyframes,
                  export=export, obstacle_counter=40, coloration_counter=9, framing_counter=1, keyframe_counter=12,
                  selected_obstacle_id=5, selected_coloration_id=5, selected_framing_id=0)
    fields.update(changes)
    return m.Project(**fields)


LEGACY_KIND_COUNT = 12       # the obstacle kinds there were when the format-1 and format-2 reference files were written


def _legacy_sample_project():
    """_sample_project() as it was before the squircles and the spiral existed: the project the format-1
    and format-2 reference files in dev/testing hold (it has only the first LEGACY_KIND_COUNT kinds)."""
    return _sample_project(obstacles=_sample_project().obstacles[:LEGACY_KIND_COUNT])


def _as_saved(channel, number):
    """A colour channel as it comes back from a file: the settings its source does not use
    are not kept, so they take the channel's defaults."""
    return m._channel_from_json(m._channel_to_json(channel), "test", number)


def _round_trip(project):
    import json
    return m.project_from_dict(json.loads(m.project_text(project)))


def test_a_project_survives_being_saved_and_loaded():
    project = _sample_project()
    loaded = _round_trip(project)
    assert m.project_signature(loaded) == m.project_signature(project)
    assert loaded.ray == project.ray and loaded.buddha_blur == project.buddha_blur
    assert [v.id for v in loaded.colorations] == [0, 5]
    for want, got in zip(project.colorations, loaded.colorations):
        assert got.color_model == want.color_model
        for number in range(3):
            assert m.coloration_channel_value(got, number) == _as_saved(m.coloration_channel_value(want, number), number)
    assert [type(o) for o in loaded.obstacles] == list(m.SHAPE_CLASSES)
    assert loaded.export == project.export
    assert [o.id for o in loaded.obstacles] == [i * 2 + 1 for i in range(len(m.SHAPE_CLASSES))]
    assert all(o.state() == q.state() for o, q in zip(loaded.obstacles, project.obstacles))
    assert all(o._frozen for o in loaded.obstacles)
    assert (loaded.obstacle_counter, loaded.coloration_counter, loaded.keyframe_counter) == (40, 9, 12)
    assert (loaded.selected_obstacle_id, loaded.selected_coloration_id) == (5, 5)
    kf = loaded.keyframes[0]
    assert (kf.id, kf.frame_index) == (2, 0)
    want = project.keyframes[0].data
    assert kf.data["coloration:5:chan1"] == _as_saved(want["coloration:5:chan1"], 1)
    assert kf.data["coloration:0:chan0"] == _as_saved(want["coloration:0:chan0"], 0)
    assert {k: v for k, v in kf.data.items() if not k.startswith("coloration:") or k.endswith("color_model")} \
        == {k: v for k, v in want.items() if not k.startswith("coloration:") or k.endswith("color_model")}
    assert isinstance(kf.data["coloration:5:chan1"], m.ChannelSettings)
    assert isinstance(kf.data["iter"], int) and kf.data["outline"] is False


def _channel(source, **changes):
    fields = dict(chan_src=source, metric_type=m.METRIC_TYPE_POINT_PAIR, metric_slot=m.ST_EUCLIDEAN, point1=m.POINT_TOP,
                  point2=m.CHOICE_ORIGIN,
                  buddha_points=_buddha_choice(m.POINT_SELF_MEAN), buddha_path_kinds=m.ALL_PATH_KINDS,
                  buddha_compression=m.BUDDHA_COMPRESS_MODULUS, buddha_brightness=3,
                  constant=m.NONE_ONE)
    fields.update(changes)
    return m.ChannelSettings(**fields)


def test_a_channel_in_a_file_has_only_the_settings_its_source_uses():
    query_points = m._channel_to_json(_channel(m.CHAN_SRC_QUERY))
    assert query_points == {"source": "query", "kind": "point_metric", "metric": "euclidean_distance",
                            "point1": "bounces_uppermost", "point2": "origin"}
    query_other = m._channel_to_json(_channel(m.CHAN_SRC_QUERY, metric_type=m.METRIC_TYPE_OTHER, metric_slot=m.ST_BOUNCE_COUNT))
    assert query_other == {"source": "query", "kind": "other_metric", "metric": "bounces_count"}
    assert m._channel_to_json(_channel(m.CHAN_SRC_BUDDHA)) == {
        "source": "buddha", "buddha_points": ["self_bounces_mean"], "buddha_path_kinds": ["escaping", "non_escaping", "terminated_by_obstacle"],
        "buddha_compression": "modulus", "buddha_brightness": "8x"}
    assert m._channel_to_json(_channel(m.CHAN_SRC_CONSTANT)) == {"source": "constant",
                                                                  "constant": "one"}
    text = m.project_text(_sample_project())
    import json
    for coloration in json.loads(text)["colorations"]:
        for channel in coloration["channels"]:
            unused = {"query": ["buddha_points", "buddha_compression", "buddha_brightness",
                                "constant"],
                      "buddha": ["kind", "metric", "point1", "point2", "constant"],
                      "constant": ["kind", "metric", "point1", "point2", "buddha_points",
                                   "buddha_compression", "buddha_brightness"]}[channel["source"]]
            assert not [name for name in unused if name in channel], channel


def test_a_channel_missing_a_setting_its_source_uses_is_refused_and_unused_ones_take_defaults():
    def refused(data, number=0):
        try:
            m._channel_from_json(data, "test", number)
        except m.ProjectError as exc:
            return str(exc)
        raise AssertionError("accepted")
    assert "'buddha_compression'" in refused({"source": "buddha", "buddha_points": []})
    assert "'buddha_brightness'" in refused({"source": "buddha", "buddha_points": [],
                                             "buddha_compression": "modulus"})
    assert "'constant'" in refused({"source": "constant"})
    assert "'metric'" in refused({"source": "query", "kind": "other_metric"})
    assert "'point2'" in refused({"source": "query", "kind": "point_metric",
                                  "metric": "horizontal_distance", "point1": "origin"})
    assert "'source'" in refused({"kind": "point_metric"})
    assert "does not suit" in refused({"source": "query", "kind": "other_metric",
                                       "metric": "horizontal_distance"})
    just_constant = m._channel_from_json({"source": "constant", "constant": "one"}, "test", 2)
    default = m.coloration_channel_value(m.ColorationConfig(0), 2)
    assert just_constant == default.replace(chan_src=m.CHAN_SRC_CONSTANT, constant=m.NONE_ONE)
    assert m._channel_from_json({"source": "query", "kind": "other_metric", "metric": "bounces_count"},
                                "test", 1).point1 == m.coloration_channel_value(m.ColorationConfig(0), 1).point1


def test_a_file_that_keeps_unused_channel_settings_still_loads_with_them():
    """The first files of format 2 held every setting of every channel; they must keep loading,
    and what they kept is kept."""
    everything = {"source": "query", "kind": "point_metric", "metric": "manhattan_distance",
                  "point1": "origin", "point2": "path_last", "buddha_points": ["path_last"],
                  "buddha_compression": "modulus", "buddha_brightness": "16x", "constant": "one"}
    channel = m._channel_from_json(everything, "test", 0)
    assert channel.chan_src == m.CHAN_SRC_QUERY and channel.metric_slot == m.ST_MANHATTAN
    assert channel.buddha_compression == m.BUDDHA_COMPRESS_MODULUS and channel.buddha_brightness == 4
    assert channel.constant == m.NONE_ONE and channel.buddha_points[m.POINT_END] is True


def test_unused_channel_settings_do_not_count_as_a_change_to_the_project():
    """What is not saved cannot make the project 'unsaved'."""
    project = _sample_project()
    sig = m.project_signature(project)
    coloration = m.coloration_config_from_snapshot(project.colorations[0])
    assert coloration.chan_src[0] == m.CHAN_SRC_QUERY
    coloration.buddha_compression[0] = m.BUDDHA_COMPRESS_MODULUS            # unused: a query channel
    coloration.constant_choice[0] = m.NONE_ONE
    coloration.buddha_brightness[0] = 4
    edited = project.replace(colorations=(m.snapshot_coloration(coloration), project.colorations[1]))
    assert m.project_signature(edited) == sig
    coloration.chan_src[0] = m.CHAN_SRC_BUDDHA                              # now they are used
    changed = project.replace(colorations=(m.snapshot_coloration(coloration), project.colorations[1]))
    assert m.project_signature(changed) != sig


def test_a_project_that_cannot_be_saved_counts_as_having_unsaved_changes():
    class FakeApp:
        pass
    app = FakeApp()
    ok = _sample_project()
    app.saved_signature = m.project_signature(ok)
    app.capture_project = lambda show_error=True: ok
    app.is_dirty = m.App.is_dirty.__get__(app)
    assert app.is_dirty() is False
    app.capture_project = lambda show_error=True: ok.replace(ray=ok.ray.replace(seg_mult=float("nan")))
    assert app.is_dirty() is True
    app.capture_project = lambda show_error=True: None
    assert app.is_dirty() is True


def test_a_project_file_is_pretty_printed_versioned_json():
    import json
    text = m.project_text(_sample_project())
    assert text.startswith('{\n  "format": "light-bounces-project",\n  "format_version": 9,')
    assert text.endswith("}\n") and "\r" not in text
    assert all(len(line) < 120 for line in text.split("\n"))            # one value per line
    data = json.loads(text)
    assert data["format"] == m.PROJECT_FORMAT_NAME and data["format_version"] == 9
    assert data["program_version"] == m.FILE_VERSION
    assert [o["kind"] for o in data["obstacles"]] == [cls.KEY for cls in m.SHAPE_CLASSES]
    assert data["obstacles"][10]["kind"] == "lune" and data["selected"] == {"obstacle": 5, "coloration": 5, "framing": 0}
    assert m.PROJECT_EXTENSION == ".lightbounce"


def test_saving_and_loading_a_project_file_on_disk(tmp=None):
    path = os.path.join(tempfile.mkdtemp(), "mine" + m.PROJECT_EXTENSION)
    project = _sample_project()
    m.save_project(path, project)
    assert m.project_signature(m.load_project(path)) == m.project_signature(project)
    m.save_project(path, _sample_project(obstacle_counter=99))              # replaced in place
    assert m.load_project(path).obstacle_counter == 99
    assert os.listdir(os.path.dirname(path)) == ["mine" + m.PROJECT_EXTENSION]
    with open(path, "rb") as f:
        assert f.read().count(b"\n") > 100


def _dict_of(project=None):
    import json
    return json.loads(m.project_text(project or _sample_project()))


def _refused(data, expected_text):
    try:
        m.project_from_dict(data)
    except m.ProjectError as exc:
        assert expected_text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected a ProjectError mentioning {expected_text!r}")


def test_a_file_that_is_not_a_usable_project_is_refused_with_a_clear_message():
    d = _dict_of()
    _refused([], "not a Prismatic Fractals project")
    _refused({**d, "format": "something-else"}, "not a Prismatic Fractals project")
    _refused({**d, "format_version": 99}, "newer version")
    _refused({**d, "format_version": "1"}, "format version")
    _refused({**d, "format_version": 0}, "format version")
    _refused({**d, "colorations": []}, "no colorations")
    d2 = _dict_of()
    d2["obstacles"][0]["kind"] = "HexagonShape"
    _refused(d2, "HexagonShape")
    d3 = _dict_of()
    d3["obstacles"][0]["fields"]["sides"] = 4
    _refused(d3, "does not have")
    d4 = _dict_of()
    d4["obstacles"][1]["fields"]["sides"] = 2.5
    _refused(d4, "sides")
    d5 = _dict_of()
    d5["obstacles"][1]["fields"]["sides"] = 500
    _refused(d5, "outside")
    d6 = _dict_of()
    d6["obstacles"][1]["id"] = d6["obstacles"][0]["id"]
    _refused(d6, "two obstacles have the id")
    d7 = _dict_of()
    d7["colorations"][1]["id"] = 0
    _refused(d7, "two colorations have the id")
    d8 = _dict_of()
    d8["colorations"][0]["color_model"] = "CMYK"
    _refused(d8, "colour model")
    d9 = _dict_of()
    d9["colorations"][0]["channels"].pop()
    _refused(d9, "3 channels")
    d10 = _dict_of()
    d10["colorations"][0]["channels"][0]["metric"] = "no_such_metric"
    _refused(d10, "no_such_metric")
    d10b = _dict_of()
    d10b["colorations"][0]["channels"][0].update(kind="point_metric", metric="bounces_count")
    _refused(d10b, "does not suit")
    d10c = _dict_of()
    d10c["colorations"][1]["channels"][1]["buddha_points"] = ["bounces_mean", "a_future_point"]
    _refused(d10c, "a_future_point")
    d10d = _dict_of()
    d10d["obstacles"][0]["fields"]["in_effect"] = 1                      # a number, not a key
    _refused(d10d, "ray effect")
    d10e = _dict_of()
    d10e["simulation"]["emit_mode"] = "sideways"
    _refused(d10e, "sideways")
    d10f = _dict_of()
    d10f["keyframes"][0]["settings"]["obst:1:out_effect"] = "explode"
    _refused(d10f, "explode")
    d11 = _dict_of()
    d11["keyframes"][1]["id"] = d11["keyframes"][0]["id"]
    _refused(d11, "two keyframes have the id")
    d12 = _dict_of()
    d12["keyframes"][0]["settings"]["iter"] = [1, 2]
    _refused(d12, "iter")
    d13 = _dict_of()
    d13["natural_termination"]["iter"] = "many"
    _refused(d13, "damaged")
    d14 = _dict_of()
    d14["colorations"][0]["channels"][0].pop("metric")
    _refused(d14, "no 'metric'")
    d15 = _dict_of()
    d15["keyframes"][0]["settings"]["coloration:5:chan7"] = {"source": "query"}
    _refused(d15, "not a channel")


def test_loading_a_file_that_is_not_json_or_not_text_or_missing_is_refused():
    folder = tempfile.mkdtemp()
    bad = os.path.join(folder, "bad.lightbounce")
    with open(bad, "w") as f:
        f.write('{"format": "light-bounces-project", ')
    for path, text in ((bad, "line 1"), (os.path.join(folder, "missing.lightbounce"), "could not be read")):
        try:
            m.load_project(path)
        except m.ProjectError as exc:
            assert text in str(exc), str(exc)
        else:
            raise AssertionError(path)
    binary = os.path.join(folder, "binary.lightbounce")
    with open(binary, "wb") as f:
        f.write(bytes([0xff, 0xfe, 0x00, 0x80]))
    try:
        m.load_project(binary)
    except m.ProjectError as exc:
        assert "UTF-8" in str(exc)
    else:
        raise AssertionError("binary file accepted")


def test_a_project_that_cannot_be_written_as_json_says_so_and_writes_nothing():
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "x.lightbounce")
    ok = _sample_project()
    not_a_number = ok.replace(ray=ok.ray.replace(seg_mult=float("nan")))
    try:
        m.save_project(path, not_a_number)
    except m.ProjectError:
        pass
    else:
        raise AssertionError("NaN was saved")
    assert os.listdir(folder) == []
    try:
        m.save_project(os.path.join(folder, "no", "such", "dir", "x.lightbounce"), ok)
    except m.ProjectError as exc:
        assert "could not be written" in str(exc)
    else:
        raise AssertionError("an impossible path was accepted")


def test_a_file_from_another_version_is_read_leniently():
    d = _dict_of()
    del d["natural_termination"]["iter"]                    # a setting this file lacks: its default
    d["simulation"]["a_setting_from_the_future"] = 5
    d["future_section"] = {"anything": 1}
    d["export"]["a_future_choice"] = "x"
    d["counters"]["obstacle"] = 0                    # lower than the ids in use: raised above them
    d["export"]["video_format"] = "a_future_format"                   # unknown choice: control keeps its value
    d["colorations"][1]["channels"][1]["buddha_points"] = ["bounces_mean"]
    loaded = m.project_from_dict(d)
    assert "video_format" not in loaded.export
    assert loaded.natural_termination.max_iter == m.DEFAULT_NATURAL_TERMINATION_SETTINGS.max_iter
    assert loaded.obstacle_counter == max(o.id for o in loaded.obstacles) + 1
    chosen = m.coloration_channel_value(loaded.colorations[1], 1).buddha_points
    assert chosen == _buddha_choice(m.POINT_ALL_MEAN)
    d2 = _dict_of()
    d2["selected"] = {"obstacle": 12345, "coloration": 777}              # gone: first coloration, none selected
    loaded = m.project_from_dict(d2)
    assert loaded.selected_obstacle_id == -1 and loaded.selected_coloration_id == loaded.colorations[0].id


def test_the_project_signature_sees_content_but_not_selection_or_counters():
    base = _sample_project()
    sig = m.project_signature(base)
    assert m.project_signature(_sample_project()) == sig
    assert m.project_signature(_sample_project(selected_obstacle_id=1, selected_coloration_id=0,
                                               obstacle_counter=1, coloration_counter=1,
                                               keyframe_counter=1)) == sig
    moved = base.obstacles[0].with_values(r=0.75)
    assert m.project_signature(_sample_project(obstacles=(moved,) + base.obstacles[1:])) != sig
    assert m.project_signature(_sample_project(obstacles=base.obstacles[:-1])) != sig
    assert m.project_signature(base.replace(natural_termination=base.natural_termination.replace(max_iter=78))) != sig
    assert m.project_signature(base.replace(ray=base.ray.replace(seg_mult=0.9))) != sig
    assert m.project_signature(base.replace(export={**base.export, "video_fps": "25"})) != sig
    assert m.project_signature(base.replace(decorations={"outline": True, "pending": True})) != sig
    edited = (m.Keyframe(2, 0, {**base.keyframes[0].data, "iter": 41}), base.keyframes[1])
    assert m.project_signature(base.replace(keyframes=edited)) != sig
    other_coloration = m.ColorationConfig(5)
    assert m.project_signature(base.replace(colorations=(base.colorations[0], m.snapshot_coloration(other_coloration)))) != sig


def test_the_export_option_table_names_each_choice_once_with_sensible_allowed_values():
    keys = [key for key, _path, _kind, _allowed in m.EXPORT_OPTIONS]
    assert len(keys) == len(set(keys))
    paths = [path for _key, path, _kind, _allowed in m.EXPORT_OPTIONS]
    assert len(paths) == len(set(paths))
    for _key, _path, kind, allowed in m.EXPORT_OPTIONS:
        assert kind in ("choice", "text", "bool", "number")
        assert (kind == "choice") == bool(allowed)
    assert "JXL" in m.VIDEO_FORMATS and "Constant quality" in m.VIDEO_BITRATE_MODES


# A project file calls every choice by a permanent string KEY, which a Catalog translates to the
# program's own code. Both sides are pinned: a key may be added but never removed, renamed or
# pointed at another meaning; the codes themselves may be renumbered freely (the file does not
# hold them). (If you only reworded a label that a key is pinned to, update it here.)
PINNED_KEYS = {
    "CAT_CHAN_SOURCE": {"query": "CHAN_SRC_QUERY", "buddha": "CHAN_SRC_BUDDHA",
                        "constant": "CHAN_SRC_CONSTANT"},
    "CAT_METRIC_TYPE": {"point_metric": "METRIC_TYPE_POINT_PAIR", "other_metric": "METRIC_TYPE_OTHER"},
    "CAT_SEQUENCE": {"path": "SEQ_PATH", "bounces": "SEQ_BOUNCES", "boundary_bounces": "SEQ_BOUNDARY",
                     "self_bounces": "SEQ_SELF"},
    "CAT_POINT_REDUCTION": {"first": "PRED_FIRST", "last": "PRED_LAST", "mean": "PRED_MEAN",
                            "uppermost": "PRED_TOP", "lowermost": "PRED_BOTTOM", "rightmost": "PRED_RIGHT",
                            "leftmost": "PRED_LEFT"},
    "CAT_STAT_REDUCTION": {"count": "SRED_COUNT", "length": "SRED_LENGTH", "last_direction": "SRED_LAST_DIR",
                           "share": "SRED_SHARE"},
    "CAT_METRIC": {                   # the run's own measurements, and each sequence's reductions
        "horizontal_distance": "ST_HDIST", "vertical_distance": "ST_VDIST",
        "manhattan_distance": "ST_MANHATTAN", "euclidean_distance": "ST_EUCLIDEAN",
        "loop_area": "ST_LOOP_AREA", "mean_bounce_angle": "ST_ANGLES",
        "termination_reason": "ST_TERMINATION", "last_segment_ratio": "ST_LAST_RATIO",
        "bounces_count": "ST_BOUNCE_COUNT", "self_bounces_share": "ST_SELF_SHARE",
        "boundary_bounces_share": "ST_BOUNDARY_SHARE", "path_length": "ST_LENGTH",
        "path_last_direction": "ST_LAST_DIR"},
    "CAT_POINT": {
        "path_first": "POINT_START", "path_last": "POINT_END", "bounces_mean": "POINT_ALL_MEAN",
        "boundary_bounces_mean": "POINT_BOUNDARY_MEAN", "self_bounces_mean": "POINT_SELF_MEAN",
        "bounces_uppermost": "POINT_TOP", "bounces_lowermost": "POINT_BOTTOM",
        "bounces_rightmost": "POINT_RIGHT", "bounces_leftmost": "POINT_LEFT",
        "self_bounces_uppermost": "POINT_SELF_TOP", "self_bounces_lowermost": "POINT_SELF_BOTTOM",
        "self_bounces_rightmost": "POINT_SELF_RIGHT", "self_bounces_leftmost": "POINT_SELF_LEFT",
        "origin": "CHOICE_ORIGIN", "(-1,1)": "CHOICE_POINT_NEG1_POS1",
        "(-2,2)": "CHOICE_TOP_LEFT_POINT"},
    "CAT_PATH_KIND": {"escaping": "PATH_KIND_ESCAPING", "non_escaping": "PATH_KIND_NON_ESCAPING",
                      "terminated_by_obstacle": "PATH_KIND_TERMINATED_BY_OBSTACLE"},
    "CAT_BUDDHA_COMPRESSION": {"arctangent": "BUDDHA_COMPRESS_ARCTAN",
                               "modulus": "BUDDHA_COMPRESS_MODULUS"},
    "CAT_CONSTANT": {"zero": "NONE_ZERO", "one": "NONE_ONE"},
    "CAT_EMIT_MODE": {"up": "EMIT_UP", "towards_point": "EMIT_TOWARDS"},
    "CAT_CONFINE_METHOD": {"box": "CONFINE_BOX", "path_length": "CONFINE_PATH"},
    "CAT_BLUR_TYPE": {"gaussian": "BUDDHA_BLUR_GAUSSIAN", "triangular": "BUDDHA_BLUR_TRIANGULAR",
                      "bokeh": "BUDDHA_BLUR_BOKEH", "catadioptric_bokeh": "BUDDHA_BLUR_CATADIOPTRIC"},
    "CAT_EFFECT": {"reflect": "OBSTACLE_EFFECT_REFLECT", "terminate": "OBSTACLE_EFFECT_TERMINATE",
                   "pass_through": "OBSTACLE_EFFECT_PASS"},
}
PINNED_BUDDHA_POINT_KEYS = {          # key -> (style, the point or multi option, the limit that ended the rays)
    "bounces_mean": ("BUDDHA_SIMPLE", "POINT_ALL_MEAN", None),
    "self_bounces_mean": ("BUDDHA_SIMPLE", "POINT_SELF_MEAN", None),
    "boundary_bounces_mean": ("BUDDHA_SIMPLE", "POINT_BOUNDARY_MEAN", None),
    "path_last": ("BUDDHA_SIMPLE", "POINT_END", None),
    "path_last_distance_limit": ("BUDDHA_SIMPLE", "POINT_END", "TERM_DIST"),
    "path_last_perpendicular_collision": ("BUDDHA_SIMPLE", "POINT_END", "TERM_PERPENDICULAR"),
    "path_last_confined": ("BUDDHA_SIMPLE", "POINT_END", "TERM_CONFINED"),
    "all_self_bounces": ("BUDDHA_MULTI", "BUDDHA_MULTI_ALL_SELF", None),
    "all_segment_midpoints": ("BUDDHA_MULTI", "BUDDHA_MULTI_ALL_MID", None),
    "progressive_means_of_bounce_points": ("BUDDHA_MULTI", "BUDDHA_MULTI_PROG_BOUNCE", None),
    "progressive_means_of_segment_midpoints": ("BUDDHA_MULTI", "BUDDHA_MULTI_PROG_MID", None),
    "all_path_points": ("BUDDHA_MULTI", "BUDDHA_MULTI_ALL_PATH", None)}
# What the keys of format 4 (and 1-3: the migrations bring them to format 4's) meant: key -> the program's constant. Every
# one must still load, as the key _migrate_project_4_to_5 renames it to.
FORMAT_4_KEYS = {
    "CAT_METRIC": {
        "horizontal_distance": "ST_HDIST", "vertical_distance": "ST_VDIST",
        "manhattan_distance": "ST_MANHATTAN", "euclidean_distance": "ST_EUCLIDEAN",
        "bounces": "ST_BOUNCE_COUNT", "ray_bounce_fraction": "ST_SELF_SHARE",
        "boundary_bounce_fraction": "ST_BOUNDARY_SHARE", "loop_area": "ST_LOOP_AREA",
        "mean_bounce_angle": "ST_ANGLES", "ray_length": "ST_LENGTH",
        "last_segment_direction": "ST_LAST_DIR", "termination_reason": "ST_TERMINATION",
        "last_segment_ratio": "ST_LAST_RATIO"},
    "CAT_POINT": {
        "start_point": "POINT_START", "end_point": "POINT_END",
        "all_bounces_mean": "POINT_ALL_MEAN", "boundary_bounces_mean": "POINT_BOUNDARY_MEAN",
        "self_bounces_mean": "POINT_SELF_MEAN", "uppermost_bounce": "POINT_TOP",
        "lowermost_bounce": "POINT_BOTTOM", "rightmost_bounce": "POINT_RIGHT",
        "leftmost_bounce": "POINT_LEFT", "uppermost_self_bounce": "POINT_SELF_TOP",
        "lowermost_self_bounce": "POINT_SELF_BOTTOM", "rightmost_self_bounce": "POINT_SELF_RIGHT",
        "leftmost_self_bounce": "POINT_SELF_LEFT", "origin": "CHOICE_ORIGIN",
        "top_left_corner": "CHOICE_POINT_NEG1_POS1"}}
FORMAT_4_BUDDHA_POINT_KEYS = {        # key -> the format-5 key of the same thing
    "all_bounces_mean": "bounces_mean", "self_bounces_mean": "self_bounces_mean",
    "boundary_bounces_mean": "boundary_bounces_mean", "termination_point": "path_last",
    "termination_point_distance_limit": "path_last_distance_limit",
    "termination_point_perpendicular_collision": "path_last_perpendicular_collision",
    "termination_point_confined": "path_last_confined", "all_self_bounces": "all_self_bounces",
    "all_segment_midpoints": "all_segment_midpoints",
    "progressive_means_of_bounce_points": "progressive_means_of_bounce_points",
    "progressive_means_of_segment_midpoints": "progressive_means_of_segment_midpoints"}
PINNED_SHAPE_KINDS = {"ellipse": "EllipseShape", "regular_polygon": "RegularPolygonShape",
                      "moss_egg": "MossEggShape", "circular_sector": "CircularSectorShape",
                      "circular_segment": "CircularSegmentShape", "two_circle_lens": "LensShape",
                      "reuleaux_polygon": "ReuleauxPolygonShape", "capsule": "CapsuleShape",
                      "trefoil": "TrefoilShape", "quatrefoil": "QuatrefoilShape",
                      "lune": "LuneShape", "arbelos": "ArbelosShape",
                      "superelliptical_squircle": "SuperellipseShape",
                      "fernandez_guasti_squircle": "FernandezGuastiShape",
                      "archimedean_spiral": "ArchimedeanSpiralShape",
                      "snail_of_pythagoras": "PythagorasSnailShape",
                      "hugelschaffer_egg": "HugelschafferEggShape",
                      "sine_flower": "SineFlowerShape"}
PINNED_EXPORT_CHOICES = {             # catalog -> key -> the text the control holds
    "CAT_COLOR_INFO": {"none": "None", "overlay": "Overlay", "bottom_margin": "Bottom margin"},
    "CAT_FIGURE_INFO": {"none": "None", "left_margin": "Left margin"},
    "CAT_FRAMING_INFO": {"none": "None", "left_margin": "Left margin"},
    "CAT_IMAGE_SOURCE": {"current_image": "Current image", "new_image": "New image"},
    "CAT_IMAGE_FORMAT": {"jxl": "jxl", "png": "png", "qoi": "qoi"},
    "CAT_JXL_MODE": {"lossless": "Lossless", "lossy": "Lossy"},
    "CAT_VIDEO_FORMAT": {"mp4_h264": "mp4 (h264)", "gif": "GIF", "apng": "APNG",
                         "mkv_ffv1": "mkv (ffv1)", "jxl": "JXL"},
    "CAT_BITRATE_MODE": {"constant_quality": "Constant quality", "target_bitrate": "Target bitrate",
                         "variable_bitrate": "Variable bitrate"},
    "CAT_DOWNSAMPLE_FACTOR": {"none": "No change", "2x2": "2x2", "4x4": "4x4", "8x8": "8x8"},
    "CAT_DOWNSAMPLE_METHOD": {"mean": "Mean", "median": "Median", "sum": "Sum", "max": "Max",
                              "ffmpeg": "ffmpeg"},
}
PINNED_SIMULATION_KEYS = [        # the ray engine's own settings (the "simulation" section)
    "seg_mult", "emit_mode", "emit_target_r", "emit_target_theta", "emit_target_cx",
    "emit_target_cy", "emit_offset", "fuzz_on", "emit_jitter", "bnd_offset", "bnd_jitter",
    "bnd_mult", "self_offset", "self_jitter", "self_mult", "seed"]
PINNED_NATURAL_TERMINATION_KEYS = [      # a section of their own since format 4 (they were in "simulation")
    "iter", "dist", "escape_radius", "confine_lookback", "confine_method", "confine_threshold"]
PINNED_BUDDHA_SEED_KEYS = ["buddha_seed_camera", "buddha_seed_x_start", "buddha_seed_x_stop",
                           "buddha_seed_y_start", "buddha_seed_y_stop"]   # the "buddha_seeding" section
PINNED_BUDDHA_BLUR_KEYS = ["buddha_blur_radius", "buddha_blur_type"]   # a section of their own since format 3
PINNED_DECORATION_KEYS = ["outline", "pending", "ray_path", "ray_start", "ray_end", "guides",
                          "samples"]
PINNED_EXPORT_KEYS = [
    "include_all_colorations", "color_info_location", "figure_info_location", "framing_info_location", "image_source",
    "image_format", "image_jxl_mode", "image_jxl_quality", "image_width", "image_height",
    "video_fps", "video_width", "video_height", "video_start_frame",
    "video_number_files_from_start", "video_format",
    "video_bitrate_mode", "video_quality", "video_bitrate", "video_vbr_target", "video_vbr_max",
    "video_downsample_factor", "video_downsample_method", "video_jxl_mode", "video_jxl_quality"]
PINNED_OBSTACLE_FIELDS = {            # kind key -> the fields its project entry has
    "ellipse": {"width", "height"}, "regular_polygon": {"sides", "radius"},
    "moss_egg": {"moss_angle", "moss_size"}, "circular_sector": {"radius", "sector_span"},
    "circular_segment": {"radius", "segment_span"},
    "two_circle_lens": {"lens_radius1", "lens_radius2", "lens_sep"},
    "reuleaux_polygon": {"reuleaux_sides", "radius"},
    "capsule": {"capsule_length", "capsule_radius"}, "trefoil": {"radius", "lobe_size"},
    "quatrefoil": {"radius", "lobe_size"}, "lune": {"radius", "cut_radius", "cut_offset"},
    "arbelos": {"radius", "split"},
    "superelliptical_squircle": {"width", "height", "exponent"},
    "fernandez_guasti_squircle": {"radius", "squareness"},
    "archimedean_spiral": {"radius", "spiral_pitch", "spiral_fill"},
    "snail_of_pythagoras": {"snail_segments", "radius"},
    "hugelschaffer_egg": {"egg_a", "egg_b", "egg_shift"},
    "sine_flower": {"radius", "flower_petals", "flower_amplitude"}}
SHARED_OBSTACLE_FIELDS = {"rotation", "r", "theta", "cx", "cy", "in_effect", "out_effect",
                          "fire_inside"}


def test_the_keys_in_project_files_never_change():
    """Every key below may be in a saved project, so it must still exist and still mean the
    same. Adding keys is fine; the program's own codes may be renumbered."""
    for catalog_name, keys in PINNED_KEYS.items():
        catalog = getattr(m, catalog_name)
        for key, constant in keys.items():
            assert catalog.by_key.get(key) == getattr(m, constant), \
                f"{catalog_name}: the key {key!r} must still mean {constant}"
    for key, (style, ref, limit) in PINNED_BUDDHA_POINT_KEYS.items():
        _label, entry_style, entry_ref, entry_limit = m.BUDDHA_POINT_CATALOG[m.CAT_BUDDHA_POINT.by_key[key]]
        assert (entry_style, entry_ref, entry_limit) == (getattr(m, style), getattr(m, ref),
                                                          None if limit is None else getattr(m, limit)), key
    for key, class_name in PINNED_SHAPE_KINDS.items():
        assert m.CAT_SHAPE_KIND.by_key[key].__name__ == class_name, key
    for catalog_name, keys in PINNED_EXPORT_CHOICES.items():
        for key, text in keys.items():
            assert getattr(m, catalog_name).by_key.get(key) == text, f"{catalog_name}: {key}"
    for index, key in enumerate(("1x", "2x", "4x", "8x", "16x")):
        assert m.CAT_BUDDHA_BRIGHTNESS.by_key[key] == index == m.BUDDHA_BRIGHTNESS_OPTIONS.index(key)
    assert set(PINNED_SIMULATION_KEYS) <= set(m.RAY_SETTING_SPECS)
    assert set(PINNED_NATURAL_TERMINATION_KEYS) <= set(m.NATURAL_TERMINATION_SETTING_SPECS)
    assert set(PINNED_BUDDHA_BLUR_KEYS) <= set(m.BLUR_SETTING_SPECS)
    assert set(PINNED_BUDDHA_SEED_KEYS) <= set(m.BUDDHA_SEED_SETTING_SPECS)
    assert set(PINNED_EXPORT_KEYS) <= {row[0] for row in m.EXPORT_OPTIONS}
    assert set(PINNED_DECORATION_KEYS) <= {key for key, _label, _default in
                                           m.DECO_IMAGE + m.DECO_PREVIEW}
    for key, fields in PINNED_OBSTACLE_FIELDS.items():
        assert fields | SHARED_OBSTACLE_FIELDS <= set(m.CAT_SHAPE_KIND.by_key[key].FIELD_NAMES), key
    assert m.PROJECT_FORMAT_NAME == "light-bounces-project" and m.PROJECT_EXTENSION == ".lightbounce"


def test_a_catalog_translates_between_keys_and_codes_and_refuses_what_it_does_not_know():
    catalog = m.Catalog("colour", [("red", 7), ("green", 3)])
    assert catalog.key(7) == "red" and catalog.code("green", "here") == 3
    assert catalog.codes == [7, 3]
    for call in (lambda: catalog.code("blue", "the file"), lambda: catalog.code(3, "the file"),
                 lambda: catalog.key(99)):
        try:
            call()
        except m.ProjectError as exc:
            assert "colour" in str(exc)
        else:
            raise AssertionError("an unknown key or code was accepted")
    try:
        m.Catalog("bad", [("Has Capitals", 1)])
    except AssertionError:
        pass
    else:
        raise AssertionError("a key that is not lower-case snake case was accepted")


def test_every_key_a_format_4_file_can_hold_still_means_the_same_once_migrated():
    """Format 4 named each metric, point and Buddha point on its own; format 5 names them by sequence and reduction.
    Every old key must come out of the migration as a key that means what the old one did."""
    for catalog_name, renames in (("CAT_METRIC", m._FORMAT_4_METRIC_RENAMES), ("CAT_POINT", m._FORMAT_4_POINT_RENAMES)):
        for old_key, constant in FORMAT_4_KEYS[catalog_name].items():
            new_key = m._FORMAT_8_POINT_RENAMES.get(renames.get(old_key, old_key), renames.get(old_key, old_key))
            assert getattr(m, catalog_name).by_key.get(new_key) == getattr(m, constant), (catalog_name, old_key)
    assert set(m._FORMAT_4_METRIC_RENAMES) <= set(FORMAT_4_KEYS["CAT_METRIC"])
    assert set(m._FORMAT_4_POINT_RENAMES) <= set(FORMAT_4_KEYS["CAT_POINT"])
    for old_key, new_key in FORMAT_4_BUDDHA_POINT_KEYS.items():
        assert m._FORMAT_4_BUDDHA_RENAMES.get(old_key, old_key) == new_key, old_key
        assert new_key in m.CAT_BUDDHA_POINT.by_key, new_key
    assert set(m._FORMAT_4_BUDDHA_RENAMES) <= set(FORMAT_4_BUDDHA_POINT_KEYS) | set(FORMAT_4_KEYS["CAT_POINT"])


def test_a_format_4_channel_is_renamed_in_every_coloration_and_keyframe_by_the_migration():
    """(A format-4 file calls a coloration a "view": its "views" section and "view:<id>:..." keyframe keys.)"""
    channel = {"source": "query", "kind": "point_metric", "metric": "euclidean_distance",
               "point1": "uppermost_self_bounce", "point2": "end_point"}
    other = {"source": "query", "kind": "other_metric", "metric": "ray_bounce_fraction"}
    buddha = {"source": "buddha", "buddha_points": ["all_bounces_mean", "termination_point_confined", "all_self_bounces"],
              "buddha_compression": "modulus", "buddha_brightness": "1x"}
    data = {"format": m.PROJECT_FORMAT_NAME, "format_version": 4,
            "views": [{"id": 0, "mode": "RGB", "channels": [channel, other, buddha]}],
            "keyframes": [{"id": 1, "frame_index": 0, "settings": {"view:0:chan0": channel, "iter": 5}}]}
    migrated = m._migrate_project_4_to_5(data)
    assert migrated["format_version"] == 5 and data["format_version"] == 4          # the input is not touched
    assert migrated["views"][0]["channels"] == [
        {"source": "query", "kind": "point_metric", "metric": "euclidean_distance",
         "point1": "self_bounces_uppermost", "point2": "path_last"},
        {"source": "query", "kind": "other_metric", "metric": "self_bounces_share"},
        {"source": "buddha", "buddha_points": ["bounces_mean", "path_last_confined", "all_self_bounces"],
         "buddha_compression": "modulus", "buddha_brightness": "1x"}]
    assert migrated["keyframes"][0]["settings"]["view:0:chan0"]["point1"] == "self_bounces_uppermost"
    assert migrated["keyframes"][0]["settings"]["iter"] == 5


def test_a_project_file_holds_keys_for_every_choice_and_no_codes():
    data = _dict_of()
    assert data["simulation"]["emit_mode"] == "towards_point"
    assert data["natural_termination"]["confine_method"] in ("box", "path_length")
    assert data["buddha_blur"] == {"buddha_blur_radius": 0.03, "buddha_blur_type": "bokeh"}
    assert set(data["simulation"]) == set(m.RAY_SETTING_SPECS)
    assert set(data["natural_termination"]) == set(m.NATURAL_TERMINATION_SETTING_SPECS)
    assert {o["fields"]["in_effect"] for o in data["obstacles"]} <= set(m.CAT_EFFECT.by_key)
    assert all(isinstance(o["fields"]["out_effect"], str) for o in data["obstacles"])
    channel = data["colorations"][1]["channels"][1]
    assert channel == {"source": "buddha",
                       "buddha_points": ["boundary_bounces_mean", "self_bounces_mean"], "buddha_path_kinds": ["escaping", "non_escaping", "terminated_by_obstacle"],
                       "buddha_compression": "modulus", "buddha_brightness": "8x"}
    settings = data["keyframes"][0]["settings"]
    assert settings["coloration:5:chan1"] == channel and settings["iter"] == 40
    assert data["keyframes"][1]["settings"]["emit_mode"] == "towards_point"
    assert data["export"]["video_format"] == "jxl" and data["export"]["color_info_location"] == "overlay"
    assert data["export"]["video_jxl_mode"] == "lossy" and data["export"]["image_format"] == "png"


def test_self_checks_catch_a_catalog_that_misses_one_of_the_programs_codes():
    fresh = load_module()
    fresh.selfchecks.CAT_METRIC = fresh.Catalog("metric", fresh.CAT_METRIC.entries[:-1])
    _breaks_a_self_check(fresh, "the metric catalog")
    fresh = load_module()
    fresh.selfchecks.CAT_EFFECT = fresh.Catalog("ray effect", list(fresh.CAT_EFFECT.entries) + [("extra", 9)])
    _breaks_a_self_check(fresh, "the ray effect catalog")
    fresh = load_module()
    fresh.selfchecks.SETTING_CHOICE_CATALOGS = dict(fresh.SETTING_CHOICE_CATALOGS)
    del fresh.selfchecks.SETTING_CHOICE_CATALOGS["emit_mode"]
    _breaks_a_self_check(fresh, "SETTING_CHOICE_CATALOGS")
    fresh = load_module()                                 # the blur's choice is covered like any other
    fresh.selfchecks.SETTING_CHOICE_CATALOGS = dict(fresh.SETTING_CHOICE_CATALOGS)
    del fresh.selfchecks.SETTING_CHOICE_CATALOGS["buddha_blur_type"]
    _breaks_a_self_check(fresh, "SETTING_CHOICE_CATALOGS")


def test_self_checks_catch_a_shape_kind_without_its_own_key():
    fresh = load_module()
    fresh.LuneShape.KEY = fresh.ArbelosShape.KEY
    _breaks_a_self_check(fresh, "KEY")
    fresh = load_module()
    fresh.LuneShape.KEY = "Not A Key"
    _breaks_a_self_check(fresh, "KEY")


V1_PROJECT_FILE = os.path.join(ROOT, "dev", "testing",
                               "project_v1.lightbounce")


def test_a_format_1_project_file_is_converted_when_loaded():
    """dev/testing/project_v1.lightbounce is a real file written by the first version of the
    project format (numbers for choices, class names for obstacle kinds) from _sample_project();
    it must keep loading. _sample_project() must therefore not change: this file is the
    reference it is compared with."""
    import json
    with open(V1_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 1 and old["obstacles"][0]["kind"] == "EllipseShape"
    assert isinstance(old["simulation"]["emit_mode"], int)
    loaded = m.load_project(V1_PROJECT_FILE)
    assert m.project_signature(loaded) == m.project_signature(_legacy_sample_project())
    assert m.project_text(loaded).count('"format_version": 9') == 1


V2_PROJECT_FILE = os.path.join(ROOT, "dev", "testing",
                               "project_v2.lightbounce")


def test_a_format_2_project_file_is_converted_when_loaded():
    """dev/testing/project_v2.lightbounce is a real file written by format 2 from _sample_project():
    the Buddha blur's two settings are inside "simulation". Format 3 gives them a section of their
    own (the blur is not a ray-simulation setting), and the file must keep loading."""
    import json
    with open(V2_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 2 and "buddha_blur" not in old
    assert old["simulation"]["buddha_blur_radius"] == 0.03 and old["simulation"]["buddha_blur_type"] == "bokeh"
    loaded = m.load_project(V2_PROJECT_FILE)
    assert loaded.buddha_blur == m.BuddhaBlurSettings(radius=0.03, type=m.BUDDHA_BLUR_BOKEH)
    assert loaded.ray == _sample_project().ray
    assert m.project_signature(loaded) == m.project_signature(_legacy_sample_project())
    converted = json.loads(m.project_text(loaded))
    assert converted["format_version"] == 9
    assert converted["buddha_blur"] == {"buddha_blur_radius": 0.03, "buddha_blur_type": "bokeh"}
    assert set(converted["simulation"]) == set(m.RAY_SETTING_SPECS)      # the two moved, none was copied
    assert old["simulation"]["iter"] == converted["natural_termination"]["iter"] == 77


def test_a_format_2_file_without_blur_settings_loads_the_default_blur():
    import json
    with open(V2_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    for key in ("buddha_blur_radius", "buddha_blur_type"):
        del old["simulation"][key]
    assert m.project_from_dict(old).buddha_blur == m.DEFAULT_BUDDHA_BLUR
    del old["simulation"]
    loaded = m.project_from_dict(old)
    assert loaded.buddha_blur == m.DEFAULT_BUDDHA_BLUR and loaded.ray == m.DEFAULT_RAY_SETTINGS


def test_a_damaged_format_1_file_is_refused_not_half_converted():
    import json
    with open(V1_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    old["obstacles"][1]["fields"]["in_effect"] = 77
    try:
        m.project_from_dict(old)
    except m.ProjectError as exc:
        assert "ray effect" in str(exc)
    else:
        raise AssertionError("a number that is not an effect was accepted")
    old = json.load(open(V1_PROJECT_FILE, encoding="utf-8"))
    del old["views"][0]["channels"][0]["stat"]
    try:
        m.project_from_dict(old)
    except m.ProjectError as exc:
        assert "damaged" in str(exc)
    else:
        raise AssertionError("a channel without a metric was accepted")


# --------------------------------------------------------------------------- #
# Projects in the program: unsaved changes and the file dialogs
# --------------------------------------------------------------------------- #
def _fake_project_app(dirty, save_result=True):
    class FakeApp:
        pass
    app = FakeApp()
    app.project_path = None
    app.busy_with_render_or_reencode = lambda: False
    app.is_dirty = lambda: dirty
    app.saves = []
    app.save_project = lambda: app.saves.append(1) or save_result
    app.project_display_name = lambda: "Untitled"
    app.can_replace_project = m.App.can_replace_project.__get__(app)
    return app


def _with_answer(answer, fn):
    orig = (m.messagebox.askyesnocancel, m.messagebox.showinfo)
    asked = []
    m.messagebox.askyesnocancel = lambda *a, **k: asked.append(a) or answer
    m.messagebox.showinfo = lambda *a, **k: asked.append(("info",) + a)
    try:
        return fn(), asked
    finally:
        m.messagebox.askyesnocancel, m.messagebox.showinfo = orig


def test_replacing_a_project_with_no_unsaved_changes_asks_nothing():
    app = _fake_project_app(dirty=False)
    result, asked = _with_answer(None, lambda: app.can_replace_project("open another project"))
    assert result is True and asked == [] and app.saves == []


def test_unsaved_changes_are_saved_discarded_or_the_action_cancelled_as_the_user_chooses():
    app = _fake_project_app(dirty=True)
    result, asked = _with_answer(True, lambda: app.can_replace_project("start a new project"))
    assert result is True and app.saves == [1] and len(asked) == 1
    assert "Untitled" in asked[0][1] and "start a new project" in asked[0][1]
    app = _fake_project_app(dirty=True)
    assert _with_answer(False, lambda: app.can_replace_project("x"))[0] is True and app.saves == []
    app = _fake_project_app(dirty=True)
    assert _with_answer(None, lambda: app.can_replace_project("x"))[0] is False and app.saves == []
    app = _fake_project_app(dirty=True, save_result=False)      # chose Save, then cancelled the dialog
    assert _with_answer(True, lambda: app.can_replace_project("x"))[0] is False


def test_a_project_is_not_replaced_while_a_video_renders_but_the_program_may_still_close():
    app = _fake_project_app(dirty=False)
    app.busy_with_render_or_reencode = lambda: True
    result, asked = _with_answer(None, lambda: app.can_replace_project("open another project"))
    assert result is False and asked and asked[0][0] == "info"
    result, asked = _with_answer(None, lambda: app.can_replace_project("close the program",
                                                                      check_busy=False))
    assert result is True and asked == []


def _fake_save_project_app(path=None):
    class FakeApp:
        pass
    app = FakeApp()
    app.project_path = path
    app.root = None
    app.project_display_name = m.App.project_display_name.__get__(app)
    app.project_dialog_dir = lambda: tempfile.mkdtemp()
    app.captured = _sample_project()
    app.capture_project = lambda show_error=True: app.captured
    app.update_title = lambda: None
    app.save_project_to = m.App.save_project_to.__get__(app)
    app.save_project_as = m.App.save_project_as.__get__(app)
    app.save_project = m.App.save_project.__get__(app)
    return app


def test_save_goes_to_the_projects_file_and_asks_for_one_only_when_it_has_none():
    folder = tempfile.mkdtemp()
    target = os.path.join(folder, "p" + m.PROJECT_EXTENSION)
    app = _fake_save_project_app(path=target)
    assert app.save_project() is True and os.path.exists(target)
    assert app.saved_signature == m.project_signature(app.captured)

    app = _fake_save_project_app(path=None)
    seen = {}
    orig = m.filedialog.asksaveasfilename
    chosen = os.path.join(folder, "new" + m.PROJECT_EXTENSION)
    m.filedialog.asksaveasfilename = lambda *a, **k: seen.update(k) or chosen
    try:
        assert app.save_project() is True
    finally:
        m.filedialog.asksaveasfilename = orig
    assert seen["defaultextension"] == ".lightbounce" and seen["filetypes"] == m.PROJECT_FILE_TYPES
    assert seen["initialfile"] == "untitled.lightbounce"
    assert app.project_path == chosen and os.path.exists(chosen)

    app = _fake_save_project_app(path=None)                     # the dialog was cancelled
    m.filedialog.asksaveasfilename = lambda *a, **k: ""
    try:
        assert app.save_project() is False
    finally:
        m.filedialog.asksaveasfilename = orig
    assert app.project_path is None


def test_a_failed_save_reports_the_problem_and_leaves_the_project_unsaved():
    app = _fake_save_project_app(path=os.path.join(tempfile.mkdtemp(), "no", "dir", "p.lightbounce"))
    shown = []
    orig = m.messagebox.showerror
    m.messagebox.showerror = lambda *a, **k: shown.append(a)
    try:
        assert app.save_project() is False
    finally:
        m.messagebox.showerror = orig
    assert shown and "could not be written" in shown[0][1]
    assert not hasattr(app, "saved_signature")


def test_a_project_saved_with_the_old_skip_first_frame_choice_still_loads():
    """The "skip first frame" checkbox was replaced by a starting frame; its key is retired (no longer
    written) but a file that has it must still load, the key simply ignored."""
    import json
    data = json.loads(m.project_text(_sample_project()))
    data["export"]["video_skip_first_frame"] = True
    loaded = m.project_from_dict(data)
    assert "video_skip_first_frame" not in loaded.export
    assert m.project_signature(loaded) == m.project_signature(_sample_project())
    assert "video_skip_first_frame" not in m.project_text(loaded)


V3_PROJECT_FILE = os.path.join(ROOT, "dev", "testing", "project_v3.lightbounce")


def test_a_format_3_project_file_is_converted_when_loaded():
    """dev/testing/project_v3.lightbounce is a real file written by format 3 from _sample_project() (program
    version 131): every ray-simulation setting, the termination limits and the confinement among them, is in its
    "simulation" section. Format 4 gives the natural-termination settings a section of their own, and the file must
    keep loading, to exactly that project."""
    import json
    with open(V3_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 3 and "natural_stop" not in old
    assert {"iter", "dist", "escape_radius", "confine_window", "confine_method",
            "confine_threshold", "seg_mult"} <= set(old["simulation"])        # (format 3's own names)
    loaded = m.load_project(V3_PROJECT_FILE)
    assert m.project_signature(loaded) == m.project_signature(_sample_project())
    assert loaded.obstacles and len(loaded.obstacles) == len(m.SHAPE_CLASSES)
    assert loaded.natural_termination.max_iter == 77 and loaded.natural_termination.confine_lookback == 6
    converted = json.loads(m.project_text(loaded))
    assert converted["format_version"] == 9
    assert set(converted["natural_termination"]) == set(m.NATURAL_TERMINATION_SETTING_SPECS)      # the six moved ...
    assert set(converted["simulation"]) == set(m.RAY_SETTING_SPECS)                 # ... and none stayed
    assert converted["natural_termination"]["iter"] == old["simulation"]["iter"] == 77


def test_the_format_3_to_4_migration_moves_only_the_natural_termination_settings_it_finds():
    """The six settings that were in "simulation" move, unchanged, to "natural_stop"; one the file lacks stays
    lacking (its default is applied on loading), and the ray engine's own settings are left alone."""
    import json
    with open(V3_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    del old["simulation"]["escape_radius"]
    new = m._migrate_project_3_to_4(old)
    assert new["format_version"] == 4 and "escape_radius" not in new["natural_stop"]
    assert set(new["natural_stop"]) == {"iter", "dist", "confine_window", "confine_method", "confine_threshold"}
    assert "seg_mult" in new["simulation"] and "iter" not in new["simulation"]
    assert "natural_stop" not in old and "iter" in old["simulation"]          # the file's data is not modified
    assert m.project_from_dict(old).natural_termination.escape_radius == 2.0, "the default of format 8, which the migration writes into the file"
    assert m._migrate_project_3_to_4({"format_version": 3})["natural_stop"] == {}


V4_PROJECT_FILE = os.path.join(ROOT, "dev", "testing", "project_v4.lightbounce")


def test_a_format_4_project_file_is_converted_when_loaded():
    """dev/testing/project_v4.lightbounce is a real file written by format 4 from _sample_project(): its
    channels name each metric, point and Buddha point on its own ("start_point", "self_bounces_mean"). Format 5 names
    them by sequence and reduction, and the file must keep loading, to exactly that project."""
    import json
    with open(V4_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 4
    assert old["views"][0]["channels"][0]["point1"] == "start_point"
    assert old["views"][1]["channels"][1]["buddha_points"] == ["self_bounces_mean", "boundary_bounces_mean"]
    loaded = m.load_project(V4_PROJECT_FILE)
    assert m.project_signature(loaded) == m.project_signature(_sample_project())
    converted = json.loads(m.project_text(loaded))
    assert converted["format_version"] == 9
    assert converted["colorations"][0]["channels"][0]["point1"] == "path_first"
    assert converted["colorations"][0]["channels"][0]["point2"] == "path_last"
    assert converted["colorations"][1]["channels"][1]["buddha_points"] == ["boundary_bounces_mean", "self_bounces_mean"]
    assert converted["keyframes"][0]["settings"]["coloration:0:chan0"]["point1"] == "path_first"


def test_the_buddha_seeding_is_saved_in_a_section_of_its_own_and_comes_back_and_older_files_get_the_defaults():
    import json
    seeds = m.BuddhaSeedSettings(camera_pixels=False, x_start=-0.5, x_stop=0.25, y_start=-3.0, y_stop=2.0)
    project = _sample_project(buddha_seed=seeds)
    data = json.loads(m.project_text(project))
    assert data["buddha_seeding"] == {"buddha_seed_camera": False, "buddha_seed_x_start": -0.5,
                                     "buddha_seed_x_stop": 0.25, "buddha_seed_y_start": -3.0,
                                     "buddha_seed_y_stop": 2.0}
    assert m.project_from_dict(data).buddha_seed == seeds
    assert m.project_signature(project) != m.project_signature(_sample_project()), "a change of seeding is a change"
    del data["buddha_seeding"]                                     # a file from before seeding existed
    assert m.project_from_dict(data).buddha_seed == m.DEFAULT_BUDDHA_SEED


V5_PROJECT_FILE = os.path.join(ROOT, "dev", "testing", "project_v5.lightbounce")


def test_a_format_5_project_file_is_converted_when_loaded():
    """dev/testing/project_v5.lightbounce is a real file written by format 5 from _sample_project() (program
    version 139): what the program now calls a coloration is a "view" in it - the "views" section, the "view" id
    counter and selection, the "include_all_views" export choice and the "view:<id>:..." keyframe keys. Format 6
    renames them, and the file must keep loading, to exactly that project."""
    import json
    with open(V5_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 5 and "colorations" not in old
    assert [view["id"] for view in old["views"]] == [0, 5] and old["counters"]["view"] == 9
    assert old["selected"]["view"] == 5 and "include_all_views" in old["export"]
    assert {"view:5:mode", "view:5:chan1", "view:0:chan0"} <= set(old["keyframes"][0]["settings"])
    loaded = m.load_project(V5_PROJECT_FILE)
    assert m.project_signature(loaded) == m.project_signature(_sample_project())
    converted = json.loads(m.project_text(loaded))
    assert converted["format_version"] == 9 and "views" not in converted
    assert [c["id"] for c in converted["colorations"]] == [0, 5]
    assert converted["counters"]["coloration"] == 9 and "view" not in converted["counters"]
    assert converted["selected"]["coloration"] == 5 and "view" not in converted["selected"]
    assert "include_all_colorations" in converted["export"] and "include_all_views" not in converted["export"]
    assert "coloration:5:chan1" in converted["keyframes"][0]["settings"]
    assert '"view' not in m.project_text(loaded), "no key of the converted file is still called a view"


def test_the_format_5_to_6_migration_renames_only_the_view_keys_it_finds():
    """Only the five names that spell "view" change; every other section, counter and keyframe key (an obstacle's,
    a setting's) is left alone, the file's own data is not modified, and a file lacking a part is still converted."""
    import json
    with open(V5_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    new = m._migrate_project_5_to_6(old)
    assert new["format_version"] == 6 and old["format_version"] == 5 and "views" in old
    assert new["colorations"] == old["views"]
    assert new["counters"] == {"obstacle": old["counters"]["obstacle"], "coloration": old["counters"]["view"],
                               "keyframe": old["counters"]["keyframe"]}
    assert new["selected"] == {"obstacle": old["selected"]["obstacle"], "coloration": old["selected"]["view"]}
    settings = old["keyframes"][0]["settings"]
    assert new["keyframes"][0]["settings"] == {
        ("coloration:" + key[5:] if key.startswith("view:") else key): value for key, value in settings.items()}
    assert any(key.startswith("obst:") for key in new["keyframes"][0]["settings"])
    assert new["simulation"] == old["simulation"] and new["obstacles"] == old["obstacles"]
    bare = m._migrate_project_5_to_6({"format_version": 5})
    assert bare == {"format_version": 6}


V6_PROJECT_FILE = os.path.join(ROOT, "dev", "testing", "project_v6.lightbounce")


def test_a_format_6_project_file_is_converted_when_loaded():
    """dev/testing/project_v6.lightbounce is a real file written by format 6 from _sample_project() (program
    version 142): the confinement lookback is called "confine_window" in it. Format 7 renames it
    "confine_lookback", and the file must keep loading, to exactly that project."""
    import json
    with open(V6_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 6 and old["natural_stop"]["confine_window"] == 6
    assert "confine_lookback" not in old["natural_stop"]
    loaded = m.load_project(V6_PROJECT_FILE)
    assert m.project_signature(loaded) == m.project_signature(_sample_project())
    assert loaded.natural_termination.confine_lookback == 6
    converted = json.loads(m.project_text(loaded))
    assert converted["format_version"] == 9 and converted["natural_termination"]["confine_lookback"] == 6
    assert "confine_window" not in m.project_text(loaded)


def test_a_keyframe_of_a_format_6_file_that_animates_the_confinement_lookback_keeps_its_value():
    """The keyframe setting is called "confine_window" in format 6 and must still be read, as the lookback."""
    import json
    with open(V6_PROJECT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    data["keyframes"][0]["settings"]["confine_window"] = 12
    loaded = m.project_from_dict(data)
    assert loaded.keyframes[0].data["confine_lookback"] == 12 and "confine_window" not in loaded.keyframes[0].data


def test_the_format_6_to_7_migration_renames_only_the_confine_window_it_finds():
    """The natural termination setting and every keyframe's setting of that name are renamed, with their values; every
    other key is left alone, the file's own data is not modified, and a file lacking it still converts."""
    old = {"format_version": 6,
           "natural_stop": {"iter": 5, "confine_window": 9, "confine_method": "box"},
           "keyframes": [{"id": 0, "frame_index": 0, "settings": {"confine_window": 4, "iter": 7, "obst:1:width": 1.0}},
                         {"id": 1, "frame_index": 5, "settings": {"iter": 8}}]}
    new = m._migrate_project_6_to_7(old)
    assert new["format_version"] == 7 and old["format_version"] == 6
    assert new["natural_stop"] == {"iter": 5, "confine_lookback": 9, "confine_method": "box"}
    assert new["keyframes"][0]["settings"] == {"confine_lookback": 4, "iter": 7, "obst:1:width": 1.0}
    assert new["keyframes"][1]["settings"] == {"iter": 8}
    assert old["natural_stop"]["confine_window"] == 9 and "confine_window" in old["keyframes"][0]["settings"]
    assert m._migrate_project_6_to_7({"format_version": 6}) == {"format_version": 7}
    assert m._migrate_project_6_to_7({"format_version": 6, "natural_stop": {"iter": 5}})["natural_stop"] == {"iter": 5}


V7_PROJECT_FILE = os.path.join(ROOT, "dev", "testing", "project_v7.lightbounce")


def test_a_format_7_project_file_is_converted_when_loaded():
    """dev/testing/project_v7.lightbounce is a real file written by format 7 from _sample_project() (program
    version 146): the section of the natural termination settings is called "natural_stop" in it. Format 8 renames
    it "natural_termination", and the file must keep loading, to exactly that project."""
    import json
    with open(V7_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 7 and old["natural_stop"]["confine_lookback"] == 6
    assert "natural_termination" not in old
    loaded = m.load_project(V7_PROJECT_FILE)
    assert m.project_signature(loaded) == m.project_signature(_sample_project())
    assert loaded.natural_termination.max_iter == 77
    converted = json.loads(m.project_text(loaded))
    assert converted["format_version"] == 9 and converted["natural_termination"] == old["natural_stop"]
    assert "natural_stop" not in converted
    assert [c["color_model"] for c in converted["colorations"]] == [c["mode"] for c in old["colorations"]]
    assert all("mode" not in c for c in converted["colorations"])
    assert "coloration:5:color_model" in converted["keyframes"][0]["settings"]
    assert not any(k.endswith(":mode") for kf in converted["keyframes"] for k in kf["settings"])


def test_the_format_7_to_8_migration_renames_only_the_natural_stop_section_and_the_colour_models_it_finds():
    old = {"format_version": 7, "natural_stop": {"iter": 5}, "simulation": {"seg_mult": 0.5, "emit_mode": "up"},
           "colorations": [{"id": 0, "mode": "HSV", "channels": []}, {"id": 3, "channels": []}],
           "keyframes": [{"id": 0, "frame_index": 0, "settings": {"coloration:3:mode": "RGB", "coloration:3:chan0": 1,
                                                                   "emit_mode": "up", "iter": 7}}]}
    new = m._migrate_project_7_to_8(old)
    assert new == {"format_version": 8, "natural_termination": {"iter": 5},
                   "simulation": {"seg_mult": 0.5, "emit_mode": "up"},       # the emission's mode is another thing
                   "colorations": [{"id": 0, "color_model": "HSV", "channels": []}, {"id": 3, "channels": []}],
                   "keyframes": [{"id": 0, "frame_index": 0, "settings": {"coloration:3:color_model": "RGB",
                                                                           "coloration:3:chan0": 1, "emit_mode": "up",
                                                                           "iter": 7}}]}
    assert "natural_stop" in old and old["format_version"] == 7 and old["colorations"][0]["mode"] == "HSV"
    assert m._migrate_project_7_to_8({"format_version": 7}) == {"format_version": 8}


# --------------------------------------------------------------------------- #
# Projects keep the framings
# --------------------------------------------------------------------------- #
PINNED_FRAMING_KEYS = ["id", "x", "y", "diagonal", "aspect"]


def _framing_project(**changes):
    framings = (_framing(id=0), _framing(id=4, x=0.25, y=-0.5, diagonal=1.5, aspect=2.0),
                _framing(id=9, x=-3.0, diagonal=40.0, aspect=0.25))
    fields = dict(framings=framings, framing_counter=12, selected_framing_id=4)
    fields.update(changes)
    return _sample_project(**fields)


def test_framings_are_saved_in_a_section_of_their_own_and_come_back_with_their_counter_and_selection():
    import json
    project = _framing_project()
    data = json.loads(m.project_text(project))
    assert [sorted(f) for f in data["framings"]] == [sorted(PINNED_FRAMING_KEYS)] * 3
    assert data["framings"][1] == {"id": 4, "x": 0.25, "y": -0.5, "diagonal": 1.5, "aspect": 2.0}
    assert data["counters"]["framing"] == 12 and data["selected"]["framing"] == 4
    loaded = m.project_from_dict(data)
    assert loaded.framings == project.framings and loaded.framing_counter == 12 and loaded.selected_framing_id == 4
    assert m.project_signature(loaded) == m.project_signature(project)
    assert m.project_signature(project) == m.project_signature(_framing_project(selected_framing_id=0)), \
        "the selection is not content (like the other selections)"
    moved = (project.framings[0], project.framings[1].__class__(**{**project.framings[1].as_dict(), "x": 0.5}), project.framings[2])
    assert m.project_signature(project) != m.project_signature(_framing_project(framings=moved)), "a number is"
    assert m.PROJECT_FORMAT_VERSION >= 8, "framings were a section added (older files load with the defaults): no new format"


def test_a_file_without_framings_has_the_whole_world_framing_and_the_id_counter_stays_above_the_ids():
    import json
    data = json.loads(m.project_text(_framing_project()))
    del data["framings"], data["counters"]["framing"], data["selected"]["framing"]
    loaded = m.project_from_dict(data)
    assert loaded.framings == (m.DEFAULT_FRAMING,) and loaded.selected_framing_id == 0 and loaded.framing_counter == 1
    data = json.loads(m.project_text(_framing_project()))
    data["counters"]["framing"] = 2
    data["selected"]["framing"] = 777
    loaded = m.project_from_dict(data)
    assert loaded.framing_counter == 10 and loaded.selected_framing_id == 0, "above the highest id; a gone one means framing 0"


def test_a_project_with_a_damaged_framing_is_refused_and_a_missing_number_takes_its_default():
    import json
    good = json.loads(m.project_text(_framing_project()))

    def refused(mutate, text):
        data = json.loads(json.dumps(good))
        mutate(data)
        try:
            m.project_from_dict(data)
        except m.ProjectError as exc:
            assert text in str(exc), str(exc)
        else:
            raise AssertionError(f"accepted: {text}")
    refused(lambda d: d["framings"][1].update(id=0), "two framings have the id")
    refused(lambda d: d["framings"][1].update(id=-1), "id must be a whole number")
    refused(lambda d: d["framings"][1].update(x="left"), "x must be a number")
    refused(lambda d: d["framings"][1].update(x=True), "x must be a number")
    refused(lambda d: d["framings"][1].update(diagonal=0), "above 0")
    refused(lambda d: d["framings"][1].update(aspect=-2.0), "above 0")
    refused(lambda d: d["framings"][1].update(x=1e300, diagonal=1e-300), "rectangle")
    refused(lambda d: d["framings"].append("framing"), "must be an object")
    data = json.loads(json.dumps(good))
    del data["framings"][1]["aspect"]
    assert m.project_from_dict(data).framings[1].aspect == 1.0


def test_a_format_5_file_loads_with_the_whole_world_framing_selected():
    import json
    with open(V5_PROJECT_FILE, encoding="utf-8") as f:
        assert "framings" not in json.load(f)
    loaded = m.load_project(V5_PROJECT_FILE)
    assert loaded.framings == (OLD_FRAMING,) and loaded.selected_framing_id == 0, "the whole world [-1, 1], written into the file"
    assert loaded.framings != (m.DEFAULT_FRAMING,), "which is no longer the default framing"
    assert m.project_signature(loaded) == m.project_signature(_sample_project())


def test_the_keyframe_keys_of_framings_are_saved_as_numbers_and_come_back_as_floats():
    import json
    keyframes = (m.Keyframe(2, 0, {"framing:4:x": 0.5, "framing:4:diagonal": 3.0, "framing:0:aspect": 2.0, "iter": 40}),
                 m.Keyframe(7, 30, {"framing:4:x": -1.5, "framing:4:y": 0.25}))
    project = _framing_project(keyframes=keyframes)
    data = json.loads(m.project_text(project))
    assert data["keyframes"][0]["settings"]["framing:4:x"] == 0.5 and data["keyframes"][1]["settings"]["framing:4:y"] == 0.25
    loaded = m.project_from_dict(data)
    assert [kf.data for kf in loaded.keyframes] == [kf.data for kf in keyframes]
    assert all(type(v) is float for k, v in loaded.keyframes[0].data.items() if k.startswith("framing:"))
    assert m.project_signature(loaded) == m.project_signature(project)


def test_a_keyframe_with_a_damaged_framing_key_is_refused():
    import json
    base = json.loads(m.project_text(_framing_project(keyframes=(m.Keyframe(2, 0, {"iter": 40}),))))

    def refused(key, value, text):
        data = json.loads(json.dumps(base))
        data["keyframes"][0]["settings"][key] = value
        try:
            m.project_from_dict(data)
        except m.ProjectError as exc:
            assert text in str(exc), str(exc)
        else:
            raise AssertionError(f"accepted {key} = {value!r}")
    refused("framing:4:colour", 1.0, "not a framing setting")
    refused("framing:four:x", 1.0, "not a framing setting")
    refused("framing:4", 1.0, "not a framing setting")
    refused("framing:4:x:y", 1.0, "not a framing setting")
    refused("framing:4:x", "left", "must be a number")
    refused("framing:4:x", True, "must be a number")
    refused("framing:4:diagonal", 0, "above 0")
    refused("framing:4:aspect", -1.0, "above 0")
    data = json.loads(json.dumps(base))
    data["keyframes"][0]["settings"]["framing:4:x"] = -3
    assert m.project_from_dict(data).keyframes[0].data["framing:4:x"] == -3.0, "a framing's x may be anything finite"


# --------------------------------------------------------------------------- #
# Format 9: a file says explicitly what format 8 left to the defaults of its time
# --------------------------------------------------------------------------- #
V8_PROJECT_FILE = os.path.join(ROOT, "dev", "testing", "project_v8.lightbounce")


def test_a_format_8_project_file_is_converted_when_loaded():
    """dev/testing/project_v8.lightbounce is a real file written by format 8 from _sample_project() (program version 151)."""
    import json
    with open(V8_PROJECT_FILE, encoding="utf-8") as f:
        old = json.load(f)
    assert old["format_version"] == 8 and old["program_version"] == 151
    loaded = m.load_project(V8_PROJECT_FILE)
    assert m.project_signature(loaded) == m.project_signature(_sample_project())
    assert json.loads(m.project_text(loaded))["format_version"] == 9


def _format_8_file(**sections):
    """A parsed format-8 file with a coloration and what `sections` say (the rest is what a file may leave out)."""
    data = {"format": m.PROJECT_FORMAT_NAME, "format_version": 8,
            "colorations": [{"id": 0, "color_model": "RGB", "channels": [
                {"source": "constant", "constant": "zero"}, {"source": "constant", "constant": "zero"},
                {"source": "constant", "constant": "zero"}]}]}
    data.update(sections)
    return data


def test_the_format_9_migration_writes_the_defaults_of_format_8_that_a_file_leaves_out():
    data = _format_8_file(obstacles=[{"kind": "ellipse", "id": 0, "fields": {"cx": 0.25}},
                                     {"kind": "snail_of_pythagoras", "id": 1, "fields": {"snail_segments": 9}},
                                     {"kind": "capsule", "id": 2, "fields": {"capsule_radius": 0.3}}])
    migrated = m._migrate_project_8_to_9(data)
    assert migrated["format_version"] == 9 and data["format_version"] == 8, "the input is not touched"
    assert migrated["natural_termination"] == {"escape_radius": 2.0, "confine_threshold": 2.0 / 512}
    assert migrated["buddha_seeding"] == {"buddha_seed_camera": True, "buddha_seed_x_start": -1.0,
                                          "buddha_seed_x_stop": 1.0, "buddha_seed_y_start": -1.0, "buddha_seed_y_stop": 1.0}
    assert migrated["framings"] == [{"id": 0, "x": 0.0, "y": 0.0, "diagonal": 2.0 * 2.0 ** 0.5, "aspect": 1.0}]
    assert migrated["obstacles"][0]["fields"] == {"cx": 0.25, "width": 1.0, "height": 1.0}
    assert migrated["obstacles"][1]["fields"] == {"snail_segments": 9, "radius": 1.0}
    assert migrated["obstacles"][2]["fields"] == {"capsule_radius": 0.3, "capsule_length": 1.0}, "what the file holds stays"
    loaded = m.project_from_dict(data)
    assert loaded.natural_termination.escape_radius == 2.0 and loaded.natural_termination.confine_threshold == 2.0 / 512
    assert loaded.framings[0].diagonal == 2.0 * 2.0 ** 0.5
    assert (loaded.obstacles[0].width, loaded.obstacles[1].radius, loaded.obstacles[2].capsule_length) == (1.0, 1.0, 1.0)


def test_the_format_9_migration_keeps_every_value_a_file_holds():
    data = _format_8_file(
        natural_termination={"escape_radius": 3.5, "confine_threshold": 0.01, "iter": 10},
        buddha_seeding={"buddha_seed_camera": False, "buddha_seed_x_start": -0.5},
        framings=[{"id": 3, "x": 0.5, "diagonal": 1.5}],
        obstacles=[{"kind": "ellipse", "id": 0, "fields": {"width": 1.7, "height": 0.4}}])
    migrated = m._migrate_project_8_to_9(data)
    assert migrated["natural_termination"] == {"escape_radius": 3.5, "confine_threshold": 0.01, "iter": 10}
    assert migrated["buddha_seeding"]["buddha_seed_x_start"] == -0.5 and migrated["buddha_seeding"]["buddha_seed_camera"] is False
    assert migrated["buddha_seeding"]["buddha_seed_x_stop"] == 1.0, "only what the file leaves out is written"
    assert migrated["framings"] == [{"id": 3, "x": 0.5, "y": 0.0, "diagonal": 1.5, "aspect": 1.0}]
    assert migrated["obstacles"][0]["fields"] == {"width": 1.7, "height": 0.4}


def test_the_format_9_migration_leaves_the_keyframes_alone_apart_from_the_renamed_point():
    """A setting no keyframe defines is not written into one: an export takes it from what the controls held when it began."""
    keyframes = [{"id": 5, "frame_index": 30, "settings": {"escape_radius": 5.0, "iter": 20}},
                 {"id": 2, "frame_index": 10, "settings": {"seg_mult": 0.5}}]
    migrated = m._migrate_project_8_to_9(_format_8_file(keyframes=keyframes))["keyframes"]
    assert migrated == keyframes
    assert m._migrate_project_8_to_9(_format_8_file(keyframes=[]))["keyframes"] == []
    assert "keyframes" not in m._migrate_project_8_to_9(_format_8_file())


def test_a_migrated_file_does_not_depend_on_the_defaults_of_the_program():
    """The point of the migration: change the defaults (the world is going to be twice as large) and every file of format
    8 or before still loads to the same project, and an export of it - which takes a setting no keyframe defines from the
    controls, here the project's own - resolves the same settings at every keyframe."""
    import json
    files = [os.path.join(ROOT, "dev", "testing", f"project_v{n}.lightbounce") for n in range(1, 9)]
    files += [os.path.join(ROOT, "data", "projects", name) for name in sorted(os.listdir(os.path.join(ROOT, "data", "projects")))
              if name.endswith(".lightbounce")]
    changes = {"escape_radius": 4.0, "confine_threshold": 4.0 / 512, "buddha_seed_x_start": -2.0, "buddha_seed_x_stop": 2.0,
               "buddha_seed_y_start": -2.0, "buddha_seed_y_stop": 2.0}

    def read(path):
        project = m.load_project(path)
        base = m.settings_to_dict(project.ray, project.natural_termination, project.buddha_blur, project.buddha_seed,
                                  project.engine, project.mandelbrot)
        base["outline"] = bool(project.decorations.get("outline", True))
        ordered = sorted(project.keyframes, key=lambda k: k.frame_index)
        frames = [m.resolve_frame_settings(ordered, kf.frame_index, base) for kf in ordered]
        return m.project_text(project), frames

    before = {path: read(path) for path in files}
    saved_settings, saved_framing = dict(m.DEFAULT_SETTING_VALUES), m.projects.DEFAULT_FRAMING
    saved_diagonal = m.framings.DEFAULT_FRAMING_DIAGONAL
    try:
        m.DEFAULT_SETTING_VALUES.update(changes)
        m.framings.DEFAULT_FRAMING_DIAGONAL = 4.0 * 2.0 ** 0.5          # (a framing a file leaves a number of out)
        m.projects.DEFAULT_FRAMING = m.FramingSnapshot(id=0, x=0.0, y=0.0, diagonal=4.0 * 2.0 ** 0.5, aspect=1.0)
        after = {path: read(path) for path in files}
    finally:
        m.DEFAULT_SETTING_VALUES.clear()
        m.DEFAULT_SETTING_VALUES.update(saved_settings)
        m.projects.DEFAULT_FRAMING = saved_framing
        m.framings.DEFAULT_FRAMING_DIAGONAL = saved_diagonal
    for path in files:
        assert before[path][0] == after[path][0], path
        assert before[path][1] == after[path][1], path
    assert any(len(frames) > 0 for _text, frames in before.values()), "some file has keyframes"
    assert json.loads(before[files[0]][0])["format_version"] == 9


def test_the_point_choice_top_left_corner_is_the_key_minus_one_one_from_format_9_on():
    channel = {"source": "query", "kind": "point_metric", "metric": "horizontal_distance",
               "point1": "top_left_corner", "point2": "origin"}
    other = {"source": "query", "kind": "point_metric", "metric": "euclidean_distance",
             "point1": "path_first", "point2": "top_left_corner"}
    data = _format_8_file(keyframes=[{"id": 1, "frame_index": 0, "settings": {"coloration:0:chan1": other, "iter": 5}}])
    data["colorations"][0]["channels"][0] = channel
    migrated = m._migrate_project_8_to_9(data)
    assert migrated["colorations"][0]["channels"][0]["point1"] == "(-1,1)"
    assert migrated["colorations"][0]["channels"][0]["point2"] == "origin"
    assert migrated["keyframes"][0]["settings"]["coloration:0:chan1"] == {**other, "point2": "(-1,1)"}
    assert channel["point1"] == "top_left_corner", "the input is not touched"
    loaded = m.project_from_dict(data)
    assert m.CAT_POINT.by_key["(-1,1)"] == m.CHOICE_POINT_NEG1_POS1 == m.N_POINTS + 1
    assert "top_left_corner" not in m.CAT_POINT.by_key
    assert m.coloration_channel_value(loaded.colorations[0], 0).point1 == m.CHOICE_POINT_NEG1_POS1
    assert '"point1": "(-1,1)"' in m.project_text(loaded)
    assert m.POINT_LABELS[m.CHOICE_POINT_NEG1_POS1] == "(-1,1)"


def test_a_key_may_be_the_coordinates_of_a_fixed_point_and_nothing_else_odd():
    pattern = m.PROJECT_KEY_PATTERN
    assert all(pattern.fullmatch(key) for key in ("(-1,1)", "(-2,2)", "(0,0)", "(10,-3)", "self_bounces_mean", "1x"))
    assert not any(pattern.fullmatch(key) for key in ("(-1, 1)", "(-1.5,1)", "(1)", "(-1,1", "Top left", "", "a-b", "(a,b)"))


def test_a_channel_comparing_the_top_left_point_is_saved_under_the_key_of_its_coordinates():
    v = m.ColorationConfig(0)
    v.point1_choice[0], v.point2_choice[0] = m.CHOICE_TOP_LEFT_POINT, m.CHOICE_POINT_NEG1_POS1
    project = _sample_project(colorations=(m.snapshot_coloration(v),), selected_coloration_id=0)
    text = m.project_text(project)
    assert '"point1": "(-2,2)"' in text and '"point2": "(-1,1)"' in text
    loaded = _round_trip(project)
    channel = m.coloration_channel_value(loaded.colorations[0], 0)
    assert (channel.point1, channel.point2) == (m.CHOICE_TOP_LEFT_POINT, m.CHOICE_POINT_NEG1_POS1)


def test_the_kinds_of_path_a_buddha_channel_scatters_are_saved_as_keys_and_a_file_without_them_means_all_three():
    import json
    v = m.ColorationConfig(0)
    v.chan_src[1] = m.CHAN_SRC_BUDDHA
    v.buddha_path_kinds[1] = (True, False, True)
    v.buddha_path_kinds[2] = (False, False, False)                  # (not a Buddha channel: not saved)
    project = _sample_project(colorations=(m.snapshot_coloration(v),), selected_coloration_id=0)
    data = json.loads(m.project_text(project))
    channel = data["colorations"][0]["channels"][1]
    assert channel["buddha_path_kinds"] == ["escaping", "terminated_by_obstacle"]
    assert "buddha_path_kinds" not in data["colorations"][0]["channels"][2]
    loaded = _round_trip(project)
    assert m.coloration_channel_value(loaded.colorations[0], 1).buddha_path_kinds == (True, False, True)
    assert m.coloration_channel_value(loaded.colorations[0], 2).buddha_path_kinds == m.ALL_PATH_KINDS
    del channel["buddha_path_kinds"]                                # a file from before
    old = m.project_from_dict(data)
    assert m.coloration_channel_value(old.colorations[0], 1).buddha_path_kinds == m.ALL_PATH_KINDS
    channel["buddha_path_kinds"] = []                               # none ticked is a choice
    assert m.coloration_channel_value(m.project_from_dict(data).colorations[0], 1).buddha_path_kinds == (False, False, False)
    for bad, text in ((["escaping", "sideways"], "not a kind of path"), ("escaping", "must be a list")):
        channel["buddha_path_kinds"] = bad
        assert _raises(m.ProjectError, lambda: m.project_from_dict(data))
    keyframe = {"id": 1, "frame_index": 0, "settings": {"coloration:0:chan1": {
        "source": "buddha", "buddha_points": ["path_last"], "buddha_path_kinds": ["non_escaping"],
        "buddha_compression": "arctangent", "buddha_brightness": "1x"}}}
    data["colorations"][0]["channels"][1] = {"source": "buddha", "buddha_points": ["path_last"], "buddha_compression": "arctangent",
                                              "buddha_brightness": "1x"}
    data["keyframes"] = [keyframe]
    assert m.project_from_dict(data).keyframes[0].data["coloration:0:chan1"].buddha_path_kinds == (False, True, False)


def test_the_path_engine_is_saved_in_a_section_of_its_own_and_a_file_without_it_means_ray_bouncing():
    import json
    project = _sample_project()
    data = json.loads(m.project_text(project))
    assert data["engine"] == {"engine": "ray_bouncing"} and data["format_version"] == 9, "a section added: no new format"
    assert _round_trip(project).engine == m.DEFAULT_ENGINE_SETTINGS
    del data["engine"]
    assert m.project_from_dict(data).engine == m.DEFAULT_ENGINE_SETTINGS
    mandelbrot = project.replace(engine=m.EngineSettings(engine=m.ENGINE_MANDELBROT))
    assert json.loads(m.project_text(mandelbrot))["engine"] == {"engine": "mandelbrot"}
    assert m.project_signature(mandelbrot) != m.project_signature(project), "a different engine is an unsaved change"
    data = json.loads(m.project_text(mandelbrot))
    assert m.project_from_dict(data).engine == m.EngineSettings(engine=m.ENGINE_MANDELBROT), "(both engines are offered)"
    data["engine"]["engine"] = "warp_drive"
    assert _raises(m.ProjectError, lambda: m.project_from_dict(data))


def test_the_keys_of_the_path_engines_never_change():
    assert m.CAT_ENGINE.by_key == {"ray_bouncing": m.ENGINE_RAY_BOUNCING, "mandelbrot": m.ENGINE_MANDELBROT}
    assert m.CAT_ENGINE.name == "path engine"


def test_the_mandelbrot_settings_are_saved_in_a_section_of_their_own_and_a_file_without_it_has_the_defaults():
    import json
    project = _sample_project().replace(mandelbrot=m.MandelbrotSettings(
        exponent=-1.5, z0=m.ORBIT_Z0_C, mapping=m.ORBIT_MAPPING_JULIA, julia_re=0.285, julia_im=0.01, escape=m.ORBIT_ESCAPE_BEYOND, pre_seg_mult=0.4))
    data = json.loads(m.project_text(project))
    assert data["mandelbrot"] == {"orbit_exponent": -1.5, "orbit_z0": "c", "orbit_mapping": "julia", "orbit_julia_re": 0.285,
                                  "orbit_julia_im": 0.01, "orbit_escape": "beyond", "orbit_pre_seg_mult": 0.4} and data["format_version"] == 9
    assert _round_trip(project).mandelbrot == project.mandelbrot
    assert m.project_signature(project) != m.project_signature(_sample_project()), "(an unsaved change)"
    del data["mandelbrot"]
    assert m.project_from_dict(data).mandelbrot == m.DEFAULT_MANDELBROT_SETTINGS
    data = json.loads(m.project_text(project))
    data["mandelbrot"]["orbit_mapping"] = "elliptic"
    assert _raises(m.ProjectError, lambda: m.project_from_dict(data)), "a key it has no code for is refused"


def test_the_keys_of_the_mandelbrot_choices_never_change():
    assert m.CAT_ORBIT_Z0.by_key == {"zero": 0, "c": 1} and m.CAT_ORBIT_MAPPING.by_key == {"mandelbrot": 0, "julia": 1}
    assert m.CAT_ORBIT_ESCAPE.by_key == {"clip": 0, "beyond": 1}
