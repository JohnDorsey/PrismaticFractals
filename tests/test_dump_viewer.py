"""Tests: dump_viewer."""
import os
import tempfile
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# The dump viewer (dev/dump_viewer.py): loading, graph and layout. The window
# itself needs a real Tk and was checked by hand.
# --------------------------------------------------------------------------- #
def _load_viewer():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dump_viewer_under_test", os.path.join(ROOT,
                                               "dev", "dump_viewer.py"))
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    return viewer


def _viewer_graph(viewer, data, char_w=7.0):
    return viewer.layout_tree(viewer.build_graph(data), char_w)


def _write_pickle(obj):
    import pickle, tempfile
    path = os.path.join(tempfile.mkdtemp(), "x.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def test_the_viewer_reads_application_classes_back_as_placeholders_without_importing_them():
    viewer = _load_viewer()
    o = m.EllipseShape(3)
    o.r = 0.75
    with _ModuleRegistered():
        path = _write_pickle({"obstacles": [o], "snap": m.DEFAULT_RAY_SETTINGS})
    data = viewer.load_pickle(path)
    held = data["obstacles"][0]
    assert isinstance(held, viewer.Placeholder) and held._qual.endswith(".EllipseShape")
    attrs = dict(viewer.object_attributes(held))
    assert attrs["id"] == 3 and attrs["r"] == 0.75 and attrs["fire_inside"] is True
    snap = data["snap"]                          # a slotted dataclass: placeholder with named attributes
    assert isinstance(snap, viewer.Placeholder)
    attrs = dict(viewer.object_attributes(snap))
    assert attrs["seg_mult"] == m.DEFAULT_RAY_SETTINGS.seg_mult
    assert attrs["emit_offset"] == m.DEFAULT_RAY_SETTINGS.emit_offset


def test_the_viewer_tolerates_classes_that_cannot_be_found():
    viewer = _load_viewer()
    data = viewer.TolerantUnpickler.find_class
    cls = viewer.placeholder_class("no_such_module_at_all", "Thing")
    assert cls.__name__ == "Thing" and cls._qual == "no_such_module_at_all.Thing"
    assert viewer.placeholder_class("no_such_module_at_all", "Thing") is cls


def test_plain_values_arrays_and_large_array_stand_ins_are_shown_inline():
    viewer = _load_viewer()
    big = {"__standin__": "large array", "shape": (512, 512), "dtype": "float32",
           "nbytes": 1048576}
    for value, expect in [(5, "5"), (None, "None"), ("hi", "'hi'"), (True, "True"),
                          ([], "[]"), ({}, "{}"), ((1, 2), "(1, 2)")]:
        assert viewer.inline_text(value) == expect, value
    assert "ndarray (3, 4) float64" in viewer.inline_text(np.zeros((3, 4)))
    assert "large array" in viewer.inline_text(big) and "1.0 MB" in viewer.inline_text(big)
    assert viewer.inline_text([1, 2]) is None and viewer.inline_text({"a": 1}) is None
    assert viewer.inline_text(m.EllipseShape(1)) is None


def test_containers_become_boxes_with_an_arrow_per_container_row():
    viewer = _load_viewer()
    data = {"meta": {"cause": "x"}, "app": {"n": 3, "items": [1, {"deep": 2}], "empty": []},
            "module": {}, "threads": []}
    g = viewer.build_graph(data)
    root = g.root
    assert root.title == "dump" and [r.key for r in root.rows] == ["meta", "app", "module", "threads"]
    assert root.rows[0].child is not None and root.rows[2].child is None     # {} is inline
    app = root.rows[1].child
    assert [r.key for r in app.rows] == ["n", "items", "empty"]
    assert app.rows[0].child is None and app.rows[1].child is not None and app.rows[2].child is None
    items = app.rows[1].child
    assert [r.key for r in items.rows] == ["[0]", "[1]"] and items.rows[1].child.rows[0].key == "deep"
    assert len(g.nodes) == 5 and len(g.edges) == 4


def test_standins_and_shared_objects_and_cycles_are_handled():
    viewer = _load_viewer()
    shared = {"x": 1, "y": [1, 2, 3]}
    cyc = {"name": "loop"}
    cyc["me"] = cyc
    standin = {"__standin__": "tk variable", "type": "tkinter.StringVar", "name": "PY_VAR1",
               "value": "256"}
    g = viewer.build_graph({"a": shared, "b": shared, "c": cyc, "d": standin})
    kinds = sorted(n.kind for n in g.nodes)
    assert kinds.count("standin") == 1 and kinds.count("dict") == 3      # shared: one box
    a_row, b_row = g.root.rows[0], g.root.rows[1]
    assert a_row.child is b_row.child
    assert [r.key for r in g.root.rows[3].child.rows] == ["type", "name", "value"]
    assert g.root.rows[3].child.title.startswith("tk variable: PY_VAR1")
    viewer.layout_tree(g)
    tree = [e for e in g.edges if e[3]]
    extra = [e for e in g.edges if not e[3]]
    assert len(tree) == len(g.nodes) - 1
    assert len(extra) == 2          # the second reference to `shared`, and the cycle's arrow to itself


def test_very_long_containers_show_a_limited_number_of_rows_and_say_so():
    viewer = _load_viewer()
    g = viewer.build_graph({"long": list(range(viewer.MAX_ROWS + 50))})
    box = g.root.rows[0].child
    assert len(box.rows) == viewer.MAX_ROWS + 1 and "50 more" in box.rows[-1].text


def _random_nested(seed, size=120):
    import random
    rng = random.Random(seed)
    pool = [{"leaf": 0}]
    for i in range(size):
        parent_choices = [rng.choice(pool) for _ in range(rng.randint(1, 6))]
        node = {f"k{j}": (c if rng.random() < 0.8 else j) for j, c in enumerate(parent_choices)}
        for j in range(rng.randint(0, 12)):
            node[f"v{j}"] = j
        if rng.random() < 0.5:
            node["list"] = [rng.choice(pool) for _ in range(rng.randint(1, 4))]
        pool.append(node)
    return {"meta": {"a": 1}, "app": pool[-1], "module": pool[len(pool) // 2], "threads": []}


def test_layout_never_overlaps_boxes_and_the_trees_own_arrows_never_cross():
    viewer = _load_viewer()
    for seed in range(8):
        g = _viewer_graph(viewer, _random_nested(seed))
        assert len(g.nodes) > 20
        assert viewer.overlapping_pairs(g) == [], seed
        assert viewer.count_crossings(g, tree_only=True) == 0, seed


def test_layout_puts_each_box_one_column_right_of_its_tree_parent_and_orders_children_by_row():
    viewer = _load_viewer()
    g = _viewer_graph(viewer, _random_nested(3))
    for node in g.nodes:
        for child in node.tree_children:
            assert child.depth == node.depth + 1 and child.x >= node.x + node.width
        ys = [c.y for c in node.tree_children]
        assert ys == sorted(ys), "children stack top to bottom in the order of the rows pointing at them"
    assert g.root.x == 0 and g.root.depth == 0


def test_a_double_click_does_the_presss_work():
    """Regression test: Tk delivers a second quick press near the first as a
    double-click INSTEAD of a press, so a drag started right after a click never
    began. The double-click handler must do the press's work."""
    import types
    viewer = _load_viewer()
    g = viewer.build_graph({"a": {"x": 1}, "n": 5})
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.vg = viewer.apply_filters(g, {})
    calls = []
    v.on_press = lambda e: calls.append(("press", e.y))
    v.on_double_click(types.SimpleNamespace(x=10, y=10))
    assert calls == [("press", 10)]
    v.vg = None
    v.on_double_click(types.SimpleNamespace(x=10, y=10))        # nothing loaded: no-op
    assert calls == [("press", 10)]


def test_the_layout_of_a_real_shaped_dump_has_no_overlaps_or_crossings():
    """A dump as debug_dump.py writes it, with stand-ins, placeholders and shared objects."""
    viewer = _load_viewer()
    dumper = _load_dumper()

    class App:
        pass
    app = App()
    app.var = _FakeVar()
    app.widget = _FakeWidget()
    app.obstacles = [m.EllipseShape(1), m.EllipseShape(2)]
    app.handle = m.FloatEntryHandle(_FakeVar(), "{:.2f}", 1.0)
    app.points = np.zeros((200, 200))
    app.same = app.obstacles
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "x.pkl")
    with _ModuleRegistered():
        dumper.dump_everything(app, {"LIMIT": 5}, path, "manual", include_large_arrays=False)
    data = viewer.load_pickle(path)
    g = _viewer_graph(viewer, data)
    assert g.root.title == "dump" and viewer.overlapping_pairs(g) == []
    assert viewer.count_crossings(g) == 0
    titles = [n.title for n in g.nodes]
    assert any(t.startswith("widget:") for t in titles) and any(t.startswith("tk variable:") for t in titles)
    assert any("EllipseShape" in t for t in titles)
    rows = {r.key: r.text for n in g.nodes for r in n.rows}
    assert "large array" in rows["points"]


def _filter_sample():
    return {"meta": {"cause": "x"},
            "app": {"count": 3, "name": "n", "items": [1, {"deep": 2}], "empty": [],
                    "obst": m.EllipseShape(1)},
            "module": {}, "threads": []}


def _view_rows(view, title_start):
    node = next(n for n in view.nodes if n.title.startswith(title_start))
    return node, [r.key for r in node.rows]


def test_type_keys_name_what_the_filters_act_on():
    viewer = _load_viewer()
    standin = {"__standin__": "tk variable", "name": "PY_VAR1"}
    big = {"__standin__": "large array", "shape": (2,), "dtype": "f8", "nbytes": 16}
    for value, want in [(None, "NoneType"), (True, "bool"), (3, "int"), (2.5, "float"),
                        ("s", "str"), ({}, "dict"), ([1], "list"), ((1,), "tuple"),
                        ({1}, "set"), (np.zeros(2), "ndarray"), (standin, "stand-in: tk variable"),
                        (big, "stand-in: large array"), (len, "function"), (int, "class")]:
        assert viewer.type_key_of(value) == want, value
    placeholder = viewer.placeholder_class("light_bounces", "Obstacle")()
    assert viewer.type_key_of(placeholder) == "light_bounces.Obstacle"


def test_the_type_inventory_lists_box_types_and_plain_value_types_with_counts():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    boxes, leaves = viewer.type_inventory(g)
    box_types = dict(boxes)
    assert box_types["dict"] == 3 and box_types["list"] == 1 and "dump" not in box_types
    assert any(k.endswith("EllipseShape") for k in box_types)
    leaf_types = dict(leaves)
    assert leaf_types["int"] >= 3 and leaf_types["str"] >= 2
    assert "list" not in leaf_types and "dict" not in leaf_types     # listed once, as box types
    assert not set(leaf_types) & set(box_types)
    assert boxes == sorted(boxes, key=lambda kv: (-kv[1], kv[0]))


def test_with_every_level_at_3_the_view_is_the_whole_graph():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {})
    assert len(view.nodes) == len(g.nodes) and len(view.edges) == len(g.edges)
    assert all(n.hidden == 0 for n in view.nodes)
    assert {n.id for n in view.nodes} == {n.id for n in g.nodes}     # the same boxes
    for orig in g.nodes:
        assert [r.key for r in orig.rows] == [r.key for r in view.by_id[orig.id].rows]


def test_level_3_shows_the_box_but_hides_its_children_and_counts_them():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {"dict": 3})
    app, rows = _view_rows(view, "dict (5)")
    assert rows == [] and app.hidden == 5                  # the box is there, empty, 5 hidden
    assert viewer.hidden_message(app.hidden) == "5 children hidden by filters"
    assert viewer.hidden_message(1) == "1 child hidden by filters"
    assert not any(n.title.startswith("list") for n in view.nodes)   # its descendants vanished
    assert [r.key for r in view.root.rows] == ["meta", "app", "threads"][:0] or \
        {"meta", "app"} <= {r.key for r in view.root.rows}


def test_level_2_keeps_a_row_in_the_parent_without_a_box_or_an_arrow():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {"list": 2})
    app, rows = _view_rows(view, "dict (5)")
    assert "items" in rows
    row = app.rows[rows.index("items")]
    assert row.child is None and row.text == "list (2)"    # a plain row naming the object
    assert not any(n.title.startswith("list") for n in view.nodes)
    assert app.hidden == 0


def test_level_1_removes_the_row_too_and_counts_it_in_the_parent():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {"list": 1, "int": 1})
    app, rows = _view_rows(view, "dict (5)")
    assert "items" not in rows and "empty" not in rows and "count" not in rows and "name" in rows
    assert app.hidden == 3                                 # items and empty (lists), count (an int)
    assert viewer.apply_filters(g, {"dict": 1}).root.hidden == 3   # meta, app and the empty module


def test_plain_value_types_hide_rows_at_1_and_0_and_levels_2_to_4_all_show_them():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    for level in (2, 3, 4):
        app, rows = _view_rows(viewer.apply_filters(g, {"str": level}), "dict (5)")
        assert "name" in rows, level
    for level in (1, 0):
        app, rows = _view_rows(viewer.apply_filters(g, {"str": level}), "dict (5)")
        assert "name" not in rows and app.hidden == 1, level


def test_a_box_stays_visible_while_any_visible_row_points_at_it():
    viewer = _load_viewer()
    shared = {"x": 1}
    g = viewer.build_graph({"meta": {}, "app": {"a": shared, "b": [shared]}, "module": {},
                            "threads": []})
    both = viewer.apply_filters(g, {})
    assert sum(1 for n in both.nodes if n.title == "dict (1)") == 1
    only_b = viewer.apply_filters(g, {"list": 4})
    assert any(n.title == "dict (1)" for n in only_b.nodes)
    # hide the list: the box is still reached directly from "a"
    no_list = viewer.apply_filters(g, {"list": 1})
    assert any(n.title == "dict (1)" for n in no_list.nodes)
    # hide the children of the dict holding both: nothing is left to reach the box
    assert not any(n.title == "dict (1)" for n in viewer.apply_filters(g, {"dict": 3}).nodes)


def test_layout_of_a_filtered_view_uses_no_room_for_hidden_boxes():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_random_nested(5)))
    full_height = max(n.y + n.height for n in g.nodes) - min(n.y for n in g.nodes)
    view = viewer.apply_filters(g, {"dict": 3})            # every dict box loses its children
    viewer.layout_tree(view)
    short = max(n.y + n.height for n in view.nodes) - min(n.y for n in view.nodes)
    assert len(view.nodes) < len(g.nodes) and short < full_height
    assert viewer.overlapping_pairs(view) == [] and viewer.count_crossings(view) == 0
    assert all(n.height == viewer.TITLE_H + viewer.FOOTER_H + 6 for n in view.nodes
               if n is not view.root and n.hidden and not n.rows)


def test_a_box_with_hidden_children_is_tall_enough_for_its_message():
    viewer = _load_viewer()
    g = viewer.build_graph({"meta": {"a": 1}, "app": {}, "module": {}, "threads": []})
    view = viewer.apply_filters(g, {"int": 1})
    viewer.measure_nodes(view, 7.0)
    meta = next(n for n in view.nodes if n.title == "dict (1)")
    assert meta.hidden == 1 and meta.rows == []
    assert meta.height == viewer.TITLE_H + viewer.FOOTER_H + 6
    assert meta.width >= len(viewer.hidden_message(1)) * 7.0


def _with_view_options(viewer, v, group=False, links_only=False, sort_by="none", desc=False):
    import types

    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value
    v.group_links_var, v.links_only_var = Var(group), Var(links_only)
    v.sort_by_var, v.sort_desc_var = Var(sort_by), Var(desc)
    v.make_view = viewer.Viewer.make_view.__get__(v)
    return v


def _viewer_with_graph(viewer, data):
    import types
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.graph = viewer.build_graph(data)
    v.char_w = 7.0
    everything = viewer.layout_tree(viewer.apply_filters(v.graph, {}))
    v.pos = {n.id: [n.x, n.y] for n in everything.nodes}
    v.levels = {}
    v.unchecked = set()
    _with_view_options(viewer, v)
    v.min_gap_var = types.SimpleNamespace(get=lambda: v.min_gap)
    v.aspect_var = types.SimpleNamespace(get=lambda: v.aspect)
    v.min_gap, v.aspect = viewer.DEFAULT_MIN_GAP, 0.0
    v.selected = set()
    v.redraws = []
    v.redraw = lambda: v.redraws.append("redraw")
    v.focus_root = lambda zoom: v.redraws.append(("focus", zoom))
    v.zoom = 0.8
    return v


def test_filtering_keeps_positions_and_rearranging_moves_only_the_visible_boxes():
    viewer = _load_viewer()
    v = _viewer_with_graph(viewer, _random_nested(2))
    v.rebuild_view()
    start = {k: tuple(p) for k, p in v.pos.items()}
    assert all((n.x, n.y) == start[n.id] for n in v.vg.nodes)
    v.levels = {"dict": 3}
    v.refilter()
    assert v.redraws[-1] == "redraw"
    assert {k: tuple(p) for k, p in v.pos.items()} == start          # nothing moved
    assert all((n.x, n.y) == start[n.id] for n in v.vg.nodes)
    hidden_ids = set(start) - set(v.vg.by_id)
    assert hidden_ids
    v.rearrange()
    assert v.redraws[-1] == ("focus", 0.8)
    assert {k: tuple(p) for k, p in v.pos.items() if k in hidden_ids} == \
        {k: start[k] for k in hidden_ids}                            # hidden keep their place
    assert any(tuple(v.pos[k]) != start[k] for k in v.vg.by_id)      # visible ones moved closer
    assert viewer.overlapping_pairs(v.vg) == []
    v.levels = {}
    v.refilter()
    assert {n.id for n in v.vg.nodes} == set(start)                  # everything back


def test_selection_drops_boxes_the_filters_hide():
    viewer = _load_viewer()
    v = _viewer_with_graph(viewer, {"meta": {}, "app": {"a": {"b": {"c": 1}}}, "module": {},
                                    "threads": []})
    v.rebuild_view()
    deep = next(n.id for n in v.vg.nodes if n.title == "dict (1)" and n.depth == 3)
    v.selected = {v.vg.root.id, deep}
    v.levels = {"dict": 3}                     # app keeps its box but loses everything under it
    v.rebuild_view()
    assert v.vg.root.id in v.selected and deep not in v.selected


def _turns(points):
    """The angle (degrees) the path turns at each interior point."""
    import math
    pts = list(zip(points[0::2], points[1::2]))
    out = []
    for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:], pts[2:]):
        a1, a2 = math.atan2(by - ay, bx - ax), math.atan2(cy - by, cx - bx)
        out.append(abs((math.degrees(a2 - a1) + 180) % 360 - 180))
    return out


def test_connections_can_be_curved_or_straight_and_the_end_points_are_the_same():
    viewer = _load_viewer()
    straight = viewer.edge_path(10, 20, 210, 120, 1.0, True)
    curved = viewer.edge_path(10, 20, 210, 120, 1.0, False)
    assert straight == (10, 20, 210, 120)
    assert curved[:2] == (10, 20) and curved[-2:] == (210, 120)


def test_swoopy_connections_are_smooth_all_along_and_leave_and_enter_horizontally():
    viewer = _load_viewer()
    for args in [(10, 20, 210, 120), (10, 20, 900, 5000), (10, 500, 400, 20), (0, 0, 60, 0.5)]:
        points = viewer.edge_path(*args, 1.0, False)
        assert len(points) // 2 >= 13                      # many short segments, not 3
        assert max(_turns(points)) < 25, (args, max(_turns(points)))
        x, y = points[0::2], points[1::2]
        assert abs(y[1] - y[0]) < 0.25 * abs(x[1] - x[0]) + 1e-9      # leaves horizontally
        assert abs(y[-1] - y[-2]) < 0.25 * abs(x[-1] - x[-2]) + 1e-9  # enters horizontally
        assert x[1] > x[0] and x[-1] > x[-2]               # rightwards at both ends
    forward = viewer.edge_path(10, 20, 410, 320, 1.0, False)
    xs = forward[0::2]
    assert all(b >= a for a, b in zip(xs, xs[1:]))         # never doubles back
    back = viewer.edge_path(300, 20, 10, 120, 1.0, False)  # an arrow running backwards
    assert back[2] > back[0] and back[-2] > back[-4] and back[:2] == (300, 20)
    assert back[-2:] == (10, 120)


def test_longer_connections_get_more_segments():
    viewer = _load_viewer()
    short = len(viewer.edge_path(0, 0, 50, 10, 1.0, False))
    long_ = len(viewer.edge_path(0, 0, 500, 3000, 1.0, False))
    assert long_ > short and long_ // 2 <= 65


def test_connections_are_bold_with_a_big_arrowhead():
    viewer = _load_viewer()
    assert viewer.EDGE_WIDTH >= 2 and viewer.ARROW_SHAPE[0] >= 10


def test_the_gap_between_a_box_and_its_children_follows_minimum_and_aspect_ratio():
    viewer = _load_viewer()
    for seed in range(4):
        for min_gap, aspect in [(110, 0.0), (40, 0.0), (300, 0.0), (20, 0.2), (200, 0.5)]:
            g = viewer.layout_tree(viewer.build_graph(_random_nested(seed)), 7.0, min_gap, aspect)
            checked = 0
            for node in g.nodes:
                for child in node.tree_children:
                    want = node.x + node.width + max(min_gap, aspect * node.span_h)
                    assert abs(child.x - want) < 1e-6, (seed, min_gap, aspect)
                    checked += 1
            assert checked > 10
            assert viewer.overlapping_pairs(g) == [], (seed, min_gap, aspect)
            assert viewer.count_crossings(g) == 0, (seed, min_gap, aspect)


def test_span_h_is_the_vertical_space_all_the_children_take_up():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_random_nested(1)), 7.0, 110, 0.0)
    fan = max(g.nodes, key=lambda n: len(n.tree_children))
    assert len(fan.tree_children) > 2
    top = min(c.y for c in fan.tree_children)
    bottom = max(c.y + c.height for c in fan.tree_children)
    assert fan.span_h >= bottom - top - 1e-6               # blocks can be taller than the boxes
    assert all(n.span_h == 0 for n in g.nodes if not n.tree_children)


def test_a_bigger_aspect_ratio_pushes_the_children_of_a_tall_fan_further_away():
    viewer = _load_viewer()
    data = _random_nested(6)
    flat = viewer.layout_tree(viewer.build_graph(data), 7.0, 110, 0.0)
    wide = viewer.layout_tree(viewer.build_graph(data), 7.0, 110, 0.4)
    fan_f = max(flat.nodes, key=lambda n: n.span_h)
    fan_w = next(n for n in wide.nodes if n.id == fan_f.id)
    gap_f = fan_f.tree_children[0].x - (fan_f.x + fan_f.width)
    gap_w = fan_w.tree_children[0].x - (fan_w.x + fan_w.width)
    assert gap_f == 110 and gap_w > 110
    assert abs(gap_w - 0.4 * fan_w.span_h) < 1e-6
    assert max(n.x for n in wide.nodes) > max(n.x for n in flat.nodes)


def test_rearranging_uses_the_arrangement_settings():
    viewer = _load_viewer()
    v = _viewer_with_graph(viewer, _random_nested(2))
    v.rebuild_view()
    v.min_gap, v.aspect = 250, 0.3
    v.rearrange()
    for node in v.vg.nodes:
        for child in node.tree_children:
            assert abs(child.x - (node.x + node.width + max(250, 0.3 * node.span_h))) < 1e-6
    assert {k: tuple(p) for k, p in v.pos.items() if k in v.vg.by_id} == \
        {n.id: (n.x, n.y) for n in v.vg.nodes}


def test_only_the_boxes_near_the_window_are_selected_for_drawing():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_random_nested(4)))
    everything = (-1e9, -1e9, 1e9, 1e9)
    assert len(viewer.nodes_in_region(g, everything)) == len(g.nodes)
    box = g.nodes[3]
    region = (box.x + 1, box.y + 1, box.x + 2, box.y + 2)
    near = viewer.nodes_in_region(g, region)
    assert box in near and len(near) < len(g.nodes)
    assert all(n.x <= region[2] and n.x + n.width >= region[0] and n.y <= region[3]
               and n.y + n.height >= region[1] for n in near)
    assert viewer.nodes_in_region(g, (-1e9, -1e9, -1e8, -1e8)) == []


def test_only_the_rows_inside_the_window_are_drawn():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph({"meta": {}, "app": {f"k{i}": i for i in range(100)},
                                               "module": {}, "threads": []}))
    app = next(n for n in g.nodes if n.title == "dict (100)")
    top = app.y + viewer.TITLE_H
    assert viewer.visible_row_range(app, top - 500, top + 10 * viewer.ROW_H - 1) == (0, 10)
    first, last = viewer.visible_row_range(app, top + 40 * viewer.ROW_H, top + 45 * viewer.ROW_H)
    assert (first, last) == (40, 46)
    assert viewer.visible_row_range(app, top + 1e6, top + 2e6) == (100, 100)
    assert viewer.visible_row_range(app, top - 100, top - 50) == (0, 0)


def test_there_are_five_levels_and_everything_starts_at_the_top_one():
    viewer = _load_viewer()
    assert viewer.LEVELS == (4, 3, 2, 1, 0)
    assert (viewer.SHOW, viewer.NO_CHILDREN, viewer.ROW_ONLY, viewer.HIDDEN,
            viewer.LOWEST) == (4, 3, 2, 1, 0)
    g = viewer.build_graph(_filter_sample())
    assert len(viewer.apply_filters(g, {}).nodes) == len(g.nodes)


def _lowest_sample():
    return {"meta": {}, "module": {}, "threads": [],
            "app": {"only": {"a": 1, "b": 2},                      # both children are ints
                    "mixed": {"a": 1, "s": "x"},                   # an int and a str
                    "nested": {"deep": {"a": 1}},                  # a box whose only child is a box
                    "empty": {}, "leaf": 5}}


def test_the_lowest_level_hides_the_object_like_level_1_and_counts_it():
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    view = viewer.apply_filters(g, {"int": 0})
    mixed, rows = _view_rows(view, "dict (2)")
    assert "a" not in rows and "s" in rows and mixed.hidden == 1      # the int is hidden+counted
    assert "leaf" not in _view_rows(view, "dict (5)")[1]              # its row is gone too


def test_a_box_whose_children_are_all_at_the_lowest_level_hides_but_keeps_its_row():
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    view = viewer.apply_filters(g, {"int": 0})
    app, rows = _view_rows(view, "dict (5)")
    assert "only" in rows                                   # the parent's row is still there ...
    row = app.rows[rows.index("only")]
    assert row.child is None and row.text == "dict (2)"     # ... as a plain row, no arrow
    assert not any(n.title == "dict (2)" and n.rows and n.rows[0].key == "a" for n in view.nodes)
    assert app.hidden == 1                                  # only "leaf" counts as hidden in app
    assert any(n.title == "dict (2)" and [r.key for r in n.rows] == ["s"] for n in view.nodes)


def test_a_box_with_any_child_above_the_lowest_level_stays_even_if_it_is_hidden_otherwise():
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    # str at level 1 is hidden, not lowest: it still counts as a child above zero
    view = viewer.apply_filters(g, {"int": 0, "str": 1})
    mixed = next(n for n in view.nodes if n.title == "dict (2)" and n.hidden == 2)
    assert mixed.rows == []                                 # shown, empty, 2 hidden by filters
    assert "mixed" in _view_rows(view, "dict (5)")[1]
    assert any(r.child is mixed for n in view.nodes for r in n.rows)


def test_the_lowest_level_only_hides_a_parent_that_has_children_and_never_the_top_box():
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    view = viewer.apply_filters(g, {"int": 0})
    app, rows = _view_rows(view, "dict (5)")
    assert "empty" in rows                                  # an empty dict has nothing at 0
    only = next(n for n in g.nodes if [r.key for r in n.rows] == ["a", "b"])
    mixed = next(n for n in g.nodes if [r.key for r in n.rows] == ["a", "s"])
    empty = next(n for n in g.nodes if n.title == "dict (0)") if any(
        n.title == "dict (0)" for n in g.nodes) else None
    assert viewer.hides_itself(only, {"int": 0})
    assert not viewer.hides_itself(mixed, {"int": 0})
    assert not viewer.hides_itself(only, {})                # nothing is at 0 by default
    assert empty is None                                    # empty dicts are inline rows
    # the top box never hides, even with every child of it at the lowest level
    sample = {"meta": {"a": 1}, "app": {"b": 2}, "module": {"c": 3}, "threads": []}
    top = viewer.apply_filters(viewer.build_graph(sample), {"dict": 0})
    assert [r.key for r in top.root.rows] == ["threads"] and top.root.hidden == 3
    only_lowest = viewer.apply_filters(viewer.build_graph(
        {"meta": {"a": 1}, "app": {"b": 2}, "module": {"c": 3}}), {"dict": 0})
    assert only_lowest.root is not None and only_lowest.root.rows == []
    assert only_lowest.root.hidden == 3 and len(only_lowest.nodes) == 1


def test_the_hiding_does_not_cascade_to_the_grandparent():
    """{"nested": {"deep": {"a": 1}}} with ints at the lowest level: "deep" hides itself
    (its only child is at 0) but keeps its row in "nested", and "nested" - whose child
    is now a plain row above zero - stays as a box."""
    viewer = _load_viewer()
    g = viewer.build_graph(_lowest_sample())
    view = viewer.apply_filters(g, {"int": 0})
    nested = next(n for n in view.nodes if n.title == "dict (1)" and n.rows
                  and n.rows[0].key == "deep")
    assert nested.rows[0].child is None and nested.rows[0].text == "dict (1)"
    assert nested.hidden == 0
    assert not any(n.rows and n.rows[0].key == "a" for n in view.nodes)   # deep's box is gone
    assert any(r.child is nested for n in view.nodes for r in n.rows)     # nested is reached


def test_each_box_knows_its_name_inside_its_tree_parent_and_the_row_that_points_at_it():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_filter_sample()))
    assert viewer.name_in_parent(g.root) == "top box"
    for node in g.nodes:
        if node is g.root:
            continue
        assert node.parent.rows[node.parent_row].child is node
        assert viewer.name_in_parent(node) == node.parent.rows[node.parent_row].key
    names = {viewer.name_in_parent(n) for n in g.nodes}
    assert {"meta", "app", "items", "obst"} <= names
    view = viewer.apply_filters(g, {"int": 1, "list": 1})       # rows removed: indexes shift
    viewer.layout_tree(view)
    for node in view.nodes:
        if node is not view.root:
            assert node.parent.rows[node.parent_row].child is node


def _zoomed_viewer(viewer, zoom):
    import types
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.zoom, v.ox, v.oy, v.char_w = zoom, 0.0, 0.0, 7.0
    v.vg = viewer.layout_tree(viewer.apply_filters(viewer.build_graph(_filter_sample()), {}))
    return _with_history(viewer, v)


def test_the_name_line_is_only_clickable_when_zoomed_in_far_enough():
    viewer = _load_viewer()
    v = _zoomed_viewer(viewer, viewer.TWO_LINE_TITLE_ZOOM)
    child = next(n for n in v.vg.nodes if n.title == "dict (9)" or n.parent is v.vg.root)
    x0, y0 = v.to_screen(child.x, child.y)
    top_half = y0 + viewer.TITLE_H * v.zoom * 0.25
    assert v.name_hit(child, x0 + 5, top_half)
    assert not v.name_hit(child, x0 + 5, y0 + viewer.TITLE_H * v.zoom * 0.75)   # the type line
    assert not v.name_hit(child, x0 + 5000, top_half)                          # past the name
    assert not v.name_hit(v.vg.root, 5, 5)                 # the top box has no parent
    v.zoom = viewer.TWO_LINE_TITLE_ZOOM - 0.01
    assert not v.name_hit(child, x0 + 5, top_half)


def test_clicking_a_name_goes_to_its_row_in_the_parent_instead_of_starting_a_drag():
    import types
    viewer = _load_viewer()
    v = _zoomed_viewer(viewer, 2.0)
    child = next(n for n in v.vg.nodes if n.parent is v.vg.root)
    calls = []
    v.vg = v.vg
    v.node_at = lambda x, y: child.id
    v.go_to_row = lambda parent, row: calls.append((parent, row))
    v.set_selection = lambda ids: calls.append(("selection", ids))
    v.selected = set()
    x0, y0 = v.to_screen(child.x, child.y)
    v.on_press(types.SimpleNamespace(x=x0 + 4, y=y0 + 3, state=0))
    assert calls == [(child.parent, child.parent_row)] and getattr(v, "drag", None) is None
    calls.clear()
    v.canvas = types.SimpleNamespace(tag_raise=lambda tag: None)
    v.on_press(types.SimpleNamespace(x=x0 + 4, y=y0 + viewer.TITLE_H * 2.0 * 0.8, state=0))
    assert v.drag["mode"] == "nodes"                        # the type line still drags the box


def test_going_to_a_row_centres_it_and_highlights_it_until_cleared():
    import types
    viewer = _load_viewer()
    v = _zoomed_viewer(viewer, 2.0)
    parent = v.vg.root
    seen = {}
    v.canvas_size = lambda: (1000, 600)
    v.set_view = lambda z, ox, oy: seen.update(zoom=z, ox=ox, oy=oy)
    v.set_info = lambda text: seen.update(info=text)
    v.root = types.SimpleNamespace(after=lambda ms, fn: seen.update(after=(ms, fn)))
    v.flash = None
    v.go_to_row(parent, 2)
    assert seen["zoom"] == 2.0 and v.flash == (parent.id, 2)
    row_y = parent.y + viewer.TITLE_H + 2 * viewer.ROW_H + viewer.ROW_H / 2
    assert abs(seen["oy"] - (row_y - 600 / 2 / 2.0)) < 1e-9       # the row is at mid-window
    assert seen["ox"] == parent.x - 30 / 2.0 and seen["after"][0] == 2000
    redraws = []
    v.redraw = lambda: redraws.append(1)
    v.clear_flash()
    assert v.flash is None and redraws == [1]
    v.clear_flash()
    assert redraws == [1]                                   # nothing to clear the second time


def test_titles_have_room_for_two_lines():
    viewer = _load_viewer()
    assert viewer.TITLE_H >= 2 * viewer.REF_FONT_PX + 6
    assert viewer.TWO_LINE_TITLE_ZOOM > 1.0


def test_rows_that_point_at_a_box_get_a_ticked_checkbox_and_other_rows_none():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {})
    app = next(n for n in view.nodes if n.title == "dict (5)")
    checks = {r.key: r.check for r in app.rows}
    assert checks == {"count": None, "name": None, "items": True, "empty": None, "obst": True}
    for node in view.nodes:
        for r in node.rows:
            assert (r.child is not None) == (r.check is True)
            assert r.origin is not None and r.origin[0] == node.id


def test_a_cleared_checkbox_hides_the_box_but_keeps_the_row_with_an_empty_checkbox():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    full = viewer.apply_filters(g, {})
    app = next(n for n in full.nodes if n.title == "dict (5)")
    items_row = next(r for r in app.rows if r.key == "items")
    view = viewer.apply_filters(g, {}, {items_row.origin})
    app2 = next(n for n in view.nodes if n.title == "dict (5)")
    row = next(r for r in app2.rows if r.key == "items")
    assert row.child is None and row.check is False and row.text == "list (2)"
    assert app2.hidden == 0                                  # not counted as hidden by filters
    assert not any(n.title.startswith("list") for n in view.nodes)   # the box is gone ...
    assert len(view.nodes) < len(full.nodes)
    assert not any(n.rows and n.rows[0].key == "deep" for n in view.nodes)   # ... and its contents
    assert [r.key for r in app2.rows] == [r.key for r in app.rows]           # same rows


def test_a_box_must_pass_both_its_checkbox_and_the_filters():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    app = next(n for n in viewer.apply_filters(g, {}).nodes if n.title == "dict (5)")
    origin = next(r for r in app.rows if r.key == "items").origin

    def list_visible(levels, unchecked):
        return any(n.title.startswith("list") for n in viewer.apply_filters(g, levels, unchecked).nodes)
    assert list_visible({}, set())                          # ticked and allowed
    assert not list_visible({}, {origin})                   # cleared
    assert not list_visible({"list": 2}, set())             # the filter keeps it a plain row
    assert not list_visible({"list": 1}, set())             # the filter hides the row
    assert not list_visible({"list": 2}, {origin})
    assert not list_visible({"dict": 3}, set())             # its parent's children are hidden
    assert list_visible({"list": 4}, set())
    # with the filter hiding the row, the checkbox has nothing to show (and keeps its state)
    hidden = viewer.apply_filters(g, {"list": 1}, {origin})
    assert "items" not in [r.key for n in hidden.nodes for r in n.rows]


def test_a_shared_box_stays_while_another_ticked_row_still_points_at_it():
    viewer = _load_viewer()
    shared = {"x": 1}
    g = viewer.build_graph({"meta": {}, "app": {"a": shared, "b": shared}, "module": {},
                            "threads": []})
    app = next(n for n in viewer.apply_filters(g, {}).nodes if n.title == "dict (2)")
    a, b = (next(r for r in app.rows if r.key == k).origin for k in ("a", "b"))
    def boxes(unchecked):
        return sum(1 for n in viewer.apply_filters(g, {}, unchecked).nodes if n.title == "dict (1)")
    assert boxes(set()) == 1 and boxes({a}) == 1 and boxes({b}) == 1 and boxes({a, b}) == 0


def test_boxes_with_checkboxes_are_wider_by_the_checkbox_room():
    viewer = _load_viewer()
    g = viewer.build_graph({"meta": {"a": 1}, "app": {"b": {"c": 1}}, "module": {}, "threads": []})
    view = viewer.apply_filters(g, {})
    viewer.measure_nodes(view, 7.0)
    app = next(n for n in view.nodes if n.rows and n.rows[0].key == "b")
    plain = next(n for n in view.nodes if n.rows and n.rows[0].key == "a")
    widest = lambda n: max(len(n.title) + 1, *(len(r.key) + 2 + len(r.text) for r in n.rows))
    assert abs(app.width - (widest(app) * 7.0 + 2 * viewer.PAD + viewer.CHECK_W)) < 1e-9
    assert abs(plain.width - (widest(plain) * 7.0 + 2 * viewer.PAD)) < 1e-9


def _with_history(viewer, v):
    """Give a stub viewer the real history methods (a recording menu and set_view)."""
    v.back_stack, v.forward_stack, v.flash = [], [], None
    v.vg = getattr(v, "vg", None)
    v.menu_states = {}
    v.view_menu = __import__("types").SimpleNamespace(
        entryconfigure=lambda label, state: v.menu_states.__setitem__(label, state))
    v.views_shown = []
    v.set_view = lambda zoom, ox, oy: (setattr(v, "zoom", zoom), setattr(v, "ox", ox),
                                       setattr(v, "oy", oy), v.views_shown.append((zoom, ox, oy)))
    for name in ("current_view", "navigate", "nav_back", "nav_forward", "show_view",
                 "update_nav_menu"):
        setattr(v, name, getattr(viewer.Viewer, name).__get__(v))
    return v


def _click_setup(viewer):
    import types
    g = viewer.build_graph({"meta": {}, "app": {"one": {"x": 1}, "num": 5}, "module": {},
                            "threads": []})
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.graph = g
    v.zoom, v.ox, v.oy, v.char_w = 1.0, 0.0, 0.0, 7.0
    v.levels, v.unchecked, v.selected = {}, set(), set()
    _with_view_options(viewer, v)
    v.min_gap_var = types.SimpleNamespace(get=lambda: 110)
    v.aspect_var = types.SimpleNamespace(get=lambda: 0.0)
    v.pos = {}
    everything = viewer.layout_tree(viewer.apply_filters(g, {}))
    v.pos = {n.id: [n.x, n.y] for n in everything.nodes}
    v.rebuild_view()
    v.link_block_until = 0.0
    _with_history(viewer, v)
    v.log = []
    v.redraw = lambda: v.log.append("redraw")
    v.go_to = lambda node: v.log.append(("go_to", node.title))
    v.canvas = types.SimpleNamespace(tag_raise=lambda tag: None,
                                     configure=lambda **kw: v.log.append(("cursor", kw)))
    v.set_selection = lambda ids: setattr(v, "selected", set(ids))
    v.set_info = lambda text: v.log.append(("info", text))
    app = next(n for n in v.vg.nodes if n.rows and n.rows[0].key == "one")
    v.node_at = lambda x, y: app.id
    v.cursor = ""
    return v, app


def _at_row(viewer, v, node, row, dx=10):
    x0, y0 = v.to_screen(node.x, node.y)
    return x0 + dx, y0 + (viewer.TITLE_H + row * viewer.ROW_H + viewer.ROW_H / 2)


def test_clicking_a_link_row_jumps_to_its_box_but_dragging_from_it_moves_the_box():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    x, y = _at_row(viewer, v, app, 0)
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    v.on_release(types.SimpleNamespace(x=x, y=y, state=0))
    assert ("go_to", "dict (1)") in v.log
    v.log.clear()
    v.link_block_until = 0.0
    v.canvas.move = lambda *a: None
    v.update_edges = lambda ids: None
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    v.on_motion(types.SimpleNamespace(x=x + 30, y=y + 20, state=0))
    v.on_release(types.SimpleNamespace(x=x + 30, y=y + 20, state=0))
    assert not any(isinstance(e, tuple) and e[0] == "go_to" for e in v.log)   # a drag, no jump
    assert "redraw" in v.log
    v.log.clear()
    x2, y2 = _at_row(viewer, v, app, 1)                       # a plain value row: no link
    v.on_press(types.SimpleNamespace(x=x2, y=y2, state=0))
    v.on_release(types.SimpleNamespace(x=x2, y=y2, state=0))
    assert not any(isinstance(e, tuple) and e[0] == "go_to" for e in v.log)


def test_a_quick_second_click_does_not_jump_again():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    x, y = _at_row(viewer, v, app, 0)
    press = lambda: (v.on_press(types.SimpleNamespace(x=x, y=y, state=0)),
                     v.on_release(types.SimpleNamespace(x=x, y=y, state=0)))
    press()
    press()                                                   # as the second click of a double-click
    assert [e for e in v.log if isinstance(e, tuple) and e[0] == "go_to"] == [("go_to", "dict (1)")]


def test_clicking_a_checkbox_toggles_it_and_does_not_start_a_drag_or_jump():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    right = v.to_screen(app.x, app.y)[0] + app.width
    x = right - viewer.PAD - viewer.CHECK_SIZE / 2
    _x, y = _at_row(viewer, v, app, 0)
    assert v.checkbox_hit(app, 0, x, y) and not v.checkbox_hit(app, 1, x, y)
    assert not v.checkbox_hit(app, 0, x - 60, y)
    origin = app.rows[0].origin
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    assert origin in v.unchecked and getattr(v, "drag", None) is None
    assert "redraw" in v.log
    app2 = next(n for n in v.vg.nodes if n.id == app.id)
    assert app2.rows[0].check is False and app2.rows[0].child is None
    assert not any(n.title == "dict (1)" and n.rows and n.rows[0].key == "x" for n in v.vg.nodes)
    v.node_at = lambda x_, y_: app2.id
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))      # tick it again
    assert origin not in v.unchecked
    assert not any(isinstance(e, tuple) and e[0] == "go_to" for e in v.log)


def test_show_all_also_ticks_every_checkbox_again():
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    v.level_vars = {}
    v.unchecked = {app.rows[0].origin}
    v.levels = {"dict": 3}
    v.reset_filters()
    assert v.unchecked == set() and v.levels == {}


def test_the_hand_cursor_shows_over_links_checkboxes_and_names_only():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    x, y = _at_row(viewer, v, app, 0)
    v.drag = None
    v.on_hover(types.SimpleNamespace(x=x, y=y, state=0))
    assert v.cursor == "hand2"                                # a link row
    x2, y2 = _at_row(viewer, v, app, 1)
    v.on_hover(types.SimpleNamespace(x=x2, y=y2, state=0))
    assert v.cursor == ""                                     # a plain value row
    right = v.to_screen(app.x, app.y)[0] + app.width
    v.on_hover(types.SimpleNamespace(x=right - viewer.PAD - 5, y=y, state=0))
    assert v.cursor == "hand2"                                # its checkbox


def _stub_with_places(viewer):
    """A viewer whose link jumps just move the view (by 100 world units per jump)."""
    v = _zoomed_viewer(viewer, 1.0)
    v.jumps = 0

    def jump():
        v.jumps += 1
        v.ox += 100.0
    v.jump = jump
    return v


def test_link_jumps_are_remembered_and_back_and_forward_walk_through_them():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    start = v.current_view()
    v.navigate(v.jump)
    second = v.current_view()
    v.navigate(v.jump)
    third = v.current_view()
    assert v.back_stack == [start, second] and v.forward_stack == []
    v.nav_back()
    assert v.current_view() == second and v.forward_stack == [third]
    v.nav_back()
    assert v.current_view() == start and v.back_stack == [] and v.forward_stack == [third, second]
    v.nav_back()                                             # nowhere further back: nothing happens
    assert v.current_view() == start
    v.nav_forward()
    assert v.current_view() == second
    v.nav_forward()
    assert v.current_view() == third and v.forward_stack == []
    v.nav_forward()
    assert v.current_view() == third


def test_a_new_link_jump_forgets_the_places_forward_would_have_gone_to():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    v.navigate(v.jump)
    v.navigate(v.jump)
    v.nav_back()
    assert v.forward_stack
    v.navigate(v.jump)
    assert v.forward_stack == []


def test_going_back_restores_the_exact_view_including_the_zoom_and_clears_the_highlight():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    v.zoom, v.ox, v.oy = 2.5, 40.0, -12.0

    def zoom_jump():
        v.zoom, v.ox, v.oy = 0.5, 9000.0, 300.0
        v.flash = (1, 2)
    v.navigate(zoom_jump)
    v.nav_back()
    assert v.current_view() == (2.5, 40.0, -12.0) and v.flash is None
    v.nav_forward()
    assert v.current_view() == (0.5, 9000.0, 300.0)


def test_the_history_is_limited():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    for _ in range(viewer.HISTORY_LIMIT + 50):
        v.navigate(v.jump)
    assert len(v.back_stack) == viewer.HISTORY_LIMIT


def test_the_menu_entries_are_greyed_out_when_there_is_nowhere_to_go():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    v.update_nav_menu()
    assert v.menu_states == {"Navigate back": "disabled", "Navigate forward": "disabled"}
    v.navigate(v.jump)
    assert v.menu_states == {"Navigate back": "normal", "Navigate forward": "disabled"}
    v.nav_back()
    assert v.menu_states == {"Navigate back": "disabled", "Navigate forward": "normal"}
    v.nav_forward()
    assert v.menu_states == {"Navigate back": "normal", "Navigate forward": "disabled"}


def test_back_and_forward_do_nothing_with_no_file_loaded():
    viewer = _load_viewer()
    v = _stub_with_places(viewer)
    v.navigate(v.jump)
    v.vg = None
    v.nav_back()
    v.nav_forward()
    assert v.views_shown == []


def test_clicking_a_link_row_or_a_name_is_remembered_but_dragging_and_the_go_to_menu_are_not():
    import types
    viewer = _load_viewer()
    v, app = _click_setup(viewer)
    x, y = _at_row(viewer, v, app, 0)
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    v.on_release(types.SimpleNamespace(x=x, y=y, state=0))
    assert len(v.back_stack) == 1                            # a click on a link row
    v.link_block_until = 0.0
    v.canvas.move = lambda *a: None
    v.update_edges = lambda ids: None
    v.on_press(types.SimpleNamespace(x=x, y=y, state=0))
    v.on_motion(types.SimpleNamespace(x=x + 30, y=y + 20, state=0))
    v.on_release(types.SimpleNamespace(x=x + 30, y=y + 20, state=0))
    assert len(v.back_stack) == 1                            # a drag is not a jump
    # a click on a box's name (two-line titles) is a jump too
    v2 = _zoomed_viewer(viewer, 2.0)
    child = next(n for n in v2.vg.nodes if n.parent is v2.vg.root)
    v2.node_at = lambda x_, y_: child.id
    v2.go_to_row = lambda parent, row: None
    v2.selected = set()
    x0, y0 = v2.to_screen(child.x, child.y)
    v2.on_press(types.SimpleNamespace(x=x0 + 4, y=y0 + 3, state=0))
    assert len(v2.back_stack) == 1 and v2.forward_stack == []


def test_the_side_buttons_map_to_back_and_forward_per_platform():
    viewer = _load_viewer()
    assert viewer.nav_button_numbers("win32") == (4, 5)      # checked with real side-button clicks
    assert viewer.nav_button_numbers("x11") == (8, 9)        # 4 and 5 are the wheel there


def test_the_view_menu_and_keyboard_are_wired_to_back_and_forward():
    """The menu entries (with shortcut hints) and the key and mouse bindings are made
    in Viewer.__init__, which needs a real window; here the wiring is read from its
    source, and tests above cover what the actions do. The real window was checked
    by hand."""
    import inspect
    viewer = _load_viewer()
    source = inspect.getsource(viewer.Viewer.__init__)
    assert 'label="Navigate back", accelerator="Alt+Left"' in source
    assert 'label="Navigate forward", accelerator="Alt+Right"' in source
    assert "command=self.nav_back" in source and "command=self.nav_forward" in source
    assert '"<Alt-Left>"' in source and '"<Alt-Right>"' in source
    assert "nav_button_numbers(windowing)" in source
    assert 'if windowing == "x11":' in source             # the wheel bindings only there


def test_an_arrow_is_drawn_only_while_it_is_long_enough_on_the_screen():
    viewer = _load_viewer()
    short, long_ = (0, 0, 100, 0), (0, 0, 100000, 40000)
    assert viewer.arrow_visible(*short, 1.0) and viewer.arrow_visible(*long_, 1.0)
    assert not viewer.arrow_visible(*short, 0.05)            # 5 px: gone
    assert viewer.arrow_visible(*long_, 0.006)               # 840 px: still there, zoomed right out
    edge = (0, 0, 300, 200)                                  # the length is measured as |dx| + |dy|
    assert viewer.arrow_visible(*edge, viewer.MIN_ARROW_PX / 500)
    assert not viewer.arrow_visible(*edge, viewer.MIN_ARROW_PX / 501)
    assert viewer.arrow_visible(300, 200, 0, 0, viewer.MIN_ARROW_PX / 500)   # direction does not matter


def test_longer_arrows_stay_visible_to_a_smaller_zoom_than_shorter_ones():
    viewer = _load_viewer()
    g = viewer.layout_tree(viewer.build_graph(_random_nested(3)))
    edges = sorted((viewer.edge_points(p, r, c) for p, r, c, _ in g.edges),
                   key=lambda e: abs(e[2] - e[0]) + abs(e[3] - e[1]))
    assert len(edges) > 20

    def gone_below(edge):
        lo, hi = 1e-9, 1e3
        for _ in range(80):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if not viewer.arrow_visible(*edge, mid) else (lo, mid)
        return hi
    zooms = [gone_below(e) for e in edges]
    assert zooms == sorted(zooms, reverse=True)              # longer -> vanishes at a smaller zoom
    assert zooms[0] > zooms[-1]
    # at a zoom where the shortest has gone, the longest is still drawn: not all at once
    z = zooms[0] * 0.99
    assert not viewer.arrow_visible(*edges[0], z) and viewer.arrow_visible(*edges[-1], z)


def test_zoomed_out_until_the_short_arrows_vanish_the_long_ones_remain():
    viewer = _load_viewer()
    rows = {f"k{i}": {"leaf": i} for i in range(60)}
    g = viewer.layout_tree(viewer.build_graph({"meta": {}, "app": rows, "module": {}, "threads": []}))
    edges = [viewer.edge_points(p, r, c) for p, r, c, _ in g.edges]
    lengths = sorted(abs(e[2] - e[0]) + abs(e[3] - e[1]) for e in edges)
    zoom = viewer.MIN_ARROW_PX / lengths[len(lengths) // 2]          # the median arrow is 12 px
    drawn = [e for e in edges if viewer.arrow_visible(*e, zoom)]
    assert 0 < len(drawn) < len(edges)                               # some, but not all, remain
    assert min(abs(e[2] - e[0]) + abs(e[3] - e[1]) for e in drawn) * zoom >= viewer.MIN_ARROW_PX
    assert all(abs(e[2] - e[0]) + abs(e[3] - e[1]) >= lengths[len(lengths) // 2] - 1e-9
               for e in drawn)


def _arrange_data():
    return {"meta": {}, "module": {}, "threads": [],
            "app": {"item10": 1, "item2": "hello world", "Alpha": {"x": 1}, "zeta": [1, 2, 3],
                    "beta": {"a": {"b": {"c": 1}}}}}


def _app_rows(viewer, data, group=False, sort_by="none", desc=False, levels=None, unchecked=(),
              links_only=False):
    g = viewer.build_graph(data)
    view = viewer.apply_filters(g, levels or {}, set(unchecked), links_only)
    viewer.arrange_rows(view, group, sort_by, desc)
    app = next(n for n in view.nodes if n.title == "dict (5)")
    return g, view, [r.key for r in app.rows]


def test_descendant_counts_handle_chains_sharing_cycles_and_self_references():
    """A box's descendants are the different boxes reachable from it, itself not counted."""
    viewer = _load_viewer()
    shared = {"x": {"y": 1}}
    cyc1, cyc2 = {}, {}
    cyc1["to2"] = cyc2
    cyc2["to1"] = cyc1
    selfish = {}
    selfish["me"] = selfish
    g = viewer.build_graph({"meta": {}, "module": {"c": cyc1, "self": selfish}, "threads": [],
                            "app": {"s1": shared, "s2": shared, "own": {"z": 1}}})
    below = {}
    for n in g.nodes:
        for r in n.rows:
            if r.child is not None:
                below[r.key] = r.child.desc_all
    assert below["x"] == 0                                   # {"y": 1} has no box below it
    assert below["s1"] == below["s2"] == 1                   # the shared box reaches x only
    assert below["own"] == 0
    assert below["c"] == 1 and below["to2"] == 1 and below["to1"] == 1   # each reaches the other
    assert below["self"] == 0 and below["me"] == 0           # a box pointing at itself: nothing else
    # a chain of four boxes under "a": a, b, c, d reach 3, 2, 1, 0 others
    chain = viewer.build_graph({"meta": {}, "app": {"a": {"b": {"c": {"d": {"e": 1}}}}},
                                "module": {}, "threads": []})
    counts = {r.key: r.child.desc_all for n in chain.nodes for r in n.rows if r.child is not None}
    assert [counts[k] for k in ("a", "b", "c", "d")] == [3, 2, 1, 0]
    assert chain.root.desc_all == 5                          # app, a, b, c, d
    # the function also works on just a part of the graph (the visible boxes)
    view = viewer.apply_filters(chain, {"dict": 3})
    assert viewer.descendant_counts(view.nodes)[view.root.id] == len(view.nodes) - 1


def test_sizes_count_everything_a_value_holds_and_arrays_their_data():
    viewer = _load_viewer()
    big = np.zeros((100, 100))
    small = {"n": 1}
    g = viewer.build_graph({"meta": {}, "app": {"arr": big, "small": small,
                                                "standin": {"__standin__": "large array",
                                                            "shape": (9,), "dtype": "f8",
                                                            "nbytes": 123456},
                                                "nest": {"one": small, "s": "x" * 500}},
                            "module": {}, "threads": []})
    app = next(n for n in g.nodes if n.title == "dict (4)")
    rows = {r.key: r for r in app.rows}
    assert rows["arr"].size >= big.nbytes
    assert rows["standin"].size >= 123456
    assert rows["nest"].size > rows["small"].size + 500      # holds the small box and a long string
    assert rows["nest"].child.size == rows["nest"].size
    assert app.size >= rows["arr"].size + rows["standin"].size + rows["nest"].size
    assert g.root.size >= app.size


def test_arrange_rows_does_nothing_unless_asked():
    viewer = _load_viewer()
    _g, view, keys = _app_rows(viewer, _arrange_data())
    assert keys == ["item10", "item2", "Alpha", "zeta", "beta"]


def test_links_can_be_grouped_first_keeping_the_order_inside_each_group():
    viewer = _load_viewer()
    _g, _view, keys = _app_rows(viewer, _arrange_data(), group=True)
    assert keys == ["Alpha", "zeta", "beta", "item10", "item2"]


def test_a_row_only_counts_as_a_link_while_its_box_is_visible():
    viewer = _load_viewer()
    g = viewer.build_graph(_arrange_data())
    app_orig = next(n for n in g.nodes if n.title == "dict (5)")
    zeta_origin = (app_orig.id, next(i for i, r in enumerate(app_orig.rows) if r.key == "zeta"))
    _g, _v, keys = _app_rows(viewer, _arrange_data(), group=True, unchecked=[zeta_origin])
    # zeta's box is hidden by its checkbox, so its row is no longer a link
    assert keys[:2] == ["Alpha", "beta"] and keys.index("zeta") > keys.index("beta")
    _g, _v, keys = _app_rows(viewer, _arrange_data(), group=True, levels={"list": 2})
    assert keys == ["Alpha", "beta", "item10", "item2", "zeta"]


def test_rows_sort_by_name_with_numbers_in_natural_order_ascending_and_descending():
    viewer = _load_viewer()
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="name")
    assert keys == ["Alpha", "beta", "item2", "item10", "zeta"]
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="name", desc=True)
    assert keys == ["zeta", "item10", "item2", "beta", "Alpha"]
    assert viewer.natural_key("item10") > viewer.natural_key("item2")
    assert viewer.natural_key("B") > viewer.natural_key("a")          # case-insensitive


def test_grouping_comes_first_then_the_sort_within_each_group():
    viewer = _load_viewer()
    _g, _v, keys = _app_rows(viewer, _arrange_data(), group=True, sort_by="name")
    assert keys == ["Alpha", "beta", "zeta", "item2", "item10"]
    _g, _v, keys = _app_rows(viewer, _arrange_data(), group=True, sort_by="name", desc=True)
    assert keys == ["zeta", "beta", "Alpha", "item10", "item2"]


def test_rows_sort_by_size_in_bytes():
    viewer = _load_viewer()
    g, _v, asc = _app_rows(viewer, _arrange_data(), sort_by="size")
    sizes = {r.key: r.size for n in g.nodes if n.title == "dict (5)" for r in n.rows}
    assert [sizes[k] for k in asc] == sorted(sizes.values())
    assert asc.index("item10") < asc.index("item2")                  # an int is smaller than a str
    assert asc.index("Alpha") < asc.index("beta")                    # beta holds much more
    _g, _v, desc = _app_rows(viewer, _arrange_data(), sort_by="size", desc=True)
    assert [sizes[k] for k in desc] == sorted(sizes.values(), reverse=True)


def test_rows_sort_by_total_descendants_including_hidden_ones():
    viewer = _load_viewer()
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="descendants")
    assert keys == ["item10", "item2", "Alpha", "zeta", "beta"]     # beta has 2 below; ties stay put
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="descendants", desc=True)
    assert keys == ["beta", "item10", "item2", "Alpha", "zeta"]
    # hiding beta's child box does not change the total
    g = viewer.build_graph(_arrange_data())
    beta = next(r.child for n in g.nodes for r in n.rows if r.key == "beta")
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="descendants",
                             unchecked=[(beta.id, 0)])
    assert keys[-1] == "beta"


def test_rows_sort_by_visible_descendants_which_follow_the_filters():
    viewer = _load_viewer()
    g = viewer.build_graph(_arrange_data())
    beta = next(r.child for n in g.nodes for r in n.rows if r.key == "beta")
    _g, _v, everything = _app_rows(viewer, _arrange_data(), sort_by="visible", desc=True)
    assert everything[0] == "beta"                                  # 2 visible descendants
    # with beta's "a" box hidden (its checkbox), beta has none left: back to the original order
    _g, _v, keys = _app_rows(viewer, _arrange_data(), sort_by="visible", desc=True,
                             unchecked=[(beta.id, 0)])
    assert keys == ["item10", "item2", "Alpha", "zeta", "beta"]
    g2, view, _ = _app_rows(viewer, _arrange_data(), sort_by="visible", unchecked=[(beta.id, 0)])
    assert viewer.descendant_counts(view.nodes)[next(
        n.id for n in view.nodes if n.id == beta.id)] == 0
    assert g2.root.desc_all >= 5                                    # the full count is unchanged


def test_the_note_about_more_entries_always_stays_last_and_edges_follow_the_moved_rows():
    viewer = _load_viewer()
    many = {f"k{i}": {"v": i} for i in range(viewer.MAX_ROWS + 5)}
    g = viewer.build_graph({"meta": {}, "app": many, "module": {}, "threads": []})
    view = viewer.apply_filters(g, {})
    viewer.arrange_rows(view, True, "name", True)
    app = next(n for n in view.nodes if len(n.rows) == viewer.MAX_ROWS + 1)
    assert app.rows[-1].key == "\u2026" and app.rows[0].key == f"k{viewer.MAX_ROWS - 1}"
    for node, i, child, _tree in view.edges:
        assert node.rows[i].child is child
    assert len(view.edges) == sum(1 for n in view.nodes for r in n.rows if r.child is not None)
    viewer.layout_tree(view)
    assert viewer.overlapping_pairs(view) == [] and viewer.count_crossings(view) == 0
    ys = [c.y for c in app.tree_children]
    assert ys == sorted(ys)                                         # stacked in the new row order
    assert [c.parent_row for c in app.tree_children] == sorted(c.parent_row
                                                               for c in app.tree_children)


def test_hide_rows_that_are_not_links_counts_them_in_the_hidden_message():
    viewer = _load_viewer()
    g = viewer.build_graph(_filter_sample())
    view = viewer.apply_filters(g, {}, set(), True)
    app = next(n for n in view.nodes if n.title == "dict (5)")
    assert [r.key for r in app.rows] == ["items", "obst"]           # only the link rows
    assert app.hidden == 3                                          # count, name, empty
    assert viewer.hidden_message(app.hidden) == "3 children hidden by filters"
    assert all(r.child is not None for n in view.nodes for r in n.rows)   # every row left is a link
    # boxes with no link rows at all show just the message
    leaf = next(n for n in view.nodes if n.rows == [] and n.hidden)
    assert leaf.hidden >= 1
    # rows hidden by a checkbox or by a filter level are not links either
    items = next(r for r in next(n for n in viewer.apply_filters(g, {}).nodes
                                 if n.title == "dict (5)").rows if r.key == "items")
    cleared = viewer.apply_filters(g, {}, {items.origin}, True)
    app2 = next(n for n in cleared.nodes if n.title == "dict (5)")
    assert [r.key for r in app2.rows] == ["obst"] and app2.hidden == 4
    row_only = viewer.apply_filters(g, {"list": 2}, set(), True)
    app3 = next(n for n in row_only.nodes if n.title == "dict (5)")
    assert [r.key for r in app3.rows] == ["obst"] and app3.hidden == 4
    off = viewer.apply_filters(g, {}, set(), False)
    assert next(n for n in off.nodes if n.title == "dict (5)").hidden == 0


def test_the_view_options_reach_the_view_and_show_all_clears_the_links_only_filter():
    viewer = _load_viewer()
    v = _viewer_with_graph(viewer, _arrange_data())
    v.links_only_var.set(True)
    v.group_links_var.set(True)
    v.sort_by_var.set("name")
    v.rebuild_view()
    app = next(n for n in v.vg.nodes if n.title == "dict (5)")
    assert [r.key for r in app.rows] == ["Alpha", "beta", "zeta"] and app.hidden == 2
    v.level_vars = {}
    v.reset_filters()
    assert v.links_only_var.get() is False
    app = next(n for n in v.vg.nodes if n.title == "dict (5)")
    assert app.hidden == 0 and len(app.rows) == 5


def _wheel_stub(viewer):
    import types
    v = viewer.Viewer.__new__(viewer.Viewer)
    v.vg = object()
    v.zoom, v.ox, v.oy = 1.0, 500.0, 300.0
    v.drawn = (0, 0, 10 ** 6, 10 ** 6)
    v.log = []
    v.canvas = types.SimpleNamespace(move=lambda tag, dx, dy: v.log.append(("move", dx, dy)))
    v.canvas_size = lambda: (1000, 600)
    v.redraw = lambda: v.log.append("redraw")
    v.schedule_redraw = lambda: v.log.append("schedule")
    return v


def test_the_plain_wheel_scrolls_up_and_down_and_shift_scrolls_sideways():
    import types
    viewer = _load_viewer()
    v = _wheel_stub(viewer)
    step = viewer.WHEEL_SCROLL_PX
    v.on_wheel(types.SimpleNamespace(x=10, y=10, delta=120, state=0))          # wheel up
    assert v.log[-1] == ("move", 0, step) and v.oy == 300.0 - step and v.ox == 500.0
    v.on_wheel(types.SimpleNamespace(x=10, y=10, delta=-120, state=0))         # wheel down
    assert v.log[-1] == ("move", 0, -step) and v.oy == 300.0
    v.on_wheel(types.SimpleNamespace(x=10, y=10, delta=120, state=0x1))        # Shift, wheel up
    assert v.log[-1] == ("move", step, 0) and v.ox == 500.0 - step and v.oy == 300.0
    v.on_wheel(types.SimpleNamespace(x=10, y=10, delta=-120, state=0x1))       # Shift, wheel down
    assert v.log[-1] == ("move", -step, 0) and v.ox == 500.0
    assert v.zoom == 1.0                                                        # neither zooms


def test_the_scroll_distance_is_in_window_pixels_whatever_the_zoom():
    import types
    viewer = _load_viewer()
    v = _wheel_stub(viewer)
    v.zoom = 4.0
    v.on_wheel(types.SimpleNamespace(x=0, y=0, delta=120, state=0x1))
    assert v.ox == 500.0 - viewer.WHEEL_SCROLL_PX / 4.0
    v.on_wheel(types.SimpleNamespace(x=0, y=0, delta=120, state=0))
    assert v.oy == 300.0 - viewer.WHEEL_SCROLL_PX / 4.0


def test_ctrl_wheel_zooms_at_the_pointer_and_scrolling_redraws_at_the_edge_of_what_is_drawn():
    import types
    viewer = _load_viewer()
    v = _wheel_stub(viewer)
    v.on_wheel(types.SimpleNamespace(x=200, y=100, delta=120, state=0x4))
    assert v.zoom == 1.2 and v.log[-1] == "schedule" and (v.ox, v.oy) != (500.0, 300.0)
    v.on_wheel(types.SimpleNamespace(x=200, y=100, delta=-120, state=0x4))
    assert abs(v.zoom - 1.0) < 1e-12
    v.on_wheel(types.SimpleNamespace(x=200, y=100, delta=120, state=0x1 | 0x4))   # Ctrl wins: zoom
    assert v.zoom == 1.2
    v.zoom = 1.0
    v.drawn = (v.ox, v.oy, v.ox + 1000, v.oy + 600)          # the drawn region is just the window
    v.log.clear()
    v.on_wheel(types.SimpleNamespace(x=0, y=0, delta=120, state=0))
    assert v.log[-1] == "redraw"                             # scrolled out of the drawn region
    v.vg = None
    v.log.clear()
    v.scroll_by(5, 5)
    assert v.log == []                                       # nothing loaded: nothing to move


def test_the_x11_wheel_buttons_use_the_same_rules():
    import types
    viewer = _load_viewer()
    v = _wheel_stub(viewer)
    v.wheel(10, 10, 0, 1)
    assert v.log[-1] == ("move", 0, viewer.WHEEL_SCROLL_PX)
    v.wheel(10, 10, 0x1, -1)
    assert v.log[-1] == ("move", -viewer.WHEEL_SCROLL_PX, 0)
    v.wheel(10, 10, 0x4, 1)
    assert v.zoom == 1.2


def test_the_view_menu_and_sidebar_offer_the_new_options():
    import inspect
    viewer = _load_viewer()
    init = inspect.getsource(viewer.Viewer.__init__)
    assert 'label="Group link rows first"' in init
    assert 'label="Sort objects within their parent"' in init
    assert "for value, label in SORT_CHOICES:" in init
    assert 'label="Ascending"' in init and 'label="Descending"' in init
    labels = [label for _value, label in viewer.SORT_CHOICES]
    assert labels == ["None (original order)", "Name (alphabetical)", "Size in bytes",
                      "Total descendants (including hidden ones)", "Visible descendants"]
    assert "Hide rows that aren't links" in inspect.getsource(viewer.Viewer.build_sidebar)
