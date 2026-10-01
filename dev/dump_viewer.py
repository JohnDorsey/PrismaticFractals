"""
Dump viewer (development aid): explore a .pkl written by debug_dump.py (or most
any pickle of nested dicts, lists and objects) as a tree of boxes joined by arrows.

Run:   python dump_viewer.py [dump.pkl]
       (with no argument it offers the newest file in dev/dumps)

Mostly standalone: it needs only Python's standard library (tkinter) and, if
installed, numpy (to read arrays). It does NOT import light_bounces.py, so it
starts instantly and works without numba; classes that belong to the
application are read back as placeholder objects showing their attributes.

What you see
  Every container (dict, list, tuple, set) and object is a box with one row per
  entry. A row whose value is a plain value (number, string, array summary, ...)
  shows it inline; a row whose value is another container or object ends in an
  arrow to that object's own box. Stand-ins written by debug_dump.py (Tk
  variables, widgets, ...) are boxes too (purple). An object referenced from
  several places is one box with several arrows (the extra ones red, dashed
  unless very long).
  Hover a row to read its full text in the panel at the bottom.

Controls
  scroll wheel            scroll up / down
  Shift + scroll wheel    scroll sideways
  Ctrl + scroll wheel     zoom in / out towards the mouse pointer
  right-drag              pan
  left-drag an object     move it (all selected objects move together)
  left-drag on background select the objects the rectangle touches
                          (hold Ctrl to add to the selection)
  left-click background   deselect
  click a link row        a row that ends in an arrow is a link (underlined, hand
                          cursor): click it to jump to the box it points at
  its checkbox            (at the right of such a row) shows or hides that box;
                          a box is visible only if its checkbox is ticked AND the
                          filters below allow it
  click a box's name      (zoomed in far enough, its title has two lines and the first
                          is the name it has inside its parent) jump to its row there
  F   fit everything      1   actual size, centred on the top box
  Esc deselect            Ctrl+O   open another dump   R   rearrange
  mouse back / forward    (the side buttons), Alt+Left / Alt+Right, or View >
                          Navigate back / forward: return to the place you were
                          before you last clicked a link, or go on again (a place
                          is where the view was; only link clicks - rows and box
                          names - are remembered, not View > Go to)
  View > Go to            jump to the top box or to any of its children
  View > Connections      swoopy curves or straight lines

Arrangement (top of the left sidebar): "Rearrange" lays the visible boxes out
again using two settings -
  minimum horizontal spacing   the least room between a box and its children
  spacing aspect ratio         the room between a box and its children is at least
                               this multiple of the vertical space all those
                               children take up (0 = just the minimum), so a box
                               with a tall fan of children gets them further away
                               and its arrows come out less steep

View > Group link rows first   in every box, rows that point at a visible box come first
View > Sort objects            within their parent, order the rows (and so the stacked
                               boxes after a rearrange) by name, size in bytes (an
                               estimate of the memory the value and everything under it
                               takes), or total number of descendant boxes - counting all
                               of them, or only the visible ones - ascending or descending

Filters (left sidebar, its two lists can be folded): for every type of object in the
file, one of five levels
  4  show these objects and their children (each child still follows its own level)
  3  show these objects but hide their children
  2  hide these objects, but let each appear as a row inside its parent
  1  hide these objects, and their rows inside their parents too
  0  as 1, and also hide the parent of such an object if all of that parent's
     children are at 0 (the parent still appears as a row inside its own parent)
A box that has children hidden by a filter says how many at its bottom.
"Hide rows that aren't links" also hides, in every box, each row that is not a
clickable link (plain values, rows whose box is hidden or unticked); they are counted
in that message too.
Only rows that point at a box have a checkbox: hiding a plain value's row would
leave nothing to tick again. "Show all" also ticks every checkbox again.
Hiding and showing leaves everything where it is; "Rearrange" lays out the
visible boxes again so that hidden ones take up no room.

The starting layout is a left-to-right tree: boxes are placed so that none
overlap, and so that the arrows of the tree itself cross nothing (each box's
children are stacked in the order of the rows that point at them).

Speed: only the boxes near the window are on the canvas. Panning moves those few
items; zooming redraws them, skipping the text when it would be too small to
read and each arrow that would be shorter than a few pixels (so zoomed far out
only the long arrows are drawn).
"""
import os
import pickle
import re
import sys
import time
import types

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

try:
    import numpy as np
except Exception:                                   # noqa: BLE001
    np = None

# ---- geometry, in world units (pixels at zoom 1) -------------------------------------
ROW_H = 16
TITLE_H = 34                 # room for two lines of title (used when zoomed in far)
FOOTER_H = 18
CHECK_SIZE = 10              # the checkbox on a row that points at a box
CHECK_W = 22                 # extra box width it takes
LINK_CLICK_GUARD_S = 0.6     # a quick second click (a double-click) does not jump again
HISTORY_LIMIT = 200          # places remembered for navigating back and forward
WHEEL_SCROLL_PX = 120        # how far one wheel click scrolls (plain: up/down, Shift: sideways)
SORT_NONE, SORT_NAME, SORT_SIZE, SORT_DESC_ALL, SORT_DESC_VISIBLE = (
    "none", "name", "size", "descendants", "visible")
SORT_CHOICES = [(SORT_NONE, "None (original order)"), (SORT_NAME, "Name (alphabetical)"),
                (SORT_SIZE, "Size in bytes"),
                (SORT_DESC_ALL, "Total descendants (including hidden ones)"),
                (SORT_DESC_VISIBLE, "Visible descendants")]
PAD = 8
DEFAULT_MIN_GAP = 110        # minimum horizontal room between a box and its children
MIN_GAP_RANGE = (0, 600)
ASPECT_RANGE = (0.0, 1.0)
ROW_GAP = 18
REF_FONT_PX = 12
MAX_ROWS = 400               # rows shown per box; the rest summarized in one row
MAX_KEY = 30
MAX_VAL = 46
HIDE_TEXT_BELOW_PX = 5
STRAIGHT_BELOW_ZOOM = 0.05   # curves cost more to draw; below this everything is straight
MIN_ARROW_PX = 12            # an arrow shorter than this on the screen is not drawn, so zoomed
                             # far out only the longer ones remain (and cost nothing to draw)
MIN_ZOOM, MAX_ZOOM = 0.004, 12.0
START_ZOOM = 0.8
DRAW_MARGIN = 0.3            # extra drawn around the window, as a fraction of its size
# Filter levels (see the module docstring); every type starts at SHOW.
SHOW, NO_CHILDREN, ROW_ONLY, HIDDEN, LOWEST = 4, 3, 2, 1, 0
LEVELS = (SHOW, NO_CHILDREN, ROW_ONLY, HIDDEN, LOWEST)
TWO_LINE_TITLE_ZOOM = 1.5    # zoomed in this far, a title is two lines: name, then type

FILL = {"root": "#ffffff", "dict": "#dbe7ff", "list": "#dcf3d6", "tuple": "#e6f2cf",
        "set": "#d6f0f0", "object": "#ffe6c2", "standin": "#ecd5f4"}
TITLE_FILL = {"root": "#e0e0e0", "dict": "#b9cffb", "list": "#b5e3ab", "tuple": "#cfe5a0",
              "set": "#a9e0e0", "object": "#ffd08a", "standin": "#d9b0e8"}
SELECTED_OUTLINE = "#ff8f00"
EDGE_COLOR = "#4a4a4a"
EXTRA_EDGE_COLOR = "#c0392b"
EDGE_WIDTH = 2
DASH_MAX_PX = 1500           # extra arrows longer than this (on screen) are solid: a dashed
                             # wide line tens of thousands of pixels long paints very slowly
ARROW_SHAPE = (11, 13, 4)
FOOTER_COLOR = "#a04000"


# ===================================================================== loading
class Placeholder:
    """Stands in for any class that cannot (or should not) be imported when
    reading a dump: keeps whatever the pickle puts into it."""
    _qual = "?"

    def __new__(cls, *args, **kwargs):
        # pickle builds a namedtuple as cls.__new__(cls, *values), without calling
        # __init__, so the values have to be captured here.
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state

    def append(self, item):
        self.__dict__.setdefault("items", []).append(item)

    def extend(self, items):
        self.__dict__.setdefault("items", []).extend(items)

    def __setitem__(self, key, value):
        self.__dict__.setdefault("pairs", []).append((key, value))


_placeholder_classes = {}


def placeholder_class(module, name):
    key = (module, name)
    if key not in _placeholder_classes:
        _placeholder_classes[key] = type(name, (Placeholder,), {
            "__module__": module, "_qual": f"{module}.{name}"})
    return _placeholder_classes[key]


# Application modules are never imported here (that would pull in numba).
APPLICATION_MODULES = ("__main__", "light_bounces", "lightbounces", "debug_dump", "test_light_bounces", "tests",
                       "test_harness")


class TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in APPLICATION_MODULES:
            return placeholder_class(module, name)
        try:
            return super().find_class(module, name)
        except Exception:                           # noqa: BLE001
            return placeholder_class(module, name)


def load_pickle(path):
    with open(path, "rb") as f:
        return TolerantUnpickler(f).load()


# ===================================================================== the graph
class Row:
    __slots__ = ("key", "text", "full", "child", "type_key", "origin", "check", "size",
                 "desc_all")

    def __init__(self, key, text, full, child=None, type_key=None, origin=None, check=None,
                 size=0, desc_all=0):
        self.key, self.text, self.full, self.child = key, text, full, child
        self.type_key = type_key        # what the row's value is (None: not a value)
        self.origin = origin            # (id of the original box, index of the original row)
        self.check = check              # None: no checkbox; else whether it is ticked
        self.size = size                # estimated bytes of the value (and all under it)
        self.desc_all = desc_all        # boxes under the value (0 for a plain value)


class Node:
    def __init__(self, nid, kind, title, type_key=None):
        self.id = nid
        self.kind = kind
        self.title = title
        self.type_key = type_key        # the type the filters act on
        self.rows = []
        self.hidden = 0                 # entries left out by filters (view nodes only)
        self.width = self.height = 0.0
        self.x = self.y = 0.0
        self.depth = 0
        self.parent = None              # the tree parent (first to reach this node)
        self.tree_children = []
        self.parent_row = -1            # the row of the tree parent that points at this box
        self.span_h = 0.0               # vertical space all its tree children take up
        self.size = 0                   # estimated bytes of the value (and all under it)
        self.desc_all = 0               # boxes under this one, hidden ones included


def _short(text, limit):
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    return text if len(text) <= limit else text[:limit - 1] + "\u2026"


def is_ndarray(value):
    return (np is not None and isinstance(value, np.ndarray)) or \
        (type(value).__name__ == "ndarray" and type(value).__module__.startswith("numpy"))


def _size_text(n):
    units = ["B", "KB", "MB", "GB"]
    n, i = float(n), 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.0f} B" if i == 0 else f"{n:.1f} {units[i]}"


def inline_text(value):
    """Short text for a value that is shown inside its row, or None if the value
    needs a box of its own."""
    if value is None or isinstance(value, (bool, int, float, complex)):
        return repr(value)
    if isinstance(value, (str, bytes)):
        return _short(repr(value), MAX_VAL)
    if is_ndarray(value):
        return f"ndarray {tuple(value.shape)} {value.dtype} ({_size_text(value.nbytes)})"
    if isinstance(value, type):
        return f"<class {value.__module__}.{value.__qualname__}>"
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        return f"<function {getattr(value, '__qualname__', value)}>"
    if isinstance(value, dict):
        if value.get("__standin__") == "large array":
            return (f"<large array {tuple(value.get('shape', ()))} {value.get('dtype')} "
                    f"({_size_text(value.get('nbytes', 0))})>")
        return "{}" if not value else None
    if isinstance(value, list):
        return "[]" if not value else None
    if isinstance(value, tuple):
        if not value:
            return "()"
        if len(value) <= 6 and all(v is None or isinstance(v, (bool, int, float, str))
                                   for v in value):
            return _short(repr(value), MAX_VAL)
        return None
    if isinstance(value, (set, frozenset)):
        return "set()" if not value else None
    return None


def shallow_size(value):
    """Estimated bytes one value takes itself (arrays: their data; a large-array stand-in:
    the size it records)."""
    if is_ndarray(value):
        return int(value.nbytes)
    if isinstance(value, dict) and value.get("__standin__") == "large array":
        return int(value.get("nbytes", 0) or 0)
    try:
        return sys.getsizeof(value)
    except Exception:                               # noqa: BLE001
        return 0


def deep_size(value, memo, active=None):
    """Estimated bytes of a value and everything it holds. `memo` (id -> size) makes each
    shared value be measured once; a value met again while it is being measured
    (a cycle) counts as 0 there."""
    key = id(value)
    if key in memo:
        return memo[key]
    active = set() if active is None else active
    if key in active:
        return 0
    if isinstance(value, (str, bytes, int, float, bool, complex, type(None))) or is_ndarray(value):
        return shallow_size(value)
    active.add(key)
    total = shallow_size(value)
    if isinstance(value, dict):
        if value.get("__standin__") != "large array":
            for k, v in value.items():
                total += shallow_size(k) + deep_size(v, memo, active)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            total += deep_size(v, memo, active)
    elif not isinstance(value, type):
        for _name, v in object_attributes(value):
            total += deep_size(v, memo, active)
    active.discard(key)
    memo[key] = total
    return total


def object_attributes(value):
    """[(name, value)] for an object that is neither a container nor a plain value."""
    if isinstance(value, Placeholder):
        attrs = []
        state = getattr(value, "state", None)
        if isinstance(state, dict):
            attrs += list(state.items())
        elif isinstance(state, tuple) and len(state) == 2 and \
                (state[0] is None or isinstance(state[0], dict)) and isinstance(state[1], dict):
            attrs += list((state[0] or {}).items()) + list(state[1].items())
        elif state is not None:
            attrs.append(("state", state))
        if getattr(value, "args", ()):
            attrs += [(f"arg {i}", a) for i, a in enumerate(value.args)]
        if getattr(value, "items", None):
            attrs += [(f"[{i}]", a) for i, a in enumerate(value.items)]
        if getattr(value, "pairs", None):
            attrs += [(repr(k), v) for k, v in value.pairs]
        return attrs
    attrs = []
    if hasattr(value, "__dict__"):
        attrs += list(vars(value).items())
    for cls in type(value).__mro__:
        for slot in getattr(cls, "__slots__", ()):
            if isinstance(slot, str) and hasattr(value, slot):
                attrs.append((slot, getattr(value, slot)))
    return attrs


def _object_title(value):
    if isinstance(value, Placeholder):
        return value._qual
    t = type(value)
    return f"{t.__module__}.{t.__qualname__}"


def type_key_of(value):
    """The name of the type a value is filtered as (dict, list, tuple, set, int, str,
    ndarray, ..., stand-in: <kind>, or an object's class name)."""
    if value is None:
        return "NoneType"
    if isinstance(value, dict):
        kind = value.get("__standin__")
        return f"stand-in: {kind}" if isinstance(kind, str) else "dict"
    if isinstance(value, (bool, int, float, complex, str, bytes, list, tuple, set)):
        return type(value).__name__
    if isinstance(value, frozenset):
        return "set"
    if is_ndarray(value):
        return "ndarray"
    if isinstance(value, type):
        return "class"
    if isinstance(value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        return "function"
    return _object_title(value)


class Graph:
    def __init__(self):
        self.nodes = []
        self.by_id = {}                  # id(python object) -> Node
        self.edges = []                  # (parent node, row index, child node, is_tree_edge)
        self.keepalive = []
        self.size_memo = {}
        self.root = None

    def new_node(self, kind, title, type_key=None):
        node = Node(len(self.nodes), kind, title, type_key)
        self.nodes.append(node)
        return node

    def node_for(self, value):
        """The box for a container or object (made once per Python object), or
        None if the value is shown inline."""
        if inline_text(value) is not None:
            return None
        key = id(value)
        if key in self.by_id:
            return self.by_id[key]
        self.keepalive.append(value)
        type_key = type_key_of(value)
        if isinstance(value, dict):
            standin = value.get("__standin__") if isinstance(value.get("__standin__"), str) else None
            if standin is not None:
                detail = value.get("name") or value.get("type") or value.get("path") or ""
                node = self.new_node("standin", _short(f"{standin}: {detail}".rstrip(": "), 60),
                                     type_key)
                items = [(k, v) for k, v in value.items() if k != "__standin__"]
            else:
                node = self.new_node("dict", f"dict ({len(value)})", type_key)
                items = list(value.items())
            entries = [(k if isinstance(k, str) else repr(k), v) for k, v in items]
        elif isinstance(value, (list, tuple, set, frozenset)):
            kind = "list" if isinstance(value, list) else "tuple" if isinstance(value, tuple) \
                else "set"
            node = self.new_node(kind, f"{type(value).__name__} ({len(value)})", type_key)
            entries = [(f"[{i}]", v) for i, v in enumerate(value)]
        else:
            attrs = object_attributes(value)
            node = self.new_node("object", f"{_object_title(value)} ({len(attrs)})", type_key)
            entries = [(str(k), v) for k, v in attrs]
        self.by_id[key] = node
        self.fill_rows(node, entries)
        node.size = deep_size(value, self.size_memo)
        return node

    def fill_rows(self, node, entries):
        shown = entries[:MAX_ROWS]
        for name, value in shown:
            text = inline_text(value)
            if text is not None:
                full = f"{name} = {self.full_text(value)}"
                node.rows.append(Row(_short(str(name), MAX_KEY), text, full, None,
                                     type_key_of(value), size=deep_size(value, self.size_memo)))
                continue
            child = self.node_for(value)
            node.rows.append(Row(_short(str(name), MAX_KEY), child.title,
                                 f"{name} -> {child.title}", child, child.type_key,
                                 size=child.size))
            self.edges.append((node, len(node.rows) - 1, child, None))
        if len(entries) > MAX_ROWS:
            node.rows.append(Row("\u2026", f"{len(entries) - MAX_ROWS} more entries not shown",
                                 f"{len(entries) - MAX_ROWS} more entries not shown"))

    @staticmethod
    def full_text(value):
        try:
            text = repr(value)
        except Exception as exc:                    # noqa: BLE001
            text = f"<repr failed: {exc!r}>"
        return text if len(text) <= 4000 else text[:4000] + "..."


def build_graph(data):
    """The graph of boxes for a loaded dump (a debug_dump dict, or any object)."""
    graph = Graph()
    if isinstance(data, dict) and {"meta", "app", "module"} <= set(data):
        root = graph.new_node("root", "dump", None)
        entries = [(k, data[k]) for k in ("meta", "app", "module", "threads") if k in data]
        entries += [(k, v) for k, v in data.items()
                    if k not in ("meta", "app", "module", "threads")]
        graph.fill_rows(root, entries)
    else:
        root = graph.node_for(data)
        if root is None:
            root = graph.new_node("root", "value", None)
            root.rows.append(Row("value", inline_text(data), Graph.full_text(data), None,
                                 type_key_of(data)))
    graph.root = root
    if root.size == 0:
        root.size = deep_size(data, graph.size_memo)
    counts = descendant_counts(graph.nodes)
    for node in graph.nodes:
        node.desc_all = counts[node.id]
    for node in graph.nodes:
        for row in node.rows:
            if row.child is not None:
                row.desc_all = row.child.desc_all
    return graph


def descendant_counts(nodes):
    """{box id: how many different boxes can be reached from it through its rows}, for any
    list of boxes (the whole graph, or just the visible ones). Boxes that point at each
    other (a cycle) are handled: they all reach each other. Uses reachability sets kept
    as integers, combined in the order Tarjan's algorithm finishes strongly connected
    groups, so it stays fast for a few thousand boxes."""
    index = {n.id: i for i, n in enumerate(nodes)}
    succ = [[index[r.child.id] for r in n.rows if r.child is not None and r.child.id in index]
            for n in nodes]
    count = len(nodes)
    order, low, on_stack, stack = [-1] * count, [0] * count, [False] * count, []
    reach = [0] * count
    counter = 0
    for start in range(count):
        if order[start] != -1:
            continue
        work = [(start, iter(succ[start]))]
        order[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack[start] = True
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if order[nxt] == -1:
                    order[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack[nxt] = True
                    work.append((nxt, iter(succ[nxt])))
                    advanced = True
                    break
                if on_stack[nxt]:
                    low[node] = min(low[node], order[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == order[node]:
                group = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    group.append(member)
                    if member == node:
                        break
                mask = 0
                cyclic = len(group) > 1 or node in succ[node]
                members = set(group)
                for m in group:
                    if cyclic:
                        mask |= 1 << m
                    for c in succ[m]:
                        if c not in members:
                            mask |= (1 << c) | reach[c]
                for m in group:
                    reach[m] = mask
    return {nodes[i].id: reach[i].bit_count() - ((reach[i] >> i) & 1) for i in range(count)}


# ===================================================================== filters
def type_inventory(graph):
    """(box types, plain value types): each a list of (type name, count) sorted by
    count then name. Box types count boxes; plain value types count the rows that
    show one. A type is listed once: a type that has boxes (say dict) is a box type
    even though its empty ones show inline, and its one level covers both."""
    boxes, leaves = {}, {}
    for node in graph.nodes:
        if node is not graph.root and node.type_key is not None:
            boxes[node.type_key] = boxes.get(node.type_key, 0) + 1
        for row in node.rows:
            if row.child is None and row.type_key is not None:
                leaves[row.type_key] = leaves.get(row.type_key, 0) + 1

    def ordered(counts):
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered(boxes), ordered({k: n for k, n in leaves.items() if k not in boxes})


class ViewGraph:
    """The boxes that are visible under a set of filter levels (new Node objects with
    the same ids as the originals, holding only the rows that are shown)."""
    def __init__(self):
        self.nodes = []
        self.by_id = {}
        self.edges = []
        self.root = None


def row_level(row, levels):
    """The filter level of the value a row shows (rows that are not values: SHOW)."""
    return SHOW if row.type_key is None else levels.get(row.type_key, SHOW)


def hides_itself(node, levels):
    """Does a box disappear because everything in it is at the LOWEST level? (It needs
    at least one entry; an entry that is not a value counts as shown.)"""
    return bool(node.rows) and all(row_level(r, levels) == LOWEST for r in node.rows)


def apply_filters(graph, levels, unchecked=frozenset(), links_only=False):
    """Which boxes and rows are visible. `levels` maps a type name to 4, 3, 2, 1 or 0
    (missing: 4); see the module docstring. `unchecked` holds (box id, row index)
    origins whose checkbox is cleared: such a row stays (plain, with an empty
    checkbox) but its box is not shown on its account. A row that points at a box
    the filters show gets a ticked checkbox. The top box is always shown and always
    shows its children."""
    view = ViewGraph()

    def make(orig):
        node = Node(orig.id, orig.kind, orig.title, orig.type_key)
        view.nodes.append(node)
        view.by_id[orig.id] = node
        return node
    view.root = make(graph.root)
    queue = [(graph.root, view.root)]
    while queue:
        orig, node = queue.pop(0)
        if orig is not graph.root and levels.get(orig.type_key, SHOW) == NO_CHILDREN:
            node.hidden = len(orig.rows)             # its children are hidden
            continue
        for i, row in enumerate(orig.rows):
            level = row_level(row, levels)
            if level <= HIDDEN:
                node.hidden += 1
                continue
            origin = (orig.id, i)
            if (row.child is not None and level >= NO_CHILDREN
                    and not hides_itself(row.child, levels)):
                if origin not in unchecked:
                    child = view.by_id.get(row.child.id)
                    if child is None:
                        child = make(row.child)
                        queue.append((row.child, child))
                    node.rows.append(Row(row.key, row.text, row.full, child, row.type_key,
                                         origin, True, row.size, row.desc_all))
                    view.edges.append((node, len(node.rows) - 1, child, None))
                    continue
                check = False                        # cleared: a plain row, no box
            else:
                check = None                         # a plain row (or naming a hidden box)
            if links_only:
                node.hidden += 1
                continue
            node.rows.append(Row(row.key, row.text, row.full, None, row.type_key, origin,
                                 check, row.size, row.desc_all))
    return view


def natural_key(text):
    """Case-insensitive sort key that orders digit runs as numbers ([2] before [10])."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def arrange_rows(view, group_links=False, sort_by=SORT_NONE, descending=False):
    """Reorder the rows of every box of a ViewGraph: with `group_links`, rows that link to
    a box come first; then, within each group, by `sort_by` (name, size in bytes, total
    descendants, or visible descendants), ascending or `descending`; ties keep their
    order. A row that is not a value (the 'more entries' note) always stays last. The
    view's edges are rebuilt to follow the moved rows."""
    if not group_links and sort_by == SORT_NONE:
        return view
    visible = descendant_counts(view.nodes) if sort_by == SORT_DESC_VISIBLE else None

    def value_key(row):
        if sort_by == SORT_NAME:
            return natural_key(row.key)
        if sort_by == SORT_SIZE:
            return row.size
        if sort_by == SORT_DESC_ALL:
            return row.desc_all
        if sort_by == SORT_DESC_VISIBLE:
            return visible[row.child.id] if row.child is not None else 0
        return 0
    for node in view.nodes:
        if sort_by != SORT_NONE:
            node.rows.sort(key=value_key, reverse=descending)
        node.rows.sort(key=lambda r: (r.type_key is None and r.child is None,
                                      group_links and r.child is None))
    view.edges = [(node, i, row.child, None) for node in view.nodes
                  for i, row in enumerate(node.rows) if row.child is not None]
    return view


def name_in_parent(node):
    """The name a box has inside its tree parent (the key of the row pointing at it)."""
    if node.parent is None:
        return "top box"
    if 0 <= node.parent_row < len(node.parent.rows):
        return node.parent.rows[node.parent_row].key
    return "?"


def hidden_message(count):
    return f"{count} {'child' if count == 1 else 'children'} hidden by filters"


# ===================================================================== layout
def measure_nodes(graph, char_w):
    for node in graph.nodes:
        widest = len(node.title) + 1
        for row in node.rows:
            widest = max(widest, len(row.key) + 2 + len(row.text))
        if node.hidden:
            widest = max(widest, len(hidden_message(node.hidden)) + 1)
        node.width = widest * char_w + 2 * PAD
        if any(r.check is not None for r in node.rows):
            node.width += CHECK_W
        node.height = (TITLE_H + len(node.rows) * ROW_H + (FOOTER_H if node.hidden else 0) + 6)


def span_tree(graph):
    """Give every reachable box its depth, tree parent and tree children (the first
    row that points at a box makes that arrow the tree's), and flag each edge as a
    tree edge or an extra one (to a box reached another way)."""
    root = graph.root
    for node in graph.nodes:
        node.parent, node.tree_children, node.depth = None, [], 0
        node.parent_row = -1
        node.span_h = 0.0
    seen = {root.id}
    order = [root]
    stack = [(root, iter(range(len(root.rows))))]
    while stack:
        node, rows = stack[-1]
        for i in rows:
            child = node.rows[i].child
            if child is not None and child.id not in seen:
                seen.add(child.id)
                child.parent = node
                child.parent_row = i
                child.depth = node.depth + 1
                node.tree_children.append(child)
                order.append(child)
                stack.append((child, iter(range(len(child.rows)))))
                break
        else:
            stack.pop()
    first, flagged = set(), []
    for p, i, c, _ in graph.edges:
        is_tree = c.parent is p and (p.id, c.id) not in first
        if is_tree:
            first.add((p.id, c.id))
        flagged.append((p, i, c, is_tree))
    graph.edges = flagged
    return order


def place_tree(graph, order, min_gap=DEFAULT_MIN_GAP, aspect=0.0):
    """Assign every box x, y (world units). Each box's children are stacked in row
    order in a block beside it, blocks never sharing a vertical range - so nothing
    overlaps and the tree's own arrows do not cross. The children start to the right
    of their parent by max(min_gap, aspect * the vertical space all the children take
    up), so a tall fan of children is kept further away (and its arrows less steep)."""
    root = graph.root
    block = {}
    for node in reversed(order):
        total = sum(block[c.id] for c in node.tree_children)
        total += ROW_GAP * max(0, len(node.tree_children) - 1)
        node.span_h = total
        block[node.id] = max(node.height, total)
    tops = {root.id: 0.0}
    root.x = 0.0
    for node in order:
        top = tops[node.id]
        node.y = top + (block[node.id] - node.height) / 2
        child_x = node.x + node.width + max(min_gap, aspect * node.span_h)
        child_top = top + (block[node.id] - node.span_h) / 2
        for child in node.tree_children:
            child.x = child_x
            tops[child.id] = child_top
            child_top += block[child.id] + ROW_GAP


def layout_tree(graph, char_w=7.0, min_gap=DEFAULT_MIN_GAP, aspect=0.0):
    """Measure and place every box of a Graph or ViewGraph (see place_tree)."""
    measure_nodes(graph, char_w)
    place_tree(graph, span_tree(graph), min_gap, aspect)
    return graph


def overlapping_pairs(graph):
    """Pairs of boxes whose rectangles overlap (should be none after layout_tree)."""
    boxes = sorted(graph.nodes, key=lambda n: n.x)
    bad = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if b.x >= a.x + a.width:
                break
            if a.y < b.y + b.height and b.y < a.y + a.height:
                bad.append((a, b))
    return bad


def edge_points(parent, row, child):
    """Straight-line endpoints of an arrow in world units: from the right edge of the
    parent's row to the middle of the child's title bar."""
    return (parent.x + parent.width, parent.y + TITLE_H + row * ROW_H + ROW_H / 2,
            child.x, child.y + TITLE_H / 2)


def arrow_visible(x0, y0, x1, y1, zoom):
    """Is an arrow (end points in world units) long enough on the screen at `zoom` to be
    worth drawing? The longer it is, the further out it stays visible."""
    return (abs(x1 - x0) + abs(y1 - y0)) * zoom >= MIN_ARROW_PX


def edge_path(x0, y0, x1, y1, zoom, straight):
    """Canvas coordinates of a connection: two points, or a swoopy curve - a cubic
    Bezier that leaves and enters horizontally, drawn as many short segments so it
    is smooth all along, ends included."""
    if straight:
        return (x0, y0, x1, y1)
    reach = max(40 * zoom, abs(x1 - x0) * 0.45)
    c1x, c2x = x0 + reach, x1 - reach
    length = abs(x1 - x0) + abs(y1 - y0) + 2 * reach
    steps = max(12, min(64, int(length / 10)))
    points = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        points += [a * x0 + b * c1x + c * c2x + d * x1, a * y0 + b * y0 + c * y1 + d * y1]
    return tuple(points)


def count_crossings(graph, tree_only=True):
    """How many pairs of arrows (drawn straight) cross each other."""
    segments = []
    for parent, row, child, is_tree in graph.edges:
        if tree_only and not is_tree:
            continue
        segments.append(edge_points(parent, row, child))

    def side(ax, ay, bx, by, px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    count = 0
    for i, (ax, ay, bx, by) in enumerate(segments):
        for cx, cy, dx, dy in segments[i + 1:]:
            if {(ax, ay), (bx, by)} & {(cx, cy), (dx, dy)}:
                continue                               # sharing an endpoint is not crossing
            d1 = side(ax, ay, bx, by, cx, cy)
            d2 = side(ax, ay, bx, by, dx, dy)
            d3 = side(cx, cy, dx, dy, ax, ay)
            d4 = side(cx, cy, dx, dy, bx, by)
            if d1 * d2 < 0 and d3 * d4 < 0:
                count += 1
    return count


def nodes_in_region(graph, region):
    """The boxes whose rectangles touch `region` (left, top, right, bottom, world)."""
    x0, y0, x1, y1 = region
    return [n for n in graph.nodes
            if n.x <= x1 and n.x + n.width >= x0 and n.y <= y1 and n.y + n.height >= y0]


def visible_row_range(node, top, bottom):
    """(first, last+1) of the rows of `node` that touch world y range [top, bottom]."""
    base = node.y + TITLE_H
    first = min(len(node.rows), max(0, int((top - base) // ROW_H)))
    last = min(len(node.rows), int((bottom - base) // ROW_H) + 1)
    return first, max(first, last)


# ===================================================================== the window
def nav_button_numbers(windowing):
    """(back, forward) mouse button numbers of Tk for a mouse's side buttons. Checked
    on Windows with real side-button clicks: back arrives as button 4, forward as
    5. On X11 buttons 4 and 5 are the scroll wheel and the side buttons are 8 and
    9."""
    return (8, 9) if windowing == "x11" else (4, 5)


def newest_dump(folder):
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pkl")]
    except OSError:
        return None
    return max(files, key=os.path.getmtime) if files else None


class Viewer:
    def __init__(self, root):
        self.root = root
        root.title("Dump viewer")
        root.geometry("1500x900")
        self.font = tkfont.Font(family="Consolas", size=-REF_FONT_PX)
        self.bold = tkfont.Font(family="Consolas", size=-REF_FONT_PX, weight="bold")
        self.link = tkfont.Font(family="Consolas", size=-REF_FONT_PX, underline=True)
        self.char_w = max(1, self.font.measure("0"))
        self.graph = None               # every box, as loaded
        self.vg = None                  # the boxes the filters leave visible
        self.pos = {}                   # box id -> [x, y]: where each box is (hidden ones too)
        self.levels = {}                # type name -> 3/2/1/0
        self.level_vars = {}
        self.zoom, self.ox, self.oy = 1.0, 0.0, 0.0   # screen = (world - origin) * zoom
        self.selected = set()
        self.drag = None
        self.rects, self.edge_items, self.edges_of = {}, {}, {}
        self.drawn = (0, 0, 0, 0)       # the world region that is on the canvas
        self.font_px = REF_FONT_PX
        self.redraw_job = None
        self.back_stack = []            # views left by clicking a link (oldest first)
        self.forward_stack = []         # views to come back to with "forward"
        self.unchecked = set()          # (box id, row index) whose checkbox is cleared
        self.link_block_until = 0.0     # no link clicks before this time (see on_release)
        self.flash = None               # (box id, row) highlighted after jumping to a row
        self.cursor = ""
        self.path = None

        self.straight_var = tk.BooleanVar(value=False)
        self.min_gap_var = tk.DoubleVar(value=DEFAULT_MIN_GAP)
        self.aspect_var = tk.DoubleVar(value=0.0)
        self.group_links_var = tk.BooleanVar(value=False)    # View > Group link rows first
        self.links_only_var = tk.BooleanVar(value=False)     # sidebar: hide rows that aren't links
        self.sort_by_var = tk.StringVar(value=SORT_NONE)     # View > Sort objects ...
        self.sort_desc_var = tk.BooleanVar(value=False)
        self.section_open = {"boxes": True, "values": True}
        menu = tk.Menu(root, tearoff=False)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open...", accelerator="Ctrl+O", command=self.open_dialog)
        menu.add_cascade(label="File", menu=file_menu)
        view_menu = tk.Menu(menu, tearoff=False)
        view_menu.add_command(label="Fit everything", accelerator="F", command=self.fit)
        view_menu.add_command(label="Actual size (centred on the top box)", accelerator="1",
                              command=self.actual_size)
        view_menu.add_command(label="Rearrange visible boxes", accelerator="R",
                              command=self.rearrange)
        self.goto_menu = tk.Menu(view_menu, tearoff=False)
        view_menu.add_cascade(label="Go to", menu=self.goto_menu)
        view_menu.add_separator()
        view_menu.add_command(label="Navigate back", accelerator="Alt+Left",
                              command=self.nav_back, state="disabled")
        view_menu.add_command(label="Navigate forward", accelerator="Alt+Right",
                              command=self.nav_forward, state="disabled")
        view_menu.add_separator()
        view_menu.add_radiobutton(label="Swoopy connections", variable=self.straight_var,
                                  value=False, command=self.schedule_redraw)
        view_menu.add_radiobutton(label="Straight connections", variable=self.straight_var,
                                  value=True, command=self.schedule_redraw)
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Group link rows first", variable=self.group_links_var,
                                  command=self.refilter)
        sort_menu = tk.Menu(view_menu, tearoff=False)
        for value, label in SORT_CHOICES:
            sort_menu.add_radiobutton(label=label, variable=self.sort_by_var, value=value,
                                      command=self.refilter)
        sort_menu.add_separator()
        sort_menu.add_radiobutton(label="Ascending", variable=self.sort_desc_var, value=False,
                                  command=self.refilter)
        sort_menu.add_radiobutton(label="Descending", variable=self.sort_desc_var, value=True,
                                  command=self.refilter)
        view_menu.add_cascade(label="Sort objects within their parent", menu=sort_menu)
        menu.add_cascade(label="View", menu=view_menu)
        self.view_menu = view_menu
        root.config(menu=menu)

        self.build_sidebar()
        main_area = tk.Frame(root)
        main_area.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(main_area, background="#2b2b2b", highlightthickness=0)
        self.info = tk.Text(main_area, height=5, wrap="word", font=("Consolas", 10),
                            background="#f4f4f4")
        self.info.pack(side="bottom", fill="x")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.set_info("Open a dump with File > Open (Ctrl+O).")

        c = self.canvas
        c.bind("<MouseWheel>", self.on_wheel)
        windowing = root.tk.call("tk", "windowingsystem")
        if windowing == "x11":                       # there 4 and 5 are the scroll wheel
            c.bind("<Button-4>", lambda e: self.wheel(e.x, e.y, e.state, 1))
            c.bind("<Button-5>", lambda e: self.wheel(e.x, e.y, e.state, -1))
        back_button, forward_button = nav_button_numbers(windowing)
        root.bind(f"<ButtonPress-{back_button}>", lambda e: self.nav_back())
        root.bind(f"<ButtonPress-{forward_button}>", lambda e: self.nav_forward())
        root.bind("<Alt-Left>", lambda e: self.nav_back())
        root.bind("<Alt-Right>", lambda e: self.nav_forward())
        for keysym, action in (("XF86Back", self.nav_back), ("XF86Forward", self.nav_forward)):
            try:                                     # keyboards with browser keys
                root.bind(f"<{keysym}>", lambda e, a=action: a())
            except tk.TclError:
                pass
        c.bind("<ButtonPress-3>", self.on_pan_start)
        c.bind("<B3-Motion>", self.on_pan)
        c.bind("<ButtonRelease-3>", lambda e: self.redraw())
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_motion)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>", self.on_hover)
        c.bind("<Double-Button-1>", self.on_double_click)
        c.bind("<Configure>", lambda e: self.schedule_redraw())
        root.bind("<Key-f>", lambda e: self.fit())
        root.bind("<Key-1>", lambda e: self.actual_size())
        root.bind("<Key-r>", lambda e: self.rearrange())
        root.bind("<Escape>", lambda e: self.clear_selection())
        root.bind("<Control-o>", lambda e: self.open_dialog())

    # ---- sidebar ------------------------------------------------------------------------
    def build_sidebar(self):
        bg = "#f0f0f0"
        side = tk.Frame(self.root, width=290, background=bg)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        arrange = tk.LabelFrame(side, text=" Arrangement ", background=bg,
                                font=("Segoe UI", 9, "bold"))
        arrange.pack(fill="x", padx=8, pady=(8, 4))
        tk.Button(arrange, text="Rearrange", command=self.rearrange).pack(
            anchor="w", padx=6, pady=(4, 2))
        tk.Label(arrange, text="Minimum horizontal spacing", background=bg, anchor="w",
                 font=("Segoe UI", 8)).pack(fill="x", padx=6)
        tk.Scale(arrange, variable=self.min_gap_var, from_=MIN_GAP_RANGE[0],
                 to=MIN_GAP_RANGE[1], resolution=5, orient="horizontal", background=bg,
                 highlightthickness=0, length=250).pack(padx=6)
        tk.Label(arrange, text="Spacing aspect ratio", background=bg, anchor="w",
                 font=("Segoe UI", 8)).pack(fill="x", padx=6)
        tk.Scale(arrange, variable=self.aspect_var, from_=ASPECT_RANGE[0],
                 to=ASPECT_RANGE[1], resolution=0.01, orient="horizontal", background=bg,
                 highlightthickness=0, length=250).pack(padx=6, pady=(0, 4))

        head = tk.Frame(side, background=bg)
        head.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(head, text="Filters", font=("Segoe UI", 11, "bold"),
                 background=bg).pack(side="left")
        tk.Button(head, text="Show all", command=self.reset_filters).pack(side="right")
        tk.Label(side, justify="left", anchor="w", background=bg, wraplength=270,
                 font=("Segoe UI", 8), text=(
                     "4  show, and show children\n3  show, hide children\n"
                     "2  hide, but keep as a row in the parent\n"
                     "1  hide, and hide the row too\n"
                     "0  as 1, and hide the parent if it has\n"
                     "    nothing above 0 left")).pack(anchor="w", padx=8)
        tk.Checkbutton(side, text="Hide rows that aren't links", variable=self.links_only_var,
                       command=self.refilter, background=bg, anchor="w",
                       font=("Segoe UI", 9)).pack(fill="x", padx=8, pady=(4, 0))
        holder = tk.Frame(side, background=bg)
        holder.pack(fill="both", expand=True, pady=(4, 0))
        self.side_canvas = tk.Canvas(holder, background=bg, highlightthickness=0)
        scroll = tk.Scrollbar(holder, orient="vertical", command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.side_canvas.pack(side="left", fill="both", expand=True)
        self.filter_frame = tk.Frame(self.side_canvas, background=bg)
        self.side_canvas.create_window((0, 0), window=self.filter_frame, anchor="nw")
        self.filter_frame.bind("<Configure>", lambda e: self.side_canvas.configure(
            scrollregion=self.side_canvas.bbox("all")))
        self.side_canvas.bind("<MouseWheel>", lambda e: self.side_canvas.yview_scroll(
            -1 if e.delta > 0 else 1, "units"))

    def fill_sidebar(self):
        bg = "#f0f0f0"
        for child in self.filter_frame.winfo_children():
            child.destroy()
        self.level_vars = {}
        boxes, leaves = type_inventory(self.graph)
        for name, title, items in (("boxes", "Containers and objects", boxes),
                                   ("values", "Plain values", leaves)):
            if not items:
                continue
            header = tk.Label(self.filter_frame, font=("Segoe UI", 9, "bold"), background=bg,
                              cursor="hand2", anchor="w")
            header.pack(fill="x", padx=6, pady=(8, 2))
            body = tk.Frame(self.filter_frame, background=bg)
            header.bind("<Button-1>", lambda e, n=name, h=header, t=title, b=body:
                        self.toggle_section(n, h, t, b))
            self.show_section(name, header, title, body)
            for key, count in items:
                row = tk.Frame(body, background=bg)
                row.pack(fill="x", padx=6, pady=1)
                var = tk.IntVar(value=self.levels.get(key, SHOW))
                self.level_vars[key] = var
                for level in LEVELS:
                    tk.Radiobutton(row, text=str(level), variable=var, value=level,
                                   indicatoron=False, width=2, font=("Segoe UI", 8),
                                   command=lambda k=key: self.on_level_changed(k)
                                   ).pack(side="left")
                shown = key.rsplit(".", 1)[-1] if "." in key else key
                shown = shown if len(shown) <= 30 else shown[:29] + "\u2026"
                label = tk.Label(row, text=f" {shown} ({count})", anchor="w",
                                 font=("Segoe UI", 8), background=bg)
                label.pack(side="left", fill="x")
                label.bind("<Enter>", lambda e, k=key: self.set_info(k))

    def show_section(self, name, header, title, body):
        """Set a filter list's header arrow and show or hide its rows."""
        opened = self.section_open[name]
        header.configure(text=("\u25bc " if opened else "\u25b6 ") + title)
        if opened:
            body.pack(fill="x", after=header)
        else:
            body.pack_forget()

    def toggle_section(self, name, header, title, body):
        self.section_open[name] = not self.section_open[name]
        self.show_section(name, header, title, body)

    def on_level_changed(self, key):
        self.levels[key] = self.level_vars[key].get()
        self.refilter()

    def reset_filters(self):
        """Every type back to the top level, every checkbox ticked, all rows shown."""
        self.levels = {}
        self.unchecked = set()
        self.links_only_var.set(False)
        for var in self.level_vars.values():
            var.set(SHOW)
        self.refilter()

    def toggle_row(self, origin):
        """Tick or clear the checkbox of the row with this origin."""
        if origin in self.unchecked:
            self.unchecked.discard(origin)
        else:
            self.unchecked.add(origin)
        self.refilter()

    # ---- loading ------------------------------------------------------------------------
    def open_dialog(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = filedialog.askopenfilename(
            title="Open a dump", initialdir=os.path.join(here, "dumps")
            if os.path.isdir(os.path.join(here, "dumps")) else here,
            filetypes=[("Pickle dumps", "*.pkl"), ("All files", "*.*")])
        if path:
            self.open(path)

    def open(self, path):
        try:
            data = load_pickle(path)
        except Exception as exc:                    # noqa: BLE001
            messagebox.showerror("Dump viewer", f"Could not read {path}:\n{exc}")
            return
        self.path = path
        self.root.update()               # the window needs its real size before drawing
        self.graph = build_graph(data)
        self.levels, self.unchecked, self.selected = {}, set(), set()
        self.back_stack, self.forward_stack = [], []
        self.update_nav_menu()
        everything = layout_tree(self.make_view(), self.char_w, self.min_gap_var.get(),
                                 self.aspect_var.get())
        self.pos = {n.id: [n.x, n.y] for n in everything.nodes}
        self.root.title(f"Dump viewer - {os.path.basename(path)}")
        self.fill_sidebar()
        self.rebuild_view()
        self.goto_menu.delete(0, "end")
        self.goto_menu.add_command(label="Top box", command=lambda: self.go_to(self.vg.root))
        for row in self.graph.root.rows:
            if row.child is not None:
                self.goto_menu.add_command(
                    label=f"{row.key}: {row.child.title}",
                    command=lambda orig=row.child: self.go_to_id(orig.id))
        self.focus_root(START_ZOOM)
        meta = data.get("meta") if isinstance(data, dict) else None
        summary = f"{len(self.graph.nodes)} boxes, {len(self.graph.edges)} arrows"
        if isinstance(meta, dict):
            summary += (f" | cause {meta.get('cause')}, file version {meta.get('file_version')}, "
                        f"{meta.get('time')}, large arrays "
                        f"{'included' if meta.get('large_arrays_included') else 'left out'}")
        self.set_info(summary + "\nHover a row to read it in full here.")

    # ---- the visible graph --------------------------------------------------------------
    def make_view(self):
        """The visible boxes under the filters, the checkboxes and "hide rows that aren't
        links", with their rows grouped and sorted as View asks."""
        view = apply_filters(self.graph, self.levels, self.unchecked, self.links_only_var.get())
        return arrange_rows(view, self.group_links_var.get(), self.sort_by_var.get(),
                            self.sort_desc_var.get())

    def rebuild_view(self):
        """Work out what the filters leave visible (boxes keep the positions they
        had), and size the boxes to their visible rows."""
        self.vg = self.make_view()
        measure_nodes(self.vg, self.char_w)
        span_tree(self.vg)
        for node in self.vg.nodes:
            node.x, node.y = self.pos[node.id]
        self.edges_of = {}
        for k, (parent, _row, child, _tree) in enumerate(self.vg.edges):
            self.edges_of.setdefault(parent.id, []).append(k)
            self.edges_of.setdefault(child.id, []).append(k)
        self.selected &= set(self.vg.by_id)

    def refilter(self):
        self.rebuild_view()
        self.redraw()

    def rearrange(self):
        """Lay out the visible boxes again, so hidden ones take up no room."""
        if self.graph is None:
            return
        place_tree(self.vg, span_tree(self.vg), self.min_gap_var.get(), self.aspect_var.get())
        for node in self.vg.nodes:
            self.pos[node.id] = [node.x, node.y]
        self.focus_root(self.zoom)

    # ---- drawing ------------------------------------------------------------------------
    def to_screen(self, x, y):
        return (x - self.ox) * self.zoom, (y - self.oy) * self.zoom

    def canvas_size(self):
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def view_region(self, margin=0.0):
        vw, vh = self.canvas_size()
        z = self.zoom
        return (self.ox - margin * vw / z, self.oy - margin * vh / z,
                self.ox + (1 + margin) * vw / z, self.oy + (1 + margin) * vh / z)

    def schedule_redraw(self):
        """Redraw once, when Tk is next idle (many wheel clicks then cost one redraw)."""
        if self.redraw_job is None and self.vg is not None:
            self.redraw_job = self.root.after_idle(self.redraw)

    def redraw(self):
        if self.redraw_job is not None:
            self.root.after_cancel(self.redraw_job)
            self.redraw_job = None
        if self.vg is None:
            return
        c = self.canvas
        c.delete("all")
        self.rects, self.edge_items = {}, {}
        px = max(1, round(REF_FONT_PX * self.zoom))
        if px != self.font_px:                      # no text items exist just now: cheap
            self.font_px = px
            self.font.configure(size=-px)
            self.bold.configure(size=-px)
            self.link.configure(size=-px)
        region = self.view_region(DRAW_MARGIN)
        self.drawn = region
        show_text = px >= HIDE_TEXT_BELOW_PX
        nodes = nodes_in_region(self.vg, region)
        ids = {n.id for n in nodes}
        for k, (parent, row, child, is_tree) in enumerate(self.vg.edges):
            x0, y0, x1, y1 = edge_points(parent, row, child)
            if not arrow_visible(x0, y0, x1, y1, self.zoom):
                continue
            if (parent.id in ids or child.id in ids
                    or (min(x0, x1) <= region[2] and max(x0, x1) >= region[0]
                        and min(y0, y1) <= region[3] and max(y0, y1) >= region[1])):
                straight = self.straight_var.get() or self.zoom < STRAIGHT_BELOW_ZOOM
                coords = self.edge_coords(k, straight)
                dashed = (not is_tree
                          and abs(coords[-2] - coords[0]) + abs(coords[-1] - coords[1])
                          < DASH_MAX_PX)
                self.edge_items[k] = c.create_line(
                    *coords, arrow="last", arrowshape=ARROW_SHAPE, width=EDGE_WIDTH,
                    fill=EDGE_COLOR if is_tree else EXTRA_EDGE_COLOR,
                    dash=(7, 4) if dashed else (), tags=("edge",))
        for node in nodes:
            self.draw_node(node, show_text, region)

    def draw_node(self, node, show_text, region):
        c, z = self.canvas, self.zoom
        x0, y0 = self.to_screen(node.x, node.y)
        x1, y1 = x0 + node.width * z, y0 + node.height * z
        tag = f"n{node.id}"
        selected = node.id in self.selected
        self.rects[node.id] = c.create_rectangle(
            x0, y0, x1, y1, fill=FILL[node.kind],
            outline=SELECTED_OUTLINE if selected else "#333333", width=3 if selected else 1,
            tags=("node", tag))
        if not show_text:
            return
        c.create_rectangle(x0, y0, x1, y0 + TITLE_H * z, fill=TITLE_FILL[node.kind],
                           outline="#333333", tags=("node", tag))
        if z >= TWO_LINE_TITLE_ZOOM:
            c.create_text(x0 + PAD * z, y0 + TITLE_H * z * 0.26, anchor="w",
                          text=name_in_parent(node), font=self.link if node.parent else self.bold,
                          fill="#0b3d91" if node.parent else "#555555", tags=("node", tag))
            c.create_text(x0 + PAD * z, y0 + TITLE_H * z * 0.74, anchor="w", text=node.title,
                          font=self.bold, fill="#111111", tags=("node", tag))
        else:
            c.create_text(x0 + PAD * z, y0 + TITLE_H * z / 2, anchor="w", text=node.title,
                          font=self.bold, fill="#111111", tags=("node", tag))
        first, last = visible_row_range(node, region[1], region[3])
        for i in range(first, last):
            row = node.rows[i]
            link = row.child is not None
            if self.flash == (node.id, i):
                c.create_rectangle(x0 + 2, y0 + (TITLE_H + i * ROW_H) * z, x1 - 2,
                                   y0 + (TITLE_H + (i + 1) * ROW_H) * z, fill="#ffe680",
                                   outline="#ff8f00", tags=("node", tag))
            c.create_text(x0 + PAD * z, y0 + (TITLE_H + i * ROW_H + ROW_H / 2) * z, anchor="w",
                          text=f"{row.key}: {row.text}",
                          font=self.link if link else self.font,
                          fill="#0b3d91" if link else "#777777" if row.check is False
                          else "#111111", tags=("node", tag))
            if row.check is not None:
                self.draw_checkbox(node, i, row.check, x1, y0)
        if node.hidden:
            c.create_text(x0 + PAD * z,
                          y0 + (TITLE_H + len(node.rows) * ROW_H + FOOTER_H / 2) * z,
                          anchor="w", text=hidden_message(node.hidden), font=self.font,
                          fill=FOOTER_COLOR, tags=("node", tag))

    def draw_checkbox(self, node, i, ticked, x1, y0):
        c, z = self.canvas, self.zoom
        cx = x1 - (PAD + CHECK_SIZE / 2) * z
        cy = y0 + (TITLE_H + i * ROW_H + ROW_H / 2) * z
        half = CHECK_SIZE / 2 * z
        tags = ("node", f"n{node.id}")
        c.create_rectangle(cx - half, cy - half, cx + half, cy + half, fill="#ffffff",
                           outline="#333333", tags=tags)
        if ticked:
            c.create_line(cx - half * 0.6, cy, cx - half * 0.1, cy + half * 0.6,
                          cx + half * 0.7, cy - half * 0.6, fill="#1b7a1b",
                          width=max(1, round(1.6 * z)), tags=tags)

    def checkbox_hit(self, node, row, x, y):
        """Is window position (x, y) on the checkbox of row `row` of `node`?"""
        if not 0 <= row < len(node.rows) or node.rows[row].check is None:
            return False
        x0, y0 = self.to_screen(node.x, node.y)
        right = x0 + node.width * self.zoom
        cy = y0 + (TITLE_H + row * ROW_H + ROW_H / 2) * self.zoom
        return (right - (PAD + CHECK_SIZE + 3) * self.zoom <= x <= right - (PAD - 3) * self.zoom
                and abs(y - cy) <= ROW_H / 2 * self.zoom)

    def edge_coords(self, k, straight=None):
        parent, row, child, _ = self.vg.edges[k]
        x0, y0, x1, y1 = edge_points(parent, row, child)
        (sx0, sy0), (sx1, sy1) = self.to_screen(x0, y0), self.to_screen(x1, y1)
        if straight is None:
            straight = self.straight_var.get() or self.zoom < STRAIGHT_BELOW_ZOOM
        return edge_path(sx0, sy0, sx1, sy1, self.zoom, straight)

    def update_edges(self, node_ids):
        done = set()
        for nid in node_ids:
            for k in self.edges_of.get(nid, ()):
                if k not in done and k in self.edge_items:
                    done.add(k)
                    self.canvas.coords(self.edge_items[k], *self.edge_coords(k))

    # ---- view ---------------------------------------------------------------------------
    def zoom_at(self, x, y, factor):
        """Zoom by `factor` keeping the world point under window position (x, y)."""
        new = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))
        if new == self.zoom:
            return
        wx, wy = self.ox + x / self.zoom, self.oy + y / self.zoom
        self.zoom = new
        self.ox, self.oy = wx - x / new, wy - y / new
        self.schedule_redraw()

    def on_wheel(self, event):
        self.wheel(event.x, event.y, event.state, 1 if event.delta > 0 else -1)

    def wheel(self, x, y, state, direction):
        """One wheel click (`direction` 1 = away from you): with Ctrl held, zoom at the
        pointer; else with Shift held, scroll sideways; otherwise scroll up and down."""
        shift, ctrl = bool(state & 0x1), bool(state & 0x4)
        if ctrl:
            self.zoom_at(x, y, 1.2 if direction > 0 else 1 / 1.2)
        elif shift:
            self.scroll_by(direction * WHEEL_SCROLL_PX, 0)
        else:
            self.scroll_by(0, direction * WHEEL_SCROLL_PX)

    def scroll_by(self, dx, dy):
        """Move everything by (dx, dy) window pixels (the few items on the canvas, then a
        redraw only if the window has left the region that was drawn)."""
        if self.vg is None:
            return
        self.canvas.move("all", dx, dy)
        self.ox -= dx / self.zoom
        self.oy -= dy / self.zoom
        seen = self.view_region()
        d = self.drawn
        if seen[0] < d[0] or seen[1] < d[1] or seen[2] > d[2] or seen[3] > d[3]:
            self.redraw()

    def set_view(self, zoom, ox, oy):
        self.zoom = min(MAX_ZOOM, max(MIN_ZOOM, zoom))
        self.ox, self.oy = ox, oy
        self.redraw()

    def fit(self):
        if self.vg is None:
            return
        self.root.update_idletasks()
        nodes = self.vg.nodes
        x0 = min(n.x for n in nodes)
        y0 = min(n.y for n in nodes)
        x1 = max(n.x + n.width for n in nodes)
        y1 = max(n.y + n.height for n in nodes)
        vw, vh = self.canvas_size()
        zoom = min(vw / max(1, x1 - x0), vh / max(1, y1 - y0)) * 0.96
        zoom = min(MAX_ZOOM, max(MIN_ZOOM, zoom))
        self.set_view(zoom, (x0 + x1) / 2 - vw / 2 / zoom, (y0 + y1) / 2 - vh / 2 / zoom)

    def focus_root(self, zoom):
        """Zoom to `zoom` (1.0 = actual size) with the top box at the left edge of the
        window and vertically centred - a readable starting view, since a dump's
        tree is usually far taller than it is wide."""
        if self.vg is None:
            return
        self.root.update_idletasks()
        zoom = min(MAX_ZOOM, max(MIN_ZOOM, zoom))
        vw, vh = self.canvas_size()
        top = self.vg.root
        self.set_view(zoom, top.x - 20 / zoom, top.y + top.height / 2 - vh / 2 / zoom)

    def actual_size(self):
        self.focus_root(1.0)

    def go_to(self, node):
        """Pan (keeping the zoom) so the top-left of `node` is near the window's
        top-left corner."""
        self.set_view(self.zoom, node.x - 30 / self.zoom, node.y - 30 / self.zoom)
        self.set_info(f"{node.title}  ({len(node.rows)} rows)")

    def current_view(self):
        return (self.zoom, self.ox, self.oy)

    def navigate(self, action):
        """Run a link jump (`action`), remembering the view it leaves so "back" returns
        to it; a new jump forgets what "forward" would have gone to."""
        self.back_stack.append(self.current_view())
        del self.back_stack[:-HISTORY_LIMIT]
        self.forward_stack.clear()
        action()
        self.update_nav_menu()

    def nav_back(self):
        """Return to the view before the last link jump."""
        if not self.back_stack or self.vg is None:
            return
        self.forward_stack.append(self.current_view())
        self.show_view(self.back_stack.pop())

    def nav_forward(self):
        """Go on to the view "back" came from."""
        if not self.forward_stack or self.vg is None:
            return
        self.back_stack.append(self.current_view())
        self.show_view(self.forward_stack.pop())

    def show_view(self, view):
        self.flash = None
        self.set_view(*view)
        self.update_nav_menu()

    def update_nav_menu(self):
        """Grey out the View menu's back / forward entries when there is nowhere to go."""
        for label, stack in (("Navigate back", self.back_stack),
                             ("Navigate forward", self.forward_stack)):
            self.view_menu.entryconfigure(label, state="normal" if stack else "disabled")

    def go_to_row(self, parent, row):
        """Centre the window (keeping the zoom) on row `row` of box `parent`, and
        highlight that row for a couple of seconds."""
        z = self.zoom
        vh = self.canvas_size()[1]
        self.flash = (parent.id, row)
        self.set_view(z, parent.x - 30 / z, parent.y + TITLE_H + row * ROW_H + ROW_H / 2
                      - vh / 2 / z)
        self.set_info(f"{parent.title}\n{parent.rows[row].full}")
        self.root.after(2000, self.clear_flash)

    def clear_flash(self):
        if self.flash is not None:
            self.flash = None
            self.redraw()

    def name_hit(self, node, x, y):
        """Is window position (x, y) on the name line of `node`'s two-line title?"""
        if self.zoom < TWO_LINE_TITLE_ZOOM or node.parent is None:
            return False
        x0, y0 = self.to_screen(node.x, node.y)
        width = (len(name_in_parent(node)) * self.char_w + 2 * PAD) * self.zoom
        return x0 <= x <= x0 + width and y0 <= y <= y0 + TITLE_H * self.zoom / 2

    def go_to_id(self, nid):
        node = self.vg.by_id.get(nid)
        if node is None:
            self.set_info("That box is hidden by the filters.")
        else:
            self.go_to(node)

    def row_at(self, node, event_y):
        """The index of the row of `node` at window y, or -1 (title bar, edges)."""
        y0 = self.to_screen(node.x, node.y)[1]
        below_title = (event_y - y0) / self.zoom - TITLE_H
        row = int(below_title // ROW_H) if below_title >= 0 else -1
        return row if 0 <= row < len(node.rows) else -1

    # ---- mouse --------------------------------------------------------------------------
    def node_at(self, x, y):
        for item in reversed(self.canvas.find_overlapping(x, y, x, y)):
            tags = self.canvas.gettags(item)
            if "node" in tags:
                for t in tags:
                    if t.startswith("n") and t[1:].isdigit():
                        return int(t[1:])
        return None

    def on_double_click(self, event):
        """Tk sends a second quick press near the first as a double-click *instead of*
        a press, so this must do the press's work too (else a drag started right
        after a click would not begin)."""
        if self.vg is not None:
            self.on_press(event)

    def on_pan_start(self, event):
        self.pan_last = (event.x, event.y)

    def on_pan(self, event):
        """Move the few items on the canvas; redraw only if the window has left the
        region that was drawn."""
        dx, dy = event.x - self.pan_last[0], event.y - self.pan_last[1]
        self.pan_last = (event.x, event.y)
        self.scroll_by(dx, dy)

    def on_press(self, event):
        if self.vg is None:
            return
        ctrl = bool(event.state & 0x4)
        nid = self.node_at(event.x, event.y)
        if nid is not None and self.name_hit(self.vg.by_id[nid], event.x, event.y):
            node = self.vg.by_id[nid]
            self.navigate(lambda: self.go_to_row(node.parent, node.parent_row))
            return
        if nid is not None:
            node = self.vg.by_id[nid]
            row = self.row_at(node, event.y)
            if self.checkbox_hit(node, row, event.x, event.y):
                self.toggle_row(node.rows[row].origin)
                return
            link = node.rows[row].child if row >= 0 else None
            if nid not in self.selected:
                if not ctrl:
                    self.set_selection(set())
                self.set_selection(self.selected | {nid})
            for n in self.selected:
                self.canvas.tag_raise(f"n{n}")
            self.drag = {"mode": "nodes", "last": (event.x, event.y), "start": (event.x, event.y),
                         "moved": False, "link": link}
        else:
            self.drag = {"mode": "band", "start": (event.x, event.y), "moved": False,
                         "rect": None, "ctrl": ctrl}

    def on_motion(self, event):
        d = self.drag
        if d is None:
            return
        if d["mode"] == "nodes":
            dx, dy = event.x - d["last"][0], event.y - d["last"][1]
            d["last"] = (event.x, event.y)
            if abs(event.x - d["start"][0]) + abs(event.y - d["start"][1]) >= 3:
                d["moved"] = True
            for nid in self.selected:
                node = self.vg.by_id[nid]
                node.x += dx / self.zoom
                node.y += dy / self.zoom
                self.pos[nid] = [node.x, node.y]
                self.canvas.move(f"n{nid}", dx, dy)
            self.update_edges(self.selected)
        else:
            x0, y0 = d["start"]
            if not d["moved"] and abs(event.x - x0) + abs(event.y - y0) < 4:
                return
            d["moved"] = True
            if d["rect"] is None:
                d["rect"] = self.canvas.create_rectangle(x0, y0, event.x, event.y,
                                                         outline="#ffcc80", dash=(4, 3))
            else:
                self.canvas.coords(d["rect"], x0, y0, event.x, event.y)

    def on_release(self, event):
        d, self.drag = self.drag, None
        if d is None:
            return
        if d["mode"] == "nodes":
            now = time.monotonic()
            if d["link"] is not None and not d["moved"] and now >= self.link_block_until:
                self.link_block_until = now + LINK_CLICK_GUARD_S   # a double-click is one jump
                self.navigate(lambda: self.go_to(d["link"]))   # a click (not a drag) on a link
            else:
                self.redraw()                # arrows to boxes that were off screen, and so on
            return
        if d["rect"] is not None:
            self.canvas.delete(d["rect"])
        if not d["moved"]:
            self.set_selection(set())                  # a plain click on the background
            return
        x0, y0 = d["start"]
        hit = set()
        for item in self.canvas.find_overlapping(min(x0, event.x), min(y0, event.y),
                                                 max(x0, event.x), max(y0, event.y)):
            tags = self.canvas.gettags(item)
            if "node" in tags:
                for t in tags:
                    if t.startswith("n") and t[1:].isdigit():
                        hit.add(int(t[1:]))
        self.set_selection((self.selected | hit) if d["ctrl"] else hit)

    def on_hover(self, event):
        if self.vg is None or self.drag is not None:
            return
        nid = self.node_at(event.x, event.y)
        node = self.vg.by_id[nid] if nid is not None else None
        row = self.row_at(node, event.y) if node is not None else -1
        clickable = node is not None and (
            self.name_hit(node, event.x, event.y) or self.checkbox_hit(node, row, event.x, event.y)
            or (row >= 0 and node.rows[row].child is not None))
        cursor = "hand2" if clickable else ""
        if cursor != self.cursor:
            self.cursor = cursor
            self.canvas.configure(cursor=cursor)
        if node is None:
            return
        if self.checkbox_hit(node, row, event.x, event.y):
            tick = "ticked" if node.rows[row].check else "cleared"
            self.set_info(f"{node.title}\nCheckbox ({tick}): shows or hides the box this row "
                          "points at (it must also be allowed by the filters).")
        elif row >= 0:
            self.set_info(f"{node.title}\n{node.rows[row].full}")
        else:
            extra = f", {hidden_message(node.hidden)}" if node.hidden else ""
            self.set_info(f"{node.title}  ({len(node.rows)} rows shown{extra})")

    # ---- selection ----------------------------------------------------------------------
    def set_selection(self, ids):
        for nid in self.selected - ids:
            if nid in self.rects:
                self.canvas.itemconfigure(self.rects[nid], outline="#333333", width=1)
        for nid in ids - self.selected:
            if nid in self.rects:
                self.canvas.itemconfigure(self.rects[nid], outline=SELECTED_OUTLINE, width=3)
        self.selected = set(ids)

    def clear_selection(self):
        self.set_selection(set())

    def set_info(self, text):
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.configure(state="disabled")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = tk.Tk()
    viewer = Viewer(root)
    path = argv[0] if argv else newest_dump(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "dumps"))
    if path:
        root.after(50, lambda: viewer.open(path))
    else:
        root.after(50, viewer.open_dialog)
    root.mainloop()


if __name__ == "__main__":
    main()
