"""Which part of lightbounces/core.py (what is left of the old single file) uses which other part (run: python dev/refactor/deps_map.py).

Cuts the file into the regions in ANCHORS (each region starts at the first top-level definition named
there and runs to the next anchor, so moving code around does not stale it), then reports, for every
region, which names it uses from other regions, and every use of a region defined LATER in the file -
the "back-references" that would become import cycles if the regions were separate modules.

Used to make dev/refactor_plan.md. After step 1 of the plan the only back-references left are the
kernels (and the constants they need) being below geometry and shapes.
"""
import ast
import collections
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lightbounces", "core.py")
ANCHORS = [("consts", None), ("consts2+blur", "world_radius_to_pixels"), ("records", "strict_record"),
           ("ui_helpers", "make_scrollable_tab"), ("keyframe", "Keyframe"), ("shapes", "ShapeParam"),
           ("settings", "SettingEntry"), ("misc_utils", "ensure_project_dir"), ("image_io", "png_rgba"),
           ("naming", "default_filename"), ("geometry", "ellipse_bound"),
           ("metrics", "ST_HDIST"), ("views", "COLOR_MODELS"), ("project", "PROJECT_EXTENSION"),
           ("selfchecks", "run_self_checks"), ("coloration", "make_mapping"),
           ("kernels", "_circle_roots"), ("app", "App")]
REGIONS = []


def build_regions(tree, total_lines):
    """REGIONS = [(name, first line, last line)] from ANCHORS (the first anchor is line 1)."""
    starts = []
    for name, anchor in ANCHORS:
        if anchor is None:
            starts.append((name, 1))
            continue
        for node in tree.body:
            names = ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                     [x.id for t in getattr(node, "targets", []) for x in ast.walk(t)
                      if isinstance(x, ast.Name)])
            if anchor in names:
                first = node.lineno
                while first > 1 and LINES[first - 2].startswith("#"):
                    first -= 1
                starts.append((name, first))
                break
        else:
            print(f"(region {name}: anchor {anchor!r} is not in core.py any more - it has been moved out)")
    assert [s for _n, s in starts] == sorted(s for _n, s in starts), \
        "an anchor is out of order: a region would be empty or negative - move the anchor"
    for k, (name, first) in enumerate(starts):
        last = starts[k + 1][1] - 1 if k + 1 < len(starts) else total_lines
        REGIONS.append((name, first, last))


def region_of(line):
    for name, first, last in REGIONS:
        if first <= line <= last:
            return name
    return None


LINES = []


def main():
    global LINES
    with open(PATH, encoding="utf-8") as handle:
        text = handle.read()
    LINES = text.split("\n")
    tree = ast.parse(text)
    build_regions(tree, len(LINES))
    defined = {}
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [x.id for t in node.targets for x in ast.walk(t) if isinstance(x, ast.Name)]
        for name in names:
            defined.setdefault(name, (region_of(node.lineno), node.lineno))
    uses = collections.defaultdict(lambda: collections.defaultdict(list))
    for node in tree.body:
        here = region_of(node.lineno)
        top = getattr(node, "name", "<assignment>")
        for x in ast.walk(node):
            if (isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id in defined
                    and defined[x.id][0] != here):
                uses[(here, defined[x.id][0])][x.id].append((x.lineno, top))
    order = {name: i for i, (name, _a, _b) in enumerate(REGIONS)}
    print("region -> regions it uses (number of distinct names)")
    for name, _a, _b in REGIONS:
        row = {there: len(names) for (here, there), names in uses.items() if here == name}
        print(f"  {name:13s} {dict(sorted(row.items(), key=lambda kv: -kv[1]))}")
    print("\nback-references (a region using one defined later in the file):")
    for (here, there), names in sorted(uses.items(), key=lambda kv: (order[kv[0][0]], order[kv[0][1]])):
        if order[there] > order[here]:
            for name, where in names.items():
                print(f"  {here} -> {there}: {name} (defined line {defined[name][1]}), "
                      f"used at line {where[0][0]} in {where[0][1]}")


if __name__ == "__main__":
    main()
