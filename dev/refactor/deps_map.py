"""Which part of light_bounces.py uses which other part (run: python dev/refactor/deps_map.py).

Cuts the file into the regions in REGIONS (by line number), then reports, for every region, which
names it uses from other regions, and every use of a region defined LATER in the file - the
"back-references" that would become import cycles if the regions were separate modules.

The line numbers below are those of light_bounces.py at FILE_VERSION 113 (before any move). After
step 1 of dev/refactor_plan.md they are stale: update REGIONS from the file's section comments
first. This script was used to make the plan; it is kept so the picture can be redone.
"""
import ast
import collections
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "light_bounces.py")
REGIONS = [("consts", 1, 361), ("consts2+blur", 362, 594), ("records", 595, 959),
           ("ui_helpers", 960, 1111), ("keyframe", 1112, 1147), ("shapes", 1148, 1881),
           ("settings", 1882, 2182), ("misc_utils", 2183, 2447), ("image_io", 2448, 2862),
           ("geometry", 2863, 3371), ("metrics", 3372, 3786), ("views", 3787, 3988),
           ("project", 3989, 4536), ("selfchecks", 4537, 4686), ("coloration", 4687, 5006),
           ("kernels", 5007, 6230), ("app", 6231, 11000)]


def region_of(line):
    for name, first, last in REGIONS:
        if first <= line <= last:
            return name
    return None


def main():
    with open(PATH, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
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
