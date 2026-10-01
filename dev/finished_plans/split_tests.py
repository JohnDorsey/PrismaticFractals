"""One-off tool used by step 8 of dev/refactor_plan.md: splits test_light_bounces.py into tests/<area>.py files (kept as history).

    python dev/refactor/split_tests.py [--go]

Each top-level statement of the old file (with the banner and comments above it) goes to the area its section belongs to (AREAS maps the line
of a section's title to an area); a helper used by more than one area (and everything such a helper uses) goes to tests/support.py. Without --go
it only reports. Run on the single-file version; the original is read from git HEAD~ or the path given by --source.
"""
import ast
import collections
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SOURCE = os.path.join(ROOT, "test_light_bounces.py")

# section title line -> area (a section continues until the next entry)
AREAS = {
    34: "selfchecks", 78: "inputs", 115: "kernels", 183: "shapes", 202: "images", 304: "images", 321: "shapes",
    678: "settings", 722: "views", 968: "export", 1181: "export", 1393: "export", 1607: "export", 1649: "preview",
    2097: "video", 2139: "video", 2584: "video", 2694: "coloration", 2757: "coloration", 2913: "inputs",
    3015: "coloration", 3366: "timeline", 3468: "shapes", 3603: "kernels", 3658: "kernels", 3661: "kernels",
    3859: "kernels", 3928: "kernels", 4006: "widgets", 4026: "kernels", 4114: "shapes", 4230: "shapes", 4476: "shapes",
    4493: "images", 4543: "coloration", 4586: "settings", 4606: "kernels", 4937: "kernels", 5331: "debug",
    5455: "export", 5562: "export", 5650: "coloration", 5743: "coloration", 5831: "images", 5932: "coloration",
    5944: "timeline", 6145: "timeline", 6501: "debug", 6978: "debug", 7107: "dump_viewer", 8512: "settings",
    8680: "settings", 8868: "settings", 8949: "kernels", 9010: "views", 9043: "projects", 9691: "projects",
    9807: "package",
}


def stores(node):
    names = []
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
            names.append(x.id)
    return names


def loads(node):
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def main(go):
    text = open(SOURCE, "rb").read().decode("utf-8")
    lines = text.split("\n")
    tree = ast.parse(text)
    body = tree.body
    runner_at = next(n.lineno for n in body if isinstance(n, ast.ClassDef) and n.name == "SkipTest")
    starts = sorted(AREAS)
    chunks = []                                   # (node, first_line_index, last_line_index)
    prev_end = 0
    for node in body:
        last = node.end_lineno
        chunks.append((node, prev_end, last))     # 0-based slice lines[prev_end:last]
        prev_end = last
    tail = "\n".join(lines[prev_end:])
    # which area owns each node
    def area_of(node):
        if node.lineno >= runner_at - 3:
            return "runner"
        mapped = [s for s in starts if s <= node.lineno]
        return AREAS[mapped[-1]] if mapped else "header"
    owner, defined_in, node_loads = {}, {}, {}
    for i, (node, _a, _b) in enumerate(chunks):
        owner[i] = area_of(node)
        node_loads[i] = loads(node)
        for name in stores(node):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                defined_in.setdefault(name, i)
    header = [i for i in owner if owner[i] == "header"]
    header_names = {n for i in header for n in stores(chunks[i][0])} | {
        (a.asname or a.name).split(".")[0] for i in header for a in getattr(chunks[i][0], "names", [])}
    # usage areas of every defined name
    users = collections.defaultdict(set)
    for i in owner:
        for name in node_loads[i]:
            if name in defined_in and defined_in[name] != i:
                users[name].add(owner[i])
    shared_nodes = set()
    for name, i in defined_in.items():
        if owner[i] in ("header", "runner"):
            continue
        if len(users[name] | {owner[i]}) > 1:
            shared_nodes.add(i)
    changed = True
    while changed:                                # whatever a shared node uses must be shared too
        changed = False
        for i in list(shared_nodes):
            for name in node_loads[i]:
                j = defined_in.get(name)
                if j is not None and j != i and owner[j] not in ("header", "runner") and j not in shared_nodes:
                    shared_nodes.add(j)
                    changed = True
    dest = {}
    for i in owner:
        dest[i] = "support" if i in shared_nodes else owner[i]
    report = collections.Counter(dest.values())
    tests_per = collections.Counter(chunks[i][0].name for i in owner
                                    if isinstance(chunks[i][0], ast.FunctionDef) and chunks[i][0].name.startswith("test_")
                                    for _ in [0] if False)
    count = collections.Counter(dest[i] for i in owner
                                if isinstance(chunks[i][0], ast.FunctionDef) and chunks[i][0].name.startswith("test_"))
    print("nodes per file:", dict(report))
    print("tests per file:", dict(count), "total", sum(count.values()))
    print("shared helpers:", sorted(n for n, i in defined_in.items() if i in shared_nodes)[:80])
    if not go:
        return
    out = collections.defaultdict(list)
    for i, (node, a, b) in enumerate(chunks):
        if dest[i] in ("header", "runner"):
            continue
        chunk = "\n".join(lines[a:b])
        chunk = chunk.replace("os.path.dirname(os.path.abspath(__file__))", "ROOT")
        out[dest[i]].append(chunk)
    tests_dir = os.path.join(ROOT, "tests")
    os.makedirs(tests_dir, exist_ok=True)
    stdlib = ["dataclasses", "os", "shutil", "sys", "tempfile", "threading", "traceback"]
    for name, chunk_list in out.items():
        body_text = "\n".join(chunk_list).strip("\n") + "\n"
        used = set(re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", body_text))
        imports = [f"import {mod}" for mod in stdlib if mod in used] + (["import numpy as np"] if "np" in used else [])
        if name == "support":
            head = ('"""Shared by the test files of this folder: the one loaded copy of the package (`m`), the root folder, the\n'
                    'skip exception, and every helper, fake app and fixture used by more than one area."""\n'
                    "import dataclasses\nimport os\nimport shutil\nimport sys\nimport tempfile\nimport threading\nimport traceback\n\n"
                    "import numpy as np\n\nfrom test_harness import load_module\n\n"
                    "ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the folder with light_bounces.py\n"
                    "m = load_module()\n\n\n"
                    "class SkipTest(Exception):\n"
                    '    """Raised by a test that can\'t run in this environment (e.g. needs a real external tool that isn\'t\n'
                    '    installed); reported as SKIP, not PASS."""\n\n\n')
            text_out = head + body_text + "\n\n__all__ = [n for n in dir() if not n.startswith(\"__\")]\n"
        else:
            head = f'"""Tests: {name}."""\n' + "\n".join(imports) + ("\n" if imports else "") + "\nfrom .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers\n\n\n"
            text_out = head + body_text
        with open(os.path.join(tests_dir, ("test_" + name if name != "support" else name) + ".py"), "wb") as f:
            f.write(text_out.encode("utf-8"))
    with open(os.path.join(tests_dir, "__init__.py"), "wb") as f:
        f.write(b'"""The test suite: one module per area (test_*.py), shared helpers in support.py, run by runner.py."""\n')


if __name__ == "__main__":
    main("--go" in sys.argv)
