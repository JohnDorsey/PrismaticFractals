"""Moves top-level definitions out of PrismaticFractals/core.py into a new module of the package.

Used by the steps of dev/finished_plans/refactor_plan.md (import it from a throw-away script):

    import extract_tool as x
    x.extract("image_io", ["write_png", "read_png", ...], doc="PNG/QOI/JXL writers ...")

What it does, so a step stays a pure move:
  * cuts each named top-level definition (with its decorators and the comment lines directly above it) out of
    core.py, in file order, and writes them unchanged into PrismaticFractals/<module>.py;
  * gives the new module the imports it needs: the standard-library / numpy / tkinter imports of core.py that
    its code uses, and `from .other import name, ...` for every name it uses that another package module
    defines. A name it uses that is STILL defined in core.py is an import cycle: it stops and lists them
    (move those too, or move the definition they need first);
  * adds to core.py `from .<module> import name, ...` for the names core.py still uses from it;
  * rewrites every read of a PATCHABLE name (tests patch these) into an attribute read of its owner module
    (`paths.EXPORT_DIR`, `external.ffmpeg`) - never a by-value import (rule R2 of the plan).
Files are written with LF line endings (the working copy is LF; git normalises).
"""
import ast
import builtins
import collections
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PKG = os.path.join(ROOT, "PrismaticFractals")
CORE = os.path.join(PKG, "core.py")

def _patchable_globals():
    """name -> owner module, read from PATCHABLE_GLOBALS in PrismaticFractals/__init__.py (the one list); such a name
    is always read as `<owner>.<name>` at call time."""
    with open(os.path.join(PKG, "__init__.py"), encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "PATCHABLE_GLOBALS":
            return ast.literal_eval(node.value)
    raise SystemExit("PATCHABLE_GLOBALS not found in PrismaticFractals/__init__.py")


PATCHABLE = _patchable_globals()
NOT_PACKAGE = ("__init__", "core")


def read(path):
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8").replace("\r\n", "\n")


def write(path, text):
    with open(path, "wb") as handle:
        handle.write(text.encode("utf-8"))


def names_of(node):
    """The module-level names a top-level statement defines."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    names = []
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Try, ast.If)):
        for x in ast.walk(node):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                names.append(x.id)
            elif isinstance(x, (ast.Import, ast.ImportFrom)):
                names += [(a.asname or a.name).split(".")[0] for a in x.names]
            elif isinstance(x, (ast.FunctionDef, ast.ClassDef)) and isinstance(node, (ast.Try, ast.If)):
                names.append(x.name)
    return names


def span(lines, node):
    """First and last line index (0-based, inclusive) of a statement with its decorators and the comment
    lines directly above it."""
    first = node.lineno - 1
    if getattr(node, "decorator_list", None):
        first = min(d.lineno for d in node.decorator_list) - 1
    while first > 0 and lines[first - 1].startswith("#") and not lines[first - 1].startswith("# ---"):
        first -= 1                       # (a banner line of dashes stays where it is)
    last = node.end_lineno - 1
    while last + 1 < len(lines) and re.match(r"^\s+#", lines[last + 1]):
        last += 1                        # the rest of a trailing comment that goes on over several lines
    return first, last


def loaded_names(nodes):
    return {x.id for node in nodes for x in ast.walk(node)
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def package_modules():
    """module name -> text for every package module except __init__ and core."""
    out = {}
    for name in sorted(os.listdir(PKG)):
        if name.endswith(".py") and name[:-3] not in NOT_PACKAGE:
            out[name[:-3]] = read(os.path.join(PKG, name))
    return out


def definitions(text):
    """name -> the first top-level statement defining it."""
    result = {}
    for node in ast.parse(text).body:
        for name in names_of(node):
            result.setdefault(name, node)
    return result


def owners():
    """name -> module for every top-level name of the package modules (not core)."""
    result = {}
    for module, text in package_modules().items():
        for name, node in definitions(text).items():
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                result.setdefault(name, module)
    return result


def import_lines(core_tree, used, skip=()):
    """The import statements of core.py (and its numba try-block) restricted to the names in `used`."""
    out = []
    for node in core_tree.body:
        if isinstance(node, ast.Import):
            parts = [a for a in node.names if (a.asname or a.name).split(".")[0] in used]
            if parts:
                out.append("import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "")
                                                 for a in parts))
        elif isinstance(node, ast.ImportFrom) and not node.level:
            parts = [a for a in node.names if (a.asname or a.name) in used]
            if parts:
                out.append(f"from {node.module} import " + ", ".join(
                    a.name + (f" as {a.asname}" if a.asname else "") for a in parts))
    return out


NUMBA_BLOCK = '''try:
    from numba import njit, prange
except ImportError:  # works, but far too slowly to be practical
    print("WARNING: numba not found - falling back to pure Python (very slow).")
    print("         Install it with:  pip install numba")

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    prange = range'''


def wrap_import(prefix, names):
    names = sorted(names)
    one = f"{prefix} import {', '.join(names)}"
    if len(one) <= 100:
        return one
    body, line = [], "    "
    for n in names:
        piece = n + ", "
        if len(line) + len(piece) > 98:
            body.append(line.rstrip())
            line = "    "
        line += piece
    body.append(line.rstrip(", "))
    return f"{prefix} import (\n" + "\n".join(body) + "\n)"


def rewrite_patchables(text, own_module):
    """Reads of a PATCHABLE name become `<owner>.<name>` (not in the owner module itself). Returns
    (new text, set of owner modules now needed)."""
    tree = ast.parse(text)
    lines = text.split("\n")
    edits, needed = [], set()
    defined_here = set()
    for node in tree.body:
        defined_here.update(names_of(node))
    for x in ast.walk(tree):
        if (isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id in PATCHABLE
                and PATCHABLE[x.id] != own_module):
            edits.append((x.lineno - 1, x.col_offset, x.id))
            needed.add(PATCHABLE[x.id])
    for line, col, name in sorted(edits, reverse=True):
        lines[line] = lines[line][:col] + f"{PATCHABLE[name]}.{name}" + lines[line][col + len(name):]
    return "\n".join(lines), needed


def ensure_owner_imports(text, own_module):
    """`from . import paths, external` after the last top-level import, for every owner module of a patchable
    name that the text uses (as `paths.X`) and is not itself (adds only what is missing)."""
    tree = ast.parse(text)
    owners_needed = {owner for owner in set(PATCHABLE.values()) if owner != own_module
                     and any(isinstance(x, ast.Name) and x.id == owner for x in ast.walk(tree))}
    have = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
            have.update(a.name for a in node.names)
    missing = sorted(set(owners_needed) - have)
    if not missing:
        return text
    lines = text.split("\n")
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = node.end_lineno
        elif isinstance(node, ast.Try) and last:
            last = node.end_lineno
    lines[last:last] = [f"from . import {', '.join(missing)}"]
    return "\n".join(lines)


def register_in_init(module):
    """Add `module` to the package's `from . import ...` line and to _MODULES (just before core: modules are
    listed bottom-up, so a name defined in two modules takes the higher one)."""
    path = os.path.join(PKG, "__init__.py")
    text = read(path)
    match = re.search(r"^from \. import (.*)$", text, flags=re.M)
    names = sorted(set(match.group(1).split(", ")) | {module})
    text = text.replace(match.group(0), "from . import " + ", ".join(names))
    match = re.search(r"^_MODULES = \((.*)\)$", text, flags=re.M)
    items = [n for n in match.group(1).split(", ") if n]
    if module not in items:
        items.insert(items.index("core"), module)
    text = text.replace(match.group(0), "_MODULES = (" + ", ".join(items) + ")")
    write(path, text)


def extract(module, names, doc, numba=False, allow_core=(), keep_in_core=(), dry_run=False):
    """See the module docstring. `numba` puts the numba import block at the top of the new module.
    `allow_core` lists names the new module may use although core.py still defines them (only for
    names that are really local variables). `dry_run` only reports (and still stops on a cycle).
    Returns a short report."""
    core_text = read(CORE)
    core_tree = ast.parse(core_text)
    core_lines = core_text.split("\n")
    wanted = set(names)
    chosen, found = [], set()
    for node in core_tree.body:
        hit = wanted & set(names_of(node))
        if hit:
            chosen.append(node)
            found |= hit
    missing = wanted - found
    if missing:
        raise SystemExit(f"not top-level definitions of core.py: {sorted(missing)}")
    blocks, cut_spans = [], []
    for node in chosen:
        first, last = span(core_lines, node)
        blocks.append("\n".join(core_lines[first:last + 1]))
        cut_spans.append((first, last))
    body = ""
    for k, (block, node) in enumerate(zip(blocks, chosen)):
        if k:
            simple = lambda n: isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            if cut_spans[k][0] == cut_spans[k - 1][1] + 1:
                body += "\n"                         # they were adjacent in core.py
            else:
                body += "\n\n" if simple(node) and simple(chosen[k - 1]) else "\n\n\n"
        body += block
    for name, owner in PATCHABLE.items():       # a patchable name moved into its owner module is a plain name there
        if owner == module:
            body = re.sub(rf"\b{owner}\.{name}\b", name, body)

    mod_path = os.path.join(PKG, module + ".py")
    existing = read(mod_path) if os.path.exists(mod_path) else None
    here = set()
    for node in chosen:
        here.update(names_of(node))
    if existing is not None:
        here.update(definitions(existing))

    others = owners()
    core_defs = definitions(core_text)
    remaining_defs = {n for n, node in core_defs.items()
                      if node not in chosen and not isinstance(node, (ast.Import, ast.ImportFrom))}
    used = loaded_names(chosen) - set(dir(builtins))
    provided = {"njit", "prange"} if numba else set()       # the numba block is copied into the new module
    cycle = sorted(n for n in used if n in remaining_defs and n not in here and n not in allow_core
                   and n not in PATCHABLE and n not in provided)
    if cycle:
        raise SystemExit(f"{module}: would need names that are still defined in core.py (import cycle): {cycle}")
    from_other = collections.defaultdict(set)
    for n in used:
        if n in others and n not in here and n not in PATCHABLE and others[n] != module:
            from_other[others[n]].add(n)
    if dry_run:
        return (f"dry run, nothing written: {module} would take {len(chosen)} definitions "
                f"({sum(l - f + 1 for f, l in cut_spans)} lines) and import from "
                f"{ {k: sorted(v) for k, v in from_other.items()} }")

    # core.py without the cut definitions
    keep = [True] * len(core_lines)
    for first, last in cut_spans:
        for i in range(first, last + 1):
            keep[i] = False
    new_core = "\n".join(l for l, k in zip(core_lines, keep) if k)
    new_core = re.sub(r"\n{4,}", "\n\n\n", new_core)

    # the new module
    header_used = used | {x.value.id for node in chosen for x in ast.walk(node)
                          if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)}
    imports = import_lines(core_tree, header_used)
    lines_out = [f'"""{doc}"""']
    lines_out += imports
    if numba:
        lines_out += ["", NUMBA_BLOCK]
    for other, ns in sorted(from_other.items()):
        lines_out.append(wrap_import(f"from .{other}", ns))
    header = "\n".join(lines_out)
    if existing is None:
        text = header + "\n\n\n" + body + "\n"
    else:
        text = existing.rstrip("\n") + "\n\n\n" + body + "\n"
    text, needed = rewrite_patchables(text, module)
    text = ensure_owner_imports(text, module)
    write(mod_path, text)

    # what core.py still uses from the new module
    new_defs = definitions(read(mod_path))
    core_used = loaded_names(ast.parse(new_core).body) if new_core.strip() else set()
    core_needs = sorted(n for n in core_used if n in new_defs and n not in PATCHABLE
                        and not isinstance(new_defs[n], (ast.Import, ast.ImportFrom))
                        and n not in definitions(new_core))
    new_core, needed_core = rewrite_patchables(new_core, "core")
    imp = []
    if core_needs:
        imp.append(wrap_import(f"from .{module}", core_needs))
    tree = ast.parse(new_core)
    last = 0
    for k, node in enumerate(tree.body):
        if k == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            last = node.end_lineno                       # the module docstring
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Try)):
            last = node.end_lineno
        else:
            break
    cl = new_core.split("\n")
    cl[last:last] = imp
    new_core = "\n".join(cl)
    new_core = ensure_owner_imports(new_core, "core")
    register_in_init(module)
    write(CORE, new_core)
    return (f"{module}: moved {len(chosen)} definitions ({len(body.splitlines())} lines); "
            f"core now imports {len(core_needs)} names from it; it imports from "
            f"{ {k: len(v) for k, v in from_other.items()} }")
