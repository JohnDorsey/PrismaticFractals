"""Moves methods of `class App` (lightbounces/core.py) into a mixin class in a new module.

    python dev/refactor/extract_methods.py MODULE MixinName DOCFILE [--go] first_method:last_method [name ...]

Each argument is either one method name or an anchor range `first:last` meaning every method of App from `first` to `last` in file order. Without --go it
only reports. The new module gets `class MixinName:` with the methods unchanged, the imports they need (standard library / numpy / tkinter from core.py,
`from .other import ...` for names the package modules define; a name still defined in core.py is an import cycle and stops it), and reads of patchable
globals rewritten to `paths.X`. core.py loses the methods, imports the mixin and lists it among the bases of App. Names that are really local variables
(i, s, key...) are reported as cycles or imported by mistake: run dev/refactor/prune_imports.py on both files afterwards.
"""
import ast
import builtins
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_tool as x                                   # noqa: E402

BOGUS = {"i", "s", "c", "p", "label", "tab", "key", "spec", "_tab", "_label", "cls", "field", "is_float", "_is_float", "line", "idx",
         "record_class", "table", "kind", "catalog", "value", "o", "v", "k", "n", "r", "t", "d", "e", "f", "g", "w", "h", "m", "a", "b"}


def method_span(lines, node):
    first, last = x.span(lines, node)
    while first > 0 and re.match(r"^    #", lines[first - 1]):
        first -= 1                                          # indented comment lines directly above (section banners too)
    return first, last


def main(argv):
    go = "--go" in argv
    args = [a for a in argv if not a.startswith("--")]
    module, mixin, docfile, specs = args[0], args[1], args[2], args[3:]
    doc = open(docfile, encoding="utf-8").read().strip()
    core_text = x.read(x.CORE)
    core_tree = ast.parse(core_text)
    core_lines = core_text.split("\n")
    app = next(n for n in core_tree.body if isinstance(n, ast.ClassDef) and n.name == "App")
    methods = [n for n in app.body if isinstance(n, ast.FunctionDef)]
    order = [m.name for m in methods]
    wanted = []
    for item in specs:
        if ":" in item:
            a, b = item.split(":")
            wanted += order[order.index(a):order.index(b) + 1]
        else:
            wanted.append(item)
    seen = set()
    wanted = [n for n in wanted if not (n in seen or seen.add(n))]
    chosen = [m for m in methods if m.name in wanted]
    assert len(chosen) == len(wanted), set(wanted) - {m.name for m in chosen}
    blocks, cuts = [], []
    for m in chosen:
        first, last = method_span(core_lines, m)
        blocks.append("\n".join(core_lines[first:last + 1]))
        cuts.append((first, last))
    body = "\n\n".join(blocks)
    # names the methods read
    used = x.loaded_names(chosen) - set(dir(builtins))
    others = x.owners()
    core_defs = x.definitions(core_text)
    core_top = {n for n, node in core_defs.items() if not isinstance(node, (ast.Import, ast.ImportFrom))}
    cycle = sorted(n for n in used if n in core_top and n not in BOGUS and n not in x.PATCHABLE)
    if cycle:
        raise SystemExit(f"{module}: needs names still defined in core.py: {cycle}")
    from_other = {}
    for n in used:
        if n in others and n not in x.PATCHABLE:
            from_other.setdefault(others[n], set()).add(n)
    print(f"{mixin}: {len(chosen)} methods, {sum(l - f + 1 for f, l in cuts)} lines; imports from "
          f"{ {k: len(v) for k, v in from_other.items()} }")
    if not go:
        return
    header_used = used | {y.value.id for m in chosen for y in ast.walk(m)
                          if isinstance(y, ast.Attribute) and isinstance(y.value, ast.Name)}
    lines_out = [f'"""{doc}"""'] + x.import_lines(core_tree, header_used)
    for other, ns in sorted(from_other.items()):
        lines_out.append(x.wrap_import(f"from .{other}", ns))
    text = "\n".join(lines_out) + f"\n\n\nclass {mixin}:\n" + body + "\n"
    text, _needed = x.rewrite_patchables(text, module)
    text = x.ensure_owner_imports(text, module)
    x.write(os.path.join(x.PKG, module + ".py"), text)
    keep = [True] * len(core_lines)
    for first, last in cuts:
        for i in range(first, last + 1):
            keep[i] = False
    new_core = "\n".join(l for l, k in zip(core_lines, keep) if k)
    new_core = re.sub(r"\n{4,}", "\n\n\n", new_core)
    new_core = re.sub(r"(\n    def [^\n]*\n(?:        [^\n]*\n|\n)*?)\n{2,}(    def )", r"\1\n\2", new_core) if False else new_core
    m = re.search(r"^class App(\(([^)]*)\))?:", new_core, flags=re.M)
    bases = [b.strip() for b in (m.group(2) or "").split(",") if b.strip()] + [mixin]
    new_core = new_core.replace(m.group(0), f"class App({', '.join(bases)}):")
    tree = ast.parse(new_core)
    last = 0
    for k, node in enumerate(tree.body):
        if k == 0 and isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            last = node.end_lineno
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Try)):
            last = node.end_lineno
        else:
            break
    cl = new_core.split("\n")
    cl[last:last] = [f"from .{module} import {mixin}"]
    new_core = "\n".join(cl)
    new_core = x.ensure_owner_imports(new_core, "core")
    x.write(x.CORE, new_core)
    x.register_in_init(module)


if __name__ == "__main__":
    main(sys.argv[1:])
