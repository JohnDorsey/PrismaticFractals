"""Remove unused imported names from a module's top-level import statements and rewrap the statement
(python dev/refactor/prune_imports.py PrismaticFractals/core.py [name ...]: also removes the named ones even if used -
use that for the local-variable names extract_tool.py wrongly imports). Run it after every extraction; then compile and run the suite."""
import ast, re, sys, textwrap

path = sys.argv[1]
raw = open(path, "rb").read().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
text = raw.replace("\r\n", "\n")
tree = ast.parse(text)
used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
explicit = set(sys.argv[2:])
lines = text.split("\n")
edits = []
for node in tree.body:
    if isinstance(node, ast.ImportFrom):
        keep = [a for a in node.names if (a.asname or a.name) in used and (a.asname or a.name) not in explicit]
        if len(keep) == len(node.names):
            continue
        if explicit is not None and not explicit and False:
            pass
        edits.append((node, keep))
    elif isinstance(node, ast.Import):
        keep = [a for a in node.names if (a.asname or a.name).split(".")[0] in used and (a.asname or a.name) not in explicit]
        if len(keep) != len(node.names):
            edits.append((node, keep))
removed = []
for node, keep in sorted(edits, key=lambda e: -e[0].lineno):
    for a in node.names:
        if a not in keep:
            removed.append(a.asname or a.name)
    first, last = node.lineno - 1, node.end_lineno - 1
    if not keep:
        new = []
    elif isinstance(node, ast.Import):
        new = ["import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)]
    else:
        module = "." * node.level + (node.module or "")
        names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
        one = f"from {module} import {names}"
        if len(one) <= 100:
            new = [one]
        else:
            body = textwrap.wrap(names, width=96, initial_indent="    ", subsequent_indent="    ")
            new = [f"from {module} import ("] + body + [")"]
    lines[first:last + 1] = new
open(path, "wb").write("\n".join(lines).replace("\n", nl).encode("utf-8"))
print("removed:", sorted(removed))
