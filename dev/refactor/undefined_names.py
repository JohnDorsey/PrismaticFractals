"""Names a module's functions read as globals that the module neither defines nor imports (python dev/refactor/undefined_names.py).

Tests only run some of the program's methods; after moving code between modules this finds the NameError that a rarely used method
would raise. Exact (Python's symbol tables), not a guess. Prints nothing but a summary when every module is clean."""
import builtins
import glob
import os
import symtable
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def walk(table, out):
    for child in table.get_children():
        if child.get_type() == "function":
            out |= {s.get_name() for s in child.get_symbols() if s.is_global() and s.is_referenced()}
        walk(child, out)


def check(path):
    source = open(path, encoding="utf-8").read()
    top = symtable.symtable(source, path, "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    used = set()
    walk(top, used)
    return sorted(n for n in used if n not in defined and not hasattr(builtins, n))


bad = 0
for path in sorted(glob.glob(os.path.join(ROOT, "lightbounces", "**", "*.py"), recursive=True)):
    missing = check(path)
    if missing:
        bad += 1
        print(f"{os.path.relpath(path, ROOT)}: {missing}")
print("undefined global names:", "none" if not bad else f"in {bad} module(s)")
sys.exit(1 if bad else 0)
