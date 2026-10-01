"""What the @njit functions of a module read (run: python dev/refactor/kernel_reads.py [path to a .py file]).

For every top-level @njit function it lists, exactly (using Python's symbol tables, not a guess), the GLOBAL names the
function reads, which of them are other @njit functions it calls, and groups the @njit functions into call-connected
components. numba freezes the value of every global a kernel reads into the compiled code and only checks the
modification time of the kernel's own file when it reuses a cached compile (see dev/refactor/numba_cache_check.py), so
every such global, and every function a kernel calls, has to be defined in that same file. Default file:
lightbounces/kernels.py (pass lightbounces/core.py to see what the kernels read before step 3 of dev/refactor_plan.md
moved them).

kernel_rule_violations() turns that into a verdict and is what the guard test in test_light_bounces.py runs on every
file of the package (python dev/refactor/kernel_reads.py also prints its verdict).
"""
import ast
import builtins
import collections
import os
import sys
import symtable

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# The only package names a kernels file may import: numba's decorator and parallel range (decorators and iterators,
# not values a compiled function freezes in). Written the way import_origins() reports them.
ALLOWED_PACKAGE_IMPORTS = {(".external", "njit"), (".external", "prange")}


def is_njit(node):
    return isinstance(node, ast.FunctionDef) and any("njit" in ast.unparse(d) for d in node.decorator_list)


def function_tables(table):
    """Every function symbol table in `table`, nested ones included, as (name, lineno, table)."""
    out = []
    for child in table.get_children():
        if child.get_type() == "function":
            out.append((child.get_name(), child.get_lineno(), child))
        out += function_tables(child)
    return out


def global_reads(table):
    """The global names read by a function table and the functions nested inside it."""
    names = set()
    for symbol in table.get_symbols():
        if symbol.is_global() and symbol.is_referenced():
            names.add(symbol.get_name())
    for child in table.get_children():
        names |= global_reads(child)
    return names


def symtable_global_reads(source):
    """{name of a top-level @njit function: set of global names it reads, builtins removed}."""
    tree = ast.parse(source)
    top = symtable.symtable(source, "<module>", "exec")
    tables = {(name, lineno): t for name, lineno, t in function_tables(top)}
    result = {}
    for node in tree.body:
        if is_njit(node):
            table = tables[(node.name, node.lineno)]
            result[node.name] = {n for n in global_reads(table) if not hasattr(builtins, n)}
    return result


def defined_names(tree):
    """name -> 'function' / 'class' / 'import' / 'assignment' for every top-level binding."""
    kinds = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            kinds[node.name] = "function"
        elif isinstance(node, ast.ClassDef):
            kinds[node.name] = "class"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                kinds[(alias.asname or alias.name).split(".")[0]] = "import"
        else:
            for x in ast.walk(node):
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    kinds.setdefault(x.id, "assignment")
                elif isinstance(x, (ast.Import, ast.ImportFrom)):
                    for alias in x.names:
                        kinds[(alias.asname or alias.name).split(".")[0]] = "import"
                elif isinstance(x, ast.FunctionDef):
                    kinds.setdefault(x.name, "function")
    return kinds


def top_level_statements(tree):
    """Every statement that is not inside a function or class body (those inside a module-level try/if/for count)."""
    pending = list(tree.body)
    while pending:
        node = pending.pop(0)
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for field in ("body", "orelse", "finalbody", "handlers"):
                pending.extend(getattr(node, field, []))


def import_origins(tree):
    """{name a top-level import binds: where it comes from}. A package-relative import keeps its leading dots
    ('.external'); `from . import external` gives '.' + the name ('.external'), as an import of the module itself."""
    origins = {}
    for node in top_level_statements(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                origins[(alias.asname or alias.name).split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                origins[alias.asname or alias.name] = module if node.module else module + alias.name
    return origins


def kernel_rule_violations(source, kernels_file=False):
    """The ways `source` breaks the rule that a file's @njit functions read only what the same file defines
    (see the docstring of lightbounces/kernels.py for why), as a list of sentences; empty means it keeps the rule.
    A kernel may read: a builtin, a name defined or imported in its own file from OUTSIDE the package (numpy, math,
    collections...), or njit/prange from .external. It may not read a name imported from another module of the
    package: numba would freeze that module's current value into the compiled code and only look at this file's
    modification time, so a later edit there leaves the kernel running stale code. `kernels_file=True` is the file
    that holds the kernel ABI: it may import nothing from the package except ALLOWED_PACKAGE_IMPORTS."""
    tree = ast.parse(source)
    reads = symtable_global_reads(source)
    kinds = defined_names(tree)
    origins = import_origins(tree)
    problems = []
    for function, names in sorted(reads.items()):
        for name in sorted(names):
            origin = origins.get(name, "")
            if name not in kinds:
                problems.append(f"{function} reads {name}, which this file neither defines nor imports")
            elif origin.startswith(".") and (origin, name) not in ALLOWED_PACKAGE_IMPORTS:
                problems.append(f"{function} reads {name}, imported from {origin}: numba would freeze its value in and "
                                f"never notice a later change in that file")
    if kernels_file:
        for node in top_level_statements(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                module = "." * node.level + (node.module or "")
                for alias in node.names:
                    here = (module, alias.name) if node.module else (module + alias.name, alias.name)
                    if here not in ALLOWED_PACKAGE_IMPORTS:
                        problems.append(f"the kernels file imports {alias.name} from {module}; it may import only "
                                        f"{sorted(n for _, n in ALLOWED_PACKAGE_IMPORTS)} from the package")
    return problems


def main(path):
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    reads = symtable_global_reads(source)
    kinds = defined_names(tree)
    njit_names = set(reads)
    print(f"{path}: {len(njit_names)} top-level @njit functions")
    problems = kernel_rule_violations(source, kernels_file=os.path.basename(path) == "kernels.py")
    print("rule check:", "ok" if not problems else f"{len(problems)} problem(s)")
    for problem in problems:
        print("  PROBLEM:", problem)
    calls = {name: sorted(r & njit_names) for name, r in reads.items()}
    # call-connected components (undirected)
    parent = {n: n for n in njit_names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for name, callees in calls.items():
        for callee in callees:
            parent[find(name)] = find(callee)
    groups = collections.defaultdict(list)
    for name in njit_names:
        groups[find(name)].append(name)
    print("\ncall-connected groups (each must live in one file with everything it reads):")
    for members in sorted(groups.values(), key=len, reverse=True):
        print(f"  {len(members)}: {sorted(members)}")
    constants = set()
    for name, r in sorted(reads.items()):
        for n in r:
            if n not in njit_names:
                constants.add(n)
    print(f"\nglobal names read that are not @njit functions: {len(constants)}")
    by_kind = collections.defaultdict(list)
    for n in sorted(constants):
        by_kind[kinds.get(n, "NOT DEFINED HERE")].append(n)
    for kind, names in by_kind.items():
        print(f"  {kind} ({len(names)}): {names if kind != 'assignment' else '(listed below)'}")
    # line numbers of the assignments, to see how scattered they are
    where = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        where.setdefault(x.id, node.lineno)
    assignments = sorted(by_kind.get("assignment", []), key=lambda n: where.get(n, 0))
    if assignments:
        print("  assignments by line:", ", ".join(f"{n}:{where.get(n)}" for n in assignments))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "lightbounces", "kernels.py"))
