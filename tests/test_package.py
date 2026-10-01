"""Tests: package."""
import os
import sys
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# The package: modules, re-exports, patchable globals, the launcher
# --------------------------------------------------------------------------- #
PACKAGE_DIR = os.path.join(ROOT, "lightbounces")


def _package_sources():
    import ast
    for filename in sorted(os.listdir(PACKAGE_DIR)):
        if filename.endswith(".py"):
            with open(os.path.join(PACKAGE_DIR, filename), encoding="utf-8") as handle:
                yield filename[:-3], ast.parse(handle.read())


def test_patchable_globals_are_read_as_attributes_of_their_owner_never_imported_by_value():
    """Tests patch these (m.paths.EXPORT_DIR = ...). A `from .paths import EXPORT_DIR` copies the value when
    the module is imported, so a later patch would silently do nothing; every module therefore reads
    `paths.EXPORT_DIR` at call time. Only the owner module may use the bare name."""
    import ast
    owners = m.PATCHABLE_GLOBALS
    assert set(owners.values()) == {"paths", "external", "debug"}
    for module, tree in _package_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                bad = [a.name for a in node.names if a.name in owners and owners[a.name] != module]
                assert not bad, f"lightbounces/{module}.py imports {bad} by value"
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in owners
                    and owners[node.id] != module and module != "__init__"):
                raise AssertionError(f"lightbounces/{module}.py line {node.lineno} reads {node.id} directly; "
                                     f"read {owners[node.id]}.{node.id}")
    for name, owner in owners.items():
        assert hasattr(getattr(m, owner), name), f"{owner} does not define {name}"
        assert not hasattr(m, name), f"the package must not re-export {name}: it would be a stale copy after a patch"


def test_the_package_re_exports_every_name_of_every_module_as_the_same_object():
    import ast
    seen = {}
    for module, tree in _package_sources():
        if module == "__init__":
            continue
        for node in tree.body:
            for name in ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                         [x.id for t in getattr(node, "targets", []) for x in ast.walk(t)
                          if isinstance(x, ast.Name)]):
                if name.startswith("__") or name in m.PATCHABLE_GLOBALS:
                    continue
                seen.setdefault(name, []).append(module)
                assert hasattr(m, name), f"{module}.{name} is not reachable as lightbounces.{name}"
    assert len(seen) > 400
    for name, modules in seen.items():
        if len(modules) == 1:
            assert getattr(m, name) is getattr(getattr(m, modules[0]), name), name


def test_every_load_of_the_package_is_an_independent_copy():
    first, second = load_module(), load_module()
    assert first is not second and first.core is not second.core and first.EllipseShape is not second.EllipseShape
    first.LuneShape.KEY = "changed"
    assert second.LuneShape.KEY == "lune"
    assert set(first.__lb_modules__) >= {"lightbounces", "lightbounces.core", "lightbounces.paths"}
    assert not [name for name in sys.modules if name.startswith("lightbounces")], \
        "a copy must not stay in sys.modules (the next copy would not start clean)"


def test_the_launcher_exposes_the_programs_names_and_names_the_output_files_after_itself():
    import importlib.util
    path = os.path.join(ROOT, "light_bounces.py")
    spec = importlib.util.spec_from_file_location("launcher_under_test", path)
    launcher = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(launcher)
        package = sys.modules["lightbounces"]
        assert launcher.App is package.core.App and launcher.main is package.core.main
        assert launcher.paths is package.paths and launcher.paths.PROGRAM_NAME == "light_bounces"
        assert launcher.Shape is package.Shape
    finally:
        for name in [n for n in sys.modules if n == "lightbounces" or n.startswith("lightbounces.")]:
            del sys.modules[name]


def test_the_program_folder_is_the_one_with_the_launcher():
    assert m.paths.SCRIPT_DIR == ROOT
    assert os.path.isfile(os.path.join(m.paths.SCRIPT_DIR, "light_bounces.py"))
    assert m.paths.EXPORT_DIR == os.path.join(m.paths.SCRIPT_DIR, "exports")
    assert m.paths.TEMP_DIR == os.path.join(m.paths.EXPORT_DIR, "TEMP")


def test_no_name_is_defined_in_two_modules_of_the_package():
    """The package re-exports every module's names, later modules winning, so a second definition of a name in another
    module would silently shadow the first for everything that reads it through the package - and the two copies
    could drift apart (a moved definition left behind in the module it came from is the likely way to get here)."""
    import ast
    homes = {}
    for module, tree in _package_sources():
        if module == "__init__":
            continue
        for node in tree.body:
            names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else \
                [x.id for t in getattr(node, "targets", []) for x in ast.walk(t) if isinstance(x, ast.Name)]
            for name in names:
                homes.setdefault(name, set()).add(module)
    assert len(homes) > 400
    twice = {name: sorted(modules) for name, modules in homes.items() if len(modules) > 1}
    assert not twice, f"defined in more than one module: {twice}"


def _load_dev_tool(*parts):
    """A script from dev/ loaded by its path (the dev tools are not part of the package)."""
    import importlib.util
    path = os.path.join(ROOT, "dev", *parts) + ".py"
    spec = importlib.util.spec_from_file_location("dev_tool_under_test_" + parts[-1], path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_the_numba_kernels_read_only_names_defined_in_their_own_file():
    """numba freezes the value of every global a kernel reads into the compiled code, and when it reuses a cached
    compile it only compares the modification time of the file the kernel is in (verified with
    dev/refactor/numba_cache_check.py). So a constant or another @njit function living in a different file than
    the kernel that uses it would leave the kernel running stale code after an edit there: every file with @njit
    functions must define everything its kernels read, and kernels.py (the kernel ABI) may import nothing from the
    package but njit/prange."""
    reads = _load_dev_tool("refactor", "kernel_reads")
    kernels_found = {}
    for folder, _dirs, files in os.walk(PACKAGE_DIR):
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(folder, filename)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            problems = reads.kernel_rule_violations(source, kernels_file=(filename == "kernels.py"))
            assert not problems, f"{os.path.relpath(path, PACKAGE_DIR)}: " + "; ".join(problems)
            found = len(reads.symtable_global_reads(source))
            if found:
                kernels_found[os.path.relpath(path, PACKAGE_DIR)] = found
    assert kernels_found.get("kernels.py", 0) >= 20, "the scan should have found the ray kernels it protects"
    assert kernels_found.get("image_io.py", 0) >= 1, "...and the QOI encoder"
    assert not (set(kernels_found) - {"kernels.py", "image_io.py"}), \
        f"@njit functions in a new file: {sorted(kernels_found)} - they belong in kernels.py unless they call no other kernel"


def test_the_kernel_rule_check_catches_each_way_of_breaking_the_rule():
    """The check above must not pass just because it looks at nothing: feed it small sources that break the rule in
    each way (and one that keeps it)."""
    reads = _load_dev_tool("refactor", "kernel_reads")
    head = "import numpy as np\nfrom math import sqrt\nfrom .external import njit, prange\n"
    good = head + (
        "LIMIT = 3\n"
        "@njit(cache=True)\n"
        "def inner(x):\n    return sqrt(x) * LIMIT\n"
        "@njit(cache=True)\n"
        "def outer(a):\n    out = np.empty(3)\n    for i in prange(3):\n        out[i] = inner(a)\n    return out\n")
    assert reads.kernel_rule_violations(good, kernels_file=True) == []

    def problems(source, kernels_file=False):
        return reads.kernel_rule_violations(source, kernels_file=kernels_file)

    undefined = head + "@njit(cache=True)\ndef k(x):\n    return x * MISSING\n"
    found = problems(undefined)
    assert len(found) == 1 and found[0].startswith("k reads MISSING")

    constant = "from .constants import LIMIT\n" + head + "@njit(cache=True)\ndef k(x):\n    return x * LIMIT\n"
    found = problems(constant)
    assert len(found) == 1 and "k reads LIMIT, imported from .constants" in found[0]

    callee = "from .helpers import inner\n" + head + "@njit(cache=True)\ndef k(x):\n    return inner(x)\n"
    found = problems(callee)
    assert len(found) == 1 and "k reads inner, imported from .helpers" in found[0]

    module = "from . import paths\n" + head + "@njit(cache=True)\ndef k(x):\n    return x * paths.SCALE\n"
    found = problems(module)
    assert len(found) == 1 and "k reads paths, imported from .paths" in found[0]

    unused_import = "from .constants import UNUSED\n" + good         # fine in an ordinary file...
    assert problems(unused_import) == []
    found = problems(unused_import, kernels_file=True)              # ...but not in the kernel ABI file
    assert len(found) == 1 and "imports UNUSED from .constants" in found[0]

    shadowed = "from .constants import LIMIT\n" + good              # imported AND assigned: the kernel may see either
    found = problems(shadowed)
    assert len(found) == 1 and "inner reads LIMIT, imported from .constants" in found[0]
    assert problems("from . import external\n" + good, kernels_file=True)

    # imports from outside the package, and a name only used outside the kernels, are fine
    assert problems(head + "import json\nOTHER = json.dumps({})\n@njit\ndef k(x):\n    return sqrt(x)\n",
                    kernels_file=True) == []


def test_kernels_read_their_rule_and_obstacle_arguments_by_field_name_never_by_number():
    """numba cannot forbid indexing a namedtuple, so the convention is enforced here:
    in the kernel sources, `fuzz`, `confine`, `emission` and `obstacles` are only
    ever read as `name.field`."""
    import ast
    import inspect
    import textwrap
    checked = 0
    for fn in (m.emits_ray, m.trace_path, m.render_rows, m.render_rows_wh,
               m.scatter_multi_buddha):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn.py_func)))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in ("fuzz", "confine", "emission", "obstacles")):
                raise AssertionError(
                    f"{fn.py_func.__name__} indexes {node.value.id}[...] (line {node.lineno}); "
                    "read it by field name")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in ("fuzz", "confine", "emission", "obstacles"):
                checked += 1
    assert checked >= 20, "the scan should have found the field reads it is protecting"
