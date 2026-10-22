"""Tests: package."""
import ast
import os
import sys
import numpy as np

from .support import *      # noqa: F401,F403 - m, ROOT, SkipTest and the shared helpers


# --------------------------------------------------------------------------- #
# The package: modules, re-exports, patchable globals, the launcher
# --------------------------------------------------------------------------- #
PACKAGE_DIR = os.path.join(ROOT, "PrismaticFractals")


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
                assert not bad, f"PrismaticFractals/{module}.py imports {bad} by value"
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in owners
                    and owners[node.id] != module and module != "__init__"):
                raise AssertionError(f"PrismaticFractals/{module}.py line {node.lineno} reads {node.id} directly; "
                                     f"read {owners[node.id]}.{node.id}")
    for name, owner in owners.items():
        assert hasattr(getattr(m, owner), name), f"{owner} does not define {name}"
        assert not hasattr(m, name), f"the package must not re-export {name}: it would be a stale copy after a patch"


def test_the_modules_below_the_window_never_import_tkinter_or_the_window_modules():
    """The window is the app_* modules, core and widgets. Everything else (kernels, settings, engines, projects, the colour
    pipeline, export text...) must stay usable without Tk, so that another front end (or a script) can drive it."""
    window = lambda name: name.startswith("app_") or name in ("core", "widgets")
    for module, tree in _package_sources():
        if window(module) or module == "__init__":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported = [node.module.split(".")[0]]
            elif isinstance(node, ast.ImportFrom):                   # from .x import ...  /  from . import x
                imported = [node.module.split(".")[0]] if node.module else [a.name for a in node.names]
            else:
                continue
            for name in imported:
                assert name != "tkinter" and not window(name), f"PrismaticFractals/{module}.py (line {node.lineno}) imports {name}"


def test_each_engine_description_says_whether_it_owns_the_orbit_settings_and_whether_its_paths_bounce():
    """The window and the figure margin ask the description, not the engine's code (a third engine needs no edit there)."""
    ray, orbit = m.engine_description(m.ENGINE_RAY_BOUNCING), m.engine_description(m.ENGINE_MANDELBROT)
    assert (ray.uses_orbit_settings, ray.path_has_bounces) == (False, True)
    assert (orbit.uses_orbit_settings, orbit.path_has_bounces) == (True, False)


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
                assert hasattr(m, name), f"{module}.{name} is not reachable as PrismaticFractals.{name}"
    assert len(seen) > 400
    for name, modules in seen.items():
        if len(modules) == 1:
            assert getattr(m, name) is getattr(getattr(m, modules[0]), name), name


def test_every_load_of_the_package_is_an_independent_copy():
    first, second = load_module(), load_module()
    assert first is not second and first.core is not second.core and first.EllipseShape is not second.EllipseShape
    first.LuneShape.KEY = "changed"
    assert second.LuneShape.KEY == "lune"
    assert set(first.__lb_modules__) >= {"PrismaticFractals", "PrismaticFractals.core", "PrismaticFractals.paths"}
    assert not [name for name in sys.modules if name.startswith("PrismaticFractals")], \
        "a copy must not stay in sys.modules (the next copy would not start clean)"


def test_the_launcher_exposes_the_programs_names_and_names_the_output_files_after_itself():
    import importlib.util
    path = os.path.join(ROOT, "PrismaticFractals.py")
    spec = importlib.util.spec_from_file_location("launcher_under_test", path)
    launcher = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(launcher)
        package = sys.modules["PrismaticFractals"]
        assert launcher.App is package.core.App and launcher.main is package.core.main
        assert launcher.paths is package.paths and launcher.paths.PROGRAM_NAME == "PrismaticFractals"
        assert launcher.Shape is package.Shape
    finally:
        for name in [n for n in sys.modules if n == "PrismaticFractals" or n.startswith("PrismaticFractals.")]:
            del sys.modules[name]


def test_the_program_folder_is_the_one_with_the_launcher():
    assert m.paths.SCRIPT_DIR == ROOT
    assert os.path.isfile(os.path.join(m.paths.SCRIPT_DIR, "PrismaticFractals.py"))
    assert m.paths.EXPORT_DIR == os.path.join(m.paths.SCRIPT_DIR, "data", "exports")
    assert m.paths.PROJECT_DIR == os.path.join(m.paths.SCRIPT_DIR, "data", "projects")
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
    for fn in (m.fires_ray, m.trace_path, m.trace_rows,
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


def test_a_hasattr_guard_in_the_window_code_names_an_attribute_something_sets():
    """Regression: `refresh_color_controls_for_selected_coloration` guarded itself with `hasattr(self, "mode_var")`, an attribute
    that a rename had left nothing to create, so the guard held for good and the method never did its work (and the one test that
    called it set the attribute by hand). Every attribute a `hasattr(self, "x")` names must be assigned somewhere in the package."""
    import ast
    import glob
    import re
    trees = {}
    for path in glob.glob(os.path.join(PACKAGE_DIR, "*.py")):
        with open(path, encoding="utf-8") as handle:
            trees[path] = ast.parse(handle.read())
    assigned = {node.attr for tree in trees.values() for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)}
    templates = []                                      # setattr(self, f"framing_{field}_var", ...): a name with a hole in it
    for tree in trees.values():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr"
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.JoinedStr)):
                templates.append("".join(re.escape(part.value) if isinstance(part, ast.Constant) else ".+" for part in node.args[1].values))
    guards = [(os.path.basename(path), node.args[1].value) for path, tree in trees.items() for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr"
              and len(node.args) == 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"
              and isinstance(node.args[1], ast.Constant)]
    assert len(guards) >= 8, "the scan should have found the guards it is checking"
    missing = [f"{name}: hasattr(self, {attr!r})" for name, attr in guards
               if attr not in assigned and not any(re.fullmatch(template, attr) for template in templates)]
    assert not missing, "nothing ever assigns " + "; ".join(missing)
