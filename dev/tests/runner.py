"""Finds and runs the tests (the folder's test_*.py files and support.py); used by dev/test_light_bounces.py.

    python dev/test_light_bounces.py                 every test (what to run after every edit)
    python dev/test_light_bounces.py kernels colorations   only the tests of those areas (dev/tests/test_<area>.py)
    python dev/test_light_bounces.py -k golden       only the tests whose name contains the text (any number of -k)
    python dev/test_light_bounces.py --list          the areas and how many tests each has

A test is a plain function named test_* taking no arguments that raises AssertionError on failure (SkipTest, from support, to skip).
Tests run in name order, so the order does not depend on the file they are in. Exit code 1 if any failed."""
import glob
import importlib
import os
import sys
import traceback

from .support import SkipTest

HERE = os.path.dirname(os.path.abspath(__file__))


def area_names():
    return sorted(os.path.basename(p)[5:-3] for p in glob.glob(os.path.join(HERE, "test_*.py")))


def collect(areas=()):
    """{test name: function} for the given areas (all when empty); a name defined twice is an error."""
    wanted = list(areas) or area_names()
    tests = {}
    for area in wanted:
        if area not in area_names():
            raise SystemExit(f"no such test area: {area} (areas: {', '.join(area_names())})")
        module = importlib.import_module(f"tests.test_{area}")
        for name, fn in vars(module).items():
            if name.startswith("test_") and callable(fn) and getattr(fn, "__module__", None) == module.__name__:
                if name in tests:
                    raise SystemExit(f"test {name} is defined in two files")
                tests[name] = fn
    return tests


def run_all(areas=(), contains=()):
    tests = [(n, f) for n, f in sorted(collect(areas).items()) if not contains or any(c in n for c in contains)]
    passed, failed, skipped = [], [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
            print(f"  PASS  {name}")
        except SkipTest as e:
            skipped.append(name)
            print(f"  SKIP  {name}: {e}")
        except Exception:                          # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=3, file=sys.stdout)
            print()
    print(f"\n{len(passed)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped, {len(tests)} total")
    return not failed


def main(argv):
    if "--list" in argv:
        for area in area_names():
            print(f"{area:14} {len(collect([area])):4} tests")
        return 0
    areas, contains, i = [], [], 0
    while i < len(argv):
        if argv[i] == "-k":
            contains.append(argv[i + 1])
            i += 2
        else:
            areas.append(argv[i])
            i += 1
    return 0 if run_all(areas, contains) else 1
