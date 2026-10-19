"""Runs the test suite (the tests themselves are in the dev/tests/ folder, one file per area).

    python dev/test_PrismaticFractals.py                 every test
    python dev/test_PrismaticFractals.py kernels colorations   only those areas (dev/tests/test_<area>.py)
    python dev/test_PrismaticFractals.py -k golden       only tests whose name contains the text
    python dev/test_PrismaticFractals.py --list          the areas

One test: (from dev/) python -c "from tests.test_kernels import *; test_name()" (see dev/tests/runner.py for how tests are found)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from tests.runner import main
    sys.exit(main(sys.argv[1:]))
