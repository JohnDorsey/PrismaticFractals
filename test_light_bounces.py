"""Runs the test suite (the tests themselves are in the tests/ folder, one file per area).

    python test_light_bounces.py                 every test
    python test_light_bounces.py kernels views   only those areas (tests/test_<area>.py)
    python test_light_bounces.py -k golden       only tests whose name contains the text
    python test_light_bounces.py --list          the areas

One test: python -c "from tests.test_kernels import *; test_name()" (see tests/runner.py for how tests are found)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from tests.runner import main
    sys.exit(main(sys.argv[1:]))
