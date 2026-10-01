"""Re-checks the numba fact the module split depends on (run: python dev/refactor/numba_cache_check.py).

numba's on-disk cache (cache=True) only looks at the modification time of the file a kernel is
defined in. So when a CONSTANT a kernel reads, or another @njit function it CALLS, lives in a
different file and is changed there, the kernel keeps running its old cached code. This script
builds two tiny throw-away projects in a temp folder and shows it.

Expected result with the numba this project was developed with (0.6x): both cases print
"STALE (as documented)". If either prints "fresh", numba's behaviour has changed and the rule
in CLAUDE.md ("kernels, the constants they read and the njit functions they call live in one
file") could be relaxed - tell the user.
"""
import os
import subprocess
import sys
import tempfile
import textwrap


def write(folder, name, text):
    with open(os.path.join(folder, name), "w", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(text).lstrip("\n"))


def run(folder, script):
    out = subprocess.run([sys.executable, script], cwd=folder, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    return out.stdout.strip()


def case(title, files, edit, script, before, after_stale, after_fresh):
    with tempfile.TemporaryDirectory() as folder:
        for name, text in files.items():
            write(folder, name, text)
        write(folder, "run.py", script)
        first = run(folder, "run.py")
        assert first == before, (title, first)
        name, text = edit
        write(folder, name, text)
        os.utime(os.path.join(folder, name), None)
        second = run(folder, "run.py")
        verdict = ("STALE (as documented)" if second == after_stale
                   else "fresh" if second == after_fresh else f"unexpected: {second}")
        print(f"{title}: before = {first}, after editing the other file = {second} -> {verdict}")
        return verdict.startswith("STALE")


stale_constant = case(
    "a constant defined in another file",
    {"consts.py": "K = 1\n",
     "kern.py": """
        from numba import njit
        from consts import K

        @njit(cache=True)
        def f(x):
            return x + K
        """},
    ("consts.py", "K = 2\n"), "import kern\nprint(kern.f(10))\n", "11", "11", "12")
stale_callee = case(
    "an @njit function called from another file",
    {"helper.py": """
        from numba import njit

        @njit(cache=True)
        def helper(x):
            return x + 1
        """,
     "caller.py": """
        from numba import njit
        from helper import helper

        @njit(cache=True)
        def caller(x):
            return helper(x) * 10
        """},
    ("helper.py", """
        from numba import njit

        @njit(cache=True)
        def helper(x):
            return x + 2
        """), "import caller\nprint(caller.caller(1))\n", "20", "20", "30")
print("rule still needed:", stale_constant or stale_callee)
