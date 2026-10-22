"""Used by the CI workflow: turns the test suite's output (a file) into one GitHub error annotation naming every failing test with the first lines of
its traceback, so a failure can be read from the run's annotations without downloading the log. Prints nothing when nothing failed."""
import re
import sys


def main(path, limit=3500):
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    failing = re.findall(r"^\s+FAIL\s+(\S+)", text, re.M)
    if not failing:
        return
    first = text.index("Traceback") if "Traceback" in text else 0
    message = "failing: " + ", ".join(failing) + "\n" + text[first:first + limit]
    message = message.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
    print(f"::error title=failing tests::{message}")


if __name__ == "__main__":
    main(sys.argv[1])
