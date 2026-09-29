@echo off
rem Fake `cjxl` for test_light_bounces.py (Windows). POSIX counterpart: cjxl.
rem Usage mimics the real tool's `cjxl <input> <output> ...`: copies <input> to
rem <output> and exits 0. If the FAKE_CJXL_FAIL_ON environment variable is set
rem and <input>'s path contains that text, instead prints a message to stderr
rem and exits 1, so tests can exercise the failure path.
setlocal EnableDelayedExpansion
set "input=%~1"
if defined FAKE_CJXL_FAIL_ON (
  if not "!input:%FAKE_CJXL_FAIL_ON%=!"=="!input!" (
    echo fake cjxl: simulated encoder failure 1>&2
    exit /b 1
  )
)
copy /Y "%~1" "%~2" >nul
exit /b 0
