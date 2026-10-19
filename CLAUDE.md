# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository: a Tkinter program that shoots rays (or iterates z -> z^p + c) from every pixel and colours what they record. The feature description is the docstring of `PrismaticFractals/__init__.py`; every module has its own docstring. Read those, not this file, to learn how the code works.

## What belongs in this file

- Only what the code cannot tell you: rules a test or tool enforces without explaining, gotchas, decisions and their reasons, things that were measured, policies, and the user's standing preferences.
- **Do not describe how the code works here** (modules, classes, signatures, data flow, algorithms, UI layout): that is the job of docstrings, comments and the package docstring. If something is hard to find in the code, improve the docstring instead.
- No per-change history ("since FILE_VERSION N"): that is `git log`. Keep the file small (the aim is under 20 KB).

## Commands

There is a git repo (the user commits between turns), no build step, no linter.

```bash
python -m compileall -q PrismaticFractals dev PrismaticFractals.py   # compile-check (silent when fine)
python dev/test_PrismaticFractals.py               # full suite (~15 s); custom runner, not pytest
python PrismaticFractals.py                        # launch the Tkinter app
```

- Requirements: `numpy`, `numba` (without it everything is pure Python and unusably slow), `ffmpeg-python` + the `ffmpeg` binary for video, optionally `cjxl` (JPEG XL animations, and stills above 2^28 pixels, which ffmpeg's libjxl cannot encode).
- Folders: `PrismaticFractals/` the program, `PrismaticFractals.py` the launcher, `data/` what the program reads and writes (`projects/` is committed; `exports/` and `debug/` are gitignored), `dev/` everything for development (`tests/`, `refactor/` measurement and check tools, `testing/` fixtures, `tools/` standalone helpers, `finished_plans/` history).
- **Line endings:** working files are CRLF (`core.autocrlf=true`): every `.py`, `.md`, `.cmd`, `.gitignore`. LF stays for `dev/testing/cjxl` (pinned by `.gitattributes`) and the `.lightbounce` project files (the program writes LF). The Edit tool preserves endings; from a script read and write bytes, and scan for lone LF and for CR CR LF afterwards. Never `sed -i`.
- **Numba:** kernels are cached in `__pycache__` (`cache=True`), so the first run after editing a kernel is slow. numba only checks the modification time of the file a kernel is defined in, so a kernel keeps running stale code when a constant it reads, or another `@njit` function it calls, changes in a *different* file (verified: `dev/refactor/numba_cache_check.py`). Hence `kernels.py` defines every `@njit` function that calls another, every constant a kernel reads and the kernel argument types, and imports nothing from the package except `njit`/`prange`; a test (checker: `dev/refactor/kernel_reads.py`) enforces it. A new kernel constant goes in `kernels.py`, never elsewhere.
- Moving the package folder with `git mv` also moves its ignored `__pycache__`, whose numba caches still name the old modules: delete it (`ModuleNotFoundError: No module named '<old name>'` out of a kernel call is that).

## Tests

- Plain `test_*` functions in `dev/tests/test_<area>.py`, run by name: `python dev/test_PrismaticFractals.py <areas>`, `-k text`, `--list`. A test name must be unique across all files. Put a test in the area of what it tests; shared helpers, fake apps and fixtures live in `dev/tests/support.py`, which freezes `__all__` at its end (so helpers go above it).
- The harness (`dev/test_harness.py`) stubs Tk, so only plain functions and classes are testable. Anything needing a live window is checked by the real-window tools in `dev/refactor/`: `project_roundtrip_check.py` after changing project, window or generation code; `engine_switch_check.py` after changing engines, the coloration editor or the bottom-row layout; `undefined_names.py` after moving code. A change that must not alter an image is run through `image_hashes.py` and `project_images.py` against their baselines (in `dev/refactor/baselines/`), and the result quoted.
- Read names through the package (`m.parse_dist_limit`), but patch a global in the module that owns it (`m.paths.EXPORT_DIR = ...`); `from .paths import EXPORT_DIR` is forbidden (it copies the value) and a test scans for it. Tests must patch `EXPORT_DIR`, `TEMP_DIR`, `DEV_DIR` and `DEBUG_DIR` to a temp folder so nothing lands in `data/`.
- Bug fixes get a regression test whose docstring says what went wrong. A test must not set an attribute by hand to get past a `hasattr` guard (a scan test checks the guards).
- Tests written for the old [-1, 1] world keep it through `support.py` (`UNIT_WORLD`, `UNIT_ESCAPE_RADIUS`, `UNIT_PIXEL`, `UNIT_CONFINE`, `_unit_circle`, `_unit_scene`; `_termination()` defaults to them).
- Kernel tests: scenes must start rays *inside* a firing shape (`_kernel_trace` puts a unit-circle figure first), compiled kernels use their own RNG (seed it from compiled code, `_seed_kernel_rng`), and `dev/testing/classic_boundary_golden.npz` (recorded output of the old fixed-bounding-shape implementation) must be kept.
- A test that needs a Buddha option the Mandelbrot engine does not offer uses "All self bounces".
- The cjxl tests are hermetic (`_PathContext` in `test_video.py` strips any real `cjxl` from PATH and puts the fakes in `dev/testing/` first); keep the two fakes (`cjxl`, `cjxl.cmd`) behaving identically.

## Rules that are easy to break

- **Registries must agree.** `run_self_checks()` fails at import when tables drift (settings vs records vs controls, shape parameters, `TERM_*` dicts, point and metric catalogs, project catalogs). Adding a setting means: the record's field, its default, a row in its spec table (the row's `tab` is the tab its control is in, and titles its keyframe section), the control, and its reading in `current_*_settings` / `preview_settings`. Adding a metric, point, Buddha option or any coded choice means adding its project-file key. A self-check failure at import is the intended alarm.
- **Settings records** (`strict_record`) are frozen, keyword-only, with every argument required: a new field must be passed everywhere.
- **Kernel arguments** are namedtuples (numba needs them) read only by field name. A `parallel=True` function cannot use, inside its `prange`, a tuple containing tuples; it must be given the one-level pieces and rebuild the tuples inside the loop (why `trace_rows` is a plain `njit` wrapper over `_trace_rows_parallel`; verified).
- **Path engines.** `trace_engine_path` is the one switch; no Python code calls `trace_path` outside the kernels (a test scans). A choice's id is global; an engine stores some of them at positions of its own, so stored data is read through the figure's layout, never `FULL_LAYOUT`, and an unstored slot raises `KeyError`. A new coloration channel source must list what it reads in `demand_layout` (a random-coloration test compares). The image on screen belongs to the engine it was made with: after a switch, `coloration_fits` makes recolouring skip silently until Start. The pure rules of what an engine accepts are in `engines.py`; `correct_settings` is the one correction. Design decisions D1-D40 are in `dev/finished_plans/Engine_Refactor_Design.md` (D38 revised at 168: the Mandelbrot engine offers the two midpoint options; D5 revised at 170: only Emissions and fuzzing are ignored by it).
- **Two segment length multipliers.** `seg_mult` (key never renamed) is the AFTER one, used by both engines; `orbit_pre_seg_mult` is BEFORE, Mandelbrot only. For the orbit, AFTER only moves the recorded point (z is the engine's whole state, so a carried-on pulled-back point would be BEFORE).
- **Every pixel <-> world conversion goes through the world rectangle** (`WorldRect`); never use a fixed 512-pixel grid.
- **The default world is [-2, 2]** (it was [-1, 1]; the Mandelbrot set reaches -2), and everything measured in world units doubled with it. The size enters a metric in two named constants, `DISTANCE_METRIC_DIVISOR` and `LENGTH_METRIC_DIVISOR`: they are global, so a project file from before draws its distance and path-length channels differently (its geometry and settings are unchanged; format 9 wrote the old values explicitly).
- **What defines an image's figure:** the engine and its settings, termination limits, segment multipliers, fuzzing, emission and the obstacles (Start generates them; Revert restores them). The Buddha blur and the Buddha seeding are live (they recolour what exists); a framing change generates the same figure again, seen by the new camera. None of them needs Start.
- **Exports take a setting no keyframe defines from what the controls held when the export began, never from the program's defaults** (`resolve_frame_settings(..., base)`). Keyframes snapshot everything; float settings interpolate linearly, the rest hold. Keyframe thumbnails and per-keyframe interpolation methods were removed ahead of a planned keyframe overhaul.
- **The figure settings margin** has blocks of fixed height whatever the numbers, so all frames of a video match.
- **Termination-reason metric values are evenly spaced 0..1:** a new reason means rescaling them. `TERM_CONFINED` is also (mis)used for a degenerate zero-length normal and for the pass-through crossing safety cap; splitting them out would re-space the metric.
- **Declined / not built:** circle-radius detection and periodic-orbit detection for confinement; the Buddha "multi" options as sequence x transform; separate parallel drivers per engine (D28, see Measured).
- **Adding a shape kind:** a `Shape` subclass (`KIND`, `NAME`, `PARAMS`, `__slots__`, bound function), a `SHAPE_*` code, an entry in `SHAPE_CLASSES`, and a branch in the three kernels `boundary_hit`, `obstacle_hit`, `inside_boundary` (arc loops need none; a kind numbered after them needs its `inside_boundary` branch placed BEFORE the range check that sends arc-loop codes onward). A name too long for the margin's `Obstacle #N: NAME` line gives a `SHORT_NAME`. The Hügelschäffer egg's equation was transcribed from mathcurve.com's page, whose parametrization as transcribed lacked a leading `cos t` (checked against the equation).
- **Pixel font (`FONT_5X8`, `GLYPH_ROWS` = 8):** the glyphs are Adafruit's classic font (BSD-2-Clause, credited with its full license text in `THIRD_PARTY_NOTICES.md`; do not remove that when touching the table), 5 cells wide, 8 rows tall - the 8th row holds the tails of `, g p q y`, so any code that lays out text must use `GLYPH_ROWS`, never a literal 7. Theta and the multiplication sign are this program's own drawings. Standing permission: whenever a symbol would be useful in drawn text, add the glyph yourself without asking (8 rows of 5 cells in the style of the existing ones, a test, a line in the report; say in the notices if it is not your own drawing). Never reword text to avoid a missing character; `render_text_line` raises `KeyError` on one.
- **Terminology:** the *figure* is the mathematical result of perfectly simulating the rules (it can never be generated whole); every picture the program makes is an *image* (a view of it). Never call an obstacle or an exported picture "the figure".

## Projects and files

- `.lightbounce` files are versioned JSON; every choice in one is a permanent readable string KEY (a `Catalog` per kind of choice), never a number. **A key, once a file can contain it, is never renamed, removed or given another meaning** (`test_the_keys_in_project_files_never_change` pins them). The extension and the format's name stay as they are.
- Older formats are converted one version at a time (`PROJECT_MIGRATIONS`); an old migration spells out the old names of its time, so do not "fix" them. `dev/testing/project_v1..v8.lightbounce` are real files that must keep loading, so what `_sample_project()` in the tests describes must not change.
- `App.apply_project` is the one way a project becomes the program's state (Open and New both use it); the file is fully parsed and validated first and nothing is loaded partially.
- **The program never deletes files on its own;** the one exception is the Video panel's Clear button (`clear_temp_frames`, after confirmation, `frame_NNNNN.png` in `TEMP_DIR` only). A video render refuses to start if `TEMP_DIR` is not empty.
- Output file names derive from the launcher's name (`PROGRAM_NAME`), so renaming the launcher renames them. A JXL name carries the quality, `name.q90.jxl`, and `q100` always means lossless (so lossy stops at 95).
- The package, launcher and test runner share the name `PrismaticFractals` (renamed from `light_bounces` / `lightbounces` at FILE_VERSION 174): from this folder `import PrismaticFractals` finds the package directory, not the launcher. `dev/finished_plans/` is history and still uses the old names; the dump viewer keeps the old module names so old dumps still read.

- The program is licensed GPL-3.0-or-later (`LICENSE`, Copyright (C) 2026 John Dorsey; the header is in the launcher and the package docstring, not in every file). Everything third-party is installed by the user, never vendored; when a dependency or borrowed technique is added, list it in `THIRD_PARTY_NOTICES.md` (with the license, and the source of a borrowed technique in a comment at its place).

## Measured (this machine; the numbers are the reason for decisions)

- Each call added to the parallel kernel's loop body costs first-kernel-call time when it is loaded from the cache (the engine switch about 0.8 s, `measure_path` about 0.4 s); together the two engines cost about 3 s more at every launch (cold 13.9 -> 17 s, from the numba cache 7.8 -> 10.6 s). Separate drivers per engine, chosen from Python, would remove most of it and reverse D28; the user said it is fine for now. Warm run times are unchanged.
- Stored layouts matter for memory: a 3840 x 2160 export peaks at about 0.9 GB with the default coloration (2.9 GB before an export kept only the slots its colorations read), about 1.0 GB with two simple Buddha options, and 2.3 GB with a Buddha channel on all 28 simple points. The live image keeps every slot (320 bytes a pixel).
- Touching any file except `kernels.py` costs no recompile; touching `kernels.py` costs about 6 s once.
- A 25600 x 12800 lossless JXL through `cjxl` takes about 18 s and 2.1 GB.

## Tools that need care

- **Wallcrop (`dev/tools/wallcrop.nim`, standalone Nim, unrelated to the package; the file you edit is the file you run):** needs Nim with a C compiler, the `vips` programs (libvips) and a 64-bit `SDL2.dll`. **It rewrites two lines of its own source (`const LastImage`, `const LastSaveDir`) with absolute paths, including the Windows user name, whenever an image is opened or a folder chosen: never commit those lines with real paths in them.** Tested by `--selftest` (run by `test_wallcrop.py`).
- The dump viewer (`dev/tools/dump_viewer.py`) is standalone (tkinter only) and never imports the package; dumps come from `dev/debug_dump.py`, loaded by path, and go to the gitignored `data/debug/`.

## Working preferences for this project

- After every edit: compile-check, then run the full test suite, not just tests that seem related.
- Flag risk and disagree with a plan when warranted, rather than silently complying or silently deviating. If a request has an unintended side effect, say so and explain the tradeoff before proceeding.
- Keep a visible, incrementing version counter in a long-lived file: `FILE_VERSION` in `PrismaticFractals/version.py`, bumped by exactly 1 once per completed task or turn that changed the code, not per individual edit.
