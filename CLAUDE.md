# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

No git repo, no build step, no linter. Two source files matter: `light_bounces.py` (the whole app, ~8.6k lines) and `test_light_bounces.py` (+ `test_harness.py`).

```bash
python -m py_compile light_bounces.py      # compile-check
python test_light_bounces.py               # full suite (~8s); custom runner, not pytest
python light_bounces.py                    # launch the Tkinter app
```

- There is no single-test CLI flag. Tests are plain `test_*` functions (no args, raise `AssertionError`) discovered from module globals by `run_all()`. To run one: `python -c "import test_light_bounces as t; t.test_name()"`.
- Requirements: `numpy`, `numba` (without it everything runs in pure Python, unusably slow), `ffmpeg-python` + the `ffmpeg` binary for video, optionally `cjxl` for JPEG XL output.
- Baseline (verified on Windows, 2026-09-29): all 196 pass, none fail or skip, provided `cjxl` and `ffmpeg` are on PATH. Without the real `cjxl`/`ffmpeg`, `test_convert_apng_to_jxl_with_the_real_cjxl_produces_a_jxl_file` reports SKIP (not PASS); everything else still passes.
- cjxl tests are hermetic: `_PathContext` (in the test file) strips any real `cjxl` from PATH, then either leaves none (`_no_cjxl()`) or puts `dev/testing/` first (`_fake_cjxl(fail_on=...)`). The fakes there (`cjxl.cmd` for Windows, `cjxl` for POSIX) copy input to output, or fail if the input path contains `$FAKE_CJXL_FAIL_ON`. Keep the two fakes behaviorally identical.
- `convert_apng_to_jxl` runs the path from `shutil.which("cjxl")`, not the bare name, because on Windows a bare name won't launch `.cmd`/`.bat` shims.
- Source files use CRLF line endings, except `dev/testing/cjxl` (the POSIX fake), which must stay LF or its `#!/bin/bash` line breaks. When editing from a Python script, read and write in binary mode (or `newline=''`) so line endings aren't silently changed.
- Numba caches compiled kernels in `__pycache__` (`cache=True`); first run after editing a kernel is slow.

## Architecture

**Single-file Tkinter app.** `light_bounces.py` shoots a ray from every pixel inside a mirrored bounding shape; rays reflect off the shape, off their own earlier path, and off optional obstacles. Per-ray measurements are stored per pixel, so recoloring is instant without re-simulating. The module docstring (top of file) is the authoritative feature description; read it first.

Layout of the file, in order:
1. Constants/parameters, "Buddha" (density-scatter) blur kernels.
2. Data classes: `Keyframe`, `Obstacle`, `ViewConfig`, `Snapshot`, plus `SETTING_INFO`/`DEFAULT_SETTING_VALUES`/`SETTING_FLOAT_KEYS`/`SHAPE_PARAM_INFO`/`OBST_FIELD_INFO` registries.
3. Pure helpers: keyframe interpolation (`resolve_frame_settings`, `resolve_value`, `resolve_obstacle_arrays`, `resolve_views`), filename/settings text, parsers, hand-written PNG/QOI/JXL encoders and PNG decoder, image utils.
4. Geometry: bounding shapes are built by `make_bound(shape, params)` into packed arrays (arcs+edges, or circles) that numba kernels can consume.
5. Coloration tables/metric catalogs, then `run_self_checks()` (called at import).
6. Numba kernels: `arcs_edges_hit`/`circles_*_hit` → `boundary_hit`/`obstacle_hit` → `trace_path` (one ray) → `compute_stats`/`compute_points` → `render_rows(_wh)` and `scatter_multi_buddha` (parallel, row-banded).
7. `class App` (~4.4k lines, lines ~4200–8580): all UI, generation threads, timeline/keyframes, video render (`_render_worker`, `encode_video_output`).

Key design points that span multiple places:
- **Keyframes snapshot everything** needed to generate an image, independent of the live controls. Float settings are interpolated between keyframes (method from the later one); non-numeric settings hold the earlier value. Dynamic lists (obstacles, views) use keys like `"obst:<id>:<field>"` / `"view:<id>:..."` so each list item's fields animate independently and list position never matters.
- **Parallel registries must stay in sync**, and `run_self_checks()` enforces this at import time (e.g. `SETTING_INFO` vs `DEFAULT_SETTING_VALUES`, `Obstacle.__slots__` vs `OBST_FIELD_INFO`, `SHAPE_PARAM_INFO` vs `Snapshot._fields`, per-`TERM_*` dicts, point-catalog lengths). When adding a setting, shape param, termination code, or metric, update every table; a self-check failure at import is the intended alarm. Tests deliberately break a fresh module copy to confirm each check fires.
- **The program never deletes files**, including its own. A video render refuses to start if `TEMP/` is non-empty or two keyframes share a frame index.
- Output filenames derive from the script's own filename, so renaming the file changes them.

## Testing

`test_harness.load_module()` stubs `tkinter`/`ttk`/dialogs with no-op widgets so the real module imports headlessly and its plain functions/classes can be tested. Each call returns a fresh module object (use it for tests that mutate module state). Anything needing a live Tk mainloop is out of scope and checked by hand. Bug fixes get a regression test whose docstring says what went wrong.

## Working preferences for this project

- After every edit: compile-check, then run the full test suite, not just
  tests that seem related.
- Flag risk and disagree with a plan when warranted, rather than silently
  complying or silently deviating. If a request has an unintended side
  effect, say so and explain the tradeoff before proceeding.
- Keep a visible, incrementing version counter in a long-lived file,
  updated on every delivery. This is `FILE_VERSION` near the top of
  `light_bounces.py`: bump it by exactly 1 once per
  completed task/turn that changed the file, not per individual edit.
