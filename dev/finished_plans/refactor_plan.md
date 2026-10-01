# Refactor plan: splitting light_bounces.py into a package

Status: **FINISHED 2026-10-01** (FILE_VERSION 122). Final figures: light_bounces.py 10,990 lines -> a 23-line launcher + 38 modules in `lightbounces/` (largest: kernels.py 1,390, shapes.py 786, app_timeline.py ~700; core.py 216), tests split into 17 area files (594 tests, 15 s), kernel timing unchanged (warm 4.1 s, cold 6.5 s) and no longer slowed by edits outside kernels.py. Original status line follows:
Status (history):  Created at FILE_VERSION 113 (light_bounces.py 10,990 lines, 578 tests, suite ~13 s).

READ THIS FILE FIRST in every prompt that continues the refactor, and update the progress log (bottom) before
finishing the prompt. When the whole plan is done, **move this file to `dev/finished_plans/`** (create the folder)
and remove the pointer to it from CLAUDE.md - see "Finishing" below. Never delete it.

## Reporting rule (the user asked for this)
At the end of every step's summary, say what was done, what the next step is, and **remind the user which effort
level to set before they tell Claude to continue** (the recommendation is in each step below).

## Goal
CLAUDE.md no longer requires a single-file program. Split the ~11k-line file (and later its tests) into a package
so that (1) work on one area reads/edits a few hundred lines instead of searching 11k, (2) edits to UI/project code
stop invalidating the numba kernel cache (touching light_bounces.py costs a ~2.4 s recompile of two entry points,
more for the whole app), (3) one area's tests can be run alone, (4) the layering becomes enforced by imports.
**No behaviour changes while moving code** (rule R5).

## Decisions (D = decision; "assumed" = Claude's default, the user has not objected - confirm if it matters)
- D1 (assumed): a package directory `lightbounces/`; `light_bounces.py` stays as the thin launcher (`python
  light_bounces.py` keeps working; PROGRAM_NAME - and so output file names - still comes from the launcher's file name).
- D2: the step order below. NOTE it differs from the order first proposed in chat (projects before kernels): while
  writing this plan the dependency check showed the project code needs the metric/point/channel ids (ST_*, POINT_*,
  CHAN_*, KIND_*), which are scattered through the file but are part of the kernel ABI and must live in `kernels.py`
  (R1). So kernels come first and projects later.
- D3 (assumed): the user commits between prompts; Claude does not commit unless told.
- D4: FILE_VERSION moves to `lightbounces/version.py` in step 2 and is re-exported (`m.FILE_VERSION` keeps working).
  It is still bumped by exactly 1 per prompt that changes code.
- D5: `App` is split into mixin classes (files, not a new architecture); `App.__init__` keeps creating all state.
- D6: tests are split last (step 8), after the code. Until then the single `test_light_bounces.py` keeps working.

## Rules (R) - each is either already a test or must become one
- R1 Numba cache. Verified twice (`python dev/refactor/numba_cache_check.py`): numba only looks at the mtime of the
  file a kernel is defined in, so a kernel keeps STALE code when a constant it reads, or an @njit function it calls,
  is changed in another file. Therefore **every @njit function that calls another, every constant a kernel reads
  (74 of them today, defined between lines 323 and 5444), and the kernel argument namedtuples (FuzzRule, ConfineRule,
  EmissionRule, ObstacleArrays) live in ONE file, `lightbounces/kernels.py`**; every other module imports them from
  there. `test_the_numba_kernels_read_only_names_defined_in_their_own_file` (checker: `dev/refactor/kernel_reads.py`) fails
  if an @njit function in ANY file of the package reads a name its own file does not define, or one imported from another
  module of the package, and if kernels.py imports anything from the package but njit/prange (done in step 3).
- R2 Patchable globals are read as module attributes at call time. `from paths import TEMP_DIR` copies the value at
  import and makes a later patch do nothing. Patchable names: EXPORT_DIR, TEMP_DIR, PROJECT_DIR, DEV_DIR, ffmpeg,
  ImageGrab, dump_app_state. They are used only as `paths.X`, `external.X`, `debug.X`. A test scans the package source
  for `from ... import <patchable name>` and fails (step 2).
- R3 Tests read through the package re-export (`m.some_name` keeps working for every public AND underscore name) but
  WRITE (patch) on the owning module: `m.paths.EXPORT_DIR = ...`, `m.external.ffmpeg = ...`. Patching `m.messagebox`,
  `m.filedialog`, `m.ttk`, `m.tk`, `m.subprocess` attributes still works (shared module objects) as long as code calls
  `messagebox.showinfo(...)` through the module, never `from tkinter.messagebox import showinfo`.
- R4 Imports go one way (bottom-up layers, see "Target modules"). No function-level imports to dodge a cycle.
- R5 Moving code changes no logic. Anything that needs a fix is done in step 1 or reported and done in its own prompt.
- R6 FILE_VERSION: +1 per prompt that changes code (D4).
- R7 After every edit: `python -m py_compile` the changed files and run the FULL suite
  (`python test_light_bounces.py`; baseline "586 passed, 0 failed, 0 skipped" after step 3; the count grows when tests are added). At the end
  of every step also run `python dev/refactor/project_roundtrip_check.py` (needs a display; must print "all checks
  passed"), and update CLAUDE.md.
- R8 Line endings. CORRECTED in step 3 (the first wording, "the working files are LF", was an assumption that is false):
  git stores LF (`core.autocrlf=true`), and the working files are a mix - CRLF: `CLAUDE.md`, `.gitignore`, `test_harness.py`,
  `dev/debug_dump.py`, `dev/dump_viewer.py`, `dev/testing/cjxl.cmd`; LF: `test_light_bounces.py`, `light_bounces.py`,
  `lightbounces/*`, `dev/refactor/*`, this file, `dev/testing/cjxl` (must be LF). CLAUDE.md documents "source files use CRLF",
  so the working tree has drifted from it. Verified in the session transcript: `test_light_bounces.py` was already rewritten
  with `newline='\n'` by a script on 2026-09-29, i.e. it was LF before this refactor began (git cannot show that: it stores LF
  either way); the package files are new and written LF by this refactor's tools. Keep each existing file's endings, write new
  files LF, use the Edit tool or binary mode.
  OPEN for the user: normalise the LF source files to CRLF (matches CLAUDE.md and what a fresh Windows checkout gives)?
  If yes it needs its own prompt; `extract_tool.py` writes LF today and would have to learn to keep CRLF first.
- R9 Dumps (`dev/debug_dump.py`, `dev/dump_viewer.py`, `_load_test_dump`) pickle classes by module path
  (`light_bounces.Shape` becomes `lightbounces.shapes.EllipseShape`...) and map `__main__` to the launcher: update them
  in the step that moves the classes (3, 5, 7) and keep the dump tests green.
- R10 The program never deletes files on its own (project rule); Claude's own scratch files go in the session scratch
  folder, repository helper scripts in `dev/refactor/`.

## Facts measured before the refactor (so results can be compared)
- Layout of light_bounces.py (v113): constants 1-594, records 595-959, UI helpers 960-1111, Keyframe 1112-1147, shapes
  1148-1881, settings registry 1882-2182, misc utils 2183-2447, image writers+text 2448-2862, geometry 2863-3371,
  metrics/catalogs 3372-3786, views 3787-3988, projects 3989-4536, self-checks 4537-4686, coloration/info 4687-5006,
  kernels 5007-6230, `App` 6262-10980 (216 methods, 4,719 lines), `main` 10983.
  (`python dev/refactor/deps_map.py` redraws the dependency picture; it finds its regions by anchor names, so it survives moves.)
- Kernel facts: 26 @njit functions in 3 call-connected groups (QOI encoder; Buddha blur; the ray kernels), 74 constants
  read by kernels; no @njit function calls a non-njit function.
- Timing (`python dev/refactor/kernel_cost.py`, run it twice): warm cache ~4.0 s for the first kernel calls; after
  merely `touch`ing light_bounces.py ~6.4 s. Goal after step 3: touching a UI/project file leaves the warm figure.
- Tests: 578, ~13 s. `m.X = ...` patches of module globals: EXPORT_DIR 20, ffmpeg 10, TEMP_DIR 6, dump_app_state 4,
  ImageGrab+DEV_DIR (one helper at the "Points m.DEV_DIR at a scratch directory" context manager). About 21 patch sites in
  17 tests (each patch is an assignment plus a restore). Another 27 assignments patch shared stdlib/Tk objects and need no
  change. 19 `load_module()` uses want a fresh copy; 15 tests use `inspect.getsource`.
- `ruff`/pytest are not used; the runner is `run_all()` in test_light_bounces.py (discovers `test_*` in module globals).

## Back-references to fix in step 1 (an earlier region using a later one; all found by deps_map.py)
True cycles (two regions need each other):
1. shapes <-> geometry: `SHAPE_CLASSES` is used by `make_bound` and `offset_bound` (geometry region) while the shape
   classes call the bound builders. Fix: move `make_bound` and `offset_bound` next to the shapes (after SHAPE_CLASSES).
2. shapes <-> settings: `resolve_obstacles` (shapes region) calls `resolve_value`; `settings_emission` (settings region)
   calls `obstacle_center_from` (shapes region). Fix: `obstacle_center_from` moves to the settings region; `resolve_obstacles`
   and `resolve_obstacle_arrays` move to the settings region after `resolve_value` (they only read fields by name).
Placement problems (one direction, wrong place):
3. `OBST_BOUND_LEN` (defined in the coloration region) used by `build_obstacle_arrays`: move to the kernel-constants area.
4. `DEFAULT_SETTING_VALUES` (after the project code) used by `resolve_setting`: move up beside DEFAULT_RAY_SETTINGS.
5. `MODES` (coloration region) used by `_view_from_json`: move to the views region / constants.
6. `jxl_quality_tag` (image writers) used by `image_filename`: put the two together.
7. `inside_boundary_grid` (a kernel) used by `render_obstacle_thumbnail`/`ring_layer_masks`: just needs kernels to be a lower layer.
8. `inside_boundary`, `boundary_hit` (kernels) used by `estimate_area_numeric`/`sample_boundary_xy` (geometry): same.
Also: `image_io` needs `FONT_5X7` (line 781) and `TEMP_DIR` (line 905): the font moves into image_io, TEMP_DIR into paths.

## Target modules (bottom-up layers; a module imports only from modules above it in this list)
1. `version.py` - FILE_VERSION.
2. `paths.py` - SCRIPT_DIR, EXPORT_DIR, TEMP_DIR, PROJECT_DIR, DEV_DIR, ensure_export_dir, ensure_project_dir.
3. `external.py` - optional dependencies as attributes: ffmpeg, ImageGrab, cjxl_available().
4. `widgets.py` - ModifiedOutline, Tooltip, FloatEntryHandle, make_scrollable_tab, detect_modified_keys, JxlQualityControls.
5. `image_io.py` - PNG/QOI/JXL writers and readers, downsample_image, make_grid, bilinear_resize, FONT_5X7 + text
   rendering, preview_placeholder.
6. `kernels.py` - ALL kernel-ABI constants (SIZE, PIXEL, EPS, TERM_*, SHAPE_*, ST_*, POINT_*, CHAN_*, KIND_*, N_*, RAW_*,
   OBSTACLE_EFFECT_*, LAYOUT_*...), the rule/array namedtuples, every @njit function (incl. QOI encoder and Buddha blur).
   Whether the QOI encoder stays here or in image_io is decided in step 3 by what keeps R1 true (it calls no other kernel,
   so it may stay in image_io with its own constants).
7. `constants.py` - the remaining limits/defaults/UI sizes that no kernel reads.
8. `settings.py` - strict_record, records, RAY_SETTING_SPECS, resolve_* (value/setting/frame/obstacles/views), Keyframe.
9. `geometry.py` - bound builders (ellipse_bound... arbelos_bound), pack_*, region_arcs, estimate_area_numeric...
10. `shapes.py` - Shape and its 12 subclasses, make_bound, offset_bound, ring_layer_masks, thumbnails, build_obstacle_arrays.
11. `metrics.py` - STAT_LABELS, POINT catalogs, Buddha point catalog, make_mapping, point metrics.
12. `views.py` - ViewConfig, ViewSnapshot, ChannelSettings, ViewSettings, resolve_views.
13. `export_text.py` - coloration/figure info text, margins, layout_info_columns.
14. `projects.py` - Catalogs, Project, project_to_dict/from_dict, migrations, EXPORT_OPTIONS.
15. `selfchecks.py` - run_self_checks (called from `__init__` after everything is imported).
16. `app/` - `core.py` (class App, __init__), mixins: `projects_mixin.py`, `export_mixin.py` (saving + video),
    `timeline_mixin.py`, `obstacle_editor_mixin.py`, `coloration_mixin.py`, `preview_mixin.py`, `generation_mixin.py`,
    `debug_mixin.py` (+ `debug.py` holding dump_app_state, take_ui_screenshot).
17. `__init__.py` - re-exports every name (public and underscore) of every module so `m.name` keeps working; runs the self-checks.

## Steps (one prompt each; the effort level is what to set BEFORE telling Claude to continue)

### Step 1 - prepare inside the single file. Recommended effort: HIGH - DONE (FILE_VERSION 114)
- [x] Fixed back-references 1-6 above in light_bounces.py as pure moves (full suite after each; verified afterwards that the
      multiset of non-blank lines is identical before and after). Details: `offset_bound`/`make_bound` now sit right after
      `SHAPE_CLASSES`; `obstacle_center_from` sits before `emission_rule` (a module-level default calls it at import, so it
      must come before `EMISSION_DEFAULT`); `resolve_obstacles` follows `resolve_value`; `resolve_obstacle_arrays` STAYS in the
      shapes region (it needs `build_obstacle_arrays`), so **shapes import settings, never the reverse** (settings is below
      geometry and shapes in the target order); `OBST_BOUND_LEN` is beside `DEFAULT_SIDES`; `DEFAULT_SETTING_VALUES` follows
      `SETTING_WIDGET_ATTRS`; `MODES`/`MODE_NAMES` precede the Views section; `default_filename`/`image_filename` follow
      `preview_placeholder` (after the image writers whose `jxl_quality_tag` they use).
- [x] `dev/refactor/deps_map.py` now cuts regions at named anchor definitions (not line numbers), so it keeps working after
      moves. Its output: no region uses a later one except shapes -> geometry/kernels/settings and geometry -> kernels, which
      are only file-order facts (they point down the target layer order, see "Target modules").
- [x] Behaviour unchanged: 578 passed; `project_roundtrip_check.py` all 15 checks passed.
- Note for step 3: the neighbours of `OBST_BOUND_LEN` (`OBST_OBJ_BASE`, ...) and the rest of the 74 kernel constants still
  have to be gathered into kernels.py.

### Step 2 - package skeleton and leaf modules. Recommended effort: HIGH - DONE (FILE_VERSION 115)
- [x] `lightbounces/`: `version.py`, `paths.py`, `external.py`, `widgets.py`, `image_io.py`, `core.py` (everything else, for now),
      `__init__.py` (re-exports every name of every module, public and underscore, except the patchable globals).
      `light_bounces.py` is now the launcher: it imports the package, re-exports its names (so `import light_bounces as lb` still
      works in scripts), sets `paths.PROGRAM_NAME` from its own file name and runs `main()`. The long feature-description docstring
      moved to `lightbounces/__init__.py`.
- [x] `test_harness.load_module()` imports the whole package afresh and returns it (modules kept in `m.__lb_modules__`, removed from
      sys.modules); `_ModuleRegistered` registers all of a copy's modules; tests read through the package and patch/rebind in the owner.
- [x] Patch sites re-pointed: `m.paths.*` (EXPORT_DIR 20, TEMP_DIR 6, DEV_DIR), `m.external.*` (ffmpeg 10, ImageGrab),
      `m.core.dump_app_state` (4, moves with the debug module in step 7), `fresh.core.*` for the 7 rebinds on fresh copies
      (SETTING_FLOAT_KEYS, POINT_MENU_ORDER, CAT_METRIC, CAT_EFFECT, RAY_CHOICE_CATALOGS) - these move to the owning module in later steps.
- [x] Guard test for rule R2 (+ re-export, independent copies, launcher, program-folder tests): 5 new tests, 583 in all.
- [x] Dumps: `dev/dump_viewer.py` APPLICATION_MODULES knows `lightbounces`; `debug_dump.load_dump` defaults to it; a real startup dump
      reads back with classes `lightbounces.core.*`. `dump_app_state` passes `globals()` (it used `sys.modules[__name__].__dict__`).
- [x] `python light_bounces.py` starts cleanly (25 s, empty stderr); `project_roundtrip_check.py` 15/15.
- Learned: `dev/refactor/extract_tool.py` does the moves (cut, compute imports, refuse a cycle, rewrite patchable reads, register in
  `__init__`); use it for steps 3-7. A heredoc in this environment halves backslashes: write any script containing `\n` or regexes
  with the Write tool, not a heredoc. The names `i`, `label`, `tab` are false "cycle" reports (module-level loop variables in core);
  pass them in `allow_core`.

### Step 3 - kernels (the numba-sensitive step). Recommended effort: HIGHEST (max) - DONE (FILE_VERSION 116)
- [x] `lightbounces/kernels.py` (1,390 lines) holds all 25 @njit functions of the ray/render/Buddha-scatter group (one
      call-connected group), the 74 constants they read plus the rest of each numbered family (TERM_*, SHAPE_* incl.
      SHAPE_FIRST_ARC_LOOP, EMIT_*, CONFINE_*, EPS family, KIND_*, OBSTACLE_EFFECT_*, ST_*/N_STATS, POINT_*/N_POINTS,
      RAW_*/N_RAW, OBST_OBJ_BASE, PARITY_DX/DY, ESCAPE_OBJ, SIZE, PIXEL) and the namedtuples FuzzRule, ConfineRule, EmissionRule,
      ObstacleArrays. It imports only numpy, math, collections and `njit`/`prange` from `external.py` (the numba import and its
      pure-Python fallback now live in `external.py` only; `image_io.py` shares them). core.py imports the 85 names it still uses.
      The QOI encoder stays in `image_io.py`: it calls no other kernel and reads only `np`, so R1 holds there too.
- [x] Guard tests: `test_the_numba_kernels_read_only_names_defined_in_their_own_file` (every file of the package, found by
      walking the folder, so a future `app/` subpackage is covered), `test_the_kernel_rule_check_catches_each_way_of_breaking_the_rule`
      (small broken sources: undefined name, constant imported from another module, callee imported, module imported, an unused
      package import in kernels.py, plus a good source that must pass) and `test_no_name_is_defined_in_two_modules_of_the_package`
      (a moved definition left behind would silently shadow its twin). Also mutation-checked on scratch copies of the REAL
      kernels.py (not kept): importing EPS from another module, renaming a callee away and renaming SHAPE_LENS were all reported.
      Checker: `dev/refactor/kernel_reads.py` (exact, via Python's `symtable`); `python dev/refactor/kernel_reads.py` prints its verdict.
- [x] Compare (all MEASURED, `python dev/refactor/kernel_cost.py`, time of the first kernel calls): cold after editing kernels.py
      6.2 s; warm 3.9-4.0 s (baseline before the refactor 4.0 s); after `touch lightbounces/core.py` 3.9 s; after touching
      `__init__.py`, `widgets.py`, `paths.py`, `external.py` 4.0 s; after `touch lightbounces/kernels.py` 6.3 s (a recompile, once),
      then 4.1 s. Before the split, touching the one big file cost 6.4 s every time - that cost is gone for every file except
      kernels.py. `numba_cache_check.py` still prints "STALE (as documented)" twice (a fact about numba, not about this code).
      Golden-data test passes (it is in the suite). Pure-Python fallback checked by hand with numba blocked: the package imports,
      prints the warning, kernels run as plain functions.
- [x] R9: nothing to change. A real startup dump of the new code reads back (`debug_dump.load_dump`): classes `lightbounces.core.App`,
      `lightbounces.kernels...`, `lightbounces.widgets...`; the viewer accepts any `lightbounces.*` module.
- [x] Housekeeping: a stray comment ("Hints shown under the shape parameter controls") that sat above the RAW_* constants since before
      this refactor was dropped; the empty banners left behind in core.py and the unused imports there (atan, floor, struct,
      subprocess, zlib, njit, prange, JXL_LOSSY_MAX_QUALITY) were removed. CLAUDE.md compile command is now `python -m compileall -q ...`.
- Learned: R8 (line endings) in this plan was wrong - corrected above, with an open question for the user.
- Done: suite 586 passed / 0 failed / 0 skipped (13.9 s warm), `project_roundtrip_check.py` all 15 checks passed.

### Step 4 - constants, settings, records. Recommended effort: HIGH - DONE (FILE_VERSION 117)
- [x] `lightbounces/constants.py` (142 definitions, 258 lines): every plain constant that was between the old "Parameters"
      banner and the record classes (limits, defaults, the Buddha-blur and shape-parameter ranges, UI colours/sizes, TERM_* lookup
      tables, polling periods) - none of them read by a kernel; it imports only kernel names (KIND_*, TERM_*, SIZE, ...). The
      blur *functions* (`world_radius_to_pixels`, `build_buddha_blur_kernel`, `scatter_points_raw`, `blur_counts_image`) stay in
      core.py for now (they are coloration code: step 6).
- [x] `lightbounces/settings.py` (41 definitions, 459 lines): `strict_record`, `FuzzParams`, `BuddhaBlurSettings`, `RaySettings` +
      their defaults, `KeyframeData`, `Keyframe`, `remove_keyframe_keys`, `snapshot_keyframes`, `SettingEntry`, `RaySettingSpec`,
      `RAY_SETTING_SPECS` and everything derived from it, `fuzz_rule`/`confine_rule`/`emission_rule` (+ defaults, `settings_*`,
      `obstacle_center_from`), `resolve_value`/`resolve_obstacles`/`resolve_setting`/`resolve_frame_settings`, `effective_dist`. It imports
      only constants and kernels; no cycle (the tool refuses one). `ObstacleRing` and `resolve_obstacle_arrays` stay with the shapes.
- [x] Done: suite 586 passed / 0 failed / 0 skipped; `project_roundtrip_check.py` all 15 checks passed; touching constants.py,
      settings.py and core.py leaves the first kernel calls at 4.0 s (warm).
- Learned: `extract_tool.py` mistakes local/comprehension variables for names (it added `key`, `label`, `spec`, `tab` to core's import
  and `from . import paths` to settings because of a local list called `paths`): read the generated import blocks, drop such
  names, run the unused-import check (scratch script) after every extraction. A test that rebinds a name on `fresh.core` (e.g.
  `SETTING_FLOAT_KEYS`) works only because the reader (`run_self_checks`) is still in core.py; when step 6 moves the self-checks,
  re-point those 5 rebinds (lines ~59, 4488, 9416-9423 of the test file) to `fresh.selfchecks`.

### Step 5 - geometry and shapes. Recommended effort: HIGH - DONE (FILE_VERSION 119)
- [x] `lightbounces/geometry.py` (474 lines, 25 definitions): the bound builders `ellipse_bound` ... `arbelos_bound`, `pack_arcs_edges`/`pack_circles`,
      `ARC_FIELDS` & co, `region_arcs`, `arc_loop_bound`, `estimate_area_numeric`, `find_interior_point`, `sample_boundary_xy`, `chain_arcs_xy`. Its docstring
      carries the packed-layout description that was a banner in core.py. Imports only constants and kernels.
- [x] `lightbounces/shapes.py` (786 lines, 39 definitions): `ShapeParam` and the shared parameter rows, `Shape` and its 12 subclasses, `SHAPE_CLASSES`/`SHAPE_NAMES`,
      `offset_bound`, `make_bound`, `obstacle_tab_title`, thumbnails, `arc_loop_distance`, `ring_layer_masks`, `draw_obstacle_outline`,
      `default_starting_obstacle`, `build_obstacle_arrays`, `obstacles_signature`, `resolve_obstacle_arrays`, `ObstacleRing`. Imports constants, kernels, settings
      (`resolve_obstacles`, `strict_record`) and geometry; no cycle.
- [x] Characterization (scratch script, not kept; run before and after): for each of the 12 kinds, a default and a moved/rotated variant - params, bound, boundary
      samples, outline, extent, describe/shape_text/info lines, centre, thumbnail, ring masks, drawn outline, offset_bound out/in, make_bound, inside grid - plus
      `build_obstacle_arrays`, `obstacles_signature`, tab titles, and the geometry helpers called directly: 34 digests, ALL IDENTICAL before and after.
- [x] Suite 594 passed / 0 failed / 0 skipped; `project_roundtrip_check.py` all checks passed; touching geometry.py and shapes.py leaves the kernel cache warm (4.1 s);
      a real startup dump with `lightbounces.shapes` classes reads back.
- Learned: `extract_tool.py` again added local variable names (`label`, `field`, `is_float`, `p`) to import lists; `prune_imports.py` (scratch) removed them. That
  pruner is now `dev/refactor/prune_imports.py` (python dev/refactor/prune_imports.py lightbounces/core.py [names]): run it after every extraction.

### Step 6 - metrics, views, export text, projects, self-checks. Recommended effort: MEDIUM-HIGH - DONE (FILE_VERSION 120)
- [x] `metrics.py` (393 lines: metric/point/Buddha catalogs and options, `buddha_compress`, the four Buddha blur functions, `make_mapping`), `views.py` (262:
      the DEFAULT_* channel tables, MODES, `ViewConfig`, `ViewSnapshot`, `ChannelSettings`, `ViewSettings`, `resolve_views`, `hsv_to_rgb_np`/`channels_to_rgb`),
      `export_text.py` (310: option lists, `settings_suffix`, `default_filename`/`image_filename`, coloration/figure info text, layout and drawing, `apply_info_locations`),
      `projects.py` (603: PROJECT_* constants, `Catalog` + every `CAT_*`, `EXPORT_OPTIONS`, `Project`, JSON, migrations 1->2 and 2->3, `project_from_dict`),
      `selfchecks.py` (186: `run_self_checks`). `__init__` calls `selfchecks.run_self_checks()` after all modules are loaded (it used to run when core was imported).
- [x] The 8 test rebinds on `fresh.core.*` now rebind on `fresh.selfchecks.*` (where the checks read the tables); every self-check test still fires.
- [x] Suite 594 passed / 0 failed / 0 skipped; `project_roundtrip_check.py` all checks passed; `project_v1`/`project_v2` files still load (in the suite); touching
      projects.py/views.py leaves the kernel cache warm (4.2 s). No characterization script this time: the moved code is covered by the project, view and export tests.
- Left in core.py (5,200 lines): the keyframe-list text helpers (`DIFF_PRESENCE_CHOICES`, `short_value_text`...), parsers (`parse_*`), window/dump helpers
  (`shift_into_area`, `take_ui_screenshot`, `dump_app_state` ...), `SAMPLE_COLORS`, `default_map_help`, DECO_* tables and `App`. Step 7 decides their homes
  (parsers and keyframe text next to the mixins that use them; the window/dump helpers into `debug.py`).

### Step 7 - split `App` into mixins. Recommended effort: MEDIUM - DONE in one prompt (FILE_VERSION 121)
- [x] First the leftover helpers got homes: `parsing.py` (parse_*), `window.py`, `debug.py` (unused_path, take_ui_screenshot, `dump_app_state` - now a patchable
      global owned by `debug`, read as `debug.dump_app_state`; the 4 tests patch `m.debug.dump_app_state`; it dumps the globals of the module that defines
      `App.__init__`), `ui_text.py`, and SAMPLE_COLORS/DECO_* into constants.py.
- [x] 13 mixins, flat modules (a deviation: the plan said an `app/` folder; flat names `app_*.py` were simpler and the guard tests walk folders anyway):
      app_projects (20 methods), app_export (11), app_render (4), app_video (22), app_timeline (35), app_debug (10), app_generation (5), app_preview (11),
      app_recolor (18), app_obstacles (16), app_coloration (16), app_controls (40), app_layout (9). `core.py` is now 216 lines: `class App(...mixins)`,
      `__init__`, `main()`. Methods were moved unchanged by `dev/refactor/extract_methods.py`.
- [x] Suite 594 passed / 0 failed / 0 skipped after EVERY mixin; `undefined_names.py` reports none; `project_roundtrip_check.py` all checks passed (the real
      window builds every tab); kernel cache warm (4.3 s).
- Not verified by running: video render / ffmpeg paths and the Debug menu items through the real window (covered by the unit tests with fakes and by the
  undefined-names check, not by a click-through).

### Step 8 - tests, documentation, closing. Recommended effort: HIGH - DONE (FILE_VERSION 122)
- [x] `test_light_bounces.py` (10,053 lines) split by `dev/refactor/split_tests.py` into `tests/` : `support.py` (the one loaded package `m`, `ROOT`, `SkipTest`, 31 shared
      helpers/fakes) and 17 area files (coloration 58, debug 31, dump_viewer 83, export 43, images 16, inputs 11, kernels 102, package 9, preview 33, projects 28,
      selfchecks 4, settings 28, shapes 61, timeline 39, video 28, views 18, widgets 2 = 594). Each statement went, with the banner above it, to its section's area; a
      helper used by two areas (and what it uses) went to support. `tests/runner.py` finds tests by name; `test_light_bounces.py` is now a launcher with area, `-k`
      and `--list` options. Verified: the sorted list of 594 test names with their outcomes is IDENTICAL before and after the split.
- [x] `dev/dump_viewer.py` treats classes from `tests.*` as application classes (they used to come from `test_light_bounces`).
- [x] Clean-cache run (`NUMBA_CACHE_DIR` pointed at a temporary scratch folder; the real cache was not touched): first kernel calls 6.5 s cold, 4.1 s warm - the
      same as before the refactor (4.0 / 6.4). Whole suite 15 s either way.
- [x] CLAUDE.md: the refactor paragraph removed, Commands/Testing rewritten for `tests/`, the Architecture layout replaced by the final module map and the dev tools.

## Interlude between steps 4 and 5 (a change the user asked for, FILE_VERSION 118)
`BuddhaBlurSettings` was taken out of `RaySettings`; the two are used side by side (`KeyframeData(ray, buddha_blur, outline)`,
`Project.ray`/`.buddha_blur`, `App.generated_settings`/`generated_blur`, `baseline`/`baseline_blur`, `current_ray_settings()`/
`current_blur_settings()`, `colorize_view(..., blur, settings, fz, obst_arrays)`). Settings code is generic over records:
`SETTING_TABLES` (record class -> its spec table), `settings_to_dict(*records)`, `settings_from_dict(record_class, values)`,
`record_leaf_paths`, `SettingSpec` (was `RaySettingSpec`), `SETTING_CHOICE_CATALOGS` (was `RAY_CHOICE_CATALOGS`). Project format 3 has a
`buddha_blur` section. Effects on the remaining steps: `settings.py` already holds both records (nothing to redo); the
self-check wording changed ("SETTING_TABLES", "the RaySettings fields and the rows of its settings table disagree"), re-point the
`fresh.core.*` rebinds to `fresh.selfchecks` in step 6 as noted there (now also `SETTING_CHOICE_CATALOGS`); the App mixins of step 7
should keep `current_blur_settings`/`blur_of_current_image` with the coloration code, and `current_ray_settings` with generation.

The blur was then made live (same prompt as the start of step 5, FILE_VERSION 119): `generated_blur`/`baseline_blur` are gone, Start and Revert
ignore it, `on_buddha_blur_changed` -> coalesced `refresh_blur` recolors from the stored counts.

## Finishing
1. Create `dev/finished_plans/` and move this file there (keep the name, add the completion date and the final figures at
   the top). Also move `dev/refactor/deps_map.py` there (its line numbers are history); keep `numba_cache_check.py`,
   `kernel_cost.py`, `project_roundtrip_check.py` and `kernel_reads.py` (the guard test loads it) in `dev/refactor/` (still
   useful) and mention them in CLAUDE.md. `extract_tool.py` (moves definitions between modules): keep it there too unless the user
   prefers it archived with this plan.
2. Remove the pointer to this plan from CLAUDE.md.
3. Tell the user it is done.

## Progress log (update at the end of every prompt)
| Step | Status | FILE_VERSION | Suite | Notes |
|---|---|---|---|---|
| plan | written | 113 | 578 passed | helper scripts in dev/refactor/ verified working |
| 1 | done | 114 | 578 passed | 6 pure moves; deps_map anchored on names; roundtrip check 15/15 |
| 2 | done | 115 | 583 passed | package + 5 leaf modules; launcher; harness reloads the package; R2 guard test; extract_tool.py |
| 3 | done | 116 | 586 passed | kernels.py (25 @njit + 74 constants + families + 4 namedtuples); guard test + checker; touching any other file leaves the warm 4.0 s (was 6.4 s), touching kernels.py 6.3 s; roundtrip 15/15; R8 corrected |
| 4 | done | 117 | 586 passed | constants.py (142 defs) + settings.py (41 defs); no cycles; kernel cache stays warm; roundtrip 15/15 |
| (interlude, user request) | done | 118 | 590 passed | (the blur was made live at 119, see below) the Buddha blur is no longer inside RaySettings: its own record beside it in KeyframeData, Project, App state, colorize_view; SETTING_TABLES / settings_to_dict / settings_from_dict; project format 3 (+ migration 2->3, project_v2.lightbounce); roundtrip check 21/21 |
| 5 | done | 119 | 594 passed | geometry.py (25 defs) + shapes.py (39 defs); 34 shape digests identical; roundtrip ok; kernel cache warm |
| 6 | done | 120 | 594 passed | metrics, views, export_text, projects, selfchecks; self-checks run from __init__; test rebinds on fresh.selfchecks; roundtrip ok |
| 7 | done | 121 | 594 passed | helpers (parsing, window, debug, ui_text) + 13 app_* mixins; core.py 216 lines; dump_app_state patchable in debug |
| 8 | done | 122 | 594 passed | tests/ (17 areas + support + runner); identical outcomes; clean-cache timing equal to the baseline; plan finished |
