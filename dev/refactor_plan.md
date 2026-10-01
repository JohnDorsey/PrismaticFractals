# Refactor plan: splitting light_bounces.py into a package

Status: **in progress**. Created at FILE_VERSION 113 (light_bounces.py 10,990 lines, 578 tests, suite ~13 s).

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
  there. A test parses kernels.py and fails if an @njit function uses a name not defined in kernels.py (step 3).
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
  (`python test_light_bounces.py`; baseline "578 passed, 0 failed"; the count grows when tests are added). At the end
  of every step also run `python dev/refactor/project_roundtrip_check.py` (needs a display; must print "all checks
  passed"), and update CLAUDE.md.
- R8 Line endings: the working files are LF (git autocrlf=true); write files with scripts in binary mode or the Edit
  tool, and keep `dev/testing/cjxl` (POSIX fake) LF.
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
  (`python dev/refactor/deps_map.py` redraws the dependency picture; its REGIONS go stale after step 1.)
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

### Step 1 - prepare inside the single file. Recommended effort: HIGH
- [ ] Fix back-references 1-6 above in light_bounces.py (small moves, full suite after each); 7-8 need no change yet.
- [ ] Re-point `dev/refactor/deps_map.py` REGIONS at the new line numbers; run it; the only remaining back-references
      should be 7, 8 (kernels/geometry/shapes ordering) and nothing else.
- [ ] Add no behaviour; bump FILE_VERSION; CLAUDE.md pointer to this plan (already added when the plan was created).
- Done when: suite green, roundtrip check passes, progress log updated.

### Step 2 - package skeleton and leaf modules. Recommended effort: HIGH (MEDIUM would be enough for the pure moves)
- [ ] Create `lightbounces/` with `version.py`, `paths.py`, `external.py`, `widgets.py`, `image_io.py` (+ FONT_5X7),
      `__init__.py` re-exporting everything; the launcher `light_bounces.py` keeps `main()` and imports the package.
- [ ] `test_harness.load_module()` loads the whole package fresh (drop `lightbounces*` from sys.modules, re-import) and
      returns an object whose attributes include underscore names; `_ModuleRegistered` and `_load_test_dump` adapted (R9).
- [ ] Re-point the patch sites: EXPORT_DIR/TEMP_DIR -> `m.paths`, ffmpeg -> `m.external`; ImageGrab/DEV_DIR/dump_app_state
      stay on the monolith until step 7 (still read through the module that owns them).
- [ ] Add the R2 guard test (no `from ... import` of a patchable name).
- Done when: suite green, roundtrip check passes, `python light_bounces.py` starts, dumps still load.

### Step 3 - kernels (the numba-sensitive step). Recommended effort: HIGHEST (max)
- [ ] Gather every kernel-ABI constant and the rule/array namedtuples from all over the file into `kernels.py` with all
      @njit functions (R1). Everything else imports them from there.
- [ ] Add the R1 guard test (AST: every name an @njit function loads is defined in kernels.py or is a builtin/numpy/numba name).
- [ ] Compare: golden test (in suite) passes; `kernel_cost.py` warm figure unchanged; touching a non-kernel file does NOT
      make the next run slower (this is the point of the split); `numba_cache_check.py` still prints the staleness (it is a
      fact about numba, not about this code).
- [ ] Update `dev/debug_dump.py`/viewer class paths if needed (R9).
- Done when: suite green, roundtrip check passes, the timing comparison recorded in the progress log.

### Step 4 - constants, settings, records. Recommended effort: HIGH
- [ ] `constants.py`, `settings.py` (strict_record, records, RAY_SETTING_SPECS, resolve_*, Keyframe).
- Done when: suite green, roundtrip check passes.

### Step 5 - geometry and shapes. Recommended effort: HIGH
- [ ] `geometry.py` then `shapes.py` (with make_bound/offset_bound/ring_layer_masks/thumbnails/build_obstacle_arrays).
- [ ] Re-run the characterization idea used for the Shape refactor if wanted: per-shape params/bound/outline/thumbnail
      identical before and after (write a scratch script; the golden test covers the kernels).
- Done when: suite green, roundtrip check passes.

### Step 6 - metrics, views, export text, projects, self-checks. Recommended effort: MEDIUM-HIGH
- [ ] `metrics.py`, `views.py`, `export_text.py`, `projects.py`, `selfchecks.py`.
- Done when: suite green, roundtrip check passes; `dev/testing/project_v1.lightbounce` still loads.

### Step 7 - split `App` into mixins (one or two prompts). Recommended effort: MEDIUM
- [ ] 7a: project/export/video/timeline methods. 7b: obstacle editor, Coloration tab, preview, generation, debug (+ `debug.py`,
      moving dump_app_state/ImageGrab/DEV_DIR and re-pointing their patches).
- Mixins are plain classes with methods using `self.x`; `App.__init__` stays in core.py. No attribute renames.
- Done when: suite green, roundtrip check passes, the window opens and every tab builds.

### Step 8 - tests, documentation, closing. Recommended effort: HIGH
- [ ] Split `test_light_bounces.py` by area into `tests/` (shared helpers in one file, a runner that discovers across
      files and can run one area); full suite must still be one command and must report the same count.
- [ ] Clean-cache run (set a temporary numba cache dir; do not delete the real one), timing comparison with the "Facts" above.
- [ ] Rewrite the Architecture part of CLAUDE.md as the final module map; remove the "refactor in progress" pointer.
- [ ] Finish: see below.

## Finishing
1. Create `dev/finished_plans/` and move this file there (keep the name, add the completion date and the final figures at
   the top). Also move `dev/refactor/deps_map.py` there (its line numbers are history); keep `numba_cache_check.py`,
   `kernel_cost.py` and `project_roundtrip_check.py` in `dev/refactor/` (still useful) and mention them in CLAUDE.md.
2. Remove the pointer to this plan from CLAUDE.md.
3. Tell the user it is done.

## Progress log (update at the end of every prompt)
| Step | Status | FILE_VERSION | Suite | Notes |
|---|---|---|---|---|
| plan | written | 113 | 578 passed | helper scripts in dev/refactor/ verified working |
| 1 | not started | | | |
| 2 | not started | | | |
| 3 | not started | | | |
| 4 | not started | | | |
| 5 | not started | | | |
| 6 | not started | | | |
| 7 | not started | | | |
| 8 | not started | | | |
