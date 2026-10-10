# Engine refactor plan

The ordered work for the path-engine refactor. What is to be built is decided in `Engine_Refactor_Design.md` (decisions
D1 to D39, cited below as D<n>); this document only orders the work. It holds no decisions of its own: where it says
"assumption" it names something the design document does not settle.

## How the work is done

- One step per delivery, in the order below. Each step ends with a report that says what was verified and what was only
  read, the next step, and its recommended effort. `FILE_VERSION` goes up by 1 once per step that changes code.
- Tests that lock in current behaviour come first; the change second; then the same tests must pass unmodified. A step
  that cannot keep an image bit-identical says so, and says what it compared instead.
- After every edit: compile check, then the full suite. Line endings stay CRLF (`.py`, `.md`); `.lightbounce` files LF.
- The effort levels are the recommended level for doing the step: low, medium, high, xhigh.
- Mandelbrot is NOT offered in the engine dropdown until step 14 (earlier steps reach it only through tests and a
  test-double engine), so that no half-built engine can be selected.
- Assumptions to confirm: (a) the Video panel's Clear message is not an export, so it is appended and does not clear the
  status box (D21 names only a re-encode); (b) every migration added by this plan extends project format 9 in place
  (format 9 is introduced in step 2); a value missing from a file takes its default, so adding a setting needs no new
  format, only a rename or a change of meaning does.

## Measurement tools (step 1 makes them permanent)

The tools used so far live in a scratch folder and are lost with it. Step 1 moves them into `dev/refactor/`:
`hash_scenes.py` (kernel hashes of fixed scenes), `equiv.py` (export images against the last commit), `live_equiv.py`
(the live colouring paths), `big_export.py` (peak memory of a real large export), `undefined_names.py` (already there).
Every step that is meant to leave images unchanged is verified with them, and the numbers are quoted in its report.

---

## Step 1. Lock in today's behaviour and build the scale check  (effort: medium)

Goal: make the doubled world (step 3) provably a pure change of scale for the ray-bouncing engine.

- Move the measurement tools into `dev/refactor/` and record fresh baselines (kernel, export and live hashes).
- Write the scale check: render each baseline scene with every length doubled (shapes, positions, framing, escape radius,
  confinement threshold, distance limit, emission target) and compare with the baseline. Believed, not yet measured:
  doubling by a power of two is exact in floating point except the absolute tolerances `EPS`, `SELF_TIE_EPS` and
  `RETRO_EPS`; the check shows which scenes differ and why. The metrics whose divisors change (D23) are compared after
  accounting for the divisors.
- Tests that use `WORLD_RECT_DEFAULT` or a default framing with unit-sized shapes are listed (they will need an explicit
  old rectangle in step 3).

Done when: the baselines are stored, the scale check runs (and its known exceptions are listed), nothing else changed.
Decisions: none; it prepares D10, D23.

Result (done, no program code changed, so `FILE_VERSION` stays 151):
- Tools in `dev/refactor/`: `image_hashes.py` (kernel 32, export 20, live 45 hashes), `project_images.py` (the 13 files of
  `projects/` and `dev/testing/project_v*.lightbounce`, 38 pictures), `scale_check.py`, `big_export.py`. Baselines are in
  `dev/refactor/baselines/` and re-run identical; `image_hashes.py` reproduces every hash recorded before it existed.
- The scale check (136 scenes: every shape kind but the snail, default and mid-range parameters, segment multipliers 1.0
  and 0.7, fuzzing, emission Towards): at scale 2, 119,507 of 119,632 traced pixels are exactly twice as large, 80 agree to
  rounding, 45 diverge (0.038%), 36 of them where a ray runs back along its own path; no metric differs on an agreeing
  pixel. Cause of the divergences, read from `trace_path` and confirmed on one ray: absolute tolerances (the `1e-14` test
  for a segment parallel to an earlier one; `EPS`; `SELF_TIE_EPS`) decide ties differently when lengths are doubled, and
  root searches (squircles, egg, spiral) stop at absolute tolerances. Consequence for step 3: files that keep their own
  numbers (step 2 pins them) are bit-identical; the NEW default scene (the radius-2 circle at 45 degrees, which retraces
  its own path) will differ from today's default picture in a few percent of its pixels (15 of 448 traced pixels in the
  small circle tested) - the pixels where which of two coinciding hits wins is decided by rounding.
- Tests that will need an explicit old rectangle or values in step 3 (counts of lines mentioning them): `WORLD_RECT_DEFAULT`
  in `test_kernels.py` (15), `test_coloration.py` (12), `test_framings.py` (7); `DEFAULT_PIXEL_WORLD_WIDTH` in
  `test_kernels.py` (12), `test_framings.py` (2), `test_shapes.py` (3); `DEFAULT_ESCAPE_RADIUS` in `test_kernels.py` (6);
  `DEFAULT_FRAMING` in `test_framings.py` (29) and six other files; `_start_scene` / `default_starting_obstacle` users in
  `test_export.py` (10), `test_kernels.py` (15+4), `test_framings.py` (7), `test_preview.py` (4), `test_coloration.py` (3);
  the literal zone `(-1.0, 1.0, -1.0, 1.0)` of the seeding tests in `test_kernels.py`.

## Step 2. Project format 9: pin the old values, rename the top-left key, add the snail radius  (effort: high)

Goal: make every project file say explicitly what it means, BEFORE any default changes, so that step 3 cannot alter an
old file's image. Defaults are not changed in this step; images are bit-identical.

- Format 9 and `_migrate_project_8_to_9` (D33): write explicitly into a file wherever a missing value would take a
  default that D10 or D23 changes: the framings section (the old default framing), `escape_radius` (2.0), the
  confinement threshold (`2/512`), a Buddha seeding zone (-1, 1, -1, 1), and every obstacle value that has a doubled
  default. Keyframes that hold values are left as they are.
- D24: the point choice "Top left corner" becomes the key "(-1,1)" (its label in the dropdowns is "(-1,1)"), in
  colorations and in keyframe values. The migrations of older formats keep the old key.
- D25: the snail of Pythagoras gets a `radius` (shared field name; for now default 1, the size it has); a file with no
  snail radius loads it as 1. `snail_bound` takes the radius.
- Pin the new keys in `test_the_keys_in_project_files_never_change`; add a test per migration (a file of each older
  format, including `dev/testing/project_v1` to `v7.lightbounce` and the six in `projects/`, loads and renders the same
  image as before).

Done when: all fixtures and projects render the same images before and after; the migration tests pass.
Decisions: D24 (first half), D25 (first half), D33. Risk: keyframe values hold point choices, so the key rename must be
checked in keyframes too.

Result (done; `FILE_VERSION` 152; 758 tests pass, 9 new):
- `_migrate_project_8_to_9` and the `_FORMAT_8_*` tables in `lightbounces/projects.py`; `dev/testing/project_v8.lightbounce`
  is a new real format-8 fixture. All 14 files of `projects/` and `dev/testing/` render the same pictures
  (`project_images.py`), and `image_hashes.py` and `scale_check.py` are unchanged.
- Beyond what the plan listed: (a) a video used to take a setting no keyframe defines (escape radius, confinement threshold, the Buddha seeding zone...) from the program's DEFAULTS, so the first version of the migration pinned those into the first keyframe. That was replaced in FILE_VERSION 153 by D40 (an export takes them from the live values when it began; `resolve_frame_settings(keyframes, frame, base)`), so the migration no longer touches keyframes, apart from the renamed point. The pictures of `kf<frame>` renders of the real projects changed where their live values differ from the defaults (baseline `projects_FILE_VERSION_153.json`); the live pictures did not. (b) `PROJECT_KEY_PATTERN` now also admits the coordinates of a
  fixed point, `(-1,1)`. (c) `CHOICE_TOP_LEFT_CORNER` is renamed `CHOICE_POINT_NEG1_POS1` (label, key and short name
  "(-1,1)"), so that step 3's "Top left point" can have its own name. (d) The snail's `radius` is the shared field (range
  0 to the shared maximum, default 1 for now); at 1 its bound is bit-identical to before.
- Verified that the migration makes a file independent of the defaults: `test_a_migrated_file_does_not_depend_on_the_defaults_of_the_program` changes the defaults the way step 3 will (escape radius, confinement threshold, seeding zone, the framing's diagonal) and checks that every old file loads to the same project text and resolves the same settings at every keyframe (with the project's own values as the base); it fails if any of the framing, termination or seeding pins is removed (tried one by one). The obstacle pins are checked by literal values, because the class defaults will change in step 3.
- Not run: `dev/refactor/project_roundtrip_check.py` (needs a real window); the new snail control in the obstacle editor
  is built from the shape's `PARAMS` like the other shapes' (the self-checks pass), but I have not looked at it in the window.

## Step 3. The doubled world  (effort: high)

Goal: apply D10, D23 and the second half of D24 and D25. Ray-bouncing images of the migrated files stay the same.

- Kernel constants `WORLD_RECT_DEFAULT` = (-2, 2, -2, 2) and `DEFAULT_PIXEL_WORLD_WIDTH` = 4/512 (touching `kernels.py`
  recompiles; the numba cache rule stays).
- The distance divisors become ONE named constant in `kernels.py` (point distances divide by 4, the length is
  `atan(length / 10)`), read by the kernel (`compute_run_measurements`, `sequence_summary`) and by `metrics.py`; a
  self-check keeps the copies in step; the help texts that say "divided by 2" and "length / 5" are updated.
- Defaults and maxima (D10): framing 0, the Buddha seeding zone, the default escape radius 4.0, the bar graph
  truncation at 2 (D37), the default and maximum of every parameter D10 lists (ellipse and superellipse width and
  height, radius, lens radii and separation, capsule length and radius, lune cut radius and offset, egg half-length and
  half-width, Moss egg size, spiral pitch), the maxima of `r`, `cx`, `cy`, and the minimum of `cx` and `cy` (-2). A new
  snail's default radius is 2 (D25). The starting obstacle is the circle of radius 2.
- D24: the new point choice "Top left point", key "(-2,2)".
- `default_project`, thumbnails and every place that assumed a [-1, 1] world are found by search and checked
  (`image_io.make_grid`'s default rectangle, the corkscrew and bar graph units, the ring layer pixel).
- Tests that relied on the old default rectangle get an explicit one; the classic golden data stays valid with it.

Done when: the scale check passes; migrated fixtures and projects are image-identical; a fresh start shows the radius-2
circle filling the (-2, 2) image exactly as the unit circle filled the old one; the full suite passes.
Decisions: D10, D23, D24, D25, D37. Risks: the long list of constants (a missed one shows as a wrong default, which the
scale check and the fixtures catch); the absolute tolerances.

Result (done; `FILE_VERSION` 154; 769 tests pass, 6 new; `Rendering_For_Humans.md` no longer exists, the user deleted it):
- Everything D10, D23, D24 and D37 list is changed (`CLAUDE.md`, "The default world is [-2, 2]", has the list); the two
  metric divisors are the named constants `DISTANCE_METRIC_DIVISOR` (4) and `LENGTH_METRIC_DIVISOR` (10) in `kernels.py`,
  imported by `metrics.py`; the new fixed point is `CHOICE_TOP_LEFT_POINT`, "Top left point", key "(-2,2)", listed second.
- Measured: `scale_check.py` (now with the snail, 144 scenes): 99.985% of 251,000 traced pixels exactly twice as large, the
  rest at ray tie-breaks. `fresh_start_check.py`: a fresh start in the new world differs from the old start in 0.04-0.14% of
  its drawn pixels (not the 3% the small circle of step 1 suggested). `image_hashes.py` against `hashes_step3_before.json`
  (scenes written out in the old world, distance and length columns left out): identical.
- The recorded golden data (the old fixed-shape implementation's output) still matches, its distance and length columns
  converted by the new divisors.
- NOT as the plan hoped: because the divisors are global, 12 of the 14 files in `projects/` and `dev/testing` draw their
  distance (and, in `lune_experiments`, length) channels differently now - a quarter as bright for a distance - while their
  geometry and settings are exactly as before (restoring the old distance divisor in Python gives back every one of those
  pictures but the length channel's). Baseline `projects_FILE_VERSION_154.json` is recorded after the change.
- Tests written for the old world keep it through `tests/support.py` (`UNIT_WORLD`, `UNIT_ESCAPE_RADIUS`, `UNIT_PIXEL`,
  `UNIT_CONFINE`, `_unit_circle`, `_unit_scene`, `SAMPLE_SIZES`); the sample project spells out its sizes.

## Step 4. Buddha termination-kind filter and "All path points" (Ray Bouncing)  (effort: medium)

Goal: D16 and D38 for the ray-bouncing engine, independent of Mandelbrot. With all three boxes ticked, images are
bit-identical.

- A per-channel filter, three checkboxes: escaping, non-escaping, terminated by obstacle (D16: a perpendicular
  collision, the iteration limit, the distance limit and confinement are non-escaping). Its project keys; a file
  without it loads all three ticked. It is animated like the other channel settings. The existing options "Path: Last
  point (distance limit / perpendicular collision / confined)" stay.
- `demand_layout` keeps `ST_TERMINATION` whenever a filter is not all three; stored-point and multi-option scatters
  both apply it (a multi option knows the reason from its own trace).
- The new multi option "All path points" (start included), generated like the other multi options.
- Tests: a property test (random filters) that the filtered scatter equals a brute-force scatter of the paths of the
  ticked kinds; the existing `test_a_demand_layout_keeps_everything_the_colour_chain_reads` is extended.

Done when: hashes identical with the filter all ticked; the property test passes. Decisions: D16, D38 (ray-bouncing
half).

Result (done; `FILE_VERSION` 155; 778 tests pass, 9 new):
- A channel has `buddha_path_kinds` (three flags; `CAT_PATH_KIND` keys; default all ticked; a file without it loads all
  ticked; saved for Buddha channels only; carried by the keyframe's channel record) with three checkboxes ("Paths:") in its
  Buddha controls; "All path points" is the new multi option (catalog entry 36, the last, so no old index moved).
- A perpendicular collision is non-escaping (the redefinition recorded in D16). A path of an unticked kind scatters nothing,
  for the stored options (masked by the stored termination metric, which `demand_layout` keeps for a limited channel) and for
  the kernels (`term_allowed`, one flag per `TERM_*` code).
- Verified: with all three ticked, `image_hashes.py` and `project_images.py` are identical to their baselines. The property
  test runs every set of ticked kinds x three options (a stored point, all path points, all midpoints) x both kinds of seeding
  against a reference written from `trace_path` alone, on a scene that has escaping, obstacle-terminated, iteration-limit and
  perpendicular endings; removing either the flags or the stored mask makes it fail (tried).
- Not looked at: the three new checkboxes in the window (built like the point checkboxes; the layout row of the compression
  and brightness controls moved down one).

## Step 5. The engine as a setting, with one engine  (effort: medium)

Goal: the choice of engine exists end to end, with only Ray Bouncing available.

- `CAT_ENGINE` ("ray_bouncing", "mandelbrot"), an engine record and table in `SETTING_TABLES`, the Simulation tab's
  dropdown (with only "Ray Bouncing" offered for now), `GeneratedFigure.engine`, the keyframe key, the project section,
  and the figure settings margin line. The engine is figure-defining (D8: a change needs Start; it is in the Revert
  baseline and Start's comparison). New settings stay out of file names (D32).
- Self-checks keep the new table in step with the record.

Done when: save, New, Open and Start round-trip the engine; images identical. Decisions: D6, D8, D32.

Result (done; `FILE_VERSION` 156; 786 tests pass, 8 new): `EngineSettings`, `ENGINE_SETTING_SPECS`, `CAT_ENGINE`, the Simulation
tab's dropdown (Ray Bouncing only; a project that says Mandelbrot is refused until step 14 offers it),
`GeneratedFigure.engine`, `baseline_engine`, the keyframe key `engine`, the project section `engine` and the `Engine:` block
of the figure settings margin. The pair `(ray, natural_termination)` that Start, `settings_of` and the export calls pass
around is unchanged - the engine travels beside it, as the blur and the seeding do - so no file name and no existing call
changed. Images identical (`image_hashes.py`, `project_images.py`). Not looked at: the dropdown in the window.

## Step 6. The compatibility core, tested with a test-double engine  (effort: high)

Goal: the rules of D7, D11, D17, D34, D35 and D39 as pure functions, before any interface uses them.

- Global choice ids (D34): the catalogs of metrics, points and Buddha options become the union of what the engines
  offer, each engine offering a subset; `StoredLayout` is the id-to-position map of an engine. For the ray engine the ids
  do not change.
- The engine description: which choices it offers, the defaults it uses where the usual default is not offered (D39),
  the ranges it accepts (clamping, D17), which settings it ignores (for the greyed state, D13).
- ONE correction function (D35): given settings and an engine it returns the corrected settings and a list of
  (tab, setting, old, new); a dropdown choice falls back to the engine's default, a number is clamped. Every place a
  render starts will call it (wired in later steps).
- Engine memory (D18): what was reset, and which settings the user or a keyframe has changed since, kept for the
  session; "go back" is its inverse.
- A test-double engine that offers a small subset and has different ranges stands in for Mandelbrot in the tests
  (including a number setting with a range that differs between engines, as D17 asks).

Done when: the pure functions are fully tested, including round trips of switch, switch back and go back. Decisions:
D7, D11, D17, D18, D34, D35, D39. Risk: the biggest design piece; ids and defaults touch every catalog.

Result (done; `FILE_VERSION` 157; 810 tests pass, 24 new): `lightbounces/engines.py`, pure, nothing calls it yet.
- Global ids (D34) needed no change: Mandelbrot's choices are a subset of the existing catalogs (its "Path" sequence and metrics
  are already there), so an `EngineDescription` holds sets of ids and the ray engine's offers all of them. `StoredLayout` as an
  engine's own id-to-position map is step 8 (it needs the engine's tables passed in).
- `EngineDescription` (offered metrics, points, Buddha options; accepted values of other choices by key pattern; ranges; the
  defaults used where the usual one is not offered; the tabs and keys it ignores), `correct_settings` (the one correction),
  `correct_channel`, `Correction`, `correction_message` (the wording of D7, D20, D36), `switch_engine` with `EngineMemory`, `go_back`,
  `new_obstacle_effects` (D35). The test double `small_engine()` in `tests/support.py` has the shape of D31 and a range for the
  iteration limit and the segment multiplier (ranges that differ between two doubles are tested).
- Verified: round trips (random settings, 300 each: switch and switch back gives the same settings; correcting twice changes
  nothing; correcting under the ray engine changes nothing), switch + go back, a setting changed meanwhile stays, four deliberate
  breakages of the code each make a test fail (tried). Hashes and project pictures identical.
- Interpretations to confirm are listed in the report of this step.

## Step 7. The engine switch inside the drivers  (effort: high)

Goal: D28 with only the ray engine, so the cost is measured before a second engine exists.

- `trace_rows_stored`, `scatter_from_seeds` and every Python caller of `trace_path` (the hover preview, the warm-up)
  gain the engine, and a per-path switch calls the engine's path function; both engines' argument bundles travel
  together (the unused one a dummy), and the "one-level pieces" rule of the parallel driver is kept.
- Measure first-call compile time (`dev/refactor/kernel_cost.py`), warm run time against the baseline, and the numba
  cache behaviour after touching `kernels.py`.

Done when: hashes identical; run time within noise of the baseline; the compile time is reported. Decisions: D2, D28.
Risk: numba compile time grows with every branch; if it is much worse than expected, stop and report before step 8.

Result (done; `FILE_VERSION` 158; 815 tests pass, 5 new): the four drivers take a `PathArgs(engine, ray, orbit)` as `engine_args` and call
`trace_engine_path`, the one switch (`trace_path` for the ray engine, `orbit_path` for the other); the hover preview, the warm-up
and the drivers all go through it, and a test scans the modules so that nothing else calls `trace_path`. `orbit_path` is a stub (a
path of its start point) until step 9, so the second branch is real but small. `OrbitArgs` has the fields of D14 and D29.
- Images identical (`image_hashes.py`, `project_images.py`); the ray engine through the switch gives exactly `trace_path`'s paths.
- Measured against a copy of the package from before the change (`dev/refactor/driver_bench.py`, alternating processes): warm run
  times within noise (e.g. 512 x 512 start scene 0.095-0.100 s both ways); first kernel calls cold 15.1-15.2 s against 13.9 s
  (+9%), from the cache 8.4 s against 7.8 s (+7%); touching another file recompiles nothing, touching `kernels.py` recompiles
  (15.4 s). The real orbit function (step 9) adds its own compile time and is re-measured there.
- Obstacles still live inside the `RayArgs` (the orbit function is handed `engine_args.ray.obstacles`); taking them out of it would
  change `trace_path`'s signature and every test that calls it, and nothing needs it yet. A decision for step 9, if wanted.

## Step 8. Per-engine layouts and sequence tables  (effort: high)

Goal: D3. The ray engine's tables are passed in instead of read as globals, so another engine can bring its own.

- `sequence_summary`, `compute_sequences` and `in_sequence` take the engine's slot tables; the stat and point counts
  are the engine's; `StoredLayout` is built from the engine's layout and the coloration's demand.
- The golden test and the hashes must be identical.

Done when: identical hashes and memory numbers (the quoted 860 MB default export is re-measured). Decisions: D3, D34.

Result (done; `FILE_VERSION` 159; 822 tests pass, 7 new): `SequenceTables` (the id -> position maps of an engine's metrics and
characteristic points, and its sequences) travel in `PathArgs.tables`; `sequence_summary` / `compute_sequences` write through them, the
drivers size their per-path arrays by them, and `measure_path` is the third engine switch. `in_sequence` was NOT given tables: the
sequences' ids are global like the metrics' (D34), so its rule by tag stays one function and an engine lists the sequences it has.
`StoredLayout` is the id -> position map of an engine (`engine_layout(description)`, `GeneratedFigure.layout`); an export's kept layout is
mapped onto the engine's positions (`stat_sources` / `point_sources`).
- Verified: hashes and project pictures identical; the 3840 x 2160 default export has the same image hash and 874 -> 880 MB peak
  (985 -> 988 MB with two simple Buddha options); the golden test passes; the test double's tables are exercised by the reductions,
  the drivers (the orbit branch) and the seeded Buddha scatter; four deliberate breakages each make a test fail (tried).
- Measured (alternating old/new): warm run times within noise; first kernel calls cold 15.1 -> 15.5 s (+3%), from the cache 8.5 -> 9.3 s (+10%).
  The first `trace_rows` call from the cache (the parallel kernel is reloaded) went 5.8 s (before step 7) -> 6.7 s (step 7) -> 7.5 s (step 8):
  about 1.7 s of start-up time in all, of which the engine switch is ~0.8 s and `measure_path` ~0.4 s (attributed by removing each in turn).
  Step 9's real orbit function will add more. If start-up time matters, the switch could move out of the parallel loop (one parallel
  driver per engine behind the same wrapper), which D28 puts the other way round; say if you want that looked at.

## Step 9. The Mandelbrot path function and its tests  (effort: high)

Goal: the engine itself, reachable only from tests.

- FIRST, decide (the user asked for this reminder at step 7): do the obstacles stay inside the ray engine's `RayArgs`
  (`engine_args.ray.obstacles`, which `orbit_path` is handed) or move out into the `PathArgs` beside the two engines' bundles?
  Moving them changes `trace_path`'s signature and every test that calls it.

- A path function in `kernels.py` (D2): the orbit of z_0 under z -> z^p + c, or the Julia form; p any real number
  (integers by multiplication; a non-finite value ends the path as an escape, D26); z_0 chosen between 0 and c;
  the pixel's location becomes c (Mandelbrot) or z_0 (Julia); the shared escape and confinement checks and the
  distance limit; the escape behaviour of D29 (clip default); a new point kind for orbit points.
- Obstacles through the existing machinery (D12): each segment z_k to z_(k+1) is tested, terminate and pass are
  honoured, and an incompatible Reflect terminates the path (D11, last resort). Gating by the pixel's location,
  flagging at the first point (D30).
- Its own sequence ("Path") and metrics (D31) built on step 8's tables.
- Tests (D26): known counts at known points (with the escape radius 2 set explicitly: c = 0 never escapes, c = 1 escapes at
  the third point, c = 2 and c = -2 at the second (the test is `>=`; this list first said c = 2 escapes at the third, which is
  wrong), c = -1 cycles to the limit), a Julia case, a real-exponent case, the clip behaviour, an obstacle that terminates on a
  segment, and an estimate of the set's area (about 1.5066).

Done when: those tests pass and the ray engine's hashes are untouched. Decisions: D12, D26, D28, D29, D30, D31.

Result (done; `FILE_VERSION` 160; 847 tests pass, 25 new): the obstacles decision was made first (they leave `RayArgs`; `PathArgs`
has an `obstacles` field and `trace_path` takes them as an argument; hashes identical). `orbit_path` is the real function (see
CLAUDE.md, "The orbit engine"): the orbit of z -> z^p + c or the Julia form, a whole exponent by multiplication and another by the
principal value, non-finite points ending the path as an escape at the last finite point, the shared iteration, distance, escape
(`>=`) and confinement ends, clip or no clip, every segment tested against the obstacles (terminate and pass honoured, reflect
terminating as the last resort), the path beginning at the origin for z_0 = 0 while the driver gates at the pixel (D30), a new
point tag `TAG_ORBIT`, its measurements (`compute_orbit_measurements`) and its Path sequence through step 8's tables.
- Done here although the plan puts it in step 11: the Mandelbrot `EngineDescription` (`engines.MANDELBROT_ENGINE`), because the
  tables need its layout - the metrics, points and Buddha options of D31 and D38, the defaults of D39, no Reflect and pass-through
  for a new obstacle (D11, D35), Emissions and Mutations ignored (D5). Step 11 keeps: the controls, the switch's popup, the hover
  preview and its legend for an orbit, and the real-engine runs of the step 6 tests (two now exist).
- Verified: the ray engine's hashes and project pictures identical; four deliberate breakages of the orbit function (flags taken at the
  pixel, segments untested, no clip, the pass-through flip forgotten) each make tests fail (tried); the set's area from the drivers
  is 1.5249 (500 x 500, 200 iterations) and 1.5079 (1000 x 1000, 2000 iterations) against 1.50659.
- Measured: warm run times unchanged; first kernel calls cold 15.5 -> 17.0 s, from the cache 9.3 -> 10.6 s. Since before step 7 the
  start-up cost is +3 s (cold +22%, the parallel kernel's load from the cache +49%), all of it the switches inside the parallel loop.
  Not looked at: nothing offers the engine, so nothing in the window.

## Step 10. The Mandelbrot settings and the Simulation tab  (effort: medium)

Goal: D6 and D14 as controls (the dropdown still offers only Ray Bouncing).

- A settings record and table: exponent, z_0 (0 or c), pixel maps to c or to z_0, Julia constant (re, im), escape
  behaviour (D29); keyframe-animated where numeric; project section; self-checks.
- The controls under the engine dropdown; within them the Julia constant is greyed in Mandelbrot mode and the z_0 choice
  in Julia mode; a note under the z_0 option when z_0 = 0 explains flagging at the origin and gating at c (D30).

Done when: save, Open and keyframes round-trip them; a hand check in the window. Decisions: D6, D14, D26, D29, D30.

Result (done; `FILE_VERSION` 161; 855 tests pass, 8 new): `MandelbrotSettings` (exponent, z_0, pixel mapping, Julia constant, escape
behaviour) with its table, catalogs, project section `mandelbrot`, keyframe keys (numbers blend, choices hold), `KeyframeData` /
`Project` / `GeneratedFigure` fields, and the controls under the engine dropdown with the greying and the D30 note. Figure-defining
only when the Mandelbrot engine is selected. Verified: save and Open round-trip it (tests, and the real window through
`project_roundtrip_check.py`, which builds the new controls); the ray engine's hashes and project pictures identical; an image
rendered through the app with the Mandelbrot engine and with a Julia set differs from the ray one. Not looked at: how the controls
look and grey in the window, and a keyframe loading them (headless tests only). Two stale dev tools (`project_roundtrip_check.py`,
`project_images.py`) were repaired. Greying by the engine is done here for these controls only; step 13 makes it one resolver.

## Step 11. The Mandelbrot catalog, defaults and obstacle effects  (effort: high)

Goal: what the Mandelbrot engine offers and uses.

(Step 9 already wrote the engine's `EngineDescription`: offered choices, defaults, effects and ignored tabs. What is left is below.)

- Offered choices (D31, D38): Path sequence, its seven point reductions, the metrics of D31, Buddha options (the
  Path stored points, "Last point (distance limit)" and "(confined)", "All path points"; not the midpoints, the self
  bounces or the perpendicular option). Defaults of its own (D39).
- Obstacles: Reflect not offered; a new obstacle made while Mandelbrot is selected is pass-through both ways (D35);
  the replacement of Reflect with pass-through on a switch (D11).
- The hover preview and its bar graph and corkscrew work for an orbit.

Done when: with the test engine replaced by the real one, the step 6 tests run against Mandelbrot; a Buddhabrot renders
sensibly. Decisions: D11, D31, D35, D38, D39.

Result (done; `FILE_VERSION` 162; 862 tests pass, 7 new): the interface offers what the selected engine offers - obstacle effects, the
Colorations tab's metrics, points and Buddha checkboxes (greyed when not offered) - and a new obstacle gets the engine's effects (pass-through
for Mandelbrot); the hover preview, its legends and its Numbers tab work for an orbit. The step 6 tests run against the real description
(`test_engines.py`: the Mandelbrot description and a switch ray -> Mandelbrot -> ray that restores everything). A Buddhabrot of the
escaping orbits and one of the others render through the app's own path, each symmetric about the real axis and unlike each other.
- Found and fixed on the way: an export of a Buddha channel with only multi options and a limited kind of path raised KeyError (ray engine too).
- Left for step 12 by design: nothing CORRECTS a value the engine does not offer (see the CLAUDE.md bullet), so switching the dropdown by hand
  to Mandelbrot (it is not offered yet) fails on the default coloration. The window was checked only by `project_roundtrip_check.py`
  (which found one build-order bug); how the filtered dropdowns and the greyed checkboxes look was not looked at.

## Step 12. The interface behaviour  (effort: high)

Goal: every message and flow of the design, now that both engines are real (still reached through a developer switch).

- Switching engine: the D7 popup (grouped by tab, old -> new, [ok] [go back]) and engine memory (D18).
- Keyframes: the save popup with its suggested changes and [ok] / [go back] that cancels the save (D19); selecting a
  keyframe with invalid values corrects them and shows the message box (D36); the project-open message (D20).
- Rendering: the correction function at every render entry (Start, a still export, each video frame before its margin
  text, thumbnails, the hover preview; D35, D20), errors to the status box for video, popups for a still export.
- The status box clears only when an export (or a re-encode) starts; "n/total" on the progress bar, or a line per frame
  if that is not easy (D21). A project opened with corrections is unsaved (D20).

Done when: each flow is covered by a headless test where it can be, and checked by hand in the window where it cannot.
Decisions: D7, D18, D19, D20, D21, D36. Risk: Tk dialogs are stubbed in the test harness; the real window is the check.

Result (done; `FILE_VERSION` 163; 882 tests pass, 20 new): every flow of the design exists and is reached through the developer switch
`LIGHT_BOUNCES_ENGINES=all`. The switch with its D7 notice, [ok] / [go back] and the engine memory; selecting a keyframe with values the
engine does not accept (D36); the guard on saving into keyframes (D19: `keyframe_fixes`, tested on 100 random keyframe lists that applying its
fixes leaves nothing more to fix); the correction at Start, a still export, every video frame (D20: status box), a new coloration and an opened
project (unsaved afterwards); the status box cleared only when an export starts and the frame count over the progress bar (D21).
- Verified in the real window (`dev/refactor/engine_switch_check.py`, all its checks, and the screenshots looked at): Mandelbrot resets the
  obstacle and the default coloration with a notice; an image, the hover preview and a Julia set with it; going back and switching back restore
  everything and the ray image is pixel-identical; saving the engine into a keyframe offers the fix for the keyframe before it; the frame count
  shows over the progress bar. The ray engine's hashes and project pictures are identical.
- Found on the way: the image on screen keeps the engine it was made with, so a switch (or going back) made the colour chain read data that is
  not there (KeyError): `coloration_fits` now leaves the image alone until Start. And the previous engine must be put into the captured
  settings by hand, since the dropdown already shows the new one.
- Choices of mine to confirm: D20 says an error message per frame; the status box gets the full list once and a short line per later frame
  with the same corrections. The notice is a window of its own with [ok] and [go back] (`show_notice`), since the message boxes of tk have no such
  buttons. Deleting a keyframe's setting is not guarded. Not looked at: the notice window itself (the checks replace it).

## Step 13. Greying and the margin  (effort: medium)

Goal: D4, D5, D13, D15, D22.

- One resolver for whether a control is enabled: the engine allows it AND its own rule (for example the seeding entries
  while "camera pixels" is ticked) allows it. The Emissions and Mutations tabs are greyed for Mandelbrot; no tab or
  control disappears (D4).
- The figure settings margin shows settings the engine ignores in medium grey (a per-line colour in the margin text).
- Greyed settings still save, load, animate and colour solid obstacle outlines (D13, D22); the sequence preview and the
  guide lines stay visible (D15).

Done when: switching engine in the window greys and restores the right controls, with the modified outlines still
correct. Decisions: D4, D5, D13, D15, D22.

Result (done; `FILE_VERSION` 164; 887 tests pass, 5 new): `refresh_enabled_states` is the one resolver (engine AND the control's own rule);
Emissions and Mutations grey whole under Mandelbrot and come back, fuzzing's own rule never enables what the engine ignores; the figure
settings margin lists ignored settings in medium grey (`GreyLine`) and gains the orbit's own settings block (grey under the ray engine). Checked in
the real window (tabs greyed and back, the Termination tab untouched, the margin picture looked at). Not looked at: the blue modified outlines on a
greyed control (they are drawn by the same frames as before; nothing about them was changed).

## Step 14. Switch it on and finish  (effort: medium)

- Offer "Mandelbrot" in the dropdown.
- Run `dev/refactor/project_roundtrip_check.py` in the real window; check the full set of flows once by hand.
- Re-measure: warm and cold compile time, large-export memory for both engines, a Mandelbrot export and a Buddhabrot.
- Update `CLAUDE.md` and the module docstring; save a memory note for anything a later
  session must not rediscover.
- Review `Engine_Refactor_Design.md` for decisions marked "revisit" and for anything no longer true.

Result (done; `FILE_VERSION` 165; 888 tests pass): Mandelbrot is in the dropdown for everyone (`OFFERED_ENGINES`; the environment switch of
step 12 is gone; a project may name either engine). `project_roundtrip_check.py` and `engine_switch_check.py` pass in the real window; hashes and
project pictures of the ray engine identical. `CLAUDE.md` and the module docstring describe the engines (the docstring also said an ellipse is
0 to 2 wide: fixed to 0 to 4). A bug found while checking: CRLF doubled to CR CR LF in six places of `CLAUDE.md`, from my own writes.
- Re-measured: warm run times unchanged; first kernel calls cold 16.9 s and from the cache 10.5 s (13.9 and 7.8 s before the refactor);
  3840 x 2160 exports, peak memory: ray default 895 MB (874 before the layouts), two simple Buddha options 1006 MB, Mandelbrot default 967 MB,
  Mandelbrot Buddhabrot with all path points 1056 MB. The ray images have the same hashes as at the start of the refactor's step 8.
- The design review: nothing is marked "revisit" except D2 (a third engine: none exists), and D27 (a perpendicular collision for orbits "may change":
  parked). Where the program departs from the letter of a decision, on purpose: D20 (a message per frame: the first in full, then a short line
  while the corrections stay the same), D18 ("changed since" is a comparison of the value with the one the reset put there, so typing the reset
  value back counts as unchanged), D13 (the margin also lists the orbit's own settings, in grey under the ray engine), D21 (the progress text uses
  the recipe of the link and works in the real window). D1 and D9 hold as written.
- Not done, for you to decide: the start-up cost (about 3 s a launch) of keeping the engine switches inside the parallel loop (D28). Not looked at:
  the notice window and the blue modified outlines on greyed controls.

---

## Parked (not decisions, not scheduled)

- Revisit D2 (one kernels file) when a third engine is added.
- A perpendicular collision for Mandelbrot paths (D27 says this may change), and reflecting an orbit's segments off a
  reflective obstacle the way a ray bounces (mentioned as a possibility, not decided).
- A smooth (fractional) iteration count as a metric.
- If a setting that affects a dashed obstacle outline ever depends on the engine, ask again (D22).

## Effort summary

| Step | Effort |
|---|---|
| 1 Lock in, scale check | medium |
| 2 Format 9, pinned values, keys | high |
| 3 The doubled world | high |
| 4 Buddha filter, all path points | medium |
| 5 Engine as a setting | medium |
| 6 Compatibility core | high |
| 7 Engine switch in the drivers | high |
| 8 Per-engine layouts | high |
| 9 Mandelbrot path function | high |
| 10 Mandelbrot settings and tab | medium |
| 11 Mandelbrot catalog, defaults, effects | high |
| 12 Interface behaviour | high |
| 13 Greying and margin | medium |
| 14 Switch on and finish | medium |
