# Rendering for Humans

A guide to how Light Bounces turns settings into pixels, followed by assessments of the code that does it.

- **Snapshot:** `lightbounces.version.FILE_VERSION` = 142, 727 tests passing, 2026-10-08.
- **Audience:** someone who knows Python and numpy but has not read `lightbounces.kernels` (2,466 lines) and does not want to start there.
- **Scope:** the path from a ray's start point to a coloured pixel, with most attention on data types, on how *sequences* become *colours*, and on what the arrays holding sequences look like before Buddha scattering. Part 2 assesses cleanliness, consistency, readiness for several sequence engines (ray bouncing now, the Mandelbrot set next) and names that deserve renaming.

## Conventions used in this document

- **Fully qualified names.** `lightbounces.kernels.trace_path` is the function `trace_path` in the module `lightbounces/kernels.py`. The package also re-exports every name (so `lightbounces.trace_path` works), but the defining module is always given here.
- **Methods of the window.** `lightbounces.core.App` has no methods of its own except `__init__`; it inherits 14 mixin classes, one per module. `lightbounces.app_render.RenderMixin.apply_coloration` is therefore called as `self.apply_coloration(...)` on the `App`. The name always carries the mixin so you can find the source.
- **Verified vs read.** Every factual claim is tagged where it matters:
  **[V]** = I ran code in this session and observed it (dtypes, shapes, sizes, timings, the worked example);
  **[R]** = I read it in the source and did not run it.
  Opinions are labelled as such, with a confidence.
- **"Pixel", "image", "figure".** Following the project's terminology (`CLAUDE.md`): the *figure* is the mathematical result of the rules; every picture is an *image* of it. A *coloration* is how stored numbers become colours; a *framing* is the camera.

---

# Part 1 — How rendering works

## 1.1 The pipeline on one page

```
 settings records        scene (Shape objects)        framing (x, y, diagonal, aspect)
        |                        |                               |
        |  settings_ray_args     |  build_obstacle_arrays        |  framing_world_rect
        v  settings_natural_termination v                               v
 kernel arguments:  RayArgs, NaturalTerminationRule, ObstacleArrays, WorldRect     (namedtuples / numpy arrays)
        |
        v   once per pixel, in parallel:  kernels._trace_rows_parallel
 pixel centre (x, y) --fires_ray?--no--> pixel_state = PIXEL_NO_RAY (2)
        | yes
        v
 trace_path ---> path_points (max_iter+2, 2) f64   path_point_tags (max_iter+2,) f64   raw (11,) f64   term, npts     [one ray, thrown away]
        v
 compute_run_measurements      ---> out (24,) f64   (the ray's own run measurements)
 compute_sequences  ---> out (24,) f64   (sequence scalars)     ray_characteristic_points (28, 2) f64  (sequence points)    [one ray, thrown away]
        v   copied, narrowed to float32
 stats (h, w, 24) f32   characteristic_points (h, w, 28, 2) f32   pixel_state (h, w) u8                    [kept: the stored image data]
        v
 coloration = 3 channels, each  Query | Constant | Buddha
   Query    : read slots of stats / characteristic_points   --> value in [0, 1]
   Constant : 0 or 1
   Buddha   : points --> counts (h, w) f64 --> FFT blur --> compress --> [0, 1]
        v
 channels_to_rgb(mode, v0, v1, v2) --> x 255 --> rint --> uint8 (h, w, 3), BACKGROUND where pixel_state != PIXEL_TRACED
        v
 outlines (+ name overlay on screen only)  -->  Tk PhotoImage (PPM)  |  PNG / QOI / JXL  |  video frames
```

Three things the diagram does not say:

1. **The whole path of a ray is never kept.** Only reductions of it survive into `stats` and `characteristic_points`. That is why some Buddha options must trace their rays a second time (section 1.10).
2. **Everything after `pixel_state`, `stats` and `characteristic_points` is cheap and repeatable.** Changing a coloration, the Buddha blur or the Buddha seeding never re-runs `trace_path` for the image's own rays; it only re-reads the stored arrays. Changing a framing re-runs it automatically; changing a ray setting, a termination limit or an obstacle re-runs it when you press Start.
3. **There are three separate implementations of "stored numbers to colour"** (live rows, whole image, single hovered ray) and **three decoders of the Buddha option list**. They agree today; section 2.1 lists them as the main duplication.

## 1.2 Vocabulary

| Term | Meaning here | Defined / held in |
|---|---|---|
| figure | what the rules produce, in the limit | not a data structure |
| image | one picture of the figure: a pixel grid over a world rectangle | `lightbounces.core.App.img` and friends |
| framing | camera: centre `x`, `y`, `diagonal`, `aspect` (width / height), world units | `lightbounces.framings.FramingConfig` (live), `FramingSnapshot` (frozen) |
| world rectangle | the rectangle of the world a framing covers, as four floats | `lightbounces.kernels.WorldRect` (`world_rect` as a parameter) |
| ray / path | one traced bounce path from one start point | `path_points`, `path_point_tags` inside `lightbounces.kernels.trace_path` |
| fire | a pixel "fires" a ray if its centre is inside a shape with *fire rays inside* on | `lightbounces.kernels.fires_ray` |
| pixel_state | `uint8` per pixel: `PIXEL_PENDING` 0 = its row is not rendered yet, `PIXEL_TRACED` 1 = its ray was traced and stored, `PIXEL_NO_RAY` 2 = reached, no ray fires there | `lightbounces.app_preview.PreviewMixin.pixel_state` (made in `make_image_buffers`) |
| sequence | a named, filtered view of one path (`SEQ_PATH`, `SEQ_BOUNCES`, `SEQ_BOUNDARY`, `SEQ_SELF`) | `lightbounces.kernels.in_sequence` |
| reduction | a fixed summary of a sequence: 7 point reductions (`PRED_*`), 4 scalar reductions (`SRED_*`) | `lightbounces.kernels.sequence_summary` |
| slot | an index into a ray's stored array: 28 point slots, 24 metric slots | `lightbounces.kernels.POINT_SLOT`, `STAT_SLOT` |
| run measurement | a number about the ray's own run, not about any sequence (loop area, bounce angles, termination reason, last-segment ratio, and four placeholder distances) | `lightbounces.kernels.compute_run_measurements` |
| metric | the UI word for a metric slot (`stats[..., slot]`); always scaled to 0..1 | `lightbounces.metrics.METRIC_LABELS` |
| coloration | colour model (`"RGB"`/`"HSV"`) + three channels | `lightbounces.colorations.ColorationConfig` |
| channel | one of the three colour components, with a *source*: Query, Constant or Buddha | `lightbounces.metrics.CHAN_SRC_*` |
| counts | `float64 (h, w)` histogram of where Buddha points landed | `lightbounces.core.App.buddha_counts` |
| seed point | a start point for the rays Buddha channels scatter (not the RNG seed) | `lightbounces.settings.BuddhaSeedSettings` |
| zone | the world rectangle seed points are laid over | `BuddhaSeedSettings.x_start` ... `y_stop` |

## 1.3 Coordinates: world, world rectangle, pixel

**World coordinates** are plain `float64` (x to the right, y up). The default scene is the unit circle at the origin.

**A framing makes a world rectangle.** `lightbounces.framings.framing_size`: height = diagonal / sqrt(1 + aspect²), width = aspect × height. `lightbounces.framings.framing_world_rect` returns `WorldRect(x - width/2, x + width/2, y - height/2, y + height/2)` as Python floats. The default diagonal is `lightbounces.constants.DEFAULT_FRAMING_DIAGONAL` = 2·sqrt(2), chosen so that at aspect 1 the rectangle is exactly `WorldRect(-1.0, 1.0, -1.0, 1.0)` = `lightbounces.kernels.WORLD_RECT_DEFAULT` **[V]** (`framing_world_rect(DEFAULT_FRAMING) == WORLD_RECT_DEFAULT` holds exactly, and a test pins it).

**A framing also makes a pixel grid.** `lightbounces.framings.framing_image_size(f, box=PREVIEW_BOX)`: aspect ≥ 1 gives `(512, max(1, round(512 / aspect)))`, otherwise `(max(1, round(512 * aspect)), 512)`. So the live image is at most 512 on its longer side, and pixels are only approximately square when the rounding bites. Exports choose their own width and height; if their shape differs from the rectangle's the picture stretches (as non-square exports always did).

**Pixel centre to world** (identical arithmetic in `lightbounces.kernels._trace_rows_parallel`, `lightbounces.kernels.scatter_from_seeds`, `lightbounces.image_io.make_grid` and `lightbounces.app_preview.PreviewMixin.pixel_center`, so a hover ray starts exactly where the image's own ray did **[V]**):

```
x = x_start + (i + 0.5) * ((x_stop - x_start) / w)
y = y_stop  - (j + 0.5) * ((y_stop  - y_start) / h)        # row 0 is the top
```

**World to pixel** (floor of the same thing, written out separately in `lightbounces.kernels.scatter_one_buddha_point`, `lightbounces.app_recolor.RecolorMixin.scatter_buddha_rows` and `lightbounces.app_render.RenderMixin.apply_coloration`; as floats in `lightbounces.app_preview.PreviewMixin.world_to_canvas` and `lightbounces.shapes.Shape.outline_coords`):

```
col = floor((x - x_start) * (w / (x_stop - x_start)))
row = floor((y_stop - y) * (h / (y_stop - y_start)))       # outside 0 <= col < w, 0 <= row < h: the point is not counted
```

Two fixed constants survive from the era of one fixed image: `lightbounces.kernels.PREVIEW_BOX` = 512 (now "the longer side of the live image") and `lightbounces.kernels.DEFAULT_PIXEL_WORLD_WIDTH` = 2/512 (the world width of one pixel of the default framing at that size). They set the default confinement threshold and the outline thickness (section 2.4 suggests renaming them).

**Precision.** Kernels work in `float64`. Stored sample points are narrowed to `float32` (section 1.8); `float32` spacing near 0.75 is 5.96e-8 **[V]**, which equals one pixel's width when the world rectangle is about 3.05e-5 wide, roughly 6.6×10⁴ times narrower than the default (arithmetic **[V]**; consequence for deep zoom in section 2.3).

## 1.4 Data types: the inventory

There are four families. Know which one you are looking at.

### A. Settings the user edits (Python, strict, frozen)

All are `lightbounces.settings.strict_record` dataclasses: frozen, slotted, keyword-only with every field required, type-checked at construction (an `int` is accepted for a `float` field, a `bool` is not).

| Record | Holds | Module |
|---|---|---|
| `RaySettings` | segment multiplier, `fuzz: FuzzParams`, emission mode / target / offset | `lightbounces.settings` |
| `NaturalTerminationSettings` | `max_iter`, `max_dist` (0 = none), `escape_radius`, three confinement settings | `lightbounces.settings` |
| `BuddhaBlurSettings`, `BuddhaSeedSettings` | blur radius/type; camera-pixels flag and zone | `lightbounces.settings` |
| `FramingSnapshot` | `id`, `x`, `y`, `diagonal`, `aspect` (all float except id) | `lightbounces.framings` |
| `ColorationSettings` | resolved coloration: `mode: str`, then ten per-channel **tuples** (section 1.9) | `lightbounces.colorations` |
| `ChannelSettings` | one channel as a keyframe stores it (9 fields) | `lightbounces.colorations` |

The *live* versions are mutable: `lightbounces.colorations.ColorationConfig` (12 slots; per-channel **lists** edited in place) and `lightbounces.framings.FramingConfig`. Snapshots copy lists to tuples so a render thread cannot see a half-made edit.

### B. Kernel arguments (numba-friendly namedtuples and arrays) **[V]**

Built only by their builders, read in kernels only by field name.

| Type | Fields (type) | Builder |
|---|---|---|
| `lightbounces.kernels.NaturalTerminationRule` | `max_iter` (int), `max_dist` (float; `inf` = no limit), `escape_radius` (float), `confine` (`ConfineRule`) | `lightbounces.settings.natural_termination_rule`, `settings_natural_termination` |
| `lightbounces.kernels.ConfineRule` | `lookback` (int: how many recent path points), `method` (int), `threshold` (float, world units) | `lightbounces.settings.confine_rule` |
| `lightbounces.kernels.RayArgs` | `seg_mult` (float), `fuzz` (`FuzzRule`: `enabled` bool + 7 floats), `emission` (`EmissionRule`: `mode` int, `target_x`, `target_y`, `offset` floats), `obstacles` (`ObstacleArrays`) | `lightbounces.settings.ray_args`, `settings_ray_args` |
| `lightbounces.kernels.ObstacleArrays` | `shapes` int64 (n,), `bounds` float64 (n, 205), `cx`, `cy` float64 (n,), `effects` int64 (n, 2), `fire_inside` bool (n,) | `lightbounces.shapes.build_obstacle_arrays` |
| `lightbounces.kernels.WorldRect` | `x_start`, `x_stop`, `y_start`, `y_stop` (floats) | `lightbounces.framings.framing_world_rect` |

`bounds` has 205 columns because `lightbounces.constants.OBST_BOUND_LEN` is sized for the largest packing (the snail of Pythagoras, 100 segments).

A numba limitation shapes the drivers: a `parallel=True` function cannot use a tuple that contains other tuples inside its `prange` loop. `lightbounces.kernels.trace_rows` is therefore a plain wrapper that takes `termination` and `engine_args` whole and passes their one-level pieces to `lightbounces.kernels._trace_rows_parallel` (which has **17 parameters**), where the whole tuples are rebuilt per pixel **[R]**.

### C. Per-ray scratch arrays (inside the kernels, one ray at a time) **[V]**

| Name | dtype | shape | Meaning |
|---|---|---|---|
| `path_points` | float64 | (`max_iter` + 2, 2) | the path: start, bounce points, optional end point |
| `path_point_tags` | float64 | (`max_iter` + 2,) | a *tag* per path point (`TAG_START` 0.0 ... `TAG_PERPENDICULAR` 6.0) |
| `raw` | float64 | (11,) | `RAW_*` run measurements (bounce count, termination code, deflection sum, ...); integers are stored as floats |
| `out` | float64 | (24,) | the ray's 24 metric slots, filled by `compute_run_measurements` and `compute_sequences` |
| `ray_characteristic_points` | float64 | (28, 2) | the ray's 28 stored sample points |
| `term`, `npts` | int | scalar | why it terminated (`TERM_*`), how many valid rows of `path_points` |

### D. Per-image arrays (kept) **[V]**

| Attribute (all on `lightbounces.core.App`) | dtype | shape | Notes |
|---|---|---|---|
| `stats` | float32 | (h, w, 24) | metric slots; 96 bytes per pixel |
| `characteristic_points` | float32 | (h, w, 28, 2) | sample points; 224 bytes per pixel |
| `pixel_state` | uint8 | (h, w) | 0 pending, 1 ray traced and stored, 2 no ray here |
| `img` | uint8 | (h, w, 3) | the coloured image; pre-filled with `BACKGROUND` = (12, 12, 16) |
| `frame`, `image_for_screen` | uint8 | (h, w, 3) | `img` + outlines (what is exported), then + coloration-name overlay (screen only) |
| `gx`, `gy` | float64 | (h, w) | world coordinates of every pixel centre |
| `buddha_counts` | list of 3 | each `None` or float64 (h, w) | raw Buddha histograms per channel |
| `buddha_counts_blurred` | list of 3 | each `None` or float64 (h, w) | cached blurred copy |
| `hover_path_points`, `hover_path_point_tags`, `hover_raw`, `hover_vec`, `hover_ray_characteristic_points` | float64 | (10001, 2), (10001,), (11,), (24,), (28, 2) | scratch for the hovered ray; 10001 = `ITER_LIMIT_MAX` + 2 |

**Stored bytes per pixel = 321** (96 + 224 + 1). At 512×512 that is 84 MB; at 1920×1080, 666 MB; at 3840×2160, 2,663 MB **[V]**. `lightbounces.app_render.RenderMixin.render_full_image` allocates these whole for an export, then adds one `float64 (h, w)` counts array per Buddha channel (66 MB each at 4K) and the FFT's complex padded copies (not measured).

### E. Code spaces: the integers and floats that mean things

Each family below is a set of numbers whose *meaning is only the family*. Mixing families is the commonest source of confusion in this codebase (section 2.2).

| Family | Values | Type | Defined in |
|---|---|---|---|
| `TERM_*` | CONFINED 0, ITER 1, DIST 2, OBSTACLE 3, PERPENDICULAR 4, ESCAPED 5 | int | `lightbounces.kernels` |
| `TAG_START` ... `TAG_ESCAPED` | 0.0 ... 5.0 point *tags* | **float** (stored in a float64 array) | `lightbounces.kernels` |
| `SHAPE_*` | 0 ... 17 obstacle class codes (also `Shape.KIND`) | int | `lightbounces.kernels` |
| `SEQ_*`, `PRED_*`, `SRED_*` | 4 sequences, 7 point reductions, 4 scalar reductions | int | `lightbounces.kernels` |
| `ST_*`, `POINT_*` | 13 legacy metric slots, 13 legacy point slots (the *first* slots) | int | `lightbounces.kernels` |
| point choice ids | 0-27 = stored slots; 28 = `CHOICE_ORIGIN`; 29 = `CHOICE_TOP_LEFT_CORNER` (no slot) | int | `lightbounces.metrics` |
| `mapping[c]` values | a metric slot 0-23, or `NONE_ZERO` = -1, `NONE_ONE` = -2, `CHAN_BUDDHA` = -3 | int (sentinels share the field with slots) | `lightbounces.metrics` |
| `CHAN_SRC_*` | QUERY 0, BUDDHA 1, CONSTANT 2 | int | `lightbounces.metrics` |
| `METRIC_TYPE_POINT_PAIR`, `METRIC_TYPE_OTHER` | channel metric type: point comparison 0, other 1 (the project file's key for it is still `kind`) | int | `lightbounces.metrics` |
| `BUDDHA_SIMPLE`, `BUDDHA_MULTI` | option style 0, 1 | int | `lightbounces.metrics` |
| `BUDDHA_MULTI_ALL_SELF` ... | `"all_self"`, `"all_mid"`, `"prog_bounce"`, `"prog_mid"` | **str** | `lightbounces.metrics` |
| `BUDDHA_COMPRESS_*` | ARCTAN 0, MODULUS 1 | int | `lightbounces.metrics` |
| `ColorationSettings.mode` | `"RGB"`, `"HSV"` | **str** | `lightbounces.colorations.COLOR_MODELS` |

Termination codes are not in the order of their stored metric values: `ST_TERMINATION` stores ITER 0.0, CONFINED 0.2, DIST 0.4, OBSTACLE 0.6, PERPENDICULAR 0.8, ESCAPED 1.0 (`lightbounces.constants.TERM_STAT_VALUE`, duplicated as literals in `lightbounces.kernels.compute_run_measurements`; a test, `test_every_termination_reason_maps_to_its_own_stat_value_in_compute_stats`, ties them together).

## 1.5 Stage 1: from settings to kernel arguments

`lightbounces.app_generation.GenerationMixin.start` (button) and `begin_generation` (also run by a framing change) turn the live controls into the kernel arguments and keep them on the `App` as `gen_termination` (a `NaturalTerminationRule`) and `gen_engine_args` (a `RayArgs` whose `obstacles` are the scene *as generated*). Two consequences:

- Editing a control afterwards changes nothing on screen until Start: the image is of `generated_settings`, `generated_natural_termination` and the frozen obstacles.
- A new framing re-runs `begin_generation` with those same kept arguments: a new image of the *same figure*, no Start needed.

The world rectangle comes from `lightbounces.app_preview.PreviewMixin.set_image_framing`, which also reallocates (or clears) the per-image arrays when the framing changes.

## 1.6 Stage 2: tracing one ray, `lightbounces.kernels.trace_path`

Signature: `trace_path(x0, y0, path_points, path_point_tags, termination, engine_args, raw)` returns `(term, npts)`. It is one 365-line function **[V]** (the longest in the package).

What goes in `path_points`:

- `path_points[0]` is the start point (`TAG_START`).
- `path_points[1] ... path_points[nb]` are the **bounce points** (`nb` = number of bounces), tagged `TAG_BOUNDARY` (bounced off a shape's edge) or `TAG_SELF` (bounced off the ray's own earlier path).
- One more point may follow, tagged `TAG_DISTCUT`, `TAG_OBSTACLE_TERM` or `TAG_ESCAPED`, when the ray was ended by the distance limit, a *terminate* edge or by escaping. A ray ended by exactly perpendicular incidence also gets a final point, tagged `TAG_PERPENDICULAR` (since FILE_VERSION 144; it used to reuse `TAG_SELF`/`TAG_BOUNDARY`), and `nb` is not increased, so it is *not* a bounce point. Membership is by tag alone (section 1.7).
- With the segment multiplier below 1, recorded bounce points are *pulled back* along their segments; the ray itself continues from the true hit.

Why a ray stops (`term`): `TERM_ITER` (limit reached), `TERM_DIST`, `TERM_ESCAPED` (nothing ahead; the last point lies `escape_radius` from the origin), `TERM_OBSTACLE`, `TERM_PERPENDICULAR`, and `TERM_CONFINED`, which `lightbounces.kernels.is_confined` decides from the last `confine.lookback` points (bounding box under the threshold both ways, or summed segment length under it). `TERM_CONFINED` is also reused for two safety exits (a zero-length normal, a pass-through crossing overflow) **[R]**.

Cost shape **[R]**: each bounce tests every earlier segment of the same path for a self-hit, so one ray costs O(bounces²). Long rays dominate render time; the reductions that follow are comparatively cheap.

**Worked example [V].** Default scene (unit circle, reflecting both ways, fires inside), iteration limit 40, a ray from (0.31, -0.22): `term` = 0 (`TERM_CONFINED`), `npts` = 20, 19 bounces. `kinds[:10]` = `[0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0]`: the start, three boundary bounces, a self bounce, a boundary bounce, then self bounces. The first points were (0.31, -0.22), (0.9206, 0.3906), (-0.9334, 0.3589), (0.4218, -0.9067), (0.3828, -0.1472).

## 1.7 Stage 3: from a path to sequences and their reductions

A *sequence* is a rule for which path points count. `lightbounces.kernels.in_sequence(seq, k, path_point_tags)` is its single definition, decided by the point's tag alone (since FILE_VERSION 144; it used to take `nb` and `npts` too):

| Sequence | Label | Point `k` belongs if |
|---|---|---|
| `SEQ_PATH` (0) | Path | always (start, bounces and any end point) |
| `SEQ_BOUNCES` (1) | All bounces | `path_point_tags[k]` is `TAG_BOUNDARY` or `TAG_SELF` |
| `SEQ_BOUNDARY` (2) | Boundary bounces | `path_point_tags[k] == TAG_BOUNDARY` |
| `SEQ_SELF` (3) | Self bounces | `path_point_tags[k] == TAG_SELF` |

`lightbounces.kernels.sequence_summary` walks the path once per sequence and writes **every** reduction of it. `lightbounces.kernels.compute_sequences` loops over the four. Both write into the ray's `ray_characteristic_points` and `out`.

**Point reductions** (each yields one (x, y); an empty sequence reports the path's end point for every one): `PRED_FIRST`, `PRED_LAST`, `PRED_MEAN`, `PRED_TOP` (largest y, first on a tie), `PRED_BOTTOM`, `PRED_RIGHT`, `PRED_LEFT`.

**Scalar reductions** (each yields one number in 0..1):

| `SRED_*` | Formula |
|---|---|
| `SRED_COUNT` | min(points / `max_iter`, 1) |
| `SRED_LENGTH` | atan(polyline length / 5) × 2 / π |
| `SRED_LAST_DIR` | direction of the last segment, -180..180° mapped to 0..1; 0.5 with fewer than two points |
| `SRED_SHARE` | min(points / bounces, 1); 0 if the ray has no bounces |

### Where each reduction is stored (the slot tables) **[V]**

The slot numbering keeps the 13 legacy numbers (`ST_*`, `POINT_*`) as the *first* slots and numbers the rest of the 4×7 and 4×4 grids in order.

Point slots, `lightbounces.kernels.POINT_SLOT[sequence, reduction]` (shape (4, 7), int64):

| sequence \ reduction | first | last | mean | top | bottom | right | left |
|---|---|---|---|---|---|---|---|
| Path | 0 | 1 | 13 | 14 | 15 | 16 | 17 |
| All bounces | 18 | 19 | 2 | 5 | 6 | 7 | 8 |
| Boundary bounces | 20 | 21 | 3 | 22 | 23 | 24 | 25 |
| Self bounces | 26 | 27 | 4 | 9 | 10 | 11 | 12 |

Metric slots for the scalar reductions, `lightbounces.kernels.STAT_SLOT[sequence, reduction]` (shape (4, 4)):

| sequence \ reduction | count | length | last dir | share |
|---|---|---|---|---|
| Path | 13 | 9 | 10 | 14 |
| All bounces | 4 | 15 | 16 | 17 |
| Boundary bounces | 18 | 19 | 20 | 6 |
| Self bounces | 21 | 22 | 23 | 5 |

The remaining 8 metric slots are written by `lightbounces.kernels.compute_run_measurements`, not by any sequence:

| Slot | Name | Note |
|---|---|---|
| 0, 1, 2, 3 | Horizontal, Vertical, Manhattan, Euclidean distance | **placeholders**: start-to-end values; colouring recomputes them from whichever two points a channel chose (section 1.9) |
| 7 | Loop area / same-perimeter circle | 0 if the ray never touches itself or the segment multiplier is below 1 |
| 8 | Mean bounce angle / 180° | |
| 11 | Termination reason | the six values above |
| 12 | Last segment / uncut length | 1 unless the distance limit cut the last segment |

Two observations from the formulas, confirmed in the worked pixel below **[V]**: `Path: Share of the bounces` (slot 14) and `All bounces: Share of the bounces` (slot 17) equal 1.0 for every ray that has at least one bounce, because the path has at least as many points as bounces; they carry almost no information.

The 28 point slots plus the two fixed choices (`CHOICE_ORIGIN` = (0, 0) and `CHOICE_TOP_LEFT_CORNER` = (-1, 1), both unrelated to the ray, neither stored) make 30 point choices; `lightbounces.metrics.CHOICE_ORIGIN` is defined as `N_POINTS`, so its id would change if the number of characteristic points changed.

## 1.8 Stage 4: the stored arrays and the generation loop

**One pixel's journey into the arrays** (`lightbounces.kernels._trace_rows_parallel`, a `prange` over all pixels of the requested rows) **[R]**:

1. Compute the pixel centre (section 1.3).
2. `fires_ray(x, y, obstacles)`; if false, set `pixel_state[j, i] = PIXEL_NO_RAY` and `continue`.
3. If fuzzing, seed numba's RNG with `rng_seed_base + i * 1000003 + j * 97`, so every pixel's ray is reproducible and a re-trace elsewhere gives the same ray.
4. Allocate `path_points`, `path_point_tags`, `raw`, `out`, `ray_characteristic_points` (fresh `np.empty` per pixel).
5. `trace_path`, `compute_run_measurements`, `compute_sequences`.
6. Copy `out` into `stats[j, i, :]` and `ray_characteristic_points` into `characteristic_points[j, i, :, :]`: **the float64 to float32 narrowing happens here**. Set `pixel_state[j, i] = PIXEL_TRACED`.

**Row batching.** `lightbounces.app_generation.GenerationMixin.tick` runs on the Tk main loop. Its budget is `PERIOD` × `COMPUTE_FRACTION` = 0.1 s × 0.55 per refresh; it aims for about three batches and rescales `batch_rows` by 0.5 to 2 after each. After the rows it calls `scatter_buddha_rows` for just those rows and `recolor`, then `draw`. If the live coloration has a Buddha channel it recolours **all** rows computed so far (a new count can change any pixel already drawn), otherwise only the new rows.

**`pixel_state` has three values (since FILE_VERSION 145; it used to be a two-valued `status`).** `PIXEL_PENDING` (0) means the pixel's row has not been rendered yet, `PIXEL_TRACED` (1) that a ray was fired and stored, `PIXEL_NO_RAY` (2) that the pixel was reached but lies outside every firing shape. Only `PIXEL_TRACED` pixels get a colour from a Query *or a Buddha* channel (section 1.9, 1.10). A finished render leaves no pixel pending. `App.rows_done` is still the scheduler's "next row" counter, but it is no longer needed to interpret the array.

**What is not stored:** the path (`path_points`), the tags (`path_point_tags`) and `raw`. Everything about a path that can be asked later is one of the 28 point slots and 24 metric slots computed at trace time. This is the central design choice: recolouring is instant because the question "which sequence, which reduction?" was answered for *all* of them in advance.

**Other holders of the same arrays.** `lightbounces.app_render.RenderMixin.render_full_image` builds a fresh `(stats, characteristic_points, pixel_state)` triple at the export's size and discards it afterwards. `lightbounces.app_render.RenderMixin.simulated_thumbnail_data` builds a small triple (48×48 box) per framing, cached in `App.thumb_sims` until the next generation.

## 1.9 Stage 5: from stored numbers to a colour

**The coloration's data.** Ten parallel per-channel tuples in `lightbounces.colorations.ColorationSettings` (`metric_types`, `point_metric`, `other_metric`, `point1_choice`, `point2_choice`, `chan_src`, `buddha_points`, `buddha_compression`, `buddha_brightness`, `constant_choice`; plus `color_model`). `lightbounces.metrics.make_mapping` reduces them to `mapping`, a tuple of 3 ints: the metric slot each channel uses, or a sentinel (`NONE_ZERO`, `NONE_ONE`, `CHAN_BUDDHA`).

**A sequence never becomes a colour directly.** A channel gets to a sequence by exactly three doors:

| Source (`chan_src`) | Reads | Becomes a value in 0..1 by |
|---|---|---|
| Query, *other* metric (`metric_types[c] == METRIC_TYPE_OTHER`) | one metric slot `s` of `stats` | `np.clip(stats[..., s][computed].astype(np.float64), 0, 1)` |
| Query, *point comparison* (`metric_types[c] == METRIC_TYPE_POINT_PAIR`) | two **point** slots (or the two fixed points) | `lightbounces.metrics.extract_point_xy` for each, then `lightbounces.metrics.point_comparison_metric`: horizontal = abs(dx) × 0.5, vertical = abs(dy) × 0.5, Manhattan = (abs(dx) + abs(dy)) × 0.5, Euclidean = hypot(dx, dy) × 0.5, each clipped to 0..1 |
| Constant | nothing | zeros or ones |
| Buddha | chosen **points** (or whole re-traced sequences) | histogram, blur, compress (section 1.10) |

So a sequence's *scalars* (count, length, last direction, share) are colours as they are; a sequence's *points* become colours only as a **distance between two of them** or as a **place to drop a count**. `extract_point_xy` converts the stored float32 points of the fired pixels into two float64 arrays of length n, where `n = int(computed.sum())`.

**Then:** `lightbounces.colorations.channels_to_rgb(mode, v0, v1, v2)` multiplies by 255 (RGB), or runs `hsv_to_rgb_np` first (HSV; hue wraps with `% 1.0`); the three float64 arrays are stacked, `np.rint`-rounded (half to even), clipped and cast to `uint8`, and assigned **only into fired pixels**: `img[computed] = ...`. Every other pixel keeps `BACKGROUND`.

**Worked pixel [V]** (default scene, 64×48 image, iteration limit 40, default coloration; pixel i = 40, j = 24, centre (0.265625, -0.0208333)):

| Item | Value |
|---|---|
| `pixel_state[24, 40]` | 1 |
| `characteristic_points[24, 40, 0]` (Path: First point) | (0.265625, -0.0208333) |
| `characteristic_points[24, 40, 1]` (Path: Last point) | (0.521502, -0.152772) |
| channel 0 = point comparison, horizontal: abs(0.265625 - 0.521502) × 0.5 | 0.127938 (`stats[0]` holds the same value only because the default points are Path first and last, the placeholder pair) |
| channel 1 = point comparison, vertical | 0.065969 |
| channel 2 = other metric `ST_BOUNCE_COUNT` (`stats[4]`, All bounces: count / limit = 19/40) | 0.475 |
| × 255, rounded | 33, 17, 121 |
| `apply_coloration(...)[24, 40]` | [33, 17, 121] (identical) |
| unfired corner pixel | [12, 12, 16] = `BACKGROUND` |
| `stats[24, 40, 11]` (Termination reason) | 0.2 (confined) |

**Three implementations of this chain** (all produce the same numbers; none calls another):

- `lightbounces.app_recolor.RecolorMixin.recolor(r0, r1)`: the live image, rows `[r0, r1)`, writes into `self.img`.
- `lightbounces.app_render.RenderMixin.apply_coloration(...)`: a whole image of any size (exports, non-live thumbnails, framing thumbnails); returns a new `uint8 (h, w, 3)`.
- `lightbounces.app_recolor.RecolorMixin.component_values(vec, ray_characteristic_points, x0, y0)`: one hovered ray, returns three floats.

## 1.10 Stage 6: Buddha scattering (and what holds sequences before it)

**The idea.** A Buddha channel does not colour a pixel by anything about *its* ray. It scatters chosen points of *many* rays into a histogram `counts` the size of the image, optionally blurs it, and then colours each pixel by the (compressed) value of the histogram **at that pixel's own location**.

**Which points.** `lightbounces.metrics.BUDDHA_POINT_CATALOG` has 35 entries **[V]**; a channel chooses any subset (`buddha_points`: a tuple of 35 bools), and all chosen options accumulate into the one histogram:

| Entries | Style | One point per fired pixel? |
|---|---|---|
| 0-27: each stored point slot (e.g. "Self bounces: Mean point") | simple | yes: one point per fired ray per option |
| 28, 29, 30: "Path: Last point" only for rays ended by distance limit / perpendicular hit / confinement | simple, filtered | yes, for the rays that ended that way |
| 31-34: All self bounces, All segment midpoints, Progressive means of bounce points, Progressive means of segment midpoints | multi | no: many points per ray (up to about `max_iter`) |

### The arrays that hold sequences, before scattering **[V]**

This is the data the scatterer has to work from. Read it top to bottom as the life of a sequence.

| # | Array | dtype | Shape | What it holds | Lifetime |
|---|---|---|---|---|---|
| 1 | `path_points` | float64 | (`max_iter` + 2, 2) | **the whole sequence**: every path point in order | one ray; the only place the full path exists |
| 2 | `path_point_tags` | float64 | (`max_iter` + 2,) | the tag of each point, which makes sequences (`in_sequence`) | one ray |
| 3 | `ray_characteristic_points` | float64 | (28, 2) | each of the 4 sequences **reduced to 7 points** | one ray |
| 4 | `characteristic_points` | float32 | (h, w, 28, 2) | `ray_characteristic_points` of every fired pixel, narrowed | the image |
| 5 | `stats[..., 11]` | float32 | (h, w) | termination reason (0.0, 0.2, ... 1.0), used to select rays for options 28-30 | the image |
| 6 | `pixel_state` | uint8 | (h, w) | which pixels have data (and which were reached) | the image |
| 7 | `xs`, `ys` | float32 | (n,) | the chosen point of each selected pixel, pulled from #4 (a temporary) | one scatter call |
| 8 | `col`, `row` | int64 | (n,) | their pixel indices | one scatter call |
| 9 | `counts` = `App.buddha_counts[c]` | float64 | (h, w) | the histogram (raw, unblurred) | until Start, a seeding change, or a coloration edit |
| 10 | `App.buddha_counts_blurred[c]` | float64 | (h, w) | blurred copy, made on demand | until the next scatter into #9 |

**So before scattering, a "sequence" is present as:** a 28-row slice of 7 reductions per sequence (rows of #4), never as the sequence itself. The simple options read exactly one row of #4 per option. The four multi options need the full sequence (#1, #2) and therefore **re-trace the rays**. The re-trace is bit-for-bit the same ray as the image's own, because the same fuzz seed formula is used.

### Four routes into `counts`

| Route | When | Reads | Code | Parallel? |
|---|---|---|---|---|
| Stored points | camera pixels on, simple options | #4 (float32) | `lightbounces.app_recolor.RecolorMixin.scatter_buddha_rows` (live), `lightbounces.app_render.RenderMixin.apply_coloration` (export), then `lightbounces.metrics.scatter_points_raw` (`np.add.at`) | no (numpy) |
| Re-trace, camera zone | camera pixels on, multi options | re-traced `path_points`, `path_point_tags` (float64) | `lightbounces.kernels.scatter_multi_buddha` = `scatter_from_seeds` with the world rectangle as zone | no (single thread by design: a point can land on any pixel) |
| Re-trace, own zone | camera pixels off, **all** options | re-traced rays from a seed grid over the zone | `lightbounces.metrics.scatter_seeded_buddha` then `lightbounces.kernels.scatter_from_seeds` | no |
| Export / thumbnail | any | same three as above, into a fresh array | `lightbounces.app_render.RenderMixin.apply_coloration` | no |

`scatter_from_seeds(row0, row1, zone, world_rect, ...)` separates **where seeds are** (`zone`, a `WorldRect`; one seed per output pixel, laid at regular spacings) from **what `counts` covers** (`world_rect`). A seed fires only if `fires_ray` is true there. Because the stored-points route reads float32 points and the re-trace route float64, the two "should" agree but can differ by a point on a pixel edge: 2 of 262,144 cells in the default 512×512 image **[V, earlier session]**.

### After scattering

1. **Blur.** `lightbounces.metrics.build_buddha_blur_kernel(radius_px, type)` makes a `(K, 3)` float64 table (row offset, column offset, weight); `lightbounces.metrics.blur_counts_image` turns it into a dense kernel and convolves the **whole** histogram once by FFT (`np.fft.rfft2` on arrays padded to (h + k - 1, w + k - 1)). Cost depends on the image size, not on the radius. The radius in pixels is `radius × min(w, h)` (`lightbounces.metrics.world_radius_to_pixels`, which is not about world units despite its name).
2. **Compress** (`lightbounces.metrics.buddha_compress`): arctangent, (2/π)·atan(count / 8) (8 maps to 0.5, 80 to 0.9365 **[V]**), or modulus, (count % 256) / 255 (300 maps to 0.1725 **[V]**). After the blur the counts are floats, though the docstring says "non-negative int".
3. **Brightness**: multiply by 1, 2, 4, 8 or 16, clip to 1.
4. **Colour**: the value is read at `counts[row, col]` **only for fired pixels**, then it joins the other two channels in `channels_to_rgb`. Consequence: a Buddha image is visible only inside the shapes that fire rays, even where counts exist elsewhere.

**Volume [V]**, default circle, 60 iterations, 256×256, options all-midpoints + progressive bounces + progressive midpoints: 2,842,576 points, about 43 per pixel. **`np.add.at` costs 9.0 ms per 200,000 points; `np.bincount` does the same in 1.6 ms [V]**. That is about 7 ms saved per channel per full image, so it is an optimisation to know about, not a priority.

## 1.11 Stage 7: drawing, exporting, thumbnails, video

- **On screen.** `lightbounces.app_preview.PreviewMixin.draw`: `np.copyto(image_for_export, img)`; obstacle outlines are drawn into `image_for_export` (ring thickness = `OBST_RING_LAYER_PX` pixels of `outline_pixel` world units, so zoom does not thicken them); `image_for_screen` = `image_for_export` + the coloration-name overlay; the pixels go to Tk as a binary PPM (`"P6 w h 255\n"` + `tobytes()`) via `PhotoImage.configure(data=..., format="PPM")`. The hover ray, guide lines and seed-zone outline are Tk canvas items on top, not part of any array.
- **Current image export.** Saves `App.frame` (so outlines yes, name overlay no). With several colorations selected it recolours the stored arrays per coloration through `lightbounces.app_render.RenderMixin.render_current_image` and joins them side by side (`np.concatenate(axis=1)`).
- **New image / video frame.** `lightbounces.app_render.RenderMixin.render_full_image(data, out_w, out_h, obst_arrays, colorations, framing)`: new arrays, one `trace_rows` over all rows with the framing's world rectangle, `apply_coloration` per coloration, outlines from `lightbounces.image_io.make_grid` and `lightbounces.shapes.ring_layer_masks`. A video resolves obstacles, colorations and the framing per frame from keyframes (`lightbounces.settings.resolve_obstacles`, `lightbounces.colorations.resolve_colorations`, `lightbounces.framings.resolve_framing`).
- **Margins.** `lightbounces.export_text.apply_info_locations` adds the coloration text (overlay or bottom margin) and a left margin of figure settings and/or framing settings (`left_margin_blocks`, `add_left_info_margin`); the margin's size depends only on image size and obstacle count, so every video frame matches.
- **Thumbnails.** Coloration thumbnails are of framing 0, framing thumbnails are coloured by coloration 0 (`lightbounces.app_render.RenderMixin.render_coloration_thumbnail`, `render_framing_thumbnail`, `render_thumbnail`); when the image on screen is of framing 0 the first uses the stored arrays (the live one reuses `App.img`), otherwise a small simulation.

## 1.12 Threads and timing

- `trace_rows` is `nogil=True`; its inner driver is a numba `prange` across pixels, so one batch of rows uses all cores. It runs on the Tk main thread during generation, in the time slices described in section 1.8.
- Buddha scattering, blur, recolour and all export colouring are single-threaded numpy or numba. Exports and videos run on a background `threading.Thread` and touch no Tk widget.
- Measured throwaway for orientation **[V]**, a Mandelbrot orbit using the shared `NaturalTerminationRule` checks, 512×512, parallel: 16 ms at 256 iterations for the orbit alone, 43 ms with all four sequences reduced; 41 and 103 ms at 1000 iterations. Reducing all four sequences therefore multiplies the cost of the orbit by about 2.5 to 3.7 (the shorter the orbit, the larger the factor; 9 to 33 ms at 64 iterations), and when rays are long the cost is dominated by `trace_path` itself.

## 1.13 Index: where to look

| To understand | Read |
|---|---|
| the constants and every code space | `lightbounces.kernels` lines 1-300 |
| one ray | `lightbounces.kernels.trace_path` |
| sequences and reductions | `lightbounces.kernels.in_sequence`, `sequence_summary`, `compute_sequences`, `compute_run_measurements` |
| per-pixel driver | `lightbounces.kernels._trace_rows_parallel` |
| generated labels and choice tables | `lightbounces.metrics` (from `SEQUENCE_LABELS` down) |
| the Buddha catalog and compression | `lightbounces.metrics.BUDDHA_POINT_CATALOG`, `buddha_compress` |
| colour models and the coloration records | `lightbounces.colorations` |
| camera | `lightbounces.framings`, `lightbounces.app_preview.PreviewMixin.set_image_framing` |
| live generation | `lightbounces.app_generation.GenerationMixin.tick` |
| live recolouring and Buddha counts | `lightbounces.app_recolor.RecolorMixin` |
| whole-image colouring, exports, thumbnails | `lightbounces.app_render.RenderMixin` |

---

# Part 2 — Assessments

## 2.0 Scorecard

| Area | Verdict | Confidence |
|---|---|---|
| Code cleanliness | Strong foundations (strict records, import-time self-checks, 727 tests, a bit-identical-refactor habit). Four hot spots hold most of the debt (section 2.1). | high: measured |
| Consistency | Very good in the settings layer, weakest in the sequence/colour layer, where code spaces mix types and several words mean two or three things. | medium-high |
| Readiness for multiple sequence engines | **Medium; roughly 5 on a 10-point difficulty scale.** The camera/world-rectangle blocker named in the last report is gone. What remains is a set of well-contained, mostly mechanical global tables and three duplicated drivers. | medium (my estimate, one prototype) |
| Naming | About 14 renames would remove real confusion; roughly half are internal-only and cheap. | high |

## 2.1 Code cleanliness

### Measured **[V]**

- 14,558 lines in `lightbounces/` (36 modules); 746 functions and classes, **64% with a docstring** (475). `lightbounces.kernels` has 51 of 58; `lightbounces.shapes` only 33 of 129 (many one-line methods).
- No `TODO`/`FIXME` markers. 74 lines longer than 120 characters, 19 longer than 160 (the project has no linter or limit, so this is information, not a violation). Spelling: "color" 201 times, "colour" 35 times, in comments and strings.
- Three unused imports (`lightbounces.app_render`, `lightbounces.geometry`, `lightbounces.projects`) were found; **removed in FILE_VERSION 143**.

### What is genuinely good

- **Strict records and spec tables.** One row per setting, validated at construction, enforced at import by `lightbounces.selfchecks.run_self_checks`. This turns drift into a loud failure.
- **The kernel ABI discipline** (one file defines every kernel and every constant a kernel reads, because numba's cache only checks that file's modification time) is documented, tested and has never been broken since it was written.
- **Verification culture.** Golden data, pipeline hashes (38 outputs identical across the framing work), and tests that name the bug they fixed.
- **Layering.** Modules import only from modules above them; the kernels import nothing from the package.

### Hot spots (in order of how much cleanup they would buy)

1. **Duplication of the colour chain.** `RecolorMixin.recolor`, `RenderMixin.apply_coloration` and `RecolorMixin.component_values` each re-implement "channel source to value". Any new channel source or metric type must be added three times. A single function `channel_values(coloration, stats, characteristic_points, selection)` would serve all three.
2. **Duplication of the Buddha option decoder.** Three places decode `BUDDHA_POINT_CATALOG` into "slots, termination filters, multi flags": `RecolorMixin.scatter_buddha_rows`, `RenderMixin.apply_coloration`, `lightbounces.metrics.scatter_seeded_buddha`. The first two identify rays ended by a given reason by comparing the stored float metric to `TERM_STAT_VALUE[...]` within 0.01 (`np.abs(term_stat - target) < 0.01`); the third and the kernels compare the integer `term`. Two mechanisms for one question.
3. **The per-ray chain is repeated.** `trace_path`, then `compute_run_measurements`, then `compute_sequences` appear together in `lightbounces.kernels._trace_rows_parallel` and `lightbounces.app_preview.PreviewMixin.update_preview` (and `warm_up`), and `scatter_from_seeds` has a partial copy. A `trace_and_reduce` kernel would be the natural unit, and is the natural point to put an engine switch (section 2.3).
4. **Four representations of one coloration.** `ColorationConfig` (12 slots, lists), `ColorationSnapshot` (12 fields, tuples), `ColorationSettings` (11 fields, tuples), `ChannelSettings` (9 fields), with `decompose_chans` and `set_channel` converting between them, and ten parallel per-channel sequences instead of three channel records. A `Channel` record in a 3-tuple would remove most of the glue. (Opinion; medium confidence, because the keyframe format stores `ChannelSettings` and a change here needs care.)

### Other findings

- **Size of single units.** `lightbounces.kernels.trace_path` is 365 lines; `lightbounces.selfchecks.run_self_checks` 195; `lightbounces.core.App.__init__` 157 (70 attributes); the biggest UI builders 100-207 lines. `trace_path` is long but linear; splitting it is risky and low-yield. `App.__init__` is the more expensive kind of long, because 256 methods across 14 mixins read and write those 70 attributes implicitly.
- **That implicit coupling has a measurable price.** This session, adding a framing list meant editing about two dozen test fake apps by hand (`_give_image_geometry`, `_give_framings` in `tests/support.py` now exist to stop that).
- **Long parameter lists** invite ordering mistakes: `kernels._trace_rows_parallel` 17, `app_export.ExportMixin._save_new_image_worker` 14 (plus `self`), `kernels.scatter_from_seeds` 14, `app_render.RenderMixin.apply_coloration` 12 (plus `self`), `app_video.VideoMixin._render_worker` 11 (plus `self`), `export_text.apply_info_locations` 11. The four booleans `want_all_self, want_all_mid, want_prog_bounce, want_prog_mid` always travel together and would be one small tuple or bit mask.
- **`lightbounces.kernels` is 2,466 lines, 17% of the package.** About 1,350 of them (lines 279-1634) are shape geometry; `trace_path` starts at 1713. The size is a consequence of the numba cache rule, not of poor structure.
- **A parameter is shadowed.** `lightbounces.kernels.trace_path(x0, y0, path_points, path_point_tags, termination, engine_args, raw)` receives a `NaturalTerminationRule` named `stop`, unpacks it at the top, then reused `stop` as a boolean flag inside the loop (**fixed in FILE_VERSION 147: the parameter is `termination`, the flag `ended`**). It works only because the fields were copied first.
- **Stale comments and user-visible text** (found while writing this guide; none is a bug in computation; **all six items below were corrected in FILE_VERSION 143**, the text shown is what they said before):
  - `lightbounces.app_coloration.ColorationMixin.build_buddha_blur_controls`: the hint text says "A multiple of the view's smallest dimension (2 units): 0.1 means a blur radius of 0.2 units". The radius is `radius × min(w, h)` pixels, i.e. a fraction of the image's shorter side. This is wrong for every non-default framing and **is visible to the user**.
  - `lightbounces.__init__` docstring (lines 3-6): "the simulation itself always runs in that fixed square"; and `lightbounces.constants` (lines 26-31) about the blur radius being "always 2.0".
  - `lightbounces.metrics` (the "Constant source" comment): "Stored in the same slot other_metric[c] already occupies"; true before `constant_choice` existed, false now.
  - `lightbounces.metrics.CHOICE_TOP_LEFT_CORNER` and its description: "the top-left corner of the simulation's [-1, 1] square". With framings it is a world point, not the image's corner.
  - `lightbounces.kernels` module docstring lists the kernel argument types as `FuzzRule, ConfineRule, EmissionRule, ObstacleArrays`; `NaturalTerminationRule`, `RayArgs` and `Window` (now `WorldRect`) were missing. `kernels.PREVIEW_BOX`'s comment says "image size in pixels".
  - `lightbounces.metrics.buddha_compress`'s docstring says "non-negative int"; it receives blurred floats.
- **Performance footnote.** `np.add.at` in `scatter_points_raw` (section 1.10) and a fresh `np.empty` set of five arrays per pixel inside the `prange` loop (**[R]**, believed cheap relative to a trace; not measured).

## 2.2 Consistency

1. **One kind of thing, several number types.** Point tags are *floats* (`TAG_BOUNDARY = 1.0`) because they live in a float64 array and are compared with `==`; every other code family is `int`; the four multi-point Buddha options are *strings* (`"all_self"`) while their siblings are ints; the colour model is the string `"RGB"`/`"HSV"`. A `uint8` tag array and ints everywhere would remove the float comparisons.
2. **Sentinels share a field with real values.** `mapping[c]` and `ColorationConfig.other_metric` hold metric slots 0-23 *or* -1, -2, -3. Anything iterating over them must know to check the sentinels first (`lightbounces.metrics.make_mapping` and every consumer do).
3. **Pixel-to-world and world-to-pixel arithmetic is written out nine times** (four forward, five inverse; section 1.3), always with the same formula. They agree today, and the framing work verified that with hashes; one `Window`-based helper per language (kernels and Python) would make that agreement structural.
4. **Two numeric paths for the same point.** Stored points are float32; re-traced points are float64. The difference is real (2 of 262,144 cells **[V]**) and is documented, but it means "camera pixels" is not exactly equal to "zone = world rectangle".
5. **Normalisation constants are hard-wired to a 2-unit world.** `compute_run_measurements` and `point_comparison_metric` scale distances by 0.5 (a 2-unit world), and the length reduction uses `atan(length / 5)`. They are world-unit measures, so a zoomed framing yields small values: consistent, but arbitrary for an engine whose plane is a different size.
6. **Three families named `KIND_*`, plus a local variable `kinds` that meant two things** (section 2.4). **Fixed in FILE_VERSION 146:** point tags are `TAG_*` and `path_point_tags`; a channel's metric type is `METRIC_TYPE_*` and `metric_types`; `Shape.KIND` is the one `KIND` left.
7. **List versus tuple.** Live coloration fields are lists mutated in place; snapshots and settings are tuples; `make_mapping` returns a tuple. Correct, and a regular source of copy-or-share questions.
8. **Where things disagree on naming the same concept** (all in section 2.4): the share-of-self-bounces metric is called `ST_SELF_SHARE` in code and "Self bounces: Share of the bounces" in the UI; `fires_ray`, "fire rays inside" and "Emissions" are three words for starting a ray.
9. **Good:** error handling is uniform by layer (strict records raise `TypeError`; `lightbounces.parsing.parse_*` return `None` for the entry to show as invalid; files raise `lightbounces.projects.ProjectError`); the project-file keys are permanent and tested; the generated label tables make UI text and file keys consistent by construction.

## 2.3 Readiness for multiple sequence engines (ray bouncing plus the Mandelbrot set)

### What changed since the previous assessment

The largest blocker then, a fixed [-1, 1] world rectangle hard-coded in about ten places, is **gone**: every pixel/world conversion goes through `lightbounces.kernels.WorldRect`, `lightbounces.framings` and `lightbounces.app_preview.PreviewMixin.set_image_framing`, with the default verified bit-identical. A Mandelbrot engine would get pan, zoom and aspect ratio from the Framing tab as it stands. The `zone` versus `world_rect` split in `lightbounces.kernels.scatter_from_seeds` (where seeds start versus what `counts` covers) is also exactly what a Buddhabrot needs.

### What already fits a second engine **[V]**

A throwaway Mandelbrot orbit (z ← z² + c from z = 0, c = the pixel centre) written in about 25 lines against the **existing** `lightbounces.kernels.NaturalTerminationRule`, `beyond_escape_radius`, `is_confined` and `compute_sequences`, in a `prange` over 512×512: it ran at 16 ms (256 iterations) and 43 ms with all four sequences reduced. 83% of the pixels escaped; `Path: Point count / limit` (slot 13) is already the normalised escape time (0.008 to 1.0); every pixel fires. The default escape radius, 2.0, equals the Mandelbrot bailout. Confinement worked as a convergence test for interior points.

### What is ray-specific but sits where an engine-neutral thing would be

| Item (fully qualified) | Why it blocks | Difficulty to generalise |
|---|---|---|
| `lightbounces.kernels._trace_rows_parallel`, `trace_rows`, `scatter_from_seeds`, `scatter_multi_buddha` | call `fires_ray`, `trace_path`, `compute_run_measurements` directly and take `RayArgs`'s pieces as parameters | medium: an engine id switch inside the drivers (numba cannot cleanly dispatch on a function passed into a cached kernel) |
| `lightbounces.kernels.in_sequence` plus the tags `KIND_*` | sequence membership is coded for ray tags; the Mandelbrot prototype had to pretend its points were "boundary bounces" to fill a sequence | medium: bit-mask tags (uint8) and per-engine membership masks |
| `lightbounces.kernels.N_SEQS`, `N_POINTS`, `N_STATS`, `POINT_SLOT`, `STAT_SLOT`, `POINT_ALIASES`, `STAT_ALIASES` | compile-time globals of the ray engine: every pixel stores 28 points and 24 metrics whether an engine has one sequence or four | medium-high: per-engine layouts mean per-engine array shapes, labels, project keys |
| `lightbounces.kernels.compute_run_measurements` and `RAW_*` | run measurements (loop area, bounce angle, last-segment ratio, `seg_mult`) are ray concepts; Mandelbrot would want smooth escape time, final modulus | medium |
| `lightbounces.kernels.TERM_OBSTACLE`, `TERM_PERPENDICULAR` inside the shared `TERM_STAT_VALUE` spacing | adding or removing a reason respaces the termination metric for everyone | low-medium |
| `lightbounces.kernels.fires_ray` and the per-pixel start | a pixel's role differs: ray = start point in a scene; Mandelbrot = parameter c; Julia = z₀ with a fixed c | medium: a start hook per engine |
| `lightbounces.kernels.scatter_trace_points` and the four multi options | hard-wired to the self-bounce sequence, segment midpoints and progressive means; **no option exists for "every point of a sequence, for orbits that ended with reason T"**, which is a Buddhabrot | medium |
| `lightbounces.metrics.SEQUENCE_LABELS` ..., `BUDDHA_POINT_CATALOG`, `CONSTANT_POINT_COORDS` | one global table of labels built at import from the ray engine's sequences | medium: per-engine tables, plus a channel naming a sequence its engine lacks needs a defined behaviour |
| `lightbounces.projects.CAT_SEQUENCE`, `CAT_POINT`, `CAT_METRIC`, `CAT_BUDDHA_POINT` | the permanent file keys are the union of one engine's names (`self_bounces_mean`...) | medium: keys across engines must be a union, added without renaming |
| `lightbounces.core.App.gen_engine_args`, `gen_termination`, `generated_settings`, `settings_of_current_image`; `lightbounces.app_generation.GenerationMixin.tick` | the app holds one engine's arguments under ray names and calls its kernel directly | medium |
| `lightbounces.app_preview.PreviewMixin.update_preview`, the Sequence preview panel (`lightbounces.app_layout.LayoutMixin.build_info_quadrant`), `lightbounces.settings.RaySettings`, the Emissions, Mutations and Obstacles tabs | ray-only UI and preview; the empty Simulation tab is the obvious engine selector | medium-high |

### Mandelbrot-specific needs, one by one

| Need | Status today |
|---|---|
| a pan/zoom plane | **done** (framings; default world rectangle for the set would be a different default framing) |
| every pixel starts an orbit | needs a start rule instead of `fires_ray` |
| escape-time colouring | **available** as `Path: Point count / limit`; *smooth* colouring needs one new run-measurement slot |
| interior detection | **available**: `TERM_ITER` plus confinement as convergence |
| Julia sets | one extra engine setting (c) and a start rule |
| Buddhabrot | needs the "all points of a sequence, filtered by termination" option; and the seed grid is one seed per output pixel, far less sampling than the usual random c over many millions (a design property of the seeding as specified) |
| deep zoom | stored `characteristic_points` are float32 (limit about 6.6×10⁴× the default world rectangle's width, section 1.3); `stats` (escape time) are unaffected, so only point-based features (Buddha simple options, distance metrics) degrade; float64 storage would add 224 bytes per pixel at 28 points (448 instead of 224 for `characteristic_points`, 545 in all) |
| memory | a one-sequence engine pays 321 bytes per pixel for a layout built for four sequences (believed about 90 bytes would do; not measured) |

### The numba cache rule

`lightbounces.kernels` must contain every kernel and every constant a kernel reads, because numba checks only that file's modification time. Putting a second engine's kernels in another file would let them run stale when a shared constant changes. Two answers: keep one file (it grows, with clear section headers), or write a custom cache key that hashes all kernel files (about 50 lines; I believe feasible and have not tried it). This decision should be made *before* the engine files are created.

### Opinion and a suggested order

My view (medium confidence): readiness is good for a **Mandelbrot image behind a switch** and only fair for a **clean engine abstraction**. A hack that hard-wires a second driver would work in a couple of working sessions, but it would bake an engine switch through the drivers, the UI and the project format, and you would redo it. I recommend, in order, each step verifiable against the existing golden data and the 38 pipeline hashes:

1. **Fix the stale text and rename the confusing words** (section 2.4) while nothing depends on them. Low risk, high clarity.
2. **Introduce a `trace_and_reduce` kernel** and make the three call sites use it (section 2.1, item 3). Bit-identical.
3. **Unify the colour chain and the Buddha decoder** (section 2.1, items 1 and 2) so engines touch each once.
4. **Decide the numba cache question**, then create an `Engine` descriptor: its sequences and reductions, slot layout, run-measurement slots, termination reasons, kernel entry points, settings record and tab.
5. **Replace `in_sequence` and the float `path_point_tags` with bit-mask tags** so `sequence_summary` is data-driven (I reported the feasibility and cost of this earlier; it was not done at your request, and nothing else depends on it first).
6. **Generalise the Buddha multi options** to "every point of sequence S, for orbits ended by reason T".
7. **Add float64 storage as an option**, then the Mandelbrot engine itself.

## 2.4 Opportunities to rename things to avoid confusion

Classification of cost: **internal** = identifiers and comments only; **key** = also a permanent project-file key, so it needs a format migration (the project is at format 6) and should normally be left alone, renaming only the attribute; **UI** = user-visible text.

### Status after FILE_VERSION 148

Done, by your decisions on this table: **rows 1, 2, 3, 5 (slimmed), 6, 7, 8 (partly), 9 and the medium-value rows 12, 13, 14, 15, 16, 17**. Not done, by decision: rows 4 (done earlier, in 145), 10, the rest of row 8, and the medium-value rows 19 and 22.

- Row 1: point tags are `TAG_*` (`TAG_START` ... `TAG_PERPENDICULAR`, `TAG_COLOR`); a channel's metric type is `METRIC_TYPE_POINT_PAIR` / `METRIC_TYPE_OTHER` / `METRIC_TYPE_LABELS` / `DEFAULT_METRIC_TYPES`, `ColorationConfig.metric_types`, `ChannelSettings.metric_type`, `CAT_METRIC_TYPE` (the project file's key `kind` is unchanged). `Shape.KIND` is untouched.
- Row 2: `FuzzParams.rng_seed`, `rng_seed_base`, `rng_seed_var` / `rng_seed_entry`, `last_rng_seed`, `randomize_rng_seed`, `parse_rng_seed`, `DEFAULT_RNG_SEED`. The setting key `seed` is unchanged. "Seed" keeps meaning a Buddha start point.
- Row 3: `ConfineRule.window` is `lookback`; the setting and project key `confine_window` is `confine_lookback` (**project format 7**, migration `_migrate_project_6_to_7`); the UI says "Lookback". `Window` is `WorldRect`, `WINDOW_DEFAULT` is `WORLD_RECT_DEFAULT`, `image_window` / `framing_window` are `image_world_rect` / `framing_world_rect`, and the parameters are `world_rect`.
- Row 5 (slimmed): "stat" is the stored side (`stats`, `ST_*`, `STAT_SLOT`, `N_STATS`, `STAT_ALIASES`, `STAT_REDUCTION_*`, `TERM_STAT_VALUE`); "metric" is the chosen side (`METRIC_LABELS`, `METRIC_SHORT`, `METRIC_DESCRIPTIONS`, `POINT_METRICS`, `OTHER_METRICS`, `RUN_METRIC_INFO`, `constant_label_to_metric`..., and the field `ChannelSettings.metric_slot`, formerly `stat`).
- Row 6: `pts` is `path_points`, `kinds` (tags) is `path_point_tags`, the (h, w, 28, 2) array is `characteristic_points` (`App.characteristic_points`), and one ray's (28, 2) float64 array `sp` is `ray_characteristic_points` (`App.hover_ray_characteristic_points`; inside `recolor` the whole-image slice is just `characteristic_points`).
- Row 7: `compute_run_measurements`. Row 9: `trace_rows` / `_trace_rows_parallel`.
- Row 8 (partly): the `RayArgs` bundle is passed around as `engine_args` (`gen_engine_args`, the tests' `_engine_args`); `RayArgs`, `ray_args` and `settings_ray_args` keep their names, and `ray` stays what it is for the `RaySettings` of a project or export. The flag in `trace_path` is `ended`.
- Termination is never shortened to "stop" any more (your instruction in 147): the rule is `NaturalTerminationRule` (parameter `termination`, `gen_termination`), the settings `NaturalTerminationSettings` (`natural_termination`, `generated_natural_termination`, `NATURAL_TERMINATION_SETTING_SPECS`...), the UI section is "Termination limits", and the project file's section is `natural_termination` (**project format 8**, migration `_migrate_project_7_to_8`). The words "stops" / "stopped" in running text that does not name one of these (and `x_stop` / `y_stop`, the end of a range) were left alone.

- Row 12: `ST_BOUNCE_COUNT`, `ST_SELF_SHARE`, `ST_BOUNDARY_SHARE` (formerly `ST_BOUNCES`, `ST_RAY_FRAC`, `ST_BOUNDARY_FRAC`).
- Row 13: `PREVIEW_BOX` (formerly `SIZE`: the longer side of the image on screen) and `DEFAULT_PIXEL_WORLD_WIDTH` (formerly `PIXEL`). Near-conflict: `constants.PREVIEW_SIZE` (150) is the side of the Video panel's frame preview, a different "preview".
- Row 14: `App.image_for_export` (formerly `frame`: the image plus outlines, what is exported) and `App.image_for_screen` (formerly `display_frame`).
- Row 15: `CHOICE_ORIGIN`, `CHOICE_TOP_LEFT_CORNER` (formerly `POINT_ORIGIN`, `POINT_TOP_LEFT_CORNER`).
- Row 16: `fires_ray` (formerly `emits_ray`) and the comments that said a ray was "emitted" from a pixel now say "fired". "Emission" now means the direction only, except in one UI label (Buddha seeding: "use camera pixels as emission seed points") that I left for you to decide.
- Row 17: a coloration's color model is `color_model` (formerly `mode`): `ColorationConfig.color_model`, `COLOR_MODELS` / `COLOR_MODEL_NAMES` / `DEFAULT_COLOR_MODEL`, `color_model_var` / `color_model_combo`, `on_color_model_changed`, and the project keys `color_model` (in a coloration) and `coloration:<id>:color_model` (in a keyframe), through the same format-8 migration as `natural_termination` (no format 9: no format-8 file exists yet). The export margin line still reads "Mode:RGB".

### High value

| # | Current name(s) | The confusion | Suggested |
|---|---|---|---|
| 1 | `lightbounces.kernels.KIND_START`...`KIND_ESCAPED` (point tags, floats); `lightbounces.metrics.KIND_POINT`, `KIND_OTHER` (channel metric type); `Shape.KIND` / `SHAPE_*` (obstacle class); the identifier `kinds` in `trace_path(..., kinds, ...)` (point tags) **and** `ColorationConfig.kinds` (channel types) | **Three unrelated meanings of "kind", and two meanings of the same variable name.** Reading `make_mapping(kinds, ...)` next to `trace_path(..., kinds, ...)` invites mistakes | point tags: `TAG_*` and `tags`; channel types: `METRIC_TYPE_POINT_PAIR`, `METRIC_TYPE_OTHER` and `metric_types`. Internal. (The file keys `point_metric` / `other_metric` are permanent: key) |
| 2 | "seed": `FuzzParams.seed`, `seed_base`, `seed_var`, `last_seed`, `randomize_seed` (RNG) versus `BuddhaSeedSettings`, `scatter_seeded_buddha`, `seed_job`, "seed point grid" (Buddha start points) | two concepts, one word, in the same call (`scatter_from_seeds(..., seed_base, ...)`) | RNG: `rng_seed`, `rng_seed_base`; keep "seed" for Buddha start points, which is the term you chose. Internal (the project key `seed` is permanent) |
| 3 | "window": `ConfineRule.window` / `confine_window` (a count of path points), `lightbounces.kernels.Window` (world rectangle), `App.image_window`, the module `lightbounces.window` (operating-system window placement) | four meanings | `ConfineRule.window` to `lookback`; `Window` to `WorldWindow` or `WorldRect`. `confine_window` is a project key: leave the key, rename the attribute |
| 4 | `status` (now `lightbounces.app_preview.PreviewMixin.pixel_state`) | 0 meant both "no ray fires" and "not computed yet" | **Done in FILE_VERSION 145**: renamed `pixel_state`, three values `PIXEL_PENDING` / `PIXEL_TRACED` / `PIXEL_NO_RAY` |
| 5 | `stats`, `ST_*`, `STAT_SLOT`, `STAT_LABELS`; UI "metric"; module `lightbounces.metrics` | three prefixes and two nouns for the same array | `metrics` (array), `METRIC_*` slots; keep `lightbounces.metrics` |
| 6 | `pts`, `points`, `sp` | path points of one ray / stored sample points of the image / one ray's sample points: three near-identical names for three different arrays | `path_points`, `sequence_points` (the (h, w, 28, 2) array), `ray_sequence_points`; and `kinds` to `path_tags` |
| 7 | `lightbounces.kernels.compute_stats` | it fills only 8 of 24 slots (4 of them placeholders), the name suggests all | `compute_run_measurements` |
| 8 | `ray` / `RayArgs` as the parameter of `trace_path`, `render_rows_wh`, `scatter_*` | reads as "one ray", is the engine's argument bundle; also `stop` (a rule) shadowed by a boolean `stop` inside `trace_path` | `engine_args` / `RayEngineArgs`; `natural_stop`; local flag `ended` |
| 9 | `render_rows_wh`, `render_full_image`, `render_current_image`, `render_*_thumbnail` | "render" means *simulate* in the first and *simulate plus colour* in the others | `trace_rows` (no `_wh`: the fixed-size version no longer exists) for the simulation; keep `render_*` for the full pipeline |
| 10 | `lightbounces.metrics.POINT_TOP_LEFT_CORNER` and its label "Top left corner" | with framings it is the world point (-1, 1), not the image's corner | `POINT_WORLD_TOP_LEFT`; UI label "World point (-1, 1)". The project key `top_left_corner` is permanent: UI plus internal only |
| 11 | `lightbounces.metrics.world_radius_to_pixels` | not about world units; it is a fraction of the image's shorter side | `blur_radius_to_pixels`; and fix the user-visible hint text (section 2.1) |

### Medium value

| # | Current | Issue | Suggested |
|---|---|---|---|
| 12 | `ST_RAY_FRAC`, `ST_BOUNDARY_FRAC`, `ST_BOUNCES` | "ray" here means the ray's *own path* (self bounces), contrary to its use everywhere else | `ST_SELF_SHARE`, `ST_BOUNDARY_SHARE`, `ST_BOUNCE_COUNT` |
| 13 | `lightbounces.kernels.SIZE`, `PIXEL` | `SIZE` is the longer side of the live image, `PIXEL` the world width of one pixel of the *default* framing | `PREVIEW_BOX`, `DEFAULT_PIXEL_WORLD_WIDTH` |
| 14 | `App.frame`, `display_frame` (image buffers) | "frame" also means an animation frame (keyframes, `frame_index`, video frames) | `image_for_export`, `image_for_screen` |
| 15 | `POINT_ORIGIN`, `POINT_TOP_LEFT_CORNER` | they look like slots (like `POINT_START`) but are choice ids with no slot | `CHOICE_ORIGIN`... |
| 16 | `emits_ray`, `fire_inside`, "Emissions" tab | three words for "starts a ray" / "sets its direction" | `fires_ray`; keep "Emissions" for direction only |
| 17 | `ColorationConfig.mode` | the colour model; collides in spirit with `emit_mode`, `confine_method` | `color_model` (project key `mode` permanent) |
| 18 | `generated_settings`, `generated_natural_stop`, `gen_stop`, `gen_ray`, `settings_of_current_image` | five names for "what the image on screen was made with" | one record, `GeneratedFigure`, holding the settings, natural stop, obstacles and kernel arguments |
| 19 | `BUDDHA_MULTI_ALL_SELF = "all_self"`... | strings among ints | ints (or a `MultiOptions` mask) |
| 20 | `ColorationMixin` in `app_coloration.py` | also builds the Buddha blur and Buddha seeding tabs and the thumbnail list | split `app_buddha.py` |
| 21 | `RecolorMixin` in `app_recolor.py` | also owns Buddha counts and hover values | `app_color_pipeline.py` or split |
| 22 | `Simulation` tab (empty) versus the project section `simulation` (which holds the ray engine's settings) | same word, different thing | when the tab becomes the engine selector, call the section `ray_engine` in a format migration |

### Spelling

The code uses "color" in identifiers and a mix of "color" and "colour" in comments and strings; pick one for new text (the identifiers settle it as "color").

---

# Appendix A — What was verified and how

**[V] Run in this session:** every dtype, shape and byte count in sections 1.4 and 1.10; the slot tables and 24 metric / 30 point labels; the 35 Buddha catalog entries; the worked ray and the worked pixel (hand arithmetic equals `apply_coloration` output, [33, 17, 121]); `framing_window(DEFAULT_FRAMING) == WORLD_RECT_DEFAULT`; float32 spacing; `np.add.at` versus `np.bincount`; the Mandelbrot prototype timings and escape fraction; line, function, docstring, import and long-line counts.

**[R] Read, not run:** the control flow of `trace_path`, the per-pixel allocation inside the `prange` loop, the O(bounces²) self-hit scan, the structure and threading of the generation loop, every "duplicated in three places" claim (verified by reading all three bodies), the numba nested-tuple limitation (documented in `CLAUDE.md`, verified in an earlier session), the claim that about 90 bytes per pixel would suffice for a one-sequence engine, and the 2-of-262,144 float32/float64 disagreement (measured in an earlier session, not re-run today).

# Appendix B — Reproduce the key numbers

```python
# From the project folder, in a Python session:
import numpy as np
from tests.support import m                        # the loaded package
arrays = m.build_obstacle_arrays([m.default_starting_obstacle(0)])
termination = m.natural_termination_rule(40, float("inf"), 2.0, m.CONFINE_DEFAULT)
engine_args = m.ray_args(1.0, m.fuzz_rule(m.DEFAULT_FUZZ), m.EMISSION_DEFAULT, arrays)

w, h = 64, 48
stats = np.zeros((h, w, m.N_STATS), dtype=np.float32)
characteristic_points = np.zeros((h, w, m.N_POINTS, 2), dtype=np.float32)
pixel_state = np.zeros((h, w), dtype=np.uint8)
m.trace_rows(stats, characteristic_points, pixel_state, 0, h, w, h, m.WORLD_RECT_DEFAULT, termination, engine_args, 0)

print(stats.itemsize * m.N_STATS + characteristic_points.itemsize * m.N_POINTS * 2 + pixel_state.itemsize)   # 321 bytes per pixel
print(m.POINT_SLOT)                                # the (4, 7) point slot table
print(m.STAT_SLOT)                                 # the (4, 4) scalar slot table
i, j = 40, 24                                      # the worked pixel of section 1.9
print(stats[j, i, [0, 1, 4, 11]])                  # 0.12794  0.06597  0.475  0.2
print(characteristic_points[j, i, [m.POINT_START, m.POINT_END]])  # (0.265625, -0.02083) and (0.52150, -0.15277)
```
