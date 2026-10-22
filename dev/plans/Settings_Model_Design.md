# Settings model: design note and spike plan

Status: PROPOSAL, nothing built. Needs the author's sign-off on the choices in "Decisions to confirm" before the spike starts.

## Why

Today the Tk controls ARE the program's state for the 37 "static" settings (the six records of `SETTING_TABLES`). Verified in the code:

- `current_natural_termination_settings()` (and the ray, blur, seeding, engine and Mandelbrot ones) build a record by reading widget variables (`self.iter_var.get()`, `self.seg_mult_scale.get()`...); `capture_all_settings()` is the sum of them, and an export reads `*_var.get()` directly (22 calls in `app_export.py`).
- Everything that makes an entry usable is written by hand, once per setting: a `last_<name>` attribute for the last valid value (`core.py` 89-97), a parser call and an `Invalid.TEntry` style switch in `on_limit_var_changed`, an error sentence, a row in `on_limit_commit`, a `trace_add` in `core.py` 181-186, and a copy of the whole thing in `preview_natural_termination_settings` (`app_preview.py` 163-172). `set_setting_control` is the one setter for loading a value into a control.
- A custom UI, a headless driver, or Canopy writing a value would each have to re-implement all of that.

Not a problem (do not touch): obstacles, colorations and framings are already model objects (`App.obstacles`, `.colorations`, `.framings`); the keyframe code only reads and writes them. Only the six records are widget-held.

## Proposal

A new pure module `settings_model.py` (layer 3, after `settings.py`; imports no Tk; covered by the layering test). One object, `SettingsModel`, owned by `App` as `self.model`:

- Holds, per setting key, the **last valid value** and an **error** (None, or the message for the text last submitted).
- `model.value(key)`, `model.record(RecordClass)` (the record from the current values, or **None while any of its settings has an error** - exactly what `current_*_settings()` returns now), `model.error(key)`.
- `model.submit(key, text)`: the view reports what the user typed; the model parses it with the setting's parser. Valid: the value changes, the error clears. Invalid: the value stays, the error is set. Returns the error.
- `model.set(key, value)`: programmatic write (keyframe load, Revert, project open, Canopy). Clears any error. This replaces `set_setting_control` as the one setter.
- `model.subscribe(listener)`; a listener gets the set of changed keys. `with model.batch():` defers notification to one call at the end (replaces `kf_applying` plus the hand-batched refreshes).
- A parser table, one row per setting: text -> value or None, and the error sentence. It is the existing `parse_*` functions plus their messages, moved next to the spec rows (`SettingSpec` gains `parse` and `message`).

The window becomes a **view** of it. A small binding helper in the window layer connects one control: typing calls `submit`, a model change rewrites the control, `error` drives the red style and the message label, focus-out rewrites the text from `value` (the current "put back the last valid value"). The Tk variable stays as the widget's display buffer; it is no longer read by anyone else. `preview_*_settings` reads `model.value` for the last valid values (no more copies of the parsing).

Everything the spec tables already say (path, kind, names, tab, label) drives the binding generically, so the 37 settings need rows, not hand-written glue.

## What stays as it is

- The records (`NaturalTerminationSettings`...), the spec tables, `settings_to_dict` / `settings_from_dict`, projects, keyframes, `GeneratedFigure`.
- **The two polling loops** (`poll_modified` at 100 ms, `poll_project_title` at 1000 ms). Correction to what I said earlier: the model does not remove them. They deliberately catch every edit path, including the obstacle, coloration and framing editors, which do not go through the model. Events could replace them only after those three also publish changes; I would not plan on it, and the polling is cheap (the code measures ~0.06 ms a capture).
- Tk widgets, layout, the keyframe machinery.

## Spike (one family, to measure the real cost)

Family: **natural termination** (6 settings, all four text kinds plus one choice, so it covers the awkward parts: int, float, "0 = none", a dropdown, the confinement method greying the threshold label).

Changes: `settings_model.py` (the model, the parser rows for this family); the binding helper; `current_natural_termination_settings`, `on_limit_var_changed` / `on_limit_commit` / `core.py` traces for these six, the six `last_*` attributes, `preview_natural_termination_settings`, and the keys' branch in `set_setting_control` go through the model. The other 31 settings are untouched, and `capture_all_settings` / keyframes / projects keep working because they still call `current_natural_termination_settings()`.

Done means, all measured:
1. The existing 913 tests pass **unmodified** (the first rule for a risky refactor); new model tests (a parse table test, error/last-valid semantics, batch notification, record() is None while invalid) run with no Tk.
2. `project_roundtrip_check.py` and `engine_switch_check.py` pass in the real window, and I type into the entries and check the red style, message, commit-restores, keyframe select / modified outline / Revert by hand in the real window.
3. A report of lines added and removed, and how many places a seventh setting would need to touch (today: about seven).
4. A verdict: roll out to the other five records (one family per step, each verified), or stop and say why.

Work on a branch (`settings-model-spike`), merged only if the verdict is "roll out". FILE_VERSION is bumped once per merged step.

## Risks

- **Timing of invalid states.** Today `current_*` returns None as the user types garbage, and Start / the buttons depend on it. `record()` must keep that exactly; tests cover it, but the real-window check is what proves the focus-out and keyframe-selection edge cases.
- **Two sources of truth during the transition.** While some families are migrated and others not, a reader must use the right one. Mitigation: migrate a whole record at a time, and delete the old `last_*` and parsing code in the same step (never leave both).
- **Loops** (model change -> control write -> trace -> submit). The binding ignores the write it made itself; tested headlessly with a fake control.
- **Test fakes.** Many fake apps in `support.py` set `current_*_settings` lambdas; they keep working because those remain the public readers. Nothing in them needs to know about the model.
- Cost is unmeasured until the spike; my guess (belief, not measurement) is a day's work for the spike and a step per remaining record.

## Decisions (confirmed by the author)

1. The model holds values only; the view owns draft text.
2. `SettingSpec` gains `parse` and `message` at roll-out (the spike used a separate table, `NATURAL_TERMINATION_PARSERS`).
3. Polling stays for now. **Events for everything is a later goal** (obstacles, colorations and framings would also have to publish changes before `poll_modified` and `poll_project_title` can go).
4. The spike is on the branch `settings-model-spike`, merged only on a "roll out" verdict.
5. This note lives in `dev/plans/`.

## Spike result (natural termination, FILE_VERSION 178)

Verified:
- Full suite 927 passed (913 before plus 14 new, none Tk). The real-window checks `project_roundtrip_check.py`, `engine_switch_check.py` and the new `settings_model_check.py` (types into the real entries) pass. `image_hashes.py` is IDENTICAL to its baseline; `project_images.py` gives the same hashes as the code before the spike (3 files differ from its old FILE_VERSION 154 baseline both before and after, so that is not the spike's).
- Production code: +249 / -97 lines of which the model is 151 (new file, reused by every later family); `app_controls.py` shrank 862 -> 826 lines and `core.py` lost its six `last_*` attributes and six traces.
- A seventh natural termination text setting now needs: the record field and default, its spec row, its parser row (with its error sentence), the control, and a line in `on_model_changed`'s message grouping when it has a message. No handler, no `last_*`, no trace, no commit row, no preview copy. (Before: about ten places.)

Deviation from the plan's first criterion ("existing tests pass unmodified"): 30 tests could not, because their fixtures set `app.iter_var` / `app.confine_*_var` and expected `current_*` to read them (the widgets-as-state design). The assertions are unchanged except that three read `app.model.value(...)` where they read `app.last_*`; the fixtures now get a real model through `_give_settings_model` (`dev/tests/support.py`) and tests type with `var.set(...)`. Every remaining fake-app test of another family will need the same one-line fixture change when its family moves.

Found on the way: generated key events (`event_generate("<Return>")`) are not delivered when the window has no real focus, so the real-window check asserts that the binding exists and calls `on_limit_commit`; and `FloatEntryHandle` (the number entries of the other families) has the same last-valid logic written again, which the roll-out replaces.

Verdict: roll out. Next family: the Buddha blur and seeding (smallest, no engine coupling), then the ray settings, the engine and orbit settings last.

## Rollout 1: Buddha blur and Buddha seeding (FILE_VERSION 179)

Done, on `main` (the spike branch was fast-forward merged first). The model now holds 13 settings: the six natural termination ones, the two blur ones and the five seeding ones.

- `SettingSpec` gained `parse`, `message` and `lenient` (decision 2); `NATURAL_TERMINATION_PARSERS` and `SettingParser` are gone; `MODEL_SETTING_SPECS` is the rows that have a parser, and `SettingsModel(specs)` is built from them.
- **Lenient settings** (new): text that is no number is silently ignored and the last valid value stays; no error is held or shown, `record()` is never None. The blur radius (also clamped into its range) and the seeding zone behave this way, exactly as `FloatEntryHandle.get` and `last_buddha_seed` did. A checkbox (`bool` kind) holds a bool.
- Both families react when committed or when the program loads them, not on every keystroke (as before); only the natural termination keys trigger `on_model_changed`'s refresh. A reaction style per family is therefore a property of the family's handlers, not of the model.
- The blur radius is no longer a `FloatEntryHandle` (`add_model_entry` builds label, entry and variable from the spec row; the control attributes are now `buddha_blur_radius_var` / `buddha_blur_radius_entry`). `last_buddha_seed` is gone.
- Bindings are derived from the spec rows (`spec.control`, `spec.widget`): a new managed setting needs no binding code.
- Checks as for the spike: 932 tests, the three real-window checks plus `project_roundtrip_check.py` (which wrote the blur radius through the old handle and now uses `model.set`) and `engine_switch_check.py`, image hashes identical, project pictures identical to before. No test needed more than a fixture change; only the seeding fixture was rebuilt around a real model.
- Found: a real-window check must not leave an entry invalid before capturing a project (`capture_project` then raises a modal dialog and the check hangs); the check now types a valid value first.
- Next: the ray settings (segment multiplier, emission, fuzzing, seed) - these use `FloatEntryHandle` throughout (the sliders-as-entries of `add_angle_entry`), so that family also retires `FloatEntryHandle`.
