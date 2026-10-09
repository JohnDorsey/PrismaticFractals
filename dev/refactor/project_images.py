"""The image each project file makes, as hashes: proof that a change to how project files are read or written (a new
format, a migration, new defaults) leaves every real file drawing the same picture.

    python dev/refactor/project_images.py <out.json>                  record
    python dev/refactor/project_images.py <out.json> <baseline.json>  record and compare (exit code 1 if different)

Every file of projects/ and dev/testing/project_v*.lightbounce is loaded and rendered at 96 x 72 pixels the way a "New
image" export renders it: with the project's own settings, obstacles, colorations and selected framing (render
"current"), and at the frame of each of its keyframes (render "kf<frame>"; a setting no keyframe defines is the project's
own value, as in an export), through App.render_full_image. A file that
does not load is recorded with the error text, so a migration that breaks one shows up as a difference."""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np                                                 # noqa: E402
import tests.support as sup                                        # noqa: E402

m = sup.m
W, H = 96, 72


def render(project, keyframes, frame, base):
    """The picture of `project` at `frame` of `keyframes` (a one-off keyframe of its current values for "current").
    `base` is every static setting as the project holds it: what an export falls back to for a setting no keyframe
    defines."""
    data = m.resolve_frame_settings(keyframes, frame, base)
    obst_values = m.resolve_obstacles(list(project.obstacles), keyframes, frame)
    colorations = m.resolve_colorations(list(project.colorations), keyframes, frame)
    framing_by_id = {f.id: f for f in project.framings}
    framing = m.resolve_framing(framing_by_id.get(project.selected_framing_id, project.framings[0]), keyframes, frame)
    app = sup._fake_render_app()
    rgb = app.render_full_image(data, W, H, m.build_obstacle_arrays(obst_values), colorations, framing)
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()[:16]


def hashes():
    res = {}
    files = sorted(glob.glob(os.path.join("projects", "*.lightbounce")) + glob.glob(os.path.join("dev", "testing", "project_v*.lightbounce")))
    for path in files:
        name = path.replace(os.sep, "/")
        try:
            project = m.load_project(path)
        except Exception as exc:                                    # noqa: BLE001
            res[name] = {"load error": str(exc)}
            continue
        entry = {}
        values = m.settings_to_dict(project.ray, project.natural_termination, project.buddha_blur, project.buddha_seed,
                                   project.engine)
        values["outline"] = bool(project.decorations.get("outline", True))
        entry["current"] = render(project, [m.Keyframe(-1, 0, values)], 0, values)
        ordered = sorted(project.keyframes, key=lambda k: k.frame_index)
        for frame in sorted({k.frame_index for k in ordered}):
            entry[f"kf{frame}"] = render(project, ordered, frame, values)
        res[name] = entry
    return res


if __name__ == "__main__":
    out = hashes()
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    print(len(out), "files;", sum(len(v) for v in out.values()), "pictures hashed; load errors:",
          [k for k, v in out.items() if "load error" in v])
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            before = json.load(f)
        bad = [k for k in sorted(set(out) | set(before)) if out.get(k) != before.get(k)]
        for k in bad:
            print("DIFFERENT ", k)
        print("IDENTICAL to" if not bad else f"{len(bad)} files DIFFERENT from", sys.argv[2])
        sys.exit(1 if bad else 0)
