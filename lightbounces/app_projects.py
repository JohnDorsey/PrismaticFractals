"""The File menu and project handling of the window: capturing the whole program state as a Project, applying a Project (Open and New alike), unsaved-change tracking, the window title, and the Save / Save As / Open / New / close flows."""
import os
import sys
from tkinter import filedialog, messagebox
from .constants import PROJECT_POLL_MS
from .paths import ensure_project_dir
from .projects import (
    EXPORT_OPTIONS, PROJECT_EXTENSION, PROJECT_FILE_TYPES, Project, ProjectError, _is_number,
    load_project, project_signature, save_project, coloration_config_from_snapshot
)
from .settings import (
    DEFAULT_FUZZ, DEFAULT_RAY_SETTINGS, EMISSION_DEFAULT, NATURAL_TERMINATION_DEFAULT, fuzz_rule, ray_args,
    settings_to_dict, snapshot_keyframes
)
from .shapes import build_obstacle_arrays
from .colorations import snapshot_coloration
from .framings import framing_config_from_snapshot, snapshot_framing


class ProjectsMixin:
    # ---- projects ---------------------------------------------------------- #
    def start_project_handling(self):
        """Called once the window is built: the starting state is the "new project" one;
        remember it (File > New returns to it) and when a project is saved or has unsaved
        changes."""
        self.project_path = None
        self.default_project = self.capture_project()
        self.saved_signature = project_signature(self.default_project)
        self.title_shown = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)
        for sequence, command in (("<Control-n>", self.new_project),
                                  ("<Control-o>", self.open_project),
                                  ("<Control-s>", self.save_project),
                                  ("<Control-Shift-S>", self.save_project_as)):
            self.root.bind_all(sequence, lambda _event, command=command: command())
        self.update_title()
        self.root.after(PROJECT_POLL_MS, self.poll_project_title)

    def export_option_control(self, path):
        control = self
        for name in path.split("."):
            control = getattr(control, name)
        return control

    def capture_export_options(self):
        """{EXPORT_OPTIONS key: value} from the Export designer and video panel controls."""
        values = {}
        for key, path, kind, _allowed in EXPORT_OPTIONS:
            value = self.export_option_control(path).get()
            if kind == "number":
                value = float(value)
                value = int(value) if value.is_integer() else value
            elif kind == "bool":
                value = bool(value)
            else:
                value = str(value)
            values[key] = value
        return values

    def apply_export_options(self, values):
        """Put saved Export designer / video choices into their controls. A value that is
        missing, of the wrong kind, or not one of the allowed choices leaves its control alone."""
        for key, path, kind, allowed in EXPORT_OPTIONS:
            value = values.get(key)
            if kind == "bool":
                ok = isinstance(value, bool)
            elif kind == "number":
                ok = _is_number(value)
            elif kind == "choice":
                ok = value in allowed.codes
            else:
                ok = isinstance(value, str)
            if ok:
                self.export_option_control(path).set(value)

    def capture_project(self, show_error=True):
        """A Project holding everything now set up (None, after telling the user unless
        `show_error` is False, while an entry holds an invalid value)."""
        current = self.current_generation_settings()
        if current is None:
            if show_error:
                messagebox.showerror("Invalid settings", "Fix the highlighted entries first.")
            return None
        ray, natural_termination = current
        selected = self.selected_obstacle
        return Project(
            ray=ray, natural_termination=natural_termination, buddha_blur=self.current_blur_settings(),
            buddha_seed=self.current_buddha_seed_settings(),
            decorations={key: bool(var.get()) for key, var in self.deco.items()},
            obstacles=tuple(o.snapshot() for o in self.obstacles),
            colorations=tuple(snapshot_coloration(v) for v in self.colorations),
            framings=tuple(snapshot_framing(f) for f in self.framings),
            keyframes=tuple(snapshot_keyframes(self.keyframes)),
            export=self.capture_export_options(), obstacle_counter=self._obst_counter,
            coloration_counter=self._coloration_counter, framing_counter=self._framing_counter,
            keyframe_counter=self._kf_counter,
            selected_obstacle_id=selected.id if selected is not None else -1,
            selected_coloration_id=self.selected_coloration_id, selected_framing_id=self.selected_framing_id)

    def discard_generated_image(self):
        """Back to the blank state before the first Start (stopping a generation in progress)."""
        if self.tick_id is not None:
            self.root.after_cancel(self.tick_id)
            self.tick_id = None
        self.running = False
        self.done = False
        self.generated_settings = None
        self.generated_natural_termination = None
        self.gen_termination = NATURAL_TERMINATION_DEFAULT
        self.gen_engine_args = ray_args(DEFAULT_RAY_SETTINGS.seg_mult, fuzz_rule(DEFAULT_FUZZ), EMISSION_DEFAULT,
                                build_obstacle_arrays([]))
        self.gen_obstacles_sig = ()
        self.gen_obstacle_snapshots = []
        self.gen_obstacles_by_id = {}
        self.set_image_framing(self.selected_framing_snapshot())    # (a blank image of the selected framing)
        self.batch_rows = 1
        self.thumb_sims = {}
        self.coloration_thumb_cache = {}
        self.canvas.delete("pending")
        self.pinned_px = None
        self.clear_overlay()

    def apply_project(self, project):
        """Make the program hold exactly `project`: the controls, obstacles, colorations, keyframes
        and export choices are replaced, and the image goes blank until Start. Used by
        File > Open and File > New alike, so a loaded project cannot differ from a fresh one."""
        self.discard_generated_image()
        self.kf_applying = True
        try:
            for key, value in settings_to_dict(project.ray, project.natural_termination, project.buddha_blur,
                                               project.buddha_seed).items():
                self.set_setting_control(key, value)
            self.baseline = project.ray
            self.baseline_natural_termination = project.natural_termination
            for key, value in project.decorations.items():
                if key in self.deco:
                    self.deco[key].set(value)
            self.obstacles = [o.duplicate(o.id) for o in project.obstacles]
            self._obst_counter = project.obstacle_counter
            self.selected_obstacle = next(
                (o for o in self.obstacles if o.id == project.selected_obstacle_id),
                self.obstacles[0] if self.obstacles else None)
            self.colorations = [coloration_config_from_snapshot(v) for v in project.colorations]
            self._coloration_counter = project.coloration_counter
            self.selected_coloration_id = project.selected_coloration_id
            self.framings = [framing_config_from_snapshot(f) for f in project.framings]
            self._framing_counter = project.framing_counter
            self.selected_framing_id = project.selected_framing_id
            self.keyframes = snapshot_keyframes(project.keyframes)
            self._kf_counter = project.keyframe_counter
            self.selected_keyframe = None
            self.kf_modified = set()
            self.kf_last_seen = None
            self.apply_export_options(project.export)
        finally:
            self.kf_applying = False
        self.on_limit_var_changed()
        self.on_fuzz_changed()
        self.update_fuzz_enabled_state()
        self.on_save_source_changed()
        self.on_save_format_changed()
        self.on_export_format_changed()
        self.on_bitrate_mode_changed()
        self.on_downsample_factor_changed()
        self.refresh_obstacle_editor()
        self.rebuild_obstacle_list()
        self.set_image_framing(self.selected_framing_snapshot())    # (the image is blank: nothing generated yet)
        self.refresh_framing_controls_for_selected_framing()
        self.refresh_color_controls_for_selected_coloration()
        self.rebuild_coloration_list()                  # (which also rebuilds the framings' list)
        self.refresh_live_preview_state()
        self.on_buddha_seeds_changed()
        self.on_decoration_changed("outline")
        self.on_decoration_changed("pending")
        self.on_shape_param_changed()
        self.rebuild_keyframe_list()
        self.refresh_timeline_diffs()
        self.refresh_modified_outlines()
        self.update_buttons()
        self.draw()

    def project_display_name(self):
        return os.path.basename(self.project_path) if self.project_path else "Untitled"

    def is_dirty(self):
        """Whether there are changes not saved in the project (an entry holding an invalid
        value counts as one)."""
        project = self.capture_project(show_error=False)
        if project is None:
            return True
        try:
            return project_signature(project) != self.saved_signature
        except ProjectError:                       # something that cannot be saved is a change
            return True

    def update_title(self):
        title = f"{'*' if self.is_dirty() else ''}{self.project_display_name()} - " \
                f"{os.path.basename(sys.argv[0]) or 'Light bounces'}"
        if title != self.title_shown:
            self.title_shown = title
            self.root.title(title)

    def poll_project_title(self):
        try:
            self.update_title()
        finally:
            self.root.after(PROJECT_POLL_MS, self.poll_project_title)

    def mark_project_saved(self):
        """The project as it now is counts as saved (after New, Open and Save)."""
        project = self.capture_project(show_error=False)
        if project is not None:
            self.saved_signature = project_signature(project)
        self.update_title()

    def can_replace_project(self, action, check_busy=True):
        """Whether the current project may be replaced (or the program closed): not while a
        video renders, and if it has unsaved changes only once the user has chosen to save
        them or not."""
        if check_busy and self.busy_with_render_or_reencode():
            messagebox.showinfo("A video is rendering",
                                f"Wait for the video render to finish (or cancel it) before "
                                f"you {action}.")
            return False
        if not self.is_dirty():
            return True
        answer = messagebox.askyesnocancel(
            "Unsaved changes",
            f"Save the changes to {self.project_display_name()} before you {action}?")
        if answer is None:
            return False
        return self.save_project() if answer else True

    def new_project(self):
        if not self.can_replace_project("start a new project"):
            return
        self.apply_project(self.default_project)
        self.project_path = None
        self.mark_project_saved()

    def open_project(self):
        path = filedialog.askopenfilename(
            parent=self.root, title="Open project", initialdir=self.project_dialog_dir(),
            filetypes=PROJECT_FILE_TYPES)
        if not path:
            return
        try:
            project = load_project(path)
        except ProjectError as exc:
            messagebox.showerror("Could not open project", f"{path}\n\n{exc}")
            return
        if not self.can_replace_project("open another project"):
            return
        self.apply_project(project)
        self.project_path = path
        self.mark_project_saved()

    def project_dialog_dir(self):
        return os.path.dirname(self.project_path) if self.project_path else ensure_project_dir()

    def save_project(self):
        """Save to the project's file (asking for one first if it has none). True if saved."""
        if self.project_path is None:
            return self.save_project_as()
        return self.save_project_to(self.project_path)

    def save_project_as(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save project as", initialdir=self.project_dialog_dir(),
            initialfile=self.project_display_name() if self.project_path
            else "untitled" + PROJECT_EXTENSION,
            defaultextension=PROJECT_EXTENSION, filetypes=PROJECT_FILE_TYPES)
        return bool(path) and self.save_project_to(path)

    def save_project_to(self, path):
        project = self.capture_project()
        if project is None:
            return False
        try:
            save_project(path, project)
        except ProjectError as exc:
            messagebox.showerror("Could not save project", str(exc))
            return False
        self.project_path = path
        self.saved_signature = project_signature(project)
        self.update_title()
        return True

    def on_close_window(self):
        if self.can_replace_project("close the program", check_busy=False):
            self.root.destroy()
