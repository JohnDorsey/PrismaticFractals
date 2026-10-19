"""Framings and the Framing tab: the list of framings (the camera: where it is and how far it sees) with thumbnails, creating/duplicating/deleting/selecting one, the entries that edit the selected one, and what a change of framing does to the image on screen (generate it again, as seen by the new framing)."""
import tkinter as tk
from tkinter import ttk
from .constants import RIGHT_W, SELECTED_OUTLINE, SELECTED_OUTLINE_PX, THUMB_SIZE
from .framings import (
    FRAMING_FIELDS, FramingConfig, framing_is_usable, same_framing, snapshot_framing
)
from .image_io import encode_png_bytes
from .parsing import parse_positive_number, parse_target_coordinate
from .widgets import ThumbnailStrip

# How each of a framing's entries is read: its parser, the label beside it, and the key-press check it gets.
FRAMING_ENTRIES = (
    ("x", "X (world units):", parse_target_coordinate, "validate_signed_float"),
    ("y", "Y (world units):", parse_target_coordinate, "validate_signed_float"),
    ("diagonal", "Diagonal (world units):", parse_positive_number, "validate_positive_float"),
    ("aspect", "Aspect ratio (width / height):", parse_positive_number, "validate_positive_float"),
)


class FramingMixin:
    # ---- framings (Framing tab) -------------------------------------------------------------- #
    def framing_by_id(self, framing_id):
        """The live FramingConfig for one framing id, or None if it doesn't exist."""
        for f in self.framings:
            if f.id == framing_id:
                return f
        return None

    def selected_framing_obj(self):
        return self.framing_by_id(self.selected_framing_id)

    def selected_framing_snapshot(self):
        """The selected framing as a FramingSnapshot: what the image is made as, and what exports and background
        threads read."""
        return snapshot_framing(self.selected_framing_obj())

    def all_framing_ids(self):
        """Every framing's id, in list order."""
        return [f.id for f in self.framings]

    def new_framing(self):
        """Add a new (default: the whole world) framing right after the selected one, and select it - which also
        makes it the one the image is of."""
        f = FramingConfig(self._framing_counter)
        self._framing_counter += 1
        idx = self.all_framing_ids().index(self.selected_framing_id)
        self.framings.insert(idx + 1, f)
        self.selected_framing_id = f.id
        self.on_framing_changed()

    def duplicate_framing(self):
        """A copy of the selected framing, inserted right after it and selected."""
        src = self.selected_framing_obj()
        if src is None:
            return
        f = FramingConfig(self._framing_counter)
        self._framing_counter += 1
        for field in FRAMING_FIELDS:
            setattr(f, field, getattr(src, field))
        idx = self.all_framing_ids().index(self.selected_framing_id)
        self.framings.insert(idx + 1, f)
        self.selected_framing_id = f.id
        self.on_framing_changed()

    def delete_framing(self):
        """Removes the selected framing, unless it's the only one left; framing 0 becomes the selected one."""
        if len(self.framings) <= 1:
            return
        f = self.selected_framing_obj()
        if f is None:
            return
        self.framings.remove(f)
        self.purge_keyframe_settings(f"framing:{f.id}:")
        self.selected_framing_id = self.framings[0].id
        self.on_framing_changed()

    def select_framing(self, framing_id):
        """Selects a framing - which also makes it the one the image on screen and every export are of."""
        if framing_id == self.selected_framing_id:
            return
        self.selected_framing_id = framing_id
        self.on_framing_changed()

    def on_framing_changed(self):
        """The selection of framings or the selected framing itself changed (another one selected, one made, copied
        or deleted, or its numbers edited): show its numbers, make the image an image of it, and refresh the
        thumbnails."""
        self.refresh_framing_controls_for_selected_framing()
        self.apply_selected_framing()
        self.rebuild_coloration_list()          # (which also rebuilds the framings' list)

    def apply_selected_framing(self):
        """Make the image on screen an image of the selected framing. If one has been generated, that is a new
        generation (of the same figure: the settings it was started with, not the controls' present ones); until the
        first Start there is only the blank image, which just takes the new shape. Nothing happens when the image is
        of those very numbers already. A pinned ray preview is let go: its pixel is not the same place any more."""
        snapshot = self.selected_framing_snapshot()
        if same_framing(self.image_framing, snapshot):
            self.image_framing = snapshot       # (the same camera; maybe another framing's id)
            return
        self.pinned_px = None
        self.canvas.configure(cursor="crosshair")
        if self.generated is not None:
            self.begin_generation()
            return
        self.set_image_framing(snapshot)
        self.clear_overlay()
        self.refresh_live_preview_state()
        self.update_buttons()

    # ---- the Framing tab's controls ---------------------------------------------------------- #
    def build_framing_controls(self, parent):
        """The framings list (New/Dupe/Del + a strip of thumbnails like the Colorations tab's), then the
        "Edit selected framing" panel: where the camera is (x, y: the middle of what it sees), how far across it sees
        (the diagonal of what it sees) and the shape of that (width / height). Entries commit on Enter or when
        focus leaves, an invalid one going back to its last valid value."""
        self.framing_strip = ThumbnailStrip(
            parent, THUMB_SIZE, self.small_font, SELECTED_OUTLINE, SELECTED_OUTLINE_PX, self.new_framing,
            self.duplicate_framing, self.delete_framing, self.select_framing)
        self.framing_strip.frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))

        edit = ttk.LabelFrame(parent, text="Edit selected framing")
        edit.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))
        edit.columnconfigure(0, weight=1)
        framing = self.selected_framing_obj()
        self.last_framing = {field: getattr(framing, field) for field in FRAMING_FIELDS}
        for row, (field, label, _parse, validator) in enumerate(FRAMING_ENTRIES):
            var = tk.StringVar(value=str(self.last_framing[field]))
            setattr(self, f"framing_{field}_var", var)
            ttk.Label(edit, text=label).grid(row=row, column=0, sticky="w", padx=(8, 6), pady=2)
            vcmd = (self.root.register(getattr(self, validator)), "%P")
            entry = ttk.Entry(edit, textvariable=var, width=20, justify="right", validate="key", validatecommand=vcmd)
            entry.grid(row=row, column=1, padx=(0, 8), pady=2)
            for event in ("<FocusOut>", "<Return>", "<KP_Enter>"):
                entry.bind(event, self.on_framing_commit)
            setattr(self, f"framing_{field}_entry", entry)
        ttk.Label(edit, foreground="#666", wraplength=RIGHT_W - 70,
                  text="The selected framing is what the image on screen and every export show. Changing it "
                       "generates the image again, of the same figure (the settings it was started with). "
                       "Aspect ratio 1 and diagonal 2.828... (2 sqrt 2) show the whole world, -1 to 1 both "
                       "ways. The image's shape follows the aspect ratio; an export's own width and height "
                       "stretch the picture if they differ.").grid(
            row=len(FRAMING_ENTRIES), column=0, columnspan=2, sticky="w", padx=8, pady=(4, 6))
        self.rebuild_framing_list()

    def refresh_framing_controls_for_selected_framing(self):
        """Show the selected framing's numbers in the entries (nothing, if the tab is not built yet)."""
        if not hasattr(self, "framing_x_var"):
            return
        framing = self.selected_framing_obj()
        if framing is None:
            return
        for field in FRAMING_FIELDS:
            self.last_framing[field] = getattr(framing, field)
            getattr(self, f"framing_{field}_var").set(str(getattr(framing, field)))

    def on_framing_commit(self, _event=None):
        """Enter pressed or focus left an entry: tidy every entry's text (nonsense, or numbers that would leave the
        framing no rectangle of the world to see, go back to the last valid values) and, if the selected framing changed,
        make the image an image of it."""
        framing = self.selected_framing_obj()
        if framing is None:
            return
        values = {}
        for field, _label, parse, _validator in FRAMING_ENTRIES:
            value = parse(getattr(self, f"framing_{field}_var").get())
            values[field] = self.last_framing[field] if value is None else value
        trial = FramingConfig(framing.id)
        for field, value in values.items():
            setattr(trial, field, value)
        if not framing_is_usable(trial):
            values = dict(self.last_framing)
        changed = any(values[field] != getattr(framing, field) for field in FRAMING_FIELDS)
        for field in FRAMING_FIELDS:
            self.last_framing[field] = values[field]
            getattr(self, f"framing_{field}_var").set(str(values[field]))
            setattr(framing, field, values[field])
        if changed:
            self.on_framing_changed()

    # ---- the list and its thumbnails ------------------------------------------------------------ #
    def rebuild_framing_list(self):
        """Show every framing's thumbnail (coloured by coloration 0), the selected one framed."""
        items = [(f.id, f"framing {idx}", encode_png_bytes(self.render_framing_thumbnail(f, THUMB_SIZE)),
                  f.id == self.selected_framing_id)
                 for idx, f in enumerate(self.framings)]
        self.framing_strip.rebuild(items)
