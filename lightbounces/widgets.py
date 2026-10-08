"""Small Tk building blocks that know nothing about the simulation: the blue outline around a modified control, tooltips,
a number entry with a slider-like interface, a scrollable tab, a strip of thumbnails, and the JPEG XL mode/quality row."""
import tkinter as tk
from tkinter import ttk
from .image_io import JXL_LOSSY_MAX_QUALITY, JXL_MODES


def make_scrollable_tab(tab):
    """Makes a notebook tab's contents vertically scrollable: returns
    (canvas, inner_frame), where widgets that would have gone straight into
    `tab` should go into `inner_frame` instead. Contents keep their natural
    height rather than being squeezed to fit the tab, and the width follows
    the tab's."""
    tab.rowconfigure(0, weight=1)
    tab.columnconfigure(0, weight=1)
    canvas = tk.Canvas(tab, highlightthickness=0)
    vsb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    inner = ttk.Frame(canvas)
    inner.columnconfigure(0, weight=1)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
    return canvas, inner


class ThumbnailStrip:
    """A strip of captioned thumbnails with New / Dupe / Del stacked to its left: the Obstacles tab's list turned
    sideways. The Colorations tab's list and the Framing tab's are this widget twice, so that they look and behave
    the same: grid `strip.frame` where it belongs, give `rebuild` the items to show, and the commands are called when
    a button is pressed or a thumbnail clicked (the selected thumbnail has a coloured frame around it)."""

    def __init__(self, parent, thumb_size, font, outline_color, outline_px, on_new, on_dupe, on_delete, on_select):
        self.thumb_size, self.font = thumb_size, font
        self.outline_color, self.outline_px = outline_color, outline_px
        self.on_select = on_select
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(1, weight=1)
        btn_col = ttk.Frame(self.frame)
        btn_col.grid(row=0, column=0, sticky="n", padx=(0, 6))
        ttk.Button(btn_col, text="New", width=6, command=on_new).pack(fill="x", pady=(0, 3))
        ttk.Button(btn_col, text="Dupe", width=6, command=on_dupe).pack(fill="x", pady=(0, 3))
        self.delete_button = ttk.Button(btn_col, text="Del", width=6, command=on_delete)
        self.delete_button.pack(fill="x")

        list_container = ttk.Frame(self.frame)
        list_container.grid(row=0, column=1, sticky="ew")
        list_container.columnconfigure(0, weight=1)
        canvas = tk.Canvas(list_container, height=thumb_size + 34, highlightthickness=0)
        hsb = ttk.Scrollbar(list_container, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=hsb.set)
        canvas.grid(row=0, column=0, sticky="ew")
        hsb.grid(row=1, column=0, sticky="ew")
        self.items_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.items_frame, anchor="nw")
        self.items_frame.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        # (unlike the vertical obstacle list, the inner frame's WIDTH is never pinned to the canvas's - it needs to
        # grow past it, since scrolling sideways to reach it is the whole point)

    def rebuild(self, items):
        """Show `items`, [(id, caption, png bytes of the thumbnail, selected), ...], left to right; Del is
        disabled while there is only one."""
        for w in self.items_frame.winfo_children():
            w.destroy()
        for idx, (item_id, caption, png, selected) in enumerate(items):
            cell = tk.Frame(self.items_frame, highlightthickness=self.outline_px if selected else 0,
                            highlightbackground=self.outline_color, highlightcolor=self.outline_color)
            cell.grid(row=0, column=idx, sticky="n", padx=2, pady=1)
            thumb = tk.PhotoImage(data=png, format="png")
            thumb_lbl = tk.Label(cell, image=thumb, bd=0, cursor="hand2")
            thumb_lbl.image = thumb              # keep a reference alive
            thumb_lbl.grid(row=0, column=0, padx=2, pady=(2, 0))
            thumb_lbl.bind("<Button-1>", lambda _e, i=item_id: self.on_select(i))
            lbl = tk.Label(cell, text=caption, cursor="hand2", font=self.font)
            lbl.grid(row=1, column=0, pady=(0, 2))
            lbl.bind("<Button-1>", lambda _e, i=item_id: self.on_select(i))
        self.delete_button.state(["disabled"] if len(items) <= 1 else ["!disabled"])


MODIFIED_OUTLINE = "#1e88e5"     # blue: a setting changed since the selected keyframe was

MODIFIED_OUTLINE_PX = 4          # loaded / last saved (see App.check_modified)


def find_control_label(widget):
    """The text label that names `widget`, if it has one beside it: the label to its left in the same
    grid row, or the label packed directly before it. None when there is none (or when the widget is
    placed some other way)."""
    def is_label(w):
        return isinstance(w, (ttk.Label, tk.Label))
    master = widget.master
    try:
        manager = widget.winfo_manager()
        if manager == "grid":
            info = widget.grid_info()
            row, column = int(info["row"]), int(info["column"])
            found = None
            for sibling in master.grid_slaves(row=row):
                sibling_column = int(sibling.grid_info()["column"])
                if sibling_column < column and is_label(sibling) and (
                        found is None or sibling_column > int(found.grid_info()["column"])):
                    found = sibling
            return found
        if manager == "pack":
            slaves = master.pack_slaves()
            index = slaves.index(widget)
            if index > 0 and is_label(slaves[index - 1]):
                return slaves[index - 1]
    except (tk.TclError, ValueError, KeyError):
        pass
    return None


class ModifiedOutline:
    """A blue frame drawn just behind, and MODIFIED_OUTLINE_PX around, one
    widget - without touching how that widget is laid out (it is placed in the
    widget's own parent and follows the widget's geometry) - and the same blue
    for the text of the label that names the control (a checkbox or radio button
    is its own label). place() measures from
    the parent's inner area, which for some parents (a ttk labelframe, say) is
    offset from the origin winfo_x/winfo_y use, so that offset is measured once
    per outline."""
    def __init__(self, widget):
        self.widget = widget
        self.recolored = None       # (what was recoloured, how to put it back) while shown
        self.frame = tk.Frame(widget.master, background=MODIFIED_OUTLINE)
        self.frame.place(x=0, y=0, width=1, height=1)
        self.frame.update_idletasks()
        self.dx, self.dy = self.frame.winfo_x(), self.frame.winfo_y()
        self.frame.place_forget()
        self.shown = False
        widget.bind("<Configure>", lambda _e: self._place(), add="+")

    def _place(self):
        if not self.shown:
            return
        w, n = self.widget, MODIFIED_OUTLINE_PX
        self.frame.place(x=w.winfo_x() - n - self.dx, y=w.winfo_y() - n - self.dy,
                         width=w.winfo_width() + 2 * n, height=w.winfo_height() + 2 * n)
        self.frame.lower(w)

    def _recolor(self, on):
        """Turn the control's label blue (on) or put it back as it was."""
        if not on:
            if self.recolored is not None:
                restore = self.recolored
                self.recolored = None
                try:
                    restore()
                except tk.TclError:                 # the label has been destroyed meanwhile
                    pass
            return
        w = self.widget
        try:
            if w.winfo_class() in ("TCheckbutton", "TRadiobutton"):
                original = str(w.cget("style"))
                style = f"Modified.{w.winfo_class()}"
                ttk.Style(w).configure(style, foreground=MODIFIED_OUTLINE)
                w.configure(style=style)
                self.recolored = lambda: w.configure(style=original)
                return
            label = find_control_label(w)
            if label is not None:
                original = str(label.cget("foreground"))
                label.configure(foreground=MODIFIED_OUTLINE)
                self.recolored = lambda: label.configure(foreground=original)
        except tk.TclError:
            self.recolored = None

    def show(self, on):
        if on == self.shown:
            return
        self.shown = on
        if on:
            self._place()
        else:
            self.frame.place_forget()
        self._recolor(on)


def detect_modified_keys(last_seen, current, modified):
    """The set of modified setting keys after comparing the settings seen at the
    previous check (`last_seen`) with `current`: any key whose value differs is
    added (and stays until saved or the keyframe is deselected, even if the value
    later goes back); keys that no longer exist are dropped; keys that are new
    (e.g. a just-added obstacle) are not counted as edits."""
    result = {k for k in modified if k in current}
    result.update(k for k, v in current.items() if k in last_seen and last_seen[k] != v)
    return result


class Tooltip:
    """A small hover tooltip attached to one widget: shows `text` in a
    borderless popup near the cursor on <Enter>, hides it on <Leave>.
    update(text) changes what it shows later (e.g. as a button's enabled
    state changes) without rebinding anything; an empty text shows nothing."""
    def __init__(self, widget, text=""):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def update(self, text):
        self.text = text
        if self.tip is not None and not text:
            self._hide()

    def _show(self, _event=None):
        if not self.text or self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, background="#ffffe0",
                relief="solid", borderwidth=1, padx=4, pady=2).pack()

    def _hide(self, _event=None):
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class FloatEntryHandle:
    """A .get()/.set() wrapper around a float Entry, making it a drop-in
    replacement for a tk.Scale everywhere else in the code that calls
    .get()/.set() on one of these controls. With no bounds (the default),
    accepts any float, so many-revolution angles can be typed or animated;
    with lo/hi given, the value is clamped into range - used for settings
    that keep a slider-like range but as a text field (so typing isn't
    interrupted mid-edit the way a live-validating popup would be)."""

    def __init__(self, var, fmt, default, lo=None, hi=None, exact=False):
        self.var = var
        self.fmt = fmt
        self.exact = exact          # show the value as typed (shortest exact text), not through fmt
        self.lo = lo
        self.hi = hi
        self.last_valid = self._clamp(float(default))
        self.widget = None          # set by add_angle_entry to the ttk.Entry itself

    def _clamp(self, value):
        if self.lo is not None:
            value = max(self.lo, value)
        if self.hi is not None:
            value = min(self.hi, value)
        return value

    def text(self, value):
        return repr(float(value)) if self.exact else self.fmt.format(value)

    def get(self):
        try:
            self.last_valid = self._clamp(float(self.var.get()))
        except (ValueError, tk.TclError):
            pass
        return self.last_valid

    def set(self, val):
        self.last_valid = self._clamp(float(val))
        self.var.set(self.text(self.last_valid))

    def commit(self):
        """Called on FocusOut/Return: normalize the text, or revert it if
        it's not currently a valid number."""
        try:
            self.last_valid = self._clamp(float(self.var.get()))
        except ValueError:
            pass
        self.var.set(self.text(self.last_valid))


class JxlQualityControls:
    """The JPEG XL choices - lossless, or lossy with a quality - as one row of controls
    (`frame`); used by the still-image export and by the video panel."""

    def __init__(self, parent, label="JPEG XL mode:"):
        self.frame = ttk.Frame(parent)
        ttk.Label(self.frame, text=label).pack(side="left", padx=(0, 4))
        self.mode_var = tk.StringVar(value=JXL_MODES[0])
        combo = ttk.Combobox(self.frame, textvariable=self.mode_var, values=JXL_MODES,
                             state="readonly", width=8)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        # lossy only: a quality slider on the same line
        self.quality_frame = ttk.Frame(self.frame)
        ttk.Label(self.quality_frame, text="Quality:").pack(side="left", padx=(8, 4))
        self.scale = tk.Scale(self.quality_frame, from_=5, to=JXL_LOSSY_MAX_QUALITY,
                              resolution=5, orient="horizontal", showvalue=False,
                              highlightthickness=0, width=14, sliderlength=18, length=80)
        self.scale.set(90)
        self.scale.pack(side="left")
        value_label = ttk.Label(self.quality_frame, text="90", width=3, anchor="e")
        value_label.pack(side="left", padx=(4, 0))
        self.scale.configure(command=lambda v: value_label.configure(text=f"{float(v):.0f}"))
        self.refresh()

    def lossless(self):
        return self.mode_var.get() == "Lossless"

    def quality(self):
        return int(round(float(self.scale.get())))

    def refresh(self):
        """The quality slider only matters in Lossy mode."""
        if self.lossless():
            self.quality_frame.pack_forget()
        else:
            self.quality_frame.pack(side="left")
