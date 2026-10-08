"""Framings: where the camera is and how much of the world it sees. A framing is a centre (x, y), a diagonal and an
aspect ratio (width / height), all in world units; it defines the rectangle of the world an image covers (a WorldRect: what
the kernels take), and so also the shape of the image on screen. This module has the live editable FramingConfig, its
frozen copy FramingSnapshot (what background threads and project files hold) and the arithmetic from a framing to its
world rectangle, its pixel size and the text that describes it. The whole-world framing (centre 0, 0, diagonal 2 sqrt(2),
aspect 1) covers (-1, 1, -1, 1) exactly: what every image covered before framings existed."""
import math
from .constants import (
    DEFAULT_FRAMING_ASPECT, DEFAULT_FRAMING_DIAGONAL, DEFAULT_FRAMING_X, DEFAULT_FRAMING_Y
)
from .kernels import PREVIEW_BOX, WorldRect
from .settings import resolve_value, strict_record

FRAMING_FIELDS = ("x", "y", "diagonal", "aspect")        # a framing's settings (it also has an id)

# Every field a framing can define in a keyframe, as the "framing:<id>:<field>" keys name them: (field, label, is_float).
# All four are floats, so between keyframes the camera glides: pans, zooms and changes shape linearly.
FRAMING_FIELD_INFO = [("x", "X", True), ("y", "Y", True), ("diagonal", "Diagonal", True),
                      ("aspect", "Aspect ratio", True)]


# One framing's live state, edited by the Framing tab's controls. Position in App.framings has no meaning of its own
# beyond the order of the list (and the first one, "framing 0", being the one every coloration's thumbnail is of).
class FramingConfig:
    __slots__ = ("id",) + FRAMING_FIELDS

    def __init__(self, framing_id):
        self.id = framing_id
        self.x = DEFAULT_FRAMING_X
        self.y = DEFAULT_FRAMING_Y
        self.diagonal = DEFAULT_FRAMING_DIAGONAL
        self.aspect = DEFAULT_FRAMING_ASPECT


@strict_record
class FramingSnapshot:
    """A frozen copy of one FramingConfig (see snapshot_framing), safe to read from a background thread and to keep
    after the live one has been edited or deleted."""
    id: int
    x: float
    y: float
    diagonal: float
    aspect: float


def snapshot_framing(f):
    """A FramingSnapshot of a FramingConfig (or of another snapshot)."""
    return FramingSnapshot(id=f.id, x=float(f.x), y=float(f.y), diagonal=float(f.diagonal), aspect=float(f.aspect))


DEFAULT_FRAMING = snapshot_framing(FramingConfig(0))        # the whole world


def framing_config_from_snapshot(snapshot):
    """A live FramingConfig with the contents of a FramingSnapshot."""
    framing = FramingConfig(snapshot.id)
    for field in FRAMING_FIELDS:
        setattr(framing, field, getattr(snapshot, field))
    return framing


def same_framing(a, b):
    """Whether two framings (either kind) see the same: all four numbers equal (they may have different ids)."""
    return all(getattr(a, field) == getattr(b, field) for field in FRAMING_FIELDS)


def resolve_framing(snapshot, keyframes_sorted, frame_number):
    """The FramingSnapshot of one video frame: the framing's four numbers each resolved independently (animated or
    held) from the keyframes that define its "framing:<id>:<field>" key, like any other setting - falling back to the
    snapshot's own value (at render-start time) wherever no keyframe defines one."""
    values = {field: resolve_value(f"framing:{snapshot.id}:{field}", keyframes_sorted, frame_number,
                                   getattr(snapshot, field), is_float)
              for field, _label, is_float in FRAMING_FIELD_INFO}
    return FramingSnapshot(id=snapshot.id, **{k: float(v) for k, v in values.items()})


def framing_size(f):
    """(width, height) in world units of the rectangle a framing covers: the diagonal split by the aspect ratio
    (width / height)."""
    height = f.diagonal / math.sqrt(1.0 + f.aspect * f.aspect)
    return f.aspect * height, height


def framing_world_rect(f):
    """The WorldRect (see kernels.WorldRect) a framing covers: its centre, the width and the height."""
    width, height = framing_size(f)
    return WorldRect(float(f.x - width / 2.0), float(f.x + width / 2.0),
                  float(f.y - height / 2.0), float(f.y + height / 2.0))


def framing_is_usable(f):
    """Whether a framing's numbers make a world rectangle that has a size (a diagonal too small for the centre's size, or a
    number too large to have a fraction, would leave it a line or a point) and a finite one."""
    width, height = framing_size(f)
    world_rect = framing_world_rect(f)
    return (all(math.isfinite(v) for v in world_rect) and width > 0.0 and height > 0.0
            and world_rect.x_stop > world_rect.x_start and world_rect.y_stop > world_rect.y_start)


def framing_image_size(f, box=PREVIEW_BOX):
    """(pixels across, pixels down) of the image a framing makes on screen: its aspect ratio, in the largest size
    that fits a `box` pixel square (so the whole-world framing's is box x box)."""
    if f.aspect >= 1.0:
        return box, max(1, round(box / f.aspect))
    return max(1, round(box * f.aspect)), box


def framing_outline_pixel(f):
    """The world width of one pixel of the image this framing makes on screen: what an obstacle's outline layers are
    measured in, so that they look the same however far the framing zooms in or out (DEFAULT_PIXEL_WORLD_WIDTH for the whole world)."""
    width, height = framing_size(f)
    return max(width, height) / PREVIEW_BOX


def framing_info_block(f):
    """A framing's settings as the lines of text of an exported image's margin."""
    return ["Framing", f"  x {f.x:.9g}", f"  y {f.y:.9g}", f"  diagonal {f.diagonal:.9g}",
            f"  aspect {f.aspect:.9g}"]
