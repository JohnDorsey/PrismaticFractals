"""Which slots of a ray's metrics and characteristic points an image keeps. Every ray has N_STATS metrics and N_POINTS
characteristic points (the reductions of its sequences), but a coloration reads only a few of them: the default one two
points and one metric. A StoredLayout names the slots that are kept and where they sit in the arrays of an image -
slot k of the last axis of `stats` is metric stat_ids[k], slot k of the points axis of `characteristic_points` is
characteristic point point_ids[k] - so a large export need not allocate 321 bytes for every pixel (the full layout,
FULL_LAYOUT, which the live image keeps so that any coloration can be chosen at once) but only what its colorations
need (see color_pipeline.demand_layout). Reading a slot a layout does not keep raises, never returns zeros.

The same idea is how a path engine will declare what it can store: an engine's own layout (the sequences it has), of
which a render keeps the part its colorations need."""
from dataclasses import dataclass, field
import numpy as np
from .kernels import N_POINTS, N_STATS


@dataclass(frozen=True, eq=False)
class StoredLayout:
    stat_ids: np.ndarray            # int64: the metric slots kept, ascending
    point_ids: np.ndarray           # int64: the characteristic point slots kept, ascending
    _stat_index: dict = field(repr=False)
    _point_index: dict = field(repr=False)

    @classmethod
    def of(cls, stat_slots, point_slots):
        """The layout that keeps exactly these metric slots and these characteristic point slots."""
        stat_ids = np.array(sorted({int(s) for s in stat_slots}), dtype=np.int64)
        point_ids = np.array(sorted({int(p) for p in point_slots}), dtype=np.int64)
        if len(stat_ids) and (stat_ids[0] < 0 or stat_ids[-1] >= N_STATS):
            raise ValueError(f"a metric slot outside 0..{N_STATS - 1}: {stat_ids.tolist()}")
        if len(point_ids) and (point_ids[0] < 0 or point_ids[-1] >= N_POINTS):
            raise ValueError(f"a characteristic point slot outside 0..{N_POINTS - 1}: {point_ids.tolist()}")
        return cls(stat_ids, point_ids, {int(s): k for k, s in enumerate(stat_ids)},
                   {int(p): k for k, p in enumerate(point_ids)})

    def stat_position(self, slot):
        """Where metric `slot` is in the last axis of `stats`."""
        try:
            return self._stat_index[int(slot)]
        except KeyError:
            raise KeyError(f"metric slot {int(slot)} is not stored by this layout") from None

    def point_position(self, slot):
        """Where characteristic point `slot` is in the points axis of `characteristic_points`."""
        try:
            return self._point_index[int(slot)]
        except KeyError:
            raise KeyError(f"characteristic point slot {int(slot)} is not stored by this layout") from None

    def union(self, other):
        """The layout that keeps what either keeps."""
        return StoredLayout.of(set(self.stat_ids.tolist()) | set(other.stat_ids.tolist()),
                               set(self.point_ids.tolist()) | set(other.point_ids.tolist()))

    @property
    def bytes_per_pixel(self):
        """What an image keeps for each pixel under this layout: float32 metrics, float32 (x, y) points, one state byte."""
        return 4 * len(self.stat_ids) + 8 * len(self.point_ids) + 1

    def allocate(self, out_h, out_w):
        """Zeroed (stats, characteristic_points) arrays for an out_h x out_w image under this layout."""
        return (np.zeros((out_h, out_w, len(self.stat_ids)), dtype=np.float32),
                np.zeros((out_h, out_w, len(self.point_ids), 2), dtype=np.float32))


FULL_LAYOUT = StoredLayout.of(range(N_STATS), range(N_POINTS))      # every slot, in order: what the live image keeps
