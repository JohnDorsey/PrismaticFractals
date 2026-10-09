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
from .kernels import N_POINTS, N_PRED, N_SEQS, N_SRED, N_STATS, POINT_SLOT, STAT_SLOT, SequenceTables


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

    @property
    def n_stats(self):
        """How many metrics this layout keeps."""
        return len(self.stat_ids)

    @property
    def n_points(self):
        """How many characteristic points this layout keeps."""
        return len(self.point_ids)

    def has_stat(self, slot):
        """Whether this layout keeps metric `slot`."""
        return int(slot) in self._stat_index

    def has_point(self, slot):
        """Whether this layout keeps characteristic point `slot`."""
        return int(slot) in self._point_index

    def stat_sources(self, engine_tables):
        """The positions, in the per-path metrics of the engine whose tables are `engine_tables` (SequenceTables), of the metrics
        this layout keeps, in order: what trace_rows_stored reads to fill slot k. KeyError for a metric the engine does not store."""
        return _positions(engine_tables.stat_pos, self.stat_ids, "metric")

    def point_sources(self, engine_tables):
        """The same for the characteristic points."""
        return _positions(engine_tables.point_pos, self.point_ids, "characteristic point")

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


def _positions(table, ids, what):
    positions = table[ids] if len(ids) else np.empty(0, dtype=np.int64)
    if (positions < 0).any():
        raise KeyError(f"the engine does not store the {what}(s) {ids[positions < 0].tolist()}")
    return positions.astype(np.int64)


def sequence_tables(layout):
    """The SequenceTables (kernels.py) of an engine that stores exactly what the StoredLayout `layout` keeps: the id -> position maps of
    its metrics and characteristic points, and the sequences it reduces (the sequences some stored point or metric belongs to)."""
    point_pos = np.full(N_POINTS, -1, dtype=np.int64)
    stat_pos = np.full(N_STATS, -1, dtype=np.int64)
    for slot, position in layout._point_index.items():
        point_pos[slot] = position
    for slot, position in layout._stat_index.items():
        stat_pos[slot] = position
    seqs = [seq for seq in range(N_SEQS) if any(point_pos[POINT_SLOT[seq, red]] >= 0 for red in range(N_PRED))
            or any(stat_pos[STAT_SLOT[seq, red]] >= 0 for red in range(N_SRED))]
    return SequenceTables(seqs=np.array(seqs, dtype=np.int64), point_pos=point_pos, stat_pos=stat_pos,
                          n_points=layout.n_points, n_stats=layout.n_stats)


def layout_of_tables(tables):
    """The StoredLayout of everything an engine with these SequenceTables stores (the inverse of sequence_tables; the positions must
    follow the order of the ids)."""
    stat_ids = [slot for slot in range(N_STATS) if tables.stat_pos[slot] >= 0]
    point_ids = [slot for slot in range(N_POINTS) if tables.point_pos[slot] >= 0]
    layout = StoredLayout.of(stat_ids, point_ids)
    if (layout.stat_sources(tables) != np.arange(layout.n_stats)).any() or (layout.point_sources(tables) != np.arange(layout.n_points)).any():
        raise ValueError("the positions of an engine's stored metrics and points must follow the order of their ids")
    return layout


FULL_LAYOUT = StoredLayout.of(range(N_STATS), range(N_POINTS))      # every slot, in order: what the live image keeps of the ray engine
