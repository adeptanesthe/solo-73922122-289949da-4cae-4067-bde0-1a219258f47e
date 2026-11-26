"""Mask propagation engine (pure NumPy, no framework/state).

Coordinate / optical-flow convention
------------------------------------
All frames share the same ``H x W`` grid.  For the boundary between frame
``i`` and frame ``i+1`` there are two *integer* flow arrays of shape
``H x W x 2`` with displacement order ``[dy, dx]``:

* ``fwd[i][r, c]`` maps a pixel ``(r, c)`` in frame ``i`` to
  ``(r + dy, c + dx)`` in frame ``i+1``.
* ``bwd[i][r, c]`` maps a pixel ``(r, c)`` in frame ``i+1`` to
  ``(r + dy, c + dx)`` in frame ``i``.

When a chain crosses one boundary in direction D it performs the matching
step and then a forward/backward round-trip check:

* left trace (frame ``i+1`` -> ``i``):  ``q = p + bwd[i][p]`` and require
  ``q + fwd[i][q] == p``;
* right trace (frame ``i`` -> ``i+1``): ``q = p + fwd[i][p]`` and require
  ``q + bwd[i][q] == p``.

A chain breaks for a pixel at the first step whose landing coordinate is
outside the frame or whose round-trip coordinate differs from the starting
coordinate of that step.

The implementation builds cumulative coordinate maps *outwards from each
confirmed anchor*, so every boundary is processed once per interval.  It is
mathematically identical to tracing every pixel independently all the way to
the anchor (the test-suite includes a brute-force cross-check).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Status codes (stored in the immutable per-version status matrix)
# ---------------------------------------------------------------------------

STATUS_BROKEN = 0
"""Both chains are unavailable (broken); pixel must be painted."""

STATUS_LEFT = 1
"""Only the left chain was available; its value was used."""

STATUS_RIGHT = 2
"""Only the right chain was available; its value was used."""

STATUS_AGREE = 3
"""Both chains were available and carried the same value."""

STATUS_CONFLICT = 4
"""Both chains were available but carried different values; paint needed."""

STATUS_CONFIRMED = 5
"""Frame has a user-supplied mask; never computed."""

STATUS_LABELS: dict[int, str] = {
    STATUS_BROKEN: "broken",
    STATUS_LEFT: "left",
    STATUS_RIGHT: "right",
    STATUS_AGREE: "agree",
    STATUS_CONFLICT: "conflict",
    STATUS_CONFIRMED: "confirmed",
}

LABEL_TO_STATUS = {v: k for k, v in STATUS_LABELS.items()}

NEEDS_PAINT_STATUSES = frozenset({STATUS_BROKEN, STATUS_CONFLICT})

PENDING_VALUE = -1
"""Value used in ``values`` matrices for pixels that still need painting."""

VALUE_DTYPE = np.int8
STATUS_DTYPE = np.int8
FLOW_DTYPE = np.int64
MASK_DTYPE = np.uint8


def normalize_mask(raw: np.ndarray | list, height: int, width: int,
                   name: str = "mask") -> np.ndarray:
    """Convert and validate a binary ``H x W`` mask.

    Raises ``ValueError`` with a human-readable message on any problem.
    """
    try:
        arr = np.asarray(raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name}: cannot be read as an array: {exc}") from exc

    if arr.dtype == object:
        raise ValueError(f"{name}: rows are ragged or contain non-numeric data")
    if arr.ndim != 2 or arr.shape != (height, width):
        raise ValueError(
            f"{name}: expected shape ({height}, {width}), got {arr.shape}"
        )

    # Reject NaN / Inf before integer conversion.
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: contains NaN or Infinity")

    # Validate on the original dtype so e.g. 0.5 cannot truncate onto 0/1.
    try:
        is_binary = (arr == 0) | (arr == 1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: entries must be numeric 0/1") from exc
    if not bool(np.all(is_binary)):
        raise ValueError(f"{name}: only 0 and 1 are allowed")

    return arr.astype(MASK_DTYPE)


def normalize_flows(raw_fwd: list, raw_bwd: list, frame_count: int,
                    height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate and stack forward/backward flows.

    Returns two arrays of shape ``(frame_count - 1, H, W, 2)`` int64.
    """
    boundaries = frame_count - 1
    if not isinstance(raw_fwd, list) or len(raw_fwd) != boundaries:
        raise ValueError(
            f"flows: expected {boundaries} forward boundary arrays, "
            f"got {len(raw_fwd) if isinstance(raw_fwd, list) else 'non-list'}"
        )
    if not isinstance(raw_bwd, list) or len(raw_bwd) != boundaries:
        raise ValueError(
            f"flows: expected {boundaries} backward boundary arrays, "
            f"got {len(raw_bwd) if isinstance(raw_bwd, list) else 'non-list'}"
        )

    fwd = np.zeros((boundaries, height, width, 2), dtype=FLOW_DTYPE)
    bwd = np.zeros((boundaries, height, width, 2), dtype=FLOW_DTYPE)

    for i in range(boundaries):
        fwd[i] = _normalize_one_flow(raw_fwd[i], height, width,
                                     f"forward[{i}]")
        bwd[i] = _normalize_one_flow(raw_bwd[i], height, width,
                                     f"backward[{i}]")
    return fwd, bwd


def _normalize_one_flow(raw, height: int, width: int, name: str) -> np.ndarray:
    try:
        arr = np.asarray(raw)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"flows.{name}: cannot be read as an array: {exc}") from exc
    if arr.dtype == object:
        raise ValueError(f"flows.{name}: flow field is ragged")
    if arr.ndim != 3 or arr.shape != (height, width, 2):
        raise ValueError(
            f"flows.{name}: expected shape ({height}, {width}, 2) [dy, dx], "
            f"got {arr.shape}"
        )
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"flows.{name}: contains NaN/Infinity")
        if not np.all(arr == np.round(arr)):
            raise ValueError(f"flows.{name}: optical flow must be integer")
    try:
        return arr.astype(FLOW_DTYPE, casting="safe")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"flows.{name}: components must be integers") from exc


def _coordinate_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rr, cc = np.meshgrid(
        np.arange(height, dtype=FLOW_DTYPE),
        np.arange(width, dtype=FLOW_DTYPE),
        indexing="ij",
    )
    return rr, cc


def _build_left_values(
    lo: int,
    hi: int,
    anchor_mask: np.ndarray,
    fwd: np.ndarray,
    bwd: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Trace every pixel of frames ``lo+1..hi-1`` left to frame ``lo``.

    Returns an ``(hi-lo-1, H, W)`` int8 matrix; unavailable chains are -1.
    Row ``j`` corresponds to frame ``lo + 1 + j``.
    """
    count = hi - lo - 1
    out = np.full((count, height, width), PENDING_VALUE, dtype=VALUE_DTYPE)
    if count <= 0:
        return out

    grid_r, grid_c = _coordinate_grid(height, width)

    # Cumulative map from the *previous* frame to anchor ``lo``.
    map_r = grid_r.copy()
    map_c = grid_c.copy()
    map_alive = np.ones((height, width), dtype=bool)

    for j, frame in enumerate(range(lo + 1, hi)):
        boundary = frame - 1  # boundary between frame-1 and frame

        # Dead entries may hold out-of-frame coordinates; clip only for
        # safe array indexing, their result is masked out by ``alive``.
        src_r = grid_r
        src_c = grid_c

        step = bwd[boundary][src_r, src_c]
        dst_r = src_r + step[..., 0]
        dst_c = src_c + step[..., 1]

        in_frame = (
            (dst_r >= 0) & (dst_r < height)
            & (dst_c >= 0) & (dst_c < width)
        )
        clip_r = np.clip(dst_r, 0, height - 1)
        clip_c = np.clip(dst_c, 0, width - 1)

        back = fwd[boundary][clip_r, clip_c]
        rt_r = dst_r + back[..., 0]
        rt_c = dst_c + back[..., 1]
        roundtrip_ok = (rt_r == src_r) & (rt_c == src_c)

        step_ok = in_frame & roundtrip_ok

        # Compose the crossing with the already-built map of frame-1.
        comp_r = map_r[clip_r, clip_c]
        comp_c = map_c[clip_r, clip_c]
        comp_alive = map_alive[clip_r, clip_c]

        alive = step_ok & comp_alive
        map_r = comp_r
        map_c = comp_c
        map_alive = alive

        safe_r = np.clip(map_r, 0, height - 1)
        safe_c = np.clip(map_c, 0, width - 1)
        anchored_value = anchor_mask[safe_r, safe_c].astype(VALUE_DTYPE)
        out[j] = np.where(alive, anchored_value, PENDING_VALUE)

    return out


def _build_right_combination(
    lo: int,
    hi: int,
    anchor_mask: np.ndarray,
    left_values: np.ndarray,
    values: np.ndarray,
    status: np.ndarray,
    fwd: np.ndarray,
    bwd: np.ndarray,
    height: int,
    width: int,
) -> None:
    """Trace right to frame ``hi`` and combine with precomputed left values."""
    grid_r, grid_c = _coordinate_grid(height, width)

    map_r = grid_r.copy()
    map_c = grid_c.copy()
    map_alive = np.ones((height, width), dtype=bool)

    for frame in range(hi - 1, lo, -1):
        boundary = frame  # boundary between frame and frame+1

        src_r = grid_r
        src_c = grid_c

        step = fwd[boundary][src_r, src_c]
        dst_r = src_r + step[..., 0]
        dst_c = src_c + step[..., 1]

        in_frame = (
            (dst_r >= 0) & (dst_r < height)
            & (dst_c >= 0) & (dst_c < width)
        )
        clip_r = np.clip(dst_r, 0, height - 1)
        clip_c = np.clip(dst_c, 0, width - 1)

        back = bwd[boundary][clip_r, clip_c]
        rt_r = dst_r + back[..., 0]
        rt_c = dst_c + back[..., 1]
        roundtrip_ok = (rt_r == src_r) & (rt_c == src_c)

        step_ok = in_frame & roundtrip_ok

        comp_r = map_r[clip_r, clip_c]
        comp_c = map_c[clip_r, clip_c]
        comp_alive = map_alive[clip_r, clip_c]

        alive = step_ok & comp_alive
        map_r = comp_r
        map_c = comp_c
        map_alive = alive

        safe_r = np.clip(map_r, 0, height - 1)
        safe_c = np.clip(map_c, 0, width - 1)
        right_value = np.where(
            alive,
            anchor_mask[safe_r, safe_c].astype(VALUE_DTYPE),
            np.full((height, width), PENDING_VALUE, dtype=VALUE_DTYPE),
        )

        j = frame - (lo + 1)
        lv = left_values[j]
        v_out = np.full((height, width), PENDING_VALUE, dtype=VALUE_DTYPE)
        s_out = np.full((height, width), STATUS_BROKEN, dtype=STATUS_DTYPE)

        left_ok = lv >= 0
        right_ok = right_value >= 0
        only_left = left_ok & ~right_ok
        only_right = ~left_ok & right_ok
        both = left_ok & right_ok
        agree = both & (lv == right_value)
        conflict = both & (lv != right_value)

        v_out[only_left] = lv[only_left]
        s_out[only_left] = STATUS_LEFT

        v_out[only_right] = right_value[only_right]
        s_out[only_right] = STATUS_RIGHT

        v_out[agree] = lv[agree]
        s_out[agree] = STATUS_AGREE

        # conflict leaves value at -1
        s_out[conflict] = STATUS_CONFLICT
        # neither side: STATUS_BROKEN / -1 already set

        values[lo + 1 + j] = v_out
        status[lo + 1 + j] = s_out


def propagate_interval(
    lo: int,
    hi: int,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    fwd: np.ndarray,
    bwd: np.ndarray,
    height: int,
    width: int,
    values: np.ndarray | None = None,
    status: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute frames ``lo+1 .. hi-1`` between two confirmed anchors.

    ``lo`` / ``hi`` are frame indices of the left/right confirmed frames.
    When ``values``/``status`` full-volume arrays are supplied they are
    written at the corresponding frame slices (and returned); otherwise
    fresh full-length arrays are created (mostly convenient for tests).
    """
    if not (0 <= lo < hi):
        raise ValueError(f"invalid interval: lo={lo}, hi={hi}")

    if values is None:
        values = np.full((fwd.shape[0] + 1, height, width),
                         PENDING_VALUE, dtype=VALUE_DTYPE)
    if status is None:
        status = np.full((fwd.shape[0] + 1, height, width),
                         STATUS_BROKEN, dtype=STATUS_DTYPE)

    left_values = _build_left_values(
        lo, hi, left_mask, fwd, bwd, height, width
    )
    _build_right_combination(
        lo, hi, right_mask, left_values, values, status,
        fwd, bwd, height, width,
    )
    return values, status
