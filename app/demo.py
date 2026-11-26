"""Deterministic demo scenarios used by /docs examples and /api/demo/*.

The 3x5 "chain break" scenario is hand-designed so that frame 1 exhibits
every possible pixel status:

============  ==============================================================
pixel         what happens
============  ==============================================================
(0, 0)        backward step leaves the frame (dy=-1) -> left chain broken,
              right chain ok -> status ``right``
(1, 1)        backward step lands at (1,2) but forward flow there is zero,
              so the round trip does not return -> left broken; forward
              step dy=+9 leaves frame -> right broken too -> ``broken``
(1, 3)        forward step lands at (1,2) and backward flow there is zero ->
              round-trip mismatch -> right broken, left ok -> ``left``
(2, 4)        forward step moves one column past the right edge -> right
              broken, left ok -> ``left``
(0, 4)        both identity chains ok, but first/last masks disagree ->
              ``conflict``
other pixels  identity zero flows and equal anchors -> ``agree``
============  ==============================================================

Only (1,1) ``broken`` and (0,4) ``conflict`` need painting.
"""

from __future__ import annotations

import numpy as np

from . import propagation as eng

CB_FRAME_COUNT = 3
CB_HEIGHT = 3
CB_WIDTH = 5

SUP_FRAME_COUNT = 5
SUP_HEIGHT = 3
SUP_WIDTH = 5


def _block_mask(h: int = CB_HEIGHT, w: int = CB_WIDTH) -> np.ndarray:
    m = np.zeros((h, w), dtype=eng.MASK_DTYPE)
    m[0, 1] = 1
    m[0, 2] = 1
    m[1, 1] = 1
    m[1, 2] = 1
    return m


# ---------------------------------------------------------------- chain break
def chain_break_arrays():
    """Return (frame_count, h, w, first, last, fwd, bwd)."""
    n, h, w = CB_FRAME_COUNT, CB_HEIGHT, CB_WIDTH
    first = _block_mask().copy()
    last = _block_mask().copy()
    last[0, 4] = 1  # disagreement -> conflict at (0,4)

    fwd = np.zeros((n - 1, h, w, 2), dtype=eng.FLOW_DTYPE)
    bwd = np.zeros((n - 1, h, w, 2), dtype=eng.FLOW_DTYPE)

    # Boundary 0 (frames 0<->1), traced left from frame 1 via backward flow.
    bwd[0, 0, 0] = (-1, 0)   # (0,0) -> (-1,0): out of bounds
    bwd[0, 1, 1] = (0, 1)    # (1,1) -> (1,2); fwd[0] zero -> rt mismatch

    # Boundary 1 (frames 1<->2), traced right from frame 1 via forward flow.
    fwd[1, 1, 1] = (9, 0)    # (1,1) -> (10,1): out of bounds
    fwd[1, 1, 3] = (0, -1)   # (1,3) -> (1,2); bwd[1] zero -> rt mismatch
    fwd[1, 2, 4] = (0, 1)    # (2,4) -> (2,5): out of bounds

    return n, h, w, first, last, fwd, bwd


def chain_break_completed_mask() -> np.ndarray:
    """A finished artist mask for frame 1 (supplement submission)."""
    m = _block_mask().copy()
    m[0, 4] = 1  # artist resolves the conflict towards the last frame
    # (1,1) is already 1 inside the block; the broken chain did not know it.
    return m


# ----------------------------------------------------------------- supplement
def supplement_arrays():
    """N=5 identity-flow scenario with two endpoint disagreements.

    In v1 frames 1..3 show conflicts at (0,0) and (2,4).  Submitting a frame-2
    mask copied from the first frame makes frame 1 fully agree while frame 3
    keeps conflicting - demonstrating that only the intervals adjacent to
    frame 2 are recomputed.
    """
    n, h, w = SUP_FRAME_COUNT, SUP_HEIGHT, SUP_WIDTH
    first = _block_mask().copy()
    last = _block_mask().copy()

    first[0, 0] = 1
    last[0, 0] = 0
    first[2, 4] = 0
    last[2, 4] = 1

    fwd = np.zeros((n - 1, h, w, 2), dtype=eng.FLOW_DTYPE)
    bwd = np.zeros((n - 1, h, w, 2), dtype=eng.FLOW_DTYPE)
    return n, h, w, first, last, fwd, bwd


def supplement_frame2_mask() -> np.ndarray:
    m = _block_mask()
    m[0, 0] = 1  # follow first frame
    m[2, 4] = 0  # follow first frame
    return m


def supplement_frame3_mask() -> np.ndarray:
    """Follow the last frame; submitting it leaves nothing in between."""
    m = _block_mask()
    m[0, 0] = 0
    m[2, 4] = 1
    return m


# ------------------------------------------------------------- JSON payloads
def _flow_payload(fwd: np.ndarray, bwd: np.ndarray) -> list[dict]:
    return [
        {"forward": fwd[i].tolist(), "backward": bwd[i].tolist()}
        for i in range(fwd.shape[0])
    ]


def chain_break_create_payload() -> dict:
    n, h, w, first, last, fwd, bwd = chain_break_arrays()
    return {
        "frame_count": n,
        "height": h,
        "width": w,
        "first_mask": first.tolist(),
        "last_mask": last.tolist(),
        "flows": _flow_payload(fwd, bwd),
    }


def chain_break_mask_payload() -> dict:
    return {"mask": chain_break_completed_mask().tolist()}


def supplement_create_payload() -> dict:
    n, h, w, first, last, fwd, bwd = supplement_arrays()
    return {
        "frame_count": n,
        "height": h,
        "width": w,
        "first_mask": first.tolist(),
        "last_mask": last.tolist(),
        "flows": _flow_payload(fwd, bwd),
    }


def supplement_frame2_payload() -> dict:
    return {"mask": supplement_frame2_mask().tolist()}


def supplement_frame3_payload() -> dict:
    return {"mask": supplement_frame3_mask().tolist()}
