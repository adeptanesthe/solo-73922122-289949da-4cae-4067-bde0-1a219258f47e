"""Tests for the NumPy propagation engine, including a brute-force oracle."""

from __future__ import annotations

import numpy as np
import pytest

from app import propagation as eng


# ----------------------------------------------------------- brute force oracle
def brute_left(k, lo, r, c, fwd, bwd, h, w, anchor):
    cr, cc = r, c
    for i in range(k - 1, lo - 1, -1):
        dr, dc = map(int, bwd[i, cr, cc])
        nr, nc = cr + dr, cc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        rdr, rdc = map(int, fwd[i, nr, nc])
        if nr + rdr != cr or nc + rdc != cc:
            return None
        cr, cc = nr, nc
    return int(anchor[cr, cc])


def brute_right(k, hi, r, c, fwd, bwd, h, w, anchor):
    cr, cc = r, c
    for i in range(k, hi):
        dr, dc = map(int, fwd[i, cr, cc])
        nr, nc = cr + dr, cc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        rdr, rdc = map(int, bwd[i, nr, nc])
        if nr + rdr != cr or nc + rdc != cc:
            return None
        cr, cc = nr, nc
    return int(anchor[cr, cc])


def brute_expected(lo, hi, fwd, bwd, left_mask, right_mask, h, w):
    """Independent per-pixel implementation of the spec."""
    n = fwd.shape[0] + 1
    values = np.full((n, h, w), -1, dtype=np.int8)
    status = np.full((n, h, w), eng.STATUS_BROKEN, dtype=np.int8)
    for k in range(lo + 1, hi):
        for r in range(h):
            for c in range(w):
                lv = brute_left(k, lo, r, c, fwd, bwd, h, w, left_mask)
                rv = brute_right(k, hi, r, c, fwd, bwd, h, w, right_mask)
                if lv is not None and rv is not None and lv == rv:
                    status[k, r, c] = eng.STATUS_AGREE
                    values[k, r, c] = lv
                elif lv is not None and rv is not None:
                    status[k, r, c] = eng.STATUS_CONFLICT
                elif lv is not None:
                    status[k, r, c] = eng.STATUS_LEFT
                    values[k, r, c] = lv
                elif rv is not None:
                    status[k, r, c] = eng.STATUS_RIGHT
                    values[k, r, c] = rv
                else:
                    status[k, r, c] = eng.STATUS_BROKEN
    return values, status


# ----------------------------------------------------------------- fixed cases
def test_identity_equal_masks_all_agree():
    n, h, w = 4, 3, 4
    fwd = np.zeros((n - 1, h, w, 2), dtype=np.int64)
    bwd = np.zeros_like(fwd)
    m = np.random.RandomState(0).randint(0, 2, (h, w)).astype(np.uint8)
    values, status = eng.propagate_interval(0, n - 1, m, m, fwd, bwd, h, w)
    for k in range(1, n - 1):
        assert (status[k] == eng.STATUS_AGREE).all()
        np.testing.assert_array_equal(values[k], m.astype(np.int8))


def test_identity_different_masks_conflict():
    n, h, w = 3, 2, 2
    fwd = np.zeros((2, h, w, 2), dtype=np.int64)
    bwd = np.zeros_like(fwd)
    a = np.zeros((h, w), np.uint8)
    b = np.ones((h, w), np.uint8)
    values, status = eng.propagate_interval(0, 2, a, b, fwd, bwd, h, w)
    assert (status[1] == eng.STATUS_CONFLICT).all()
    assert (values[1] == -1).all()


def test_out_of_bounds_left_gives_right_only():
    h = w = 2
    fwd = np.zeros((2, h, w, 2), dtype=np.int64)
    bwd = np.zeros_like(fwd)
    bwd[0, 0, 0] = (-1, 0)  # only pixel (0,0) breaks on the left
    a = np.zeros((h, w), np.uint8)
    b = np.zeros((h, w), np.uint8)
    _, status = eng.propagate_interval(0, 2, a, b, fwd, bwd, h, w)
    assert status[1, 0, 0] == eng.STATUS_RIGHT
    assert (status[1] != eng.STATUS_RIGHT).sum() == h * w - 1


def test_roundtrip_mismatch_breaks_chain():
    h = w = 3
    fwd = np.zeros((2, h, w, 2), dtype=np.int64)
    bwd = np.zeros_like(fwd)
    # (1,1) -> (1,2) via backward; fwd at (1,2) is zero -> no return.
    bwd[0, 1, 1] = (0, 1)
    a = np.zeros((h, w), np.uint8)
    b = np.zeros((h, w), np.uint8)
    _, status = eng.propagate_interval(0, 2, a, b, fwd, bwd, h, w)
    assert status[1, 1, 1] in (eng.STATUS_RIGHT, eng.STATUS_BROKEN)
    # right chain is fine -> RIGHT
    assert status[1, 1, 1] == eng.STATUS_RIGHT
    # landing pixel (1,2) itself stays consistent (zero/zero round trip)
    assert status[1, 1, 2] == eng.STATUS_AGREE


def test_consistent_nonzero_flow_crosses():
    """A mutually consistent fwd/bwd pair propagates the anchor value."""
    h = w = 3
    fwd = np.zeros((2, h, w, 2), dtype=np.int64)
    bwd = np.zeros_like(fwd)
    # Pixel (2,0) in frame1 -> (1,0) in frame0 ...
    bwd[0, 2, 0] = (-1, 0)
    # ... and fwd at (1,0) maps back to (2,0).
    fwd[0, 1, 0] = (1, 0)
    # NOTE: this also makes identity pixel (1,0) inconsistent, which is the
    # expected per-pixel FB check; here we only assert (2,0) survives.
    a = np.zeros((h, w), np.uint8)
    a[1, 0] = 1
    b = np.zeros((h, w), np.uint8)
    b[2, 0] = 1  # right identity chain agrees on 1 at pixel (2,0)
    values, status = eng.propagate_interval(0, 2, a, b, fwd, bwd, h, w)
    assert status[1, 2, 0] == eng.STATUS_AGREE
    assert values[1, 2, 0] == 1


# ------------------------------------------------------------- random oracle
@pytest.mark.parametrize("seed", range(60))
def test_engine_matches_bruteforce_random(seed):
    rng = np.random.RandomState(seed)
    h = int(rng.randint(1, 5))
    w = int(rng.randint(1, 5))
    n = int(rng.randint(3, 7))
    lo = 0
    hi = n - 1
    fwd = rng.randint(-2, 3, size=(n - 1, h, w, 2)).astype(np.int64)
    bwd = rng.randint(-2, 3, size=(n - 1, h, w, 2)).astype(np.int64)
    left_mask = rng.randint(0, 2, (h, w)).astype(np.uint8)
    right_mask = rng.randint(0, 2, (h, w)).astype(np.uint8)

    values, status = eng.propagate_interval(
        lo, hi, left_mask, right_mask, fwd, bwd, h, w
    )
    exp_values, exp_status = brute_expected(
        lo, hi, fwd, bwd, left_mask, right_mask, h, w
    )
    np.testing.assert_array_equal(status, exp_status)
    np.testing.assert_array_equal(values, exp_values)


@pytest.mark.parametrize("seed", range(20))
def test_engine_matches_bruteforce_random_subinterval(seed):
    """Nearest-anchor intervals inside a larger volume."""
    rng = np.random.RandomState(1000 + seed)
    h = int(rng.randint(2, 5))
    w = int(rng.randint(2, 5))
    n = 9
    lo = int(rng.randint(0, 3))
    hi = int(rng.randint(6, 9))
    fwd = rng.randint(-1, 2, size=(n - 1, h, w, 2)).astype(np.int64)
    bwd = rng.randint(-1, 2, size=(n - 1, h, w, 2)).astype(np.int64)
    lm = rng.randint(0, 2, (h, w)).astype(np.uint8)
    rm = rng.randint(0, 2, (h, w)).astype(np.uint8)

    values = np.full((n, h, w), -7, dtype=np.int8)
    status = np.full((n, h, w), -1, dtype=np.int8)
    eng.propagate_interval(lo, hi, lm, rm, fwd, bwd, h, w,
                           values=values, status=status)

    exp_values, exp_status = brute_expected(
        lo, hi, fwd, bwd, lm, rm, h, w
    )
    # Only interval interior must be written.
    for k in range(lo + 1, hi):
        np.testing.assert_array_equal(status[k], exp_status[k])
        np.testing.assert_array_equal(values[k], exp_values[k])
    # Outside interval stays untouched.
    assert (status[: lo + 1] == -1).all()
    assert (status[hi:] == -1).all()
    assert (values[: lo + 1] == -7).all()
    assert (values[hi:] == -7).all()


def test_max_grid_full_run_is_fast():
    """40 frames x 128x128 must complete comfortably within a few seconds."""
    import time

    n, h, w = 40, 128, 128
    fwd = np.zeros((n - 1, h, w, 2), dtype=np.int64)
    bwd = np.zeros_like(fwd)
    m = np.zeros((h, w), np.uint8)
    m[32:96, 32:96] = 1
    t0 = time.perf_counter()
    values, status = eng.propagate_interval(0, n - 1, m, m, fwd, bwd, h, w)
    elapsed = time.perf_counter() - t0
    assert (status[1:-1] == eng.STATUS_AGREE).all()
    assert elapsed < 5.0, f"propagation too slow: {elapsed:.2f}s"
