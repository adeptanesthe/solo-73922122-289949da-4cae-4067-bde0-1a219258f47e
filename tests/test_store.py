"""Tests for versioned store: recompute scope, snapshots, failure atomicity."""

from __future__ import annotations

import numpy as np
import pytest

from app import propagation as eng
from app.store import ProjectStore, RecomputeError


def _identity_project(store: ProjectStore, n=7, h=3, w=4):
    fwd = np.zeros((n - 1, h, w, 2), dtype=np.int64)
    bwd = np.zeros_like(fwd)
    first = np.zeros((h, w), np.uint8)
    last = np.ones((h, w), np.uint8)  # endpoints disagree everywhere
    return store.create_project(n, h, w, first, last, fwd, bwd)


def test_create_v1_confirmed_endpoints_and_conflicts():
    s = ProjectStore()
    p = _identity_project(s)
    snap = p.current_snapshot()
    assert snap.version == 1
    assert snap.confirmed_frames == (0, 6)
    assert (snap.status[0] == eng.STATUS_CONFIRMED).all()
    assert (snap.status[6] == eng.STATUS_CONFIRMED).all()
    # identity + disagreeing endpoints => every interior pixel conflicts
    for k in range(1, 6):
        assert (snap.status[k] == eng.STATUS_CONFLICT).all()


def test_submit_middle_only_recomputes_adjacent_intervals():
    s = ProjectStore()
    p = _identity_project(s)
    pid = p.project_id
    old = p.current_snapshot()

    mask3 = np.zeros((3, 4), np.uint8)  # agrees with frame 0
    new = s.submit_mask(pid, 3, mask3)

    assert new.version == 2
    assert new.confirmed_frames == (0, 3, 6)
    assert tuple(new.recomputed_frames) == (1, 2, 4, 5)
    intervals = {(iv["left_confirmed_frame"], iv["right_confirmed_frame"]):
                 tuple(iv["recomputed_frames"])
                 for iv in new.recomputed_intervals}
    assert intervals[(0, 3)] == (1, 2)
    assert intervals[(3, 6)] == (4, 5)

    # Left interval now agrees with frame 0 (all zero).
    for k in (1, 2):
        assert (new.status[k] == eng.STATUS_AGREE).all()
        assert (new.values[k] == 0).all()
    # Right interval still conflicts (frame3 zeros vs frame6 ones).
    for k in (4, 5):
        assert (new.status[k] == eng.STATUS_CONFLICT).all()

    # Old immutable snapshot completely unchanged.
    assert p.get_snapshot(1) is old
    assert old.version == 1
    for k in range(1, 6):
        assert (old.status[k] == eng.STATUS_CONFLICT).all()


def test_second_submit_uses_new_nearest_confirmed():
    s = ProjectStore()
    p = _identity_project(s)
    pid = p.project_id
    s.submit_mask(pid, 3, np.zeros((3, 4), np.uint8))
    v2 = p.current_snapshot()

    # Now insert frame 5 (zeros): nearest anchors are 3(zeros) and 5(zeros),
    # so frame 4 agrees on 0. If the code wrongly used old anchor 6 (ones),
    # frame 4 would conflict instead.
    v3 = s.submit_mask(pid, 5, np.zeros((3, 4), np.uint8))
    assert v3.confirmed_frames == (0, 3, 5, 6)
    assert tuple(v3.recomputed_frames) == (4,)

    # Frames 1,2 results must be untouched by the second submission.
    np.testing.assert_array_equal(v3.values[1:3], v2.values[1:3])
    np.testing.assert_array_equal(v3.status[1:3], v2.status[1:3])
    # Frame 4 between two zero anchors now agrees on 0.
    assert (v3.status[4] == eng.STATUS_AGREE).all()
    assert (v3.values[4] == 0).all()


def test_correct_existing_confirmed_frame():
    s = ProjectStore()
    p = _identity_project(s)
    pid = p.project_id
    s.submit_mask(pid, 3, np.zeros((3, 4), np.uint8))
    # Correct frame 3 to ones: same two adjacent intervals recomputed.
    v3 = s.submit_mask(pid, 3, np.ones((3, 4), np.uint8))
    assert tuple(v3.recomputed_frames) == (1, 2, 4, 5)
    # Left interval frame0=zeros vs frame3=ones -> conflict.
    for k in (1, 2):
        assert (v3.status[k] == eng.STATUS_CONFLICT).all()
    # Right interval frame3=ones vs frame6=ones -> agree on 1.
    for k in (4, 5):
        assert (v3.status[k] == eng.STATUS_AGREE).all()
        assert (v3.values[k] == 1).all()


def test_submit_endpoint_frame_zero_has_only_right_interval():
    s = ProjectStore()
    p = _identity_project(s, n=4)
    v2 = s.submit_mask(p.project_id, 0, np.ones((3, 4), np.uint8))
    assert v2.confirmed_frames == (0, 3)
    assert tuple(v2.recomputed_frames) == (1, 2)
    iv = list(v2.recomputed_intervals)
    assert len(iv) == 1
    assert iv[0]["side"] == "right"
    assert iv[0]["left_confirmed_frame"] == 0
    assert iv[0]["right_confirmed_frame"] == 3


def test_invalid_mask_keeps_previous_state():
    s = ProjectStore()
    p = _identity_project(s)
    with pytest.raises(ValueError):
        s.submit_mask(p.project_id, 2, np.ones((2, 2), np.uint8))
    assert p.current_version == 1
    assert p.current_snapshot().confirmed_frames == (0, 6)

    bad = np.ones((3, 4), np.uint8)
    bad[0, 0] = 7
    with pytest.raises(ValueError):
        s.submit_mask(p.project_id, 2, bad)
    assert p.current_version == 1


def test_recompute_failure_preserves_snapshot(monkeypatch):
    s = ProjectStore()
    p = _identity_project(s)
    pid = p.project_id

    def boom(*a, **k):
        raise RuntimeError("simulated recompute failure")

    monkeypatch.setattr("app.propagation.propagate_interval", boom)
    with pytest.raises(RecomputeError):
        s.submit_mask(pid, 3, np.zeros((3, 4), np.uint8))

    # Pointer and confirmed frames remain the original version.
    assert p.current_version == 1
    snap = p.current_snapshot()
    assert snap.confirmed_frames == (0, 6)
    assert 2 not in p.snapshots
