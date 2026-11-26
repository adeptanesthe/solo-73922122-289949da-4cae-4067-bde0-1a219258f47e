"""In-memory versioned project store.

Concurrency / consistency model
-------------------------------
* Every successful creation or mask submission produces a new immutable
  :class:`Snapshot` (full ``values``/``status``/``confirmed`` arrays).
* A project has a writer :class:`threading.RLock`.  Updates copy the current
  snapshot, recompute *only* the intervals adjacent to the submitted frame on
  the copies, and only then publish the new snapshot by atomically swapping
  the current-version pointer.
* Readers never take the writer lock: they dereference the current pointer
  once (or address an immutable historical version).  A multi-frame response
  is therefore always built from a single version, and an in-flight
  recomputation never blocks or contaminates a read.
* If validation or recomputation raises, the pointers of the old snapshot are
  left untouched: the original confirmed frames and results stay available.

The module-level ``recompute_barrier`` hook is an observability/test seam:
when set, it is invoked while the writer lock is held after copies have been
prepared but before the new snapshot is published.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from . import propagation as eng

# Signature: recompute_barrier(project_id: str) -> None.
# Tests assign a callable that blocks until an event is released, allowing
# deterministic proof that readers keep seeing the old version mid-update.
recompute_barrier: Callable[[str], None] | None = None


@dataclass(frozen=True)
class Snapshot:
    version: int
    created_at: datetime
    confirmed: dict[int, np.ndarray]  # frame index -> immutable uint8 HxW
    confirmed_frames: tuple[int, ...]
    values: np.ndarray  # (N, H, W) int8, -1 == needs paint
    status: np.ndarray  # (N, H, W) int8 status codes
    recomputed_frames: tuple[int, ...] = ()
    """Frames recomputed to produce this version (empty for v1 full build's
    interior is reported separately via ``built_frames`` semantics)."""
    recomputed_intervals: tuple[dict, ...] = field(default_factory=tuple)


class RecomputeError(RuntimeError):
    """Raised when interval recomputation fails after input validation."""


class ProjectNotFound(KeyError):
    pass


class VersionNotFound(KeyError):
    pass


@dataclass
class Project:
    project_id: str
    frame_count: int
    height: int
    width: int
    fwd: np.ndarray  # (N-1, H, W, 2) int64 immutable
    bwd: np.ndarray
    created_at: datetime
    lock: threading.RLock = field(default_factory=threading.RLock)
    snapshots: dict[int, Snapshot] = field(default_factory=dict)
    current_version: int = 1

    def current_snapshot(self) -> Snapshot:
        # Single atomic attribute read; the snapshot itself is frozen.
        return self.snapshots[self.current_version]

    def get_snapshot(self, version: int | None = None) -> Snapshot:
        if version is None:
            return self.current_snapshot()
        if version not in self.snapshots:
            raise VersionNotFound(version)
        return self.snapshots[version]


class ProjectStore:
    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ get
    def get(self, project_id: str) -> Project:
        project = self._projects.get(project_id)
        if project is None:
            raise ProjectNotFound(project_id)
        return project

    # --------------------------------------------------------------- create
    def create_project(
        self,
        frame_count: int,
        height: int,
        width: int,
        first_mask: np.ndarray,
        last_mask: np.ndarray,
        fwd: np.ndarray,
        bwd: np.ndarray,
    ) -> Project:
        project_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        values = np.full((frame_count, height, width),
                         eng.PENDING_VALUE, dtype=eng.VALUE_DTYPE)
        status = np.full((frame_count, height, width),
                         eng.STATUS_BROKEN, dtype=eng.STATUS_DTYPE)

        confirmed = {
            0: first_mask.copy(),
            frame_count - 1: last_mask.copy(),
        }
        values[0] = first_mask.astype(eng.VALUE_DTYPE)
        status[0] = eng.STATUS_CONFIRMED
        values[frame_count - 1] = last_mask.astype(eng.VALUE_DTYPE)
        status[frame_count - 1] = eng.STATUS_CONFIRMED

        try:
            eng.propagate_interval(
                0, frame_count - 1,
                first_mask, last_mask, fwd, bwd,
                height, width, values=values, status=status,
            )
        except Exception as exc:
            raise RecomputeError(f"initial propagation failed: {exc}") from exc

        snap = Snapshot(
            version=1,
            created_at=now,
            confirmed=dict(confirmed),
            confirmed_frames=tuple(sorted(confirmed)),
            values=values,
            status=status,
            recomputed_frames=tuple(range(1, frame_count - 1)),
            recomputed_intervals=({
                "left_confirmed_frame": 0,
                "right_confirmed_frame": frame_count - 1,
                "recomputed_frames": list(range(1, frame_count - 1)),
            },),
        )
        project = Project(
            project_id=project_id,
            frame_count=frame_count,
            height=height,
            width=width,
            fwd=fwd,
            bwd=bwd,
            created_at=now,
        )
        project.snapshots[1] = snap
        with self._guard:
            self._projects[project_id] = project
        return project

    # ------------------------------------------------- submit / correct mask
    def submit_mask(self, project_id: str, frame_index: int,
                    mask: np.ndarray) -> Snapshot:
        project = self.get(project_id)

        # Defence-in-depth validation; route layer normally does this first.
        if not (0 <= frame_index < project.frame_count):
            raise ValueError("frame_index out of range")
        mask = eng.normalize_mask(mask, project.height, project.width,
                                  name="mask")

        with project.lock:
            old = project.current_snapshot()

            # Work entirely on copies -> failure leaves ``old`` untouched.
            new_confirmed = dict(old.confirmed)
            new_confirmed[frame_index] = mask.copy()
            new_values = old.values.copy()
            new_status = old.status.copy()

            confirmed_indices = sorted(new_confirmed)
            left_anchor = max(
                (i for i in confirmed_indices if i < frame_index),
                default=None,
            )
            right_anchor = min(
                (i for i in confirmed_indices if i > frame_index),
                default=None,
            )

            intervals: list[dict] = []
            recomputed_frames: list[int] = []

            if recompute_barrier is not None:
                recompute_barrier(project_id)

            try:
                # The submitted frame itself is confirmed, never propagated.
                new_values[frame_index] = mask.astype(eng.VALUE_DTYPE)
                new_status[frame_index] = eng.STATUS_CONFIRMED

                if left_anchor is not None:
                    eng.propagate_interval(
                        left_anchor, frame_index,
                        new_confirmed[left_anchor], mask,
                        project.fwd, project.bwd,
                        project.height, project.width,
                        values=new_values, status=new_status,
                    )
                    frames = list(range(left_anchor + 1, frame_index))
                    recomputed_frames.extend(frames)
                    intervals.append({
                        "side": "left",
                        "left_confirmed_frame": left_anchor,
                        "right_confirmed_frame": frame_index,
                        "recomputed_frames": frames,
                    })

                if right_anchor is not None:
                    eng.propagate_interval(
                        frame_index, right_anchor,
                        mask, new_confirmed[right_anchor],
                        project.fwd, project.bwd,
                        project.height, project.width,
                        values=new_values, status=new_status,
                    )
                    frames = list(range(frame_index + 1, right_anchor))
                    recomputed_frames.extend(frames)
                    intervals.append({
                        "side": "right",
                        "left_confirmed_frame": frame_index,
                        "right_confirmed_frame": right_anchor,
                        "recomputed_frames": frames,
                    })
            except Exception as exc:
                # Copies are discarded; do NOT touch project.current_version.
                raise RecomputeError(
                    f"recomputation failed, previous version "
                    f"{old.version} retained: {exc}"
                ) from exc

            new_version = old.version + 1
            new_snap = Snapshot(
                version=new_version,
                created_at=datetime.now(timezone.utc),
                confirmed=new_confirmed,
                confirmed_frames=tuple(sorted(new_confirmed)),
                values=new_values,
                status=new_status,
                recomputed_frames=tuple(sorted(set(recomputed_frames))),
                recomputed_intervals=tuple(intervals),
            )
            # Publish atomically under the writer lock.
            project.snapshots[new_version] = new_snap
            project.current_version = new_version
            return new_snap


store = ProjectStore()
