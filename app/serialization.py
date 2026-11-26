"""Snapshot -> JSON-ready response dictionaries."""

from __future__ import annotations

import numpy as np

from . import propagation as eng
from .store import Project, Snapshot


def frame_dict(snap: Snapshot, frame_index: int,
               with_labels: bool = True) -> dict:
    values = snap.values[frame_index]
    status = snap.status[frame_index]
    labels = np.vectorize(eng.STATUS_LABELS.__getitem__)(status)
    needs_paint = np.isin(
        status, list(eng.NEEDS_PAINT_STATUSES)
    ).astype(np.uint8)

    result = {
        "frame": frame_index,
        "version": snap.version,
        "confirmed": frame_index in snap.confirmed,
        "mask": values.astype(int).tolist(),
        "needs_paint": needs_paint.tolist(),
        "status_codes": status.astype(int).tolist(),
    }
    if with_labels:
        result["status"] = labels.tolist()
    return result


def status_legend() -> dict:
    return {str(code): label for code, label in eng.STATUS_LABELS.items()}


def frames_response(project: Project, snap: Snapshot,
                    version_requested: int | None = None) -> dict:
    frames = [frame_dict(snap, i) for i in range(project.frame_count)]
    pending_per_frame = [
        int(np.isin(snap.status[i], list(eng.NEEDS_PAINT_STATUSES)).sum())
        for i in range(project.frame_count)
    ]
    status_counts = {
        eng.STATUS_LABELS[code]: int((snap.status == code).sum())
        for code in eng.STATUS_LABELS
    }
    return {
        "project_id": project.project_id,
        "version": snap.version,
        "requested_version": version_requested,
        "is_current_version": snap.version == project.current_version,
        "frame_count": project.frame_count,
        "height": project.height,
        "width": project.width,
        "confirmed_frames": list(snap.confirmed_frames),
        "status_legend": status_legend(),
        "pending_pixels_total": int(sum(pending_per_frame)),
        "pending_pixels_per_frame": pending_per_frame,
        "status_counts": status_counts,
        "frames": frames,
    }


def frame_response(project: Project, snap: Snapshot, frame_index: int) -> dict:
    body = frame_dict(snap, frame_index)
    body.update({
        "project_id": project.project_id,
        "version": snap.version,
        "is_current_version": snap.version == project.current_version,
        "height": project.height,
        "width": project.width,
        "status_legend": status_legend(),
    })
    return body


def project_meta(project: Project) -> dict:
    snap = project.current_snapshot()
    return {
        "project_id": project.project_id,
        "frame_count": project.frame_count,
        "height": project.height,
        "width": project.width,
        "current_version": project.current_version,
        "confirmed_frames": list(snap.confirmed_frames),
        "created_at": project.created_at.isoformat(),
        "links": {
            "all_frames": f"/api/projects/{project.project_id}/frames",
            "docs": "/docs",
        },
    }


def submit_response(project: Project, old_version: int,
                    snap: Snapshot, frame_index: int) -> dict:
    pending_per_frame = [
        int(np.isin(snap.status[i], list(eng.NEEDS_PAINT_STATUSES)).sum())
        for i in range(project.frame_count)
    ]
    return {
        "project_id": project.project_id,
        "frame": frame_index,
        "previous_version": old_version,
        "version": snap.version,
        "confirmed_frames": list(snap.confirmed_frames),
        "recomputed_frames": list(snap.recomputed_frames),
        "recomputed_intervals": list(snap.recomputed_intervals),
        "pending_pixels_total": int(sum(pending_per_frame)),
        "pending_pixels_per_frame": pending_per_frame,
        "all_frames_url": (
            f"/api/projects/{project.project_id}/frames?version={snap.version}"
        ),
        "old_version_frames_url": (
            f"/api/projects/{project.project_id}/frames?version={old_version}"
        ),
    }
