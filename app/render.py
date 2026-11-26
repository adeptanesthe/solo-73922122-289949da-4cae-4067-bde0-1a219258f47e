"""Dependency-free SVG rendering of propagation results.

Colors:
* deep indigo  -> mask value 1 (color block)
* near white   -> mask value 0
* red          -> ``broken`` chain (paint needed)
* amber        -> ``conflict`` (paint needed)
* green border -> confirmed keyframe
"""

from __future__ import annotations

from html import escape

from . import propagation as eng
from .store import Project, Snapshot

COLOR_ONE = "#4f46e5"
COLOR_ZERO = "#f1f5f9"
COLOR_BROKEN = "#ef4444"
COLOR_CONFLICT = "#f59e0b"
GRID = "#94a3b8"
CONFIRMED_STROKE = "#16a34a"

_PENDING_FILL = {
    eng.STATUS_BROKEN: COLOR_BROKEN,
    eng.STATUS_CONFLICT: COLOR_CONFLICT,
}


def _cell_fill(value: int, status: int) -> str:
    if status in _PENDING_FILL:
        return _PENDING_FILL[status]
    return COLOR_ONE if value == 1 else COLOR_ZERO


def frame_svg(snap: Snapshot, frame_index: int, cell: int = 22,
              project: Project | None = None) -> str:
    h = snap.values.shape[1]
    w = snap.values.shape[2]
    pad = 28
    label_h = 26
    width = w * cell + pad * 2
    height = h * cell + pad + label_h + 14

    values = snap.values[frame_index]
    status = snap.status[frame_index]
    confirmed = frame_index in snap.confirmed

    parts: list[str] = []
    border = (
        f'stroke="{CONFIRMED_STROKE}" stroke-width="4"'
        if confirmed else f'stroke="{GRID}" stroke-width="1"'
    )
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace">'
    )
    parts.append(f'<rect width="100%" height="100%" fill="#ffffff" rx="8"/>')
    title = (
        f"frame {frame_index} · version {snap.version} · "
        + ("confirmed" if confirmed else "propagated")
    )
    parts.append(
        f'<text x="{pad}" y="20" font-size="14" font-weight="700" '
        f'fill="#0f172a">{escape(title)}</text>'
    )
    parts.append(
        f'<g transform="translate({pad},{label_h})" {border}>'
    )

    for r in range(h):
        for c in range(w):
            x, y = c * cell, r * cell
            v = int(values[r, c])
            s = int(status[r, c])
            fill = _cell_fill(v, s)
            label = eng.STATUS_LABELS[s]
            tip = f"({r},{c}) {label}, value={v}"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="{GRID}" stroke-width="0.5">'
                f"<title>{escape(tip)}</title></rect>"
            )
            if s in (eng.STATUS_BROKEN, eng.STATUS_CONFLICT):
                parts.append(
                    f'<text x="{x + cell / 2}" y="{y + cell * 0.72}" '
                    f'text-anchor="middle" font-size="{int(cell * 0.7)}" '
                    f'fill="#ffffff" font-weight="700">?</text>'
                )
    parts.append("</g>")

    legend_y = label_h + h * cell + 18
    legend = [
        (COLOR_ONE, "value 1"),
        (COLOR_ZERO, "value 0"),
        (COLOR_BROKEN, "broken"),
        (COLOR_CONFLICT, "conflict"),
    ]
    lx = pad
    for color, text in legend:
        parts.append(
            f'<rect x="{lx}" y="{legend_y - 10}" width="12" height="12" '
            f'fill="{color}" stroke="{GRID}" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{lx + 16}" y="{legend_y}" font-size="12" '
            f'fill="#334155">{text}</text>'
        )
        lx += 24 + len(text) * 7 + 18
    if confirmed:
        parts.append(
            f'<rect x="{lx}" y="{legend_y - 10}" width="12" height="12" '
            f'fill="none" stroke="{CONFIRMED_STROKE}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{lx + 16}" y="{legend_y}" font-size="12" '
            f'fill="#334155">confirmed frame</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def sheet_svg(project: Project, snap: Snapshot, cell: int = 16) -> str:
    n, h, w = project.frame_count, project.height, project.width
    # Keep very large sheets compact; tooltips are omitted past a threshold.
    total_cells = n * h * w
    if total_cells > 40_000:
        cell = max(3, min(cell, 6))
    use_titles = total_cells <= 20_000
    gap = 26 if cell >= 10 else 12
    frame_w = w * cell
    frame_h = h * cell
    pad = 24
    width = pad * 2 + n * frame_w + max(0, n - 1) * gap
    height = pad * 2 + frame_h + 70

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'style="max-width:100%;height:auto" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad}" y="20" font-size="15" font-weight="700" '
        f'fill="#0f172a">{escape(project.project_id[:8])} · '
        f'version {snap.version} · red=broken amber=conflict '
        f'green label=confirmed</text>',
    ]

    for f in range(n):
        x0 = pad + f * (frame_w + gap)
        y0 = pad + 18
        confirmed = f in snap.confirmed
        label_color = CONFIRMED_STROKE if confirmed else "#334155"
        parts.append(
            f'<text x="{x0}" y="{y0 - 6}" font-size="13" font-weight="700" '
            f'fill="{label_color}">f{f}{"*" if confirmed else ""}</text>'
        )
        for r in range(h):
            for c in range(w):
                v = int(snap.values[f, r, c])
                s = int(snap.status[f, r, c])
                fill = _cell_fill(v, s)
                parts.append(
                    f'<rect x="{x0 + c * cell}" y="{y0 + r * cell}" '
                    f'width="{cell}" height="{cell}" fill="{fill}" '
                    f'stroke="{GRID}" stroke-width="0.4">'
                    + (
                        f"<title>f{f} ({r},{c}) "
                        f"{eng.STATUS_LABELS[s]} value={v}</title>"
                        if use_titles else ""
                    )
                    + "</rect>"
                )
        stroke = CONFIRMED_STROKE if confirmed else GRID
        sw = 3 if confirmed else 1
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{frame_w}" height="{frame_h}" '
            f'fill="none" stroke="{stroke}" stroke-width="{sw}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)
