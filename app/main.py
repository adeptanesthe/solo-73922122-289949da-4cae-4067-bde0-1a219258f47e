"""FastAPI entrypoint: versioned binary-mask propagation API."""

from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from . import demo, propagation as eng, serialization as ser
from .render import frame_svg, sheet_svg
from .schemas import MaskUpdate, ProjectCreate
from .store import (
    ProjectNotFound,
    RecomputeError,
    VersionNotFound,
    store,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

TAGS_METADATA = [
    {"name": "projects", "description": "Create projects and query propagated frames"},
    {"name": "masks", "description": "Submit / correct confirmed frame masks"},
    {"name": "visual", "description": "Dependency-free SVG previews"},
    {"name": "demo", "description": "One-click deterministic chain-break / supplement scenarios"},
]

app = FastAPI(
    title="Mask Propagation API · 蒙版传播 API",
    version="1.0.0",
    description=(
        "## 用途\n"
        "动画色块只在首尾（或若干确认帧）画好时，用相邻帧的**双向整数光流**\n"
        "把二值蒙版传播到未确认帧，并标出仍需补画的像素。\n\n"
        "## 追溯规则\n"
        "未确认帧的每个像素分别向左右最近的确认帧追溯：\n"
        "1. 任一步坐标越界，该侧链断；\n"
        "2. 去程后用反向光流做往返，坐标回不到本步起点，该侧链断；\n"
        "3. 仅一侧可用 → 取该侧值（状态 `left` / `right`）；\n"
        "4. 两侧都可用且同值 → 沿用（`agree`）；\n"
        "5. 两侧冲突（`conflict`）或都断链（`broken`）→ `mask=-1`，计入 `needs_paint`。\n\n"
        "状态码：0 broken · 1 left · 2 right · 3 agree · 4 conflict · 5 confirmed。\n\n"
        "## 版本与补交\n"
        "- 每次补交/更正蒙版只重算它与左右最近确认帧之间的帧；\n"
        "- 每次成功写入生成新的不可变版本，重算期间查询仍返回旧版；\n"
        "- 一次多帧查询中的所有帧来自同一版本（响应顶层和每帧都带 `version`）；\n"
        "- 校验或重算失败时，原确认帧与传播结果原样保留。\n\n"
        "## 在 /docs 里演示\n"
        "1. `POST /api/projects`，Request body 示例选 **“断链演示（3帧3×5）”** → Execute，复制 `project_id`；\n"
        "2. `GET /api/projects/{project_id}/frames` 查看 `needs_paint`、`status=broken/conflict`；\n"
        "3. `PUT /api/projects/{project_id}/frames/1/mask`（body 选 **“补交：第1帧成品蒙版”**）；\n"
        "4. 再查 `frames?version=1`（旧版仍在）与不带 version 的当前版；\n"
        "5. 也可直接调用 `POST /api/demo/chain-break` 和 `POST /api/demo/supplement` 一键看结果。\n"
    ),
    openapi_tags=TAGS_METADATA,
    docs_url=None,  # served by custom self-hosted route below
)


# --------------------------------------------------------- self-hosted Swagger
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui() -> HTMLResponse:
    """Swagger UI with vendored assets so /docs works without a CDN."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} · Docs",
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url="/static/favicon.svg",
    )


@app.get("/static/swagger-ui-bundle.js", include_in_schema=False)
def swagger_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "swagger-ui-bundle.js",
                        media_type="text/javascript; charset=utf-8")


@app.get("/static/swagger-ui.css", include_in_schema=False)
def swagger_css() -> FileResponse:
    return FileResponse(STATIC_DIR / "swagger-ui.css",
                        media_type="text/css; charset=utf-8")


@app.get("/static/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg",
                        media_type="image/svg+xml")


# ---------------------------------------------------------------- exception mapping
@app.exception_handler(ProjectNotFound)
async def project_not_found_handler(request, exc):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": f"project not found: {exc}"})


@app.exception_handler(VersionNotFound)
async def version_not_found_handler(request, exc):  # noqa: ANN001
    return JSONResponse(status_code=404, content={"detail": f"version not found: {exc}"})


@app.exception_handler(RecomputeError)
async def recompute_error_handler(request, exc):  # noqa: ANN001
    # State has already been preserved by the store; just report failure.
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ----------------------------------------------------------------------- helpers
def _create_arrays(payload: ProjectCreate):
    h, w = payload.height, payload.width
    try:
        first = eng.normalize_mask(payload.first_mask, h, w, "first_mask")
        last = eng.normalize_mask(payload.last_mask, h, w, "last_mask")
        raw_fwd = [flow.forward for flow in payload.flows]
        raw_bwd = [flow.backward for flow in payload.flows]
        fwd, bwd = eng.normalize_flows(
            raw_fwd, raw_bwd, payload.frame_count, h, w
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return first, last, fwd, bwd


def _get_project(project_id: str):
    try:
        return store.get(project_id)
    except ProjectNotFound as exc:
        raise HTTPException(status_code=404,
                            detail=f"project not found: {project_id}") from exc


def _get_snapshot(project, version: int | None):
    try:
        return project.get_snapshot(version)
    except VersionNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"version {version} not found for project {project.project_id}",
        ) from exc


# ------------------------------------------------------------------------- health
@app.get("/health", tags=["projects"])
def health() -> dict:
    return {"status": "ok", "service": "mask-propagation"}


# --------------------------------------------------------------------------- root
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mask Propagation API</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:860px;
      margin:48px auto;padding:0 20px;color:#0f172a;line-height:1.65}
 code,pre{background:#f1f5f9;border-radius:6px}
 code{padding:2px 6px} pre{padding:14px 16px;overflow:auto}
 a.btn{display:inline-block;background:#4f46e5;color:#fff;padding:10px 18px;
      border-radius:8px;text-decoration:none;font-weight:600;margin-right:10px}
 a.sec{background:#0f172a}
 .muted{color:#64748b}
</style></head><body>
<h1>蒙版传播 API <span class="muted">Mask Propagation</span></h1>
<p>接收首尾二值蒙版与相邻帧双向整数光流，传播中间帧并标出待补像素；
支持补交/更正、局部重算与不可变版本查询。</p>
<p>
 <a class="btn" href="/docs">打开 /docs 交互演示</a>
 <a class="btn sec" href="/redoc">ReDoc</a>
 <a class="btn sec" href="/openapi.json">OpenAPI JSON</a>
</p>
<h2>30 秒命令行自测</h2>
<pre># 1) 一键制造断链场景
curl -s -X POST http://localhost:8000/api/demo/chain-break | python -m json.tool

# 2) 一键演示补交第 2 帧与局部重算（5 帧场景）
curl -s -X POST http://localhost:8000/api/demo/supplement | python -m json.tool</pre>
<p class="muted">在 <code>/docs</code> 中按页面顶部描述的 5 步操作，可手动演示断链、
旧版本查询与补交重算。红色格=broken，琥珀格=conflict，均为待补像素。</p>
</body></html>"""


# ------------------------------------------------------------------------ projects
_CREATE_EXAMPLES = {
    "chain_break": {
        "summary": "断链演示（3 帧 · 3×5）",
        "description": (
            "含越界、往返不一致、单侧可用、两侧一致与冲突。"
            "提交后用返回的 project_id 查询 /frames。"
        ),
        "value": demo.chain_break_create_payload(),
    },
    "supplement": {
        "summary": "补交/局部重算演示（5 帧 · 3×5）",
        "description": "首尾在 (0,0)、(2,4) 冲突；补交第 2 帧后只重算第 1、3 帧。",
        "value": demo.supplement_create_payload(),
    },
}

_MASK_EXAMPLES = {
    "chain_break_frame1": {
        "summary": "断链场景：第 1 帧成品蒙版（路径 frame_index=1）",
        "value": demo.chain_break_mask_payload(),
    },
    "supplement_frame2": {
        "summary": "5 帧场景：补交第 2 帧（路径 frame_index=2）",
        "value": demo.supplement_frame2_payload(),
    },
    "supplement_frame3": {
        "summary": "5 帧场景：再补交第 3 帧（路径 frame_index=3）",
        "value": demo.supplement_frame3_payload(),
    },
}


@app.post("/api/projects", status_code=201, tags=["projects"])
def create_project(
    payload: ProjectCreate = Body(..., openapi_examples=_CREATE_EXAMPLES),
) -> dict:
    first, last, fwd, bwd = _create_arrays(payload)
    project = store.create_project(
        frame_count=payload.frame_count,
        height=payload.height,
        width=payload.width,
        first_mask=first,
        last_mask=last,
        fwd=fwd,
        bwd=bwd,
    )
    snap = project.current_snapshot()
    result = ser.frames_response(project, snap)
    result["project"] = ser.project_meta(project)
    result["next_steps"] = [
        f"GET /api/projects/{project.project_id}/frames",
        f"PUT /api/projects/{project.project_id}/frames/{{frame_index}}/mask",
        f"GET /api/projects/{project.project_id}/sheet.svg",
    ]
    return result


@app.get("/api/projects/{project_id}", tags=["projects"])
def get_project(project_id: str) -> dict:
    project = _get_project(project_id)
    return ser.project_meta(project)


@app.get("/api/projects/{project_id}/frames", tags=["projects"])
def list_frames(project_id: str, version: int | None = None) -> dict:
    """All frames from one immutable version (current if ``version`` omitted)."""
    project = _get_project(project_id)
    snap = _get_snapshot(project, version)
    return ser.frames_response(project, snap, version_requested=version)


@app.get("/api/projects/{project_id}/frames/{frame_index}", tags=["projects"])
def get_frame(project_id: str, frame_index: int,
                    version: int | None = None) -> dict:
    project = _get_project(project_id)
    if not (0 <= frame_index < project.frame_count):
        raise HTTPException(
            status_code=422,
            detail=f"frame_index must be in [0, {project.frame_count - 1}]",
        )
    snap = _get_snapshot(project, version)
    return ser.frame_response(project, snap, frame_index)


# --------------------------------------------------------------------------- masks
@app.put("/api/projects/{project_id}/frames/{frame_index}/mask", tags=["masks"])
def submit_mask(
    project_id: str,
    frame_index: int,
    payload: MaskUpdate = Body(..., openapi_examples=_MASK_EXAMPLES),
) -> dict:
    project = _get_project(project_id)
    if not (0 <= frame_index < project.frame_count):
        raise HTTPException(
            status_code=422,
            detail=f"frame_index must be in [0, {project.frame_count - 1}]",
        )

    # Validate BEFORE touching any project state -> invalid submit is a no-op.
    try:
        mask = eng.normalize_mask(
            payload.mask, project.height, project.width, "mask"
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    old_version = project.current_version
    # RecomputeError is mapped to 500 by the exception handler; the store
    # guarantees the previous snapshot remains current.
    snap = store.submit_mask(project_id, frame_index, mask)
    return ser.submit_response(project, old_version, snap, frame_index)


# -------------------------------------------------------------------------- visual
@app.get("/api/projects/{project_id}/sheet.svg", tags=["visual"],
         response_class=Response)
def get_sheet_svg(project_id: str, version: int | None = None) -> Response:
    project = _get_project(project_id)
    snap = _get_snapshot(project, version)
    svg = sheet_svg(project, snap)
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/projects/{project_id}/frames/{frame_index}/svg",
         tags=["visual"], response_class=Response)
def get_frame_svg(project_id: str, frame_index: int,
                        version: int | None = None) -> Response:
    project = _get_project(project_id)
    if not (0 <= frame_index < project.frame_count):
        raise HTTPException(status_code=422, detail="frame_index out of range")
    snap = _get_snapshot(project, version)
    svg = frame_svg(snap, frame_index, project=project)
    return Response(content=svg, media_type="image/svg+xml")


# ---------------------------------------------------------------------------- demo
@app.post("/api/demo/chain-break", tags=["demo"], status_code=201)
def demo_chain_break() -> dict:
    """One-click: build the hand-designed chain-break scenario and return it."""
    n, h, w, first, last, fwd, bwd = demo.chain_break_arrays()
    project = store.create_project(n, h, w, first, last, fwd, bwd)
    snap = project.current_snapshot()
    body = ser.frames_response(project, snap)
    body["walkthrough"] = {
        "inspect": f"/api/projects/{project.project_id}/frames",
        "visual_sheet": f"/api/projects/{project.project_id}/sheet.svg",
        "submit_completed_frame_1": {
            "method": "PUT",
            "url": f"/api/projects/{project.project_id}/frames/1/mask",
            "body": demo.chain_break_mask_payload(),
        },
        "old_version_after_submit": (
            f"/api/projects/{project.project_id}/frames?version=1"
        ),
    }
    return body


@app.post("/api/demo/supplement", tags=["demo"], status_code=201)
def demo_supplement() -> dict:
    """One-click: v1 -> submit frame 2 -> v2, returning both same-version views."""
    n, h, w, first, last, fwd, bwd = demo.supplement_arrays()
    project = store.create_project(n, h, w, first, last, fwd, bwd)
    before = project.current_snapshot()
    before_view = ser.frames_response(project, before)

    mask2 = eng.normalize_mask(
        demo.supplement_frame2_mask(), h, w, "frame2"
    )
    after = store.submit_mask(project.project_id, 2, mask2)
    after_view = ser.frames_response(project, after)

    return {
        "project_id": project.project_id,
        "submitted_frame": 2,
        "update_summary": ser.submit_response(
            project, before.version, after, 2
        ),
        "before_version_1": before_view,
        "after_version_2": after_view,
        "follow_up": {
            "submit_frame_3": {
                "method": "PUT",
                "url": f"/api/projects/{project.project_id}/frames/3/mask",
                "body": demo.supplement_frame3_payload(),
            },
            "query_old_version": (
                f"/api/projects/{project.project_id}/frames?version=1"
            ),
        },
    }
