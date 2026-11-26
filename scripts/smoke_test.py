#!/usr/bin/env python3
"""End-to-end smoke test over real HTTP using only the standard library.

Usage:
    python scripts/smoke_test.py [BASE_URL]

Default BASE_URL=http://localhost:8000 (docker compose published port).
Exits non-zero if any assertion fails.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
PASS = 0


def call(method: str, path: str, body=None, expect: int = 200):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = resp.status
            raw = resp.read()
            ctype = resp.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        code = exc.code
        raw = exc.read()
        ctype = exc.headers.get("content-type", "")
    if code != expect:
        raise AssertionError(f"{method} {path}: expected {expect}, got {code}: "
                             f"{raw[:400].decode(errors='replace')}")
    if "json" in ctype:
        return code, json.loads(raw)
    return code, raw


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS
    if not cond:
        raise AssertionError(f"FAIL: {name} {extra}")
    PASS += 1
    print(f"  ok  {name}")


def wait_health() -> None:
    import time
    for _ in range(30):
        try:
            _, h = call("GET", "/health")
            if h.get("status") == "ok":
                print(f"health ok at {BASE}")
                return
        except Exception:
            time.sleep(1)
    raise SystemExit("server did not become healthy")


def main() -> None:
    wait_health()

    # /docs and schema -------------------------------------------------------
    _, docs = call("GET", "/docs")
    check("/docs serves Swagger UI", b"swagger" in docs.lower())
    check("/docs uses vendored local assets",
          b"/static/swagger-ui-bundle.js" in docs)
    _, js = call("GET", "/static/swagger-ui-bundle.js")
    check("vendored swagger bundle reachable",
          isinstance(js, bytes) and len(js) > 1_000_000
          and b"webpack" in js[:400])
    _, oa = call("GET", "/openapi.json")
    check("openapi lists demo endpoints",
          "/api/demo/chain-break" in oa["paths"]
          and "/api/demo/supplement" in oa["paths"])

    # Chain break scenario ---------------------------------------------------
    _, demo = call("POST", "/api/demo/chain-break", expect=201)
    pid = demo["project_id"]
    f1 = demo["frames"][1]
    check("chain-break status matrix exact", f1["status"] == [
        ["right", "agree", "agree", "agree", "conflict"],
        ["agree", "broken", "agree", "left", "agree"],
        ["agree", "agree", "agree", "agree", "left"],
    ])
    check("two pending pixels (broken+conflict)",
          demo["pending_pixels_total"] == 2,
          str(demo["pending_pixels_total"]))
    check("every frame tagged version 1",
          all(f["version"] == 1 for f in demo["frames"]))

    _, frames = call("GET", f"/api/projects/{pid}/frames")
    check("GET frames same version", frames["version"] == 1
          and all(f["version"] == 1 for f in frames["frames"]))

    _, svg = call("GET", f"/api/projects/{pid}/frames/1/svg")
    check("frame SVG rendered", svg.lstrip().startswith(b"<svg"))
    _, sheet = call("GET", f"/api/projects/{pid}/sheet.svg")
    check("sheet SVG rendered", sheet.lstrip().startswith(b"<svg"))

    # Invalid submission is rejected and keeps v1 ----------------------------
    call("PUT", f"/api/projects/{pid}/frames/1/mask",
         {"mask": [[9, 9], [9, 9]]}, expect=422)
    check("invalid mask rejected with 422", True)
    _, meta = call("GET", f"/api/projects/{pid}")
    check("version still 1 after invalid submit",
          meta["current_version"] == 1)

    # Submit completed frame 1 ----------------------------------------------
    completed = {
        "mask": [
            [0, 1, 1, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    }
    _, upd = call("PUT", f"/api/projects/{pid}/frames/1/mask", completed)
    check("submit creates version 2", upd["version"] == 2
          and upd["previous_version"] == 1)
    check("adjacent anchors -> no interior recompute",
          upd["recomputed_frames"] == [])

    _, cur = call("GET", f"/api/projects/{pid}/frames")
    check("current version 2 has zero pending",
          cur["version"] == 2 and cur["pending_pixels_total"] == 0)
    check("all v2 frames internally same version",
          all(f["version"] == 2 for f in cur["frames"]))

    _, old = call("GET", f"/api/projects/{pid}/frames?version=1")
    check("old version 1 still queryable with its 2 pending pixels",
          old["version"] == 1 and old["pending_pixels_total"] == 2
          and all(f["version"] == 1 for f in old["frames"]))

    _, missing = call("GET", f"/api/projects/{pid}/frames?version=99",
                      expect=404)
    check("unknown version -> 404", True)
    call("GET", "/api/projects/does-not-exist", expect=404)
    check("unknown project -> 404", True)

    # Supplement / partial recompute scenario --------------------------------
    _, sup = call("POST", "/api/demo/supplement", expect=201)
    check("supplement before: [0,2,2,2,0]",
          sup["before_version_1"]["pending_pixels_per_frame"] == [0, 2, 2, 2, 0])
    check("supplement after:  [0,0,0,2,0]",
          sup["after_version_2"]["pending_pixels_per_frame"] == [0, 0, 0, 2, 0])
    check("only frames 1 and 3 recomputed",
          sup["update_summary"]["recomputed_frames"] == [1, 3])
    pid2 = sup["project_id"]

    _, f3upd = call("PUT", f"/api/projects/{pid2}/frames/3/mask",
                    {"mask": [
                        [0, 1, 1, 0, 0],
                        [0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1],
                    ]})
    check("frame 3 submission -> version 3", f3upd["version"] == 3)
    _, final = call("GET", f"/api/projects/{pid2}/frames")
    check("all pending resolved after frame 3",
          final["pending_pixels_total"] == 0)
    check("confirmed set [0,2,3,4]",
          final["confirmed_frames"] == [0, 2, 3, 4])

    # Validation bounds -------------------------------------------------------
    bad = demo_payload()
    bad["frame_count"] = 41
    call("POST", "/api/projects", bad, expect=422)
    check("frame_count 41 rejected", True)
    bad = demo_payload()
    bad["height"] = 129
    call("POST", "/api/projects", bad, expect=422)
    check("height 129 rejected", True)

    print(f"\nALL SMOKE CHECKS PASSED ({PASS} assertions) against {BASE}")


def demo_payload():
    # Minimal valid payload built locally (keeps script independent of app pkg).
    z35 = [[0] * 5 for _ in range(3)]
    block = [
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    flow = {"forward": [[[0, 0]] * 5 for _ in range(3)],
            "backward": [[[0, 0]] * 5 for _ in range(3)]}
    return {
        "frame_count": 3, "height": 3, "width": 5,
        "first_mask": block, "last_mask": z35,
        "flows": [flow, json.loads(json.dumps(flow))],
    }


if __name__ == "__main__":
    main()
