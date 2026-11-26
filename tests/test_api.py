"""End-to-end HTTP API tests via FastAPI TestClient."""

from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app import demo
from app.main import app

client = TestClient(app)


# ----------------------------------------------------------------------- basics
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_docs_and_openapi_present():
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower()
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/projects" in paths
    assert "/api/demo/chain-break" in paths
    post_examples = (
        paths["/api/projects"]["post"]["requestBody"]["content"]
        ["application/json"].get("examples", {})
    )
    assert "chain_break" in post_examples
    put_examples = (
        paths["/api/projects/{project_id}/frames/{frame_index}/mask"]["put"]
        ["requestBody"]["content"]["application/json"].get("examples", {})
    )
    assert "supplement_frame2" in put_examples


def test_root_landing():
    r = client.get("/")
    assert r.status_code == 200
    assert "/docs" in r.text


# ----------------------------------------------------------- chain break demo
def test_chain_break_demo_exact_statuses():
    r = client.post("/api/demo/chain-break")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["version"] == 1
    assert body["frame_count"] == 3

    f1 = body["frames"][1]
    status = f1["status"]
    expected = [
        ["right", "agree", "agree", "agree", "conflict"],
        ["agree", "broken", "agree", "left", "agree"],
        ["agree", "agree", "agree", "agree", "left"],
    ]
    assert status == expected

    needs = f1["needs_paint"]
    assert needs[0][4] == 1   # conflict
    assert needs[1][1] == 1   # broken
    assert sum(map(sum, needs)) == 2
    assert body["pending_pixels_total"] == 2
    assert body["status_counts"]["broken"] == 1
    assert body["status_counts"]["conflict"] == 1
    assert body["status_counts"]["left"] == 2
    assert body["status_counts"]["right"] == 1
    assert body["status_counts"]["agree"] == 10


def test_create_project_from_prefilled_example():
    payload = demo.chain_break_create_payload()
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    pid = body["project_id"]
    assert body["project"]["current_version"] == 1

    r = client.get(f"/api/projects/{pid}/frames")
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == 1
    # Every frame in the multi-frame response shares the same version.
    assert all(f["version"] == 1 for f in data["frames"])


# -------------------------------------------------------------- submit workflow
def test_submit_mask_versioning_and_old_version_query():
    pid = client.post("/api/projects",
                      json=demo.chain_break_create_payload()).json()["project_id"]

    # v1 has two pending pixels.
    v1 = client.get(f"/api/projects/{pid}/frames").json()
    assert v1["pending_pixels_total"] == 2

    # Submit completed frame 1.
    r = client.put(f"/api/projects/{pid}/frames/1/mask",
                   json=demo.chain_break_mask_payload())
    assert r.status_code == 200, r.text
    upd = r.json()
    assert upd["previous_version"] == 1
    assert upd["version"] == 2
    assert upd["recomputed_frames"] == []  # adjacent to both anchors

    # Current version: frame 1 fully confirmed, no pending anywhere.
    cur = client.get(f"/api/projects/{pid}/frames").json()
    assert cur["version"] == 2
    assert cur["pending_pixels_total"] == 0
    assert cur["frames"][1]["confirmed"] is True
    assert cur["frames"][1]["mask"] == demo.chain_break_completed_mask().tolist()

    # Old immutable version still served and still has the pending pixels.
    old = client.get(f"/api/projects/{pid}/frames?version=1").json()
    assert old["version"] == 1
    assert old["pending_pixels_total"] == 2
    assert all(f["version"] == 1 for f in old["frames"])

    # Single-frame endpoint honors version too.
    old_f1 = client.get(f"/api/projects/{pid}/frames/1?version=1").json()
    assert old_f1["status"][1][1] == "broken"
    new_f1 = client.get(f"/api/projects/{pid}/frames/1?version=2").json()
    assert new_f1["confirmed"] is True

    # Unknown version -> 404, unknown project -> 404.
    assert client.get(f"/api/projects/{pid}/frames?version=99").status_code == 404
    assert client.get("/api/projects/nope/frames").status_code == 404


def test_supplement_demo_endpoint_shows_partial_recompute():
    r = client.post("/api/demo/supplement")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["before_version_1"]["pending_pixels_per_frame"] == [0, 2, 2, 2, 0]
    assert body["after_version_2"]["pending_pixels_per_frame"] == [0, 0, 0, 2, 0]
    upd = body["update_summary"]
    assert upd["version"] == 2
    assert upd["recomputed_frames"] == [1, 3]


def test_supplement_manual_then_finish_frame3():
    pid = client.post("/api/projects",
                      json=demo.supplement_create_payload()).json()["project_id"]
    r = client.put(f"/api/projects/{pid}/frames/2/mask",
                   json=demo.supplement_frame2_payload())
    assert r.status_code == 200
    assert r.json()["recomputed_frames"] == [1, 3]

    r = client.put(f"/api/projects/{pid}/frames/3/mask",
                   json=demo.supplement_frame3_payload())
    assert r.status_code == 200
    # Frame 3 sits between confirmed 2 and 4, both adjacent.
    assert r.json()["recomputed_frames"] == []
    frames = client.get(f"/api/projects/{pid}/frames").json()
    assert frames["pending_pixels_total"] == 0
    assert frames["confirmed_frames"] == [0, 2, 3, 4]


# --------------------------------------------------------------- validation API
def test_reject_frame_count_out_of_range():
    payload = demo.chain_break_create_payload()
    payload["frame_count"] = 1
    assert client.post("/api/projects", json=payload).status_code == 422
    payload["frame_count"] = 41
    assert client.post("/api/projects", json=payload).status_code == 422


def test_reject_dimension_over_128():
    payload = demo.chain_break_create_payload()
    payload["height"] = 129
    assert client.post("/api/projects", json=payload).status_code == 422


def test_reject_wrong_flow_count():
    payload = demo.chain_break_create_payload()
    payload["flows"] = payload["flows"][:1]  # frame_count=3 needs 2
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 422
    assert "expected 2" in r.json()["detail"]


def test_reject_ragged_mask():
    payload = demo.chain_break_create_payload()
    payload["first_mask"] = [[0, 1, 1], [0, 1]]  # ragged / wrong width
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 422


def test_reject_nonbinary_mask():
    payload = demo.chain_break_create_payload()
    payload["first_mask"][0][0] = 2
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 422
    assert "only 0 and 1" in r.json()["detail"]


def test_reject_wrong_shape_flow():
    payload = demo.chain_break_create_payload()
    payload["flows"][0]["forward"] = [[[0, 0]]]  # valid nesting, shape (1,1,2)
    r = client.post("/api/projects", json=payload)
    assert r.status_code == 422
    assert "forward[0]" in r.json()["detail"]


def test_reject_nonexistent_frame_index_on_submit():
    pid = client.post("/api/projects",
                      json=demo.chain_break_create_payload()).json()["project_id"]
    r = client.put(f"/api/projects/{pid}/frames/9/mask",
                   json=demo.chain_break_mask_payload())
    assert r.status_code == 422


def test_invalid_update_keeps_version_and_confirmed():
    pid = client.post("/api/projects",
                      json=demo.chain_break_create_payload()).json()["project_id"]
    bad = {"mask": [[0, 1], [1, 0]]}  # wrong shape
    r = client.put(f"/api/projects/{pid}/frames/1/mask", json=bad)
    assert r.status_code == 422

    meta = client.get(f"/api/projects/{pid}").json()
    assert meta["current_version"] == 1
    assert meta["confirmed_frames"] == [0, 2]


def test_extra_fields_forbidden():
    payload = demo.chain_break_create_payload()
    payload["unexpected"] = 1
    assert client.post("/api/projects", json=payload).status_code == 422


# ---------------------------------------------------------------------- visuals
def test_svg_endpoints():
    pid = client.post("/api/projects",
                      json=demo.chain_break_create_payload()).json()["project_id"]
    r = client.get(f"/api/projects/{pid}/frames/1/svg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert r.text.lstrip().startswith("<svg")
    assert "broken" in r.text

    r = client.get(f"/api/projects/{pid}/sheet.svg")
    assert r.status_code == 200
    assert "<svg" in r.text
