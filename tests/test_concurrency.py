"""Real-socket concurrency test: reads stay on the old version during recompute."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from app import demo
from app.main import app
from app.store import store


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("uvicorn failed to start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_old_version_queryable_during_recompute(server_url):
    # Create a project over the real HTTP server.
    r = httpx.post(f"{server_url}/api/projects",
                   json=demo.supplement_create_payload(), timeout=10)
    assert r.status_code == 201, r.text
    pid = r.json()["project_id"]

    entered = threading.Event()
    release = threading.Event()

    def blocking_barrier(project_id: str) -> None:
        if project_id != pid:
            return
        entered.set()
        assert release.wait(timeout=10), "test timed out waiting to release"

    import app.store as store_mod
    original_hook = store_mod.recompute_barrier
    store_mod.recompute_barrier = blocking_barrier

    put_result: dict = {}

    def worker_put():
        try:
            resp = httpx.put(
                f"{server_url}/api/projects/{pid}/frames/2/mask",
                json=demo.supplement_frame2_payload(), timeout=15,
            )
            put_result["status"] = resp.status_code
            put_result["body"] = resp.json()
        except Exception as exc:  # pragma: no cover - failure reporting
            put_result["error"] = repr(exc)
        finally:
            release.set()

    worker = threading.Thread(target=worker_put, daemon=True)
    try:
        worker.start()
        assert entered.wait(timeout=10), "recompute barrier was not reached"

        # Recompute is now in flight (writer lock held, new version not
        # published). Reads must immediately return the complete old version.
        t0 = time.perf_counter()
        resp = httpx.get(f"{server_url}/api/projects/{pid}/frames", timeout=5)
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == 1
        assert body["is_current_version"] is True
        assert body["pending_pixels_per_frame"] == [0, 2, 2, 2, 0]
        assert all(f["version"] == 1 for f in body["frames"])
        assert elapsed < 2.0, "reader appears blocked by recomputation"

        # Explicit historical query agrees.
        resp2 = httpx.get(
            f"{server_url}/api/projects/{pid}/frames?version=1", timeout=5
        )
        assert resp2.json()["version"] == 1
    finally:
        release.set()
        store_mod.recompute_barrier = original_hook
        worker.join(timeout=15)

    assert "error" not in put_result, put_result.get("error")
    assert put_result["status"] == 200, put_result
    assert put_result["body"]["version"] == 2

    # After publication, default reads move to the new same-version snapshot.
    resp = httpx.get(f"{server_url}/api/projects/{pid}/frames", timeout=5)
    body = resp.json()
    assert body["version"] == 2
    assert body["pending_pixels_per_frame"] == [0, 0, 0, 2, 0]
    assert all(f["version"] == 2 for f in body["frames"])

    # Old version remains explicitly addressable and internally consistent.
    old = httpx.get(
        f"{server_url}/api/projects/{pid}/frames?version=1", timeout=5
    ).json()
    assert old["version"] == 1
    assert all(f["version"] == 1 for f in old["frames"])


def test_failed_recompute_over_http_keeps_state(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app as asgi_app

    tc = TestClient(asgi_app)
    pid = tc.post("/api/projects",
                  json=demo.chain_break_create_payload()).json()["project_id"]

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("app.propagation.propagate_interval", boom)
    r = tc.put(f"/api/projects/{pid}/frames/1/mask",
               json=demo.chain_break_mask_payload())
    assert r.status_code == 500

    meta = tc.get(f"/api/projects/{pid}").json()
    assert meta["current_version"] == 1
    assert meta["confirmed_frames"] == [0, 2]
    frames = tc.get(f"/api/projects/{pid}/frames").json()
    assert frames["version"] == 1
    assert frames["pending_pixels_total"] == 2
