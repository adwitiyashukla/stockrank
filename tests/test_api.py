"""API contract tests.

Skipped automatically when the optional API extras are absent, so the core suite
still runs in a minimal environment.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="install with: pip install -e '.[api]'")


@pytest.fixture(scope="module")
def client():
    starlette_testclient = pytest.importorskip(
        "starlette.testclient", reason="TestClient needs an http client backend"
    )
    from stockrank.api.main import app

    return starlette_testclient.TestClient(app)


def test_root_lists_the_service(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "StockRank"


def test_health_reports_status(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["runs_available"], list)


def test_unknown_run_returns_404(client):
    assert client.get("/runs/definitely-not-a-run/metrics").status_code == 404


def test_score_rejects_an_empty_batch(client):
    r = client.post("/score", json={"run": "baseline", "observations": []})
    assert r.status_code in (400, 404)


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/score" in schema["paths"]
    assert "/health" in schema["paths"]
