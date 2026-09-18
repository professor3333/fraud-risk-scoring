from __future__ import annotations

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from fraud.serve.app import create_app
from fraud.serve.rate_limit import RateLimitConfig, RateLimiter, client_key


def test_sliding_window_expiry_and_independent_quotas() -> None:
    now = [0.0]
    limiter = RateLimiter(RateLimitConfig(predict=2, bulk=1, explain=1), lambda: now[0])
    assert limiter.retry_after("a", "/predict", "POST") == 0
    now[0] = 10
    assert limiter.retry_after("a", "/predict", "POST") == 0
    assert limiter.retry_after("a", "/predict", "POST") == 50
    assert limiter.retry_after("b", "/predict", "POST") == 0
    assert limiter.retry_after("a", "/predict/csv", "POST") == 0
    assert limiter.retry_after("a", "/predict/batch", "POST") == 60
    assert limiter.retry_after("a", "/explain", "POST") == 0
    now[0] = 59.1
    assert limiter.retry_after("a", "/predict/", "POST") == 1
    now[0] = 60
    assert limiter.retry_after("a", "/predict", "POST") == 0
    assert limiter.retry_after("a", "/predict", "POST") == 10
    now[0] = 70
    assert limiter.retry_after("a", "/predict", "POST") == 0


def test_bucket_cap_preserves_active_quotas_and_recovers() -> None:
    now = [0.0]
    limiter = RateLimiter(RateLimitConfig(max_buckets=2, predict=1), lambda: now[0])
    assert limiter.retry_after("a", "/predict", "POST") == 0
    now[0] = 1
    assert limiter.retry_after("b", "/predict", "POST") == 0
    for i in range(100):
        assert limiter.retry_after(str(i), "/predict", "POST") == 59
    assert limiter.retry_after("a", "/predict", "POST") == 59
    now[0] = 60
    assert limiter.retry_after("new", "/predict", "POST") == 0
    assert limiter.retry_after("b", "/predict", "POST") == 1


def test_concurrent_admission_is_atomic() -> None:
    limiter = RateLimiter(RateLimitConfig(predict=3), lambda: 0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: limiter.retry_after("a", "/predict", "POST"), range(40)))
    assert results.count(0) == 3
    assert results.count(60) == 37


def test_disabled_and_unmetered_routes() -> None:
    off = RateLimiter(RateLimitConfig(enabled=False, predict=1))
    on = RateLimiter(RateLimitConfig(predict=1))
    for _ in range(3):
        assert off.retry_after("a", "/predict", "POST") == 0
        for path in ("/health", "/model-info", "/", "/single", "/outcomes", "/audit/recent"):
            assert on.retry_after("a", path, "GET") == 0
        assert on.retry_after("a", "/predict", "OPTIONS") == 0


@pytest.mark.parametrize(
    "config",
    [
        {"bulk": 0},
        {"predict": -1},
        {"enabled": "false"},
        {"max_buckets": 0},
        {"window_seconds": 0},
        {"typo": 1},
    ],
)
def test_invalid_config_fails_closed(config: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RateLimitConfig.model_validate(config)


def test_client_identity_trust_boundary() -> None:
    def request(header: str) -> Request:
        return Request(
            {
                "type": "http",
                "client": ("127.0.0.1", 1234),
                "headers": [
                    (b"cf-connecting-ip", header.encode()),
                    (b"x-forwarded-for", b"198.51.100.99"),
                ],
            }
        )

    assert client_key(request("203.0.113.1"), render=False) == "127.0.0.1"
    assert client_key(request("203.0.113.1"), render=True) == "203.0.113.1"
    assert client_key(request("2001:db8:0:0::1"), render=True) == "2001:db8::1"
    for bad in ("", "bad", "203.0.113.1, 203.0.113.2"):
        assert client_key(request(bad), render=True) == "render-unknown"


@pytest.fixture
def limited_config(served: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("FRAUD_API_KEY", raising=False)
    monkeypatch.delenv("FRAUD_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    source = Path(served["config"])
    raw = yaml.safe_load(source.read_text())
    raw["model_path"] = str(source.parents[1] / raw["model_path"])
    raw["audit_db"] = str(tmp_path / "audit.sqlite")
    raw["rate_limits"] = {"predict": 1, "bulk": 1, "explain": 1}
    cfg = tmp_path / "serving.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    return cfg


@pytest.mark.parametrize("path", ["/predict", "/predict/csv", "/predict/batch", "/explain"])
def test_rejection_precedes_body_and_model_and_is_recorded(
    limited_config: Path,
    path: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(limited_config)
    with TestClient(app) as client:
        limiter = app.state.serving.rate_limiter
        monkeypatch.setattr(limiter, "clock", lambda: 0)
        # Invalid requests consume quota too; none of these should reach inference.
        assert client.post(path, content=b"{").status_code == 422

        async def unread_body():
            raise AssertionError("rate-limited request body was read")
            yield b""  # async generator: fail on consumption, not construction

        from starlette.requests import Request as StarletteRequest

        monkeypatch.setattr(StarletteRequest, "stream", lambda _: unread_body())
        logger = logging.getLogger("fraud.serve.requests")
        monkeypatch.setattr(logger, "handlers", [caplog.handler])
        monkeypatch.setattr(logger, "propagate", False)
        with caplog.at_level(logging.INFO, logger="fraud.serve.requests"):
            response = client.post(path, content=b"not parsed", headers={"X-Request-ID": "quota"})
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"
        assert response.headers["X-Request-ID"] == "quota"
        assert "rate limit" in response.json()["detail"]
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/audit/recent").status_code == 403
    with sqlite3.connect(limited_config.parent / "audit.sqlite") as db:
        assert db.execute("SELECT count(*) FROM prediction_events").fetchone() == (0,)
        assert db.execute(
            "SELECT status_code, n_rows FROM requests WHERE request_id = 'quota'"
        ).fetchone() == (429, 0)
    lines = [json.loads(r.message) for r in caplog.records if r.name == "fraud.serve.requests"]
    assert len([r for r in lines if r["request_id"] == "quota" and r["status"] == 429]) == 1


def test_render_clients_are_separate_and_forwarded_spoofing_does_not_reset_quota(
    limited_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RENDER", "true")
    with TestClient(create_app(limited_config)) as client:
        for address in ("203.0.113.1", "203.0.113.2"):
            headers = {"CF-Connecting-IP": address}
            assert client.post("/predict", json={}, headers=headers).status_code == 422
            headers["X-Forwarded-For"] = "198.51.100.1"
            assert client.post("/predict", json={}, headers=headers).status_code == 429


def test_auth_precedes_quota_and_keys_do_not_bypass_it(
    limited_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRAUD_API_KEY", "score-key")
    with TestClient(create_app(limited_config)) as client:
        assert client.post("/predict", json={}).status_code == 401
        headers = {"X-API-Key": "score-key"}
        assert client.post("/predict", json={}, headers=headers).status_code == 422
        assert client.post("/predict", json={}, headers=headers).status_code == 429
        assert client.post("/predict", json={}).status_code == 401
