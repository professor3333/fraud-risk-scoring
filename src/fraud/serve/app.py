"""FastAPI inference service: the calibrated pipeline, loaded once, scoring raw rows (G8)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
import math
import os
import re
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from fraud.data import schema
from fraud.evaluate.explain import explain
from fraud.evaluate.policy import apply_daily_rank_policy
from fraud.pipeline.calibrated import CalibratedModel
from fraud.serve.audit import AuditLog, PredictionEvent, new_request_id, utc_now
from fraud.serve.frames import (
    monitored_fields,
    payloads_to_frame,
    request_to_frame,
    table_to_frame,
)
from fraud.serve.parity import frozen_paths, read_manifest, verify
from fraud.serve.rate_limit import RateLimitConfig, RateLimiter, client_key
from fraud.serve.schemas import (
    Action,
    AuditEvent,
    Bands,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CsvPredictionResponse,
    CsvSummary,
    ExplanationResponse,
    HealthResponse,
    ModelInfoResponse,
    OutcomesRequest,
    OutcomesResponse,
    Policy,
    PolicyApplied,
    PredictionResponse,
    RankedPrediction,
    RiskLevel,
    ScoredRow,
    TransactionRequest,
)

MAX_CSV_ROWS = 5_000
SECONDS_PER_DAY = 86_400
# Endpoints that score, explain, write labels or read the audit trail: recorded in the
# requests table, logged as one JSON line, and bounded in time. Scoring and explanation
# need X-API-Key when FRAUD_API_KEY is set; the admin endpoints (labels in, audit rows out)
# need FRAUD_ADMIN_API_KEY and are refused outright while it is unset, so an anonymous
# public demo never exposes them.
GUARDED_PREFIXES = ("/predict", "/explain", "/outcomes", "/audit")
ADMIN_PREFIXES = ("/outcomes", "/audit")
request_log = logging.getLogger("fraud.serve.requests")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # serving.yaml max_upload_bytes overrides

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_CONFIG = ROOT / "configs" / "serving.yaml"


@dataclass(frozen=True)
class ServingState:
    model: CalibratedModel
    bands: Bands
    model_version: str
    parity_rows: int
    info: dict[str, Any]
    default_review_budget: int
    audit: AuditLog | None
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    api_key: str | None = None  # FRAUD_API_KEY; None leaves the scoring endpoints open
    admin_api_key: str | None = None  # FRAUD_ADMIN_API_KEY; None disables the admin endpoints
    request_timeout_s: float = 60.0  # serving.yaml request_timeout_s
    rate_limiter: RateLimiter | None = None
    render_client_ip: bool = False


CHAMPION_FILES = (
    "model.joblib",
    "model_frozen_sample.json",
    "model_frozen_expected.json",
    "model_manifest.json",
    "model_monitor_reference.json",
)


#: ``scripts/publish_champion.py`` tags one immutable release per champion
#: ``champion-<sha12>``, so a release URL carries the artifact's own digest.
CHAMPION_TAG_DIGEST = re.compile(r"champion-([0-9a-fA-F]{8,64})/?$")


def expected_champion_digest(base_url: str, pin: str | None) -> str:
    """The sha256 (or leading hex of it) ``model.joblib`` must have, from outside the store.

    ``model.joblib`` is a pickle: ``joblib.load`` executes whatever it contains, so it
    must not be opened before it is known to be the intended artifact. The manifest and
    the frozen golden cannot establish that — they are fetched from the same base URL as
    the artifact, so anything able to serve a malicious ``model.joblib`` can serve the
    matching digest with it. The anchor has to come from somewhere the store does not
    control: FRAUD_CHAMPION_SHA256, or the ``champion-<sha12>`` release tag in the URL,
    both of which are deployer configuration (render.yaml, docs/deployment.md).

    Raises ``RuntimeError`` when neither is present: an unpinned remote fetch is exactly
    the case this check exists to refuse.
    """
    if pin:
        candidate = pin.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{8,64}", candidate):
            raise RuntimeError(
                "FRAUD_CHAMPION_SHA256 must be at least 8 hex characters of the "
                f"artifact's sha256, got {pin!r}"
            )
        return candidate
    found = CHAMPION_TAG_DIGEST.search(base_url.rstrip("/"))
    if found is None:
        raise RuntimeError(
            f"refusing to fetch an unpinned champion from {base_url}: model.joblib is a "
            "pickle and is executed when it is loaded, so its digest must be known before "
            "it is opened. Point FRAUD_CHAMPION_URL at a champion-<sha12> release "
            "(scripts/publish_champion.py) or set FRAUD_CHAMPION_SHA256."
        )
    return found.group(1).lower()


def fetch_champion(
    model_path: Path, base_url: str, token: str | None, expected_digest: str
) -> list[str]:
    """Populate the champion directory from an artifact store at startup.

    For hosts that build from the repository (Render, Koyeb — the image carries
    no weights): FRAUD_CHAMPION_URL names a base URL under which the champion
    files are served — a GitHub release's ``…/releases/download/<tag>`` (public,
    no token; scripts/publish_champion.py) or any store that takes a bearer
    token in FRAUD_CHAMPION_TOKEN. The files land next to ``model_path`` and the
    startup parity check then treats them exactly like a local champion.
    Optional files (the monitoring reference) are skipped when the store lacks
    them; everything else must exist.

    ``model.joblib`` is checked against ``expected_digest`` in memory and is written
    only if it matches, so a substituted artifact is never handed to ``joblib.load``.
    The parity check cannot do this job: it runs after the pickle has been executed.
    """
    import urllib.error
    import urllib.request

    model_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    fetched: list[str] = []
    for name in CHAMPION_FILES:
        target = model_path.parent / name
        if target.exists():
            continue
        req = urllib.request.Request(f"{base_url.rstrip('/')}/{name}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body: bytes = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and name == "model_monitor_reference.json":
                continue
            raise RuntimeError(f"could not fetch {name} from {base_url}: HTTP {exc.code}") from exc
        if name == "model.joblib":
            digest = hashlib.sha256(body).hexdigest()
            if not hmac.compare_digest(digest[: len(expected_digest)], expected_digest):
                raise RuntimeError(
                    f"refusing to load model.joblib from {base_url}: sha256 {digest[:12]} "
                    f"is not the pinned {expected_digest[:12]}. The artifact was not "
                    "written to disk and was not deserialized."
                )
        target.write_bytes(body)
        fetched.append(name)
    return fetched


def load_state(config_path: Path = DEFAULT_CONFIG) -> ServingState:
    raw = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parents[1]
    model_path = root / raw["model_path"]
    champion_url = os.environ.get("FRAUD_CHAMPION_URL")
    if not model_path.exists() and champion_url:
        expected_digest = expected_champion_digest(
            champion_url, os.environ.get("FRAUD_CHAMPION_SHA256")
        )
        fetched = fetch_champion(
            model_path, champion_url, os.environ.get("FRAUD_CHAMPION_TOKEN"), expected_digest
        )
        request_log.info(
            json.dumps(
                {"event": "champion_fetched", "files": fetched, "pinned": expected_digest[:12]}
            )
        )
    model = joblib.load(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    # A promoted champion carries a manifest (scripts/promote.py): its version, facts and
    # policy bands come from there, so a promotion never edits this config.
    manifest = read_manifest(model_path)
    if manifest is not None:
        if manifest["artifact_sha256"][:12] != digest:
            raise RuntimeError(
                f"{model_path.name} ({digest}) is not the artifact its manifest describes "
                f"({manifest['artifact_sha256'][:12]}); promote again or restore the champion"
            )
        raw["model_version"] = manifest["model_version"]
        raw["model_info"] = {**raw.get("model_info", {}), **manifest["model_info"]}
        raw["bands"] = manifest["bands"]
    bands = Bands(**raw["bands"])
    if not 0.0 <= bands.review <= bands.block <= 1.0:
        raise ValueError(f"bands must satisfy 0 <= review <= block <= 1, got {bands}")
    # Refuse to serve an artifact that does not reproduce its frozen probabilities (G8).
    if raw.get("require_parity", True):
        parity = verify(model, model_path)
        parity_rows = parity.n_rows
    else:
        parity_rows = 0 if not all(p.exists() for p in frozen_paths(model_path)) else -1
    # Prediction audit trail: serving.yaml `audit_db` (relative to the repo root), overridden
    # by FRAUD_AUDIT_DB; an explicit null disables it.
    audit_path = os.environ.get("FRAUD_AUDIT_DB", raw.get("audit_db"))
    audit = (
        AuditLog(Path(audit_path) if Path(audit_path).is_absolute() else root / audit_path)
        if audit_path
        else None
    )
    return ServingState(
        model=model,
        bands=bands,
        model_version=f"{raw['model_version']}@{digest}",
        parity_rows=parity_rows,
        info=dict(raw.get("model_info", {})),
        default_review_budget=int(raw.get("default_review_budget", 200)),
        audit=audit,
        max_upload_bytes=int(raw.get("max_upload_bytes", MAX_UPLOAD_BYTES)),
        api_key=os.environ.get("FRAUD_API_KEY") or None,
        admin_api_key=os.environ.get("FRAUD_ADMIN_API_KEY") or None,
        request_timeout_s=float(raw.get("request_timeout_s", 60.0)),
        rate_limiter=RateLimiter(RateLimitConfig.model_validate(raw.get("rate_limits", {}))),
        render_client_ip=os.environ.get("RENDER") == "true",
    )


def build_commit() -> str | None:
    """The commit this build came from, when the host says which.

    Render injects RENDER_GIT_COMMIT; other hosts can set FRAUD_BUILD_COMMIT. The
    release workflow pins the deploy to a commit and then asserts /health reports it,
    which the served version cannot do on its own: every commit between one release
    bump and the next carries the same version.
    """
    commit = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("FRAUD_BUILD_COMMIT")
    return commit.strip() or None if commit else None


def classify(p: float, bands: Bands) -> tuple[RiskLevel, Action]:
    if p >= bands.block:
        return "high", "block"
    if p >= bands.review:
        return "medium", "review"
    return "low", "approve"


LEVEL_FOR_ACTION: dict[Action, RiskLevel] = {"block": "high", "review": "medium", "approve": "low"}


def _response(
    state: ServingState, transaction_id: int, p: float, action: Action | None = None
) -> PredictionResponse:
    if action is None:
        level, action = classify(p, state.bands)
    else:
        level = LEVEL_FOR_ACTION[action]
    return PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=p,
        risk_level=level,
        action=action,
        model_version=state.model_version,
    )


def review_capacity(
    state: ServingState,
    budget: int,
    transaction_dt: np.ndarray,
    transaction_ids: np.ndarray,
) -> dict[int, int]:
    """Reviews this request may still issue per transaction day.

    The budget is per day, not per request, and it is released through the day:
    by the time of a day's latest transaction in this request, the day has made
    ``ceil(budget * fraction of the day elapsed)`` reviews available, so a morning
    batch cannot take the whole day's queue. Transactions of the day already sent
    to review by earlier requests are charged against it, except the ones in this
    request, whose new decision replaces the old one. Without an audit trail there
    is no memory of earlier requests.
    """
    ids = {int(t) for t in transaction_ids}
    days = transaction_dt // SECONDS_PER_DAY
    capacity: dict[int, int] = {}
    for day in np.unique(days):
        latest = int(transaction_dt[days == day].max())
        elapsed = (latest - int(day) * SECONDS_PER_DAY + 1) / SECONDS_PER_DAY
        released = int(math.ceil(budget * elapsed))
        spent = 0
        if state.audit is not None:
            spent = len(state.audit.reviewed_transactions(int(day)) - ids)
        capacity[int(day)] = max(released - spent, 0)
    return capacity


def assign_actions(
    state: ServingState,
    probabilities: np.ndarray,
    policy: Policy,
    review_budget: int | None,
    frame: pd.DataFrame | None = None,
) -> tuple[list[Action], PolicyApplied]:
    """Actions for a scored batch: rank-based (default) or fixed-threshold bands."""
    if policy == "rank":
        if frame is None:
            raise ValueError("the rank policy needs the scored frame (transaction days and ids)")
        budget = state.default_review_budget if review_budget is None else review_budget
        dt = frame[schema.TIME_COL].to_numpy().astype(int)
        days = dt // SECONDS_PER_DAY
        capacity = review_capacity(state, budget, dt, frame[schema.ID_COL].to_numpy())
        actions, cutoff = apply_daily_rank_policy(probabilities, days, state.bands.block, capacity)
        applied = PolicyApplied(
            policy="rank",
            block_threshold=state.bands.block,
            review_budget=budget,
            review_threshold=None,
            review_cutoff=cutoff,
            review_capacity=int(sum(capacity.values())),
            budget_accounting="audit_trail" if state.audit is not None else "per_request",
        )
        return [str(a) for a in actions], applied  # type: ignore[misc]
    actions_t: list[Action] = [classify(float(p), state.bands)[1] for p in probabilities]
    reviewed = [float(p) for p, a in zip(probabilities, actions_t, strict=True) if a == "review"]
    applied = PolicyApplied(
        policy="threshold",
        block_threshold=state.bands.block,
        review_budget=None,
        review_threshold=state.bands.review,
        review_cutoff=min(reviewed) if reviewed else None,
    )
    return actions_t, applied


def score(state: ServingState, payload: dict[str, Any]) -> PredictionResponse:
    frame = request_to_frame(payload)
    p = float(state.model.predict_proba(frame)[0, 1])
    return _response(state, int(payload[schema.ID_COL]), p)


def score_batch(
    state: ServingState,
    payloads: list[dict[str, Any]],
    policy: Policy = "rank",
    review_budget: int | None = None,
    frame: pd.DataFrame | None = None,
) -> BatchPredictionResponse:
    """Score many rows in one pass, apply the policy, return them ranked, highest risk first."""
    if frame is None:
        frame = payloads_to_frame(payloads)
    probabilities = np.asarray(state.model.predict_proba(frame)[:, 1])
    actions, applied = assign_actions(state, probabilities, policy, review_budget, frame)
    order = np.argsort(-probabilities, kind="stable")
    ranked = []
    for rank, i in enumerate(order.tolist(), start=1):
        single = _response(
            state, int(payloads[i][schema.ID_COL]), float(probabilities[i]), actions[i]
        )
        ranked.append(RankedPrediction(rank=rank, **single.model_dump()))
    counts: dict[Action, int] = {"approve": 0, "review": 0, "block": 0}
    for r in ranked:
        counts[r.action] += 1
    return BatchPredictionResponse(n=len(ranked), ranked=ranked, counts=counts, policy=applied)


def score_table(
    state: ServingState,
    table: pd.DataFrame,
    policy: Policy = "rank",
    review_budget: int | None = None,
) -> CsvPredictionResponse:
    """Score an uploaded table; apply the policy; rank rows and summarise for the analyst view."""
    frame, ignored = table_to_frame(table)
    probabilities = np.asarray(state.model.predict_proba(frame)[:, 1], dtype=float)
    actions, applied = assign_actions(state, probabilities, policy, review_budget, frame)
    order = np.argsort(-probabilities, kind="stable")
    rows: list[ScoredRow] = []
    counts = {"approve": 0, "review": 0, "block": 0}
    for rank, i in enumerate(order.tolist(), start=1):
        p = float(probabilities[i])
        action = actions[i]
        level = LEVEL_FOR_ACTION[action]
        counts[action] += 1
        record = frame.iloc[i]
        details = {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in record.items()
            if not (isinstance(v, float) and np.isnan(v)) and v is not None and v == v
        }
        card_type = record["card6"]
        rows.append(
            ScoredRow(
                rank=rank,
                transaction_id=int(record[schema.ID_COL]),
                fraud_probability=p,
                risk_level=level,
                action=action,
                amount=float(record["TransactionAmt"]),
                product=str(record["ProductCD"]),
                card_type=None if pd.isna(card_type) else str(card_type),
                has_identity=bool(record[schema.HAS_IDENTITY_COL]),
                details=details,
            )
        )
    summary = CsvSummary(
        analysed=len(rows),
        flagged=counts["review"] + counts["block"],
        high_risk=counts["block"],
        review=counts["review"],
        approve=counts["approve"],
        average_fraud_probability=float(probabilities.mean()) if len(rows) else 0.0,
        ignored_columns=ignored,
        model_version=state.model_version,
        policy=applied,
    )
    return CsvPredictionResponse(summary=summary, rows=rows)


def record_events(
    state: ServingState,
    endpoint: str,
    request_id: str,
    applied: PolicyApplied,
    rows: Sequence[tuple[int, float, str, str]],
    latency_ms: float,
    frame: pd.DataFrame | None = None,
) -> None:
    """Persist one event per scored transaction, plus the monitored input snapshot."""
    if state.audit is None:
        return
    now = utc_now()
    when: dict[int, int] = {}
    if frame is not None:
        state.audit.record_inputs(
            [{"request_id": request_id, "scored_at": now, **f} for f in monitored_fields(frame)]
        )
        when = dict(
            zip(frame[schema.ID_COL].astype(int), frame[schema.TIME_COL].astype(int), strict=True)
        )
    state.audit.record(
        [
            PredictionEvent(
                request_id=request_id,
                endpoint=endpoint,
                scored_at=now,
                transaction_id=tid,
                model_version=state.model_version,
                fraud_probability=p,
                risk_level=level,
                action=action,
                policy=applied.policy,
                block_threshold=applied.block_threshold,
                review_threshold=applied.review_threshold,
                review_budget=applied.review_budget,
                review_cutoff=applied.review_cutoff,
                batch_size=len(rows),
                latency_ms=latency_ms,
                transaction_dt=when.get(tid),
            )
            for tid, p, level, action in rows
        ]
    )


def threshold_policy_applied(state: ServingState) -> PolicyApplied:
    return PolicyApplied(
        policy="threshold",
        block_threshold=state.bands.block,
        review_budget=None,
        review_threshold=state.bands.review,
        review_cutoff=None,
    )


async def read_upload(request: Request, file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload without ever holding more than ``max_bytes`` of it.

    The declared Content-Length is checked first (cheap, covers honest clients);
    the stream is then read in chunks and abandoned the moment it exceeds the cap,
    so a dishonest or absent length cannot make the service buffer an arbitrary body.
    """
    declared = request.headers.get("content-length")
    too_large = HTTPException(413, f"upload larger than {max_bytes} bytes")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise too_large
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(1 << 20):
        size += len(chunk)
        if size > max_bytes:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


def _authorize(request: Request, state: ServingState) -> Response | None:
    """The response that refuses this call, or None when it may proceed.

    Admin routes (/outcomes, /audit/*) answer only to FRAUD_ADMIN_API_KEY and are
    closed while it is unset; scoring routes answer to FRAUD_API_KEY and are open
    while it is unset. The two keys never substitute for each other.
    """
    admin = request.url.path.startswith(ADMIN_PREFIXES)
    if admin and state.admin_api_key is None:
        return Response(
            json.dumps({"detail": "admin endpoints are disabled: set FRAUD_ADMIN_API_KEY"}),
            status_code=403,
            media_type="application/json",
        )
    expected = state.admin_api_key if admin else state.api_key
    if expected is None:
        return None
    given = request.headers.get("x-api-key", "")
    if hmac.compare_digest(given.encode(), expected.encode()):
        return None
    return Response(
        json.dumps({"detail": "missing or invalid X-API-Key"}),
        status_code=401,
        media_type="application/json",
    )


def _finish(
    request: Request,
    response: Response,
    rid: str,
    started: str,
    t0: float,
    state: ServingState | None,
) -> Response:
    """Stamp the request id, store the call, log one JSON line."""
    rid = response.headers.get("X-Request-ID", rid)
    latency = (time.perf_counter() - t0) * 1000
    rows = int(response.headers.get("X-Rows", "0"))
    if state is not None and state.audit is not None:
        state.audit.record_request(
            rid, request.url.path, started, response.status_code, latency, rows
        )
    response.headers.setdefault("X-Request-ID", rid)
    request_log.info(
        json.dumps(
            {
                "ts": started,
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(latency, 1),
                "rows": rows,
                "client": client_key(request, render=state.render_client_ip if state else False),
            }
        )
    )
    return response


def package_version() -> str:
    """pyproject.toml's version, the one source (scripts/release.py checks the tag against it)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("fraud")
    except PackageNotFoundError:  # running from a checkout without an install
        return "0.0.0"


def request_id_of(request: Request) -> str:
    return request.headers.get("x-request-id") or new_request_id()


def model_info(state: ServingState) -> ModelInfoResponse:
    pipe = state.model.pipeline
    n_inputs = sum(len(cols) for _, _, cols in pipe.named_steps["features"].transformers)
    info = state.info
    return ModelInfoResponse(
        model=str(info.get("model", "xgboost")),
        default_policy="rank",
        default_review_budget=state.default_review_budget,
        experiment=str(info.get("experiment", "")),
        version=state.model_version,
        feature_set=str(info.get("feature_set", "")),
        n_inputs=n_inputs,
        primary_metric=str(info.get("primary_metric", "pr_auc")),
        validation_pr_auc=float(info.get("validation_pr_auc", float("nan"))),
        test_pr_auc=float(info.get("test_pr_auc", float("nan"))),
        calibration=str(info.get("calibration", state.model.method)),
        bands=state.bands,
        training_window_days=tuple(info.get("training_window_days", (0, 0))),
        validation_window_days=tuple(info.get("validation_window_days", (0, 0))),
        parity_rows=state.parity_rows,
    )


def create_app(config_path: Path = DEFAULT_CONFIG) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.serving = load_state(config_path)
        yield

    app = FastAPI(title="fraud-risk-scoring", version=package_version(), lifespan=lifespan)
    if not request_log.handlers:  # one JSON object per line, ready for a log shipper
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        request_log.addHandler(handler)
        request_log.setLevel(logging.INFO)
        request_log.propagate = False

    @app.middleware("http")
    async def guard_and_record(request: Request, call_next: Any) -> Any:
        """Guarded endpoints: API key (when configured), time budget, requests table, one
        structured log line per call — success, rejection or error alike."""
        if not request.url.path.startswith(GUARDED_PREFIXES):
            return await call_next(request)
        started = utc_now()
        t0 = time.perf_counter()
        rid = request_id_of(request)
        state: ServingState | None = getattr(request.app.state, "serving", None)
        if state is not None:
            denied = _authorize(request, state)
            if denied is not None:
                return _finish(request, denied, rid, started, t0, state)
            if state.rate_limiter is not None:
                retry = state.rate_limiter.retry_after(
                    client_key(request, render=state.render_client_ip),
                    request.url.path,
                    request.method,
                )
                if retry:
                    response = Response(
                        json.dumps({"detail": f"rate limit exceeded; retry in {retry} seconds"}),
                        status_code=429,
                        headers={"Retry-After": str(retry)},
                        media_type="application/json",
                    )
                    return _finish(request, response, rid, started, t0, state)
        timeout = state.request_timeout_s if state is not None else 60.0
        try:
            response = await asyncio.wait_for(call_next(request), timeout=timeout)
        except TimeoutError:
            response = Response(
                json.dumps({"detail": f"request exceeded {timeout:g} s"}),
                status_code=504,
                media_type="application/json",
            )
        return _finish(request, response, rid, started, t0, state)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (STATIC / "dashboard.html").read_text()

    @app.get("/single", response_class=HTMLResponse, include_in_schema=False)
    def single() -> str:
        return (STATIC / "index.html").read_text()

    @app.get("/sample.csv", include_in_schema=False)
    def sample_csv() -> FileResponse:
        return FileResponse(STATIC / "sample_transactions.csv", media_type="text/csv")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        s: ServingState = request.app.state.serving
        return HealthResponse(
            status="ok",
            model_version=s.model_version,
            parity_rows=s.parity_rows,
            audit_events=s.audit.count() if s.audit is not None else None,
            auth="api_key" if s.api_key else "open",
            admin="api_key" if s.admin_api_key else "disabled",
            build_commit=build_commit(),
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    def info(request: Request) -> ModelInfoResponse:
        return model_info(request.app.state.serving)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        body: TransactionRequest,  # type: ignore[valid-type]
        request: Request,
        response: Response,
    ) -> PredictionResponse:
        state: ServingState = request.app.state.serving
        rid = request_id_of(request)
        t0 = time.perf_counter()
        payload = body.model_dump()  # type: ignore[attr-defined]
        frame = request_to_frame(payload)
        result = _response(
            state, int(payload[schema.ID_COL]), float(state.model.predict_proba(frame)[0, 1])
        )
        latency = (time.perf_counter() - t0) * 1000
        record_events(
            state, "/predict", rid, threshold_policy_applied(state),
            [(result.transaction_id, result.fraud_probability, result.risk_level, result.action)],
            latency, frame,
        )  # fmt: skip
        response.headers["X-Request-ID"] = rid
        response.headers["X-Rows"] = "1"
        return result

    @app.post("/explain", response_model=ExplanationResponse)
    def explain_one(
        body: TransactionRequest,  # type: ignore[valid-type]
        request: Request,
        response: Response,
        top_k: int = 8,
    ) -> ExplanationResponse:
        """Which inputs moved this transaction's score, and how far (docs/explanation.md).
        Not audited: it decides nothing."""
        state: ServingState = request.app.state.serving
        frame = request_to_frame(body.model_dump())  # type: ignore[attr-defined]
        e = explain(state.model, frame, top_k=max(1, min(top_k, 50)))
        response.headers["X-Rows"] = "1"
        return ExplanationResponse(
            model_version=state.model_version,
            note=(
                "contributions are log-odds on the model's raw score and sum with the bias to"
                " raw_margin; fraud_probability is the raw score after calibration. Under the"
                " label definition this explains why the transaction ranks high, not whether"
                " this purchase itself was fraudulent."
            ),
            **e.as_dict(),
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        body: BatchPredictionRequest, request: Request, response: Response
    ) -> BatchPredictionResponse:
        state: ServingState = request.app.state.serving
        rid = request_id_of(request)
        t0 = time.perf_counter()
        payloads = [t.model_dump() for t in body.transactions]  # type: ignore[attr-defined]
        frame = payloads_to_frame(payloads)
        result = score_batch(state, payloads, body.policy, body.review_budget, frame)
        latency = (time.perf_counter() - t0) * 1000
        scored = [
            (r.transaction_id, r.fraud_probability, r.risk_level, r.action) for r in result.ranked
        ]
        record_events(state, "/predict/batch", rid, result.policy, scored, latency, frame)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Rows"] = str(result.n)
        return result

    @app.post("/predict/csv", response_model=CsvPredictionResponse)
    async def predict_csv(
        request: Request,
        response: Response,
        file: UploadFile,
        review_budget: int | None = None,
        policy: Policy = "rank",
    ) -> CsvPredictionResponse:
        state: ServingState = request.app.state.serving
        rid = request_id_of(request)
        t0 = time.perf_counter()
        raw = await read_upload(request, file, state.max_upload_bytes)
        try:
            # one row past the limit is enough to know it was exceeded; never parse the rest
            table = pd.read_csv(
                io.BytesIO(raw), dtype="str", keep_default_na=False, nrows=MAX_CSV_ROWS + 1
            )
        except (ValueError, pd.errors.ParserError) as exc:
            raise HTTPException(422, f"could not parse CSV: {exc}") from exc
        if len(table) == 0:
            raise HTTPException(422, "the CSV has no rows")
        if len(table) > MAX_CSV_ROWS:
            raise HTTPException(422, f"at most {MAX_CSV_ROWS} rows per upload")
        try:
            frame, _ = table_to_frame(table)
            result = score_table(state, table, policy, review_budget)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        latency = (time.perf_counter() - t0) * 1000
        record_events(
            state, "/predict/csv", rid, result.summary.policy,
            [(r.transaction_id, r.fraud_probability, r.risk_level, r.action) for r in result.rows],
            latency, frame,
        )  # fmt: skip
        response.headers["X-Request-ID"] = rid
        response.headers["X-Rows"] = str(len(result.rows))
        return result

    @app.get("/audit/recent", response_model=list[AuditEvent])
    def audit_recent(
        request: Request, limit: int = 50, transaction_id: int | None = None
    ) -> list[AuditEvent]:
        s: ServingState = request.app.state.serving
        if s.audit is None:
            raise HTTPException(404, "prediction auditing is disabled on this server")
        return [AuditEvent(**e) for e in s.audit.recent(min(limit, 1000), transaction_id)]

    @app.post("/outcomes", response_model=OutcomesResponse)
    def post_outcomes(
        body: OutcomesRequest, request: Request, response: Response
    ) -> OutcomesResponse:
        """Delayed labels arriving for scored transactions (docs/feedback.md)."""
        s: ServingState = request.app.state.serving
        if s.audit is None:
            raise HTTPException(404, "prediction auditing is disabled on this server")
        now = utc_now()
        rows = [
            {**o.model_dump(), "is_fraud": int(o.is_fraud), "recorded_at": now,
             "source": body.source}
            for o in body.outcomes
        ]  # fmt: skip
        recorded = s.audit.record_outcomes(rows)
        clock = max(o.observed_dt for o in body.outcomes)
        total = s.audit.record_feed_run(clock, recorded, body.source)
        response.headers["X-Rows"] = str(recorded)
        return OutcomesResponse(received=len(rows), recorded=recorded, total=total)

    return app


app = create_app()
