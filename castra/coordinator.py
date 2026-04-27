"""The coordinator — a fastapi sidecar that owns shard state and brokers
work between submitters and workers.

Endpoints:

    POST /workers/register             register a new worker
    POST /workers/{id}/heartbeat       extend lease
    POST /shards/submit                submit one or more shards
    POST /shards/claim                 worker requests next pending shard
    POST /shards/{id}/complete         worker reports success
    POST /shards/{id}/fail             worker reports failure (requeues)
    GET  /shards                       list (admin / debug)
    GET  /experiments/{name}/status    snapshot for TUI/dashboard
    GET  /healthz                      liveness probe

The fastapi `app` is created via `make_app(store, ...)` so tests can inject
a fresh in-memory or temp-file Store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from castra.storage import ShardRecord, Store


class WorkerRegister(BaseModel):
    backend: str
    hostname: str | None = None


class WorkerRegistered(BaseModel):
    worker_id: str


class HeartbeatResponse(BaseModel):
    ok: bool


class ShardSubmission(BaseModel):
    type_name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SubmitRequest(BaseModel):
    experiment: str
    shards: list[ShardSubmission]


class SubmitResponse(BaseModel):
    shard_ids: list[str]


class ClaimRequest(BaseModel):
    worker_id: str
    experiment: str | None = None
    lease_seconds: int = 240


class ClaimResponse(BaseModel):
    shard_id: str
    type_name: str
    payload: dict[str, Any]
    lease_expires_at: str
    attempts: int


class CompleteRequest(BaseModel):
    worker_id: str
    result: dict[str, Any] = Field(default_factory=dict)


class FailRequest(BaseModel):
    worker_id: str
    error: str
    requeue: bool = True


def _shard_to_dict(s: ShardRecord) -> dict[str, Any]:
    return {
        "shard_id": s.shard_id,
        "experiment": s.experiment,
        "type_name": s.type_name,
        "status": s.status,
        "worker_id": s.worker_id,
        "attempts": s.attempts,
        "submitted_at": s.submitted_at,
        "claimed_at": s.claimed_at,
        "lease_expires_at": s.lease_expires_at,
        "completed_at": s.completed_at,
        "error_text": s.error_text,
    }


def make_app(store: Store, *, experiment_label: str | None = None) -> FastAPI:
    """Build a FastAPI app bound to the given Store.

    `experiment_label` is metadata exposed at /healthz; the coordinator can
    serve any experiment if requests target one explicitly.
    """
    app = FastAPI(title="castra coordinator", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "experiment_label": experiment_label,
            "shard_counts": store.shard_counts(),
        }

    @app.post("/workers/register", response_model=WorkerRegistered)
    def register_worker(req: WorkerRegister) -> WorkerRegistered:
        wid = store.register_worker(backend=req.backend, hostname=req.hostname)
        return WorkerRegistered(worker_id=wid)

    @app.post("/workers/{worker_id}/heartbeat", response_model=HeartbeatResponse)
    def heartbeat(worker_id: str) -> HeartbeatResponse:
        ok = store.heartbeat(worker_id)
        if not ok:
            raise HTTPException(status_code=404, detail="unknown worker_id")
        return HeartbeatResponse(ok=True)

    @app.post("/shards/submit", response_model=SubmitResponse)
    def submit(req: SubmitRequest) -> SubmitResponse:
        ids = store.submit_many(
            req.experiment,
            [(s.type_name, s.payload) for s in req.shards],
        )
        return SubmitResponse(shard_ids=ids)

    @app.post("/shards/claim")
    def claim(req: ClaimRequest):
        rec = store.claim_next(
            req.worker_id,
            experiment=req.experiment,
            lease_seconds=req.lease_seconds,
        )
        if rec is None:
            # No pending shards. 204 means "valid request, nothing to do."
            from fastapi.responses import Response
            return Response(status_code=204)
        return ClaimResponse(
            shard_id=rec.shard_id,
            type_name=rec.type_name,
            payload=rec.payload,
            lease_expires_at=rec.lease_expires_at or "",
            attempts=rec.attempts,
        )

    @app.post("/shards/{shard_id}/complete")
    def complete(shard_id: str, req: CompleteRequest) -> dict[str, bool]:
        rec = store.get_shard(shard_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown shard_id")
        if rec.status != "claimed":
            raise HTTPException(
                status_code=409,
                detail=f"shard not in 'claimed' state (was {rec.status!r})",
            )
        store.complete_shard(shard_id, req.result)
        return {"ok": True}

    @app.post("/shards/{shard_id}/fail")
    def fail(shard_id: str, req: FailRequest) -> dict[str, bool]:
        rec = store.get_shard(shard_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="unknown shard_id")
        store.fail_shard(shard_id, req.error, requeue=req.requeue)
        return {"ok": True}

    @app.get("/shards")
    def list_shards(experiment: str | None = None,
                    status: str | None = None) -> dict[str, Any]:
        rows = store.list_shards(experiment=experiment, status=status)
        return {"shards": [_shard_to_dict(r) for r in rows]}

    @app.get("/experiments/{name}/status")
    def experiment_status(name: str) -> dict[str, Any]:
        return {
            "experiment": name,
            "shard_counts": store.shard_counts(experiment=name),
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "backend": w.backend,
                    "hostname": w.hostname,
                    "last_heartbeat_at": w.last_heartbeat_at,
                    "status": w.status,
                }
                for w in store.list_workers()
            ],
        }

    return app


def serve(db_path: Path, *, host: str = "127.0.0.1", port: int = 8765,
          experiment_label: str | None = None) -> None:
    """Run the coordinator with uvicorn (foreground, blocking)."""
    import uvicorn

    store = Store(db_path)
    app = make_app(store, experiment_label=experiment_label)
    uvicorn.run(app, host=host, port=port, log_level="info")
