"""HTTP client for the castra coordinator.

Both submitters (training scripts) and workers use this — same wire protocol.
Sync httpx; async path can be added later if needed.
"""

from __future__ import annotations

from typing import Any

import httpx


class CoordinatorClient:
    """Thin sync client over the coordinator REST API."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "CoordinatorClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- worker lifecycle ---

    def register_worker(self, *, backend: str,
                        hostname: str | None = None) -> str:
        r = self._http.post("/workers/register",
                            json={"backend": backend, "hostname": hostname})
        r.raise_for_status()
        return r.json()["worker_id"]

    def heartbeat(self, worker_id: str) -> None:
        r = self._http.post(f"/workers/{worker_id}/heartbeat")
        r.raise_for_status()

    # --- shard ops ---

    def submit(self, experiment: str,
               shards: list[tuple[str, dict[str, Any]]]) -> list[str]:
        """Submit shards as a list of (type_name, payload) tuples."""
        body = {
            "experiment": experiment,
            "shards": [{"type_name": t, "payload": p} for t, p in shards],
        }
        r = self._http.post("/shards/submit", json=body)
        r.raise_for_status()
        return r.json()["shard_ids"]

    def claim(self, worker_id: str, *, experiment: str | None = None,
              lease_seconds: int = 240) -> dict[str, Any] | None:
        """Claim the next pending shard. Returns None if none available."""
        body: dict[str, Any] = {"worker_id": worker_id, "lease_seconds": lease_seconds}
        if experiment is not None:
            body["experiment"] = experiment
        r = self._http.post("/shards/claim", json=body)
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()

    def complete(self, shard_id: str, *, worker_id: str,
                 result: dict[str, Any]) -> None:
        r = self._http.post(
            f"/shards/{shard_id}/complete",
            json={"worker_id": worker_id, "result": result},
        )
        r.raise_for_status()

    def fail(self, shard_id: str, *, worker_id: str, error: str,
             requeue: bool = True) -> None:
        r = self._http.post(
            f"/shards/{shard_id}/fail",
            json={"worker_id": worker_id, "error": error, "requeue": requeue},
        )
        r.raise_for_status()

    # --- queries ---

    def status(self, experiment: str) -> dict[str, Any]:
        r = self._http.get(f"/experiments/{experiment}/status")
        r.raise_for_status()
        return r.json()

    def list_shards(self, *, experiment: str | None = None,
                    status: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if experiment is not None:
            params["experiment"] = experiment
        if status is not None:
            params["status"] = status
        r = self._http.get("/shards", params=params)
        r.raise_for_status()
        return r.json()["shards"]

    def healthz(self) -> dict[str, Any]:
        r = self._http.get("/healthz")
        r.raise_for_status()
        return r.json()
