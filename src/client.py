"""Typed HTTP clients for the Medical OpenEnv API."""

from __future__ import annotations

from typing import Any

import httpx


JSONDict = dict[str, Any]
RecordsPayload = list[dict[str, Any]]
KnowledgePayload = list[dict[str, Any]]


class APIError(RuntimeError):
    """Raised when the OpenEnv API returns a non-2xx response."""

    def __init__(self, status_code: int, payload: Any):
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        super().__init__(f"OpenEnv API error ({status_code}): {detail}")
        self.status_code = status_code
        self.payload = payload
        self.detail = detail


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _response_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        text = response.text.strip()
        return {"detail": text or f"HTTP {response.status_code}"}


def _raise_for_error(response: httpx.Response, payload: Any) -> None:
    if response.status_code >= 400:
        raise APIError(response.status_code, payload)


class MedicalOpenEnvClient:
    """
    Async client for the Medical OpenEnv server.

    Usage:
        async with MedicalOpenEnvClient(base_url="http://localhost:7860") as env:
            reset = await env.reset(task_id=1, seed=42)
            episode_id = reset["episode_id"]
            records = reset["observation"]["records"]
            result = await env.step(episode_id, records=records, is_final=True)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        *,
        timeout: float = 120.0,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = _normalize_base_url(base_url)
        self._timeout = timeout
        self._headers = dict(headers or {})
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def __aenter__(self) -> "MedicalOpenEnvClient":
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def sync(self) -> "SyncMedicalOpenEnvClient":
        """Create a sync wrapper client with the same base URL and settings."""
        return SyncMedicalOpenEnvClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._headers,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._headers,
                transport=self._transport,
            )
        return self._client

    async def _request(self, method: str, path: str, **kwargs) -> JSONDict:
        client = await self._get_client()
        response = await client.request(method, path, **kwargs)
        payload = _response_payload(response)
        _raise_for_error(response, payload)
        return payload

    async def tasks(self) -> JSONDict:
        return await self._request("GET", "/tasks")

    async def contract(self) -> JSONDict:
        return await self._request("GET", "/contract")

    async def reset(self, *, task_id: int = 1, seed: int = 42) -> JSONDict:
        return await self._request("POST", "/reset", json={"task_id": task_id, "seed": seed})

    async def step(
        self,
        episode_id: str,
        *,
        records: RecordsPayload | None = None,
        knowledge: KnowledgePayload | None = None,
        is_final: bool = False,
    ) -> JSONDict:
        payload: JSONDict = {"is_final": is_final}
        if records is not None:
            payload["records"] = records
        if knowledge is not None:
            payload["knowledge"] = knowledge
        return await self._request(
            "POST",
            "/step",
            params={"episode_id": episode_id},
            json=payload,
        )

    async def state(self, episode_id: str) -> JSONDict:
        return await self._request("GET", "/state", params={"episode_id": episode_id})

    async def grader(self, episode_id: str) -> JSONDict:
        return await self._request("GET", "/grader", params={"episode_id": episode_id})

    async def export(self, episode_id: str) -> JSONDict:
        return await self._request("GET", "/export", params={"episode_id": episode_id})

    async def schema(self) -> JSONDict:
        return await self._request("GET", "/schema")

    async def metadata(self) -> JSONDict:
        return await self._request("GET", "/metadata")

    async def mode(self) -> JSONDict:
        return await self._request("GET", "/mode")

    async def health(self) -> JSONDict:
        return await self._request("GET", "/health")

    async def baseline(self) -> JSONDict:
        return await self._request("GET", "/baseline")


class SyncMedicalOpenEnvClient:
    """Synchronous client for the Medical OpenEnv server."""

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        *,
        timeout: float = 120.0,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = _normalize_base_url(base_url)
        self._timeout = timeout
        self._headers = dict(headers or {})
        self._transport = transport
        self._client: httpx.Client | None = None

    @property
    def base_url(self) -> str:
        return self._base_url

    def __enter__(self) -> "SyncMedicalOpenEnvClient":
        self._get_client()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=self._headers,
                transport=self._transport,
            )
        return self._client

    def _request(self, method: str, path: str, **kwargs) -> JSONDict:
        client = self._get_client()
        response = client.request(method, path, **kwargs)
        payload = _response_payload(response)
        _raise_for_error(response, payload)
        return payload

    def tasks(self) -> JSONDict:
        return self._request("GET", "/tasks")

    def contract(self) -> JSONDict:
        return self._request("GET", "/contract")

    def reset(self, *, task_id: int = 1, seed: int = 42) -> JSONDict:
        return self._request("POST", "/reset", json={"task_id": task_id, "seed": seed})

    def step(
        self,
        episode_id: str,
        *,
        records: RecordsPayload | None = None,
        knowledge: KnowledgePayload | None = None,
        is_final: bool = False,
    ) -> JSONDict:
        payload: JSONDict = {"is_final": is_final}
        if records is not None:
            payload["records"] = records
        if knowledge is not None:
            payload["knowledge"] = knowledge
        return self._request(
            "POST",
            "/step",
            params={"episode_id": episode_id},
            json=payload,
        )

    def state(self, episode_id: str) -> JSONDict:
        return self._request("GET", "/state", params={"episode_id": episode_id})

    def grader(self, episode_id: str) -> JSONDict:
        return self._request("GET", "/grader", params={"episode_id": episode_id})

    def export(self, episode_id: str) -> JSONDict:
        return self._request("GET", "/export", params={"episode_id": episode_id})

    def schema(self) -> JSONDict:
        return self._request("GET", "/schema")

    def metadata(self) -> JSONDict:
        return self._request("GET", "/metadata")

    def mode(self) -> JSONDict:
        return self._request("GET", "/mode")

    def health(self) -> JSONDict:
        return self._request("GET", "/health")

    def baseline(self) -> JSONDict:
        return self._request("GET", "/baseline")


__all__ = [
    "APIError",
    "MedicalOpenEnvClient",
    "SyncMedicalOpenEnvClient",
]
