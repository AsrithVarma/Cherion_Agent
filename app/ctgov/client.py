"""Async HTTP client for the ClinicalTrials.gov v2 API.

The public site is fronted by a WAF that inspects the ``User-Agent``. Contrary
to the usual advice, spoofing a *browser* UA is blocked with HTTP 403 (a browser
UA with a non-browser TLS fingerprint looks like a bot); the WAF instead admits
honest library clients whose UA contains ``python-httpx``. This client therefore
sends a polite, identifiable UA that keeps that token. Every request retries with
exponential backoff on transient failures (HTTP 429 and 5xx, plus network
errors), up to three attempts total.

The API is unauthenticated and asks callers to stay around ~1 request/second;
pagination follows ``nextPageToken`` by re-sending it as ``pageToken``.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

import httpx

from app.config import get_settings

# Polite, identifiable UA. MUST retain the "python-httpx" token: the CT.gov WAF
# returns 403 for browser-spoofing or unrecognized UAs but admits this one.
_USER_AGENT = "Cheiron/1.0 (clinical-trials-viz) python-httpx"

# Retry policy.
_MAX_ATTEMPTS = 3
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_INITIAL_BACKOFF_SECONDS = 1.0


class CTGovClient:
    """Thin async wrapper around ``httpx.AsyncClient`` for the CT.gov v2 API.

    Intended to be used as an async context manager so the underlying
    connection pool is closed cleanly::

        async with CTGovClient() as client:
            page = await client.search_studies({"query.cond": "melanoma"})
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """Create the client and its pooled ``httpx.AsyncClient``.

        Args:
            base_url: Override for the API base URL; defaults to settings.
            timeout: Per-request timeout in seconds (connect/read/write/pool).
        """
        settings = get_settings()
        self._base_url = (base_url or settings.ctgov_base_url).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/json",
            },
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "CTGovClient":
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    # -- public API --------------------------------------------------------

    async def get(
        self, path: str, params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """GET an arbitrary API path and return parsed JSON.

        This is the general-purpose helper used by the introspection module to
        hit ``/studies/enums``, ``/studies/metadata``, ``/studies/search-areas``
        and ``/version``.

        Args:
            path: Path relative to the base URL (e.g. ``/studies/enums``).
            params: Optional query parameters.

        Returns:
            The parsed JSON response as a dict.
        """
        return await self._request(path, params or {})

    async def search_studies(self, params: dict[str, Any]) -> dict[str, Any]:
        """Query ``GET /studies`` and return the parsed JSON response.

        Args:
            params: Query params such as ``query.cond``, ``filter.overallStatus``,
                ``fields``, ``pageSize``, ``pageToken``, ``countTotal``.

        Returns:
            The parsed response dict: ``{studies, totalCount, nextPageToken}``.
        """
        return await self._request("/studies", params)

    async def paginate(
        self,
        params: dict[str, Any],
        max_pages: Optional[int] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield successive ``/studies`` pages, following ``nextPageToken``.

        Each response's ``nextPageToken`` is passed back as ``pageToken`` on the
        following request. Stops when there is no next token or after
        ``max_pages`` pages (whichever comes first).

        Args:
            params: Base query params. Any ``pageToken`` is ignored/overwritten.
            max_pages: Page cap; defaults to ``settings.max_pages``.

        Yields:
            One parsed response dict per page.
        """
        if max_pages is None:
            max_pages = get_settings().max_pages

        page_params = dict(params)
        page_params.pop("pageToken", None)

        for _ in range(max_pages):
            page = await self.search_studies(page_params)
            yield page

            token = page.get("nextPageToken")
            if not token:
                break
            page_params["pageToken"] = token

    # -- internals ---------------------------------------------------------

    async def _request(
        self, path: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform a GET with retry/backoff and return parsed JSON.

        Retries on network errors and on retryable status codes (429, 5xx),
        honoring an integer ``Retry-After`` header when present. Raises
        ``httpx.HTTPStatusError`` on a non-retryable error status or after the
        final attempt, and ``httpx.RequestError`` if the network keeps failing.
        """
        backoff = _INITIAL_BACKOFF_SECONDS
        last_response: Optional[httpx.Response] = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.get(path, params=params)
            except httpx.RequestError:
                # Network-level failure (timeout, connection reset, DNS, ...).
                if attempt == _MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            if response.status_code in _RETRY_STATUSES:
                last_response = response
                if attempt == _MAX_ATTEMPTS:
                    break
                await asyncio.sleep(self._retry_delay(response, backoff))
                backoff *= 2
                continue

            response.raise_for_status()
            return response.json()

        # All attempts exhausted on retryable statuses: surface the last one.
        assert last_response is not None  # loop ran at least once
        last_response.raise_for_status()
        return last_response.json()  # pragma: no cover — raise_for_status raised

    @staticmethod
    def _retry_delay(response: httpx.Response, fallback: float) -> float:
        """Return the wait before retrying: ``Retry-After`` seconds if given."""
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return float(retry_after)
        return fallback
