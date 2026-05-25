"""OTLPLiveSource — real-time OTLP HTTP endpoint backed by an in-process buffer.

The source embeds a lightweight aiohttp server that listens on
``POST /v1/traces`` and funnels incoming spans into a thread-safe buffer.
If *aiohttp* is not installed, the HTTP server is silently skipped; callers
can still push data directly via :meth:`ingest`.
"""
from __future__ import annotations

import threading
from typing import Any

from traceshap.models.span import TraceSHAPSpan
from traceshap.ingestion.sources.base import SpanSource
from traceshap.ingestion.sources.otlp_common import parse_otlp_resource_spans

try:
    import aiohttp
    from aiohttp import web as aiohttp_web

    _AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AIOHTTP_AVAILABLE = False


class OTLPLiveSource(SpanSource):
    """SpanSource that accepts OTLP JSON payloads in real time.

    Args:
        host: Bind address for the embedded HTTP server (default ``"0.0.0.0"``).
        port: TCP port for the embedded HTTP server (default ``4318``).
        source_hint: Annotation stored on every parsed span as
            ``semconv_version = "otlp-<source_hint>"``.
        auth_token: Optional Bearer token.  When set, every incoming HTTP
            request must carry the header ``Authorization: Bearer <token>``;
            requests without a valid token are rejected with HTTP 401.
        max_buffer_size: Maximum number of spans held in the buffer at once.
            Spans that would exceed this limit are silently dropped.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4318,
        source_hint: str = "otlp",
        auth_token: str | None = None,
        max_buffer_size: int = 10000,
    ) -> None:
        self.host = host
        self.port = port
        self.source_hint = source_hint
        self._auth_token = auth_token
        self._max_buffer_size = max_buffer_size

        self._buffer: list[TraceSHAPSpan] = []
        self._lock = threading.Lock()

        # aiohttp state (populated in connect() if aiohttp is available)
        self._runner: Any | None = None
        self._site: Any | None = None

    # ------------------------------------------------------------------
    # Public data-flow API (synchronous, usable without the HTTP server)
    # ------------------------------------------------------------------

    def ingest(self, data: dict) -> None:
        """Parse an OTLP JSON payload and append spans to the buffer.

        Respects *max_buffer_size*: once the buffer is full, additional spans
        from this call are silently discarded.
        """
        spans = parse_otlp_resource_spans(data, self.source_hint)
        with self._lock:
            available = self._max_buffer_size - len(self._buffer)
            if available <= 0:
                return
            self._buffer.extend(spans[:available])

    def poll(self) -> list[TraceSHAPSpan]:
        """Return all buffered spans and clear the buffer (thread-safe)."""
        with self._lock:
            spans, self._buffer = self._buffer, []
        return spans

    # ------------------------------------------------------------------
    # SpanSource interface
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Start the embedded aiohttp server if aiohttp is available."""
        if not _AIOHTTP_AVAILABLE:
            return

        app = aiohttp_web.Application()
        app.router.add_post("/v1/traces", self._handle_traces)

        self._runner = aiohttp_web.AppRunner(app)
        await self._runner.setup()
        self._site = aiohttp_web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

    async def close(self) -> None:
        """Shut down the aiohttp server if it was started."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    # ------------------------------------------------------------------
    # HTTP handler
    # ------------------------------------------------------------------

    async def _handle_traces(self, request: Any) -> Any:  # pragma: no cover
        """Handle POST /v1/traces — validate auth, parse body, call ingest()."""
        if self._auth_token is not None:
            auth_header = request.headers.get("Authorization", "")
            expected = f"Bearer {self._auth_token}"
            if auth_header != expected:
                return aiohttp_web.Response(status=401, text="Unauthorized")

        try:
            data = await request.json()
        except Exception:
            return aiohttp_web.Response(status=400, text="Invalid JSON")

        self.ingest(data)
        return aiohttp_web.Response(status=200, text="OK")
