from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from traceshap.models.enums import SpanKind
from traceshap.models.span import TraceSHAPSpan
from traceshap.ingestion.sources.base import SpanSource

# Map Hermes ShareGPT role strings to TraceSHAP SpanKind values.
_ROLE_KIND_MAP: dict[str, SpanKind] = {
    "system": SpanKind.CUSTOM,
    "human": SpanKind.CUSTOM,
    "gpt": SpanKind.LLM,
    "tool": SpanKind.TOOL,
}


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _wrap_value(value: object) -> dict:
    """Return value unchanged if it is a dict, otherwise wrap as {text: value}."""
    if isinstance(value, dict):
        return value
    return {"text": str(value)}


def _parse_timestamp(ts: str | None) -> datetime:
    """Parse an ISO-8601 timestamp string into an aware datetime (UTC)."""
    if ts is None:
        return datetime.now(tz=timezone.utc)
    # Python 3.11+ handles "Z" suffix; earlier versions need a replace.
    ts_normalised = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts_normalised)


def _convert_entry(raw_json: str) -> list[TraceSHAPSpan]:
    """Convert a single JSONL line (already decoded to str) into TraceSHAPSpan objects."""
    data = json.loads(raw_json)

    trace_id = _sha256_hex(raw_json)[:32]
    timestamp = _parse_timestamp(data.get("timestamp"))
    model = data.get("model", "")
    conversations: list[dict] = data.get("conversations", [])

    spans: list[TraceSHAPSpan] = []
    for index, message in enumerate(conversations):
        role: str = message.get("from", "")
        value = message.get("value", "")

        span_id = _sha256_hex(f"{trace_id}_{index}")[:16]
        span_kind = _ROLE_KIND_MAP.get(role, SpanKind.CUSTOM)
        wrapped = _wrap_value(value)

        span = TraceSHAPSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            span_kind=span_kind,
            name=role,
            input=wrapped,
            output=wrapped,
            start_time=timestamp,
            end_time=timestamp,
            tokens=None,
            cost=None,
            metadata={"model": model, "role": role},
            raw_attributes={"from": role, "value": value},
            semconv_version="hermes-sharegpt",
        )
        spans.append(span)

    return spans


class HermesJsonlSource(SpanSource):
    """SpanSource that reads Hermes agent trajectories from ShareGPT JSONL files."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._polled = False

    async def connect(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Hermes JSONL source not found: {self._path}")

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._polled:
            return []
        self._polled = True

        all_spans: list[TraceSHAPSpan] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                spans = _convert_entry(line)
                all_spans.extend(spans)

        return all_spans

    async def close(self) -> None:
        pass
