from __future__ import annotations

import json
from pathlib import Path

from traceshap.models.span import TraceSHAPSpan
from traceshap.ingestion.sources.base import SpanSource
from traceshap.ingestion.sources.otlp_common import (  # noqa: F401 – re-exported for back-compat
    OTEL_KIND_MAP,
    TOOL_ATTRIBUTE_KEYS,
    get_attr,
    nano_to_datetime,
    attrs_to_dict,
    infer_span_kind,
    extract_tokens,
    convert_otlp_span,
    parse_otlp_resource_spans,
)


class OTLPJsonSource(SpanSource):
    def __init__(self, path: str, source_hint: str = "otlp"):
        self._path = Path(path)
        self._source_hint = source_hint
        self._polled = False

    async def connect(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"OTLP source path not found: {self._path}")

    async def poll(self) -> list[TraceSHAPSpan]:
        if self._polled:
            return []
        self._polled = True

        files: list[Path] = []
        if self._path.is_file():
            files = [self._path]
        elif self._path.is_dir():
            files = sorted(self._path.glob("*.json"))

        all_spans: list[TraceSHAPSpan] = []
        for file_path in files:
            with open(file_path) as f:
                data = json.load(f)
            spans = parse_otlp_resource_spans(data, self._source_hint)
            all_spans.extend(spans)

        return all_spans

    async def close(self) -> None:
        pass
