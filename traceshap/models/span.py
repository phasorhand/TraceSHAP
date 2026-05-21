from dataclasses import dataclass
from datetime import datetime

from traceshap.models.enums import SpanKind


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class TraceSHAPSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    span_kind: SpanKind
    name: str
    input: dict
    output: dict
    start_time: datetime
    end_time: datetime
    tokens: TokenUsage | None
    cost: float | None
    metadata: dict
    raw_attributes: dict
    semconv_version: str

    @property
    def duration_ms(self) -> int:
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)
