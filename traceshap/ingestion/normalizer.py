import hashlib
import json
import uuid

from traceshap.models.span import TraceSHAPSpan
from traceshap.models.step import CanonicalStep
from traceshap.models.enums import SpanKind, StepType, SideEffect

SPAN_KIND_TO_STEP_TYPE: dict[SpanKind, StepType] = {
    SpanKind.LLM: StepType.DECISION,
    SpanKind.TOOL: StepType.ACTION,
    SpanKind.RETRIEVER: StepType.OBSERVATION,
    SpanKind.RERANKER: StepType.OBSERVATION,
    SpanKind.AGENT: StepType.DECISION,
    SpanKind.GUARDRAIL: StepType.VALIDATION,
    SpanKind.EVALUATOR: StepType.VALIDATION,
    SpanKind.CUSTOM: StepType.META,
}

SPAN_KIND_DEFAULT_SIDE_EFFECT: dict[SpanKind, SideEffect] = {
    SpanKind.LLM: SideEffect.PURE,
    SpanKind.RETRIEVER: SideEffect.READ_ONLY,
    SpanKind.RERANKER: SideEffect.PURE,
    SpanKind.GUARDRAIL: SideEffect.PURE,
    SpanKind.EVALUATOR: SideEffect.PURE,
    SpanKind.AGENT: SideEffect.PURE,
    SpanKind.CUSTOM: SideEffect.IRREVERSIBLE_WRITE,
}


def _hash_dict(d: dict) -> str:
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class StepNormalizer:
    def __init__(
        self,
        side_effect_overrides: dict[str, SideEffect] | None = None,
    ):
        self._side_effect_overrides = side_effect_overrides or {}

    def normalize(self, spans: list[TraceSHAPSpan]) -> list[CanonicalStep]:
        sorted_spans = sorted(spans, key=lambda s: s.start_time)
        steps: list[CanonicalStep] = []
        retry_tracker: dict[str, int] = {}

        for span in sorted_spans:
            retry_key = f"{span.name}:{_hash_dict(span.input)}"
            attempt = retry_tracker.get(retry_key, 0)
            retry_tracker[retry_key] = attempt + 1

            step = CanonicalStep(
                step_id=f"step-{uuid.uuid4().hex[:12]}",
                raw_span_ids=[span.span_id],
                node_id=span.metadata.get("node_id"),
                tool_name=span.name if span.span_kind == SpanKind.TOOL else None,
                step_type=SPAN_KIND_TO_STEP_TYPE.get(span.span_kind, StepType.META),
                attempt_index=attempt,
                loop_iteration=None,
                input_hash=_hash_dict(span.input),
                output_hash=_hash_dict(span.output),
                side_effect_class=self._get_side_effect(span),
                framework_mapping_confidence=self._estimate_mapping_confidence(span),
                tokens=span.tokens,
                cost=span.cost,
                start_time=span.start_time,
                end_time=span.end_time,
            )
            steps.append(step)

        return steps

    def _get_side_effect(self, span: TraceSHAPSpan) -> SideEffect:
        if span.name in self._side_effect_overrides:
            return self._side_effect_overrides[span.name]
        if span.span_kind == SpanKind.TOOL:
            return SideEffect.IRREVERSIBLE_WRITE
        return SPAN_KIND_DEFAULT_SIDE_EFFECT.get(span.span_kind, SideEffect.IRREVERSIBLE_WRITE)

    @staticmethod
    def _estimate_mapping_confidence(span: TraceSHAPSpan) -> float:
        if span.metadata.get("node_id"):
            return 0.95
        if span.span_kind in (SpanKind.LLM, SpanKind.TOOL):
            return 0.7
        return 0.5
