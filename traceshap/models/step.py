from dataclasses import dataclass
from datetime import datetime

from traceshap.models.enums import StepType, SideEffect
from traceshap.models.span import TokenUsage

PROTECTED_STEP_TYPES = frozenset({StepType.VALIDATION})


@dataclass
class CanonicalStep:
    step_id: str
    raw_span_ids: list[str]
    node_id: str | None
    tool_name: str | None
    step_type: StepType
    attempt_index: int
    loop_iteration: int | None
    input_hash: str
    output_hash: str
    side_effect_class: SideEffect
    framework_mapping_confidence: float
    tokens: TokenUsage | None
    cost: float | None
    start_time: datetime
    end_time: datetime

    @property
    def duration_ms(self) -> int:
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)

    @property
    def is_protected(self) -> bool:
        return self.step_type in PROTECTED_STEP_TYPES
