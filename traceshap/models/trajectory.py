from dataclasses import dataclass, field

from traceshap.models.span import TraceSHAPSpan
from traceshap.models.step import CanonicalStep
from traceshap.models.outcome import Outcome


@dataclass
class SpanNode:
    span_id: str
    children: list["SpanNode"] = field(default_factory=list)


@dataclass
class TrajectoryMeta:
    framework: str
    agent_name: str
    agent_version: str | None = None
    task_type: str | None = None


@dataclass
class Trajectory:
    trace_id: str
    spans: list[TraceSHAPSpan]
    steps: list[CanonicalStep]
    span_tree: SpanNode
    outcome: Outcome | None
    metadata: TrajectoryMeta
