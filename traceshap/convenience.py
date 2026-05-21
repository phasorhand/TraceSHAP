from __future__ import annotations

from traceshap.models.span import TraceSHAPSpan
from traceshap.models.trajectory import Trajectory, SpanNode, TrajectoryMeta
from traceshap.models.outcome import Outcome, StepAttribution
from traceshap.models.pruning import PruningReport
from traceshap.ingestion.normalizer import StepNormalizer
from traceshap.ingestion.assembler import TreeAssembler
from traceshap.attribution.engine import AttributionEngine
from traceshap.attribution.layer0_rules import Layer0Rules
from traceshap.attribution.layer1_lift import Layer1Lift
from traceshap.attribution.layer2_sequence import Layer2Sequence
from traceshap.pruning.advisor import PruningAdvisor
from traceshap.config import PruningConfig


def spans_to_trajectory(
    spans: list[TraceSHAPSpan],
    trace_id: str,
    outcome: Outcome | None = None,
    framework: str = "unknown",
    agent_name: str = "default",
) -> Trajectory:
    sorted_spans = sorted(spans, key=lambda s: s.start_time)
    normalizer = StepNormalizer()
    steps = normalizer.normalize(sorted_spans)
    span_tree = TreeAssembler.build(sorted_spans)

    return Trajectory(
        trace_id=trace_id,
        spans=sorted_spans,
        steps=steps,
        span_tree=span_tree,
        outcome=outcome,
        metadata=TrajectoryMeta(
            framework=framework,
            agent_name=agent_name,
        ),
    )


async def quick_analyze(
    spans: list[TraceSHAPSpan],
    trace_id: str,
    layers: list[int] | None = None,
    outcome: Outcome | None = None,
    framework: str = "unknown",
    agent_name: str = "default",
    include_pruning: bool = False,
    training_trajectories: list[Trajectory] | None = None,
) -> dict:
    if layers is None:
        layers = [0]

    trajectory = spans_to_trajectory(
        spans, trace_id=trace_id, outcome=outcome,
        framework=framework, agent_name=agent_name,
    )

    layer_objs = []
    for layer_id in layers:
        if layer_id == 0:
            layer_objs.append(Layer0Rules())
        elif layer_id == 1:
            l1 = Layer1Lift()
            if training_trajectories:
                l1.fit(training_trajectories)
            layer_objs.append(l1)
        elif layer_id == 2:
            l2 = Layer2Sequence()
            if training_trajectories:
                l2.fit(training_trajectories)
            layer_objs.append(l2)

    engine = AttributionEngine(layers=layer_objs)
    attributions = await engine.analyze(trajectory)

    result: dict = {
        "trajectory": trajectory,
        "attributions": attributions,
    }

    if include_pruning:
        config = PruningConfig()
        advisor = PruningAdvisor(config)
        report = advisor.analyze(trajectory, attributions)
        result["pruning_report"] = report

    return result
