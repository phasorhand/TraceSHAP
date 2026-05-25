from __future__ import annotations

from traceshap.attribution.causal.models import CausalEdge, EdgeType
from traceshap.models.step import CanonicalStep
from traceshap.models.trajectory import Trajectory


class TrajectoryGraphBuilder:
    """Build a causal dependency graph from a Trajectory.

    For each ordered pair of steps (step_a at index i, step_b at index j > i):
    - DATA_DEPENDENCY (confidence=0.7) if:
      - step_a.output_hash == step_b.input_hash and both are non-empty, OR
      - step_a's span output text (len > 20) appears in step_b's span input text
    - TEMPORAL (confidence=0.3) if j == i + 1 (adjacent) and no data dependency
    - No edge for non-adjacent pairs without a data dependency
    """

    def build(self, trajectory: Trajectory) -> list[CausalEdge]:
        steps = trajectory.steps
        if not steps:
            return []

        # Build a lookup from span_id -> span for fast access
        spans_by_id = {span.span_id: span for span in trajectory.spans}

        edges: list[CausalEdge] = []
        n = len(steps)

        for i in range(n):
            for j in range(i + 1, n):
                step_a = steps[i]
                step_b = steps[j]

                if self._has_data_dependency(step_a, step_b, spans_by_id):
                    edges.append(
                        CausalEdge(
                            source_step_id=step_a.step_id,
                            target_step_id=step_b.step_id,
                            edge_type=EdgeType.DATA_DEPENDENCY,
                            confidence=0.7,
                            evidence=(
                                f"Data dependency detected between step {step_a.step_id} "
                                f"and step {step_b.step_id}"
                            ),
                        )
                    )
                elif j == i + 1:
                    # Adjacent steps with no data dependency get a temporal edge
                    edges.append(
                        CausalEdge(
                            source_step_id=step_a.step_id,
                            target_step_id=step_b.step_id,
                            edge_type=EdgeType.TEMPORAL,
                            confidence=0.3,
                            evidence=(
                                f"Step {step_b.step_id} immediately follows step {step_a.step_id}"
                            ),
                        )
                    )
                # Non-adjacent pairs without data dependency → no edge

        return edges

    def _has_data_dependency(
        self,
        step_a: CanonicalStep,
        step_b: CanonicalStep,
        spans_by_id: dict,
    ) -> bool:
        """Return True if step_b appears to consume output produced by step_a."""
        # Hash-based check: non-empty and matching
        if step_a.output_hash and step_b.input_hash and step_a.output_hash == step_b.input_hash:
            return True

        # Content-based check: span output text (len > 20) appears in span input text
        for span_id_a in step_a.raw_span_ids:
            span_a = spans_by_id.get(span_id_a)
            if span_a is None:
                continue
            output_text: str = span_a.output.get("text", "")
            if not isinstance(output_text, str) or len(output_text) <= 20:
                continue
            # Look for this text in any of step_b's span inputs
            for span_id_b in step_b.raw_span_ids:
                span_b = spans_by_id.get(span_id_b)
                if span_b is None:
                    continue
                input_text: str = span_b.input.get("text", "")
                if isinstance(input_text, str) and output_text in input_text:
                    return True

        return False
