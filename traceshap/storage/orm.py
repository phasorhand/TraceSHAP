from datetime import datetime, timezone

from sqlalchemy import String, Integer, Float, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TrajectoryRow(Base):
    __tablename__ = "trajectories"

    trace_id: Mapped[str] = mapped_column(String, primary_key=True)
    framework: Mapped[str] = mapped_column(String)
    agent_name: Mapped[str] = mapped_column(String)
    agent_version: Mapped[str | None] = mapped_column(String, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String, nullable=True)
    outcome_success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    outcome_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome_token_cost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outcome_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class SpanRow(Base):
    __tablename__ = "spans"

    span_id: Mapped[str] = mapped_column(String, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String, ForeignKey("trajectories.trace_id"))
    parent_span_id: Mapped[str | None] = mapped_column(String, nullable=True)
    span_kind: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    input_data: Mapped[str] = mapped_column(Text)
    output_data: Mapped[str] = mapped_column(Text)
    start_time: Mapped[str] = mapped_column(String)
    end_time: Mapped[str] = mapped_column(String)
    tokens_input: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    raw_attributes: Mapped[str] = mapped_column(Text, default="{}")
    semconv_version: Mapped[str] = mapped_column(String, default="unknown")


class CanonicalStepRow(Base):
    __tablename__ = "canonical_steps"

    step_id: Mapped[str] = mapped_column(String, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String, ForeignKey("trajectories.trace_id"))
    raw_span_ids_json: Mapped[str] = mapped_column(Text)
    node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String, nullable=True)
    step_type: Mapped[str] = mapped_column(String)
    side_effect: Mapped[str] = mapped_column(String)
    attempt_index: Mapped[int] = mapped_column(Integer, default=0)
    loop_iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_hash: Mapped[str] = mapped_column(String)
    output_hash: Mapped[str] = mapped_column(String)
    mapping_confidence: Mapped[float] = mapped_column(Float)
    tokens_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_time: Mapped[str] = mapped_column(String)
    end_time: Mapped[str] = mapped_column(String)


class AttributionRunRow(Base):
    __tablename__ = "attribution_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String, ForeignKey("trajectories.trace_id"))
    config_hash: Mapped[str] = mapped_column(String)
    code_version: Mapped[str] = mapped_column(String)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    layers_used: Mapped[str] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )


class StepAttributionRow(Base):
    __tablename__ = "step_attributions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("attribution_runs.run_id"))
    step_id: Mapped[str] = mapped_column(String, ForeignKey("canonical_steps.step_id"))
    layer: Mapped[int] = mapped_column(Integer)
    quality_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_lo: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_hi: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict: Mapped[str] = mapped_column(String)
    evidence_json: Mapped[str] = mapped_column(Text, default="[]")
    calibration_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class PruneCandidateRow(Base):
    __tablename__ = "prune_candidates"

    candidate_id: Mapped[str] = mapped_column(String, primary_key=True)
    target_type: Mapped[str] = mapped_column(String)
    target_id: Mapped[str] = mapped_column(String)
    savings_json: Mapped[str] = mapped_column(Text)
    validation_plan_json: Mapped[str] = mapped_column(Text)
    decision_status: Mapped[str] = mapped_column(String, default="candidate")
    validated_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: Mapped[str | None] = mapped_column(String, nullable=True)


class CohortStatRow(Base):
    __tablename__ = "cohort_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    step_type: Mapped[str] = mapped_column(String)
    agent_name: Mapped[str] = mapped_column(String)
    agent_version: Mapped[str | None] = mapped_column(String, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String, nullable=True)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    lift_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    lift_ci_lo: Mapped[float | None] = mapped_column(Float, nullable=True)
    lift_ci_hi: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[str | None] = mapped_column(String, nullable=True)


class MarkovModelRow(Base):
    __tablename__ = "markov_models"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String)
    framework: Mapped[str] = mapped_column(String)
    model_blob: Mapped[bytes] = mapped_column()
    training_count: Mapped[int] = mapped_column(Integer, default=0)
    held_out_metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )
