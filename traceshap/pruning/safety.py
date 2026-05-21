from traceshap.models.step import CanonicalStep
from traceshap.models.enums import StepType, Verdict
from traceshap.models.outcome import StepAttribution
from traceshap.config import PruningConfig

PROTECTED_STEP_TYPES = frozenset({StepType.VALIDATION})


def is_protected_step(step: CanonicalStep) -> bool:
    return step.step_type in PROTECTED_STEP_TYPES


def is_first_or_last(step_id: str, steps: list[CanonicalStep]) -> bool:
    if not steps:
        return False
    return step_id == steps[0].step_id or step_id == steps[-1].step_id


def classify_step(
    attr: StepAttribution,
    step: CanonicalStep,
    config: PruningConfig,
    is_first_last: bool,
) -> Verdict:
    if attr.confidence is None or attr.quality_delta is None:
        return Verdict.INSUFFICIENT_EVIDENCE

    if is_protected_step(step):
        return Verdict.KEEP

    if is_first_last and config.protect_first_last:
        return Verdict.KEEP

    if (attr.confidence.lower >= -config.prune_epsilon
            and (attr.cost_delta or 0) > 0):
        return Verdict.PRUNE_CANDIDATE

    if attr.confidence.lower < -config.keep_threshold:
        return Verdict.KEEP

    return Verdict.REVIEW
