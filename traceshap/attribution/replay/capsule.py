from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RecordedIO:
    """A single recorded tool invocation captured during a live trace run."""

    step_id: str
    tool_name: str
    input_hash: str
    input_data: dict
    output_data: dict
    side_effect_class: str
    timestamp: datetime


@dataclass
class EnvironmentSnapshot:
    """Snapshot of the runtime environment at the time of recording."""

    python_version: str
    package_versions: dict[str, str]
    env_vars_hash: str
    framework_version: str
    timestamp: datetime


@dataclass
class ReplayCapsule:
    """Container holding all recorded I/O needed to replay a trace deterministically.

    Used by Layer 3 (Replay SHAP) to substitute real tool calls with their
    previously observed outputs, enabling counterfactual perturbation without
    side effects.
    """

    capsule_id: str
    trace_id: str
    created_at: datetime
    model_id: str
    model_config: dict
    recorded_ios: list[RecordedIO]
    environment_snapshot: EnvironmentSnapshot | None = None

    def lookup_io(self, tool_name: str, input_hash: str) -> RecordedIO | None:
        """Return the RecordedIO that exactly matches *tool_name* AND *input_hash*.

        Returns ``None`` if no match is found.
        """
        for rio in self.recorded_ios:
            if rio.tool_name == tool_name and rio.input_hash == input_hash:
                return rio
        return None
