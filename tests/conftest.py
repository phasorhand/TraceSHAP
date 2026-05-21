import pytest


@pytest.fixture
def sample_trace_id() -> str:
    return "trace-abc-123"


@pytest.fixture
def sample_span_id() -> str:
    return "span-001"
