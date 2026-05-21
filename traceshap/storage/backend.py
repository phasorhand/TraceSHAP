from abc import ABC, abstractmethod

from traceshap.models.trajectory import Trajectory
from traceshap.models.outcome import StepAttribution
from traceshap.models.enums import DecisionStatus


class QueryFilter:
    def __init__(
        self,
        agent_name: str | None = None,
        agent_version: str | None = None,
        task_type: str | None = None,
        framework: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ):
        self.agent_name = agent_name
        self.agent_version = agent_version
        self.task_type = task_type
        self.framework = framework
        self.limit = limit
        self.offset = offset


class CohortFilter:
    def __init__(
        self,
        agent_version: str | None = None,
        task_type: str | None = None,
    ):
        self.agent_version = agent_version
        self.task_type = task_type


class StorageBackend(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        ...

    @abstractmethod
    async def save_trajectory(self, trajectory: Trajectory) -> None:
        ...

    @abstractmethod
    async def get_trajectory(self, trace_id: str) -> Trajectory | None:
        ...

    @abstractmethod
    async def query_trajectories(self, filters: QueryFilter) -> list[Trajectory]:
        ...

    @abstractmethod
    async def save_attribution_run(
        self, run_id: str, trace_id: str, config_hash: str,
        code_version: str, layers: list[int],
        attributions: list[StepAttribution],
    ) -> None:
        ...

    @abstractmethod
    async def update_candidate_status(
        self, candidate_id: str, status: DecisionStatus,
    ) -> None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
