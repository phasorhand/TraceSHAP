from abc import ABC, abstractmethod

from traceshap.models.span import TraceSHAPSpan


class SpanSource(ABC):
    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    async def poll(self) -> list[TraceSHAPSpan]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
