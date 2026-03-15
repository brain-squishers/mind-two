from abc import ABC, abstractmethod


class QueryInput(ABC):
    @abstractmethod
    def poll_query(self) -> str | None:
        raise NotImplementedError

    def release(self) -> None:
        return None


class StaticQueryInput(QueryInput):
    def __init__(self, query: str):
        self._query = query
        self._consumed = False

    def poll_query(self) -> str | None:
        if self._consumed or not self._query:
            return None
        self._consumed = True
        return self._query
