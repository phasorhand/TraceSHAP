from traceshap.storage.sqlite import SQLiteBackend

_backend: SQLiteBackend | None = None


def set_backend(backend: SQLiteBackend) -> None:
    global _backend
    _backend = backend


def get_backend() -> SQLiteBackend:
    if _backend is None:
        raise RuntimeError("Backend not initialized")
    return _backend
