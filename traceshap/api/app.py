from fastapi import FastAPI

from traceshap.api.deps import set_backend
from traceshap.api.routes_traces import router as traces_router
from traceshap.api.routes_attribution import router as attribution_router
from traceshap.api.routes_pruning import router as pruning_router
from traceshap.storage.sqlite import SQLiteBackend


def create_app(backend: SQLiteBackend | None = None) -> FastAPI:
    app = FastAPI(title="TraceSHAP", version="0.1.0")

    if backend is not None:
        set_backend(backend)

    app.include_router(traces_router)
    app.include_router(attribution_router)
    app.include_router(pruning_router)

    return app
