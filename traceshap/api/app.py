from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from traceshap.api.deps import set_backend
from traceshap.api.routes_traces import router as traces_router
from traceshap.api.routes_attribution import router as attribution_router
from traceshap.api.routes_pruning import router as pruning_router
from traceshap.storage.sqlite import SQLiteBackend

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend" / "dist"


def create_app(backend: SQLiteBackend | None = None) -> FastAPI:
    app = FastAPI(title="TraceSHAP", version="0.1.0")

    if backend is not None:
        set_backend(backend)

    app.include_router(traces_router)
    app.include_router(attribution_router)
    app.include_router(pruning_router)

    if FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

        @app.get("/{path:path}")
        async def serve_spa(path: str):
            file_path = FRONTEND_DIR / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    return app
