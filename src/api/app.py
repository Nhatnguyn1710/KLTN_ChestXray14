"""FastAPI backend entry point — thin composition layer.

Everything else lives in routes/, services/, startup.py, state.py.
"""
import os
import sys
import argparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.api import state
from src.api.startup import initialize
from src.api.routes import health, predict


# ─── Bootstrap config (CORS needs this before startup event runs) ──
_BOOTSTRAP_CONFIG = state.load_bootstrap_config()
_CORS_ORIGINS, _CORS_ALLOW_CREDENTIALS = state.resolve_cors_settings(_BOOTSTRAP_CONFIG)


# ─── FastAPI app ───────────────────────────────────────────────────
app = FastAPI(
    title="AI X-Ray Diagnosis API",
    description="API chẩn đoán bệnh lý phổi từ ảnh X-quang ngực",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=_CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static (gradcam PNGs + Vite dist/)
app.mount("/static", StaticFiles(directory=state.STATIC_FOLDER), name="static")

# Register routers
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", include_in_schema=False)
async def frontend_index():
    """Serve React frontend (Vite build output)."""
    if not os.path.isfile(state.REACT_APP_INDEX):
        raise HTTPException(
            status_code=404,
            detail="React frontend chưa được build. Chạy `npm run build` trong frontend/",
        )
    return FileResponse(state.REACT_APP_INDEX)


@app.on_event("startup")
async def startup_event():
    """Khởi tạo models khi server start."""
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    initialize(config_path)


# ─── CLI entry ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-Ray Diagnosis FastAPI Backend")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["CONFIG_PATH"] = args.config

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
