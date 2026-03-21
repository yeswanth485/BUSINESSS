"""
PackAI — main.py FINAL VERSION
Solves both problems:
  1. Port binds instantly (DB runs in background thread)
  2. Requests wait for DB before executing (no 500 errors)
  3. CORS headers on every response including errors
"""
import os
import time
import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── App created first, before any DB import ──────────────────────────────────
app = FastAPI(title="PackAI", version="1.0.0")

# ── CORS middleware — registered first ───────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Force CORS headers on every response including 500 errors ────────────────
@app.middleware("http")
async def cors_always(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as e:
        response = JSONResponse({"detail": str(e)}, status_code=500)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response

# ── Preflight OPTIONS handler ─────────────────────────────────────────────────
@app.options("/{path:path}")
async def preflight(path: str):
    return JSONResponse({}, headers={
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "*",
    })

# ── Health — no DB needed, answers immediately ───────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "version": "1.0.0", "db": "ready" if _db_ready else "starting"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "db": "ready" if _db_ready else "starting"}

# ── Global DB state ───────────────────────────────────────────────────────────
_db_ready    = False
_db_session  = None   # sessionmaker instance

def _get_db_url():
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

# ── Background DB initializer ─────────────────────────────────────────────────
def _init_db():
    global _db_ready, _db_session
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    url = _get_db_url()
    if not url:
        print("[DB] ERROR: DATABASE_URL environment variable not set")
        return

    for attempt in range(1, 6):
        try:
            engine = create_engine(
                url,
                pool_pre_ping  = True,
                pool_size      = 3,
                max_overflow   = 5,
                pool_recycle   = 300,
                connect_args   = {"connect_timeout": 15},
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"[DB] Connected on attempt {attempt}")

            _db_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

            from app.models.models import Base
            Base.metadata.create_all(bind=engine)
            print("[DB] Tables verified")

            _db_ready = True
            return

        except Exception as e:
            print(f"[DB] Attempt {attempt}/5 failed: {e}")
            if attempt < 5:
                time.sleep(4)

    print("[DB] FATAL: Could not connect after 5 attempts")

# Start DB in background thread immediately
threading.Thread(target=_init_db, daemon=True).start()

# ── get_db — waits for DB if not yet ready ───────────────────────────────────
def get_db():
    deadline = time.time() + 20   # wait up to 20 seconds
    while not _db_ready:
        if time.time() > deadline:
            raise RuntimeError("Database is still starting up — please try again in 10 seconds")
        time.sleep(0.3)
    db = _db_session()
    try:
        yield db
    finally:
        db.close()

# ── Inject our get_db into all modules that use it ───────────────────────────
try:
    import app.core.database as _db_module
    _db_module.get_db = get_db
    print("[startup] get_db injected")
except Exception as e:
    print(f"[startup] get_db injection error: {e}")

# ── Load all API routes ───────────────────────────────────────────────────────
try:
    from app.api import (
        auth_routes, orders_routes, optimize_routes,
        inventory_routes, analytics_routes, products_routes,
    )
    app.include_router(auth_routes.router)
    app.include_router(orders_routes.router)
    app.include_router(optimize_routes.router)
    app.include_router(inventory_routes.router)
    app.include_router(analytics_routes.router)
    app.include_router(products_routes.router)
    print("[startup] All routes loaded")
except Exception as e:
    print(f"[startup] Route error: {e}")

# ── ML in background (optional) ──────────────────────────────────────────────
def _init_ml():
    try:
        from app.services.ml_service import load_models
        load_models()
        print("[ML] Models loaded")
    except Exception as e:
        print(f"[ML] Skipped: {e}")

threading.Thread(target=_init_ml, daemon=True).start()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[PackAI] Starting on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
