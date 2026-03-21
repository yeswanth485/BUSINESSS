import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── App created BEFORE any database/ML imports ────────────────────────────────
app = FastAPI(title="PackAI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def cors_backup(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(content={}, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
            "Access-Control-Allow-Headers": "*",
        })
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# ── Health check registered FIRST so Render finds the port ───────────────────
@app.get("/")
def root():
    return {"status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

# ── Import routes AFTER app is created ───────────────────────────────────────
try:
    from app.api import auth_routes, orders_routes, optimize_routes
    from app.api import inventory_routes, analytics_routes, products_routes
    app.include_router(auth_routes.router)
    app.include_router(orders_routes.router)
    app.include_router(optimize_routes.router)
    app.include_router(inventory_routes.router)
    app.include_router(analytics_routes.router)
    app.include_router(products_routes.router)
    print("[startup] All routes loaded")
except Exception as e:
    print(f"[startup] Route loading error: {e}")

# ── Database and ML loaded AFTER server binds to port ────────────────────────
@app.on_event("startup")
async def startup():
    try:
        from app.core.database import create_tables
        create_tables()
        print("[startup] Database tables ready")
    except Exception as e:
        print(f"[startup] Database error: {e}")

    try:
        from app.services.ml_service import load_models
        load_models()
        print("[startup] ML models loaded")
    except Exception as e:
        print(f"[startup] ML error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[PackAI] Starting on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
