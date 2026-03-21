import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="PackAI", version="1.0.0")

# CORS — must be first middleware registered
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Backup CORS on every response including 500 errors
@app.middleware("http")
async def force_cors(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as e:
        response = JSONResponse({"detail": str(e)}, status_code=500)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response

@app.options("/{path:path}")
async def preflight(path: str):
    return JSONResponse({}, headers={
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "*",
    })

@app.get("/")
def root():
    return {"status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

# Load routes after CORS setup
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
    print(f"[startup] Route error: {e}")

@app.on_event("startup")
async def startup():
    try:
        from app.core.database import create_tables
        create_tables()
        print("[startup] Database tables ready")
    except Exception as e:
        print(f"[startup] DB error: {e}")
    try:
        from app.services.ml_service import load_models
        load_models()
        print("[startup] ML models loaded")
    except Exception as e:
        print(f"[startup] ML error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
