import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.core.database import create_tables
from app.services.ml_service import load_models, get_loaded_models
from app.api import auth_routes, orders_routes, optimize_routes, inventory_routes, analytics_routes, products_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    load_models()
    yield

app = FastAPI(title="PackAI", version="1.0.0", lifespan=lifespan)

# Most permissive CORS — allows every origin, every method
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manual CORS headers on every response as backup
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Handle OPTIONS preflight manually
@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(
        content={"ok": True},
        headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

app.include_router(auth_routes.router)
app.include_router(orders_routes.router)
app.include_router(optimize_routes.router)
app.include_router(inventory_routes.router)
app.include_router(analytics_routes.router)
app.include_router(products_routes.router)

@app.get("/")
def root():
    return {"status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "models": get_loaded_models()}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
