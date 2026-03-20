"""
PackAI — AI Packaging Automation Platform
FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.database import create_tables
from app.services.ml_service import load_models, get_loaded_models
from app.api import (
    auth_routes,
    orders_routes,
    optimize_routes,
    inventory_routes,
    analytics_routes,
    products_routes,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables + load all ML models into memory."""
    print("[startup] Creating database tables...")
    create_tables()

    print("[startup] Loading ML models...")
    load_models()
    print(f"[startup] Models loaded: {get_loaded_models()}")

    yield

    print("[shutdown] PackAI shutting down.")


app = FastAPI(
    title       = "PackAI — AI Packaging Automation",
    description = "Hybrid rule-based + ML packaging optimizer for Indian ecommerce warehouses",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000","https://businesss-azure.vercel.app/"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(auth_routes.router)
app.include_router(orders_routes.router)
app.include_router(optimize_routes.router)
app.include_router(inventory_routes.router)
app.include_router(analytics_routes.router)
app.include_router(products_routes.router)


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":  "ok",
        "version": "1.0.0",
        "models":  get_loaded_models(),
    }
