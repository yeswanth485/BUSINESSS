"""
PackAI — AI Packaging Automation Platform
FastAPI application entry point
"""
import os
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

# ── CORS — allow ALL origins so any Vercel URL works ─────────────────────────
# This is the safest fix — allows your main URL, preview URLs, and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers     = ["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(auth_routes.router)
app.include_router(orders_routes.router)
app.include_router(optimize_routes.router)
app.include_router(inventory_routes.router)
app.include_router(analytics_routes.router)
app.include_router(products_routes.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "PackAI API is running",
        "docs":    "/docs",
        "health":  "/health",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":  "ok",
        "version": "1.0.0",
        "models":  get_loaded_models(),
    }


# ── Entry point — Render reads PORT from environment ─────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[startup] Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host   = "0.0.0.0",
        port   = port,
        reload = False,
    )
