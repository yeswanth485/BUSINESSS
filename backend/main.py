import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.database import create_tables
from app.services.ml_service import load_models, get_loaded_models
from app.api import auth_routes, orders_routes, optimize_routes, inventory_routes, analytics_routes, products_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    load_models()
    yield

app = FastAPI(title="PackAI", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_routes.router)
app.include_router(orders_routes.router)
app.include_router(optimize_routes.router)
app.include_router(inventory_routes.router)
app.include_router(analytics_routes.router)
app.include_router(products_routes.router)

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
