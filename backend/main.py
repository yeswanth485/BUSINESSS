import os, time, threading, uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

app = FastAPI(title="PackAI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def cors_always(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as e:
        response = JSONResponse({"detail": str(e)}, status_code=500)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response

@app.options("/{path:path}")
async def preflight(path: str):
    return JSONResponse({}, headers={"Access-Control-Allow-Origin":"*","Access-Control-Allow-Headers":"*","Access-Control-Allow-Methods":"*"})

@app.get("/")
def root(): return {"status":"running","version":"1.0.0"}

@app.get("/health")
def health(): return {"status":"ok","version":"1.0.0","db":"ready" if _db_ready else "starting"}

# ── One-time database fix route ───────────────────────────────────────────────
@app.get("/fix-db")
def fix_db():
    try:
        url = os.environ.get("DATABASE_URL","")
        if url.startswith("postgres://"): url = url.replace("postgres://","postgresql://",1)
        eng = create_engine(url)
        sqls = [
            "DROP TABLE IF EXISTS packaging_plan_items CASCADE",
            "DROP TABLE IF EXISTS packaging_plans CASCADE",
            "DROP TABLE IF EXISTS order_items CASCADE",
            "DROP TABLE IF EXISTS orders CASCADE",
            "DROP TABLE IF EXISTS products CASCADE",
            "DROP TABLE IF EXISTS box_inventory CASCADE",
            "DROP TABLE IF EXISTS analytics_summary CASCADE",
            "DROP TABLE IF EXISTS users CASCADE",
            """CREATE TABLE users (id SERIAL PRIMARY KEY, email VARCHAR(255) UNIQUE NOT NULL, password_hash TEXT NOT NULL, full_name VARCHAR(255) DEFAULT NULL, is_active BOOLEAN DEFAULT TRUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
            """CREATE TABLE products (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), name VARCHAR(255) NOT NULL, length FLOAT NOT NULL, width FLOAT NOT NULL, height FLOAT NOT NULL, weight FLOAT NOT NULL, category VARCHAR(100) DEFAULT NULL, fragility_level VARCHAR(20) DEFAULT 'standard', stackable BOOLEAN DEFAULT TRUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
            """CREATE TABLE orders (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), destination_zone VARCHAR(50) DEFAULT 'default', status VARCHAR(50) DEFAULT 'pending', priority VARCHAR(20) DEFAULT 'cost', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
            """CREATE TABLE order_items (id SERIAL PRIMARY KEY, order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE, product_id INTEGER REFERENCES products(id), quantity INTEGER NOT NULL DEFAULT 1)""",
            """CREATE TABLE box_inventory (id SERIAL PRIMARY KEY, box_type VARCHAR(100) UNIQUE NOT NULL, length FLOAT NOT NULL, width FLOAT NOT NULL, height FLOAT NOT NULL, max_weight FLOAT NOT NULL, cost FLOAT NOT NULL, quantity_available INTEGER DEFAULT 100, suitable_fragile BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
            """CREATE TABLE packaging_plans (id SERIAL PRIMARY KEY, order_id INTEGER REFERENCES orders(id), total_cost FLOAT, efficiency_score FLOAT, decision_reason TEXT, decision_engine VARCHAR(50) DEFAULT 'rule_based', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
            """CREATE TABLE packaging_plan_items (id SERIAL PRIMARY KEY, packaging_plan_id INTEGER REFERENCES packaging_plans(id) ON DELETE CASCADE, box_type VARCHAR(100), items JSONB, box_cost FLOAT, shipping_cost FLOAT, efficiency_score FLOAT)""",
            """CREATE TABLE analytics_summary (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), total_orders INTEGER DEFAULT 0, total_cost_saved FLOAT DEFAULT 0, avg_efficiency FLOAT DEFAULT 0, waste_percentage FLOAT DEFAULT 0, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
        ]
        with eng.connect() as c:
            for sql in sqls:
                c.execute(text(sql))
            c.commit()
        return {"status":"success","message":"All tables recreated. Now try /auth/register"}
    except Exception as e:
        return {"status":"error","message":str(e)}

# ── DB state ──────────────────────────────────────────────────────────────────
_db_ready, _SessionLocal = False, None

def _init_db():
    global _db_ready, _SessionLocal
    url = os.environ.get("DATABASE_URL","")
    if url.startswith("postgres://"): url = url.replace("postgres://","postgresql://",1)
    if not url: print("[DB] No DATABASE_URL"); return
    for attempt in range(5):
        try:
            eng = create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=5, pool_recycle=300, connect_args={"connect_timeout":15})
            with eng.connect() as c: c.execute(text("SELECT 1"))
            _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=eng)
            from app.models.models import Base
            Base.metadata.create_all(bind=eng)
            _db_ready = True
            print(f"[DB] Ready after {attempt+1} attempt(s)")
            return
        except Exception as e:
            print(f"[DB] Attempt {attempt+1}/5: {e}")
            time.sleep(4)
    print("[DB] Failed to connect")

threading.Thread(target=_init_db, daemon=True).start()

def get_db():
    deadline = time.time() + 20
    while not _db_ready:
        if time.time() > deadline: raise RuntimeError("Database starting up — please try again in 10 seconds")
        time.sleep(0.3)
    db = _SessionLocal()
    try: yield db
    finally: db.close()

try:
    import app.core.database as _dbm
    _dbm.get_db = get_db
except: pass

try:
    from app.api import auth_routes, orders_routes, optimize_routes, inventory_routes, analytics_routes, products_routes
    app.include_router(auth_routes.router)
    app.include_router(orders_routes.router)
    app.include_router(optimize_routes.router)
    app.include_router(inventory_routes.router)
    app.include_router(analytics_routes.router)
    app.include_router(products_routes.router)
    print("[startup] Routes loaded")
except Exception as e:
    print(f"[startup] Route error: {e}")

threading.Thread(target=lambda: __import__('app.services.ml_service', fromlist=['load_models']).load_models(), daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
