import os, time, threading, hashlib, hmac, uvicorn
from fastapi import FastAPI, Request, APIRouter, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

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
def root(): return {"status":"running","version":"1.0.0","frontend":"https://ai-packaging-automation-service.netlify.app"}

@app.get("/health")
def health():
    ml_ok = False
    try:
        from app.services.ml_service import is_ml_available
        ml_ok = is_ml_available()
    except Exception:
        pass
    return {
        "status":       "ok",
        "version":      "1.0.0",
        "db":           "ready" if _db_ready else "starting",
        "ml_available": ml_ok,
        "engine":       "ml_hybrid" if ml_ok else "rule_based",
    }

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
            "CREATE TABLE users (id SERIAL PRIMARY KEY, email VARCHAR(255) UNIQUE NOT NULL, password_hash TEXT NOT NULL, full_name VARCHAR(255) DEFAULT NULL, is_active BOOLEAN DEFAULT TRUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE products (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), name VARCHAR(255), length FLOAT, width FLOAT, height FLOAT, weight FLOAT, category VARCHAR(100), fragility_level VARCHAR(20) DEFAULT 'standard', stackable BOOLEAN DEFAULT TRUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE orders (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), destination_zone VARCHAR(50) DEFAULT 'default', status VARCHAR(50) DEFAULT 'pending', priority VARCHAR(20) DEFAULT 'cost', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE order_items (id SERIAL PRIMARY KEY, order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE, product_id INTEGER REFERENCES products(id), quantity INTEGER DEFAULT 1)",
            "CREATE TABLE box_inventory (id SERIAL PRIMARY KEY, box_type VARCHAR(100) UNIQUE NOT NULL, length FLOAT, width FLOAT, height FLOAT, max_weight FLOAT, cost FLOAT, quantity_available INTEGER DEFAULT 100, suitable_fragile BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE packaging_plans (id SERIAL PRIMARY KEY, order_id INTEGER REFERENCES orders(id), total_cost FLOAT, efficiency_score FLOAT, decision_reason TEXT, decision_engine VARCHAR(50) DEFAULT 'rule_based', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE packaging_plan_items (id SERIAL PRIMARY KEY, packaging_plan_id INTEGER REFERENCES packaging_plans(id) ON DELETE CASCADE, box_type VARCHAR(100), items JSONB, box_cost FLOAT, shipping_cost FLOAT, efficiency_score FLOAT)",
            "CREATE TABLE analytics_summary (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), total_orders INTEGER DEFAULT 0, total_cost_saved FLOAT DEFAULT 0, avg_efficiency FLOAT DEFAULT 0, waste_percentage FLOAT DEFAULT 0, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        ]
        with eng.connect() as c:
            for sql in sqls:
                c.execute(text(sql))
            c.commit()
        return {"status":"success","message":"Tables recreated. Now try /auth/register"}
    except Exception as e:
        return {"status":"error","message":str(e)}

# ── Password hashing — PBKDF2 stdlib, NO bcrypt, NO size limits ──────────────
def hash_password(password: str) -> str:
    salt = os.urandom(32)
    key  = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 260000)
    return salt.hex() + ':' + key.hex()

def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(':')
        salt = bytes.fromhex(salt_hex)
        key  = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 260000)
        return hmac.compare_digest(key.hex(), key_hex)
    except Exception:
        return False

# ── JWT ───────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "packai-secret-2025-xk92mz")
ALGORITHM  = "HS256"

def make_token(user_id: int, email: str) -> str:
    exp = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode({"sub": str(user_id), "email": email, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

# ── DB state ──────────────────────────────────────────────────────────────────
_db_ready, _SessionLocal = False, None

def _init_db():
    global _db_ready, _SessionLocal
    url = os.environ.get("DATABASE_URL","")
    if url.startswith("postgres://"): url = url.replace("postgres://","postgresql://",1)
    if not url: return
    for i in range(5):
        try:
            eng = create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=5, pool_recycle=300, connect_args={"connect_timeout":15})
            with eng.connect() as c: c.execute(text("SELECT 1"))
            _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=eng)
            _db_ready = True
            print("[DB] Ready")
            # Run column migrations for new cost tracking fields
            _run_migrations(eng)
            # Load ML models once at startup
            try:
                from app.services import ml_service as _ml
                _ml.load_models()
            except Exception as me:
                print(f"[ML] Load skipped: {me}")
            return
        except Exception as e:
            print(f"[DB] Attempt {i+1}: {e}")
            time.sleep(4)

def _run_migrations(eng):
    """Safe ADD COLUMN migrations — never drops data."""
    migrations = [
        "ALTER TABLE packaging_plans ADD COLUMN IF NOT EXISTS baseline_cost FLOAT DEFAULT NULL",
        "ALTER TABLE packaging_plans ADD COLUMN IF NOT EXISTS optimized_cost FLOAT DEFAULT NULL",
        "ALTER TABLE packaging_plans ADD COLUMN IF NOT EXISTS savings FLOAT DEFAULT NULL",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS sku VARCHAR(100) DEFAULT NULL",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS channel VARCHAR(50) DEFAULT NULL",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS pincode VARCHAR(10) DEFAULT NULL",
        "ALTER TABLE orders ADD COLUMN IF NOT EXISTS destination_zone VARCHAR(50) DEFAULT 'zone_b'",
    ]
    with eng.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                pass
    print("[DB] Migrations applied")

threading.Thread(target=_init_db, daemon=True).start()

def get_db():
    deadline = time.time() + 20
    while not _db_ready:
        if time.time() > deadline:
            raise RuntimeError("DB not ready — try again in 10 seconds")
        time.sleep(0.3)
    db = _SessionLocal()
    try: yield db
    finally: db.close()

# ── Auth routes ───────────────────────────────────────────────────────────────
class RegIn(BaseModel):
    email: EmailStr
    password: str
    full_name: str = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

auth_router = APIRouter(prefix="/auth", tags=["Auth"])

@auth_router.post("/register", status_code=201)
def register(payload: RegIn):
    db = next(get_db())
    try:
        existing = db.execute(text("SELECT id FROM users WHERE email=:e"), {"e": payload.email}).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed = hash_password(payload.password)
        row = db.execute(
            text("INSERT INTO users (email, password_hash, full_name) VALUES (:e,:p,:n) RETURNING id, email, full_name, created_at"),
            {"e": payload.email, "p": hashed, "n": payload.full_name}
        ).fetchone()
        db.commit()
        return {"id": row[0], "email": row[1], "full_name": row[2], "created_at": str(row[3])}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@auth_router.post("/login")
def login(payload: LoginIn):
    db = next(get_db())
    try:
        row = db.execute(
            text("SELECT id, password_hash FROM users WHERE email=:e"),
            {"e": payload.email}
        ).fetchone()
        if not row or not verify_password(payload.password, row[1]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = make_token(row[0], payload.email)
        return {"access_token": token, "token_type": "bearer", "user_id": row[0], "email": payload.email}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

app.include_router(auth_router)

# ── Other routes ──────────────────────────────────────────────────────────────
try:
    import app.core.database as _dbm
    _dbm.get_db = get_db
    from app.api import orders_routes, optimize_routes, inventory_routes, analytics_routes, products_routes
    app.include_router(orders_routes.router)
    app.include_router(optimize_routes.router)
    app.include_router(inventory_routes.router)
    app.include_router(analytics_routes.router)
    app.include_router(products_routes.router)
    print("[startup] Other routes loaded")
except Exception as e:
    print(f"[startup] Route error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
