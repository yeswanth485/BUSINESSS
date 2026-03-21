import os, time, threading, hashlib, uvicorn
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
            "CREATE TABLE products (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), name VARCHAR(255) NOT NULL, length FLOAT NOT NULL, width FLOAT NOT NULL, height FLOAT NOT NULL, weight FLOAT NOT NULL, category VARCHAR(100) DEFAULT NULL, fragility_level VARCHAR(20) DEFAULT 'standard', stackable BOOLEAN DEFAULT TRUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE orders (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), destination_zone VARCHAR(50) DEFAULT 'default', status VARCHAR(50) DEFAULT 'pending', priority VARCHAR(20) DEFAULT 'cost', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE order_items (id SERIAL PRIMARY KEY, order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE, product_id INTEGER REFERENCES products(id), quantity INTEGER NOT NULL DEFAULT 1)",
            "CREATE TABLE box_inventory (id SERIAL PRIMARY KEY, box_type VARCHAR(100) UNIQUE NOT NULL, length FLOAT NOT NULL, width FLOAT NOT NULL, height FLOAT NOT NULL, max_weight FLOAT NOT NULL, cost FLOAT NOT NULL, quantity_available INTEGER DEFAULT 100, suitable_fragile BOOLEAN DEFAULT FALSE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
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

# ── Inline auth routes — bypasses old security.py completely ─────────────────
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

SECRET_KEY = os.environ.get("SECRET_KEY","packai-secret-key-2025")
ALGORITHM  = "HS256"
pwd_ctx    = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2     = OAuth2PasswordBearer(tokenUrl="/auth/login")

def prep_pass(p: str) -> str:
    """SHA256 first — removes bcrypt 72 byte limit completely"""
    return hashlib.sha256(p.encode()).hexdigest()

def hash_pw(p: str) -> str:
    return pwd_ctx.hash(prep_pass(p))

def verify_pw(p: str, h: str) -> bool:
    return pwd_ctx.verify(prep_pass(p), h)

def make_token(user_id: int) -> str:
    exp = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode({"sub": str(user_id), "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

class RegIn(BaseModel):
    email: EmailStr
    password: str
    full_name: str = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

auth_router = APIRouter(prefix="/auth", tags=["Auth"])

@auth_router.post("/register", status_code=201)
def register(payload: RegIn, db: Session = Depends(lambda: next(get_db()))):
    from sqlalchemy import text as t
    try:
        existing = db.execute(t("SELECT id FROM users WHERE email=:e"), {"e": payload.email}).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed = hash_pw(payload.password)
        result = db.execute(
            t("INSERT INTO users (email, password_hash, full_name) VALUES (:e,:p,:n) RETURNING id, email, full_name, created_at"),
            {"e": payload.email, "p": hashed, "n": payload.full_name}
        )
        db.commit()
        row = result.fetchone()
        return {"id": row[0], "email": row[1], "full_name": row[2], "created_at": str(row[3])}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@auth_router.post("/login")
def login(payload: LoginIn, db: Session = Depends(lambda: next(get_db()))):
    from sqlalchemy import text as t
    try:
        row = db.execute(
            t("SELECT id, password_hash FROM users WHERE email=:e"),
            {"e": payload.email}
        ).fetchone()
        if not row or not verify_pw(payload.password, row[1]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = make_token(row[0])
        return {"access_token": token, "token_type": "bearer", "user_id": row[0], "email": payload.email}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(auth_router)

# ── DB state ──────────────────────────────────────────────────────────────────
_db_ready, _SessionLocal, _engine = False, None, None

def _init_db():
    global _db_ready, _SessionLocal, _engine
    url = os.environ.get("DATABASE_URL","")
    if url.startswith("postgres://"): url = url.replace("postgres://","postgresql://",1)
    if not url: return
    for attempt in range(5):
        try:
            eng = create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=5, pool_recycle=300, connect_args={"connect_timeout":15})
            with eng.connect() as c: c.execute(text("SELECT 1"))
            _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=eng)
            _engine = eng
            _db_ready = True
            print(f"[DB] Ready")
            return
        except Exception as e:
            print(f"[DB] Attempt {attempt+1}: {e}")
            time.sleep(4)

threading.Thread(target=_init_db, daemon=True).start()

def get_db():
    deadline = time.time() + 20
    while not _db_ready:
        if time.time() > deadline:
            raise RuntimeError("DB not ready yet — try again in 10s")
        time.sleep(0.3)
    db = _SessionLocal()
    try: yield db
    finally: db.close()

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
