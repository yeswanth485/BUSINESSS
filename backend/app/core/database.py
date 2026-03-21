"""
Database — lazy connection, only connects when first request arrives.
Does NOT connect at import time — this was causing port timeout.
"""
import os
import time
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

_engine       = None
_SessionLocal = None
_ready        = False


def _get_url():
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def _ensure_ready():
    """Connect and create tables on first use. Retries 3 times."""
    global _engine, _SessionLocal, _ready
    if _ready:
        return True

    url = _get_url()
    if not url:
        raise RuntimeError("DATABASE_URL not set in environment variables")

    for attempt in range(3):
        try:
            engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=3,
                max_overflow=5,
                pool_recycle=300,
                connect_args={"connect_timeout": 10},
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            _engine       = engine
            _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

            Base.metadata.create_all(bind=engine)
            _ready = True
            print(f"[DB] Connected on attempt {attempt+1}")
            return True

        except Exception as e:
            print(f"[DB] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(2)

    raise RuntimeError("Database connection failed after 3 attempts. Check DATABASE_URL in Render environment.")


def get_db():
    """FastAPI dependency — connects on first call, reuses after."""
    _ensure_ready()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Called from startup — ensures tables exist."""
    _ensure_ready()
