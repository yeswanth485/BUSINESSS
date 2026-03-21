from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

def _get_url():
    url = os.environ.get("DATABASE_URL", "postgresql://packai:packai@localhost:5432/packaidb")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

engine = create_engine(
    _get_url(),
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=5,
    pool_recycle=300,
    connect_args={"connect_timeout": 15},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create tables AND run column migrations for existing databases."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DB] Connection OK")
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
        raise

    # Create all tables from models
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables created/verified")

    # Run migrations for existing databases that are missing columns
    _run_migrations()


def _run_migrations():
    """Safe migrations — adds missing columns, never drops anything."""
    migrations = [
        # Fix users table — add columns that may be missing in old DBs
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255) DEFAULT NULL;",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;",
        # Fix products table
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS category VARCHAR(100) DEFAULT NULL;",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS fragility_level VARCHAR(20) DEFAULT 'standard';",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS stackable BOOLEAN DEFAULT TRUE;",
        # Fix orders table
        "ALTER TABLE orders ADD COLUMN IF NOT EXISTS destination_zone VARCHAR(50) DEFAULT 'default';",
        "ALTER TABLE orders ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'pending';",
        "ALTER TABLE orders ADD COLUMN IF NOT EXISTS priority VARCHAR(20) DEFAULT 'cost';",
        # Fix box_inventory table
        "ALTER TABLE box_inventory ADD COLUMN IF NOT EXISTS suitable_fragile BOOLEAN DEFAULT FALSE;",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                pass  # Column already exists or table doesn't exist yet — both are fine
    print("[DB] Migrations applied")
