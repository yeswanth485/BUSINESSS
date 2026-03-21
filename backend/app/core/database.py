"""
Database connection — with connection pooling tuned for Render free tier
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import get_settings

settings = get_settings()

# Fix postgres:// -> postgresql:// (Render gives postgres:// which SQLAlchemy 2.x rejects)
db_url = settings.DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    db_url,
    pool_pre_ping   = True,
    pool_size       = 3,      # small pool for free tier
    max_overflow    = 5,
    pool_recycle    = 300,    # recycle connections every 5 min
    pool_timeout    = 30,
    connect_args    = {"connect_timeout": 10},
    echo            = False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    try:
        # Test connection first
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DB] Connection successful")
        Base.metadata.create_all(bind=engine)
        print("[DB] Tables created/verified")
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
        raise
