import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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
    """Drop and recreate all tables with correct schema."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("[DB] Connected")

    # Drop all old tables
    with engine.connect() as conn:
        conn.execute(text("""
            DROP TABLE IF EXISTS packaging_plan_items CASCADE;
            DROP TABLE IF EXISTS packaging_plans CASCADE;
            DROP TABLE IF EXISTS order_items CASCADE;
            DROP TABLE IF EXISTS orders CASCADE;
            DROP TABLE IF EXISTS products CASCADE;
            DROP TABLE IF EXISTS box_inventory CASCADE;
            DROP TABLE IF EXISTS analytics_summary CASCADE;
            DROP TABLE IF EXISTS users CASCADE;
        """))
        conn.commit()
    print("[DB] Old tables dropped")

    # Create all tables fresh with correct schema
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name VARCHAR(255) DEFAULT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE products (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                name VARCHAR(255) NOT NULL,
                length FLOAT NOT NULL,
                width FLOAT NOT NULL,
                height FLOAT NOT NULL,
                weight FLOAT NOT NULL,
                category VARCHAR(100) DEFAULT NULL,
                fragility_level VARCHAR(20) DEFAULT 'standard',
                stackable BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE orders (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                destination_zone VARCHAR(50) DEFAULT 'default',
                status VARCHAR(50) DEFAULT 'pending',
                priority VARCHAR(20) DEFAULT 'cost',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE order_items (
                id SERIAL PRIMARY KEY,
                order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
                product_id INTEGER REFERENCES products(id),
                quantity INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE box_inventory (
                id SERIAL PRIMARY KEY,
                box_type VARCHAR(100) UNIQUE NOT NULL,
                length FLOAT NOT NULL,
                width FLOAT NOT NULL,
                height FLOAT NOT NULL,
                max_weight FLOAT NOT NULL,
                cost FLOAT NOT NULL,
                quantity_available INTEGER DEFAULT 100,
                suitable_fragile BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE packaging_plans (
                id SERIAL PRIMARY KEY,
                order_id INTEGER REFERENCES orders(id),
                total_cost FLOAT,
                efficiency_score FLOAT,
                decision_reason TEXT,
                decision_engine VARCHAR(50) DEFAULT 'rule_based',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE packaging_plan_items (
                id SERIAL PRIMARY KEY,
                packaging_plan_id INTEGER REFERENCES packaging_plans(id) ON DELETE CASCADE,
                box_type VARCHAR(100),
                items JSONB,
                box_cost FLOAT,
                shipping_cost FLOAT,
                efficiency_score FLOAT
            );
            CREATE TABLE analytics_summary (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                total_orders INTEGER DEFAULT 0,
                total_cost_saved FLOAT DEFAULT 0,
                avg_efficiency FLOAT DEFAULT 0,
                waste_percentage FLOAT DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()
    print("[DB] All tables created with correct schema")
