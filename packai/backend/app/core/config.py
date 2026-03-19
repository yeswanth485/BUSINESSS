"""
Core configuration — reads from environment variables
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "PackAI - AI Packaging Automation"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql://packai:packai@localhost:5432/packaidb"

    # JWT
    SECRET_KEY: str = "change-this-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # ML
    MODEL_PATH: str = "ml_engine/models/packaging_model.pkl"
    ENSEMBLE_MODEL_PATH: str = "ml_engine/models/ensemble_model.pkl"

    # Shipping rates (INR per kg, by zone)
    SHIPPING_RATES: dict = {
        "zone_a": 45.0,
        "zone_b": 55.0,
        "zone_c": 65.0,
        "zone_d": 80.0,
        "default": 55.0,
    }

    # Dimensional weight divisor (standard logistics)
    DIM_WEIGHT_DIVISOR: float = 5000.0

    # Shopify
    SHOPIFY_API_KEY: str = ""
    SHOPIFY_API_SECRET: str = ""
    SHOPIFY_STORE_URL: str = ""

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
