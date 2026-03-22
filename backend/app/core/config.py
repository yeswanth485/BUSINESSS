"""
Core configuration — reads from environment variables.
All types are simple (str, int, float, bool) so pydantic-settings
can parse them from Render environment variables without errors.
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
    SECRET_KEY: str = "change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # ML model paths
    MODEL_PATH: str = "ml_engine/models/packaging_model.pkl"
    ENSEMBLE_MODEL_PATH: str = "ml_engine/models/ensemble_model.pkl"

    # Dimensional weight divisor — standard India logistics
    DIM_WEIGHT_DIVISOR: float = 5000.0

    # Shipping rates INR per kg — one flat var per zone (safe for Render env vars)
    RATE_ZONE_A: float = 45.0
    RATE_ZONE_B: float = 55.0
    RATE_ZONE_C: float = 65.0
    RATE_ZONE_D: float = 80.0
    RATE_DEFAULT: float = 55.0

    # Shopify (optional)
    SHOPIFY_API_KEY: str = ""
    SHOPIFY_API_SECRET: str = ""
    SHOPIFY_STORE_URL: str = ""

    class Config:
        env_file = ".env"

    def get_shipping_rate(self, zone: str) -> float:
        """Return shipping rate for a given zone string."""
        zone_map = {
            "zone_a":  self.RATE_ZONE_A,
            "zone_b":  self.RATE_ZONE_B,
            "zone_c":  self.RATE_ZONE_C,
            "zone_d":  self.RATE_ZONE_D,
            "default": self.RATE_DEFAULT,
        }
        return zone_map.get(zone.lower(), self.RATE_DEFAULT)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
