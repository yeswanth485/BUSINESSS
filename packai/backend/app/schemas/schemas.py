"""
Pydantic schemas — request/response validation for all endpoints
"""
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List, Any
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    email:     EmailStr
    password:  str
    full_name: Optional[str] = None

    @field_validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserLogin(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      int
    email:        str


class UserOut(BaseModel):
    id:         int
    email:      str
    full_name:  Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Product ───────────────────────────────────────────────────────────────────

class ProductCreate(BaseModel):
    name:            str
    length:          float
    width:           float
    height:          float
    weight:          float
    category:        Optional[str] = None
    fragility_level: Optional[str] = "standard"
    stackable:       Optional[bool] = True

    @field_validator("length", "width", "height", "weight")
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Dimensions and weight must be greater than zero")
        return v


class ProductOut(ProductCreate):
    id:         int
    user_id:    int
    created_at: datetime

    class Config:
        from_attributes = True


# ── Order ─────────────────────────────────────────────────────────────────────

class OrderItemCreate(BaseModel):
    product_id: int
    quantity:   int = 1

    @field_validator("quantity")
    def qty_positive(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be at least 1")
        return v


class OrderCreate(BaseModel):
    destination_zone: Optional[str] = "default"
    priority:         Optional[str] = "cost"
    items:            List[OrderItemCreate]


class OrderOut(BaseModel):
    id:               int
    user_id:          int
    destination_zone: Optional[str]
    status:           str
    priority:         str
    created_at:       datetime

    class Config:
        from_attributes = True


# ── Optimize packaging ────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    order_id:    int
    destination: Optional[str] = "default"
    priority:    Optional[str] = "cost"    # cost | speed


class PackedBoxOut(BaseModel):
    box_type:        str
    items:           List[Any]
    box_cost:        float
    shipping_cost:   float
    efficiency_score: float


class OptimizeResponse(BaseModel):
    order_id:             int
    recommended_plan:     List[PackedBoxOut]
    total_cost:           float
    efficiency_score:     float
    decision_explanation: str
    decision_engine:      str
    alternatives:         List[Any] = []


# ── Box inventory ─────────────────────────────────────────────────────────────

class BoxInventoryCreate(BaseModel):
    box_type:           str
    length:             float
    width:              float
    height:             float
    max_weight:         float
    cost:               float
    quantity_available: int = 100
    suitable_fragile:   bool = False


class BoxInventoryOut(BoxInventoryCreate):
    id:         int
    created_at: datetime

    class Config:
        from_attributes = True


# ── Analytics ─────────────────────────────────────────────────────────────────

class AnalyticsOut(BaseModel):
    total_orders:        int
    total_cost_saved:    float
    avg_efficiency:      float
    waste_percentage:    float
    box_usage:           dict
    orders_by_day:       List[dict]
    efficiency_trend:    List[dict]
