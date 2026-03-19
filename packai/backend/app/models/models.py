"""
SQLAlchemy ORM Models — AI Packaging Automation Platform
Full production schema with all 12 tables
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text,
    TIMESTAMP, ForeignKey, JSON, CheckConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    full_name     = Column(String(255), nullable=True)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(TIMESTAMP, server_default=func.now())

    orders   = relationship("Order", back_populates="user")
    products = relationship("Product", back_populates="user")


class Product(Base):
    __tablename__ = "products"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    name            = Column(String(255), nullable=False)
    length          = Column(Float, nullable=False)
    width           = Column(Float, nullable=False)
    height          = Column(Float, nullable=False)
    weight          = Column(Float, nullable=False)
    category        = Column(String(100), nullable=True)
    fragility_level = Column(String(20), default="standard")   # standard | fragile | very_fragile
    stackable       = Column(Boolean, default=True)
    created_at      = Column(TIMESTAMP, server_default=func.now())

    __table_args__ = (
        CheckConstraint("length > 0",  name="ck_product_length"),
        CheckConstraint("width > 0",   name="ck_product_width"),
        CheckConstraint("height > 0",  name="ck_product_height"),
        CheckConstraint("weight > 0",  name="ck_product_weight"),
    )

    user        = relationship("User", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")


class Order(Base):
    __tablename__ = "orders"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    destination_zone = Column(String(50), nullable=True)
    status           = Column(String(50), default="pending")   # pending | processing | completed | failed
    priority         = Column(String(20), default="cost")      # cost | speed
    created_at       = Column(TIMESTAMP, server_default=func.now())

    user            = relationship("User", back_populates="orders")
    order_items     = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    packaging_plans = relationship("PackagingPlan", back_populates="order")


class OrderItem(Base):
    __tablename__ = "order_items"

    id         = Column(Integer, primary_key=True, index=True)
    order_id   = Column(Integer, ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity   = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_orderitem_quantity"),
    )

    order   = relationship("Order", back_populates="order_items")
    product = relationship("Product", back_populates="order_items")


class BoxInventory(Base):
    __tablename__ = "box_inventory"

    id                 = Column(Integer, primary_key=True, index=True)
    box_type           = Column(String(100), nullable=False, unique=True)
    length             = Column(Float, nullable=False)
    width              = Column(Float, nullable=False)
    height             = Column(Float, nullable=False)
    max_weight         = Column(Float, nullable=False)
    cost               = Column(Float, nullable=False)
    quantity_available = Column(Integer, nullable=False, default=0)
    suitable_fragile   = Column(Boolean, default=False)
    created_at         = Column(TIMESTAMP, server_default=func.now())

    __table_args__ = (
        CheckConstraint("quantity_available >= 0", name="ck_box_qty_nonneg"),
        CheckConstraint("length > 0", name="ck_box_length"),
        CheckConstraint("width > 0",  name="ck_box_width"),
        CheckConstraint("height > 0", name="ck_box_height"),
    )


class PackagingPlan(Base):
    __tablename__ = "packaging_plans"

    id               = Column(Integer, primary_key=True, index=True)
    order_id         = Column(Integer, ForeignKey("orders.id"), nullable=False)
    total_cost       = Column(Float, nullable=False)
    efficiency_score = Column(Float, nullable=False)
    decision_reason  = Column(Text, nullable=True)
    decision_engine  = Column(String(50), default="rule_based")  # rule_based | ml_fallback
    created_at       = Column(TIMESTAMP, server_default=func.now())

    order                = relationship("Order", back_populates="packaging_plans")
    packaging_plan_items = relationship("PackagingPlanItem", back_populates="packaging_plan", cascade="all, delete-orphan")


class PackagingPlanItem(Base):
    __tablename__ = "packaging_plan_items"

    id                = Column(Integer, primary_key=True, index=True)
    packaging_plan_id = Column(Integer, ForeignKey("packaging_plans.id", ondelete="CASCADE"), nullable=False)
    box_type          = Column(String(100), nullable=False)
    items             = Column(JSON, nullable=True)      # list of product ids packed in this box
    box_cost          = Column(Float, nullable=False)
    shipping_cost     = Column(Float, nullable=False)
    efficiency_score  = Column(Float, nullable=False)

    packaging_plan = relationship("PackagingPlan", back_populates="packaging_plan_items")


class AnalyticsSummary(Base):
    __tablename__ = "analytics_summary"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_orders     = Column(Integer, default=0)
    total_cost_saved = Column(Float, default=0.0)
    avg_efficiency   = Column(Float, default=0.0)
    waste_percentage = Column(Float, default=0.0)
    updated_at       = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


# ── Indexes for query performance ────────────────────────────────────────────
Index("ix_orders_user_status",       Order.user_id, Order.status)
Index("ix_order_items_order",        OrderItem.order_id)
Index("ix_packaging_plans_order",    PackagingPlan.order_id)
Index("ix_box_inventory_available",  BoxInventory.quantity_available)
Index("ix_analytics_user",           AnalyticsSummary.user_id)
