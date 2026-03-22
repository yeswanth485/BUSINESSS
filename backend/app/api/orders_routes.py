"""
Orders Routes — includes bulk CSV upload endpoint.
POST /orders/bulk-csv  accepts a list of raw order rows from frontend CSV parser,
creates products + orders + optimizes each, returns full results.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from ..core.database import get_db
from ..core.security import get_current_user
from ..models.models import Order, OrderItem, Product, User
from ..schemas.schemas import OrderCreate, OrderOut
from ..services.decision_service import optimize_packaging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orders", tags=["Orders"])


# ── Bulk CSV schema ───────────────────────────────────────────────────────────
class BulkOrderRow(BaseModel):
    order_id:    str
    product_name: str
    sku:          Optional[str] = None
    length:       float
    width:        float
    height:       float
    weight:       float
    quantity:     int = 1
    category:     Optional[str] = None
    fragility:    Optional[str] = "standard"
    pincode:      Optional[str] = None
    channel:      Optional[str] = None
    zone:         Optional[str] = "zone_b"

class BulkCSVRequest(BaseModel):
    orders:   List[BulkOrderRow]
    zone:     Optional[str] = "zone_b"
    priority: Optional[str] = "cost"


# ── Single order create ───────────────────────────────────────────────────────
@router.post("", response_model=OrderOut, status_code=201)
def create_order(
    payload: OrderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    for item in payload.items:
        product = db.query(Product).filter(
            Product.id == item.product_id,
            Product.user_id == current_user.id
        ).first()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {item.product_id} not found")

    order = Order(
        user_id          = current_user.id,
        destination_zone = payload.destination_zone,
        priority         = payload.priority,
    )
    db.add(order)
    db.flush()
    for item in payload.items:
        db.add(OrderItem(order_id=order.id, product_id=item.product_id, quantity=item.quantity))
    db.commit()
    db.refresh(order)
    return order


# ── Bulk CSV upload + auto-optimize ──────────────────────────────────────────
@router.post("/bulk-csv")
def bulk_csv_upload(
    payload: BulkCSVRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Accepts raw CSV rows from the frontend.
    For each row:
      1. Creates/upserts a Product record
      2. Creates an Order record
      3. Runs optimization → returns baseline, optimised, savings
    Returns all results in one response.
    """
    if not payload.orders:
        raise HTTPException(status_code=400, detail="No orders provided")
    if len(payload.orders) > 500:
        raise HTTPException(status_code=400, detail="Max 500 orders per batch")

    results      = []
    total_saved  = 0.0
    total_opt    = 0.0
    total_base   = 0.0
    errors       = []

    for i, row in enumerate(payload.orders):
        try:
            # Validate dimensions
            if any(v <= 0 for v in [row.length, row.width, row.height, row.weight]):
                errors.append({"order_id": row.order_id, "error": "Invalid dimensions"})
                continue

            # Upsert product (match by sku+user or create new)
            fragility = "fragile" if row.fragility in ("fragile","true","yes","1") else "standard"
            product = None
            if row.sku:
                product = db.query(Product).filter(
                    Product.user_id == current_user.id,
                    Product.sku == row.sku
                ).first()

            if not product:
                product = Product(
                    user_id         = current_user.id,
                    name            = row.product_name,
                    sku             = row.sku,
                    length          = row.length,
                    width           = row.width,
                    height          = row.height,
                    weight          = row.weight,
                    category        = row.category,
                    fragility_level = fragility,
                    channel         = row.channel,
                    pincode         = row.pincode,
                )
                db.add(product)
                db.flush()
            else:
                # Update dimensions if changed
                product.length = row.length
                product.width  = row.width
                product.height = row.height
                product.weight = row.weight
                product.fragility_level = fragility
                db.flush()

            # Create order
            zone = row.zone or payload.zone or "zone_b"
            order = Order(
                user_id          = current_user.id,
                destination_zone = zone,
                priority         = payload.priority,
                status           = "pending",
            )
            db.add(order)
            db.flush()

            db.add(OrderItem(order_id=order.id, product_id=product.id, quantity=row.quantity))
            db.flush()

            # Run optimization immediately
            result = optimize_packaging(order.id, db, destination_zone=zone)
            db.commit()

            total_opt  += result["optimized_cost"]
            total_base += result["baseline_cost"]
            total_saved+= result["savings"]

            results.append({
                "order_id":       row.order_id,
                "db_order_id":    order.id,
                "product_name":   row.product_name,
                "channel":        row.channel,
                "zone":           zone,
                "fragility":      fragility,
                "box":            result["recommended_plan"][0]["box_type"] if result["recommended_plan"] else "—",
                "baseline_cost":  result["baseline_cost"],
                "baseline_box":   result["baseline_box"],
                "optimized_cost": result["optimized_cost"],
                "savings":        result["savings"],
                "savings_percent":result["savings_percent"],
                "efficiency":     round(result["efficiency_score"] * 100, 1),
                "engine":         result["decision_engine"],
                "dim_weight":     result["recommended_plan"][0].get("dim_weight", 0) if result["recommended_plan"] else 0,
                "chargeable_wt":  result["recommended_plan"][0].get("chargeable_weight", 0) if result["recommended_plan"] else 0,
                "explanation":    result["decision_explanation"],
            })

        except Exception as e:
            db.rollback()
            logger.error(f"[Bulk] Row {i} ({row.order_id}) failed: {e}")
            errors.append({"order_id": row.order_id, "error": str(e)})

    logger.info(
        f"[Bulk] User {current_user.id}: {len(results)} succeeded, "
        f"{len(errors)} failed, ₹{total_saved:.0f} saved"
    )

    return {
        "processed":         len(results),
        "errors":            len(errors),
        "error_details":     errors,
        "results":           results,
        "summary": {
            "total_baseline_cost":  round(total_base, 2),
            "total_optimized_cost": round(total_opt, 2),
            "total_savings":        round(total_saved, 2),
            "avg_savings_per_order": round(total_saved / len(results), 2) if results else 0,
        },
    }


# ── List and get orders ───────────────────────────────────────────────────────
@router.get("", response_model=List[OrderOut])
def list_orders(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(Order)
        .filter(Order.user_id == current_user.id)
        .order_by(Order.created_at.desc())
        .offset(skip).limit(limit)
        .all()
    )


@router.get("/{order_id}", response_model=OrderOut)
def get_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    order = db.query(Order).filter(
        Order.id == order_id,
        Order.user_id == current_user.id
    ).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order
