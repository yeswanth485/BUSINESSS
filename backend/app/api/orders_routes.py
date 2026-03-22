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
    Bulk CSV / Excel upload — handles 1 to 1000 orders.

    Pipeline:
      1. Validate all rows
      2. Run ALL 5 ML models on ALL orders simultaneously (batch matrix)
      3. FFD validates each ML vote physically fits
      4. Save products + orders + packaging plans to DB
      5. Return full cost breakdown per order
    """
    if not payload.orders:
        raise HTTPException(status_code=400, detail="No orders provided")
    if len(payload.orders) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 orders per batch")

    from ..services import ml_service as _ml
    from ..services.cost_service import calculate_total_cost, calculate_efficiency_score, calculate_baseline_cost, calculate_cost_savings, DEFAULT_BOXES
    from ..models.models import PackagingPlan, PackagingPlanItem

    BOX_ORDER = ["Box_XS","Box_S","Box_M","Box_L","Box_XL","Box_XXL"]

    # ── Step 1: Validate and prepare all rows ──────────────────────────────────
    valid_rows   = []
    errors       = []
    for i, row in enumerate(payload.orders):
        if any(v <= 0 for v in [row.length, row.width, row.height, row.weight]):
            errors.append({"order_id": row.order_id, "error": "Invalid dimensions (must be > 0)"})
            continue
        if row.quantity < 1:
            errors.append({"order_id": row.order_id, "error": "Quantity must be >= 1"})
            continue
        valid_rows.append(row)

    if not valid_rows:
        raise HTTPException(status_code=400, detail="No valid rows after validation")

    # ── Step 2: Batch ML — all 5 models on all orders at once ─────────────────
    ml_batch_results = []
    ml_used = False
    if _ml.is_ml_available():
        try:
            order_dims = [(r.length, r.width, r.height, r.weight) for r in valid_rows]
            ml_batch_results = _ml.predict_batch(order_dims)
            ml_used = True
            logger.info(f"[Bulk] ML batch: {len(order_dims)} orders, {ml_batch_results[0]['models_used']} models")
        except Exception as e:
            logger.warning(f"[Bulk] ML batch failed: {e} — using FFD only")

    # ── Fetch box inventory once ───────────────────────────────────────────────
    db_boxes = db.query(BoxInventory).filter(BoxInventory.quantity_available > 0).all()
    if db_boxes:
        box_dicts = [
            {"box_type":b.box_type,"length":b.length,"width":b.width,"height":b.height,
             "max_weight":b.max_weight,"cost":b.cost,"quantity_available":b.quantity_available,
             "suitable_fragile":b.suitable_fragile}
            for b in db_boxes
        ]
    else:
        box_dicts = [dict(b) for b in DEFAULT_BOXES]
        for b in box_dicts: b["quantity_available"] = 999

    def ffd_select(voted_box, item_vol, total_wt, is_fragile):
        eligible = sorted(
            [b for b in box_dicts if not is_fragile or b.get("suitable_fragile")],
            key=lambda b: BOX_ORDER.index(b["box_type"]) if b["box_type"] in BOX_ORDER else 99
        )
        if not eligible: eligible = sorted(box_dicts, key=lambda b: b["length"]*b["width"]*b["height"])
        if voted_box:
            ml_b = next((b for b in eligible if b["box_type"]==voted_box), None)
            if ml_b:
                bvol = ml_b["length"]*ml_b["width"]*ml_b["height"]
                if bvol >= item_vol and ml_b["max_weight"] >= total_wt:
                    return ml_b, "ml_primary"
        for b in eligible:
            if b["length"]*b["width"]*b["height"] >= item_vol and b["max_weight"] >= total_wt:
                return b, "ml_ffd_corrected" if voted_box else "rule_based"
        return eligible[-1], "fallback"

    # ── Step 3: Per-order FFD validation + DB save ────────────────────────────
    results      = []
    total_opt    = 0.0
    total_base   = 0.0
    total_saved  = 0.0

    for i, row in enumerate(valid_rows):
        try:
            fragility   = "fragile" if str(row.fragility).lower() in ("fragile","true","yes","1") else "standard"
            is_fragile  = fragility == "fragile"
            zone        = row.zone or payload.zone or "zone_b"
            item_vol    = row.length * row.width * row.height * row.quantity
            total_wt    = row.weight * row.quantity

            # ML vote for this order (from batch results)
            voted_box   = ml_batch_results[i]["voted_box"] if ml_used and i < len(ml_batch_results) else None
            ml_conf     = ml_batch_results[i]["vote_confidence"] if ml_used and i < len(ml_batch_results) else 0.0
            ml_agr      = ml_batch_results[i]["agreement"] if ml_used and i < len(ml_batch_results) else 0.0

            # FFD validates
            final_box, engine = ffd_select(voted_box, item_vol, total_wt, is_fragile)

            # Cost calculation
            bvol = final_box["length"]*final_box["width"]*final_box["height"]
            from ..services.cost_service import calculate_total_cost, calculate_efficiency_score
            cost_data = calculate_total_cost(
                final_box["length"], final_box["width"], final_box["height"],
                total_wt, final_box["cost"], zone
            )
            eff = calculate_efficiency_score(item_vol, bvol)

            # Baseline cost (next-size-up)
            baseline_data = calculate_baseline_cost(
                actual_weight=total_wt, item_volume=item_vol,
                destination_zone=zone, is_fragile=is_fragile, available_boxes=box_dicts
            )
            savings_data  = calculate_cost_savings(cost_data["total_cost"], baseline_data["baseline_cost"])

            # Save product
            product = None
            if row.sku:
                product = db.query(Product).filter(
                    Product.user_id==current_user.id, Product.sku==row.sku
                ).first()
            if not product:
                product = Product(
                    user_id=current_user.id, name=row.product_name, sku=row.sku,
                    length=row.length, width=row.width, height=row.height,
                    weight=row.weight, category=row.category,
                    fragility_level=fragility, channel=row.channel, pincode=row.pincode
                )
                db.add(product)
            else:
                product.length=row.length; product.width=row.width
                product.height=row.height; product.weight=row.weight
                product.fragility_level=fragility
            db.flush()

            # Save order
            order = Order(user_id=current_user.id, destination_zone=zone,
                          priority=payload.priority, status="completed")
            db.add(order)
            db.flush()
            db.add(OrderItem(order_id=order.id, product_id=product.id, quantity=row.quantity))
            db.flush()

            # Save packaging plan
            explanation = (
                f"{'ML (5-model vote, {:.0%} agreement)'.format(ml_agr)} → {voted_box or '—'}"
                f" → FFD {engine}: {final_box['box_type']}. "
                f"Zone {zone}. Chargeable {cost_data['chargeable_weight']:.2f}kg."
            ) if ml_used else f"FFD rule engine → {final_box['box_type']}. Zone {zone}."

            plan = PackagingPlan(
                order_id=order.id,
                total_cost=round(cost_data["total_cost"],2),
                efficiency_score=round(eff,4),
                decision_reason=explanation,
                decision_engine=engine,
                baseline_cost=round(baseline_data["baseline_cost"],2),
                optimized_cost=round(cost_data["total_cost"],2),
                savings=round(savings_data["savings_inr"],2),
            )
            db.add(plan)
            db.flush()
            db.add(PackagingPlanItem(
                packaging_plan_id=plan.id,
                box_type=final_box["box_type"],
                items=[{"product_id":product.id,"name":row.product_name,"quantity":row.quantity}],
                box_cost=cost_data["box_cost"],
                shipping_cost=cost_data["shipping_cost"],
                efficiency_score=eff,
            ))

            # Reduce box inventory
            box_inv = db.query(BoxInventory).filter(BoxInventory.box_type==final_box["box_type"]).first()
            if box_inv and box_inv.quantity_available > 0:
                box_inv.quantity_available -= 1

            db.commit()

            total_opt   += cost_data["total_cost"]
            total_base  += baseline_data["baseline_cost"]
            total_saved += savings_data["savings_inr"]

            results.append({
                "order_id":       row.order_id,
                "db_order_id":    order.id,
                "product_name":   row.product_name,
                "channel":        row.channel or "—",
                "zone":           zone,
                "fragility":      fragility,
                "box":            final_box["box_type"],
                "baseline_cost":  round(baseline_data["baseline_cost"],2),
                "baseline_box":   baseline_data["baseline_box"],
                "optimized_cost": round(cost_data["total_cost"],2),
                "savings":        round(savings_data["savings_inr"],2),
                "savings_percent":round(savings_data["savings_percent"],1),
                "efficiency":     round(eff*100,1),
                "engine":         engine,
                "ml_prediction":  voted_box,
                "ml_confidence":  round(ml_conf,4),
                "ml_agreement":   round(ml_agr,4),
                "dim_weight":     cost_data["dim_weight"],
                "chargeable_wt":  cost_data["chargeable_weight"],
                "explanation":    explanation,
                "box_cost":       cost_data["box_cost"],
                "shipping_cost":  cost_data["shipping_cost"],
            })

        except Exception as e:
            try: db.rollback()
            except: pass
            logger.error(f"[Bulk] Row {i} ({row.order_id}) failed: {e}")
            errors.append({"order_id": row.order_id, "error": str(e)})

    logger.info(
        f"[Bulk] User {current_user.id}: {len(results)}/{len(valid_rows)} done, "
        f"{len(errors)} errors, ₹{total_saved:.0f} saved, "
        f"engine={'ml_batch+ffd' if ml_used else 'ffd_only'}"
    )

    return {
        "processed":         len(results),
        "errors":            len(errors),
        "error_details":     errors,
        "results":           results,
        "ml_used":           ml_used,
        "summary": {
            "total_baseline_cost":   round(total_base, 2),
            "total_optimized_cost":  round(total_opt,  2),
            "total_savings":         round(total_saved, 2),
            "avg_savings_per_order": round(total_saved / len(results), 2) if results else 0,
            "engine":                "ml_batch+ffd" if ml_used else "ffd_only",
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
