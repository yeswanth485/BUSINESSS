"""
Decision Engine — hybrid rule-based + ML packaging optimizer.

Process flow:
  1. Fetch order + items from DB
  2. Fetch available box inventory
  3. Check fragility rules
  4. Run FFD packing algorithm
  5. Calculate cost + efficiency per box
  6. Rank and select best plan
  7. If no valid plan → ML fallback
  8. Store result in DB
  9. Return structured response
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..models.models import Order, OrderItem, Product, BoxInventory, PackagingPlan, PackagingPlanItem
from .packing_service import Item, first_fit_decreasing
from .cost_service import calculate_total_cost, calculate_efficiency_score
from . import ml_service
import time


def optimize_packaging(order_id: int, db: Session, destination_zone: str = "default") -> Dict[str, Any]:
    """
    Main entry point — deterministic decision engine.
    Returns a full packaging plan for the given order.
    """
    start_time = time.time()

    # ── Step 1: Fetch order and items ─────────────────────────────────────────
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise ValueError(f"Order {order_id} not found")

    order_items = (
        db.query(OrderItem)
        .filter(OrderItem.order_id == order_id)
        .join(Product)
        .all()
    )

    if not order_items:
        raise ValueError(f"Order {order_id} has no items")

    # Determine if any item is fragile
    is_fragile = any(
        item.product.fragility_level in ("fragile", "very_fragile")
        for item in order_items
    )

    # Build Item objects for packing algorithm
    items_to_pack: List[Item] = []
    for oi in order_items:
        p = oi.product
        items_to_pack.append(Item(
            product_id = p.id,
            name       = p.name,
            length     = p.length,
            width      = p.width,
            height     = p.height,
            weight     = p.weight,
            quantity   = oi.quantity,
        ))

    # ── Step 2: Fetch available boxes ─────────────────────────────────────────
    boxes = db.query(BoxInventory).filter(BoxInventory.quantity_available > 0).all()
    box_dicts = [
        {
            "box_type":           b.box_type,
            "length":             b.length,
            "width":              b.width,
            "height":             b.height,
            "max_weight":         b.max_weight,
            "cost":               b.cost,
            "quantity_available": b.quantity_available,
            "suitable_fragile":   b.suitable_fragile,
        }
        for b in boxes
    ]

    # ── Step 3: Run FFD packing algorithm ─────────────────────────────────────
    packed_boxes = first_fit_decreasing(items_to_pack, box_dicts, fragile=is_fragile)

    decision_engine = "rule_based"
    decision_reason = ""

    if packed_boxes:
        # ── Step 4: Calculate cost + efficiency per box ───────────────────────
        plan_items        = []
        total_cost        = 0.0
        total_item_volume = sum(i.volume for i in items_to_pack)
        total_box_volume  = 0.0

        for box in packed_boxes:
            cost_data = calculate_total_cost(
                length           = box.length,
                width            = box.width,
                height           = box.height,
                actual_weight    = box.used_weight,
                box_cost         = box.cost,
                destination_zone = destination_zone,
            )
            eff = calculate_efficiency_score(box.used_volume, box.volume)
            total_cost       += cost_data["total_cost"]
            total_box_volume += box.volume

            plan_items.append({
                "box_type":        box.box_type,
                "items":           box.packed_items,
                "box_cost":        cost_data["box_cost"],
                "shipping_cost":   cost_data["shipping_cost"],
                "efficiency_score": eff,
            })

        overall_efficiency = calculate_efficiency_score(total_item_volume, total_box_volume)
        decision_reason    = f"Rule-based FFD: {len(packed_boxes)} box(es) selected, optimised for {'cost' if order.priority == 'cost' else 'speed'}."

    else:
        # ── Step 5: ML fallback ───────────────────────────────────────────────
        decision_engine = "ml_fallback"

        # Use first item's dimensions as representative for single-item ML prediction
        rep_item = items_to_pack[0]
        try:
            ml_result = ml_service.predict_packaging(
                length = rep_item.length,
                width  = rep_item.width,
                height = rep_item.height,
                weight = rep_item.weight,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Rule-based packing failed and ML models unavailable: {e}")

        # Build a minimal plan from ML result
        ml_box = ml_result["recommended_box"]
        box_record = db.query(BoxInventory).filter(BoxInventory.box_type == ml_box).first()

        if box_record:
            cost_data = calculate_total_cost(
                length           = box_record.length,
                width            = box_record.width,
                height           = box_record.height,
                actual_weight    = rep_item.weight * rep_item.quantity,
                box_cost         = box_record.cost,
                destination_zone = destination_zone,
            )
            item_vol = rep_item.volume
            box_vol  = box_record.length * box_record.width * box_record.height
            eff      = calculate_efficiency_score(item_vol, box_vol)
            total_cost = cost_data["total_cost"]
            plan_items = [{
                "box_type":        ml_box,
                "items":           [{"product_id": rep_item.product_id, "name": rep_item.name, "quantity": rep_item.quantity}],
                "box_cost":        cost_data["box_cost"],
                "shipping_cost":   cost_data["shipping_cost"],
                "efficiency_score": eff,
            }]
            overall_efficiency = eff
        else:
            # ML recommended a box not in inventory — use confidence as proxy
            total_cost         = 0.0
            overall_efficiency = ml_result["confidence_score"]
            plan_items = [{
                "box_type":        ml_box,
                "items":           [{"product_id": rep_item.product_id, "name": rep_item.name, "quantity": rep_item.quantity}],
                "box_cost":        0.0,
                "shipping_cost":   0.0,
                "efficiency_score": ml_result["confidence_score"],
            }]

        decision_reason = (
            f"ML fallback ({ml_result['model_used']}): no valid rule-based plan found. "
            f"Recommended {ml_box} with confidence {ml_result['confidence_score']:.2%}."
        )

    # ── Step 6: Store plan in DB ───────────────────────────────────────────────
    plan = PackagingPlan(
        order_id         = order_id,
        total_cost       = round(total_cost, 2),
        efficiency_score = round(overall_efficiency, 4),
        decision_reason  = decision_reason,
        decision_engine  = decision_engine,
    )
    db.add(plan)
    db.flush()

    for pi in plan_items:
        db.add(PackagingPlanItem(
            packaging_plan_id = plan.id,
            box_type          = pi["box_type"],
            items             = pi["items"],
            box_cost          = pi["box_cost"],
            shipping_cost     = pi["shipping_cost"],
            efficiency_score  = pi["efficiency_score"],
        ))

    # ── Step 7: Update box inventory quantities ────────────────────────────────
    for pi in plan_items:
        box_record = db.query(BoxInventory).filter(BoxInventory.box_type == pi["box_type"]).first()
        if box_record and box_record.quantity_available > 0:
            box_record.quantity_available -= 1

    # Update order status
    order.status = "completed"
    db.commit()
    db.refresh(plan)

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "order_id":             order_id,
        "recommended_plan":     plan_items,
        "total_cost":           round(total_cost, 2),
        "efficiency_score":     round(overall_efficiency, 4),
        "decision_explanation": decision_reason,
        "decision_engine":      decision_engine,
        "processing_time_ms":   elapsed_ms,
        "alternatives":         [],
    }
