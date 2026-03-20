"""
Decision Engine — hybrid rule-based + ML packaging optimizer.
ML is used only as fallback and only when sklearn is available.
On Render free tier without sklearn, rule-based engine handles everything.
"""
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from ..models.models import Order, OrderItem, Product, BoxInventory, PackagingPlan, PackagingPlanItem
from .packing_service import Item, first_fit_decreasing
from .cost_service import calculate_total_cost, calculate_efficiency_score
from . import ml_service
import time


def optimize_packaging(order_id: int, db: Session, destination_zone: str = "default") -> Dict[str, Any]:
    """
    Main entry point — deterministic decision engine.
    Rule-based first. ML fallback only if rules produce no result.
    """
    start_time = time.time()

    # ── Fetch order ───────────────────────────────────────────────────────────
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

    is_fragile = any(
        item.product.fragility_level in ("fragile", "very_fragile")
        for item in order_items
    )

    items_to_pack: List[Item] = []
    for oi in order_items:
        p = oi.product
        items_to_pack.append(Item(
            product_id=p.id, name=p.name,
            length=p.length, width=p.width, height=p.height,
            weight=p.weight, quantity=oi.quantity,
        ))

    # ── Fetch available boxes ─────────────────────────────────────────────────
    boxes = db.query(BoxInventory).filter(BoxInventory.quantity_available > 0).all()

    # If no inventory in DB, use sensible hardcoded defaults
    if not boxes:
        box_dicts = [
            {"box_type":"Box_XS","length":15,"width":10,"height":10,"max_weight":2, "cost":15,"quantity_available":999,"suitable_fragile":False},
            {"box_type":"Box_S", "length":25,"width":20,"height":15,"max_weight":5, "cost":25,"quantity_available":999,"suitable_fragile":False},
            {"box_type":"Box_M", "length":35,"width":25,"height":20,"max_weight":12,"cost":40,"quantity_available":999,"suitable_fragile":True},
            {"box_type":"Box_L", "length":50,"width":40,"height":30,"max_weight":20,"cost":65,"quantity_available":999,"suitable_fragile":True},
            {"box_type":"Box_XL","length":70,"width":50,"height":40,"max_weight":30,"cost":95,"quantity_available":999,"suitable_fragile":True},
            {"box_type":"Box_XXL","length":90,"width":70,"height":60,"max_weight":50,"cost":140,"quantity_available":999,"suitable_fragile":True},
        ]
    else:
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

    # ── Run FFD packing ───────────────────────────────────────────────────────
    packed_boxes = first_fit_decreasing(items_to_pack, box_dicts, fragile=is_fragile)

    decision_engine = "rule_based"
    decision_reason = ""
    plan_items      = []
    total_cost      = 0.0
    overall_efficiency = 0.0

    if packed_boxes:
        total_item_volume = sum(i.volume for i in items_to_pack)
        total_box_volume  = 0.0

        for box in packed_boxes:
            cost_data = calculate_total_cost(
                length=box.length, width=box.width, height=box.height,
                actual_weight=box.used_weight,
                box_cost=box.cost,
                destination_zone=destination_zone,
            )
            eff             = calculate_efficiency_score(box.used_volume, box.volume)
            total_cost      += cost_data["total_cost"]
            total_box_volume += box.volume
            plan_items.append({
                "box_type":         box.box_type,
                "items":            box.packed_items,
                "box_cost":         cost_data["box_cost"],
                "shipping_cost":    cost_data["shipping_cost"],
                "efficiency_score": eff,
            })

        overall_efficiency = calculate_efficiency_score(total_item_volume, total_box_volume)
        decision_reason    = (
            f"Rule-based FFD: {len(packed_boxes)} box(es) selected, "
            f"optimised for {order.priority or 'cost'}."
        )

    else:
        # ── ML fallback (only if sklearn available) ───────────────────────────
        decision_engine = "ml_fallback"
        rep             = items_to_pack[0]

        try:
            ml_result = ml_service.predict_packaging(
                length=rep.length, width=rep.width,
                height=rep.height, weight=rep.weight,
            )
            ml_box = ml_result["recommended_box"]
        except (RuntimeError, Exception):
            # sklearn not available — use best-fit rule as last resort
            ml_box = _best_fit_fallback(rep, box_dicts)
            ml_result = {"confidence_score": 0.75, "model_used": "rule_fallback"}

        box_record = next((b for b in box_dicts if b["box_type"] == ml_box), None)
        if box_record:
            cost_data = calculate_total_cost(
                length=box_record["length"], width=box_record["width"],
                height=box_record["height"],
                actual_weight=rep.weight * rep.quantity,
                box_cost=box_record["cost"],
                destination_zone=destination_zone,
            )
            item_vol = rep.volume
            box_vol  = box_record["length"] * box_record["width"] * box_record["height"]
            eff      = calculate_efficiency_score(item_vol, box_vol)
            total_cost = cost_data["total_cost"]
            overall_efficiency = eff
            plan_items = [{
                "box_type":         ml_box,
                "items":            [{"product_id": rep.product_id, "name": rep.name, "quantity": rep.quantity}],
                "box_cost":         cost_data["box_cost"],
                "shipping_cost":    cost_data["shipping_cost"],
                "efficiency_score": eff,
            }]
        else:
            overall_efficiency = 0.75
            total_cost = 0.0
            plan_items = [{
                "box_type":         ml_box,
                "items":            [{"product_id": rep.product_id, "name": rep.name, "quantity": rep.quantity}],
                "box_cost":         0.0,
                "shipping_cost":    0.0,
                "efficiency_score": 0.75,
            }]

        decision_reason = (
            f"Fallback ({ml_result['model_used']}): no valid rule-based plan found. "
            f"Recommended {ml_box}."
        )

    # ── Store plan in DB ──────────────────────────────────────────────────────
    plan = PackagingPlan(
        order_id=order_id,
        total_cost=round(total_cost, 2),
        efficiency_score=round(overall_efficiency, 4),
        decision_reason=decision_reason,
        decision_engine=decision_engine,
    )
    db.add(plan)
    db.flush()

    for pi in plan_items:
        db.add(PackagingPlanItem(
            packaging_plan_id=plan.id,
            box_type=pi["box_type"],
            items=pi["items"],
            box_cost=pi["box_cost"],
            shipping_cost=pi["shipping_cost"],
            efficiency_score=pi["efficiency_score"],
        ))

    # Update inventory quantities
    for pi in plan_items:
        box_record = db.query(BoxInventory).filter(BoxInventory.box_type == pi["box_type"]).first()
        if box_record and box_record.quantity_available > 0:
            box_record.quantity_available -= 1

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


def _best_fit_fallback(item: Item, box_dicts: list) -> str:
    """Pure rule-based last resort — pick smallest box that fits one item."""
    item_vol    = item.length * item.width * item.height
    item_weight = item.weight

    sorted_boxes = sorted(box_dicts, key=lambda b: b["length"] * b["width"] * b["height"])
    for b in sorted_boxes:
        box_vol = b["length"] * b["width"] * b["height"]
        if box_vol >= item_vol and b["max_weight"] >= item_weight:
            return b["box_type"]
    return sorted_boxes[-1]["box_type"]
