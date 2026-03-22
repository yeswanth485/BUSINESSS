"""
Decision Engine — Hybrid rule-based + ML fallback.
Returns real baseline cost, optimized cost, and savings for every order.
"""
import time
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from ..models.models import Order, OrderItem, Product, BoxInventory, PackagingPlan, PackagingPlanItem
from .packing_service import Item, first_fit_decreasing
from .cost_service import (
    calculate_total_cost, calculate_efficiency_score,
    calculate_baseline_cost, calculate_cost_savings, DEFAULT_BOXES
)
from . import ml_service

logger = logging.getLogger(__name__)


def optimize_packaging(order_id: int, db: Session, destination_zone: str = "zone_b") -> Dict[str, Any]:
    """
    Main optimization entry point.
    Returns full cost breakdown: baseline, optimized, savings.
    """
    start_time = time.time()

    # ── Fetch order and validate ──────────────────────────────────────────────
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

    # Use order zone if not overridden
    zone = destination_zone if destination_zone != "default" else (order.destination_zone or "zone_b")

    is_fragile = any(
        item.product.fragility_level in ("fragile", "very_fragile")
        for item in order_items
    )

    items_to_pack: List[Item] = []
    total_actual_weight = 0.0
    total_item_volume   = 0.0
    for oi in order_items:
        p = oi.product
        items_to_pack.append(Item(
            product_id=p.id, name=p.name,
            length=p.length, width=p.width, height=p.height,
            weight=p.weight, quantity=oi.quantity,
        ))
        total_actual_weight += p.weight * oi.quantity
        total_item_volume   += p.length * p.width * p.height * oi.quantity

    # ── Fetch box inventory ───────────────────────────────────────────────────
    db_boxes = db.query(BoxInventory).filter(BoxInventory.quantity_available > 0).all()
    if db_boxes:
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
            for b in db_boxes
        ]
    else:
        box_dicts = [dict(b) for b in DEFAULT_BOXES]
        for b in box_dicts:
            b["quantity_available"] = 999

    # ── Run FFD rule-based packing ────────────────────────────────────────────
    packed_boxes   = first_fit_decreasing(items_to_pack, box_dicts, fragile=is_fragile)
    decision_engine = "rule_based"
    plan_items      = []
    optimized_cost  = 0.0
    overall_eff     = 0.0

    if packed_boxes:
        total_box_volume = 0.0
        for box in packed_boxes:
            cost_data = calculate_total_cost(
                box.length, box.width, box.height,
                box.used_weight, box.cost, zone,
            )
            eff = calculate_efficiency_score(box.used_volume, box.volume)
            optimized_cost   += cost_data["total_cost"]
            total_box_volume += box.volume
            plan_items.append({
                "box_type":       box.box_type,
                "items":          box.packed_items,
                "box_cost":       cost_data["box_cost"],
                "shipping_cost":  cost_data["shipping_cost"],
                "efficiency_score": eff,
                "dim_weight":     cost_data["dim_weight"],
                "chargeable_weight": cost_data["chargeable_weight"],
            })
        overall_eff = calculate_efficiency_score(total_item_volume, total_box_volume)
        decision_reason = (
            f"FFD rule engine: {len(packed_boxes)} box(es) selected. "
            f"Zone: {zone}. "
            f"{'Fragile-safe boxes used. ' if is_fragile else ''}"
            f"Chargeable weight: {sum(p['chargeable_weight'] for p in plan_items):.2f}kg."
        )

    else:
        # ── ML fallback ───────────────────────────────────────────────────────
        decision_engine = "ml_fallback"
        rep = items_to_pack[0]
        logger.info(f"[ML] Fallback triggered for order {order_id}: {rep.name} {rep.length}x{rep.width}x{rep.height}cm {rep.weight}kg")

        try:
            ml_result = ml_service.predict_packaging(rep.length, rep.width, rep.height, rep.weight)
            ml_box    = ml_result["recommended_box"]
            logger.info(f"[ML] Prediction: {ml_box} (conf={ml_result['confidence_score']:.2%}, model={ml_result['model_used']})")
        except (RuntimeError, Exception) as e:
            logger.warning(f"[ML] Failed ({e}), using rule fallback")
            ml_box = _best_fit_fallback(rep, box_dicts)
            ml_result = {"confidence_score": 0.0, "model_used": "rule_fallback"}

        box_record = next((b for b in box_dicts if b["box_type"] == ml_box), None)
        if not box_record:
            box_record = box_dicts[-1]

        cost_data = calculate_total_cost(
            box_record["length"], box_record["width"], box_record["height"],
            rep.weight * rep.quantity, box_record["cost"], zone,
        )
        item_vol = rep.volume
        box_vol  = box_record["length"] * box_record["width"] * box_record["height"]
        eff      = calculate_efficiency_score(item_vol, box_vol)
        optimized_cost = cost_data["total_cost"]
        overall_eff    = eff
        plan_items = [{
            "box_type":          ml_box,
            "items":             [{"product_id": rep.product_id, "name": rep.name, "quantity": rep.quantity}],
            "box_cost":          cost_data["box_cost"],
            "shipping_cost":     cost_data["shipping_cost"],
            "efficiency_score":  eff,
            "dim_weight":        cost_data["dim_weight"],
            "chargeable_weight": cost_data["chargeable_weight"],
        }]
        decision_reason = (
            f"ML fallback ({ml_result['model_used']}, conf={ml_result['confidence_score']:.0%}): "
            f"FFD found no valid plan. Recommended {ml_box}."
        )

    # ── Calculate real baseline cost ─────────────────────────────────────────
    baseline_data = calculate_baseline_cost(
        actual_weight    = total_actual_weight,
        item_volume      = total_item_volume,
        destination_zone = zone,
        is_fragile       = is_fragile,
        available_boxes  = box_dicts,
    )
    baseline_cost = baseline_data["baseline_cost"]
    savings_data  = calculate_cost_savings(optimized_cost, baseline_cost)
    savings       = savings_data["savings_inr"]

    # ── Persist plan to DB ────────────────────────────────────────────────────
    plan = PackagingPlan(
        order_id         = order_id,
        total_cost       = round(optimized_cost, 2),
        efficiency_score = round(overall_eff, 4),
        decision_reason  = decision_reason,
        decision_engine  = decision_engine,
        baseline_cost    = round(baseline_cost, 2),
        optimized_cost   = round(optimized_cost, 2),
        savings          = round(savings, 2),
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

    # Reduce inventory for each box used
    for pi in plan_items:
        box_inv = db.query(BoxInventory).filter(BoxInventory.box_type == pi["box_type"]).first()
        if box_inv and box_inv.quantity_available > 0:
            box_inv.quantity_available -= 1

    order.status = "completed"
    db.commit()
    db.refresh(plan)

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    logger.info(f"[Engine] Order {order_id}: {decision_engine} | optimized=₹{optimized_cost:.0f} baseline=₹{baseline_cost:.0f} saved=₹{savings:.0f} | {elapsed_ms}ms")

    return {
        "order_id":             order_id,
        "recommended_plan":     plan_items,
        "baseline_cost":        round(baseline_cost, 2),
        "optimized_cost":       round(optimized_cost, 2),
        "savings":              round(savings, 2),
        "savings_percent":      round(savings_data["savings_percent"], 1),
        "total_cost":           round(optimized_cost, 2),
        "efficiency_score":     round(overall_eff, 4),
        "decision_explanation": decision_reason,
        "decision_engine":      decision_engine,
        "baseline_box":         baseline_data["baseline_box"],
        "processing_time_ms":   elapsed_ms,
        "alternatives":         [],
    }


def _best_fit_fallback(item: Item, box_dicts: list) -> str:
    item_vol = item.length * item.width * item.height
    sorted_boxes = sorted(box_dicts, key=lambda b: b["length"] * b["width"] * b["height"])
    for b in sorted_boxes:
        if b["length"] * b["width"] * b["height"] >= item_vol and b["max_weight"] >= item.weight:
            return b["box_type"]
    return sorted_boxes[-1]["box_type"]
