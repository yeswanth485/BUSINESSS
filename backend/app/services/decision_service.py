"""
Decision Engine — FFD rule-based PRIMARY, ML fallback only.

Flow per order (as required by system design):
  Step 1: FFD runs first — exact volumetric + weight + fragility check
  Step 2a: FFD succeeds → use it. Engine = "rule_based". ML never called.
  Step 2b: FFD fails (no valid box found) → ML fallback called
  Step 3:  ML fallback predicts, result validated, engine = "ml_fallback"
  Step 4:  If ML also fails → use largest available box. Engine = "last_resort"

ML does NOT override FFD. FFD decisions are final when successful.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..models.models import (
    Order, OrderItem, Product, BoxInventory,
    PackagingPlan, PackagingPlanItem
)
from .packing_service import Item, first_fit_decreasing
from .cost_service import (
    calculate_total_cost, calculate_efficiency_score,
    calculate_baseline_cost, calculate_cost_savings, DEFAULT_BOXES
)
from . import ml_service

logger = logging.getLogger(__name__)

BOX_ORDER = ["Box_XS", "Box_S", "Box_M", "Box_L", "Box_XL", "Box_XXL"]


def optimize_packaging(
    order_id: int,
    db: Session,
    destination_zone: str = "zone_b",
) -> Dict[str, Any]:
    """
    Main entry point.
    FFD runs first — ML is only called if FFD finds no valid box.
    """
    start_time = time.time()

    # ── Fetch order ────────────────────────────────────────────────────────────
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

    zone       = destination_zone if destination_zone != "default" else (order.destination_zone or "zone_b")
    is_fragile = any(
        oi.product.fragility_level in ("fragile", "very_fragile")
        for oi in order_items
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

    # ── Fetch box inventory ────────────────────────────────────────────────────
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

    plan_items     = []
    optimized_cost = 0.0
    overall_eff    = 0.0
    decision_engine = "rule_based"
    decision_reason = ""
    ml_result_data  = None

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: FFD RULE ENGINE — primary decision maker
    # ══════════════════════════════════════════════════════════════════════════
    packed_boxes = first_fit_decreasing(items_to_pack, box_dicts, fragile=is_fragile)

    if packed_boxes:
        # FFD succeeded — use it, done, ML never called
        decision_engine = "rule_based"
        total_box_vol   = 0.0
        for box in packed_boxes:
            cost_data = calculate_total_cost(
                box.length, box.width, box.height,
                box.used_weight, box.cost, zone,
            )
            eff = calculate_efficiency_score(box.used_volume, box.volume)
            optimized_cost += cost_data["total_cost"]
            total_box_vol  += box.volume
            plan_items.append({
                "box_type":          box.box_type,
                "items":             box.packed_items,
                "box_cost":          cost_data["box_cost"],
                "shipping_cost":     cost_data["shipping_cost"],
                "efficiency_score":  eff,
                "dim_weight":        cost_data["dim_weight"],
                "chargeable_weight": cost_data["chargeable_weight"],
            })

        overall_eff = calculate_efficiency_score(total_item_volume, total_box_vol)
        decision_reason = (
            f"FFD rule engine: {len(packed_boxes)} box(es). "
            f"Zone {zone}. "
            f"{'Fragile-safe boxes. ' if is_fragile else ''}"
            f"Chargeable: {sum(p['chargeable_weight'] for p in plan_items):.2f}kg."
        )
        logger.info(
            f"[FFD] Order {order_id}: {len(packed_boxes)} box(es), "
            f"cost=₹{optimized_cost:.0f}, eff={overall_eff:.1%} — ML not needed"
        )

    else:
        # ══════════════════════════════════════════════════════════════════════
        # STEP 2: ML FALLBACK — only reached when FFD finds no valid box
        # ══════════════════════════════════════════════════════════════════════
        logger.warning(
            f"[FFD] Order {order_id}: no valid box found — triggering ML fallback. "
            f"vol={total_item_volume:.0f}cm³ wt={total_actual_weight:.2f}kg "
            f"fragile={is_fragile}"
        )

        ml_box    = None
        ml_reason = "ML fallback triggered (FFD found no valid box)"

        if ml_service.is_ml_available():
            # Use representative item (largest by volume)
            rep = max(items_to_pack, key=lambda i: i.volume)
            try:
                # Validate inputs before calling ML
                ok, errs = ml_service.validate_inputs(
                    rep.length, rep.width, rep.height, rep.weight
                )
                if not ok:
                    raise ValueError(f"Input validation: {'; '.join(errs)}")

                ml_result_data = ml_service.predict_fallback(
                    rep.length, rep.width, rep.height, rep.weight,
                    order_id=order_id,
                )
                ml_box     = ml_result_data["recommended_box"]
                ml_conf    = ml_result_data["confidence_score"]
                ml_agree   = ml_result_data["agreement"]
                ml_reason  = (
                    f"ML fallback (pred_id={ml_result_data['pred_id']}, "
                    f"conf={ml_conf:.0%}, agreement={ml_agree:.0%}): "
                    f"FFD found no valid box, ML predicted {ml_box}."
                )
                decision_engine = "ml_fallback"
                logger.info(
                    f"[ML-Fallback] Order {order_id}: predicted {ml_box} "
                    f"conf={ml_conf:.1%} agree={ml_agree:.0%}"
                )
            except (ValueError, RuntimeError) as e:
                logger.error(f"[ML-Fallback] Order {order_id} failed: {e}")
                ml_box = None

        # Use ML box if available and valid, else use largest box
        if ml_box:
            box_record = next(
                (b for b in box_dicts if b["box_type"] == ml_box),
                None
            )
            if not box_record:
                # ML predicted a box not in inventory — fall back to largest
                box_record = sorted(
                    box_dicts, key=lambda b: b["length"]*b["width"]*b["height"]
                )[-1]
                ml_reason += f" Box {ml_box} not in inventory — using {box_record['box_type']}."
                decision_engine = "last_resort"
        else:
            # Both FFD and ML failed — use largest box
            box_record = sorted(
                box_dicts, key=lambda b: b["length"]*b["width"]*b["height"]
            )[-1]
            ml_reason += f" Using largest available: {box_record['box_type']}."
            decision_engine = "last_resort"

        cost_data = calculate_total_cost(
            box_record["length"], box_record["width"], box_record["height"],
            total_actual_weight, box_record["cost"], zone,
        )
        box_vol     = box_record["length"] * box_record["width"] * box_record["height"]
        overall_eff = calculate_efficiency_score(total_item_volume, box_vol)
        optimized_cost = cost_data["total_cost"]
        decision_reason = ml_reason

        all_items = [
            {"product_id": i.product_id, "name": i.name, "quantity": i.quantity}
            for i in items_to_pack
        ]
        plan_items = [{
            "box_type":          box_record["box_type"],
            "items":             all_items,
            "box_cost":          cost_data["box_cost"],
            "shipping_cost":     cost_data["shipping_cost"],
            "efficiency_score":  overall_eff,
            "dim_weight":        cost_data["dim_weight"],
            "chargeable_weight": cost_data["chargeable_weight"],
        }]

        logger.info(
            f"[Fallback] Order {order_id}: {decision_engine} → "
            f"{box_record['box_type']} cost=₹{optimized_cost:.0f}"
        )

    # ── Baseline cost ──────────────────────────────────────────────────────────
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

    # ── Persist to DB ──────────────────────────────────────────────────────────
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

    for pi in plan_items:
        box_inv = db.query(BoxInventory).filter(
            BoxInventory.box_type == pi["box_type"]
        ).first()
        if box_inv and box_inv.quantity_available > 0:
            box_inv.quantity_available -= 1

    order.status = "completed"
    db.commit()
    db.refresh(plan)

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    logger.info(
        f"[Engine] Order {order_id} done: engine={decision_engine} "
        f"box={plan_items[0]['box_type'] if plan_items else '?'} "
        f"optimized=₹{optimized_cost:.0f} baseline=₹{baseline_cost:.0f} "
        f"saved=₹{savings:.0f} {elapsed_ms}ms"
    )

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
        "ml_prediction":        ml_result_data["recommended_box"] if ml_result_data else None,
        "ml_confidence":        ml_result_data["confidence_score"] if ml_result_data else 0.0,
        "processing_time_ms":   elapsed_ms,
        "alternatives":         [],
    }
