"""
Decision Engine — ML-primary, FFD-validated hybrid.

Flow for every order:
  Step 1: ML predicts the best box (learned from 3000+ training samples)
  Step 2: FFD validates — does that box physically fit? (volume + weight check)
  Step 3a: ML box fits → use it. Engine = "ml_primary"
  Step 3b: ML box too small → FFD upgrades to smallest valid box. Engine = "ml_ffd_corrected"
  Step 3c: ML unavailable → FFD handles alone. Engine = "rule_based"

Result: ML intelligence + FFD precision = best possible box every time.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..models.models import (
    Order, OrderItem, Product, BoxInventory,
    PackagingPlan, PackagingPlanItem
)
from .packing_service import Item, Box, first_fit_decreasing
from .cost_service import (
    calculate_total_cost, calculate_efficiency_score,
    calculate_baseline_cost, calculate_cost_savings, DEFAULT_BOXES
)
from . import ml_service

logger = logging.getLogger(__name__)

# Box catalogue ordered smallest → largest (for FFD validation)
BOX_ORDER = ["Box_XS", "Box_S", "Box_M", "Box_L", "Box_XL", "Box_XXL"]


def _box_fits(box: dict, item_volume: float, total_weight: float, is_fragile: bool) -> bool:
    """Check if a box physically fits the items — volume AND weight AND fragility."""
    box_vol = box["length"] * box["width"] * box["height"]
    if box_vol < item_volume:
        return False
    if box["max_weight"] < total_weight:
        return False
    if is_fragile and not box.get("suitable_fragile", False):
        return False
    return True


def _ffd_validate_and_correct(
    ml_box_type: str,
    item_volume: float,
    total_weight: float,
    is_fragile: bool,
    box_dicts: list,
) -> tuple[dict, str, str]:
    """
    FFD validation of ML prediction.
    Returns: (selected_box_dict, engine_label, correction_note)
    """
    eligible = sorted(
        [b for b in box_dicts if not is_fragile or b.get("suitable_fragile", False)],
        key=lambda b: BOX_ORDER.index(b["box_type"]) if b["box_type"] in BOX_ORDER
                      else b["length"] * b["width"] * b["height"]
    )
    if not eligible:
        eligible = sorted(box_dicts, key=lambda b: b["length"] * b["width"] * b["height"])

    # Find ML-predicted box in inventory
    ml_box = next((b for b in eligible if b["box_type"] == ml_box_type), None)

    if ml_box and _box_fits(ml_box, item_volume, total_weight, is_fragile):
        # ML prediction is physically valid — use it
        return ml_box, "ml_primary", f"ML predicted {ml_box_type}, FFD confirmed it fits."

    # ML box doesn't fit — FFD finds smallest valid box
    for box in eligible:
        if _box_fits(box, item_volume, total_weight, is_fragile):
            if ml_box:
                note = (
                    f"ML predicted {ml_box_type} but it's too small "
                    f"(vol={item_volume:.0f}cm³ > box={ml_box['length']*ml_box['width']*ml_box['height']:.0f}cm³). "
                    f"FFD upgraded to {box['box_type']}."
                )
            else:
                note = f"ML predicted {ml_box_type} (not in inventory). FFD selected {box['box_type']}."
            return box, "ml_ffd_corrected", note

    # Nothing fits — use largest available
    largest = eligible[-1]
    return largest, "ml_ffd_corrected", f"No box fits items. Using largest available: {largest['box_type']}."


def optimize_packaging(order_id: int, db: Session, destination_zone: str = "zone_b") -> Dict[str, Any]:
    """
    ML-primary, FFD-validated optimization.
    Every order gets ML intelligence checked by FFD physics.
    """
    start_time = time.time()

    # ── Fetch and validate order ──────────────────────────────────────────────
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
        item.product.fragility_level in ("fragile", "very_fragile")
        for item in order_items
    )

    items_to_pack:      List[Item] = []
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

    # ═════════════════════════════════════════════════════════════════
    # STEP 1: ML PREDICTION (primary decision maker)
    # ═════════════════════════════════════════════════════════════════
    ml_prediction  = None
    ml_confidence  = 0.0
    ml_model_used  = "none"
    ml_all_votes   = None
    decision_engine = "rule_based"
    plan_items      = []
    optimized_cost  = 0.0
    overall_eff     = 0.0

    # Use representative item for ML (largest by volume)
    rep_item = max(items_to_pack, key=lambda i: i.volume)

    ml_all_votes = None
    if ml_service.is_ml_available():
        try:
            # Run ALL 5 models — each evaluates independently
            ml_all_votes   = ml_service.predict_all_models(
                rep_item.length, rep_item.width,
                rep_item.height, rep_item.weight,
            )
            ml_prediction  = ml_all_votes["voted_box"]
            ml_confidence  = ml_all_votes["vote_confidence"]
            ml_model_used  = f"5-model-vote"
            logger.info(
                f"[ML] Order {order_id}: voted={ml_prediction} "
                f"agreement={ml_all_votes['agreement']:.0%} "
                f"conf={ml_confidence:.1%} "
                f"votes={ml_all_votes['model_votes']}"
            )
        except Exception as e:
            logger.warning(f"[ML] Prediction failed for order {order_id}: {e}")

    # ═════════════════════════════════════════════════════════════════
    # STEP 2: FFD VALIDATION of ML prediction
    # ═════════════════════════════════════════════════════════════════
    if ml_prediction:
        selected_box, decision_engine, correction_note = _ffd_validate_and_correct(
            ml_box_type   = ml_prediction,
            item_volume   = total_item_volume,
            total_weight  = total_actual_weight,
            is_fragile    = is_fragile,
            box_dicts     = box_dicts,
        )

        # Calculate cost for the validated box
        cost_data = calculate_total_cost(
            selected_box["length"], selected_box["width"], selected_box["height"],
            total_actual_weight, selected_box["cost"], zone,
        )
        box_vol      = selected_box["length"] * selected_box["width"] * selected_box["height"]
        overall_eff  = calculate_efficiency_score(total_item_volume, box_vol)
        optimized_cost = cost_data["total_cost"]

        # Build plan items — all items packed in one ML-selected box
        all_items = [
            {"product_id": i.product_id, "name": i.name, "quantity": i.quantity}
            for i in items_to_pack
        ]
        plan_items = [{
            "box_type":          selected_box["box_type"],
            "items":             all_items,
            "box_cost":          cost_data["box_cost"],
            "shipping_cost":     cost_data["shipping_cost"],
            "efficiency_score":  overall_eff,
            "dim_weight":        cost_data["dim_weight"],
            "chargeable_weight": cost_data["chargeable_weight"],
        }]

        # Build decision explanation
        conf_pct = f"{ml_confidence:.0%}"
        if decision_engine == "ml_primary":
            decision_reason = (
                f"ML ({ml_model_used}, {conf_pct} confidence) → {selected_box['box_type']}. "
                f"FFD validated: fits {total_item_volume:.0f}cm³ in "
                f"{box_vol:.0f}cm³ box ({overall_eff*100:.0f}% space used). "
                f"Zone {zone}. {'Fragile-safe. ' if is_fragile else ''}"
                f"Chargeable: {cost_data['chargeable_weight']:.2f}kg."
            )
        else:
            decision_reason = (
                f"ML ({ml_model_used}, {conf_pct}) → FFD corrected. "
                f"{correction_note} "
                f"Final: {selected_box['box_type']} ({overall_eff*100:.0f}% eff). "
                f"Zone {zone}."
            )

        logger.info(
            f"[Engine] Order {order_id}: {decision_engine} → {selected_box['box_type']} "
            f"| eff={overall_eff:.1%} | ₹{optimized_cost:.0f} | {correction_note}"
        )

    else:
        # ═════════════════════════════════════════════════════════════
        # STEP 3: FFD-only (ML unavailable)
        # ═════════════════════════════════════════════════════════════
        decision_engine = "rule_based"
        packed_boxes    = first_fit_decreasing(items_to_pack, box_dicts, fragile=is_fragile)

        if packed_boxes:
            total_box_vol = 0.0
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
                f"FFD rule engine (ML unavailable): {len(packed_boxes)} box(es). "
                f"Zone {zone}. {'Fragile-safe. ' if is_fragile else ''}"
                f"Chargeable: {sum(p['chargeable_weight'] for p in plan_items):.2f}kg."
            )
        else:
            # Absolute last resort
            eligible = sorted(box_dicts, key=lambda b: b["length"]*b["width"]*b["height"])
            fallback  = eligible[-1]
            cost_data = calculate_total_cost(
                fallback["length"], fallback["width"], fallback["height"],
                total_actual_weight, fallback["cost"], zone,
            )
            optimized_cost = cost_data["total_cost"]
            overall_eff    = calculate_efficiency_score(
                total_item_volume, fallback["length"]*fallback["width"]*fallback["height"]
            )
            plan_items = [{
                "box_type":          fallback["box_type"],
                "items":             [{"product_id": i.product_id, "name": i.name, "quantity": i.quantity} for i in items_to_pack],
                "box_cost":          cost_data["box_cost"],
                "shipping_cost":     cost_data["shipping_cost"],
                "efficiency_score":  overall_eff,
                "dim_weight":        cost_data["dim_weight"],
                "chargeable_weight": cost_data["chargeable_weight"],
            }]
            decision_reason = f"Fallback: used largest box {fallback['box_type']}. Zone {zone}."

    # ── Baseline cost (what warehouse pays without PackAI) ────────────────────
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

    # ── Persist to DB ─────────────────────────────────────────────────────────
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
        box_inv = db.query(BoxInventory).filter(BoxInventory.box_type == pi["box_type"]).first()
        if box_inv and box_inv.quantity_available > 0:
            box_inv.quantity_available -= 1

    order.status = "completed"
    db.commit()
    db.refresh(plan)

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    logger.info(
        f"[Engine] Order {order_id} done: {decision_engine} "
        f"| box={plan_items[0]['box_type'] if plan_items else '?'} "
        f"| optimized=₹{optimized_cost:.0f} baseline=₹{baseline_cost:.0f} saved=₹{savings:.0f} "
        f"| {elapsed_ms}ms"
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
        "ml_prediction":        ml_prediction,
        "ml_confidence":        round(ml_confidence, 4),
        "ml_model_used":        ml_model_used,
        "ml_all_votes":         ml_all_votes,
        "processing_time_ms":   elapsed_ms,
        "alternatives":         [],
    }


def _best_fit_fallback(item: Item, box_dicts: list) -> str:
    item_vol = item.length * item.width * item.height
    for b in sorted(box_dicts, key=lambda b: b["length"]*b["width"]*b["height"]):
        if b["length"]*b["width"]*b["height"] >= item_vol and b["max_weight"] >= item.weight:
            return b["box_type"]
    return sorted(box_dicts, key=lambda b: b["length"]*b["width"]*b["height"])[-1]["box_type"]
