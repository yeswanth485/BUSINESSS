"""
Cost Service — Real cost calculation engine.
Calculates BASELINE cost (without optimization) and OPTIMIZED cost (with PackAI).
Savings = baseline - optimized.

Indian logistics pricing:
  Chargeable weight = max(actual_weight, dim_weight)
  Dim weight = (L x W x H) / 5000  [cm, kg]
  Shipping cost = chargeable_weight x zone_rate
  Total cost = shipping_cost + box_cost
"""
from ..core.config import get_settings

settings = get_settings()

# Standard Indian ecommerce box catalogue — ordered smallest to largest
DEFAULT_BOXES = [
    {"box_type": "Box_XS",  "length": 15, "width": 10, "height": 5,  "max_weight": 1.5, "cost": 12,  "suitable_fragile": False},
    {"box_type": "Box_S",   "length": 25, "width": 20, "height": 15, "max_weight": 5,   "cost": 22,  "suitable_fragile": False},
    {"box_type": "Box_M",   "length": 35, "width": 25, "height": 20, "max_weight": 12,  "cost": 38,  "suitable_fragile": True},
    {"box_type": "Box_L",   "length": 50, "width": 40, "height": 30, "max_weight": 20,  "cost": 62,  "suitable_fragile": True},
    {"box_type": "Box_XL",  "length": 70, "width": 50, "height": 40, "max_weight": 30,  "cost": 90,  "suitable_fragile": True},
    {"box_type": "Box_XXL", "length": 90, "width": 70, "height": 60, "max_weight": 50,  "cost": 135, "suitable_fragile": True},
]


def calculate_dimensional_weight(length: float, width: float, height: float) -> float:
    if length <= 0 or width <= 0 or height <= 0:
        return 0.0
    return round((length * width * height) / settings.DIM_WEIGHT_DIVISOR, 3)


def calculate_chargeable_weight(actual_weight: float, dim_weight: float) -> float:
    return round(max(actual_weight, dim_weight), 3)


def calculate_shipping_cost(chargeable_weight: float, destination_zone: str) -> float:
    if chargeable_weight <= 0:
        return 0.0
    rate = settings.get_shipping_rate(destination_zone)
    return round(rate * chargeable_weight, 2)


def calculate_total_cost(
    length: float, width: float, height: float,
    actual_weight: float, box_cost: float, destination_zone: str = "default"
) -> dict:
    dim_weight        = calculate_dimensional_weight(length, width, height)
    chargeable_weight = calculate_chargeable_weight(actual_weight, dim_weight)
    shipping_cost     = calculate_shipping_cost(chargeable_weight, destination_zone)
    total_cost        = round(shipping_cost + box_cost, 2)
    return {
        "dim_weight":         dim_weight,
        "chargeable_weight":  chargeable_weight,
        "shipping_cost":      shipping_cost,
        "box_cost":           round(box_cost, 2),
        "total_cost":         total_cost,
    }


def calculate_efficiency_score(item_volume: float, box_volume: float) -> float:
    if box_volume <= 0 or item_volume <= 0:
        return 0.0
    return round(min(item_volume / box_volume, 1.0), 4)


def calculate_baseline_cost(
    actual_weight: float,
    item_volume: float,
    destination_zone: str = "default",
    is_fragile: bool = False,
    available_boxes: list = None
) -> dict:
    """
    BASELINE COST = what the warehouse would pay WITHOUT PackAI.
    Logic: pick the NEXT SIZE UP from what PackAI chose — 
    this simulates the common warehouse mistake of using a too-large box.
    
    If no available_boxes provided, use DEFAULT_BOXES.
    """
    boxes = available_boxes if available_boxes else DEFAULT_BOXES
    eligible = [b for b in boxes if not is_fragile or b.get("suitable_fragile", False)]
    if not eligible:
        eligible = DEFAULT_BOXES

    # Sort by volume ascending
    eligible_sorted = sorted(eligible, key=lambda b: b["length"] * b["width"] * b["height"])

    # Find smallest box that fits (PackAI choice)
    optimized_box = None
    for b in eligible_sorted:
        bvol = b["length"] * b["width"] * b["height"]
        if bvol >= item_volume and b["max_weight"] >= actual_weight:
            optimized_box = b
            break

    if not optimized_box:
        optimized_box = eligible_sorted[-1]

    # Baseline = next box up (simulates naive warehouse choice)
    opt_idx = eligible_sorted.index(optimized_box)
    baseline_box = eligible_sorted[min(opt_idx + 1, len(eligible_sorted) - 1)]

    baseline_result = calculate_total_cost(
        baseline_box["length"], baseline_box["width"], baseline_box["height"],
        actual_weight, baseline_box["cost"], destination_zone
    )
    return {
        "baseline_box":   baseline_box["box_type"],
        "baseline_cost":  baseline_result["total_cost"],
        "baseline_ship":  baseline_result["shipping_cost"],
        "baseline_box_c": baseline_result["box_cost"],
    }


def calculate_cost_savings(optimized_cost: float, naive_cost: float) -> dict:
    if naive_cost <= 0:
        return {"savings_inr": 0.0, "savings_percent": 0.0}
    savings_inr     = round(naive_cost - optimized_cost, 2)
    savings_percent = round((savings_inr / naive_cost) * 100, 2)
    return {
        "savings_inr":     max(savings_inr, 0.0),
        "savings_percent": max(savings_percent, 0.0),
    }
