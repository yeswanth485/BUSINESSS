"""
Cost Service — dimensional weight, shipping cost, total cost calculation
"""
from ..core.config import get_settings

settings = get_settings()


def calculate_dimensional_weight(length: float, width: float, height: float) -> float:
    """Standard logistics dimensional weight formula."""
    return (length * width * height) / settings.DIM_WEIGHT_DIVISOR


def calculate_chargeable_weight(actual_weight: float, dim_weight: float) -> float:
    """Couriers charge whichever is higher — actual or dimensional."""
    return max(actual_weight, dim_weight)


def calculate_shipping_cost(chargeable_weight: float, destination_zone: str) -> float:
    """Rate per kg multiplied by chargeable weight."""
    rate = settings.SHIPPING_RATES.get(destination_zone, settings.SHIPPING_RATES["default"])
    return round(rate * chargeable_weight, 2)


def calculate_total_cost(
    length: float,
    width: float,
    height: float,
    actual_weight: float,
    box_cost: float,
    destination_zone: str = "default",
) -> dict:
    """
    Full cost breakdown for a single box shipment.

    Returns:
        dim_weight, chargeable_weight, shipping_cost, box_cost, total_cost
    """
    dim_weight        = calculate_dimensional_weight(length, width, height)
    chargeable_weight = calculate_chargeable_weight(actual_weight, dim_weight)
    shipping_cost     = calculate_shipping_cost(chargeable_weight, destination_zone)
    total_cost        = round(shipping_cost + box_cost, 2)

    return {
        "dim_weight":        round(dim_weight, 3),
        "chargeable_weight": round(chargeable_weight, 3),
        "shipping_cost":     shipping_cost,
        "box_cost":          box_cost,
        "total_cost":        total_cost,
    }


def calculate_efficiency_score(
    item_volume: float,
    box_volume: float,
) -> float:
    """
    Ratio of usable item volume to total box volume.
    Score of 1.0 = perfect fit, 0.0 = empty box.
    """
    if box_volume <= 0:
        return 0.0
    return round(min(item_volume / box_volume, 1.0), 4)
