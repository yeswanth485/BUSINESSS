"""
Packing Service — First Fit Decreasing (FFD) bin packing algorithm.

Groups multiple products into the minimum number of boxes,
minimising unused space and total cost.
"""
from typing import List, Dict, Any, Optional


class Item:
    def __init__(self, product_id: int, name: str, length: float, width: float,
                 height: float, weight: float, quantity: int = 1):
        self.product_id = product_id
        self.name       = name
        self.length     = length
        self.width      = width
        self.height     = height
        self.weight     = weight
        self.quantity   = quantity
        self.volume     = length * width * height * quantity


class Box:
    def __init__(self, box_type: str, length: float, width: float, height: float,
                 max_weight: float, cost: float):
        self.box_type    = box_type
        self.length      = length
        self.width       = width
        self.height      = height
        self.max_weight  = max_weight
        self.cost        = cost
        self.volume      = length * width * height
        self.used_volume = 0.0
        self.used_weight = 0.0
        self.packed_items: List[Dict] = []

    @property
    def remaining_volume(self) -> float:
        return self.volume - self.used_volume

    @property
    def remaining_weight(self) -> float:
        return self.max_weight - self.used_weight

    def can_fit(self, item: Item) -> bool:
        item_vol    = item.volume
        item_weight = item.weight * item.quantity
        return (item_vol <= self.remaining_volume and
                item_weight <= self.remaining_weight)

    def pack(self, item: Item):
        self.used_volume += item.volume
        self.used_weight += item.weight * item.quantity
        self.packed_items.append({
            "product_id": item.product_id,
            "name":       item.name,
            "quantity":   item.quantity,
        })


def first_fit_decreasing(
    items: List[Item],
    available_boxes: List[Dict[str, Any]],
    fragile: bool = False,
) -> Optional[List[Box]]:
    """
    FFD Algorithm:
    1. Sort items by volume descending (largest first)
    2. For each item, find the first box that fits
    3. If no existing box fits, open a new one (smallest valid box)
    4. Return list of packed boxes or None if impossible

    Args:
        items:           List of Item objects to pack
        available_boxes: Box inventory records from DB
        fragile:         Whether to restrict to fragile-safe boxes

    Returns:
        List of packed Box objects, or None if packing fails
    """
    # Filter boxes by fragility requirement
    eligible_boxes = [
        b for b in available_boxes
        if (not fragile or b.get("suitable_fragile", False))
        and b["quantity_available"] > 0
    ]

    if not eligible_boxes:
        return None

    # Sort boxes smallest to largest volume (we open the smallest that fits)
    eligible_boxes_sorted = sorted(
        eligible_boxes,
        key=lambda b: b["length"] * b["width"] * b["height"]
    )

    # Sort items by volume descending — FFD core heuristic
    sorted_items = sorted(items, key=lambda i: i.volume, reverse=True)

    opened_boxes: List[Box] = []

    for item in sorted_items:
        placed = False

        # Try to fit into an already-opened box
        for box in opened_boxes:
            if box.can_fit(item):
                box.pack(item)
                placed = True
                break

        # No existing box fits — open the smallest new box that can hold this item
        if not placed:
            item_vol    = item.volume
            item_weight = item.weight * item.quantity

            suitable = [
                b for b in eligible_boxes_sorted
                if (b["length"] * b["width"] * b["height"]) >= item_vol
                and b["max_weight"] >= item_weight
            ]

            if not suitable:
                # Item cannot fit in ANY available box
                return None

            new_box_data = suitable[0]
            new_box = Box(
                box_type   = new_box_data["box_type"],
                length     = new_box_data["length"],
                width      = new_box_data["width"],
                height     = new_box_data["height"],
                max_weight = new_box_data["max_weight"],
                cost       = new_box_data["cost"],
            )
            new_box.pack(item)
            opened_boxes.append(new_box)

    return opened_boxes if opened_boxes else None
