"""
Analytics Routes — aggregated dashboard metrics
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from ..core.database import get_db
from ..core.security import get_current_user
from ..models.models import Order, PackagingPlan, PackagingPlanItem, BoxInventory, User

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary")
def get_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user_orders = db.query(Order).filter(Order.user_id == current_user.id).all()
    order_ids   = [o.id for o in user_orders]

    if not order_ids:
        return {
            "total_orders":     0,
            "total_cost_saved": 0.0,
            "avg_efficiency":   0.0,
            "waste_percentage": 0.0,
            "box_usage":        {},
            "orders_by_day":    [],
            "efficiency_trend": [],
        }

    plans = db.query(PackagingPlan).filter(PackagingPlan.order_id.in_(order_ids)).all()

    total_cost    = sum(p.total_cost for p in plans)
    avg_eff       = sum(p.efficiency_score for p in plans) / len(plans) if plans else 0
    waste_pct     = round((1 - avg_eff) * 100, 2)

    # Box usage distribution
    plan_ids    = [p.id for p in plans]
    plan_items  = db.query(PackagingPlanItem).filter(PackagingPlanItem.packaging_plan_id.in_(plan_ids)).all()
    box_usage   = {}
    for pi in plan_items:
        box_usage[pi.box_type] = box_usage.get(pi.box_type, 0) + 1

    # Orders by day (last 30 days)
    daily = (
        db.query(
            func.date(Order.created_at).label("day"),
            func.count(Order.id).label("count"),
        )
        .filter(Order.user_id == current_user.id)
        .group_by(func.date(Order.created_at))
        .order_by(func.date(Order.created_at))
        .limit(30)
        .all()
    )
    orders_by_day = [{"day": str(r.day), "count": r.count} for r in daily]

    # Efficiency trend
    eff_trend = [
        {"order_id": p.order_id, "efficiency": round(p.efficiency_score * 100, 2)}
        for p in plans[-20:]
    ]

    # Estimated cost saved vs naive large-box approach (30% savings assumption)
    estimated_naive_cost = total_cost * 1.30
    cost_saved           = round(estimated_naive_cost - total_cost, 2)

    return {
        "total_orders":     len(user_orders),
        "total_cost_saved": cost_saved,
        "avg_efficiency":   round(avg_eff * 100, 2),
        "waste_percentage": waste_pct,
        "box_usage":        box_usage,
        "orders_by_day":    orders_by_day,
        "efficiency_trend": eff_trend,
    }
