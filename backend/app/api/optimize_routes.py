"""
Optimize Packaging Route — returns real baseline, optimized cost, and savings.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.security import get_current_user
from ..models.models import User
from ..schemas.schemas import OptimizeRequest, OptimizeResponse
from ..services.decision_service import optimize_packaging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimize-packaging", tags=["Optimization"])


@router.post("", response_model=OptimizeResponse)
def optimize(
    payload: OptimizeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        result = optimize_packaging(
            order_id         = payload.order_id,
            db               = db,
            destination_zone = payload.destination or "zone_b",
        )
        logger.info(
            f"[API] Order {payload.order_id} optimized: "
            f"baseline=₹{result['baseline_cost']} "
            f"optimized=₹{result['optimized_cost']} "
            f"saved=₹{result['savings']} "
            f"engine={result['decision_engine']}"
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[API] Optimization error for order {payload.order_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
