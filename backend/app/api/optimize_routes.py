"""
Optimize Packaging Route
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.security import get_current_user
from ..models.models import User
from ..schemas.schemas import OptimizeRequest, OptimizeResponse
from ..services.decision_service import optimize_packaging

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
            destination_zone = payload.destination or "default",
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
