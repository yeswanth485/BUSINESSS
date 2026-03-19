"""
Box Inventory Routes
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..core.database import get_db
from ..core.security import get_current_user
from ..models.models import BoxInventory, User
from ..schemas.schemas import BoxInventoryCreate, BoxInventoryOut

router = APIRouter(prefix="/inventory", tags=["Inventory"])


@router.post("", response_model=BoxInventoryOut, status_code=201)
def add_box(
    payload: BoxInventoryCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    existing = db.query(BoxInventory).filter(BoxInventory.box_type == payload.box_type).first()
    if existing:
        raise HTTPException(status_code=400, detail="Box type already exists. Use PUT to update.")

    box = BoxInventory(**payload.model_dump())
    db.add(box)
    db.commit()
    db.refresh(box)
    return box


@router.get("", response_model=List[BoxInventoryOut])
def list_boxes(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return db.query(BoxInventory).order_by(BoxInventory.box_type).all()


@router.put("/{box_type}/restock")
def restock_box(
    box_type: str,
    quantity: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    box = db.query(BoxInventory).filter(BoxInventory.box_type == box_type).first()
    if not box:
        raise HTTPException(status_code=404, detail="Box type not found")
    box.quantity_available += quantity
    db.commit()
    return {"box_type": box_type, "quantity_available": box.quantity_available}
