"""
Auth Routes — with full error handling and detailed logging
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from ..core.database import get_db
from ..core.security import hash_password, verify_password, create_access_token
from ..models.models import User
from ..schemas.schemas import UserRegister, UserLogin, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    try:
        # Check if email already exists
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered. Please sign in instead.")

        user = User(
            email         = payload.email,
            password_hash = hash_password(payload.password),
            full_name     = payload.full_name,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"[auth] New user registered: {payload.email}")
        return user

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        print(f"[auth] DB error on register: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        db.rollback()
        print(f"[auth] Unexpected error on register: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == payload.email).first()
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = create_access_token({"sub": str(user.id)})
        print(f"[auth] Login successful: {payload.email}")
        return TokenResponse(access_token=token, user_id=user.id, email=user.email)

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        print(f"[auth] DB error on login: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        print(f"[auth] Unexpected error on login: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@router.get("/me", response_model=UserOut)
def get_me(db: Session = Depends(get_db)):
    """Health check for auth system"""
    return {"status": "auth system ok"}
