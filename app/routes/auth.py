from fastapi import APIRouter, Depends, HTTPException
from fastapi_limiter.depends import RateLimiter
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import UserModel
from app.schemas.auth import Token, User, UserLogin, UserRegister
from app.utils import create_access_token, get_password_hash, validate_password_length, verify_password

router = APIRouter()


@router.post("/api/auth/register", response_model=Token)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(UserModel).filter(UserModel.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = UserModel(
        email=user_data.email,
        name=user_data.name,
        password=get_password_hash(user_data.password),
        role=user_data.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    user_obj = User(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        created_at=user.created_at.isoformat(),
    )
    return Token(access_token=access_token, token_type="bearer", user=user_obj)


@router.post("/api/auth/login", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    validate_password_length(user_data.password)

    user = db.query(UserModel).filter(UserModel.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    user_obj = User(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        created_at=user.created_at.isoformat(),
    )
    return Token(access_token=access_token, token_type="bearer", user=user_obj)
