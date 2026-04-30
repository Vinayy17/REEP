import io
import json
import os

import cloudinary
import cloudinary.uploader
import qrcode
from sqlalchemy.orm import Session

from app.models.catalog import ProductModel

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True,
)


def upload_image_to_cloudinary(file, folder="products"):
    result = cloudinary.uploader.upload(file, folder=folder, resource_type="image")
    return result["secure_url"]


def upload_qr_to_cloudinary(pil_image, folder="qr"):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    result = cloudinary.uploader.upload(buffer, folder=folder, resource_type="image")
    return result["secure_url"]


def generate_qr(data: dict):
    qr_text = json.dumps(data, separators=(",", ":"))
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=8,
        border=2,
    )
    qr.add_data(qr_text)
    qr.make(fit=True)

    image = qr.make_image(fill_color="black", back_color="white")
    return upload_qr_to_cloudinary(image, folder="product_qr")


def generate_product_code(db: Session, prefix: str = "RR") -> str:
    last_code = (
        db.query(ProductModel.product_code)
        .filter(ProductModel.product_code.like(f"{prefix}-%"))
        .order_by(ProductModel.product_code.desc())
        .first()
    )

    next_number = 1 if not last_code else int(last_code[0].split("-")[1]) + 1
    return f"{prefix}-{next_number:04d}"


__all__ = [
    "generate_product_code",
    "generate_qr",
    "upload_image_to_cloudinary",
    "upload_qr_to_cloudinary",
]
