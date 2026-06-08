"""/api/predict endpoint — thin handler that validates input and calls the service."""
import os
import uuid
import asyncio
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps

from src.api import state
from src.api.services.predict_service import run_predict_pipeline

router = APIRouter()


@router.post("/api/predict")
async def api_predict(
    image: UploadFile = File(...),
    threshold: float = Form(0.5),
    enable_gradcam: str = Form("true"),
    view_position: str = Form("auto"),
):
    if state.cnn_model is None:
        raise HTTPException(
            status_code=503,
            detail="CNN model chưa được load. Hãy train model trước."
        )

    # Validate file
    if not image.filename:
        raise HTTPException(status_code=400, detail="Chưa chọn file.")

    content_type = (image.content_type or "").strip().lower()
    ext = (os.path.splitext(image.filename)[1] or ".jpg").lower()
    allowed_image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File không phải ảnh hợp lệ.")
    if (not content_type) and ext not in allowed_image_exts:
        raise HTTPException(status_code=400, detail="Định dạng ảnh không được hỗ trợ.")

    # Parse params
    do_gradcam = enable_gradcam.lower() == "true"
    if not (0.0 <= float(threshold) <= 1.0):
        raise HTTPException(status_code=400, detail="Threshold phải trong [0, 1].")

    # Lưu file tạm
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(state.UPLOAD_FOLDER, filename)

    try:
        max_upload_bytes = state.resolve_max_upload_bytes(state.config)
        content_length = image.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File quá lớn. Giới hạn {max_upload_bytes // (1024 * 1024)}MB.",
                    )
            except ValueError:
                pass

        contents = await image.read()
        if len(contents) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File quá lớn. Giới hạn {max_upload_bytes // (1024 * 1024)}MB.",
            )
        with open(filepath, "wb") as f:
            f.write(contents)

        # Verify image integrity and capture input resolution.
        input_w, input_h = 0, 0
        try:
            with Image.open(filepath) as im_verify:
                im_verify.verify()
            with Image.open(filepath) as im:
                im = ImageOps.exif_transpose(im)
                input_w, input_h = im.size
        except Exception:
            raise HTTPException(status_code=400, detail="Không đọc được ảnh hợp lệ (PNG/JPG).")

        response = await asyncio.to_thread(
            run_predict_pipeline,
            filepath,
            image.filename,
            filename,
            float(threshold),
            do_gradcam,
            view_position,
            input_w,
            input_h,
        )

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
    finally:
        # Xoá file upload tạm
        if os.path.exists(filepath):
            os.unlink(filepath)
