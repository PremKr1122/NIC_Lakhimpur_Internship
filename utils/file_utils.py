"""
Helpers for validating and persisting uploaded files.
Kept framework-agnostic where possible, but works directly with
FastAPI's UploadFile objects.
"""

import os
import re

from fastapi import HTTPException, UploadFile

from utils.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_PDF_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
)


def _safe_filename(filename: str) -> str:
    """Strip path separators and unsafe characters (secure_filename equivalent)."""
    base = os.path.basename(filename or "upload")
    return re.sub(r"[^\w.\-]", "_", base) or "upload"


def validate_extension(filename: str, allowed: set[str]) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Allowed: {sorted(allowed)}",
        )
    return ext


async def save_upload(upload: UploadFile, folder: str, allowed: set[str]) -> tuple[str, str]:
    """
    Validate extension/size, write the file to `folder`, and return
    (full_path, safe_filename).
    """
    validate_extension(upload.filename, allowed)

    contents = await upload.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File '{upload.filename}' is {size_mb:.1f}MB, exceeds {MAX_UPLOAD_SIZE_MB}MB limit.",
        )

    safe_name = _safe_filename(upload.filename)
    full_path = os.path.join(folder, safe_name)
    with open(full_path, "wb") as f:
        f.write(contents)

    await upload.seek(0)
    return full_path, safe_name


async def save_image_upload(upload: UploadFile, folder: str) -> tuple[str, str]:
    return await save_upload(upload, folder, ALLOWED_IMAGE_EXTENSIONS)


async def save_pdf_upload(upload: UploadFile, folder: str) -> tuple[str, str]:
    return await save_upload(upload, folder, ALLOWED_PDF_EXTENSIONS)
