"""
Face detection & embedding engine.
Wraps InsightFace so the rest of the app never touches the model directly.
"""

from functools import lru_cache

import numpy as np
import cv2
from insightface.app import FaceAnalysis

from utils.config import FACE_CROP_MARGIN, FACE_RESIZE_DIM, MODEL_NAME, MODEL_ROOT


@lru_cache(maxsize=1)
def get_face_model() -> FaceAnalysis:
    """
    Load the InsightFace model once per process.

    If MODEL_ROOT is set (see utils/config.py), InsightFace loads weights from
    <MODEL_ROOT>/models/<MODEL_NAME>/ instead of downloading to ~/.insightface.
    """
    kwargs = {"name": MODEL_NAME, "providers": ["CPUExecutionProvider"]}
    if MODEL_ROOT:
        kwargs["root"] = MODEL_ROOT
        print(f"[face_engine] Loading '{MODEL_NAME}' from local root: {MODEL_ROOT}")
    else:
        print(f"[face_engine] Loading '{MODEL_NAME}' from default InsightFace cache (~/.insightface)")

    model = FaceAnalysis(**kwargs)
    model.prepare(ctx_id=0)
    return model


def detect_faces(img_bgr: np.ndarray):
    """Return a list of detected face objects for a BGR image array."""
    return get_face_model().get(img_bgr)


def crop_with_margin(img_np: np.ndarray, bbox, margin: int = FACE_CROP_MARGIN) -> np.ndarray:
    x, y, x2, y2 = map(int, bbox)
    h, w = img_np.shape[:2]
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    x2 = min(x2 + margin, w)
    y2 = min(y2 + margin, h)
    return img_np[y:y2, x:x2]


def get_embedding_for_crop(face_crop: np.ndarray):
    """
    Resize a face crop to the standard size and return its embedding,
    or None if no face could be re-detected/aligned in the crop.
    """
    resized = cv2.resize(face_crop, FACE_RESIZE_DIM)
    aligned = detect_faces(resized)
    if not aligned:
        return None, resized
    return aligned[0].embedding, resized


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def sort_faces_row_wise(faces, row_tolerance: int = 50):
    """Sort detected faces top-to-bottom, then left-to-right within each row."""
    faces = sorted(faces, key=lambda f: f.bbox[1])
    rows = []
    current_row = []

    for face in faces:
        y_center = (face.bbox[1] + face.bbox[3]) // 2
        if not current_row:
            current_row.append((face, y_center))
        else:
            prev_y = current_row[0][1]
            if abs(y_center - prev_y) <= row_tolerance:
                current_row.append((face, y_center))
            else:
                rows.append(current_row)
                current_row = [(face, y_center)]
    if current_row:
        rows.append(current_row)

    sorted_faces = []
    for row in rows:
        row_sorted = sorted(row, key=lambda item: item[0].bbox[0])
        sorted_faces.extend(item[0] for item in row_sorted)

    return sorted_faces
