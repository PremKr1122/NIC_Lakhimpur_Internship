"""
Persistence layer: known-face embeddings + their CSV metadata.
Every function here is the single source of truth for reading/writing
DATA_CSV, VOTERID_CSV and EMBEDDINGS_PKL.
"""

import csv
import os
import pickle
from typing import List, Optional

import cv2
import numpy as np

from utils.config import (
    DATA_CSV,
    EMBEDDINGS_PKL,
    UPLOAD_FOLDER_A,
    VOTERID_CSV,
)
from utils.face_engine import cosine_similarity, get_embedding_for_crop, detect_faces
from utils.config import DUPLICATE_THRESHOLD


def load_dataset() -> List[dict]:
    """Load all known face embeddings joined with their voter IDs."""
    if not os.path.exists(EMBEDDINGS_PKL) or not os.path.exists(DATA_CSV):
        return []
    with open(EMBEDDINGS_PKL, "rb") as f:
        embeddings = pickle.load(f)
    # utf-8-sig transparently strips a BOM if the file has one (e.g. saved via Excel),
    # which otherwise turns the first header key into '\ufefffilename' and breaks lookups.
    with open(DATA_CSV, newline="", encoding="utf-8-sig") as f:
        metadata = list(csv.DictReader(f))

    if metadata and "filename" not in metadata[0]:
        raise RuntimeError(
            f"'{DATA_CSV}' has an unexpected header {list(metadata[0].keys())}, "
            "expected a 'filename' column. The file may be corrupted or in the wrong format."
        )

    filename_to_voterid = {row["filename"]: row["voter_id"] for row in metadata}
    return [
        {
            "embedding": emb,
            "filename": meta["filename"],
            "voter_id": filename_to_voterid.get(meta["filename"], ""),
        }
        for emb, meta in zip(embeddings, metadata)
    ]


def read_metadata_rows() -> List[dict]:
    with open(DATA_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_metadata_rows(rows: List[dict]) -> None:
    with open(DATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id", "matched_voter_id"])
        writer.writeheader()
        writer.writerows(rows)


def append_embedding_to_store(embedding: np.ndarray) -> None:
    embeddings = []
    if os.path.exists(EMBEDDINGS_PKL):
        with open(EMBEDDINGS_PKL, "rb") as f:
            embeddings = pickle.load(f)
    embeddings.append(embedding)
    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(embeddings, f)


def save_face_and_voterid(face_crop: np.ndarray, filename: str, voter_id: str, serial: Optional[int] = None) -> None:
    cv2.imwrite(os.path.join(UPLOAD_FOLDER_A, filename), face_crop)
    with open(DATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id"])
        writer.writerow({"filename": filename, "voter_id": voter_id})
    with open(VOTERID_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id", "serial_number"])
        writer.writerow({"filename": filename, "voter_id": voter_id, "serial_number": serial if serial is not None else ""})


def register_new_face(img_path: str, voter_id: str, threshold: float = DUPLICATE_THRESHOLD):
    """
    Used by the /add endpoint. Detects the primary face in `img_path`,
    rejects it as a duplicate if too similar to an existing entry,
    otherwise stores its embedding + metadata.

    Returns (success: bool, message: str).
    """
    img = cv2.imread(img_path)
    if img is None:
        return False, "❌ Could not read the uploaded image."

    faces = detect_faces(img)
    if not faces:
        return False, "❌ No face detected in the uploaded photo."

    from utils.face_engine import crop_with_margin  # local import avoids a cycle at module load

    face_crop = crop_with_margin(img, faces[0].bbox)
    embedding, _ = get_embedding_for_crop(face_crop)
    if embedding is None:
        return False, "❌ Could not align the detected face. Try a clearer photo."

    for entry in load_dataset():
        if cosine_similarity(entry["embedding"], embedding) > threshold:
            return False, "⚠️ Duplicate image detected. Entry not added."

    append_embedding_to_store(embedding)

    filename = os.path.basename(img_path)
    with open(DATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id"])
        writer.writerow({"filename": filename, "voter_id": voter_id})
    with open(VOTERID_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id", "serial_number"])
        writer.writerow({"filename": filename, "voter_id": voter_id, "serial_number": ""})

    return True, "✅ Entry added successfully."


def write_mapping_csv(path: str, rows: List[dict]) -> None:
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["voter_id", "image"])
        writer.writeheader()
        writer.writerows(rows)