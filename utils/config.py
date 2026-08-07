"""
Central configuration: filesystem paths, thresholds, and one-time setup.
Importing this module guarantees every required folder/file exists.
"""

import os
import csv
import pickle

# Paths

UPLOAD_FOLDER_A = "static/dataset_a/images"   # known faces
UPLOAD_FOLDER_B = "static/dataset_b"          # faces extracted from uploads
PDF_UPLOADS = "static/uploads"

DATA_CSV = "static/dataset_a/metadata.csv"
EMBEDDINGS_PKL = "static/dataset_a/embeddings.pkl"
VOTERID_CSV = "static/dataset_a/voterids.csv"
MAPPING_CSV = os.path.join("mappings", "voterid_face_mapping.csv")

# Matching / validation constants

VOTER_ID_PATTERN = r"\b[A-Z]{2,4}[0-9]{5,}\b"
SIMILARITY_THRESHOLD = 55        # % similarity to flag as a match
DUPLICATE_THRESHOLD = 0.95       # cosine similarity to reject as duplicate on /add
FACE_CROP_MARGIN = 40
FACE_RESIZE_DIM = (224, 224)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_PDF_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_MB = 25

# InsightFace model location

MODEL_NAME = "buffalo_l"
MODEL_ROOT = os.environ.get("INSIGHTFACE_MODEL_ROOT")  


def _repair_missing_header(file_path: str, fieldnames: list) -> None:
    """
    If the file exists but its first line isn't the expected header
    (e.g. legacy data carried over without one), prepend it.
    This is what silently broke metadata.csv/voterids.csv before: they existed,
    so the "create if missing" check below skipped them, even though they had
    no header row at all.
    """
    with open(file_path, newline="", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()

    expected_header = ",".join(fieldnames)
    if first_line == expected_header:
        return  # header already correct, nothing to do

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        existing_content = f.read()

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        f.write(expected_header + "\n" + existing_content)

    print(f"[config] Repaired missing/incorrect header in '{file_path}' -> {expected_header}")


def ensure_storage_ready() -> None:
    """Create required directories and seed/repair CSV/pickle files."""
    for folder in (UPLOAD_FOLDER_A, UPLOAD_FOLDER_B, PDF_UPLOADS, "mappings"):
        os.makedirs(folder, exist_ok=True)

    for file_path, fieldnames in [
        (DATA_CSV, ["filename", "voter_id", "matched_voter_id"]),
        (VOTERID_CSV, ["filename", "voter_id", "serial_number"]),
    ]:
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        elif os.path.getsize(file_path) > 0:
            _repair_missing_header(file_path, fieldnames)

    if not os.path.exists(EMBEDDINGS_PKL):
        with open(EMBEDDINGS_PKL, "wb") as f:
            pickle.dump([], f)


# Run on import so every module that touches storage can rely on it existing.
ensure_storage_ready()