"""
PDF-specific helpers: rendering pages to images and extracting/pairing
voter-ID text with detected faces by spatial proximity.
"""

import re

import cv2
import numpy as np
import fitz  # PyMuPDF

from utils.config import VOTER_ID_PATTERN


def page_to_bgr_image(page, dpi: int = 300) -> np.ndarray:
    """Render a PyMuPDF page to a BGR numpy array."""
    pix = page.get_pixmap(dpi=dpi)
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)


def get_text_blocks_from_page(page):
    blocks = []
    for b in page.get_text("blocks"):
        if b[4].strip():
            blocks.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]))
    return blocks


def _clean_voter_id(raw: str) -> str:
    return raw.replace("O", "0").replace("I", "1")


def extract_voter_id_blocks(page_text_blocks):
    """Return [(voter_id, top_y), ...] sorted top-to-bottom for a page."""
    id_blocks = []
    for block in page_text_blocks:
        for match in re.findall(VOTER_ID_PATTERN, block[4]):
            cleaned = _clean_voter_id(match)
            if len(cleaned) >= 8 and not cleaned.isdigit():
                id_blocks.append((cleaned, block[1]))
    id_blocks.sort(key=lambda b: b[1])
    return id_blocks


def extract_voterids_by_proximity(faces, page_text_blocks):
    """
    Pair each detected face with the nearest not-yet-used voter ID text block
    on the page (used by the /match endpoint).
    """
    id_blocks = []
    for block in page_text_blocks:
        for match in re.findall(VOTER_ID_PATTERN, block[4]):
            cleaned = _clean_voter_id(match)
            if len(cleaned) >= 8 and not cleaned.isdigit():
                bx, by, bx2, by2 = block[:4]
                id_blocks.append(
                    {
                        "voter_id": cleaned,
                        "center": ((bx + bx2) // 2, (by + by2) // 2),
                        "assigned": False,
                        "top": by,
                        "left": bx,
                    }
                )

    id_blocks.sort(key=lambda b: (b["top"], b["left"]))

    face_voter_ids = [""] * len(faces)
    face_indices = sorted(range(len(faces)), key=lambda i: (faces[i].bbox[1], faces[i].bbox[0]))

    for i in face_indices:
        face = faces[i]
        x, y, x2, y2 = face.bbox
        face_center = ((x + x2) // 2, (y + y2) // 2)
        min_dist = float("inf")
        best_idx = -1
        for idx, id_block in enumerate(id_blocks):
            if id_block["assigned"]:
                continue
            dist = np.linalg.norm(np.array(face_center) - np.array(id_block["center"]))
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        if best_idx != -1:
            id_blocks[best_idx]["assigned"] = True
            face_voter_ids[i] = id_blocks[best_idx]["voter_id"]

    return face_voter_ids
