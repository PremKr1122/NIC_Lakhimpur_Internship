"""
FastAPI backend for face matching / duplicate-entry detection.

Run with:
    uvicorn main:app --reload
"""

import os
from typing import List

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF

from utils.config import (
    MAPPING_CSV,
    SIMILARITY_THRESHOLD,
    UPLOAD_FOLDER_A,
    UPLOAD_FOLDER_B,
    PDF_UPLOADS,
)
from utils.dataset_manager import (
    load_dataset,
    read_metadata_rows,
    write_metadata_rows,
    register_new_face,
    save_face_and_voterid,
    append_embedding_to_store,
    write_mapping_csv,
)
from utils.face_engine import (
    crop_with_margin,
    detect_faces,
    get_embedding_for_crop,
    cosine_similarity,
    sort_faces_row_wise,
)
from utils.file_utils import save_image_upload, save_pdf_upload
from utils.pdf_utils import (
    extract_voter_id_blocks,
    extract_voterids_by_proximity,
    get_text_blocks_from_page,
    page_to_bgr_image,
)
from utils.schemas import (
    AddEntryResponse,
    FaceMatch,
    FaceResult,
    MatchResponse,
    UploadPdfResponse,
    VoterIdField,
)

app = FastAPI(
    title="Fake Voter Detection API",
    description="Face matching and duplicate-entry detection across voter photo records.",
    version="1.0.0",
)

# Allow a locally-hosted frontend to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved face crops so a frontend can render <img src="..."> directly.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
async def match_faces(files: List[UploadFile] = File(..., description="One or more PDF files")):
    """
    Detect faces in the uploaded PDF(s) and check each one against the
    known dataset. Non-PDF files are reported back as warnings, not errors,
    so a mixed batch doesn't fail entirely.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    known = load_dataset()
    filename_to_voterid = {entry["filename"]: entry["voter_id"] for entry in known}
    metadata = read_metadata_rows()

    results: List[FaceResult] = []
    warnings: List[str] = []

    for upload in files:
        ext = os.path.splitext(upload.filename or "")[1].lower()
        if ext != ".pdf":
            warnings.append(f"Unsupported file type: {upload.filename} (upload a PDF).")
            continue

        pdf_path, fname = await save_pdf_upload(upload, UPLOAD_FOLDER_B)
        doc = fitz.open(pdf_path)

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            img_np = page_to_bgr_image(page)
            faces = detect_faces(img_np)
            voter_ids = extract_voterids_by_proximity(faces, get_text_blocks_from_page(page))

            face_indices = sorted(range(len(faces)), key=lambda i: (faces[i].bbox[1], faces[i].bbox[0]))

            for i in face_indices:
                face_crop = crop_with_margin(img_np, faces[i].bbox)
                embedding, resized = get_embedding_for_crop(face_crop)

                input_name = f"{fname}_page{page_number + 1}_face{i + 1}.jpg"
                cv2.imwrite(os.path.join(UPLOAD_FOLDER_B, input_name), resized)

                if embedding is None:
                    continue

                matches: List[FaceMatch] = []
                matched_voter_id = ""
                for entry in known:
                    similarity = round(cosine_similarity(embedding, entry["embedding"]) * 100, 2)
                    if similarity >= SIMILARITY_THRESHOLD:
                        matches.append(
                            FaceMatch(
                                filename=entry["filename"],
                                similarity=similarity,
                                voter_id=filename_to_voterid.get(entry["filename"], ""),
                            )
                        )
                        if not matched_voter_id:
                            matched_voter_id = filename_to_voterid.get(entry["filename"], "")

                current_voter_id = voter_ids[i] if i < len(voter_ids) else ""
                for row in metadata:
                    if row["filename"] == input_name:
                        row["voter_id"] = current_voter_id
                        row["matched_voter_id"] = matched_voter_id

                results.append(
                    FaceResult(
                        input_image=f"/static/dataset_b/{input_name}",
                        voter_id=current_voter_id,
                        matches=matches,
                    )
                )

    write_metadata_rows(metadata)
    return MatchResponse(results=results, warnings=warnings)


@app.post("/add", response_model=AddEntryResponse)
async def add_entry(
    photo: UploadFile = File(..., description="A single reference photo (jpg/jpeg/png)"),
    voter_id: str = Form(..., description="Voter ID, e.g. XYZ123456"),
):
    """Add a single reference photo to the known-faces dataset."""
    # Validates format (2-4 uppercase letters + 5+ digits) and normalizes casing.
    try:
        validated = VoterIdField(voter_id=voter_id)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    save_path, filename = await save_image_upload(photo, UPLOAD_FOLDER_A)
    success, message = register_new_face(save_path, validated.voter_id)

    return AddEntryResponse(
        success=success,
        message=message,
        filename=filename if success else None,
        voter_id=validated.voter_id if success else None,
    )


@app.post("/upload_pdf", response_model=UploadPdfResponse)
async def upload_pdf(pdf_file: UploadFile = File(..., description="A single PDF to extract faces from")):
    """
    Extract every face on each page, pair it with the nearest voter-ID text,
    and auto-register any face that has no existing match.
    """
    pdf_path, fname = await save_pdf_upload(pdf_file, PDF_UPLOADS)
    doc = fitz.open(pdf_path)

    known = load_dataset()
    filename_to_voterid = {entry["filename"]: entry["voter_id"] for entry in known}
    new_entries = 0
    results: List[FaceResult] = []
    face_voterid_map = []
    serial_counter = 1

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        img_np = page_to_bgr_image(page)

        voter_id_blocks = extract_voter_id_blocks(get_text_blocks_from_page(page))
        faces = sort_faces_row_wise(detect_faces(img_np))

        for i, face in enumerate(faces):
            voter_id = voter_id_blocks[i][0] if i < len(voter_id_blocks) else ""
            face_crop = crop_with_margin(img_np, face.bbox)
            input_name = f"{fname}page{page_number + 1}_face{i + 1}{voter_id}.jpg"
            embedding, resized = get_embedding_for_crop(face_crop)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER_B, input_name), resized)

            if embedding is None:
                continue

            matches: List[FaceMatch] = []
            for entry in known:
                similarity = round(cosine_similarity(embedding, entry["embedding"]) * 100, 2)
                if similarity >= SIMILARITY_THRESHOLD:
                    matches.append(
                        FaceMatch(
                            filename=entry["filename"],
                            similarity=similarity,
                            voter_id=filename_to_voterid.get(entry["filename"], ""),
                        )
                    )

            if not matches and voter_id:
                save_face_and_voterid(resized, input_name, voter_id, serial=serial_counter)
                append_embedding_to_store(embedding)
                new_entries += 1

            face_voterid_map.append({"voter_id": voter_id, "image": input_name})
            results.append(
                FaceResult(
                    input_image=f"/static/dataset_b/{input_name}",
                    voter_id=voter_id,
                    matches=matches,
                )
            )
            serial_counter += 1

    write_mapping_csv(MAPPING_CSV, face_voterid_map)

    return UploadPdfResponse(
        results=results,
        new_entries=new_entries,
        message=f"Processed PDF. {new_entries} new unique face(s) saved to dataset.",
    )
