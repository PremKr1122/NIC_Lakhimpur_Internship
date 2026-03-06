from flask import Flask, render_template, request, redirect, url_for, flash
import os
import csv
import numpy as np
import cv2
import pickle
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import insightface
from insightface.app import FaceAnalysis
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER_A = 'static/dataset_a/images'
UPLOAD_FOLDER_B = 'static/dataset_b'
DATA_CSV = 'static/dataset_a/metadata.csv'
EMBEDDINGS_PKL = 'static/dataset_a/embeddings.pkl'
VOTERID_CSV = 'static/dataset_a/voterids.csv'
PDF_UPLOADS = 'static/uploads'

os.makedirs(UPLOAD_FOLDER_A, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_B, exist_ok=True)
os.makedirs(PDF_UPLOADS, exist_ok=True)

for file_path, fieldnames in [
    (DATA_CSV, ["filename", "voter_id", "matched_voter_id"]),
    (VOTERID_CSV, ["filename", "voter_id", "serial_number"])
]:
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
if not os.path.exists(EMBEDDINGS_PKL):
    with open(EMBEDDINGS_PKL, 'wb') as f:
        pickle.dump([], f)

face_model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0)

def load_dataset():
    if not os.path.exists(EMBEDDINGS_PKL) or not os.path.exists(DATA_CSV):
        return []
    with open(EMBEDDINGS_PKL, 'rb') as f:
        embeddings = pickle.load(f)
    with open(DATA_CSV, newline='', encoding='utf-8') as f:
        metadata = list(csv.DictReader(f))
    filename_to_voterid = {row["filename"]: row["voter_id"] for row in metadata}
    return [{"embedding": emb, "filename": meta["filename"], "voter_id": filename_to_voterid.get(meta["filename"], "")} for emb, meta in zip(embeddings, metadata)]

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def get_text_blocks_from_page(page):
    blocks = []
    for b in page.get_text("blocks"):
        if b[4].strip():
            blocks.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]))
    return blocks

def find_nearest_voter_id(face_center, text_blocks, used_ids):
    voter_id_pattern = r'\b[A-Z]{2,4}[0-9]{5,}\b'
    min_dist = float('inf')
    nearest_id = ""

    for block in text_blocks:
        bx, by, bx2, by2 = block[0], block[1], block[2], block[3]
        text = block[4]
        matches = re.findall(voter_id_pattern, text)
        for match in matches:
            if match in used_ids:
                continue
            center = ((bx + bx2) // 2, (by + by2) // 2)
            dist = np.linalg.norm(np.array(face_center) - np.array(center))
            if dist < min_dist:
                min_dist = dist
                nearest_id = match

    if nearest_id:
        used_ids.add(nearest_id)

    return nearest_id

def extract_voterids_by_proximity(faces, page_text_blocks):
    id_blocks = []
    for block in page_text_blocks:
        block_text = block[4]
        matches = re.findall(r'\b[A-Z]{2,4}[0-9]{5,}\b', block_text)
        for match in matches:
            cleaned = match.replace('O', '0').replace('I', '1')
            if len(cleaned) >= 8 and not cleaned.isdigit():
                bx, by, bx2, by2 = block[:4]
                id_center = ((bx + bx2)//2, (by + by2)//2)
                id_blocks.append({'voter_id': cleaned, 'center': id_center, 'assigned': False, 'top': by, 'left': bx})

    id_blocks.sort(key=lambda b: (b['top'], b['left']))

    face_voter_ids = [''] * len(faces)
    face_indices = list(range(len(faces)))
    face_indices.sort(key=lambda i: (faces[i].bbox[1], faces[i].bbox[0]))

    for i in face_indices:
        face = faces[i]
        x, y, x2, y2 = face.bbox
        face_center = ((x + x2)//2, (y + y2)//2)
        min_dist = float('inf')
        best_idx = -1
        for idx, id_block in enumerate(id_blocks):
            if id_block['assigned']:
                continue
            dist = np.linalg.norm(np.array(face_center) - np.array(id_block['center']))
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        if best_idx != -1:
            id_blocks[best_idx]['assigned'] = True
            face_voter_ids[i] = id_blocks[best_idx]['voter_id']

    return face_voter_ids

def save_face_and_voterid(face_crop, filename, voter_id, serial=None):
    cv2.imwrite(os.path.join(UPLOAD_FOLDER_A, filename), face_crop)
    with open(DATA_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id"])
        writer.writerow({"filename": filename, "voter_id": voter_id})
    with open(VOTERID_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["serial", "filename", "voter_id"])
        writer.writerow({"serial": serial if serial is not None else "", "filename": filename, "voter_id": voter_id})

def append_embedding(img_path, threshold=0.95):
    img = cv2.imread(img_path)
    faces = face_model.get(img)
    if not faces:
        return False

    face = faces[0]
    x, y, x2, y2 = map(int, face.bbox)
    margin = 40
    h, w = img.shape[:2]
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    x2 = min(x2 + margin, w)
    y2 = min(y2 + margin, h)

    face_crop = img[y:y2, x:x2]
    resized = cv2.resize(face_crop, (224, 224))
    aligned = face_model.get(resized)
    if not aligned:
        return False
    embedding = aligned[0].embedding

    existing = load_dataset()
    for entry in existing:
        sim = cosine_similarity(entry['embedding'], embedding)
        if sim > threshold:
            print("⚠ Duplicate image detected, skipping entry.")
            return False

    if os.path.exists(EMBEDDINGS_PKL):
        with open(EMBEDDINGS_PKL, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = []

    embeddings.append(embedding)
    with open(EMBEDDINGS_PKL, 'wb') as f:
        pickle.dump(embeddings, f)

    voter_id = ""
    with open(DATA_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id"])
        writer.writerow({"filename": os.path.basename(img_path), "voter_id": voter_id})
    with open(VOTERID_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id"])
        writer.writerow({"filename": os.path.basename(img_path), "voter_id": voter_id})
    return True


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/match', methods=['POST'])
def match():
    known = load_dataset()
    filename_to_voterid = {entry["filename"]: entry["voter_id"] for entry in known}
    results = []
    files = request.files.getlist('match_files')

    metadata = []
    with open(DATA_CSV, newline='', encoding='utf-8') as f:
        metadata = list(csv.DictReader(f))

    for file in files:
        fname = secure_filename(file.filename)
        ext = os.path.splitext(fname)[1].lower()
        temp_results = []

        if ext == ".pdf":
            pdf_path = os.path.join(UPLOAD_FOLDER_B, fname)
            file.save(pdf_path)
            doc = fitz.open(pdf_path)
            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
                pix = page.get_pixmap(dpi=300)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                if pix.n == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                faces = face_model.get(img_np)
                voter_ids = extract_voterids_by_proximity(faces, get_text_blocks_from_page(page))

                face_indices = list(range(len(faces)))
                face_indices.sort(key=lambda i: (faces[i].bbox[1], faces[i].bbox[0]))

                for i in face_indices:
                    face = faces[i]
                    x, y, x2, y2 = map(int, face.bbox)
                    margin = 40
                    h, w = img_np.shape[:2]
                    x = max(x - margin, 0)
                    y = max(y - margin, 0)
                    x2 = min(x2 + margin, w)
                    y2 = min(y2 + margin, h)

                    face_crop = img_np[y:y2, x:x2]
                    resized = cv2.resize(face_crop, (224, 224))
                    input_name = f"{fname}_page{page_number + 1}_face{i + 1}.jpg"
                    input_path = os.path.join(UPLOAD_FOLDER_B, input_name)
                    cv2.imwrite(input_path, resized)

                    aligned = face_model.get(resized)
                    if not aligned:
                        continue
                    embedding = aligned[0].embedding

                    matches = []
                    matched_voter_id = ""
                    for entry in known:
                        sim = cosine_similarity(embedding, entry['embedding'])
                        similarity = round(sim * 100, 2)
                        if similarity >= 55:
                            matches.append({
                                "filename": entry["filename"],
                                "similarity": similarity,
                                "voter_id": filename_to_voterid.get(entry["filename"], "")
                            })
                            if not matched_voter_id:
                                matched_voter_id = filename_to_voterid.get(entry["filename"], "")

                    # Update metadata.csv: voter_id (from PDF) and matched_voter_id (from dataset)
                    for row in metadata:
                        if row["filename"] == input_name:
                            row["voter_id"] = voter_ids[i] if i < len(voter_ids) else ""
                            row["matched_voter_id"] = matched_voter_id

                    temp_results.append({
                        "input": input_name,
                        "matches": matches,
                        "voter_id": voter_ids[i] if i < len(voter_ids) else ""
                    })

        elif ext in [".jpg", ".jpeg", ".png"]:
            flash("🟡 Image match not yet implemented in match(). Upload a PDF instead.")
            continue

        else:
            flash(f"❌ Unsupported file type: {fname}")
            continue

        results.extend(temp_results)

    # Overwrite updated metadata CSV
    with open(DATA_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "voter_id", "matched_voter_id"])
        writer.writeheader()
        writer.writerows(metadata)

    return render_template("index.html", results=results)

@app.route('/add', methods=['POST'])
def add_person():
    photo = request.files['photo']
    filename = secure_filename(photo.filename)
    voter_id = request.form.get('voter_id', '')
    save_path = os.path.join(UPLOAD_FOLDER_A, filename)
    photo.save(save_path)

    img = cv2.imread(save_path)
    if append_embedding(save_path):
        flash("✅ Entry added successfully.")
        rows = []
        updated = False
        with open(DATA_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['filename'] == filename:
                    row['voter_id'] = voter_id
                    updated = True
                rows.append(row)
        if updated:
            with open(DATA_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "voter_id"])
                writer.writeheader()
                writer.writerows(rows)
        rows = []
        updated = False
        with open(VOTERID_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['filename'] == filename:
                    row['voter_id'] = voter_id
                    updated = True
                rows.append(row)
        if updated:
            with open(VOTERID_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "voter_id"])
                writer.writeheader()
                writer.writerows(rows)
    else:
        flash("⚠ Duplicate image detected. Entry not added.")
    return redirect(url_for('home'))

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    pdf = request.files['pdf_file']
    pdf_path = os.path.join(PDF_UPLOADS, secure_filename(pdf.filename))
    pdf.save(pdf_path)

    doc = fitz.open(pdf_path)
    known = load_dataset()
    filename_to_voterid = {entry["filename"]: entry["voter_id"] for entry in known}
    new_entries = 0
    results = []
    face_voterid_map = []
    serial_counter = 1

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=300)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Get faces and text blocks
        page_text_blocks = get_text_blocks_from_page(page)
        faces = face_model.get(img_np)

        # Extract voter IDs from text blocks
        voter_id_blocks = []
        voter_id_pattern = r'\b[A-Z]{2,4}[0-9]{5,}\b'
        for block in page_text_blocks:
            matches = re.findall(voter_id_pattern, block[4])
            for match in matches:
                voter_id = match.replace('O', '0').replace('I', '1')
                if len(voter_id) >= 8 and not voter_id.isdigit():
                    voter_id_blocks.append((voter_id, block[1]))  # voter_id and top Y

        # Sort both by vertical position (top Y)
        voter_id_blocks.sort(key=lambda b: b[1])

        def sort_faces_row_wise(faces, row_tolerance=50):
            # Step 1: Sort top to bottom
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

            # Step 2: Sort within each row left to right (by X)
            sorted_faces = []
            for row in rows:
                row_sorted = sorted(row, key=lambda item: item[0].bbox[0])
                sorted_faces.extend([item[0] for item in row_sorted])

            return sorted_faces

        faces = sort_faces_row_wise(faces)

        for i, face in enumerate(faces):
            voter_id = voter_id_blocks[i][0] if i < len(voter_id_blocks) else ""
            x, y, x2, y2 = map(int, face.bbox)
            margin = 40
            h, w = img_np.shape[:2]
            x = max(x - margin, 0)
            y = max(y - margin, 0)
            x2 = min(x2 + margin, w)
            y2 = min(y2 + margin, h)
            face_crop = img_np[y:y2, x:x2]
            resized = cv2.resize(face_crop, (224, 224))
            input_name = f"{os.path.basename(pdf_path)}page{page_number + 1}_face{i + 1}{voter_id}.jpg"
            input_path = os.path.join(UPLOAD_FOLDER_B, input_name)
            cv2.imwrite(input_path, resized)

            aligned = face_model.get(resized)
            if not aligned:
                continue
            embedding = aligned[0].embedding

            matches = []
            for entry in known:
                sim = cosine_similarity(embedding, entry['embedding'])
                similarity = round(sim * 100, 2)
                if similarity >= 55:
                    matches.append({
                        "filename": entry["filename"],
                        "similarity": similarity,
                        "voter_id": filename_to_voterid.get(entry["filename"], "")
                    })

            if not matches and voter_id:
                save_face_and_voterid(resized, input_name, voter_id, serial=serial_counter)
                if os.path.exists(EMBEDDINGS_PKL):
                    with open(EMBEDDINGS_PKL, 'rb') as f:
                        embeddings = pickle.load(f)
                else:
                    embeddings = []
                embeddings.append(embedding)
                with open(EMBEDDINGS_PKL, 'wb') as f:
                    pickle.dump(embeddings, f)
                new_entries += 1

            face_voterid_map.append({
                "voter_id": voter_id,
                "image": input_name
            })

            results.append({"input": input_name, "matches": matches, "voter_id": voter_id})
            serial_counter += 1

    os.makedirs("mappings", exist_ok=True)
    with open(os.path.join("mappings", "voterid_face_mapping.csv"), mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["voter_id", "image"])
        writer.writeheader()
        writer.writerows(face_voterid_map)

    flash(f"✅ Processed PDF. {new_entries} new unique face(s) saved to dataset.")
    return render_template("index.html", results=results)


if __name__ == '__main__':
    app.run(debug=True)