<div align="center">

# 🗳️ Duplicate Voter Detection System
### Using Face Recognition & AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask&logoColor=white)
![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **An intelligent Face Recognition System to automate voter verification from PDF electoral rolls, enhancing data integrity and preventing voter fraud.**  
> *Developed during internship at **NIC Lakhimpur***

[Features](#-features) · [Demo](#-screenshots) · [Installation](#-installation) · [Usage](#-usage) · [Architecture](#-architecture) · [API](#-api-routes)

---

</div>

## 📌 Overview

The **Duplicate Voter Detection System** is a Flask-based web application that uses state-of-the-art facial recognition to identify duplicate or fraudulent voter entries in electoral documents. The system can process bulk PDF voter rolls, extract faces and voter IDs automatically, and cross-match them against a trusted database — all through a clean web interface.

This project was built to address a real-world problem faced by electoral commissions: manual verification of voter lists is slow, error-prone, and vulnerable to fraud. By automating the process with AI, this system can flag duplicate registrations in seconds.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Bulk Processing** | Upload entire voter roll PDFs; faces and IDs are auto-extracted page by page |
| 🔍 **Face Matching** | Cosine similarity-based matching using InsightFace `buffalo_l` embeddings |
| 🪪 **Voter ID Extraction** | Regex-based proximity matching to associate voter IDs with detected faces |
| 🧠 **Duplicate Detection** | Embedding comparison with configurable similarity threshold (default: 95%) |
| ➕ **Single Entry Add** | Manually add individual voter photos with voter IDs to the dataset |
| 💾 **Persistent Storage** | Embeddings stored in `.pkl`, metadata in `.csv` for fast retrieval |
| 🗂️ **Export Mapping** | Generates `voterid_face_mapping.csv` for audit trails |
| 🌐 **Web Interface** | Clean, responsive HTML/CSS UI — no frontend framework required |

---

## 🖼️ Screenshots

> *(Screenshots available in the `/screenshots` folder)*

```
screenshots/
├── home_page.png
├── match_results.png
├── pdf_upload.png
└── add_entry.png
```

---

## 🏗️ Architecture

```
duplicate-voter-detection-system/
│
├── Project/
│   └── main.py                  # Flask application (core logic)
│
├── static/
│   ├── dataset_a/
│   │   ├── images/              # Trusted voter face images
│   │   ├── metadata.csv         # Filename → Voter ID → Matched ID
│   │   ├── embeddings.pkl       # Stored face embeddings
│   │   └── voterids.csv         # Serial → Filename → Voter ID
│   │
│   ├── dataset_b/               # Extracted faces from uploaded PDFs/images
│   └── uploads/                 # Temporary PDF storage
│
├── templates/
│   └── index.html               # Main web UI
│
├── mappings/
│   └── voterid_face_mapping.csv # Audit trail output
│
├── certificate/                 # Internship certificate
├── presentation/                # Project presentation
├── sample data/                 # Sample voter roll PDFs
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

```
                  ┌─────────────────────────────┐
                  │     Upload PDF / Image      │
                  └────────────┬────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  PyMuPDF (fitz)     │
                    │  Render page @300dpi│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
   ┌──────────▼──────────┐          ┌──────────▼──────────┐
   │  InsightFace        │          │   Regex Text Parser │
   │  Face Detection &   │          │ Voter ID Extraction │
   │Embedding (buffalo_l)│          │  (proximity match)  │
   └──────────┬──────────┘          └──────────┬──────────┘
              │                                │
              └────────────────┬───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Cosine Similarity  │
                    │  vs. Known Dataset  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼───────────────────┐
              │                                    │
   ┌──────────▼──────────┐              ┌──────────▼──────────┐
   │  Match Found (≥55%) │              │  No Match Found     │
   │  → Show match +     │              │  → Add to Dataset   │
   │    voter ID         │              │  → Save embedding   │
   └─────────────────────┘              └─────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/PremKr1122/duplicate-voter-detection-system-using-Face-Recognition.git
cd duplicate-voter-detection-system-using-Face-Recognition/Project
```

### 2. Install Dependencies

```bash
pip install flask opencv-python numpy insightface pymupdf werkzeug onnxruntime
```

> **Note:** InsightFace will automatically download the `buffalo_l` model on first run (~300MB). Ensure you have an active internet connection.

### 3. Run the Application

```bash
python main.py
```

Navigate to **`http://127.0.0.1:5000`** in your browser.

---

## 📖 Usage

### 📄 Upload PDF Voter Roll (Extract & Build Dataset)

1. Go to **"Upload PDF to Extract & Add Faces"**
2. Select a voter roll PDF
3. The system will:
   - Detect all faces per page
   - Extract voter IDs via proximity matching
   - Store new unique faces into the trusted dataset

### 🔍 Match / Verify Faces

1. Go to **"Recognize Faces (Images or PDFs)"**
2. Upload a PDF or image for verification
3. Results show matched faces, similarity scores, and voter IDs

### ➕ Add Single Voter Entry

1. Go to **"Single Entry to Database"**
2. Upload a clear photo and enter the Voter ID
3. The entry is saved with duplicate-check protection

---

## 🔧 Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `SIMILARITY_THRESHOLD` | `match()` route | `55%` | Minimum similarity to consider a match |
| `DUPLICATE_THRESHOLD` | `append_embedding()` | `95%` | Threshold to reject duplicate dataset entries |
| `FACE_MARGIN` | Throughout | `40px` | Padding around detected face bounding box |
| `DPI` | PDF rendering | `300` | Resolution for PDF-to-image conversion |
| `ROW_TOLERANCE` | `sort_faces_row_wise()` | `50px` | Pixel tolerance for row-wise face grouping |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Backend** | Python, Flask |
| **Face Detection & Recognition** | InsightFace (`buffalo_l` model) |
| **Image Processing** | OpenCV |
| **PDF Processing** | PyMuPDF (fitz) |
| **Numerical Computing** | NumPy |
| **Data Storage** | CSV, Pickle (`.pkl`) |
| **Frontend** | HTML5, CSS3 (vanilla) |

---

## 📡 API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Home page |
| `POST` | `/add` | Add a single voter face + ID to dataset |
| `POST` | `/match` | Match uploaded image(s) or PDF against dataset |
| `POST` | `/upload_pdf` | Extract faces from PDF and populate dataset |

---

## 📊 Output Files

| File | Path | Contents |
|---|---|---|
| `metadata.csv` | `static/dataset_a/` | `filename`, `voter_id`, `matched_voter_id` |
| `voterids.csv` | `static/dataset_a/` | `serial`, `filename`, `voter_id` |
| `embeddings.pkl` | `static/dataset_a/` | List of numpy face embedding vectors |
| `voterid_face_mapping.csv` | `mappings/` | Audit trail: voter ID ↔ face image |

---

## ⚠️ Known Limitations

- Image-based matching (`/match` with `.jpg/.png`) is not yet implemented — use PDF input
- Performance depends on PDF quality and face image resolution
- `buffalo_l` model runs on CPU by default; GPU support available via `CUDAExecutionProvider`
- Voter ID regex pattern assumes format: `2–4 uppercase letters` followed by `5+ digits`

---

## 🔮 Future Improvements

- [ ] Implement image-only matching in `/match` route  
- [ ] Add GPU acceleration support  
- [ ] Build a dashboard with statistics and charts  
- [ ] Export flagged duplicates as a PDF report  
- [ ] Add user authentication for admin access  
- [ ] REST API with JSON responses for integration  

---

## 🏛️ Acknowledgements

- Developed during internship at **National Informatics Centre (NIC), Lakhimpur**
- Face recognition powered by [InsightFace](https://github.com/deepinsight/insightface)
- PDF processing by [PyMuPDF](https://pymupdf.readthedocs.io/)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **PremKr1122**  
⭐ Star this repo if you found it useful!

</div>
