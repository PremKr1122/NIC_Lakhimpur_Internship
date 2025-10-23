# 🇮🇳 Face Recognition System for Voter Trust Assurance
### Industrial Internship Project at NIC, Lakhimpur District Centre

---

## Project Overview

This repository documents the development of an **Automated Face Recognition System** designed to enhance data integrity and prevent fraud in electoral documentation. Developed as an intern at the **National Informatics Centre (NIC)**, Lakhimpur District Centre, Assam, this system provides a secure and automated method for verifying voter records.

* **Duration:** July 3rd, 2025 to July 30th, 2025
* **Guidance:** Mr. Mithun Mukherjee, Scientist-D & DIO of NIC, Lakhimpur
* **Team:** Prem Kr Sah (222010007033), Hirak Jyoti Sarmah, Sahil Prasad, Anupom Paul

---

## Key Problem Solved

Traditional verification of voter rolls is highly **manual, slow, and error-prone**. This creates a risk of **duplicate entries** and facilitates **voter fraud** by allowing impersonation or the use of fake IDs.

Our system directly addresses this by automating face recognition from official documents, especially PDFs containing voter lists and IDs (EPIC), for accurate and efficient verification against a stored dataset.

## System Objectives

| Objective | Description |
|:----------|:------------|
| **Data Integrity** | Maintain a clean, duplicate-free database of verified citizens/voters. |
| **Fraud Prevention** | Detect duplicate faces and mismatched IDs, reducing impersonation and multiple voting. |
| **Automation** | Automate face and ID matching to significantly reduce manual checking time and human error. |
| **Transparency** | Ensure official records are validated with high accuracy, increasing public trust. |

---

## Technical Stack & Implementation

The project utilizes a robust mix of Python libraries and a web framework to create a complete document processing pipeline:

| Component | Technology / Library Used | Purpose |
|:----------|:--------------------------|:--------|
| **Backend** | **Python** and **Flask** | Built a secure, web-based interface for PDF upload and processing. |
| **Document Parsing** | **PyMuPDF** (fitz) | Extracted images and text elements from complex PDF voter list documents. |
| **Face Recognition** | **InsightFace** | Performed face detection, cropping, and generated high-accuracy face embeddings for feature extraction. |
| **Data Extraction** | **Regular Expressions (regex)** | Used to reliably extract Voter ID numbers from surrounding text in the documents. |
| **Matching Logic** | **Cosine Similarity** | Compared extracted faces with the saved dataset to find and display matched Voter IDs with accuracy. |

## ⚙️ System Flow

1. **Input:** Admin uploads PDF or image containing voter data.
2. **Processing:** System extracts images/text (PyMuPDF) and detects faces (InsightFace).
3. **Matching:** Generated face embeddings are compared using cosine similarity to map faces with correct Voter IDs.
4. **Output:** Displayed and stored matched results (Voter ID, similarity score, etc.) or saved unmatched faces for manual review.

---

## Future Work

Potential enhancements for the system include:

* Integrating a robust database (e.g., SQLite, PostgreSQL) instead of CSV files for better storage and security.
* Improving text detection for Voter ID accuracy using advanced OCR technologies like Tesseract.
* Optimizing the code to improve overall time and space complexity.

---

## Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/face-recognition-voter-system.git

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Acknowledgments

Special thanks to **Mr. Mithun Mukherjee** and the **NIC Lakhimpur team** for their guidance and support throughout this internship project.

---

**License:** [Add your license here]
