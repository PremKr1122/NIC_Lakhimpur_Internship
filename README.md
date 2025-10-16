# 🇮🇳 Face Recognition System for Voter Trust Assurance

### Industrial Internship Project at NIC, Lakhimpur District Centre

---

### 🌟 Project Overview

This repository documents the development of an **Automated Face Recognition System** designed to enhance data integrity and prevent fraud in electoral documentation. [cite_start]Developed as an intern at the **National Informatics Centre (NIC)**, Lakhimpur District Centre, Assam, this system provides a secure and automated method for verifying voter records[cite: 38, 39, 40].

* **Duration:** July 3rd, 2025 to July 30th, 2025
* [cite_start]**Guidance:** Mr. Mithun Mukherjee, Scientist-D & DIO of NIC, Lakhimpur [cite: 1, 2]
* [cite_start]**Team:** Prem Kr Sah (222010007033), Hirak Jyoti Sarmah, Sahil Prasad, Anupom Paul [cite: 5, 6]

---

### 🏆 Key Problem Solved

[cite_start]Traditional verification of voter rolls is highly **manual, slow, and error-prone**[cite: 27, 30]. [cite_start]This creates a risk of **duplicate entries** and facilitates **voter fraud** by allowing impersonation or the use of fake IDs[cite: 28, 29].

[cite_start]Our system directly addresses this by automating face recognition from official documents, especially PDFs containing voter lists and IDs (EPIC), for accurate and efficient verification against a stored dataset[cite: 23, 24].

### ✨ System Objectives

| Objective | Description |
| :--- | :--- |
| **Data Integrity** | [cite_start]Maintain a clean, duplicate-free database of verified citizens/voters[cite: 35]. |
| **Fraud Prevention** | [cite_start]Detect duplicate faces and mismatched IDs, reducing impersonation and multiple voting[cite: 36]. |
| **Automation** | [cite_start]Automate face and ID matching to significantly reduce manual checking time and human error[cite: 27, 37]. |
| **Transparency** | [cite_start]Ensure official records are validated with high accuracy, increasing public trust[cite: 33]. |

---

### 🛠️ Technical Stack & Implementation

The project utilizes a robust mix of Python libraries and a web framework to create a complete document processing pipeline:

| Component | Technology / Library Used | Purpose |
| :--- | :--- | :--- |
| **Backend** | [cite_start]**Python** [cite: 72] [cite_start]and **Flask** [cite: 73, 78] | Built a secure, web-based interface for PDF upload and processing. |
| **Document Parsing** | [cite_start]**PyMuPDF** (fitz) [cite: 63, 79] | Extracted images and text elements from complex PDF voter list documents. |
| **Face Recognition** | [cite_start]**InsightFace** [cite: 64, 80] | [cite_start]Performed face detection, cropping, and generated high-accuracy face embeddings for feature extraction[cite: 65]. |
| **Data Extraction** | [cite_start]**Regular Expressions (regex)** [cite: 67, 81] | Used to reliably extract Voter ID numbers from surrounding text in the documents. |
| **Matching Logic** | [cite_start]**Cosine Similarity** [cite: 82] | [cite_start]Compared extracted faces with the saved dataset to find and display matched Voter IDs with accuracy[cite: 49]. |

### ⚙️ System Flow

1.  [cite_start]**Input:** Admin uploads PDF or image containing voter data[cite: 60].
2.  [cite_start]**Processing:** System extracts images/text (PyMuPDF) and detects faces (InsightFace)[cite: 63, 64].
3.  [cite_start]**Matching:** Generated face embeddings are compared using cosine similarity to map faces with correct Voter IDs[cite: 65, 68].
4.  [cite_start]**Output:** Displayed and stored matched results (Voter ID, similarity score, etc.) or saved unmatched faces for manual review[cite: 69, 82, 83].

---

### 📂 Project Documentation & Certificate (Link for Resume)

This section provides direct, verifiable proof of my internship and project work, which can be linked directly from my resume.

* [**View My Official Internship Completion Certificate**](Certificate/Industrial_Internship_Certificate.jpg) **👈 Recommended for Resume Link**
* [View the Complete Project Presentation Slides](Documentation/Face_Recognition_System_Presentation.pptx)
* **Source Code:** (Upload your code files to the `Source_Code/` folder and link to the main file, e.g., `Source_Code/app.py`)

---

### 🚀 Future Work

Potential enhancements for the system include:
* [cite_start]Integrating a robust database (e.g., SQLite, PostgreSQL) instead of CSV files for better storage and security[cite: 99].
* [cite_start]Improving text detection for Voter ID accuracy using advanced OCR technologies like Tesseract[cite: 100].
* [cite_start]Optimizing the code to improve overall time and space complexity[cite: 101].
