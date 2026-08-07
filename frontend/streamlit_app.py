"""
Streamlit UI for the Fake Voter Detection system.
This is a thin client: all detection/matching logic lives in the FastAPI
backend (see main.py). Run the backend first, then this app.

    uvicorn main:app --reload
    streamlit run frontend/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Fake Voter Detection", page_icon="🗳️", layout="wide")

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; max-width: 1100px; }
        h1 { text-align: center; letter-spacing: 1px; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { background: #f4f4f4; border-radius: 8px 8px 0 0; padding: 8px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🗳️ Fake Voter Detection")
st.caption(f"Connected to API: `{API_BASE_URL}`")


def render_results(results, empty_message="No results yet."):
    if not results:
        st.info(empty_message)
        return

    cols = st.columns(3)
    for idx, result in enumerate(results):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown("**Input Face**")
                st.image(f"{API_BASE_URL}{result['input_image']}", use_container_width=True)
                if result.get("voter_id"):
                    st.caption(f"Detected Voter ID: `{result['voter_id']}`")

                if result["matches"]:
                    st.markdown("**Matches found:**")
                    for match in result["matches"]:
                        match_url = f"{API_BASE_URL}/static/dataset_a/images/{match['filename']}"
                        m1, m2 = st.columns([1, 1.4])
                        with m1:
                            st.image(match_url, use_container_width=True)
                        with m2:
                            st.write(f"**Similarity:** {match['similarity']}%")
                            st.write(f"**Voter ID:** {match['voter_id'] or 'N/A'}")
                else:
                    st.warning("No match found.")


tab_match, tab_add, tab_pdf = st.tabs(
    ["🔍 Recognize Faces", "➕ Add Single Entry", "📄 Upload PDF & Extract"]
)

# ---- Tab 1: Recognize Faces ----
with tab_match:
    st.subheader("Recognize Faces (PDFs)")
    match_files = st.file_uploader(
        "Select PDF file(s)", type=["pdf"], accept_multiple_files=True, key="match_uploader"
    )

    if st.button("Match", type="primary", key="match_button"):
        if not match_files:
            st.warning("Please upload at least one file.")
        else:
            files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in match_files]
            with st.spinner("Detecting and matching faces…"):
                try:
                    resp = requests.post(f"{API_BASE_URL}/match", files=files_payload, timeout=300)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.RequestException as exc:
                    st.error(f"Request to API failed: {exc}")
                    data = None

            if data:
                for w in data.get("warnings", []):
                    st.warning(w)
                st.session_state["match_results"] = data.get("results", [])

    if "match_results" in st.session_state:
        st.markdown("### Results")
        render_results(st.session_state["match_results"])

# ---- Tab 2: Add Single Entry ----
with tab_add:
    st.subheader("Single Entry to Database")
    with st.form("add_entry_form", clear_on_submit=True):
        photo = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"], key="add_photo")
        voter_id = st.text_input("Voter ID", placeholder="e.g. XYZ123456")
        submitted = st.form_submit_button("Add Face", type="primary")

    if submitted:
        if not photo or not voter_id.strip():
            st.warning("Please provide both a photo and a Voter ID.")
        else:
            files_payload = {"photo": (photo.name, photo.getvalue(), photo.type)}
            data_payload = {"voter_id": voter_id.strip()}
            with st.spinner("Processing photo…"):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/add", files=files_payload, data=data_payload, timeout=60
                    )
                    if resp.status_code == 422:
                        st.error(resp.json().get("detail", "Invalid Voter ID format."))
                    else:
                        resp.raise_for_status()
                        result = resp.json()
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(result["message"])
                except requests.RequestException as exc:
                    st.error(f"Request to API failed: {exc}")

# Tab 3: Upload PDF to Extract & Add Faces
with tab_pdf:
    st.subheader("Upload PDF to Extract & Add Faces")
    pdf_file = st.file_uploader("Select PDF file", type=["pdf"], key="extract_pdf")

    if st.button("Upload & Extract", type="primary", key="extract_button"):
        if not pdf_file:
            st.warning("Please upload a PDF file.")
        else:
            files_payload = {"pdf_file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
            with st.spinner("Extracting faces from PDF…"):
                try:
                    resp = requests.post(f"{API_BASE_URL}/upload_pdf", files=files_payload, timeout=300)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.RequestException as exc:
                    st.error(f"Request to API failed: {exc}")
                    data = None

            if data:
                st.success(f"✅ {data['message']}")
                st.session_state["pdf_results"] = data.get("results", [])

    if "pdf_results" in st.session_state:
        st.markdown("### Results")
        render_results(st.session_state["pdf_results"])
