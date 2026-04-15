"""
Document Classification & Invoice Extraction — Streamlit UI
Phase 4 — Visual Frontend

Run with:
    streamlit run app.py

Requires:
    pip install streamlit
"""

import streamlit as st
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path so we can import from run.py and scripts/
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from run import (
    extract_text,
    clean_text,
    classify_document,
    extract_invoice_fields,
    VECTORIZER_PATH,
    BEST_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #888;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-card h1 {
        font-size: 2.5rem;
        margin: 0;
        text-transform: uppercase;
    }
    .result-card p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .field-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .field-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #eee;
    }
    .field-table td:first-child {
        font-weight: 600;
        width: 40%;
        color: #555;
    }
    .step-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown('<p class="main-title">📄 Document Classifier & Invoice Extractor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a document to classify it and extract invoice fields — no generative AI</p>', unsafe_allow_html=True)

# Show system status
col1, col2 = st.columns(2)
with col1:
    if VECTORIZER_PATH.exists() and BEST_MODEL_PATH.exists():
        st.success("ML Model loaded")
    else:
        st.warning("Using keyword fallback")
with col2:
    st.info("4 categories: Email, Form, Invoice, Receipt")

st.divider()

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Drop a document here",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf", "txt", "csv", "md"],
    help="Supported: images (PNG, JPG), PDFs, and text files",
)

if uploaded_file is not None:
    # Save uploaded file to a temp location so our pipeline can read it
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # ---------------------------------------------------------------
        # Step 1: Extract text
        # ---------------------------------------------------------------
        with st.spinner("Extracting text..."):
            raw_text = extract_text(tmp_path)

        if not raw_text or not raw_text.strip():
            st.error("Could not extract any text from this file.")
            st.stop()

        # ---------------------------------------------------------------
        # Step 2: Clean text
        # ---------------------------------------------------------------
        with st.spinner("Cleaning text..."):
            cleaned = clean_text(raw_text)

        if not cleaned.strip():
            st.error("Text was empty after cleaning.")
            st.stop()

        # ---------------------------------------------------------------
        # Step 3: Classify
        # ---------------------------------------------------------------
        with st.spinner("Classifying document..."):
            classification = classify_document(raw_text, cleaned)

        predicted_class = classification["predicted_class"]
        confidence = classification["confidence"]
        method = classification.get("method", "unknown")
        top_confidence = confidence.get(predicted_class, 0)

        # ---------------------------------------------------------------
        # Display classification result
        # ---------------------------------------------------------------

        # Color map for each class
        colors = {
            "invoice": "#667eea",
            "email": "#28a745",
            "receipt": "#fd7e14",
            "form": "#6f42c1",
        }
        bg_color = colors.get(predicted_class, "#667eea")

        # Emoji map
        emojis = {
            "invoice": "🧾",
            "email": "📧",
            "receipt": "🛒",
            "form": "📋",
        }
        emoji = emojis.get(predicted_class, "📄")

        st.markdown(f"""
        <div class="result-card" style="background: {bg_color};">
            <h1>{emoji} {predicted_class}</h1>
            <p>Confidence: {top_confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # Confidence breakdown
        # ---------------------------------------------------------------
        st.markdown('<p class="step-header">Confidence Scores</p>', unsafe_allow_html=True)

        cols = st.columns(4)
        for i, (cls, score) in enumerate(sorted(confidence.items())):
            with cols[i]:
                cls_emoji = emojis.get(cls, "📄")
                if cls == predicted_class:
                    st.metric(
                        label=f"{cls_emoji} {cls.title()}",
                        value=f"{score:.1%}",
                        delta="predicted",
                    )
                else:
                    st.metric(
                        label=f"{cls_emoji} {cls.title()}",
                        value=f"{score:.1%}",
                    )

        # Method badge
        if method == "ml_model":
            st.caption("🤖 Classified using ML Model (TF-IDF + Logistic Regression)")
        else:
            st.caption("🔤 Classified using keyword-based fallback")

        # ---------------------------------------------------------------
        # Step 4: Invoice extraction (only if invoice)
        # ---------------------------------------------------------------
        if predicted_class == "invoice":
            st.divider()
            st.markdown('<p class="step-header">Extracted Invoice Fields</p>', unsafe_allow_html=True)

            with st.spinner("Extracting invoice fields..."):
                fields = extract_invoice_fields(raw_text)

            found = sum(1 for v in fields.values() if v is not None)
            total = len(fields)

            st.progress(found / total, text=f"{found} of {total} fields extracted")

            # Display fields as a clean table
            field_labels = {
                "invoice_number": "Invoice Number",
                "invoice_date": "Invoice Date",
                "due_date": "Due Date",
                "issuer_name": "Issuer / Vendor",
                "recipient_name": "Recipient / Bill To",
                "total_amount": "Total Amount",
            }

            table_html = '<table class="field-table">'
            for key, label in field_labels.items():
                value = fields.get(key)
                if value:
                    if key == "total_amount":
                        display = f"${float(value):,.2f}" if value.replace('.', '').replace(',', '').isdigit() else value
                    else:
                        display = value
                    table_html += f'<tr><td>{label}</td><td>{display}</td></tr>'
                else:
                    table_html += f'<tr><td>{label}</td><td style="color:#ccc;">— not found —</td></tr>'
            table_html += '</table>'

            st.markdown(table_html, unsafe_allow_html=True)

        # ---------------------------------------------------------------
        # Expandable sections
        # ---------------------------------------------------------------
        st.divider()

        with st.expander("📝 View extracted text"):
            st.text(raw_text[:2000])
            if len(raw_text) > 2000:
                st.caption(f"... showing first 2000 of {len(raw_text)} characters")

        with st.expander("🔧 View JSON output"):
            result = {
                "file": uploaded_file.name,
                "classification": predicted_class,
                "confidence": confidence,
            }
            if predicted_class == "invoice":
                result["extracted_fields"] = fields
                result["fields_extracted"] = f"{found}/{total}"
            st.json(result)

    finally:
        # Clean up temp file
        os.unlink(tmp_path)

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #aaa;">
        <p style="font-size: 3rem;">📄</p>
        <p>Upload a document to get started</p>
        <p style="font-size: 0.85rem;">Supports PNG, JPG, PDF, and TXT files</p>
    </div>
    """, unsafe_allow_html=True)