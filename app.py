"""
Document Classification & Invoice Extraction — Streamlit UI
Phase 4 — Visual Frontend (Redesigned)

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
from datetime import datetime
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
    should_run_invoice_extraction,
    VECTORIZER_PATH,
    BEST_MODEL_PATH,
)
from app_styles import get_css, ACCENT_COLORS, CLASS_EMOJIS, FIELD_ICONS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Inject CSS
# ---------------------------------------------------------------------------

st.markdown(get_css(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="sidebar-brand">📄 Document Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-tagline">Classify documents & extract invoice fields</p>', unsafe_allow_html=True)
    st.divider()

    # System status
    if VECTORIZER_PATH.exists() and BEST_MODEL_PATH.exists():
        st.success("ML Model loaded", icon="✅")
    else:
        st.warning("Using keyword fallback", icon="⚠️")
    st.info("4 categories: Email, Form, Invoice, Receipt", icon="📂")

    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf", "txt", "csv", "md"],
        help="Supported: images (PNG, JPG), PDFs, and text files",
        accept_multiple_files=True,
    )

    st.divider()

    # Recent documents quick access
    if st.session_state.history:
        st.markdown("**Recent Documents**")
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            idx = len(st.session_state.history) - 1 - i
            emoji = CLASS_EMOJIS.get(item["predicted_class"], "📄")
            label = f"{emoji} {item['filename'][:25]}"
            if st.button(label, key=f"hist_btn_{idx}", use_container_width=True):
                st.session_state.selected_history_idx = idx

        if st.button("Clear history", type="secondary", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Helper: process a single file through the pipeline
# ---------------------------------------------------------------------------

def process_file(file_obj):
    """Run the full pipeline on an uploaded file. Returns result dict or None."""
    suffix = Path(file_obj.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name

    try:
        raw_text = extract_text(tmp_path)
        if not raw_text or not raw_text.strip():
            return None, "Could not extract any text from this file."

        cleaned = clean_text(raw_text)
        if not cleaned.strip():
            return None, "Text was empty after cleaning."

        classification = classify_document(raw_text, cleaned)
        predicted_class = classification["predicted_class"]
        confidence = classification["confidence"]
        method = classification.get("method", "unknown")
        top_confidence = confidence.get(predicted_class, 0)

        # Check if invoice extraction should run (even for non-invoice classifications)
        extraction_decision = should_run_invoice_extraction(raw_text, classification)
        run_extraction = extraction_decision.get("run_extraction", False)
        override = run_extraction and predicted_class != "invoice"

        fields = None
        if run_extraction:
            fields = extract_invoice_fields(raw_text)

        result = {
            "filename": file_obj.name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_confidence": top_confidence,
            "method": method,
            "fields": fields,
            "raw_text": raw_text,
            "override": override,
        }
        return result, None
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Helper: render classification results
# ---------------------------------------------------------------------------

def render_result(result, show_expanders=True):
    """Display classification result card, confidence bars, and invoice fields."""
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    top_confidence = result["top_confidence"]
    method = result["method"]
    fields = result.get("fields")
    raw_text = result.get("raw_text", "")
    override = result.get("override", False)

    accent = ACCENT_COLORS.get(predicted_class, "#4361EE")
    emoji = CLASS_EMOJIS.get(predicted_class, "📄")

    # Classification result card
    method_text = "ML Model" if method == "ml_model" else "Keyword fallback"
    method_icon = "🤖" if method == "ml_model" else "🔤"

    st.markdown(f"""
    <div class="result-card" style="--accent-color: {accent};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <p class="class-name">{emoji} {predicted_class}</p>
                <span class="method-badge">{method_icon} {method_text}</span>
            </div>
            <div style="text-align: right;">
                <p class="confidence-value">{top_confidence:.1%}</p>
                <p class="confidence-label">confidence</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if override:
        st.info("Invoice signals detected — extracting fields despite **{}** classification.".format(predicted_class), icon="💡")

    # Confidence breakdown bars
    st.markdown('<p class="section-header">Confidence Scores</p>', unsafe_allow_html=True)

    bars_html = '<div class="confidence-container">'
    for cls in sorted(confidence.keys()):
        score = confidence[cls]
        pct = max(score * 100, 0.5)  # minimum visible width
        cls_accent = ACCENT_COLORS.get(cls, "#888")
        cls_emoji = CLASS_EMOJIS.get(cls, "📄")
        is_predicted = "predicted" if cls == predicted_class else ""
        bars_html += f"""
        <div class="confidence-row {is_predicted}">
            <span class="label">{cls_emoji} {cls.title()}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width: {pct}%; background: linear-gradient(90deg, {cls_accent}, color-mix(in srgb, {cls_accent} 70%, white)); --bar-color: {cls_accent};"></div>
            </div>
            <span class="pct">{score:.1%}</span>
        </div>
        """
    bars_html += '</div>'

    st.markdown(bars_html, unsafe_allow_html=True)

    # Invoice fields (if extracted)
    if fields is not None:
        st.divider()
        st.markdown('<p class="section-header">Extracted Invoice Fields</p>', unsafe_allow_html=True)

        found = sum(1 for v in fields.values() if v is not None)
        total = len(fields)
        st.progress(found / total, text=f"{found} of {total} fields extracted")

        field_labels = {
            "invoice_number": "Invoice Number",
            "invoice_date": "Invoice Date",
            "due_date": "Due Date",
            "issuer_name": "Issuer / Vendor",
            "recipient_name": "Recipient / Bill To",
            "total_amount": "Total Amount",
        }

        field_keys = list(field_labels.keys())
        for row_start in range(0, len(field_keys), 2):
            cols = st.columns(2)
            for col_idx in range(2):
                key_idx = row_start + col_idx
                if key_idx >= len(field_keys):
                    break
                key = field_keys[key_idx]
                label = field_labels[key]
                value = fields.get(key)
                icon = FIELD_ICONS.get(key, "")

                with cols[col_idx]:
                    if value:
                        if key == "total_amount":
                            display = f"${float(value):,.2f}" if value.replace('.', '').replace(',', '').isdigit() else value
                        else:
                            display = value
                        st.markdown(f"""
                        <div class="field-row">
                        <div class="field-card">
                            <div class="field-icon">{icon}</div>
                            <div class="field-label">{label}</div>
                            <div class="field-value">{display}</div>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="field-row">
                        <div class="field-card missing">
                            <div class="field-icon">{icon}</div>
                            <div class="field-label">{label}</div>
                            <div class="field-value">Not found</div>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)

    # Expandable sections (skip when already inside an expander)
    if show_expanders:
        st.divider()
        with st.expander("📝 View extracted text"):
            st.code(raw_text, language="text")

        with st.expander("🔧 View JSON output"):
            output = {
                "file": result["filename"],
                "classification": predicted_class,
                "confidence": confidence,
            }
            if fields is not None:
                output["extracted_fields"] = fields
                found = sum(1 for v in fields.values() if v is not None)
                output["fields_extracted"] = f"{found}/{len(fields)}"
            st.json(output)


# ---------------------------------------------------------------------------
# Main content — Tabs
# ---------------------------------------------------------------------------

st.markdown('<p class="app-header">📄 Document Classifier & Invoice Extractor</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Upload documents to classify them and extract invoice fields</p>', unsafe_allow_html=True)

tab_classify, tab_history, tab_compare = st.tabs(["🔍 Classify", "📋 History", "⚖️ Compare"])

# ---------------------------------------------------------------------------
# Tab 1: Classify
# ---------------------------------------------------------------------------

with tab_classify:
    if uploaded_files:
        if len(uploaded_files) == 1:
            # Single file
            with st.spinner("Processing document..."):
                result, error = process_file(uploaded_files[0])
            if error:
                st.error(error)
            else:
                # Add to history (avoid duplicates on rerun)
                if not st.session_state.history or st.session_state.history[-1]["filename"] != result["filename"] or st.session_state.history[-1]["timestamp"] != result["timestamp"]:
                    st.session_state.history.append(result)
                    if len(st.session_state.history) > 20:
                        st.session_state.history = st.session_state.history[-20:]
                render_result(result)
        else:
            # Batch processing
            results = []
            progress_bar = st.progress(0, text="Processing documents...")
            for i, f in enumerate(uploaded_files):
                progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {f.name}...")
                result, error = process_file(f)
                if result:
                    results.append(result)
                    # Add to history
                    st.session_state.history.append(result)
                    if len(st.session_state.history) > 20:
                        st.session_state.history = st.session_state.history[-20:]
                else:
                    results.append({"filename": f.name, "error": error})
            progress_bar.empty()

            # Summary card
            successful = [r for r in results if "error" not in r]
            failed = [r for r in results if "error" in r]

            if successful:
                from collections import Counter
                class_counts = Counter(r["predicted_class"] for r in successful)
                breakdown = ", ".join(
                    f"{count} {CLASS_EMOJIS.get(cls, '')} {cls}{'s' if count > 1 else ''}"
                    for cls, count in class_counts.most_common()
                )
                st.markdown(f"""
                <div class="summary-card">
                    <div class="summary-title">{len(successful)} document{'s' if len(successful) != 1 else ''} classified</div>
                    <div class="summary-detail">{breakdown}</div>
                </div>
                """, unsafe_allow_html=True)

            if failed:
                for r in failed:
                    st.warning(f"Failed: {r['filename']} — {r['error']}")

            # Individual results in expanders
            for r in successful:
                emoji = CLASS_EMOJIS.get(r["predicted_class"], "📄")
                with st.expander(f"{emoji} {r['filename']} — {r['predicted_class'].title()} ({r['top_confidence']:.1%})", expanded=False):
                    render_result(r, show_expanders=False)

    elif hasattr(st.session_state, "selected_history_idx"):
        # Showing a selected history item
        idx = st.session_state.selected_history_idx
        if 0 <= idx < len(st.session_state.history):
            result = st.session_state.history[idx]
            st.caption(f"Showing saved result from {result['timestamp']}")
            render_result(result)
        del st.session_state.selected_history_idx
    else:
        # Empty state
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📄</div>
            <div class="message">Upload a document to get started</div>
            <div class="hint">Supports PNG, JPG, PDF, and TXT files — single or batch upload</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2: History
# ---------------------------------------------------------------------------

with tab_history:
    if st.session_state.history:
        st.markdown('<p class="section-header">Classification History</p>', unsafe_allow_html=True)

        # Build a display table
        history_data = []
        for item in reversed(st.session_state.history):
            emoji = CLASS_EMOJIS.get(item["predicted_class"], "📄")
            history_data.append({
                "File": item["filename"],
                "Type": f"{emoji} {item['predicted_class'].title()}",
                "Confidence": f"{item['top_confidence']:.1%}",
                "Fields": f"{sum(1 for v in item['fields'].values() if v is not None)}/{len(item['fields'])}" if item.get("fields") else "—",
                "Time": item["timestamp"],
            })

        st.dataframe(
            history_data,
            use_container_width=True,
            hide_index=True,
        )

        # Expandable detail for each
        for i, item in enumerate(reversed(st.session_state.history)):
            emoji = CLASS_EMOJIS.get(item["predicted_class"], "📄")
            with st.expander(f"{emoji} {item['filename']} — {item['timestamp']}"):
                render_result(item, show_expanders=False)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📋</div>
            <div class="message">No documents classified yet</div>
            <div class="hint">Upload documents in the sidebar to build your history</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 3: Compare
# ---------------------------------------------------------------------------

with tab_compare:
    if len(st.session_state.history) >= 2:
        st.markdown('<p class="section-header">Compare Documents</p>', unsafe_allow_html=True)

        options = [f"{CLASS_EMOJIS.get(h['predicted_class'], '📄')} {h['filename']} ({h['timestamp']})" for h in st.session_state.history]

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            sel1 = st.selectbox("Document A", options, index=len(options) - 2, key="cmp_a")
        with col_sel2:
            sel2 = st.selectbox("Document B", options, index=len(options) - 1, key="cmp_b")

        idx1 = options.index(sel1)
        idx2 = options.index(sel2)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{st.session_state.history[idx1]['filename']}**")
            render_result(st.session_state.history[idx1])
        with col2:
            st.markdown(f"**{st.session_state.history[idx2]['filename']}**")
            render_result(st.session_state.history[idx2])

    elif len(st.session_state.history) == 1:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">⚖️</div>
            <div class="message">Need at least 2 documents to compare</div>
            <div class="hint">Upload one more document to enable comparison</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">⚖️</div>
            <div class="message">No documents to compare</div>
            <div class="hint">Upload and classify documents first, then compare them side by side</div>
        </div>
        """, unsafe_allow_html=True)
