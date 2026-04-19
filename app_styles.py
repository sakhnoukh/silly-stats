"""
CSS styles for the Document Classifier Streamlit app.
"""


def get_css() -> str:
    return """
<style>
    /* ------------------------------------------------------------------ */
    /* Animation                                                          */
    /* ------------------------------------------------------------------ */

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes barGrow {
        from { width: 0% !important; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(67, 97, 238, 0); }
        50% { box-shadow: 0 0 16px 2px rgba(67, 97, 238, 0.15); }
    }

    /* ------------------------------------------------------------------ */
    /* Header                                                             */
    /* ------------------------------------------------------------------ */

    .app-header {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #4361EE, #7B2D8E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.15rem;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #8B8FA3;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    /* ------------------------------------------------------------------ */
    /* Section headers                                                    */
    /* ------------------------------------------------------------------ */

    .section-header {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8B8FA3;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(to right, rgba(139,143,163,0.3), transparent);
    }

    /* ------------------------------------------------------------------ */
    /* Result card                                                        */
    /* ------------------------------------------------------------------ */

    .result-card {
        position: relative;
        background: linear-gradient(135deg,
            color-mix(in srgb, var(--accent-color) 8%, transparent),
            color-mix(in srgb, var(--accent-color) 3%, transparent));
        border: 1px solid color-mix(in srgb, var(--accent-color) 20%, transparent);
        border-left: 5px solid var(--accent-color, #4361EE);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04);
        animation: fadeInUp 0.4s ease-out, pulseGlow 2s ease-in-out 0.5s 1;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, color-mix(in srgb, var(--accent-color) 6%, transparent), transparent 70%);
        pointer-events: none;
    }
    .result-card .class-name {
        font-size: 2rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin: 0;
    }
    .result-card .confidence-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--accent-color, #4361EE);
        margin: 0;
        line-height: 1;
    }
    .result-card .confidence-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8B8FA3;
        margin-top: 0.3rem;
    }
    .result-card .method-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.75rem;
        color: #8B8FA3;
        background: rgba(139,143,163,0.1);
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        margin-top: 0.5rem;
    }

    /* ------------------------------------------------------------------ */
    /* Confidence bars                                                    */
    /* ------------------------------------------------------------------ */

    .confidence-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 0.5rem 0;
    }
    .confidence-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.4rem 0.6rem;
        border-radius: 10px;
        transition: background 0.2s ease;
        animation: fadeInUp 0.4s ease-out backwards;
    }
    .confidence-row:nth-child(1) { animation-delay: 0.05s; }
    .confidence-row:nth-child(2) { animation-delay: 0.1s; }
    .confidence-row:nth-child(3) { animation-delay: 0.15s; }
    .confidence-row:nth-child(4) { animation-delay: 0.2s; }
    .confidence-row:hover {
        background: rgba(139,143,163,0.06);
    }
    .confidence-row .label {
        width: 90px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #8B8FA3;
        text-align: right;
        flex-shrink: 0;
    }
    .confidence-row.predicted .label {
        color: inherit;
        font-weight: 700;
    }
    .confidence-row .bar-track {
        flex: 1;
        background: rgba(139,143,163,0.1);
        border-radius: 8px;
        height: 28px;
        overflow: hidden;
        position: relative;
    }
    .confidence-row .bar-fill {
        height: 100%;
        border-radius: 8px;
        min-width: 4px;
        position: relative;
        animation: barGrow 0.8s ease-out backwards;
        animation-delay: inherit;
    }
    .confidence-row.predicted .bar-fill {
        box-shadow: 0 2px 8px color-mix(in srgb, var(--bar-color) 40%, transparent);
    }
    .confidence-row .pct {
        width: 55px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #8B8FA3;
        text-align: right;
        flex-shrink: 0;
        font-variant-numeric: tabular-nums;
    }
    .confidence-row.predicted .pct {
        color: inherit;
        font-weight: 700;
    }

    /* ------------------------------------------------------------------ */
    /* Field cards                                                        */
    /* ------------------------------------------------------------------ */

    .field-card {
        background: rgba(139,143,163,0.04);
        border: 1px solid rgba(139,143,163,0.12);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        animation: fadeInUp 0.4s ease-out backwards;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.5rem;
    }
    .field-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .field-card .field-icon {
        font-size: 1.1rem;
        margin-bottom: 0.4rem;
    }
    .field-card .field-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8B8FA3;
        margin-bottom: 0.4rem;
    }
    .field-card .field-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: inherit;
    }
    .field-card.missing {
        border-style: dashed;
        border-color: rgba(139,143,163,0.15);
        background: rgba(139,143,163,0.02);
    }
    .field-card.missing .field-value {
        color: #6C7293;
        font-style: italic;
        font-weight: 400;
        font-size: 0.9rem;
    }

    /* stagger field card animations */
    .field-row:nth-child(1) .field-card { animation-delay: 0.1s; }
    .field-row:nth-child(2) .field-card { animation-delay: 0.2s; }
    .field-row:nth-child(3) .field-card { animation-delay: 0.3s; }

    /* ------------------------------------------------------------------ */
    /* Empty state                                                        */
    /* ------------------------------------------------------------------ */

    .empty-state {
        text-align: center;
        padding: 4rem 1rem;
        animation: fadeInUp 0.5s ease-out;
    }
    .empty-state .icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        filter: grayscale(0.3);
    }
    .empty-state .message {
        font-size: 1.1rem;
        font-weight: 600;
        color: #8B8FA3;
        margin-bottom: 0.35rem;
    }
    .empty-state .hint {
        font-size: 0.85rem;
        color: #6C7293;
    }

    /* ------------------------------------------------------------------ */
    /* Summary card (batch)                                               */
    /* ------------------------------------------------------------------ */

    .summary-card {
        background: linear-gradient(135deg, rgba(67,97,238,0.06), rgba(123,45,142,0.04));
        border: 1px solid rgba(67,97,238,0.15);
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0;
        animation: fadeInUp 0.4s ease-out;
    }
    .summary-card .summary-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.4rem;
    }
    .summary-card .summary-detail {
        font-size: 0.9rem;
        color: #8B8FA3;
    }

    /* ------------------------------------------------------------------ */
    /* Sidebar polish                                                     */
    /* ------------------------------------------------------------------ */

    .sidebar-brand {
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        background: linear-gradient(135deg, #4361EE, #7B2D8E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.1rem;
    }
    .sidebar-tagline {
        font-size: 0.78rem;
        color: #8B8FA3;
    }
</style>
"""


# Accent colors keyed by document class
ACCENT_COLORS = {
    "invoice": "#4361EE",
    "email": "#2EC4B6",
    "receipt": "#FF6B35",
    "form": "#7B2D8E",
}

# Emoji map keyed by document class
CLASS_EMOJIS = {
    "invoice": "🧾",
    "email": "📧",
    "receipt": "🛒",
    "form": "📋",
}

# Icons for each invoice field
FIELD_ICONS = {
    "invoice_number": "#️⃣",
    "invoice_date": "📅",
    "due_date": "⏰",
    "issuer_name": "🏢",
    "recipient_name": "👤",
    "total_amount": "💰",
}
