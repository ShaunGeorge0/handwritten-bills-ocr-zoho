# dashboard.py — Bonus feature: minimal UI to upload a bill and compare models side by side.
#
# Run locally:
#   streamlit run dashboard.py
#
# Why Streamlit: the spec's bonus asks for a "minimal UI... local run
# instructions work just as well" as a hosted link — Streamlit gets you
# there in ~150 lines with no separate frontend/backend split, and it
# reuses extractor.py / zoho_client.py directly instead of duplicating
# extraction logic behind a REST API. If you want a shareable hosted link
# specifically (Vercel/Netlify), see the note at the bottom of this file —
# but for a 6–10 hr budget this is the higher-leverage choice.

import os
import tempfile

import pandas as pd
import streamlit as st

from extractor import extract_with_gemini, extract_with_openai, extract_with_claude
from zoho_client import ZohoBooksClient

st.set_page_config(page_title="Handwritten Bill Extractor — Model Comparison", layout="wide")

MODEL_FNS = {
    "Gemini 2.5 Flash": extract_with_gemini,
    "GPT-5 Mini": extract_with_openai,
    "Claude Haiku 4.5": extract_with_claude,
}

st.title("📄 Handwritten Bill Extractor")
st.caption("Upload a bill photo, run it through multiple vision models, and compare the extracted fields side by side.")

col_upload, col_models = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader("Upload a bill/receipt image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        st.image(uploaded, caption="Uploaded bill", width=350)

with col_models:
    selected_models = st.multiselect(
        "Models to compare", options=list(MODEL_FNS.keys()), default=list(MODEL_FNS.keys())
    )
    run_button = st.button("🔍 Extract & Compare", type="primary", disabled=not uploaded)

if "results" not in st.session_state:
    st.session_state.results = {}

if run_button and uploaded:
    # write the upload to a temp file since extract_with_* take a path
    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    results = {}
    progress = st.progress(0.0, text="Running extraction...")
    for i, model_name in enumerate(selected_models):
        try:
            results[model_name] = MODEL_FNS[model_name](tmp_path)
        except Exception as e:
            results[model_name] = e
        progress.progress((i + 1) / len(selected_models), text=f"Ran {model_name}")
    progress.empty()

    os.unlink(tmp_path)
    st.session_state.results = results
    st.session_state.uploaded_name = uploaded.name

results = st.session_state.results

if results:
    st.subheader("Side-by-side comparison")

    ok_results = {k: v for k, v in results.items() if not isinstance(v, Exception)}
    failed = {k: v for k, v in results.items() if isinstance(v, Exception)}

    if ok_results:
        # one column of fields, one column per model — easy to eyeball differences
        field_names = ["vendor_name", "bill_number", "date", "currency", "subtotal", "tax_amount", "total_amount", "payment_mode"]
        table = {"Field": field_names}
        for model_name, result in ok_results.items():
            table[model_name] = [getattr(result.data, f) for f in field_names]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        st.subheader("Latency & cost")
        meta_rows = []
        for model_name, result in ok_results.items():
            meta_rows.append({
                "Model": model_name,
                "Latency (s)": round(result.latency_s, 2),
                "Input tokens": result.input_tokens,
                "Output tokens": result.output_tokens,
            })
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

        st.subheader("Line items")
        tabs = st.tabs(list(ok_results.keys()))
        for tab, (model_name, result) in zip(tabs, ok_results.items()):
            with tab:
                if result.data.line_items:
                    st.dataframe(pd.DataFrame([li.model_dump() for li in result.data.line_items]), use_container_width=True, hide_index=True)
                else:
                    st.write("No line items extracted.")

        st.divider()
        st.subheader("Post to Zoho Books")
        best_model = st.selectbox("Use this model's extraction for the expense entry", list(ok_results.keys()))
        if st.button("📤 Create Expense in Zoho Books"):
            try:
                client = ZohoBooksClient()
                resp = client.create_expense(ok_results[best_model].data)
                expense_id = resp.get("expense", {}).get("expense_id", "N/A")
                st.success(f"Expense created in Zoho Books — expense_id: {expense_id}")
            except Exception as e:
                st.error(f"Zoho API error: {e}")

    if failed:
        st.subheader("⚠️ Failures")
        for model_name, err in failed.items():
            st.error(f"{model_name}: {err}")

# --- Optional: hosted (Vercel/Netlify) alternative -------------------------
# If a shareable hosted link matters more than dev speed, swap this file for
# a small FastAPI backend (POST /extract -> JSON) deployed on Render/Fly.io
# (Vercel/Netlify don't run long-lived Python processes) plus a static
# HTML+fetch() frontend on Vercel. That's a correct approach too, just
# meaningfully more code for the same requirement — the spec explicitly
# says local run instructions are an acceptable substitute for a hosted
# link, so it's not necessary here.