# Handwritten Bill Extraction, Evaluation & Zoho Expense Integration

Structured data extraction from handwritten Indian bills using Vision LLMs — benchmarked for accuracy, latency, and cost, then deployed into Zoho Expense.

## Overview

Digital invoices are easy for LLMs to parse; handwritten bills are not. This project builds a small, human-verified benchmark to find out which vision-capable model actually handles handwritten Indian bills well, whether the accuracy gain justifies the API cost, and demonstrates that the extracted output is usable in a real accounting workflow.

This is **not** an OCR project — it's an evaluation and deployment pipeline. Two vision LLMs were benchmarked against a manually verified dataset; the stronger of the two, **Gemini 3.5 Flash-Lite**, was selected and used to push sample bills into Zoho Expense via `pushsamples.py`.

## Architecture

```
Bill Image
    │
    ▼
Vision LLM
    │
    ▼
Pydantic Validation
    │
    ▼
Evaluation Framework
    │
    ▼
Model Selected
    │
    ▼
pushsamples.py
    │
    ▼
Zoho Expense
```

## Dataset

15 handwritten Indian bills, manually annotated and human-verified:

- Multiple vendors, handwriting styles, and layouts
- Bills with and without GST
- Missing fields, plus ambiguous and non-ambiguous cases

Ground truth per bill: `vendor_name`, `bill_number`, `date`, `currency`, `total_amount`, `tax_amount`, `gst_number`, `description`, `ambiguous`.

`description` is free-text and inherently subjective — it captures the annotator's paraphrase of the bill's purpose, so near-miss wording (e.g. "Book purchase" vs. "Books purchase") counts as incorrect under exact matching. `ambiguous` flags bills where the handwriting, layout, or missing context makes ground-truth labeling itself uncertain, which is used to separate genuine model failure from label noise in the analysis below.

## Repository Structure

| File | Purpose |
|---|---|
| `schemas.py` | Pydantic schemas for extracted bill fields |
| `extractor.py` | Vision LLM extraction pipeline |
| `evaluator.py` | Benchmark runner and scoring |
| `dashboard.py` | Streamlit UI for upload, comparison, and push |
| `pushsamples.py` | Pushes extracted bills to Zoho Expense |
| `zoho_client.py` | Zoho Expense API client |
| `main.py` | Entry point |
| `ground_truth.json` | Human-verified annotations |

## Setup instructions
 
1. Clone the repository and move into it.
2. Create a virtual environment and activate it.
```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
```
3. Install dependencies.
```bash
   pip install -r requirements.txt
```
4. Copy the environment template and fill in your real keys.
```bash
   cp .env.example .env
```
   You'll need:
   - A Gemini API key from Google AI Studio
   - An OpenRouter API key ([openrouter.ai](https://openrouter.ai) — free, no credit card required; used for the Nemotron calls)
   - Zoho Books OAuth credentials (client ID, client secret and a refresh token) from a free or trial Zoho Books organization
   - The `account_id` of an expense category and the `account_id` of a paid-through account (a bank or cash account) from that organization's Chart of Accounts
5. Verify everything is wired up correctly before running the full pipeline.
```bash
   python check_env.py
```
6. Drop your 15 bill photos into `data/images/` and fill in the matching entries in `data/ground_truth.json`.
### Running it
 
Process one bill with one model and post it to Zoho:
```bash
python main.py --image data/images/bill_01.jpg --model gemini
```
Swap `--model` for `nemotron` to use the free OpenRouter model instead.
Add `--no-zoho` to skip posting.
 
Run the full evaluation across all three models and every bill in the dataset:
```bash
python evaluator.py
```
This writes `results.json` (full detail per bill) and `results.md` (a summary table, ready to paste below).
 
Launch the comparison dashboard:
```bash
streamlit run dashboard.py
```

## Evaluation Methodology

| Field | Matching |
|---|---|
| Vendor name | Fuzzy match |
| Bill number | Normalized match |
| Date | Normalized exact match |
| Currency | Exact match |
| Amounts | Tolerance-based comparison |

Cost is computed from provider pricing; latency is the average API response time per bill. Every extraction is validated against a Pydantic schema before scoring, so malformed model output is caught rather than silently scored as correct.

## Results

**Overall Summary**

| Model | Bills | Exact Match (%) | Avg Latency (s) | Cost / Bill (USD) | Cost / 100 Bills (USD) |
|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | 15 | 13.3 | 21.77 | 0.00104 | 0.10 |
| Nemotron Nano 12B VL (free) | 15 | 0.0 | 5.80 | 0.0 | 0.0 |

**Field-Wise Accuracy (%)**

| Model | Vendor | Bill # | Date | Currency | Total | GST | Description |
|---|---|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite | 100.0 | 100.0 | 80.0 | 100.0 | 93.3 | 93.3 | 20.0 |
| Nemotron Nano 12B VL | 80.0 | 86.7 | 40.0 | 100.0 | 60.0 | 93.3 | 13.3 |

**Field Reliability Ranking — Gemini 3.5 Flash-Lite**

Currency = Vendor = Bill Number (100%) → Total / GST (93.3%) → Date (80%) → Description (20%)

**Error Distribution(in Fields)**

| Model | Missing | Hallucinated | Incorrect |
|---|---|---|---|
| Gemini 3.5 Flash-Lite | 0 | 1 | 16 |
| Nemotron Nano 12B VL | 4 | 3 | 27 |

**Ambiguous vs. Non-Ambiguous (Gemini)**

| Split | Bills | Exact Match (%) | Description Accuracy (%) |
|---|---|---|---|
| Ambiguous | 4 | 25.0 | 50.0 |
| Non-Ambiguous | 11 | 9.1 | 9.1 |

Exact match is *higher* on the ambiguous subset — a small-sample effect (4 bills) rather than a real trend; the ambiguous split's description phrasing happened to align more closely with ground truth. This is a limitation of the current dataset size, not a claim about model behavior on ambiguity itself.

**Key Findings**

- Vendor name, currency, and bill number are essentially solved fields for Gemini (100%).
- Description is the hardest field for both models — driven mostly by paraphrase mismatch, not extraction failure.
- Gemini is materially stronger than Nemotron on financial fields (total: 93.3% vs 60%; date: 80% vs 40%).
- Low exact-match accuracy (13.3%) is driven by one or two difficult fields per bill (usually description or date), not wholesale extraction failure — field-level accuracy tells a more accurate story than exact match alone.
- Nemotron's zero cost comes with materially more missing, hallucinated, and incorrect fields; it is attractive only when cost is the sole constraint.

## Zoho Expense Integration

`pushsamples.py` pushes five representative bills into Zoho Expense to demonstrate an end-to-end deployment, rather than all 15 — this avoids creating duplicate expense entries during repeated experimentation while still proving the extraction output is directly usable for downstream accounting automation.

## Streamlit Dashboard

- Upload a bill image
- Compare model extractions side by side
- View extracted, validated fields
- Push selected results to Zoho Expense

## Recommendation

**Gemini 3.5 Flash-Lite** is recommended for handwritten bill extraction: higher accuracy on nearly every field, stronger performance on financial data specifically, and an operating cost (~$0.10 per 100 bills) low enough to be a non-factor. Its higher latency (21.77s vs. 5.8s) is acceptable for a batch/asynchronous expense pipeline; it would need to be revisited for a real-time use case. The evaluation framework does not yet include digital/typed invoices, so this recommendation is scoped to handwritten bills — see below.

## Future Improvements

- Per-field confidence scores
- Batch processing
- Additional Vision LLMs, enabling benchmark-driven model selection (the framework already supports adding models — it does not yet auto-select one)
- Routing between printed and handwritten documents, potentially with different models per type
- Additional accounting integrations beyond Zoho

## Limitations

- Small dataset (15 bills) — results, especially the ambiguous/non-ambiguous split, should be read as directional, not statistically robust
- Ground truth quality depends on manual annotation and is itself imperfect for ambiguous bills
- Provider pricing is subject to change
- Nemotron benchmark used a free tier, which may not reflect paid-tier rate limits or model versions
- Handwriting diversity in the dataset is limited by what was collected, not representative of all handwriting styles

## Screenshots & Live Demo

- Screenshots: `<img width="1917" height="867" alt="image" src="https://github.com/user-attachments/assets/0113e3a1-9117-4478-bfcb-d5dc7408c1d0" />
 
`
- Live demo: `https://handwritten-bills-ocr-zohogit-5ga6xsjryvk6xeujhhejcm.streamlit.app/`

## Requirement Coverage

| Task requirement | Status |
|---|---|
| Dataset of 10–15 handwritten bills | ✅ 15 bills |
| ≥2–3 multimodal LLMs benchmarked | ⚠️ 2 models benchmarked (task asks for at least 2–3) |
| Field extraction (vendor, bill #, date, amount, currency, tax) | ✅ |
| Evaluation framework with defined ground truth | ✅ |
| Per-model, per-field accuracy | ✅ |
| Cost per model | ✅ |
| Zoho API expense entry | ✅ (Zoho Expense) |
| Written recommendation with justification | ✅ |
| Bonus UI | ✅ Streamlit dashboard |
