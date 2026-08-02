# Handwritten Bill Extraction and Model Evaluation

Extracts structured expense data from photos of handwritten Indian bills using three vision-capable LLMs then posts the result to Zoho Books as an expense. Includes an evaluation framework that scores each model per field and reports API cost so the two can be weighed against each other.

## Approach

Digital invoices are easy for LLMs to read. Handwritten shop bills are not — inconsistent handwriting, faded ink and cramped totals make extraction genuinely hard and not every model handles that equally well. This project answers two questions: which model reads handwritten Indian bills most accurately and does the accuracy gained over a cheaper model actually justify its cost.

The pipeline has five stages:

1. **Ingestion** — read a bill image from disk and encode it
2. **Vision extraction** — send the image to a vision LLM with a strict prompt and a JSON schema
3. **Schema validation** — parse the model's response into a Pydantic model so every downstream step works with the same shape of data regardless of which model produced it
4. **Evaluation** — score the extraction against a human-verified ground truth and record cost and latency
5. **Zoho sync** — post the extracted fields as a real expense entry in Zoho Books

Two models were compared: Gemini 3.6 Flash (called directly against Google's API, paid/trial credits) and NVIDIA Nemotron Nano 12B VL, a free vision-capable model called through [OpenRouter](https://openrouter.ai), trained specifically for OCR/document intelligence. Routing the free model through OpenRouter rather than a single vendor's own free tier sidesteps a real problem this project ran into: two other "obvious" free vision options (Groq's Llama 4 Scout, Mistral's Pixtral) were both deprecated by their providers by mid-2026, and a second OpenRouter model tried here (Google Gemma 4 31B) was dropped after its free tier proved too rate-limited under real use. This also turns the accuracy-vs-cost question sharper rather than softer — it's now "does Gemini's non-zero cost buy enough accuracy over a $0 model to justify it."

## File guide

| File | What it does |
|---|---|
| `schemas.py` | The `ReceiptData` and `LineItem` Pydantic models — the single source of truth for what a "correctly extracted bill" looks like. Also defines a defaults-free twin schema used only for the Gemini call, since Gemini's structured output rejects fields with default values. |
| `extractor.py` | One function per model (`extract_with_gemini`, `extract_with_nemotron`). The latter calls a private `_extract_with_openrouter` helper since it's routed through OpenRouter's OpenAI-compatible endpoint. Each returns an `ExtractionResult` holding the parsed data plus token counts and latency. |
| `zoho_client.py` | Handles the Zoho OAuth refresh flow and posts an extracted bill as a new expense, including looking up or creating the vendor as a proper Zoho contact. |
| `check_env.py` | Run this first. Confirms all API keys are present and that the Zoho refresh token actually works before you run anything that costs money or writes data. |
| `main.py` | CLI entry point for processing a single bill with one chosen model, useful for spot-checking one image without running the full evaluation. |
| `evaluator.py` | The core of the project. Runs every model against every bill in the dataset, scores each field against `ground_truth.json` and writes `results.json` and `results.md`. |
| `dashboard.py` | Bonus feature. A Streamlit UI to upload a bill and compare all three models' extractions side by side without touching the CLI. |
| `data/images/` | The 15 bill photos used for evaluation. |
| `data/ground_truth.json` | Human-verified correct values for each bill in `data/images/`, used as the answer key. |

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

## Evaluation methodology

**What counts as correct.** Handwritten OCR will not produce byte-identical strings even when a human would call the extraction correct, so different fields are scored differently:

- **Vendor name and bill number** use fuzzy string similarity (`difflib`, threshold 0.85) rather than exact match. "Gupta Kirana & General Store" and "Gupta Kirana and General Store" should both count as correct since they're the same answer and requiring exact matches would mostly measure punctuation luck instead of extraction quality.
- **Date and currency** use exact match. Both are structured, unambiguous fields — a model that reads the date correctly should produce exactly `2026-02-15`, so there is no reason to allow fuzziness here.
- **Total amount and tax amount** use a tolerance match, correct if within ₹1 or 1 percent of the true value, whichever is looser, alongside the raw mean absolute error so you can see both the pass rate and the magnitude of error.

Each field's accuracy is reported separately per model rather than blended into one score, since a model that nails vendor names but consistently misreads totals is a very different result from one that's mediocre across the board and a blended number would hide that.

**Cost.** Every extraction call returns the actual input and output token counts the provider reports, which are multiplied by that provider's published per-token pricing to get a real dollar cost per bill rather than an estimate. That's then extrapolated to a cost per 100 bills for an easier real-world comparison. Pricing was current as of when this was written — token pricing moves often enough that it's worth re-checking each provider's pricing page before trusting the numbers in a final write-up.

**Latency.** Wall-clock time for each API call, averaged across all bills a model successfully processed.

## Accuracy and cost comparison

*Fill this in with the contents of `results.md` after running `python evaluator.py` on your dataset. Template below.*
Model Comparison — Handwritten Bill Extraction
Model	bill_number acc %	currency acc %	date acc %	tax_amount acc %	total_amount acc %	vendor_name acc %	Avg Latency (s)	Cost / Bill (USD)	Cost / 100 Bills (USD)
Gemini 3.5 Flash-Lite	100.0	73.3	75.0	93.3	86.7	100.0	1.9	0.00099	0.1
Nemotron Nano 12B VL (OpenRouter, free)	85.7	100.0	37.5	90.0	60.0	80.0	6.01	0.0	0.0


## Final recommendation

*Fill this in once the table above has real numbers. A few questions worth answering explicitly rather than just naming a winner:*

- **Which model has the highest accuracy on the fields that matter most for an expense entry** — total amount and date are the two fields an accounting system actually depends on being correct, so weigh those more heavily than vendor name or tax.
- **Does the more accurate model's cost premium hold up at scale.** A model that's 5 percent more accurate but 4 times the price might be worth it if you're processing a handful of bills a month and not worth it at thousands of bills a month. Use the cost-per-100-bills figure to make this concrete.
- **Is the same model right for both digital and handwritten bills, or does this warrant two different pipelines.** Digital invoices are close to a solved problem for any of these models, so it may make sense to route digital documents to the cheapest model available and reserve the more accurate model for handwritten ones specifically, rather than paying the handwriting-tier price for every document.
- **Where each model actually failed.** A quick look at `results.json` for the bills a model got wrong often reveals a pattern — faint ink, a specific handwriting style or a particular bill layout — that's more useful for the write-up than the aggregate percentage alone.

## Bonus dashboard

`dashboard.py` is a minimal Streamlit app: upload a bill photo, choose which models to run and see every extracted field laid out side by side in a table along with latency, token counts and each model's line items. A model's extraction can be posted straight to Zoho Books from the same screen. Run locally with `streamlit run dashboard.py` — no separate hosting needed for this to work as a deliverable.

## Known limitations

- Ground truth accuracy is bounded by how carefully it was annotated. A rushed or guessed ground-truth value will misrepresent every model's score on that field equally.
- Fuzzy-match and tolerance thresholds are a judgment call made for this project. A different threshold would shift the reported accuracy numbers somewhat, though it shouldn't change which model ranks best relative to the others.
- Cost figures depend on provider pricing at the time of running and will drift over time. Re-run the pricing check before reusing these numbers in a later report.
- Nemotron Nano 12B VL is a free-tier model on OpenRouter, which means $0 cost but real rate limits (requests/min and tokens/min). `extract_with_nemotron` retries on 429s with exponential backoff, but a saturated shared pool can still slow a full batch run down. Free-tier model availability on OpenRouter also rotates — a second free model (Google Gemma 4 31B) was tried and dropped from this project for exactly that reason; if `nvidia/nemotron-nano-12b-v2-vl:free` 404s later, check [openrouter.ai/models](https://openrouter.ai/models?max_price=0&modality=vision) for a current replacement.
