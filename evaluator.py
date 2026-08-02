# evaluator.py
import os
import re
import json
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from schemas import ReceiptData
from extractor import extract_with_gemini, extract_with_nemotron

GROUND_TRUTH_PATH = "data/ground_truth.json"
IMAGES_DIR = "data/images"
RESULTS_MD_PATH = "results.md"
RESULTS_JSON_PATH = "results.json"

# Gemini rate below is USD per 1M tokens, current as of Aug 2026 — VERIFY
# again before you trust the number in your write-up, it moves over time:
#   Gemini:     https://ai.google.dev/gemini-api/docs/pricing
#   OpenRouter: https://openrouter.ai/models (per-model "$0/M" tag = free)
PRICING_USD_PER_MILLION_TOKENS = {
    "Gemini 3.5 Flash-Lite": {"input": 0.30, "output": 2.50},
    "Nemotron Nano 12B VL (OpenRouter, free)": {"input": 0.0, "output": 0.0},
}

# --- CHANGE ------------------------------------------------------------------
# Bills were being extracted one at a time, fully sequentially — with
# per-bill latencies of 5-100+s seen on Gemini's free tier alone, a 15-bill
# run could take 20-40+ minutes end to end with nothing to show for it until
# it finished (or silently hung — see REQUEST_TIMEOUT_S in extractor.py).
# These are I/O-bound network calls, so running a few at once is safe and
# roughly linear in speedup.
#
# CHANGE: dropped from 3 to 2 concurrent requests. A run at 3 produced
# several 504s from Gemini and a cluster of malformed-response failures from
# OpenRouter's free Nemotron pool, both concentrated right at the start of
# the run when all 3 requests hit at once — both providers' free tiers were
# genuinely struggling under that burst, not just being slow. 2 is still a
# real speedup over one-at-a-time while giving both providers more headroom;
# extractor.py's retry logic (4 attempts, 5/10/20/40s backoff) is the second
# line of defense for whatever a request still hits at this concurrency.
# -------------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS = 2

# --- CHANGE ------------------------------------------------------------------
# Ground truth schema changed: tax_amount -> gst_number, plus two new fields
# (description, ambiguous). This is a substantial rewrite of the evaluator:
#
#   1. Scores gst_number and description instead of tax_amount.
#   2. Every field gets a null-aware, four-way classification — correct /
#      incorrect / missing / hallucinated — instead of a plain boolean match.
#      "The model correctly said nothing" and "the model invented a value"
#      are very different failure modes and are now counted separately,
#      rather than both just collapsing into "wrong".
#   3. Adds Exact Match Accuracy (every scored field correct on a bill),
#      per-field accuracy, and Missing/Hallucinated/Incorrect field counts,
#      each as an overall metric and as a per-model one.
#   4. Adds ambiguous vs. non-ambiguous stratification using the new
#      `ambiguous` ground-truth field ("Yes"/"No").
#   5. Adds a description-matching strategy (see description_match below),
#      since a description is a semantic label, not a fact with one correct
#      string — exact match or character-level fuzzy match (difflib, still
#      used for vendor_name) would both misscore it.
#   6. results.md now has seven sections instead of one summary table:
#      overall summary, field-wise accuracy, missing/hallucinated/incorrect
#      counts, ambiguous vs. non-ambiguous, failure analysis, sample
#      predictions, and suggestions for model improvement.
# -------------------------------------------------------------------------------

SCORED_FIELDS = ["vendor_name", "bill_number", "date", "currency", "total_amount", "gst_number", "description"]

FUZZY_MATCH_THRESHOLD = 0.85   # vendor_name / bill_number — OCR'd free text
AMOUNT_TOLERANCE_ABS = 1.0     # ₹1 absolute tolerance
AMOUNT_TOLERANCE_PCT = 0.01    # or 1%, whichever is looser

# --- Description matching --------------------------------------------------
# A description ("Grocery purchase" vs. "Grocery store bill") is a semantic
# label, not OCR'd free text — the two strings can share almost no characters
# in the same positions and still mean the same thing, which is exactly the
# case difflib's character-level SequenceMatcher (used below for vendor_name)
# handles badly. Without a paid embeddings API, the practical proxy used here
# is word-level (Jaccard) overlap after stripping punctuation and a short
# stopword list, plus a substring check as a second pass so a shorter,
# strictly-correct description ("Grocery purchase") isn't penalized against a
# more detailed truth ("Grocery store purchase"). 0.4 is a deliberately loose
# threshold given descriptions here are typically only 2-4 content words —
# losing even one shared word swings the Jaccard score a long way. Re-tune
# this against your own dataset once you have real description pairs to
# eyeball; document whatever you land on in the write-up, same as
# FUZZY_MATCH_THRESHOLD below.
# -----------------------------------------------------------------------------
DESCRIPTION_MATCH_THRESHOLD = 0.4
_DESCRIPTION_STOPWORDS = {"a", "an", "the", "of", "for", "and", "on", "at", "in", "to", "with", "or"}


def fuzzy_match(a, b) -> tuple[bool, float]:
    a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
    score = difflib.SequenceMatcher(None, a, b).ratio()
    return score >= FUZZY_MATCH_THRESHOLD, score


def amount_match(pred: float, truth: float) -> bool:
    return abs(pred - truth) <= max(AMOUNT_TOLERANCE_ABS, AMOUNT_TOLERANCE_PCT * truth)


def gst_match(pred, truth) -> tuple[bool, float]:
    """GSTIN is a fixed-format printed code, not free text — closer to
    date/currency (exact match) than to vendor_name (fuzzy match). Still
    normalizes case and strips spaces/hyphens first, since OCR sometimes
    inserts stray whitespace around a printed code without actually
    misreading a character."""
    a = re.sub(r"[\s\-]", "", str(pred or "")).upper()
    b = re.sub(r"[\s\-]", "", str(truth or "")).upper()
    matched = a == b
    return matched, (1.0 if matched else 0.0)


def _description_tokens(text) -> set[str]:
    text = re.sub(r"[^\w\s]", "", str(text or "").lower())
    return {w for w in text.split() if w not in _DESCRIPTION_STOPWORDS}


def description_match(pred, truth) -> tuple[bool, float]:
    pred_tokens, truth_tokens = _description_tokens(pred), _description_tokens(truth)
    if not pred_tokens or not truth_tokens:
        return False, 0.0
    union = pred_tokens | truth_tokens
    jaccard = len(pred_tokens & truth_tokens) / len(union) if union else 0.0
    pred_norm, truth_norm = " ".join(sorted(pred_tokens)), " ".join(sorted(truth_tokens))
    substring_hit = pred_norm in truth_norm or truth_norm in pred_norm
    return (jaccard >= DESCRIPTION_MATCH_THRESHOLD or substring_hit), round(jaccard, 3)


def _normalize(value):
    """None / "" / whitespace-only all mean 'genuinely absent'. Needed
    because Gemini's schema can't express Optional and returns "" instead of
    null (see GeminiReceiptData in schemas.py), while ground truth correctly
    uses JSON null — without this, every Gemini bill would look like a
    hallucination on any field it can't see."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def classify_field(field: str, extracted: ReceiptData, truth: dict) -> dict:
    """Null-aware, four-way classification for one field on one bill:
      - correct:      both present and match, OR both genuinely absent
      - incorrect:    both present, values don't match
      - missing:      truth has a value, model returned nothing
      - hallucinated: truth has no value, model invented one anyway — a
                      materially worse failure than a plain miss, since a
                      confident fabrication is more dangerous downstream
                      than an honest "not found".
    """
    truth_val = _normalize(truth.get(field))
    extracted_val = _normalize(getattr(extracted, field, None))

    if truth_val is None and extracted_val is None:
        return {"status": "correct", "truth": None, "extracted": None}
    if truth_val is None and extracted_val is not None:
        return {"status": "hallucinated", "truth": None, "extracted": extracted_val}
    if truth_val is not None and extracted_val is None:
        return {"status": "missing", "truth": truth_val, "extracted": None}

    sim = None
    if field in ("vendor_name", "bill_number"):
        matched, sim = fuzzy_match(extracted_val, truth_val)
    elif field == "date":
        matched = extracted_val == truth_val
    elif field == "currency":
        matched = extracted_val.upper() == truth_val.upper()
    elif field == "total_amount":
        try:
            matched = amount_match(float(extracted_val), float(truth_val))
        except ValueError:
            matched = False
    elif field == "gst_number":
        matched, sim = gst_match(extracted_val, truth_val)
    elif field == "description":
        matched, sim = description_match(extracted_val, truth_val)
    else:
        matched = extracted_val == truth_val

    result = {"status": "correct" if matched else "incorrect", "truth": truth_val, "extracted": extracted_val}
    if sim is not None:
        result["similarity"] = sim
    if field == "total_amount":
        try:
            result["abs_error"] = round(abs(float(extracted_val) - float(truth_val)), 2)
        except ValueError:
            pass
    return result


def score_bill(extracted: ReceiptData, truth: dict) -> dict:
    """Per-field classification for one bill, across every scored field."""
    return {field: classify_field(field, extracted, truth) for field in SCORED_FIELDS}


def compute_stats(records: list[dict]):
    """Aggregate stats over a list of per-bill field-classification dicts.
    Returns None for an empty slice (e.g. no ambiguous bills flagged yet) so
    callers can skip that section in the report instead of dividing by zero."""
    if not records:
        return None
    n = len(records)
    exact_matches = sum(1 for r in records if all(f["status"] == "correct" for f in r["fields"].values()))

    field_accuracy, missing, hallucinated, incorrect = {}, 0, 0, 0
    for field in SCORED_FIELDS:
        statuses = [r["fields"][field]["status"] for r in records]
        correct = sum(1 for s in statuses if s == "correct")
        field_accuracy[field] = round(100 * correct / len(statuses), 1)
        missing += sum(1 for s in statuses if s == "missing")
        hallucinated += sum(1 for s in statuses if s == "hallucinated")
        incorrect += sum(1 for s in statuses if s == "incorrect")

    return {
        "bills": n,
        "exact_match_accuracy": round(100 * exact_matches / n, 1),
        "field_accuracy": field_accuracy,
        "missing_field_count": missing,
        "hallucinated_field_count": hallucinated,
        "incorrect_field_count": incorrect,
    }


def evaluate_models():
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Error: {GROUND_TRUTH_PATH} not found!")
        return

    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    models = {
        "Gemini 3.5 Flash-Lite": extract_with_gemini,
        "Nemotron Nano 12B VL (OpenRouter, free)": extract_with_nemotron,
    }

    summary = {}
    raw_results = {}

    for model_name, extract_fn in models.items():
        print(f"\n=== Benchmarking {model_name} ===")
        bill_records = []
        total_time = total_input_tokens = total_output_tokens = 0.0
        successful_images = failures = 0

        bills_to_run = []
        for img_file, truth in ground_truth.items():
            img_path = os.path.join(IMAGES_DIR, img_file)
            if not os.path.exists(img_path):
                print(f"  Skipping {img_file} (image file missing)")
                continue
            bills_to_run.append((img_file, img_path, truth))

        total_images = len(bills_to_run)

        # CHANGE: bills for this model now run concurrently (bounded pool,
        # see MAX_CONCURRENT_REQUESTS above) instead of one after another.
        # All shared state (bill_records, the running totals) is only ever
        # touched here in the main thread as each future completes — worker
        # threads only run extract_fn and return/raise, so this stays
        # race-free without needing a lock.
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as pool:
            future_to_bill = {
                pool.submit(extract_fn, img_path): (img_file, truth)
                for img_file, img_path, truth in bills_to_run
            }

            for future in as_completed(future_to_bill):
                img_file, truth = future_to_bill[future]
                try:
                    result = future.result()
                    successful_images += 1
                    total_time += result.latency_s
                    total_input_tokens += result.input_tokens
                    total_output_tokens += result.output_tokens

                    field_results = score_bill(result.data, truth)
                    bill_records.append({
                        "file": img_file,
                        "ambiguous": str(truth.get("ambiguous", "No")).strip().lower() == "yes",
                        "fields": field_results,
                        "latency_s": round(result.latency_s, 2),
                        "extracted": result.data.model_dump(),
                    })

                    n_wrong = sum(1 for f in field_results.values() if f["status"] != "correct")
                    flag = "OK" if n_wrong == 0 else f"{n_wrong} field(s) off"
                    print(f"  - {img_file}: {result.latency_s:.2f}s | {flag}")

                except Exception as e:
                    failures += 1
                    print(f"  - {img_file}: FAILED ({e})")

        # Results complete in whatever order finishes first under
        # concurrency — resort by filename so results.json/results.md read
        # the same as they did before this change (bill_01, bill_02, ...).
        bill_records.sort(key=lambda r: r["file"])

        if total_images == 0:
            continue

        overall = compute_stats(bill_records)
        ambiguous_stats = compute_stats([r for r in bill_records if r["ambiguous"]])
        non_ambiguous_stats = compute_stats([r for r in bill_records if not r["ambiguous"]])

        rates = PRICING_USD_PER_MILLION_TOKENS.get(model_name, {"input": 0, "output": 0})
        total_cost_usd = (
            total_input_tokens / 1_000_000 * rates["input"]
            + total_output_tokens / 1_000_000 * rates["output"]
        )
        avg_latency = round(total_time / successful_images, 2) if successful_images else None
        cost_per_bill = round(total_cost_usd / successful_images, 5) if successful_images else None
        cost_per_100 = round(cost_per_bill * 100, 2) if cost_per_bill is not None else None

        summary[model_name] = {
            "Bills Attempted": total_images,
            "Bills Succeeded": successful_images,
            "Failures": failures,
            "Avg Latency (s)": avg_latency,
            "Total Tokens (in/out)": f"{total_input_tokens}/{total_output_tokens}",
            "Cost per Bill (USD)": cost_per_bill,
            "Cost per 100 Bills (USD)": cost_per_100,
            "Overall": overall,
            "Ambiguous": ambiguous_stats,
            "Non-Ambiguous": non_ambiguous_stats,
        }
        raw_results[model_name] = bill_records

    print("\n" + "=" * 55)
    print("           BENCHMARKING SUMMARY REPORT           ")
    print("=" * 55)
    print(json.dumps(summary, indent=2, default=str))

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump({"summary": summary, "raw": raw_results}, f, indent=2, default=str)

    write_markdown_report(summary, raw_results)
    print(f"\nWrote {RESULTS_JSON_PATH} and {RESULTS_MD_PATH} — paste the markdown into your README write-up.")


def _fmt(v):
    return "—" if v is None else str(v)


def write_markdown_report(summary: dict, raw_results: dict):
    """Generates a ready-to-paste markdown report for the required write-up.
    Expanded from a single accuracy/cost table into seven sections covering
    overall performance, per-field accuracy, error-type breakdown, the
    ambiguous/non-ambiguous split, concrete failure examples, a sample of raw
    predictions, and a few rule-based improvement suggestions per model."""
    if not summary:
        return

    lines = ["# Model Comparison — Handwritten Bill Extraction", ""]

    # 1. Overall summary --------------------------------------------------
    lines.append("## 1. Overall Summary\n")
    header = ["Model", "Bills Attempted", "Bills Succeeded", "Failures", "Exact Match Acc (%)",
              "Avg Latency (s)", "Cost / Bill (USD)", "Cost / 100 Bills (USD)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for model_name, m in summary.items():
        row = [
            model_name, m["Bills Attempted"], m["Bills Succeeded"], m["Failures"],
            _fmt(m["Overall"]["exact_match_accuracy"] if m["Overall"] else None),
            _fmt(m["Avg Latency (s)"]), _fmt(m["Cost per Bill (USD)"]), _fmt(m["Cost per 100 Bills (USD)"]),
        ]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines.append("")

    # 2. Field-wise accuracy ------------------------------------------------
    lines.append("## 2. Field-Wise Accuracy\n")
    header = ["Model"] + [f"{f} (%)" for f in SCORED_FIELDS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for model_name, m in summary.items():
        fa = m["Overall"]["field_accuracy"] if m["Overall"] else {}
        row = [model_name] + [_fmt(fa.get(f)) for f in SCORED_FIELDS]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines.append("")

    # 3. Missing / hallucinated / incorrect ---------------------------------
    lines.append("## 3. Missing, Hallucinated & Incorrect Fields\n")
    lines.append("| Model | Missing | Hallucinated | Incorrect |")
    lines.append("|---|---|---|---|")
    for model_name, m in summary.items():
        o = m["Overall"] or {}
        lines.append(
            f"| {model_name} | {o.get('missing_field_count', '—')} | "
            f"{o.get('hallucinated_field_count', '—')} | {o.get('incorrect_field_count', '—')} |"
        )
    lines.append("")

    # 4. Ambiguous vs. non-ambiguous ----------------------------------------
    lines.append("## 4. Ambiguous vs. Non-Ambiguous Bills\n")
    any_ambiguous = any(m["Ambiguous"] for m in summary.values())
    if not any_ambiguous:
        lines.append(
            '_No bills are currently flagged `"ambiguous": "Yes"` in ground truth — '
            "this section will populate once some are._\n"
        )
    else:
        lines.append(
            "| Model | Ambiguous Exact Match (%) | Ambiguous Bills | "
            "Non-Ambiguous Exact Match (%) | Non-Ambiguous Bills |"
        )
        lines.append("|---|---|---|---|---|")
        for model_name, m in summary.items():
            a, na = m["Ambiguous"], m["Non-Ambiguous"]
            lines.append(
                f"| {model_name} | {_fmt(a['exact_match_accuracy'] if a else None)} | {a['bills'] if a else 0} | "
                f"{_fmt(na['exact_match_accuracy'] if na else None)} | {na['bills'] if na else 0} |"
            )
    lines.append("")

    # 5. Failure analysis -----------------------------------------------------
    lines.append("## 5. Failure Analysis\n")
    for model_name, records in raw_results.items():
        failing = [r for r in records if any(f["status"] != "correct" for f in r["fields"].values())]
        lines.append(f"### {model_name} — {len(failing)} bill(s) with at least one field off\n")
        if not failing:
            lines.append("_No field-level misses._\n")
            continue
        lines.append("| Bill | Field | Status | Ground Truth | Extracted |")
        lines.append("|---|---|---|---|---|")
        for r in failing:
            for field, f in r["fields"].items():
                if f["status"] != "correct":
                    lines.append(f"| {r['file']} | {field} | {f['status']} | {_fmt(f['truth'])} | {_fmt(f['extracted'])} |")
        lines.append("")

    # 6. Sample predictions ----------------------------------------------------
    lines.append("## 6. Sample Predictions\n")
    for model_name, records in raw_results.items():
        lines.append(f"### {model_name}\n")
        sample = records[:5]
        lines.append("| Bill | Vendor (extracted) | Total (extracted) | Date (extracted) | GST # (extracted) | Description (extracted) |")
        lines.append("|---|---|---|---|---|---|")
        for r in sample:
            e = r["extracted"]
            lines.append(
                f"| {r['file']} | {e.get('vendor_name','')} | {e.get('total_amount','')} | "
                f"{e.get('date','')} | {_fmt(e.get('gst_number'))} | {_fmt(e.get('description'))} |"
            )
        lines.append("")

    # 7. Suggestions for model improvement --------------------------------------
    lines.append("## 7. Suggestions for Model Improvement\n")
    for model_name, m in summary.items():
        o = m["Overall"] or {}
        fa = o.get("field_accuracy", {})
        suggestions = []
        for field, acc in fa.items():
            if acc < 70:
                suggestions.append(
                    f"- `{field}` accuracy is low ({acc}%) — check which specific bills fail on this field "
                    "(Section 5) before assuming it's a model-ceiling issue rather than a prompt or "
                    "image-quality one."
                )
        if o.get("hallucinated_field_count", 0) > 0:
            suggestions.append(
                f"- {o['hallucinated_field_count']} hallucinated field(s) — consider strengthening the "
                'prompt\'s "never guess, use empty/0 if not visible" instruction, or adding a couple of '
                "few-shot examples that show an explicitly-empty field."
            )
        if o.get("missing_field_count", 0) > 0:
            suggestions.append(
                f"- {o['missing_field_count']} missing field(s) — check whether these are genuinely "
                "illegible on the source image or a case the model reliably skips regardless of legibility."
            )
        if not suggestions:
            suggestions.append(
                "- No clear systemic weakness stood out in this run — differences are likely down to "
                "individual bill difficulty rather than a fixable pattern."
            )
        lines.append(f"**{model_name}**\n")
        lines.extend(suggestions)
        lines.append("")

    with open(RESULTS_MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    evaluate_models()