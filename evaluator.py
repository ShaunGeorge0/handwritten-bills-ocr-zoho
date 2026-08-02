# evaluator.py
import os
import json
import difflib
from schemas import ReceiptData
from extractor import extract_with_gemini, extract_with_openai, extract_with_claude

GROUND_TRUTH_PATH = "data/ground_truth.json"
IMAGES_DIR = "data/images"
RESULTS_MD_PATH = "results.md"
RESULTS_JSON_PATH = "results.json"

# --- CHANGE -----------------------------------------------------------------
# The original evaluator had NO cost tracking at all, even though the task
# spec explicitly requires "Report cost per model (API cost per bill, or per
# 100 bills extrapolated)". extract_with_* now return token usage, so we can
# compute this from what the API actually reports rather than guessing.
#
# Rates below are USD per 1M tokens, current as of Aug 2026 (cross-checked
# against provider + third-party pricing pages at the time this was written).
# VERIFY these again before you trust the numbers in your write-up — token
# pricing on all three providers has moved multiple times in 2026:
#   Gemini:  https://ai.google.dev/gemini-api/docs/pricing
#   OpenAI:  https://platform.openai.com/docs/pricing
#   Claude:  https://www.anthropic.com/pricing
# ------------------------------------------------------------------------------
PRICING_USD_PER_MILLION_TOKENS = {
    "Gemini 2.5 Flash": {"input": 0.15, "output": 1.25},
    "GPT-5 Mini": {"input": 0.25, "output": 2.00},
    "Claude Haiku 4.5": {"input": 1.00, "output": 5.00},
}

# --- CHANGE -----------------------------------------------------------------
# The original evaluator only scored vendor_name (exact, case-insensitive)
# and date (exact), plus a total_amount MAE. That's not "per field" scoring
# as the spec asks for, and exact-string vendor matching is much too strict
# for handwritten-OCR output — "Gupta Kirana & General Store" vs "Gupta
# Kirana and General Store" is functionally correct but would score as a
# total miss.
#
# Fix: fuzzy similarity (difflib, stdlib — swap in rapidfuzz for something
# faster/better if you want) for free-text fields, exact match for
# structured fields (date, currency), and tolerance-based matching for
# numeric fields. FUZZY_MATCH_THRESHOLD is the "what counts as correct"
# judgment call the spec asks you to make explicit — 0.85 similarity is a
# reasonable starting point for OCR'd shop names; document your own
# reasoning for whatever you pick in the write-up.
# ------------------------------------------------------------------------------
FUZZY_MATCH_THRESHOLD = 0.85
AMOUNT_TOLERANCE_ABS = 1.0     # ₹1 absolute tolerance
AMOUNT_TOLERANCE_PCT = 0.01    # or 1%, whichever is looser


def fuzzy_match(a: str, b: str) -> tuple[bool, float]:
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    score = difflib.SequenceMatcher(None, a, b).ratio()
    return score >= FUZZY_MATCH_THRESHOLD, score


def amount_match(pred: float, truth: float) -> bool:
    return abs(pred - truth) <= max(AMOUNT_TOLERANCE_ABS, AMOUNT_TOLERANCE_PCT * truth)


def score_bill(extracted: ReceiptData, truth: dict) -> dict:
    """Returns a per-field correctness dict for one bill. Only scores fields
    that are actually present in the ground-truth annotation, so ground_truth.json
    doesn't need every optional field filled in for every bill."""
    scores = {}

    if "vendor_name" in truth:
        matched, sim = fuzzy_match(extracted.vendor_name, truth["vendor_name"])
        scores["vendor_name"] = {"match": matched, "similarity": round(sim, 3)}

    if "bill_number" in truth:
        matched, sim = fuzzy_match(extracted.bill_number or "", truth["bill_number"] or "")
        scores["bill_number"] = {"match": matched, "similarity": round(sim, 3)}

    if "date" in truth:
        scores["date"] = {"match": extracted.date == truth["date"]}

    if "currency" in truth:
        scores["currency"] = {"match": (extracted.currency or "").upper() == truth["currency"].upper()}

    if "total_amount" in truth:
        mae = abs(extracted.total_amount - truth["total_amount"])
        scores["total_amount"] = {"match": amount_match(extracted.total_amount, truth["total_amount"]), "abs_error": round(mae, 2)}

    if truth.get("tax_amount") is not None and extracted.tax_amount is not None:
        mae = abs(extracted.tax_amount - truth["tax_amount"])
        scores["tax_amount"] = {"match": amount_match(extracted.tax_amount, truth["tax_amount"]), "abs_error": round(mae, 2)}

    return scores


def evaluate_models():
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Error: {GROUND_TRUTH_PATH} not found!")
        return

    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    models = {
        "Gemini 2.5 Flash": extract_with_gemini,
        "GPT-5 Mini": extract_with_openai,
        "Claude Haiku 4.5": extract_with_claude,
    }

    summary = {}
    raw_results = {}

    for model_name, extract_fn in models.items():
        print(f"\n=== Benchmarking {model_name} ===")
        field_scores: dict[str, list[bool]] = {}
        total_time = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        total_images = 0
        failures = 0

        for img_file, truth in ground_truth.items():
            img_path = os.path.join(IMAGES_DIR, img_file)
            if not os.path.exists(img_path):
                print(f"  Skipping {img_file} (image file missing)")
                continue

            total_images += 1
            try:
                result = extract_fn(img_path)
                total_time += result.latency_s
                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens

                bill_scores = score_bill(result.data, truth)
                for field, s in bill_scores.items():
                    field_scores.setdefault(field, []).append(s["match"])

                raw_results.setdefault(model_name, {})[img_file] = {
                    "extracted": result.data.model_dump(),
                    "scores": bill_scores,
                    "latency_s": round(result.latency_s, 2),
                }

                amt_note = bill_scores.get("total_amount", {}).get("abs_error", "n/a")
                print(f"  - {img_file}: {result.latency_s:.2f}s | total_amount abs error: {amt_note}")

            except Exception as e:
                failures += 1
                print(f"  - {img_file}: FAILED ({e})")

        if total_images == 0:
            continue

        # --- per-field accuracy, reported separately (not blended) ---
        per_field_accuracy = {
            field: round(100 * sum(matches) / len(matches), 1)
            for field, matches in field_scores.items()
        }

        # --- cost: computed from tokens the API actually reported ---
        rates = PRICING_USD_PER_MILLION_TOKENS.get(model_name, {"input": 0, "output": 0})
        total_cost_usd = (
            total_input_tokens / 1_000_000 * rates["input"]
            + total_output_tokens / 1_000_000 * rates["output"]
        )
        cost_per_bill = total_cost_usd / total_images if total_images else 0.0

        summary[model_name] = {
            "Bills Processed": total_images,
            "Failures": failures,
            "Avg Latency (s)": round(total_time / total_images, 2),
            "Per-Field Accuracy (%)": per_field_accuracy,
            "Total Tokens (in/out)": f"{total_input_tokens}/{total_output_tokens}",
            "Cost per Bill (USD)": round(cost_per_bill, 5),
            "Cost per 100 Bills (USD)": round(cost_per_bill * 100, 2),
        }

    print("\n" + "=" * 55)
    print("           BENCHMARKING SUMMARY REPORT           ")
    print("=" * 55)
    print(json.dumps(summary, indent=2))

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump({"summary": summary, "raw": raw_results}, f, indent=2)

    write_markdown_report(summary)
    print(f"\nWrote {RESULTS_JSON_PATH} and {RESULTS_MD_PATH} — paste the markdown table into your README write-up.")


def write_markdown_report(summary: dict):
    """Generates a ready-to-paste markdown table for the required write-up
    (accuracy/cost table across models) — the original evaluator only printed
    to stdout, with nothing saved for the deliverable."""
    if not summary:
        return

    all_fields = sorted({f for m in summary.values() for f in m["Per-Field Accuracy (%)"]})

    lines = ["# Model Comparison — Handwritten Bill Extraction", ""]
    header = ["Model"] + [f"{f} acc %" for f in all_fields] + ["Avg Latency (s)", "Cost / Bill (USD)", "Cost / 100 Bills (USD)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for model_name, m in summary.items():
        row = [model_name]
        for f in all_fields:
            row.append(str(m["Per-Field Accuracy (%)"].get(f, "—")))
        row += [str(m["Avg Latency (s)"]), str(m["Cost per Bill (USD)"]), str(m["Cost per 100 Bills (USD)"])]
        lines.append("| " + " | ".join(row) + " |")

    with open(RESULTS_MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    evaluate_models()