# extractor.py
import os
import json
import time
import base64
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import openai  # also used for the OpenRouter calls below — OpenRouter exposes an OpenAI-compatible endpoint

from schemas import ReceiptData, GeminiReceiptData

load_dotenv()

# --- CHANGE ------------------------------------------------------------
# Ground truth schema swapped tax_amount for gst_number and added
# description (schemas.py). The prompt now asks for both explicitly instead
# of leaving the model to guess what "gst_number" or "description" mean from
# the JSON schema field names alone.
# -------------------------------------------------------------------------
PROMPT_TEXT = """
Analyze this receipt/bill image carefully and extract all information strictly according to the requested JSON schema.
- Convert all dates to YYYY-MM-DD format.
- Ensure total_amount correctly reflects the final payable sum.
- If individual line items are visible, capture them in line_items. Otherwise, capture the overall expense.
- If a GST number (GSTIN) is printed or handwritten on the bill, capture it in gst_number exactly as shown.
  If no GST number is visible, use an empty string ("").
- Write a short one-line description of what this bill is for (e.g. "Grocery purchase", "Printing services",
  "Restaurant bill"), inferred from the vendor name and any visible line items.
- If a field is not visible on the bill, use an empty string ("") for text fields or 0 for numbers — never guess.
"""

# --- CHANGE ---------------------------------------------------------------
# Swapped the two paid comparison models (OpenAI, Claude) for one free
# vision-capable model routed through OpenRouter (https://openrouter.ai),
# so two of the three models in the comparison run at $0 marginal API cost.
# Gemini is untouched — still called directly against Google's API, not
# through OpenRouter.
#
# Model IDs, current as of Aug 2026 — re-verify before you run the eval:
#   Gemini:     https://ai.google.dev/gemini-api/docs/models
#   OpenRouter: https://openrouter.ai/models?max_price=0&modality=vision
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.5-flash-lite"                      # ~$0.30 / $2.50 per 1M tokens (in/out), current as of Aug 2026
OPENROUTER_NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"  # 12B, vision + tool-calling, tuned for OCR/document intelligence, 128K context, $0/$0

# --- CHANGE ------------------------------------------------------------------
# Both clients are created once at import time and reused across every call
# instead of a fresh client per bill (connection reuse), with an explicit
# timeout so a stalled request fails instead of hanging the whole run.
#
# Retry constants are shared by both extractors below. Both Gemini's free
# tier (504 DEADLINE_EXCEEDED under load) and OpenRouter's free pool (429s,
# and occasionally a malformed/empty response instead of a clean error code)
# throw transient failures under real batch-run conditions — a single-shot
# request isn't reliable on either provider's no-cost tier. 4 retries with
# 5/10/20/40s backoff (~75s worst case) trades a bit of time for not losing
# an entire bill to what's usually a temporary provider hiccup.
# -------------------------------------------------------------------------------
REQUEST_TIMEOUT_S = 90
MAX_RETRIES = 4
BASE_BACKOFF_S = 5  # 5, 10, 20, 40s

GEMINI_CLIENT = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000),  # this SDK takes milliseconds
)
OPENROUTER_CLIENT = openai.OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=REQUEST_TIMEOUT_S,  # this SDK takes seconds
)


@dataclass
class ExtractionResult:
    """Wraps the parsed receipt together with what we need to compute cost/latency,
    since the original approach only returned ReceiptData and had no way to score
    cost per model (a required deliverable in the task spec)."""
    data: ReceiptData
    input_tokens: int
    output_tokens: int
    latency_s: float
    model_name: str


def get_mime_type(image_path: str) -> str:
    """Mime type only, no file I/O — split out of encode_image() so callers
    that don't need the base64 string (extract_with_gemini) don't pay for
    reading and base64-encoding the whole image just to throw it away."""
    ext = Path(image_path).suffix.lower()
    if ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def encode_image(image_path: str) -> tuple[str, str]:
    """Helper to convert local image to base64 string and identify mime type."""
    mime_type = get_mime_type(image_path)

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded_string, mime_type


def extract_with_gemini(image_path: str) -> ExtractionResult:
    """Extract receipt data using Gemini 3.5 Flash-Lite.

    CHANGE: retries on genai_errors.ServerError (5xx — includes the
    504 DEADLINE_EXCEEDED seen under concurrent free-tier load) instead of
    failing the bill on the first transient server error.
    """
    encoded_bytes = Path(image_path).read_bytes()
    mime_type = get_mime_type(image_path)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()
            response = GEMINI_CLIENT.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=encoded_bytes, mime_type=mime_type),
                    PROMPT_TEXT
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    # Gemini's response_schema rejects Pydantic fields with defaults —
                    # GeminiReceiptData is the defaults-free twin, see schemas.py.
                    response_schema=GeminiReceiptData,
                )
            )
            latency = time.time() - start

            parsed_json = json.loads(response.text)
            receipt = ReceiptData(**parsed_json)

            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0

            return ExtractionResult(receipt, input_tokens, output_tokens, latency, GEMINI_MODEL)

        except genai_errors.ServerError as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                raise
            wait_s = BASE_BACKOFF_S * (2 ** attempt)
            print(f"  [{GEMINI_MODEL}] server error on attempt {attempt + 1}/{MAX_RETRIES} "
                  f"({e}) — retrying in {wait_s}s")
            time.sleep(wait_s)

    raise last_error  # unreachable — loop always returns or raises above


# --- CHANGE -----------------------------------------------------------------
# extract_with_openai and extract_with_claude are gone. In their place: a
# helper that talks to OpenRouter (https://openrouter.ai) plus a thin wrapper
# for the one free model kept (Nemotron), so main.py/dashboard.py/evaluator.py
# keep calling a one-function-per-model API, unchanged.
#
# Structured output is enforced with a forced tool call (tool_choice pinned
# to "record_receipt") rather than OpenAI's response_format=<pydantic model>
# .parse() helper, since that strict-schema path isn't guaranteed to work
# against arbitrary third-party models routed through OpenRouter.
#
# CHANGE: retry now covers more than just 429s. Under real batch-run load,
# OpenRouter's free pool doesn't only fail with a clean rate-limit error —
# it sometimes returns a 200 with an empty/malformed choices list (surfacing
# as a cryptic "'NoneType' object is not subscriptable" if indexed blindly),
# or a 5xx from an overloaded upstream provider. All of these are now
# explicitly checked for and treated as retryable, with a clear log line
# instead of a raw Python TypeError.
# ------------------------------------------------------------------------------

def _extract_with_openrouter(image_path: str, model_id: str) -> ExtractionResult:
    """Shared implementation for both OpenRouter-hosted free vision models."""
    b64_image, mime_type = encode_image(image_path)

    tool_schema = {
        "type": "function",
        "function": {
            "name": "record_receipt",
            "description": "Save structured receipt data",
            "parameters": ReceiptData.model_json_schema(),
        },
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()
            response = OPENROUTER_CLIENT.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT_TEXT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                            }
                        ]
                    }
                ],
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": "record_receipt"}},
                # Optional: attributes these calls to this project on OpenRouter's
                # public leaderboards. Safe to delete, doesn't affect the free tier.
                extra_headers={"HTTP-Referer": "https://github.com/", "X-Title": "Taxor Bill Extractor"},
            )
            latency = time.time() - start

            # CHANGE: explicit checks instead of indexing straight into
            # response.choices[0] — an overloaded free provider can return a
            # 200 with choices=None or an empty list, which used to surface
            # as an opaque 'NoneType' object is not subscriptable crash.
            if not response.choices:
                raise RuntimeError(f"{model_id} returned no choices (likely an overloaded free-tier provider)")

            message = response.choices[0].message
            if not message.tool_calls:
                raise RuntimeError(f"{model_id} failed to produce a tool call.")

            parsed_json = json.loads(message.tool_calls[0].function.arguments)
            receipt = ReceiptData(**parsed_json)

            usage = response.usage
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

            return ExtractionResult(receipt, input_tokens, output_tokens, latency, model_id)

        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError, RuntimeError) as e:
            last_error = e
            if attempt == MAX_RETRIES - 1:
                raise
            wait_s = BASE_BACKOFF_S * (2 ** attempt)
            print(f"  [{model_id}] transient error on attempt {attempt + 1}/{MAX_RETRIES} "
                  f"({e}) — retrying in {wait_s}s")
            time.sleep(wait_s)

    raise last_error  # unreachable — loop always returns or raises above


def extract_with_nemotron(image_path: str) -> ExtractionResult:
    """Extract receipt data using NVIDIA Nemotron Nano 12B VL via OpenRouter
    (free tier) — trained specifically for OCR/document intelligence, which
    makes it a natural second comparison point for handwritten bills."""
    return _extract_with_openrouter(image_path, OPENROUTER_NEMOTRON_MODEL)