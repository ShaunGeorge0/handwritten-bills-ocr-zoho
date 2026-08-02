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
import openai  # also used for the OpenRouter calls below — OpenRouter exposes an OpenAI-compatible endpoint

from schemas import ReceiptData, GeminiReceiptData

load_dotenv()

PROMPT_TEXT = """
Analyze this receipt/bill image carefully and extract all information strictly according to the requested JSON schema.
- Convert all dates to YYYY-MM-DD format.
- Ensure total_amount correctly reflects the final payable sum.
- If individual line items are visible, capture them in line_items. Otherwise, capture the overall expense.
- If a field is not visible on the bill, use an empty string ("") for text fields or 0 for numbers — never guess.
"""

# --- CHANGE ---------------------------------------------------------------
# Swapped the two paid comparison models (OpenAI, Claude) for one free
# vision-capable model routed through OpenRouter (https://openrouter.ai),
# so two of the three models in the comparison run at $0 marginal API cost.
# Gemini is untouched — still called directly against Google's API, not
# through OpenRouter.
#
# Google Gemma 4 31B (also via OpenRouter) was tried here too but dropped —
# its free tier sits on a shared, easily-saturated rate-limit pool and kept
# 429ing under real usage. Nemotron didn't have that problem in practice.
# If you want a third model back, check OpenRouter's current free
# vision-capable list before picking one — free-tier availability rotates:
#   https://openrouter.ai/models?max_price=0&modality=vision
#
# Model IDs, current as of Aug 2026 — re-verify before you run the eval:
#   Gemini:     https://ai.google.dev/gemini-api/docs/models
#   OpenRouter: https://openrouter.ai/models?max_price=0&modality=vision
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.5-flash-lite"                      # ~$0.30 / $2.50 per 1M tokens (in/out), current as of Aug 2026
OPENROUTER_NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"  # 12B, vision + tool-calling, tuned for OCR/document intelligence, 128K context, $0/$0


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


def encode_image(image_path: str) -> tuple[str, str]:
    """Helper to convert local image to base64 string and identify mime type."""
    mime_type = "image/jpeg"
    ext = Path(image_path).suffix.lower()
    if ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded_string, mime_type


def extract_with_gemini(image_path: str) -> ExtractionResult:
    """Extract receipt data using Gemini 3.5 Flash-Lite."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    encoded_bytes = Path(image_path).read_bytes()
    _, mime_type = encode_image(image_path)

    start = time.time()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=encoded_bytes, mime_type=mime_type),
            PROMPT_TEXT
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            # CHANGE: was response_schema=ReceiptData, which has Field defaults
            # and raises ValueError on every call. Use the defaults-free twin.
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


# --- CHANGE -----------------------------------------------------------------
# extract_with_openai and extract_with_claude are gone. In their place: a
# helper that talks to OpenRouter (https://openrouter.ai) plus a thin wrapper
# for the one free model kept (Nemotron), so main.py/dashboard.py/evaluator.py
# keep calling a one-function-per-model API, unchanged. Kept as a separate
# helper + wrapper (rather than inlining) so adding a second/third OpenRouter
# model back later is a two-line change, not a rewrite.
#
# Structured output is enforced with a forced tool call (tool_choice pinned
# to "record_receipt"), the same pattern the original extract_with_claude
# used — not OpenAI's response_format=<pydantic model> .parse() helper, since
# that strict-schema path isn't guaranteed to work against arbitrary
# third-party models routed through OpenRouter. Forced tool-calling is safe
# here specifically because Nemotron is tagged "Tools"-capable on OpenRouter
# as of Aug 2026 — re-check that tag on OpenRouter's model page before
# swapping in a different free model.
# ------------------------------------------------------------------------------

RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_BACKOFF_S = 5  # 5, 10, 20, 40, 80s — total worst case ~2.5min before giving up


def _extract_with_openrouter(image_path: str, model_id: str) -> ExtractionResult:
    """Shared implementation for both OpenRouter-hosted free vision models.

    CHANGE: free-tier models on OpenRouter share one rate-limited pool across
    everyone NOT using their own linked provider key (see the 429 remedy_hint
    OpenRouter returns: "add your own key to accumulate your rate limits").
    That pool routinely saturates under a batch run like evaluator.py's
    15-bills x N-models loop, so a plain single-shot request isn't reliable
    here the way it was for OpenAI/Claude's paid, per-account rate limits.
    Retry with exponential backoff specifically on 429s; anything else
    (auth errors, malformed schema, etc.) still fails immediately.
    """
    client = openai.OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    b64_image, mime_type = encode_image(image_path)

    tool_schema = {
        "type": "function",
        "function": {
            "name": "record_receipt",
            "description": "Save structured receipt data",
            "parameters": ReceiptData.model_json_schema(),
        },
    }

    for attempt in range(RATE_LIMIT_MAX_RETRIES):
        try:
            start = time.time()
            response = client.chat.completions.create(
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
            break
        except openai.RateLimitError:
            if attempt == RATE_LIMIT_MAX_RETRIES - 1:
                raise
            wait_s = RATE_LIMIT_BASE_BACKOFF_S * (2 ** attempt)
            print(f"  [{model_id}] rate-limited on OpenRouter's shared free pool — "
                  f"retrying in {wait_s}s (attempt {attempt + 1}/{RATE_LIMIT_MAX_RETRIES})")
            time.sleep(wait_s)

    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    message = response.choices[0].message
    if not message.tool_calls:
        raise RuntimeError(f"{model_id} failed to produce a tool call.")

    parsed_json = json.loads(message.tool_calls[0].function.arguments)
    receipt = ReceiptData(**parsed_json)

    return ExtractionResult(receipt, input_tokens, output_tokens, latency, model_id)


def extract_with_nemotron(image_path: str) -> ExtractionResult:
    """Extract receipt data using NVIDIA Nemotron Nano 12B VL via OpenRouter
    (free tier) — trained specifically for OCR/document intelligence, which
    makes it a natural second comparison point for handwritten bills."""
    return _extract_with_openrouter(image_path, OPENROUTER_NEMOTRON_MODEL)