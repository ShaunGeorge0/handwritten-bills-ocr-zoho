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
import openai
import anthropic

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
# Model IDs, current as of Aug 2026. Verify against each provider's live
# model list before you run the eval — this is the part that goes stale
# fastest:
#   Gemini:  https://ai.google.dev/gemini-api/docs/models
#   OpenAI:  https://platform.openai.com/docs/models
#   Claude:  https://docs.claude.com/en/docs/about-claude/models
#
# gemini-1.5-flash and gpt-4o-mini and claude-3-5-sonnet-20241022 (the
# original picks) are all retired/404 as of this writing.
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.6-flash"          # ~$1.50 / $7.50 per 1M tokens (in/out), current as of Aug 2026        # ~$0.15 / $1.25 per 1M tokens (in/out)
OPENAI_MODEL = "gpt-5-mini"                # ~$0.25 / $2.00 per 1M tokens, vision-capable
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # ~$1.00 / $5.00 per 1M tokens
# NOTE: if you swap in claude-sonnet-5 or claude-opus-4-8 for a higher-tier
# comparison, drop the `temperature=0.1` argument below — those models reject
# non-default temperature/top_p/top_k with a 400 error. Haiku 4.5 still accepts it.


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
    """Extract receipt data using Gemini 2.5 Flash."""
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


def extract_with_openai(image_path: str) -> ExtractionResult:
    """Extract receipt data using OpenAI structured outputs."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    b64_image, mime_type = encode_image(image_path)

    start = time.time()
    # CHANGE: client.beta.chat.completions.parse -> client.chat.completions.parse
    # (the .parse() helper has been promoted out of .beta in current SDK versions)
    response = client.chat.completions.parse(
        model=OPENAI_MODEL,
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
        response_format=ReceiptData,
    )
    latency = time.time() - start

    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    return ExtractionResult(response.choices[0].message.parsed, input_tokens, output_tokens, latency, OPENAI_MODEL)


def extract_with_claude(image_path: str) -> ExtractionResult:
    """Extract receipt data using Claude tool calling."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    b64_image, mime_type = encode_image(image_path)

    tool_schema = {
        "name": "record_receipt",
        "description": "Save structured receipt data",
        "input_schema": ReceiptData.model_json_schema()
    }

    start = time.time()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        temperature=0.1,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "record_receipt"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64_image
                        }
                    },
                    {"type": "text", "text": PROMPT_TEXT}
                ]
            }
        ]
    )
    latency = time.time() - start

    input_tokens = getattr(response.usage, "input_tokens", 0) or 0
    output_tokens = getattr(response.usage, "output_tokens", 0) or 0

    for content in response.content:
        if content.type == "tool_use" and content.name == "record_receipt":
            receipt = ReceiptData(**content.input)
            return ExtractionResult(receipt, input_tokens, output_tokens, latency, CLAUDE_MODEL)

    raise RuntimeError("Claude failed to produce structured output.")