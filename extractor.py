# extractor.py
import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

# SDK Imports
from google import genai
from google.genai import types
import openai
import anthropic

from schemas import ReceiptData

load_dotenv()

PROMPT_TEXT = """
Analyze this receipt/bill image carefully and extract all information strictly according to the requested JSON schema.
- Convert all dates to YYYY-MM-DD format.
- Ensure total_amount correctly reflects the final payable sum.
- If individual line items are visible, capture them in line_items. Otherwise, capture the overall expense.
"""

def encode_image(image_path: str) -> tuple[str, str]:
    """Helper to convert local image to base64 and identify mime type."""
    mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif image_path.lower().endswith(".webp"):
        mime_type = "image/webp"

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    return encoded_string, mime_type


def extract_with_gemini(image_path: str) -> ReceiptData:
    """Extract receipt data using Gemini 1.5 Flash."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    encoded_bytes = Path(image_path).read_bytes()
    mime_type, _ = encode_image(image_path)
    
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            types.Part.from_bytes(data=encoded_bytes, mime_type=mime_type),
            PROMPT_TEXT
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ReceiptData,
            temperature=0.1
        )
    )
    
    parsed_json = json.loads(response.text)
    return ReceiptData(**parsed_json)


def extract_with_openai(image_path: str) -> ReceiptData:
    """Extract receipt data using OpenAI gpt-4o-mini."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    b64_image, mime_type = encode_image(image_path)
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
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
        temperature=0.1
    )
    
    return response.choices[0].message.parsed


def extract_with_claude(image_path: str) -> ReceiptData:
    """Extract receipt data using Anthropic Claude 3.5 Sonnet."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    b64_image, mime_type = encode_image(image_path)
    
    tool_schema = {
        "name": "record_receipt",
        "description": "Save structured receipt data",
        "input_schema": ReceiptData.model_json_schema()
    }

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
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

    for content in response.content:
        if content.type == "tool_use" and content.name == "record_receipt":
            return ReceiptData(**content.input)

    raise RuntimeError("Claude failed to produce structured output.")