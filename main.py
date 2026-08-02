# main.py
import os
import argparse
from extractor import extract_with_gemini, extract_with_openai, extract_with_claude
from zoho_client import ZohoBooksClient


def process_bill(image_path: str, model_provider: str = "gemini", post_to_zoho: bool = True):
    print(f"\n=== Processing Image: {image_path} using [{model_provider.upper()}] ===")

    # Select LLM Extractor
    if model_provider.lower() == "gemini":
        result = extract_with_gemini(image_path)
    elif model_provider.lower() == "openai":
        result = extract_with_openai(image_path)
    elif model_provider.lower() == "claude":
        result = extract_with_claude(image_path)
    else:
        raise ValueError("Unsupported provider. Choose 'gemini', 'openai', or 'claude'.")

    # CHANGE: extract_with_* now return an ExtractionResult (data + token
    # usage + latency) instead of a bare ReceiptData, so the pipeline can
    # report cost — pull the receipt out of it.
    receipt = result.data

    print("\n--- Extracted Receipt Data ---")
    print(receipt.model_dump_json(indent=2))
    print(f"\n[latency: {result.latency_s:.2f}s | tokens: {result.input_tokens} in / {result.output_tokens} out]")

    # Sync with Zoho Books
    if post_to_zoho:
        print("\n--- Syncing Expense with Zoho Books ---")
        client = ZohoBooksClient()
        result_zoho = client.create_expense(receipt)
        expense_id = result_zoho.get("expense", {}).get("expense_id", "N/A")
        print(f" [✔] Expense Created Successfully in Zoho! Expense ID: {expense_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten Bill OCR & Zoho Books Pipeline")
    parser.add_argument("--image", type=str, default="data/images/bill_01.jpg", help="Path to receipt image")
    parser.add_argument("--model", type=str, default="gemini", choices=["gemini", "openai", "claude"], help="Vision model provider")
    parser.add_argument("--no-zoho", action="store_true", help="Skip posting to Zoho Books")

    args = parser.parse_args()

    if os.path.exists(args.image):
        process_bill(args.image, model_provider=args.model, post_to_zoho=not args.no_zoho)
    else:
        print(f" [✘] Error: Image file '{args.image}' not found.")