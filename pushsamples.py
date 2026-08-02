# push_samples.py
#
# Pushes a sample of extracted bills into Zoho Books as real expense entries
# — demonstrating the pipeline's output is actually usable downstream, not
# just accurate on paper. Uses Gemini 3.5 Flash-Lite specifically because
# it's the more accurate model per this project's own evaluation
# (results.md) — this script demonstrates the pipeline working end-to-end
# with the more trustworthy extraction, it doesn't re-run the comparison.
#
# Usage:
#   python push_samples.py                # first 5 images in data/images/, sorted
#   python push_samples.py --count 3      # push fewer/more than 5
#   python push_samples.py --dry-run      # extract + print, skip the Zoho POST

import os
import glob
import argparse

from extractor import extract_with_gemini
from zoho_client import ZohoBooksClient

IMAGES_DIR = "data/images"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def get_sample_images(count: int) -> list[str]:
    """Sorted rather than left in glob/OS order, so 'first N' is
    deterministic across machines and reruns — bill_01.jpg, bill_02.jpg, ...
    rather than whatever order the filesystem happens to return them in."""
    all_images = sorted(
        f for f in glob.glob(os.path.join(IMAGES_DIR, "*"))
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )
    return all_images[:count]


def push_samples(count: int = 5, dry_run: bool = False):
    images = get_sample_images(count)
    if not images:
        print(f"No images found in {IMAGES_DIR}/")
        return

    mode = "(dry run — no Zoho writes)" if dry_run else ""
    print(f"Pushing {len(images)} sample bill(s) to Zoho Books using Gemini 3.5 Flash-Lite {mode}\n")

    client = None if dry_run else ZohoBooksClient()
    succeeded, failed = [], []

    for img_path in images:
        img_name = os.path.basename(img_path)
        print(f"--- {img_name} ---")
        try:
            result = extract_with_gemini(img_path)
            receipt = result.data
            print(
                f"  Extracted: {receipt.vendor_name!r} | {receipt.date} | "
                f"{receipt.currency} {receipt.total_amount} | "
                f"{receipt.description or '(no description)'}"
            )

            if dry_run:
                print("  [dry-run] Skipped Zoho push.")
                succeeded.append(img_name)
                continue

            resp = client.create_expense(receipt)
            expense_id = resp.get("expense", {}).get("expense_id", "N/A")
            print(f"  [OK] Created in Zoho Books — expense_id: {expense_id}")
            succeeded.append(img_name)

        except Exception as e:
            print(f"  [FAILED] {e}")
            failed.append(img_name)
        print()

    print("=" * 50)
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a sample of extracted bills to Zoho Books as expenses.")
    parser.add_argument("--count", type=int, default=5, help="Number of bills to push (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Extract and print, but skip creating Zoho expenses")
    args = parser.parse_args()

    push_samples(count=args.count, dry_run=args.dry_run)