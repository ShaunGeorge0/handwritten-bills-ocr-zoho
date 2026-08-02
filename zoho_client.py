# zoho_client.py
import os
import requests
from dotenv import load_dotenv
from schemas import ReceiptData

load_dotenv()


class ZohoBooksClient:
    def __init__(self):
        self.accounts_url = os.getenv("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.in")
        self.api_base_url = os.getenv("ZOHO_API_BASE_URL", "https://www.zohoapis.in/books/v3")
        self.client_id = os.getenv("ZOHO_CLIENT_ID")
        self.client_secret = os.getenv("ZOHO_CLIENT_SECRET")
        self.refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")
        self.organization_id = os.getenv("ZOHO_ORGANIZATION_ID")
        self.expense_account_id = os.getenv("ZOHO_EXPENSE_ACCOUNT_ID")
        # CHANGE: this was missing entirely. Zoho Books' Create Expense API
        # requires paid_through_account_id (the cash/bank account the money
        # left from) in addition to account_id (the expense category it's
        # booked against) — without it the API returns a validation error.
        self.paid_through_account_id = os.getenv("ZOHO_PAID_THROUGH_ACCOUNT_ID")
        self._vendor_cache: dict[str, str] = {}

    def get_access_token(self) -> str:
        """Exchanges permanent refresh token for a live 1-hour access token."""
        url = f"{self.accounts_url}/oauth/v2/token"
        params = {
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token"
        }
        response = requests.post(url, params=params)
        if response.status_code == 200 and "access_token" in response.json():
            return response.json()["access_token"]
        else:
            raise RuntimeError(f"Failed to fetch Zoho access token: {response.text}")

    # --- CHANGE -------------------------------------------------------
    # The original code sent `"vendor_name": receipt.vendor_name` directly
    # in the expense payload. Zoho Books doesn't accept a free-text vendor
    # name on write — it wants `vendor_id`, the ID of an existing Contact
    # record (contact_type=vendor). A field it doesn't recognize is just
    # silently dropped, so the original code would "work" but every expense
    # would land with no vendor attached.
    #
    # This looks up a vendor contact by name and creates one if it doesn't
    # exist yet, so the expense is actually attributed to the right vendor.
    # If this lookup/create fails for any reason, we fall back to leaving
    # vendor_id off and putting the name in the description instead, so a
    # flaky contacts call never blocks the expense from being created.
    # --------------------------------------------------------------------
    def get_or_create_vendor_id(self, access_token: str, vendor_name: str) -> str | None:
        if vendor_name in self._vendor_cache:
            return self._vendor_cache[vendor_name]

        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        params = {"organization_id": self.organization_id, "contact_name": vendor_name}

        try:
            search = requests.get(f"{self.api_base_url}/contacts", headers=headers, params=params)
            contacts = search.json().get("contacts", []) if search.status_code == 200 else []
            for c in contacts:
                if c.get("contact_name", "").strip().lower() == vendor_name.strip().lower():
                    self._vendor_cache[vendor_name] = c["contact_id"]
                    return c["contact_id"]

            create = requests.post(
                f"{self.api_base_url}/contacts",
                headers={**headers, "Content-Type": "application/json"},
                params={"organization_id": self.organization_id},
                json={"contact_name": vendor_name, "contact_type": "vendor"},
            )
            if create.status_code in (200, 201):
                contact_id = create.json()["contact"]["contact_id"]
                self._vendor_cache[vendor_name] = contact_id
                return contact_id
        except (requests.RequestException, KeyError, ValueError):
            pass

        return None  # caller falls back to putting the name in the description

    def create_expense(self, receipt: ReceiptData) -> dict:
        """Posts structured receipt data to Zoho Books as a new expense entry."""
        access_token = self.get_access_token()
        url = f"{self.api_base_url}/expenses"

        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Content-Type": "application/json"
        }

        params = {
            "organization_id": self.organization_id
        }

        vendor_id = self.get_or_create_vendor_id(access_token, receipt.vendor_name)

        # CHANGE: receipt.description (schemas.py) now exists — use it when the
        # model produced one instead of a generic fallback string. Ground truth
        # also moved from tax_amount to gst_number; there's no native GSTIN
        # field on Zoho Books' expense payload, so it's appended to the
        # description text instead, same as the vendor-name fallback below.
        bill_summary = receipt.description or "Auto-ingested handwritten bill"
        description = f"{bill_summary} ({receipt.vendor_name}). Line items: {len(receipt.line_items)}"
        if receipt.gst_number:
            description += f" | GSTIN: {receipt.gst_number}"
        if not vendor_id:
            description = f"[Vendor: {receipt.vendor_name}] " + description

        payload = {
            "account_id": self.expense_account_id,
            "paid_through_account_id": self.paid_through_account_id,  # CHANGE: now included, required
            "date": receipt.date,
            "amount": receipt.total_amount,
            "reference_number": receipt.bill_number or "OCR-AUTO",
            "description": description,
            "currency_id": None,  # optional: set if you post in a non-base currency
        }
        if vendor_id:
            payload["vendor_id"] = vendor_id  # CHANGE: correct field name (was vendor_name)
        payload = {k: v for k, v in payload.items() if v is not None}

        response = requests.post(url, headers=headers, params=params, json=payload)

        if response.status_code in (200, 201):
            return response.json()
        else:
            raise RuntimeError(f"Zoho API Error ({response.status_code}): {response.text}")