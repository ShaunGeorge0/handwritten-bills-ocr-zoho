# zoho_client.py
import os
import re
import requests
from dotenv import load_dotenv
from schemas import ReceiptData

load_dotenv()


def _get_config(key: str, default: str | None = None) -> str | None:
    """Reads a config value from, in order: OS env vars (.env locally),
    then Streamlit secrets (st.secrets — used on Streamlit Community Cloud,
    where .env is never present because it's gitignored). Whitespace is
    stripped either way, since a stray trailing space/newline pasted into
    .env or secrets.toml turns a valid client_id/secret into an invalid one
    and Zoho reports that back as the generic "invalid_client" error.

    CHANGE: previously this was a bare os.getenv() call inline in __init__.
    On a local run with a properly filled-in .env that's fine, but on
    Streamlit Cloud there is no .env file (it's excluded via .gitignore),
    so every one of these returned None and the OAuth request was sent
    with client_id="None" / client_secret="None" — which Zoho correctly
    rejects as invalid_client. This fallback makes the same code work in
    both environments without needing two separate config paths.
    """
    val = os.getenv(key)
    if val is None:
        try:
            import streamlit as st
            val = st.secrets.get(key)
        except Exception:
            val = None
    return val.strip() if isinstance(val, str) else (val if val is not None else default)


# CHANGE: Keyword rules used to route each receipt to the right Zoho expense
# category instead of a single hardcoded account. Matched (word-boundary,
# case-insensitive) against `description` and `vendor_name` together — the
# two free-text fields most likely to carry a signal about what was bought.
# Checked in order; the first category with a hit wins, so if a bill could
# plausibly match more than one, whichever list it's checked against first
# takes priority. Anything with no match falls back to "other_expenses".
CATEGORY_KEYWORDS = {
    "printing": [
        "print", "printing", "printer", "photocopy", "xerox", "stationery",
        "stationary", "cartridge", "toner", "binding",
    ],
    "travel": [
        "travel", "travels", "transport", "transportation", "bus", "taxi",
        "cab", "auto", "rickshaw", "fuel", "petrol", "diesel", "fare",
        "toll", "parking", "train", "flight", "airfare", "car hire",
        "hire charges",
    ],
    "meal_entertainment": [
        "food", "restaurant", "meal", "lunch", "dinner", "breakfast",
        "snack", "grocery", "groceries", "catering", "cafe", "café",
        "coffee", "tea", "beverage", "hotel", "entertainment", "movie",
        "bakery", "sweets", "sweet",
    ],
    # CHANGE: new category for physical inputs bought to make or sell
    # something (as opposed to office supplies like paper/toner, which stay
    # under "printing"). "paper" was dropped from printing's keyword list
    # and left unassigned here on purpose — bare "paper" is too ambiguous
    # between printer paper and a raw-material paper stock to route safely.
    "rawmaterials_consumables": [
        "flower", "flowers", "bouquet", "design", "designs", "cement",
        "material", "materials", "raw material", "consumable",
        "consumables", "wire", "supplies", "ingredient", "ingredients",
        "fabric", "cloth", "yarn", "timber", "wood", "paint", "steel",
        "iron rod",
    ],
}


class ZohoBooksClient:
    def __init__(self):
        self.accounts_url = _get_config("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.in")
        self.api_base_url = _get_config("ZOHO_API_BASE_URL", "https://www.zohoapis.in/books/v3")
        self.client_id = _get_config("ZOHO_CLIENT_ID")
        self.client_secret = _get_config("ZOHO_CLIENT_SECRET")
        self.refresh_token = _get_config("ZOHO_REFRESH_TOKEN")
        self.organization_id = _get_config("ZOHO_ORGANIZATION_ID")
        # CHANGE: expenses used to all go to a single hardcoded account.
        # Now each category has its own account_id, with ZOHO_ACCOUNT_OTHER_EXPENSES
        # as the fallback for anything that doesn't match a keyword rule below.
        # ZOHO_EXPENSE_ACCOUNT_ID is kept as a last-resort default in case the
        # other_expenses account isn't configured either, so a missing .env
        # entry fails loudly at the Zoho API rather than crashing here.
        self.expense_account_id = _get_config("ZOHO_EXPENSE_ACCOUNT_ID")
        self.category_account_ids = {
            "printing": _get_config("ZOHO_ACCOUNT_PRINTING"),
            "travel": _get_config("ZOHO_ACCOUNT_TRAVEL"),
            # NOTE: your .env uses ZOHO_RAWMATERIALS_CONSUMABLES (no "ACCOUNT"
            # in the name, unlike the others) — matched here exactly as you
            # have it. Rename it if you'd rather keep the naming consistent.
            "rawmaterials_consumables": _get_config("ZOHO_RAWMATERIALS_CONSUMABLES"),
            "meal_entertainment": _get_config("ZOHO_ACCOUNT_MEAL_ENTERTAINMENT"),
            "other_expenses": _get_config("ZOHO_ACCOUNT_OTHER_EXPENSES"),
        }
        # CHANGE: this was missing entirely. Zoho Books' Create Expense API
        # requires paid_through_account_id (the cash/bank account the money
        # left from) in addition to account_id (the expense category it's
        # booked against) — without it the API returns a validation error.
        self.paid_through_account_id = _get_config("ZOHO_PAID_THROUGH_ACCOUNT_ID")
        self._vendor_cache: dict[str, str] = {}

        # CHANGE: fail fast with a clear, actionable message instead of
        # letting a missing credential travel all the way to Zoho and come
        # back as an opaque "invalid_client". This is the single most common
        # cause of that error — on Streamlit Community Cloud in particular,
        # .env is never deployed (it's gitignored), so without this check
        # the app would call Zoho with client_id/client_secret literally
        # set to None and silently degrade into a confusing OAuth error.
        missing = [
            name for name, val in [
                ("ZOHO_CLIENT_ID", self.client_id),
                ("ZOHO_CLIENT_SECRET", self.client_secret),
                ("ZOHO_REFRESH_TOKEN", self.refresh_token),
                ("ZOHO_ORGANIZATION_ID", self.organization_id),
            ] if not val
        ]
        if missing:
            raise RuntimeError(
                "Missing Zoho credential(s): " + ", ".join(missing) + ". "
                "Set these in a local .env file (see .env.example), or, if this "
                "is running on Streamlit Community Cloud, add them under "
                "App settings -> Secrets (they are NOT read from .env there, "
                "since .env is gitignored and never deployed)."
            )

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
        data = response.json() if response.content else {}
        if response.status_code == 200 and "access_token" in data:
            return data["access_token"]

        error = data.get("error", response.text)
        if error == "invalid_client":
            raise RuntimeError(
                "Failed to fetch Zoho access token: invalid_client. This means Zoho "
                "rejected your ZOHO_CLIENT_ID / ZOHO_CLIENT_SECRET pair. Check that: "
                "(1) both values are copied exactly from the Zoho API Console with no "
                "extra spaces/newlines, (2) the refresh token in ZOHO_REFRESH_TOKEN was "
                f"generated using this SAME client_id/secret pair, and (3) ZOHO_ACCOUNTS_URL "
                f"(currently '{self.accounts_url}') matches the data center your Zoho "
                "client was registered in (accounts.zoho.in for India, accounts.zoho.com "
                "for the US/global, accounts.zoho.eu for Europe, etc.) — a mismatch here "
                "also surfaces as invalid_client."
            )
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

    def categorize_expense(self, receipt: ReceiptData) -> str:
        """Picks the Zoho expense account_id for a receipt based on keyword
        matches in its description and vendor name (checked in this order:
        printing, travel, meal & entertainment, raw materials & consumables).
        Falls back to ZOHO_ACCOUNT_OTHER_EXPENSES (or, if that's unset, the
        legacy single ZOHO_EXPENSE_ACCOUNT_ID) when nothing matches."""
        haystack = f"{receipt.description or ''} {receipt.vendor_name or ''}".lower()

        for category in ("printing", "travel", "meal_entertainment", "rawmaterials_consumables"):
            keywords = CATEGORY_KEYWORDS[category]
            if any(re.search(rf"\b{re.escape(kw)}\b", haystack) for kw in keywords):
                account_id = self.category_account_ids.get(category)
                if account_id:
                    return account_id
                # Keyword matched but that category's account_id isn't
                # configured in .env — fall through to other_expenses
                break

        return self.category_account_ids.get("other_expenses") or self.expense_account_id

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
            "account_id": self.categorize_expense(receipt),  # CHANGE: was self.expense_account_id (single category)
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