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

    def get_access_token(self) -> str:
        """Exchanges refresh token for a fresh 1-hour access token."""
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

    def create_expense(self, receipt: ReceiptData) -> dict:
        """Posts structured receipt data to Zoho Books as a new expense."""
        access_token = self.get_access_token()
        url = f"{self.api_base_url}/expenses"
        
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "organization_id": self.organization_id
        }

        # Map ReceiptData schema to Zoho Expense API format
        payload = {
            "account_id": self.expense_account_id,
            "date": receipt.date,
            "amount": receipt.total_amount,
            "vendor_name": receipt.vendor_name,
            "reference_number": receipt.bill_number or "OCR-AUTO",
            "description": f"Auto-ingested handwritten bill. Line items: {len(receipt.line_items)}"
        }

        response = requests.post(url, headers=headers, params=params, json=payload)
        
        if response.status_code in (200, 201):
            return response.json()
        else:
            raise RuntimeError(f"Zoho API Error ({response.status_code}): {response.text}")