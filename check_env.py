import os
import requests
from dotenv import load_dotenv

load_dotenv()

def check_all():
    print("=== Checking Configured Environment Variables ===")
    keys = [
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "ZOHO_CLIENT_ID",
        "ZOHO_CLIENT_SECRET",
        "ZOHO_REFRESH_TOKEN",
        "ZOHO_ORGANIZATION_ID",
        "ZOHO_EXPENSE_ACCOUNT_ID"
    ]
    
    missing = False
    for k in keys:
        val = os.getenv(k)
        if val:
            print(f" [✔] {k}: Configured")
        else:
            print(f" [✘] {k}: MISSING")
            missing = True

    if missing:
        print("\n[!] Please fill in missing values in your .env file.")
        return

    print("\n=== Testing Zoho Access Token Generation ===")
    try:
        url = f"{os.getenv('ZOHO_ACCOUNTS_URL')}/oauth/v2/token"
        params = {
            "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN"),
            "client_id": os.getenv("ZOHO_CLIENT_ID"),
            "client_secret": os.getenv("ZOHO_CLIENT_SECRET"),
            "grant_type": "refresh_token"
        }
        res = requests.post(url, params=params)
        if res.status_code == 200 and "access_token" in res.json():
            print(" [✔] Zoho OAuth Refresh Token works! Access token generated successfully.")
        else:
            print(f" [✘] Zoho OAuth Error: {res.text}")
    except Exception as e:
        print(f" [✘] Connection Error: {e}")

if __name__ == "__main__":
    check_all()