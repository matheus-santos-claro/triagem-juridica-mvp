import os
import json
import gspread
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_google_client():
    credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not credentials_json:
        return None

    service_account_info = json.loads(credentials_json)

    credentials = Credentials.from_service_account_info(
        service_account_info,
        scopes=SCOPES
    )

    return gspread.authorize(credentials)


def save_interaction_to_google_sheets(payload: dict):
    try:
        sheet_id = os.getenv("GOOGLE_SHEET_ID")

        if not sheet_id:
            print("GOOGLE_SHEET_ID não configurado.")
            return

        client = get_google_client()

        if client is None:
            print("Google client não inicializado.")
            return

        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet("raw_logs")

        worksheet.append_row(
            [
                payload.get("timestamp", ""),
                payload.get("source", ""),
                payload.get("user_input", ""),
                payload.get("assistant_output", ""),
                payload.get("status", ""),
                payload.get("score", ""),
                payload.get("next_question", ""),
            ],
            value_input_option="USER_ENTERED"
        )

    except Exception as e:
        print(f"Erro ao salvar no Google Sheets: {type(e).__name__}: {e}")
        return