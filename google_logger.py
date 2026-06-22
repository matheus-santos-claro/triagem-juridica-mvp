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

def save_lead_summary_to_google_sheets(payload: dict):
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

        try:
            worksheet = spreadsheet.worksheet("lead_summary")
        except Exception:
            worksheet = spreadsheet.add_worksheet(
                title="lead_summary",
                rows=1000,
                cols=30
            )

            worksheet.append_row(
                [
                    "timestamp",
                    "conversation_id",
                    "nome",
                    "telefone",
                    "status",
                    "score",
                    "natureza_caso",
                    "nexo_trabalho",
                    "tipo_vinculo",
                    "data_evento",
                    "cirurgia",
                    "sequela",
                    "reducao_capacidade",
                    "resumo_final",
                    "cnis_nome",
                    "cnis_cpf",
                    "cnis_vinculos",
                    "cnis_beneficios",
                    "cnis_resumo",
                    "cnis_pontos_atencao",
                ],
                value_input_option="USER_ENTERED"
            )

        worksheet.append_row(
            [
                payload.get("timestamp", ""),
                payload.get("conversation_id", ""),
                payload.get("nome", ""),
                payload.get("telefone", ""),
                payload.get("status", ""),
                payload.get("score", ""),
                payload.get("natureza_caso", ""),
                payload.get("nexo_trabalho", ""),
                payload.get("tipo_vinculo", ""),
                payload.get("data_evento", ""),
                payload.get("cirurgia", ""),
                payload.get("sequela", ""),
                payload.get("reducao_capacidade", ""),
                payload.get("resumo_final", ""),
                payload.get("cnis_nome", ""),
                payload.get("cnis_cpf", ""),
                payload.get("cnis_vinculos", ""),
                payload.get("cnis_beneficios", ""),
                payload.get("cnis_resumo", ""),
                payload.get("cnis_pontos_atencao", ""),
            ],
            value_input_option="USER_ENTERED"
        )

    except Exception as e:
        print(f"Erro ao salvar lead_summary no Google Sheets: {type(e).__name__}: {e}")
        return