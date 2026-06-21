import json
import pdfplumber
from openai import OpenAI


def extract_text_from_pdf(uploaded_file) -> str:
    text_parts = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text_parts.append(text)

    return "\n\n".join(text_parts)


def analyze_cnis_with_llm(cnis_text: str) -> dict:
    client = OpenAI()

    prompt = f"""
Você é um assistente jurídico previdenciário especializado em leitura preliminar de CNIS.

Analise o texto extraído de um CNIS e retorne APENAS JSON válido.

Objetivo:
Extrair um resumo simples para uma demo de triagem jurídica.

Campos esperados:
{{
  "nome": "",
  "cpf": "",
  "data_nascimento": "",
  "nit": "",
  "vinculos": [
    {{
      "origem": "",
      "tipo": "",
      "data_inicio": "",
      "data_fim": "",
      "ultima_remuneracao": "",
      "indicadores": []
    }}
  ],
  "beneficios": [
    {{
      "especie": "",
      "data_inicio": "",
      "data_fim": "",
      "situacao": ""
    }}
  ],
  "indicadores_encontrados": [],
  "competencia_mais_antiga": "",
  "competencia_mais_recente": "",
  "ultima_remuneracao_identificada": "",
  "resumo_cnis": "",
  "pontos_atencao": [],
  "confianca_extracao": "baixa|media|alta"
}}

Regras:
- Não invente dados.
- Se não encontrar algum campo, use string vazia ou lista vazia.
- Não calcule valor da causa.
- Não faça conclusão jurídica definitiva.
- O resumo deve ser objetivo, em linguagem simples.

Texto do CNIS:
\"\"\"
{cnis_text[:50000]}
\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Responda somente com JSON válido, sem markdown."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except Exception:
        return {
            "erro": "Falha ao interpretar JSON retornado pelo LLM.",
            "raw_response": raw
        }


def build_cnis_summary_card(cnis_analysis: dict) -> dict:
    if not cnis_analysis:
        return {}

    return {
        "nome": cnis_analysis.get("nome", ""),
        "cpf": cnis_analysis.get("cpf", ""),
        "data_nascimento": cnis_analysis.get("data_nascimento", ""),
        "total_vinculos": len(cnis_analysis.get("vinculos", [])),
        "total_beneficios": len(cnis_analysis.get("beneficios", [])),
        "competencia_mais_antiga": cnis_analysis.get("competencia_mais_antiga", ""),
        "competencia_mais_recente": cnis_analysis.get("competencia_mais_recente", ""),
        "ultima_remuneracao_identificada": cnis_analysis.get("ultima_remuneracao_identificada", ""),
        "resumo_cnis": cnis_analysis.get("resumo_cnis", ""),
        "pontos_atencao": cnis_analysis.get("pontos_atencao", []),
        "confianca_extracao": cnis_analysis.get("confianca_extracao", ""),
    }