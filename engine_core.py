# - imports

import os
import re
import json
import uuid
import unicodedata

from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional


MODEL_NAME = "gpt-4.1-mini"

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60

PLANNER_WEIGHTS = {
    "required": 100,
    "needs_confirmation": 80,
    "inconsistent": 90,
    "activation_rule": 30,
    "unanswered": 10,
    "asked_penalty": 20,
}

CLASSIFICATION_THRESHOLDS = {
    "quente": 80,
    "morno": 50,
    "frio": 0,
}

# - dataclasses

@dataclass
class SessionState:
    conversation_id: str
    lead_id: Optional[str]
    created_at: str
    updated_at: str
    channel_source: str
    domain: str
    current_phase: str


@dataclass
class ItemState:
    item_id: str
    slug: str

    value_raw: Any = None
    value_normalized: Any = None

    status: str = "unanswered"
    confidence: float = 0.0
    source: str = "system"

    evidence: List[str] = field(default_factory=list)

    answered_at: Optional[str] = None
    updated_at: Optional[str] = None

    asked_count: int = 0
    last_question_type: Optional[str] = None

    needs_confirmation: bool = False
    conflict_with: List[str] = field(default_factory=list)

    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DerivedState:
    active_items: List[str] = field(default_factory=list)
    pending_items: List[str] = field(default_factory=list)
    confirmation_queue: List[str] = field(default_factory=list)
    inconsistent_items: List[str] = field(default_factory=list)
    dispensed_items: List[str] = field(default_factory=list)
    completion_ratio: float = 0.0
    critical_missing_items: List[str] = field(default_factory=list)


@dataclass
class ClassificationState:
    score_total: int = 0
    commercial_status: str = "frio"
    disqualification_reason: Optional[str] = None
    rule_hits: List[Dict[str, Any]] = field(default_factory=list)
    justification_bullets: List[str] = field(default_factory=list)
    recommended_next_steps: List[str] = field(default_factory=list)


@dataclass
class PlannerState:
    next_item_slug: Optional[str] = None
    next_question_type: Optional[str] = None
    next_question_text: Optional[str] = None
    planning_reason: Optional[str] = None
    candidate_items_ranked: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AuditState:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    extraction_events: List[Dict[str, Any]] = field(default_factory=list)
    validation_events: List[Dict[str, Any]] = field(default_factory=list)
    rule_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    planning_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationState:
    session: SessionState
    items: Dict[str, ItemState]
    derived: DerivedState
    classification: ClassificationState
    planner: PlannerState
    audit: AuditState

@dataclass
class MessageContext:
    message_id: str
    text: str
    source: str
    timestamp: str


@dataclass
class UpdateProposal:
    item_id: str
    slug: str

    value_raw: Any
    value_normalized: Any

    extraction_mode: str
    confidence: float

    evidence: str

    completeness: str
    validation_hint: str
    conflicts_with_current: bool
    suggested_action: str

    is_explicit_correction: bool = False


@dataclass
class GlobalSignals:
    possible_disqualification: bool = False
    urgency_signal: bool = False
    out_of_scope_signal: bool = False
    multiple_items_answered: bool = False
    contradiction_detected: bool = False


@dataclass
class ExtractionResult:
    message_context: MessageContext
    proposed_updates: List[UpdateProposal]
    global_signals: GlobalSignals
    extraction_notes: List[str]

# - utilitários

def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def to_dict(obj: Any) -> dict:
    return asdict(obj)


def pretty(obj: Any):
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)

    print(json.dumps(obj, indent=4, ensure_ascii=False))


def normalize_value(value: Any) -> Any:
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    text = value.strip().lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    text = text.replace("-", "_")
    text = text.replace(" ", "_")

    return text


def safe_str(value: Any) -> str:
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def build_initial_state(
    schema_df: pd.DataFrame,
    channel_source: str = "chat",
    domain: str = "acidente_trabalho"
) -> ConversationState:

    items = {}

    for _, row in schema_df.iterrows():
        slug = safe_str(row["slug"])

        items[slug] = ItemState(
            item_id=safe_str(row["item_id"]),
            slug=slug
        )

    timestamp = now_utc()

    return ConversationState(
        session=SessionState(
            conversation_id=str(uuid.uuid4()),
            lead_id=None,
            created_at=timestamp,
            updated_at=timestamp,
            channel_source=channel_source,
            domain=domain,
            current_phase="triagem"
        ),
        items=items,
        derived=DerivedState(),
        classification=ClassificationState(),
        planner=PlannerState(),
        audit=AuditState()
    )

## ValidationRegistry

class ValidationRegistry:

    def __init__(self):
        self.validators = {}

        self.register("texto_nao_vazio", self.texto_nao_vazio)
        self.register("valor_em_lista", self.valor_em_lista)
        self.register("booleano_sim_nao", self.booleano_sim_nao)
        self.register("data_nao_futura", self.data_nao_futura)

    def register(self, name: str, func):
        self.validators[name] = func

    def execute(self, name: str, value: Any, **kwargs) -> bool:
        name = safe_str(name)

        if not name:
            return True

        validator = self.validators.get(name)

        if validator is None:
            return True

        return validator(value, **kwargs)

    def texto_nao_vazio(self, value: Any, **kwargs) -> bool:
        return safe_str(value) != ""

    def valor_em_lista(
        self,
        value: Any,
        valores_permitidos: Optional[List[str]] = None,
        **kwargs
    ) -> bool:

        if valores_permitidos is None:
            return True

        value_norm = normalize_value(value)

        allowed = [
            normalize_value(v)
            for v in valores_permitidos
        ]

        return value_norm in allowed

    def booleano_sim_nao(self, value: Any, **kwargs) -> bool:
        return normalize_value(value) in ["sim", "nao"]

    def data_nao_futura(self, value: Any, **kwargs) -> bool:
        try:
            parsed = pd.to_datetime(value, dayfirst=True)
            return parsed <= pd.Timestamp.now(tz="UTC")
        except Exception:
            return False


#validator = ValidationRegistry()

# - DSLContext

class DSLContext:

    def __init__(self, state: ConversationState):
        self.state = state

    def _get_item(self, slug: str) -> ItemState:
        slug = safe_str(slug).strip('"').strip("'")
        return self.state.items[slug]

    def ANSWERED(self, slug: str) -> bool:
        return self._get_item(slug).status not in ["unanswered", "empty"]

    def CONFIRMED(self, slug: str) -> bool:
        return self._get_item(slug).status == "confirmed"

    def PENDING_CONFIRMATION(self, slug: str) -> bool:
        return self._get_item(slug).needs_confirmation

    def EMPTY(self, slug: str) -> bool:
        item = self._get_item(slug)
        return item.value_normalized is None or item.status in ["unanswered", "empty"]

    def INCONSISTENT(self, slug: str) -> bool:
        return self._get_item(slug).status == "inconsistent"

    def VALUE(self, slug: str) -> Any:
        return self._get_item(slug).value_normalized

    def CONFIDENCE(self, slug: str) -> float:
        return self._get_item(slug).confidence

    @property
    def SCORE_TOTAL(self) -> int:
        return self.state.classification.score_total

# - DSLInterpreter

class DSLInterpreter:

    def evaluate(
        self,
        expression: Any,
        state: ConversationState
    ) -> bool:

        expression = safe_str(expression)

        if not expression:
            return True

        ctx = DSLContext(state)

        expr = expression

        expr = self._expand_short_field_comparisons(
            expr,
            state
        )

        expr = self._quote_function_args(expr)

        expr = self._normalize_logical_operators(expr)

        expr = self._normalize_in_operators(expr)

        expr = self._normalize_comparisons(expr)

        safe_globals = {
            "__builtins__": {}
        }

        safe_locals = {
            "ANSWERED": ctx.ANSWERED,
            "CONFIRMED": ctx.CONFIRMED,
            "PENDING_CONFIRMATION": ctx.PENDING_CONFIRMATION,
            "EMPTY": ctx.EMPTY,
            "INCONSISTENT": ctx.INCONSISTENT,
            "VALUE": ctx.VALUE,
            "CONFIDENCE": ctx.CONFIDENCE,
            "SCORE_TOTAL": ctx.SCORE_TOTAL,
        }

        return bool(
            eval(
                expr,
                safe_globals,
                safe_locals
            )
        )

    # ---------------------------------

    def _expand_short_field_comparisons(
        self,
        expr: str,
        state: ConversationState
    ) -> str:

        slugs = sorted(
            state.items.keys(),
            key=len,
            reverse=True
        )

        for slug in slugs:

            pattern_not_in = (
                rf'(?<![A-Za-z0-9_"\'])'
                rf'\b{re.escape(slug)}\b'
                rf'\s+NOT\s+IN\s+\[([^\]]+)\]'
            )

            expr = re.sub(
                pattern_not_in,
                lambda m: f'VALUE("{slug}") NOT IN [{m.group(1)}]',
                expr,
                flags=re.IGNORECASE
            )

            pattern_in = (
                rf'(?<![A-Za-z0-9_"\'])'
                rf'\b{re.escape(slug)}\b'
                rf'\s+IN\s+\[([^\]]+)\]'
            )

            expr = re.sub(
                pattern_in,
                lambda m: f'VALUE("{slug}") IN [{m.group(1)}]',
                expr,
                flags=re.IGNORECASE
            )

            pattern_compare = (
                rf'(?<![A-Za-z0-9_"\'])'
                rf'\b{re.escape(slug)}\b'
                rf'\s*(=|!=|>=|<=|>|<)\s*'
                rf'([A-Za-z0-9_]+)'
            )

            expr = re.sub(
                pattern_compare,
                lambda m: f'VALUE("{slug}") {m.group(1)} {m.group(2)}',
                expr
            )

        return expr

    # ---------------------------------

    def _quote_function_args(
        self,
        expr: str
    ) -> str:

        functions = [
            "ANSWERED",
            "CONFIRMED",
            "PENDING_CONFIRMATION",
            "EMPTY",
            "INCONSISTENT",
            "VALUE",
            "CONFIDENCE",
        ]

        for fn in functions:

            pattern = rf"{fn}\(([^)\"']+)\)"

            expr = re.sub(
                pattern,
                lambda m: f'{fn}("{safe_str(m.group(1))}")',
                expr,
                flags=re.IGNORECASE
            )

        return expr

    # ---------------------------------

    def _normalize_logical_operators(
        self,
        expr: str
    ) -> str:

        expr = re.sub(
            r"\bAND\b",
            "and",
            expr,
            flags=re.IGNORECASE
        )

        expr = re.sub(
            r"\bOR\b",
            "or",
            expr,
            flags=re.IGNORECASE
        )

        expr = re.sub(
            r"\bNOT\b",
            "not",
            expr,
            flags=re.IGNORECASE
        )

        return expr

    # ---------------------------------

    def _normalize_in_operators(
        self,
        expr: str
    ) -> str:

        expr = re.sub(
            r'(VALUE\("[^"]+"\))\s+not\s+in\s+\[([^\]]+)\]',
            lambda m: (
                f'{m.group(1)} not in '
                f'{[normalize_value(x.strip()) for x in m.group(2).split(",")]}'
            ),
            expr,
            flags=re.IGNORECASE
        )

        expr = re.sub(
            r'(VALUE\("[^"]+"\))\s+in\s+\[([^\]]+)\]',
            lambda m: (
                f'{m.group(1)} in '
                f'{[normalize_value(x.strip()) for x in m.group(2).split(",")]}'
            ),
            expr,
            flags=re.IGNORECASE
        )

        return expr

    # ---------------------------------

    def _normalize_comparisons(
        self,
        expr: str
    ) -> str:

        expr = re.sub(
            r'(?<![<>=!])=(?!=)',
            '==',
            expr
        )

        expr = re.sub(
            r'(VALUE\("[^"]+"\)\s*==\s*)([a-zA-Z_][a-zA-Z0-9_]*)',
            lambda m: f'{m.group(1)}"{normalize_value(m.group(2))}"',
            expr
        )

        expr = re.sub(
            r'(VALUE\("[^"]+"\)\s*!=\s*)([a-zA-Z_][a-zA-Z0-9_]*)',
            lambda m: f'{m.group(1)}"{normalize_value(m.group(2))}"',
            expr
        )

        return expr


#dsl = DSLInterpreter()

# - OpenAIExtractor

class OpenAIExtractor:

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def __init__(
        self,
        schema_df: pd.DataFrame,
        model_name: str = MODEL_NAME
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.schema_df = schema_df
        self.model_name = model_name
        self.schema_context = self._build_schema_context()

    def _build_schema_context(self) -> List[dict]:

        context = []

        for _, row in self.schema_df.iterrows():
            context.append({
                "item_id": safe_str(row.get("item_id")),
                "slug": safe_str(row.get("slug")),
                "label": safe_str(row.get("label")),
                "descricao_semantica": safe_str(row.get("descricao_semantica")),
                "grupo": safe_str(row.get("grupo")),
                "tipo_valor": safe_str(row.get("tipo_valor")),
                "multiplo": safe_str(row.get("multiplo")),
                "valores_permitidos": safe_str(row.get("valores_permitidos")),
                "aceita_inferencia": safe_str(row.get("aceita_inferencia")),
                "aceita_parcial": safe_str(row.get("aceita_parcial")),
                "aceita_nao_sabe": safe_str(row.get("aceita_nao_sabe")),
                "pergunta_principal": safe_str(row.get("pergunta_principal")),
            })

        return context

    def extract(
        self,
        message: str,
        state: ConversationState
    ) -> ExtractionResult:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self._build_system_prompt()
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(message, state)
                }
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content
        data = self._safe_json_loads(content)

        return self._parse_response(message, data)

    def _build_system_prompt(self) -> str:

        return f"""
Você é um extrator estruturado para triagem jurídica trabalhista/previdenciária.

Sua tarefa é analisar a mensagem do usuário e extrair TODOS os campos possíveis de acordo com o schema.

Você NÃO decide elegibilidade.
Você NÃO calcula score.
Você NÃO classifica o lead.
Você NÃO dá aconselhamento jurídico.
Você apenas transforma linguagem natural em propostas estruturadas.

POLÍTICA DE CONFIANÇA:
- confidence >= 0.85: informação forte.
- 0.60 <= confidence < 0.85: informação aproveitável, mas deve ser confirmada.
- confidence < 0.60: não aplicar automaticamente.

NORMALIZAÇÃO:
- valores categóricos devem ser retornados em snake_case, minúsculo, sem acento.
- exemplos:
  - "CLT", "carteira assinada", "registrado" -> "clt"
  - "Acidente de Trabalho" -> "acidente_trabalho"
  - "sim" -> "sim"
  - "não" -> "nao"

PADRÕES DE DOENÇA OCUPACIONAL:

Quando o usuário relatar sintomas que surgiram ou pioraram após esforço repetitivo, carga de peso, movimentos repetitivos, postura de trabalho, anos de trabalho pesado ou atividade laboral contínua, considere fortemente:

natureza_caso = "doenca_ocupacional"

Exemplos:
- "dor na coluna depois de anos carregando peso"
- "começou depois de muito esforço no trabalho"
- "trabalhei anos fazendo movimento repetitivo"
- "a dor apareceu por causa do serviço"
- "tenho lesão por esforço repetitivo"

Nesses casos:
- se houver relação clara com trabalho, extraia nexo_trabalho = "sim"
- se o usuário disser "não sei se foi do trabalho", extraia nexo_trabalho = "nao_sabe"
- se houver dor crônica, limitação, perda de força ou dificuldade para trabalhar, extraia houve_sequela = "sim"
- se houver limitação para exercer função, extraia reducao_capacidade = "sim"

IMPORTANTE:
- Não confunda atividade de risco com ocupação.
- "carregando peso" não é ocupação; é evidência de esforço laboral.
- Só preencha ocupacao_evento se o usuário disser função/cargo, como pedreiro, auxiliar, motorista, operador, enfermeiro etc.

EXEMPLO:
Mensagem:
"Trabalhei muitos anos carregando peso. Comecei a ter dor forte na coluna por causa do serviço. Fiquei com limitação para trabalhar. Tenho carteira assinada."

Extração esperada:
natureza_caso = "doenca_ocupacional"
nexo_trabalho = "sim"
tipo_vinculo = "clt"
houve_sequela = "sim"
reducao_capacidade = "sim"
descricao_caso = resumo natural do relato

CORREÇÃO EXPLÍCITA POSTERIOR:

Quando o usuário corrigir uma informação anterior de forma clara, marque:
is_explicit_correction = true

Exemplos de correção explícita:
- "na verdade..."
- "corrigindo..."
- "pensando melhor..."
- "não era..."
- "não foi..."
- "não teve relação..."
- "me enganei..."
- "era MEI, não CLT"
- "não aconteceu no trabalho"
- "não teve nada a ver com o trabalho"

Nesses casos:
- extraia o novo valor corrigido;
- use suggested_action = "apply_directly";
- use validation_hint = "valid";
- conflicts_with_current pode ser true;
- confidence deve ser alta se a correção for clara.

Exemplos:
Mensagem:
"Depois pensei melhor, não teve relação com o trabalho."

Retorno esperado:
slug = "nexo_trabalho"
value_normalized = "nao"
is_explicit_correction = true
suggested_action = "apply_directly"

Mensagem:
"Na verdade eu era MEI prestando serviço."

Retorno esperado:
slug = "tipo_vinculo"
value_normalized = "mei"
is_explicit_correction = true
suggested_action = "apply_directly"

CONFIRMAÇÕES CURTAS:

Quando o estado atual tiver algum item com needs_confirmation=true e a mensagem do usuário for uma confirmação curta, como:
- "sim"
- "isso"
- "correto"
- "confirmo"
- "exatamente"
- "foi isso"
- "foi essa data"
- "sim, foi"
- "está certo"

Então você deve retornar proposta para o item pendente de confirmação, mantendo o value_normalized atual, com:
- extraction_mode = "explicit"
- confidence >= 0.95
- validation_hint = "valid"
- suggested_action = "apply_directly"
- conflicts_with_current = false
- is_explicit_correction = false

Se houver mais de um item pendente de confirmação, priorize o item que foi a última pergunta do bot, usando last_question_type e asked_count do estado.

- Se a mensagem do usuário for apenas confirmação curta ("sim", "isso", "correto"), use o item que está com needs_confirmation=true e que foi perguntado mais recentemente.
- Nesses casos, confirme o valor atual em vez de criar um novo valor.

DATAS:
- Se o usuário não informou data, não preencha data_evento.
- Datas completas devem ser dd-mm-yyyy.
- Datas aproximadas, como "março de 2025", podem ser 01-mm-yyyy, mas precisam de validation_hint="needs_confirmation".
- Se o usuário disser "não lembro", "não sei", "não tenho certeza", use value_normalized="nao_sabe" somente se o schema permitir; caso contrário, não aplique automaticamente.

TEMPO DESEMPREGADO:

Para o campo tempo_desempregado, normalize assim:
- "menos"
- "menos de 12 meses"
- "menos que 12 meses"
- "menos de um ano"
- "menos de 1 ano"
- "há pouco tempo"

=> value_normalized = "menos_12_meses"

- "mais"
- "mais de 12 meses"
- "mais que 12 meses"
- "mais de um ano"
- "mais de 1 ano"
- "há mais de um ano"
- "dois anos"
- "mais de dois anos"

=> value_normalized = "mais_12_meses"

Se a última pergunta do bot foi sobre tempo_desempregado e o usuário responder apenas "menos" ou "mais", interprete como resposta direta a tempo_desempregado.

FORA DE ESCOPO E CONTRADIÇÃO DE NEXO:

Este sistema atende apenas casos de:
- acidente de trabalho
- doença ocupacional

Se o usuário informar temas como aposentadoria, tempo de contribuição, BPC/LOAS, pensão, revisão de benefício ou outros temas previdenciários sem relação com acidente/doença ocupacional:
- natureza_caso = "fora_escopo"
- nexo_trabalho = "nao"
- possible_disqualification = true
- out_of_scope_signal = true
- suggested_action = "apply_directly"

Se o usuário corrigir uma informação anterior e disser que o evento ocorreu fora do trabalho, por exemplo:
- "foi jogando bola"
- "foi no futebol"
- "foi em casa"
- "não estava trabalhando"
- "não teve relação com trabalho"
- "foi no fim de semana"
- "na verdade foi fora do trabalho"

então:
- nexo_trabalho = "nao"
- natureza_caso = "fora_escopo"
- is_explicit_correction = true
- suggested_action = "apply_directly"
- validation_hint = "valid"

SUGGESTED_ACTION:
- apply_directly: quando for explícito, válido e forte.
- apply_and_confirm_later: quando for inferido, parcial, aproximado ou confiança média.
- ask_now: quando houver ambiguidade importante.
- ignore: quando o campo já está respondido e a nova mensagem só confirma sem alterar.
- mark_inconsistency: quando contradiz o estado atual.

SCHEMA DISPONÍVEL:
{json.dumps(self.schema_context, ensure_ascii=False, indent=2)}

RETORNE APENAS JSON VÁLIDO NESTE FORMATO:

{{
  "proposed_updates": [
    {{
      "item_id": "...",
      "slug": "...",
      "value_raw": "...",
      "value_normalized": "...",
      "extraction_mode": "explicit|inferred|derived",
      "confidence": 0.0,
      "evidence": "...",
      "completeness": "complete|partial|ambiguous",
      "validation_hint": "valid|needs_confirmation|invalid|inconsistent",
      "conflicts_with_current": false,
      "suggested_action": "apply_directly|apply_and_confirm_later|ask_now|ignore|mark_inconsistency",
      "is_explicit_correction": false
    }}
  ],
  "global_signals": {{
    "possible_disqualification": false,
    "urgency_signal": false,
    "out_of_scope_signal": false,
    "multiple_items_answered": false,
    "contradiction_detected": false
  }},
  "extraction_notes": []
}}
"""

    def _build_user_prompt(
        self,
        message: str,
        state: ConversationState
    ) -> str:

        current_items = []

        for slug, item in state.items.items():
            current_items.append({
                "slug": slug,
                "status": item.status,
                "value_normalized": item.value_normalized,
                "confidence": item.confidence,
                "needs_confirmation": item.needs_confirmation,
            })

        return f"""
DATA ATUAL:
{datetime.now(UTC).date().isoformat()}

MENSAGEM DO USUÁRIO:
{message}

ESTADO ATUAL:
{json.dumps(current_items, ensure_ascii=False, indent=2)}

INSTRUÇÕES:
- Extraia todas as informações possíveis da mensagem.
- Analise contra todo o schema, não apenas contra a última pergunta.
- Não repita campo já confirmado, exceto se a mensagem trouxer confirmação explícita, correção ou conflito.
- Se a mensagem responder uma confirmação pendente, retorne o mesmo slug com o valor confirmado.
- Não use slugs fora do schema.
- Em casos de doença ocupacional, não espere a palavra exata "doença ocupacional".
- Relatos de dor/lesão desenvolvida após anos de esforço no trabalho podem indicar doenca_ocupacional.
- Se o usuário disser que ficou com limitação para trabalhar, isso deve preencher reducao_capacidade="sim" e normalmente houve_sequela="sim".
- Não transforme "carregando peso" em ocupação; use isso como evidência de esforço laboral.

"""

    def _safe_json_loads(self, content: str) -> dict:
        try:
            return json.loads(content)
        except Exception:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)

            if match:
                return json.loads(match.group(0))

            raise ValueError(f"Resposta da OpenAI não é JSON válido:\n{content}")

    def _normalize_by_field_type(self, slug, value):
        row = self.schema_df[self.schema_df["slug"] == slug].iloc[0]
        tipo = safe_str(row.get("tipo_valor")).lower()

        if tipo in ["texto", "texto_livre", "string"]:
            return value

        return normalize_value(value)


    def _parse_response(
        self,
        message: str,
        data: dict
    ) -> ExtractionResult:

        message_context = MessageContext(
            message_id=str(uuid.uuid4()),
            text=message,
            source="openai",
            timestamp=now_utc()
        )

        proposals = []

        valid_slugs = set(self.schema_df["slug"].astype(str).str.strip())

        for proposal in data.get("proposed_updates", []):

            slug = safe_str(proposal.get("slug"))

            if slug not in valid_slugs:
                continue

            proposals.append(
                UpdateProposal(
                    item_id=safe_str(proposal.get("item_id")),
                    slug=slug,
                    value_raw=proposal.get("value_raw"),
                    value_normalized=self._normalize_by_field_type(
                        slug,
                        proposal.get("value_normalized")),
                    extraction_mode=safe_str(proposal.get("extraction_mode")) or "inferred",
                    confidence=float(proposal.get("confidence", 0.5)),
                    evidence=safe_str(proposal.get("evidence")),
                    completeness=safe_str(proposal.get("completeness")) or "ambiguous",
                    validation_hint=safe_str(proposal.get("validation_hint")) or "needs_confirmation",
                    conflicts_with_current=bool(proposal.get("conflicts_with_current", False)),
                    suggested_action=safe_str(proposal.get("suggested_action")) or "apply_and_confirm_later",
                    is_explicit_correction=bool(
                        proposal.get("is_explicit_correction", False)
                    )
                )
            )                

        global_signals = GlobalSignals(
            **data.get("global_signals", {})
        )

        return ExtractionResult(
            message_context=message_context,
            proposed_updates=proposals,
            global_signals=global_signals,
            extraction_notes=data.get("extraction_notes", [])
        )


#extractor = OpenAIExtractor(schema_df)

# - StateUpdateEngine

class StateUpdateEngine:

    COUNTABLE_STATUSES = {"confirmed", "inferred", "partial"}

    def apply(
        self,
        state: ConversationState,
        extraction: ExtractionResult
    ) -> ConversationState:

        state.audit.messages.append(to_dict(extraction.message_context))

        for proposal in extraction.proposed_updates:
            self._apply_single(state, proposal)

        self._recalculate_derived(state)
        state.session.updated_at = now_utc()

        return state

    def _apply_single(
        self,
        state: ConversationState,
        proposal: UpdateProposal
    ):

        if proposal.slug not in state.items:
            return

        item = state.items[proposal.slug]

        state.audit.extraction_events.append(to_dict(proposal))

        if proposal.suggested_action == "ignore":
            self._maybe_upgrade_existing_item(item, proposal)
            return

        if proposal.confidence < MEDIUM_CONFIDENCE:
            return

        new_value = proposal.value_normalized

        # confirmação de item pendente com o mesmo valor
        if (
            item.value_normalized is not None
            and normalize_value(item.value_normalized) == normalize_value(new_value)
            and proposal.suggested_action == "apply_directly"
        ):
            self._confirm_existing_item(item, proposal)
            return
        
        # conflito real
        if (
            item.value_normalized is not None
            and normalize_value(item.value_normalized) != normalize_value(new_value)
        ):

            if proposal.is_explicit_correction:
                self._apply_explicit_correction(
                    item,
                    proposal,
                    new_value
                )
                return

            item.status = "inconsistent"
            item.conflict_with.append(str(proposal.value_normalized))
            item.updated_at = now_utc()
            item.history.append({
                "timestamp": now_utc(),
                "previous_value": item.value_normalized,
                "proposed_value": proposal.value_normalized,
                "confidence": proposal.confidence,
                "source": proposal.extraction_mode,
                "action": "conflict_detected",
            })
            return        

        now = now_utc()

        item.value_raw = proposal.value_raw
        item.value_normalized = new_value
        item.confidence = proposal.confidence
        item.source = proposal.extraction_mode
        item.updated_at = now
        item.answered_at = item.answered_at or now

        if proposal.evidence:
            item.evidence.append(proposal.evidence)

        item.history.append({
            "timestamp": now,
            "value": new_value,
            "confidence": proposal.confidence,
            "source": proposal.extraction_mode,
            "suggested_action": proposal.suggested_action,
        })

        if (
            proposal.suggested_action == "apply_and_confirm_later"
            or proposal.validation_hint == "needs_confirmation"
            or proposal.confidence < HIGH_CONFIDENCE
        ):
            item.status = "inferred"
            item.needs_confirmation = True
        else:
            item.status = "confirmed"
            item.needs_confirmation = False

    def _confirm_existing_item(
        self,
        item: ItemState,
        proposal: UpdateProposal
    ):

        now = now_utc()

        item.status = "confirmed"
        item.needs_confirmation = False
        item.confidence = max(item.confidence, proposal.confidence)
        item.updated_at = now

        if proposal.evidence:
            item.evidence.append(proposal.evidence)

        item.history.append({
            "timestamp": now,
            "value": proposal.value_normalized,
            "confidence": proposal.confidence,
            "source": proposal.extraction_mode,
            "action": "confirmation_accepted",
        })

    def _maybe_upgrade_existing_item(
        self,
        item: ItemState,
        proposal: UpdateProposal
    ):

        if proposal.confidence > item.confidence:
            item.confidence = proposal.confidence
            item.updated_at = now_utc()

        if (
            item.needs_confirmation
            and proposal.confidence >= HIGH_CONFIDENCE
        ):
            item.status = "confirmed"
            item.needs_confirmation = False

    def _recalculate_derived(
        self,
        state: ConversationState
    ):

        derived = state.derived

        derived.active_items.clear()
        derived.pending_items.clear()
        derived.confirmation_queue.clear()
        derived.inconsistent_items.clear()
        derived.dispensed_items.clear()

        answered = 0
        total = len(state.items)

        for slug, item in state.items.items():

            if item.status in self.COUNTABLE_STATUSES:
                answered += 1
                derived.active_items.append(slug)

            if item.status in ["unanswered", "empty"]:
                derived.pending_items.append(slug)

            if item.needs_confirmation:
                derived.confirmation_queue.append(slug)

            if item.status == "inconsistent":
                derived.inconsistent_items.append(slug)

            if item.status == "dispensed":
                derived.dispensed_items.append(slug)

        derived.completion_ratio = answered / total if total else 0.0

    def _apply_explicit_correction(
        self,
        item: ItemState,
        proposal: UpdateProposal,
        new_value: Any
    ):

        now = now_utc()

        previous_value = item.value_normalized
        previous_status = item.status

        item.value_raw = proposal.value_raw
        item.value_normalized = new_value
        item.status = "confirmed"
        item.confidence = proposal.confidence
        item.source = proposal.extraction_mode
        item.needs_confirmation = False
        item.conflict_with = []
        item.updated_at = now
        item.answered_at = item.answered_at or now

        if proposal.evidence:
            item.evidence.append(proposal.evidence)

        item.history.append({
            "timestamp": now,
            "previous_value": previous_value,
            "new_value": new_value,
            "previous_status": previous_status,
            "confidence": proposal.confidence,
            "source": proposal.extraction_mode,
            "action": "explicit_correction_applied",
            "evidence": proposal.evidence,
        })


#updater = StateUpdateEngine()

# - RuleEngine

class RuleEngine:

    def __init__(self, dsl: DSLInterpreter):
        self.dsl = dsl

    def evaluate(
        self,
        state: ConversationState,
        rules_df: pd.DataFrame
    ) -> ConversationState:

        state.classification.rule_hits.clear()
        state.classification.justification_bullets.clear()
        state.classification.recommended_next_steps.clear()
        state.classification.disqualification_reason = None

        state.audit.rule_evaluations.clear()

        score_total = 0
        disqualified = False

        active_rules = self._get_active_rules(rules_df)

        disqualification_rules = active_rules[
            active_rules["tipo_regra"].apply(
                lambda x: normalize_value(x) == "desqualificacao"
            )
        ]

        score_rules = active_rules[
            active_rules["tipo_regra"].apply(
                lambda x: normalize_value(x) == "score"
            )
        ]

        classification_rules = active_rules[
            active_rules["tipo_regra"].apply(
                lambda x: normalize_value(x) == "classificacao"
            )
        ]

        # 1. Desqualificação primeiro
        for _, rule in disqualification_rules.iterrows():

            evaluation = self._evaluate_rule(
                rule,
                state
            )

            state.audit.rule_evaluations.append(
                evaluation
            )

            if not evaluation["result"]:
                continue

            disqualified = True

            hit = self._build_hit(rule)

            state.classification.rule_hits.append(
                hit
            )

            justificativa = safe_str(
                rule.get("justificativa_template")
            )

            proximo_passo = safe_str(
                rule.get("proximo_passo_template")
            )

            state.classification.disqualification_reason = justificativa

            if justificativa:
                state.classification.justification_bullets.append(
                    justificativa
                )

            if proximo_passo:
                state.classification.recommended_next_steps.append(
                    proximo_passo
                )

            break

        if disqualified:

            state.classification.score_total = 0
            state.classification.commercial_status = "desqualificado"

            return state

        # 2. Score
        for _, rule in score_rules.iterrows():

            evaluation = self._evaluate_rule(
                rule,
                state
            )

            state.audit.rule_evaluations.append(
                evaluation
            )

            if not evaluation["result"]:
                continue

            hit = self._build_hit(rule)

            state.classification.rule_hits.append(
                hit
            )

            score_total += hit["score_delta"]

            justificativa = safe_str(
                rule.get("justificativa_template")
            )

            proximo_passo = safe_str(
                rule.get("proximo_passo_template")
            )

            if justificativa:
                state.classification.justification_bullets.append(
                    justificativa
                )

            if proximo_passo:
                state.classification.recommended_next_steps.append(
                    proximo_passo
                )

        state.classification.score_total = score_total

        # 3. Classificação depois do score_total
        matched_classification = False

        for _, rule in classification_rules.iterrows():

            evaluation = self._evaluate_rule(
                rule,
                state
            )

            state.audit.rule_evaluations.append(
                evaluation
            )

            if not evaluation["result"]:
                continue

            matched_classification = True

            hit = self._build_hit(rule)

            state.classification.rule_hits.append(
                hit
            )

            status = normalize_value(
                rule.get("status_resultante")
            )

            if status:
                state.classification.commercial_status = status
            else:
                state.classification.commercial_status = self._classify(
                    score_total
                )

            justificativa = safe_str(
                rule.get("justificativa_template")
            )

            proximo_passo = safe_str(
                rule.get("proximo_passo_template")
            )

            if justificativa:
                state.classification.justification_bullets.append(
                    justificativa
                )

            if proximo_passo:
                state.classification.recommended_next_steps.append(
                    proximo_passo
                )

            break

        if not matched_classification:
            state.classification.commercial_status = self._classify(
                score_total
            )

        # 4. Limitadores de status comercial
        self._apply_status_caps(
            state
        )

        return state

    def _apply_status_caps(
        self,
        state: ConversationState
    ):

        current_status = state.classification.commercial_status

        if current_status == "desqualificado":
            return

        caps = []

        if self._item_value(state, "houve_sequela") == "nao":
            caps.append({
                "max_status": "frio",
                "reason": "Caso sem sequela informada limita a prioridade comercial.",
                "rule_name": "cap_sem_sequela"
            })

        if (
            self._item_value(state, "data_evento") is None
            or self._item_status(state, "data_evento") in [
                "unanswered",
                "empty",
                "inferred"
            ]
            or self._item_needs_confirmation(state, "data_evento")
        ):
            caps.append({
                "max_status": "morno",
                "reason": "Data do evento ausente ou pendente de confirmação limita o status a morno.",
                "rule_name": "cap_data_pendente"
            })

        if self._item_value(state, "tipo_vinculo") in [
            "autonomo",
            "mei"
        ]:
            caps.append({
                "max_status": "morno",
                "reason": "Vínculo autônomo/MEI limita o status inicial a morno.",
                "rule_name": "cap_vinculo_autonomo_mei"
            })

        final_status = current_status

        for cap in caps:

            capped_status = self._min_status(
                final_status,
                cap["max_status"]
            )

            if capped_status != final_status:

                state.classification.justification_bullets.append(
                    cap["reason"]
                )

                state.classification.rule_hits.append({
                    "rule_id": "CAP",
                    "rule_name": cap["rule_name"],
                    "rule_type": "status_cap",
                    "score_delta": 0,
                    "status_resultante": cap["max_status"]
                })

            final_status = capped_status

        state.classification.commercial_status = final_status

    def _item_value(
        self,
        state: ConversationState,
        slug: str
    ) -> Optional[str]:

        item = state.items.get(slug)

        if item is None:
            return None

        if item.value_normalized is None:
            return None

        return normalize_value(
            item.value_normalized
        )

    def _item_status(
        self,
        state: ConversationState,
        slug: str
    ) -> Optional[str]:

        item = state.items.get(slug)

        if item is None:
            return None

        return item.status

    def _item_needs_confirmation(
        self,
        state: ConversationState,
        slug: str
    ) -> bool:

        item = state.items.get(slug)

        if item is None:
            return False

        return bool(
            item.needs_confirmation
        )

    def _min_status(
        self,
        current: str,
        cap: str
    ) -> str:

        order = {
            "desqualificado": 0,
            "frio": 1,
            "morno": 2,
            "quente": 3
        }

        reverse = {
            0: "desqualificado",
            1: "frio",
            2: "morno",
            3: "quente"
        }

        current_rank = order.get(
            normalize_value(current),
            1
        )

        cap_rank = order.get(
            normalize_value(cap),
            current_rank
        )

        return reverse[
            min(
                current_rank,
                cap_rank
            )
        ]

    def _get_active_rules(
        self,
        rules_df: pd.DataFrame
    ) -> pd.DataFrame:

        if "ativa" not in rules_df.columns:
            active = rules_df.copy()
        else:

            def is_active(x):
                if pd.isna(x):
                    return False

                value = str(x).strip().lower()

                return value in [
                    "1",
                    "1.0",
                    "sim",
                    "s",
                    "true",
                    "yes",
                    "ativo",
                    "ativa",
                    "verdadeiro"
                ]

            active = rules_df[
                rules_df["ativa"].apply(is_active)
            ].copy()

        if "prioridade" in active.columns:
            active = active.sort_values(
                "prioridade",
                ascending=False
            )

        return active

    def _evaluate_rule(
        self,
        rule: pd.Series,
        state: ConversationState
    ) -> dict:

        condition = safe_str(
            rule.get("condicao")
        )

        try:
            result = self.dsl.evaluate(
                condition,
                state
            )
            error = None

        except Exception as e:
            result = False
            error = str(e)

        return {
            "rule_id": safe_str(rule.get("regra_id")),
            "rule_name": safe_str(rule.get("nome_regra")),
            "rule_type": safe_str(rule.get("tipo_regra")),
            "condition": condition,
            "result": result,
            "error": error,
        }

    def _build_hit(
        self,
        rule: pd.Series
    ) -> dict:

        return {
            "rule_id": safe_str(rule.get("regra_id")),
            "rule_name": safe_str(rule.get("nome_regra")),
            "rule_type": safe_str(rule.get("tipo_regra")),
            "score_delta": self._parse_score(rule.get("score_delta")),
            "status_resultante": safe_str(rule.get("status_resultante")),
        }

    def _parse_score(
        self,
        value: Any
    ) -> int:

        try:
            if pd.isna(value):
                return 0

            return int(
                float(value)
            )

        except Exception:
            return 0

    def _classify(
        self,
        score: int
    ) -> str:

        if score >= CLASSIFICATION_THRESHOLDS["quente"]:
            return "quente"

        if score >= CLASSIFICATION_THRESHOLDS["morno"]:
            return "morno"

        return "frio"

    def diagnose(
        self,
        state: ConversationState,
        rules_df: pd.DataFrame
    ) -> List[dict]:

        diagnostics = []

        for _, rule in self._get_active_rules(
            rules_df
        ).iterrows():

            diagnostics.append(
                self._evaluate_rule(
                    rule,
                    state
                )
            )

        return diagnostics

# - PlannerEngine

class PlannerEngine:

    def __init__(
        self,
        schema_df: pd.DataFrame,
        dsl: DSLInterpreter
    ):
        self.schema_df = schema_df
        self.dsl = dsl

    def plan(
        self,
        state: ConversationState
    ) -> ConversationState:

        if is_case_complete(state):
            state.planner.next_item_slug = None
            state.planner.next_question_type = None
            state.planner.next_question_text = None
            state.planner.planning_reason = "Caso completo ou encerrado."
            state.planner.candidate_items_ranked = []

            state.audit.planning_events.append({
                "slug": None,
                "score": None,
                "question_text": None,
                "question_type": None,
                "reason": "Caso completo ou encerrado."
            })

            return state

        candidates = []

        for _, row in self.schema_df.iterrows():
            candidate = self._build_candidate(row, state)

            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        state.planner.candidate_items_ranked = candidates

        if not candidates:
            state.planner.next_item_slug = None
            state.planner.next_question_type = None
            state.planner.next_question_text = None
            state.planner.planning_reason = "Nenhum item pendente."
            return state

        best = candidates[0]
        item = state.items[best["slug"]]

        state.planner.next_item_slug = best["slug"]
        state.planner.next_question_type = best["question_type"]
        state.planner.next_question_text = best["question_text"]
        state.planner.planning_reason = best["reason"]

        item.asked_count += 1
        item.last_question_type = best["question_type"]

        state.audit.planning_events.append(best)

        return state

    def _is_eligibility_core_item(self, slug: str) -> bool:
        return slug in [
            "natureza_caso",
            "nexo_trabalho",
            "descricao_caso",
            "tipo_vinculo",
            "data_evento"
        ]

    def _build_candidate(
        self,
        row: pd.Series,
        state: ConversationState
    ) -> Optional[dict]:

        slug = safe_str(row["slug"])

        if slug not in state.items:
            return None

        item = state.items[slug]

        if item.status == "confirmed":
            return None

        if self._is_dispensed(row, state):
            item.status = "dispensed"
            item.updated_at = now_utc()
            return None

        score = 0
        reasons = []

      
        if item.needs_confirmation:

            if self._is_eligibility_core_item(slug):
                score += PLANNER_WEIGHTS["needs_confirmation"]
                reasons.append("confirmação pendente de item crítico")
            else:
                score += 35
                reasons.append("confirmação pendente complementar")

        if item.status == "inconsistent":
            score += PLANNER_WEIGHTS["inconsistent"]
            reasons.append("item inconsistente")

        if item.status in ["unanswered", "empty"]:
            score += PLANNER_WEIGHTS["unanswered"]

        if self._is_required(row):
            score += PLANNER_WEIGHTS["required"]
            reasons.append("item obrigatório")

        base_priority = self._parse_int(row.get("prioridade_base"), default=0)
        score += base_priority

        if self._activation_rule_blocks(row, state):
            return None

        if safe_str(row.get("regra_ativacao")):
            score += PLANNER_WEIGHTS["activation_rule"]
            reasons.append("regra ativada")

        score -= item.asked_count * PLANNER_WEIGHTS["asked_penalty"]

        question_type, question_text = self._choose_question(row, item)

        return {
            "slug": slug,
            "score": score,
            "question_text": question_text,
            "question_type": question_type,
            "reason": ", ".join(reasons) if reasons else "prioridade dinâmica",
        }

    def _is_required(self, row: pd.Series) -> bool:
        return safe_str(row.get("obrigatorio")).lower() in ["1", "sim", "true", "yes"]

    def _is_dispensed(
        self,
        row: pd.Series,
        state: ConversationState
    ) -> bool:

        expr = safe_str(row.get("regra_dispensa"))

        if not expr:
            return False

        try:
            return self.dsl.evaluate(expr, state)
        except Exception:
            return False

    def _activation_rule_blocks(
        self,
        row: pd.Series,
        state: ConversationState
    ) -> bool:

        expr = safe_str(row.get("regra_ativacao"))

        if not expr:
            return False

        try:
            return not self.dsl.evaluate(expr, state)
        except Exception:
            return True

    def _choose_question(
        self,
        row: pd.Series,
        item: ItemState
    ) -> tuple[str, str]:

        if item.status == "inconsistent":
            question = safe_str(row.get("pergunta_reparo"))
            question_type = "repair"

        elif item.needs_confirmation:
            question = safe_str(row.get("pergunta_confirmacao"))
            question_type = "confirmation"

        else:
            question = safe_str(row.get("pergunta_principal"))
            question_type = "principal"

        bad_values = [
            "nao_sabe",
            "nao_sei",
            "nao sabe",
            "não sei",
            "unknown",
            "none",
            "null",
            ""
        ]

        if (
            "{valor}" in question
            and normalize_value(item.value_normalized) in bad_values
        ):
            question = (
                safe_str(row.get("pergunta_principal"))
                or
                safe_str(row.get("pergunta_reparo"))
                or
                f"Você poderia informar: {safe_str(row.get('label'))}?"
            )

        question = question.replace(
            "{valor}",
            display_value(item.value_normalized)
        )

        if not question:
            question = f"Você poderia informar: {safe_str(row.get('label'))}?"

        return question_type, question

    def _parse_int(self, value: Any, default: int = 0) -> int:
        try:
            if pd.isna(value):
                return default
            return int(float(value))
        except Exception:
            return default


#planner = PlannerEngine(schema_df, dsl)

# - ConversationEngine

class ConversationEngine:

    def __init__(
        self,
        schema_df: pd.DataFrame,
        rules_df: pd.DataFrame,
        extractor: OpenAIExtractor,
        updater: StateUpdateEngine,
        rule_engine: RuleEngine,
        planner: PlannerEngine
    ):

        self.schema_df = schema_df
        self.rules_df = rules_df

        self.extractor = extractor
        self.updater = updater
        self.rule_engine = rule_engine
        self.planner = planner

        self.reset()

    # ---------------------------------

    def reset(self):

        self.state = build_initial_state(
            self.schema_df
        )

    # ---------------------------------

    def _current_item_value(
        self,
        slug: str
    ) -> Optional[str]:

        item = self.state.items.get(slug)

        if item is None:
            return None

        if item.value_normalized is None:
            return None

        return normalize_value(
            item.value_normalized
        )

    def process(
        self,
        user_message: str,
        debug: bool = True
    ) -> dict:

        user_message = self._apply_deterministic_out_of_scope(
            user_message
        )

        user_message = self._apply_deterministic_nexo_conflict(
            user_message
        )

        user_message = self._apply_deterministic_short_answer(
            user_message
        )

        extraction = self.extractor.extract(
            user_message,
            self.state
        )

        self.state = self.updater.apply(
            self.state,
            extraction
        )

        self.state = self.rule_engine.evaluate(
            self.state,
            self.rules_df
        )

        self.state = self.planner.plan(
            self.state
        )

        assistant_message = self._build_assistant_message()

        response = {

            "assistant_message":
                assistant_message,

            "classification":
                to_dict(
                    self.state.classification
                ),

            "planner":
                to_dict(
                    self.state.planner
                ),

            "state":
                to_dict(
                    self.state
                ) if debug else None,

            "extraction":
                to_dict(
                    extraction
                ) if debug else None
        }

        return response

    def _apply_deterministic_out_of_scope(
        self,
        user_message: str
    ) -> str:

        text = normalize_value(user_message)

        out_of_scope_terms = [
            "aposentadoria",
            "aposentar",
            "me_aposentar",
            "tempo_de_contribuicao",
            "tempo_de_contribuição",
            "tempo_de_servico",
            "tempo_de_serviço",
            "contribuicao",
            "contribuição",
            "bpc",
            "loas",
            "beneficio",
            "benefício",
            "pensao",
            "pensão",
            "revisao",
            "revisão",
            "salario_maternidade",
            "salário_maternidade",
            "auxilio_reclusao",
            "auxílio_reclusão",
        ]

        strong_out_of_scope_phrases = [
            "quero_me_aposentar",
            "posso_me_aposentar",
            "ja_posso_me_aposentar",
            "já_posso_me_aposentar",
            "ja_deu_meu_tempo",
            "já_deu_meu_tempo",
            "deu_meu_tempo",
            "tempo_de_aposentadoria",
            "tempo_de_contribuicao",
            "tempo_de_contribuição",
            "tempo_de_servico",
            "tempo_de_serviço",
            "quero_dar_entrada_na_aposentadoria",
            "dar_entrada_na_aposentadoria",
        ]

        work_injury_terms = [
            "acidente",
            "doenca_ocupacional",
            "doença_ocupacional",
            "dor",
            "lesao",
            "lesão",
            "sequela",
            "cirurgia",
            "machuquei",
            "quebrei",
            "cai",
            "caí",
            "trabalho",
            "servico",
            "serviço",
            "empresa",
            "obra",
            "andaime",
        ]

        has_out_of_scope = any(
            term in text
            for term in out_of_scope_terms
        )

        has_strong_out_of_scope = any(
            term in text
            for term in strong_out_of_scope_phrases
        )

        has_work_injury_signal = any(
            term in text
            for term in work_injury_terms
        )

        if (
            has_strong_out_of_scope
            or
            (
                has_out_of_scope
                and not has_work_injury_signal
            )
        ):
            return (
                user_message
                + "\n\nInterpretação determinística: "
                "fora_escopo_previdenciario=true; "
                "natureza_caso=fora_escopo; "
                "nexo_trabalho=nao; "
                "is_explicit_correction=true"
            )

        return user_message

    def _apply_deterministic_nexo_conflict(
        self,
        user_message: str
    ) -> str:

        text = normalize_value(user_message)

        current_nexo = self._current_item_value(
            "nexo_trabalho"
        )

        if current_nexo not in [
            "sim",
            "nao_sabe"
        ]:
            return user_message

        no_work_context_terms = [
            "jogando_bola",
            "jogando_futebol",
            "futebol",
            "volei",
            "vôlei",
            "volei_de_praia",
            "vôlei_de_praia",
            "em_casa",
            "na_minha_casa",
            "fora_do_trabalho",
            "nao_foi_no_trabalho",
            "não_foi_no_trabalho",
            "nao_estava_trabalhando",
            "não_estava_trabalhando",
            "nao_tem_relacao",
            "não_tem_relação",
            "nada_a_ver_com_trabalho",
            "fim_de_semana",
            "lazer",
            "academia",
            "correndo",
            "brincando",
        ]

        explicit_negation_terms = [
            "na_verdade",
            "pensando_melhor",
            "corrigindo",
            "me_enganei",
            "foi_jogando",
            "eu_cai_jogando",
            "cai_jogando",
            "caí_jogando",
        ]

        has_no_work_context = any(
            term in text
            for term in no_work_context_terms
        )

        has_explicit_correction = any(
            term in text
            for term in explicit_negation_terms
        )

        if has_no_work_context or has_explicit_correction:
            return (
                user_message
                + "\n\nInterpretação determinística: "
                "correcao_explicita=true; "
                "nexo_trabalho=nao; "
                "natureza_caso=fora_escopo"
            )

        return user_message

    def _apply_deterministic_short_answer(
        self,
        user_message: str
    ) -> str:

        text = normalize_value(user_message)

        target_slug = self.state.planner.next_item_slug

        if not target_slug:
            return user_message

        yes_values = [
            "sim",
            "s",
            "isso",
            "correto",
            "certo",
            "claro",
            "com_certeza",
            "exatamente",
            "confirmo",
            "positivo",
            "foi_isso",
            "foi_sim",
            "sim_foi",
        ]

        no_values = [
            "nao",
            "não",
            "n",
            "negativo",
            "de_jeito_nenhum",
            "nao_foi",
            "não_foi",
            "nao_precisei",
            "não_precisei",
            "nao_tive",
            "não_tive",
        ]

        boolean_slugs = [
            "nexo_trabalho",
            "houve_cirurgia",
            "houve_sequela",
            "reducao_capacidade",
        ]

        if target_slug in boolean_slugs:

            if text in yes_values or text.startswith("sim"):
                return (
                    user_message
                    + f"\n\nInterpretação determinística: "
                    f"{target_slug}=sim"
                )

            if text in no_values or text.startswith("nao") or text.startswith("não"):
                return (
                    user_message
                    + f"\n\nInterpretação determinística: "
                    f"{target_slug}=nao"
                )

            if target_slug == "reducao_capacidade" and text in [
                "diminuiu",
                "diminui",
                "reduziu",
                "reduz",
                "perdi_forca",
                "perdi_força",
                "nao_consigo_trabalhar",
                "não_consigo_trabalhar",
                "nao_consigo_carregar_peso",
                "não_consigo_carregar_peso",
            ]:
                return (
                    user_message
                    + "\n\nInterpretação determinística: "
                    "reducao_capacidade=sim"
                )

            if target_slug == "houve_sequela" and text in [
                "dor",
                "tenho_dor",
                "muita_dor",
                "fiquei_com_dor",
                "fiquei_com_sequela",
                "ficou_sequela",
                "limitacao",
                "limitação",
                "fiquei_limitado",
                "perdi_forca",
                "perdi_força",
            ]:
                return (
                    user_message
                    + "\n\nInterpretação determinística: "
                    "houve_sequela=sim"
                )

        if target_slug == "tempo_desempregado":

            if text in [
                "menos",
                "menos_12_meses",
                "menos_de_12_meses",
                "menos_de_um_ano",
                "menos_de_1_ano",
                "menos_que_12_meses",
                "menos_que_um_ano",
                "menos_que_1_ano",
            ]:
                return (
                    user_message
                    + "\n\nInterpretação determinística: "
                    "tempo_desempregado=menos_12_meses"
                )

            if text in [
                "mais",
                "mais_12_meses",
                "mais_de_12_meses",
                "mais_de_um_ano",
                "mais_de_1_ano",
                "mais_que_12_meses",
                "mais_que_um_ano",
                "mais_que_1_ano",
                "dois_anos",
                "2_anos",
            ]:
                return (
                    user_message
                    + "\n\nInterpretação determinística: "
                    "tempo_desempregado=mais_12_meses"
                )

            if text in [
                "nao_sei",
                "não_sei",
                "nao_sabe",
                "não_sabe",
                "não_lembro",
                "nao_lembro",
            ]:
                return (
                    user_message
                    + "\n\nInterpretação determinística: "
                    "tempo_desempregado=nao_sabe"
                )

        return user_message

    # ---------------------------------

    def _build_assistant_message(
        self
    ) -> str:

        state = self.state

        if is_case_complete(state):
            return build_completion_message(state)

        if state.planner.next_question_text:
            return state.planner.next_question_text

        return (
            "Obrigado. Já tenho informações suficientes para gerar um resumo "
            "preliminar do caso."
        )

    # ---------------------------------

    def get_state(
        self
    ) -> ConversationState:

        return self.state

    # ---------------------------------

    def generate_summary_payload(
        self
    ) -> dict:

        state = self.state

        items = {}

        for slug, item in state.items.items():

            if item.value_normalized is not None:

                items[slug] = {
                    "value":
                        item.value_normalized,

                    "status":
                        item.status,

                    "confidence":
                        item.confidence
                }

        return {

            "classification":
                to_dict(
                    state.classification
                ),

            "items":
                items
        }

# - load_csv

def load_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "cp1252"]
    separators = [";", ","]

    last_error = None

    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, encoding=encoding, sep=sep)

                if len(df.columns) > 1:
                    df.columns = [c.strip() for c in df.columns]
                    return df

            except Exception as e:
                last_error = e

    raise RuntimeError(f"Erro ao carregar CSV: {path}. Último erro: {last_error}")


SCHEMA_PATH = r'C:\Users\matheus_santos\Desktop\chatbot_ADV\prototipo\datasets\schema_itens.csv'
RULES_PATH = r'C:\Users\matheus_santos\Desktop\chatbot_ADV\prototipo\datasets\regras_dominio.csv'

#schema_df = load_csv(SCHEMA_PATH)
#rules_df = load_csv(RULES_PATH)

#print("schema_df:", schema_df.shape)
#print("rules_df:", rules_df.shape)

# - build_initial_state

def humanize_value(value: str) -> str:
    if value is None:
        return "Não informado"

    mapping = {
        "sim": "Sim",
        "nao": "Não",
        "nao_sabe": "Não sabe",
        "clt": "CLT",
        "mei": "MEI",
        "autonomo": "Autônomo",
        "desempregado": "Desempregado",
        "acidente_trabalho": "Acidente de trabalho",
        "doenca_ocupacional": "Doença ocupacional",
        "quente": "Quente",
        "morno": "Morno",
        "frio": "Frio",
        "desqualificado": "Desqualificado",
    }

    normalized = normalize_value(value)

    return mapping.get(
        normalized,
        str(value).replace("_", " ").capitalize()
    )


def generate_case_summary(state: ConversationState) -> dict:
    classification = state.classification

    summary = {
        "nome": humanize_value(get_item_display_value(state, "nome_completo")),
        "descricao_caso": get_item_display_value(state, "descricao_caso"),
        "natureza_caso": humanize_value(get_item_display_value(state, "natureza_caso")),
        "nexo_trabalho": humanize_value(get_item_display_value(state, "nexo_trabalho")),
        "ocupacao_evento": humanize_value(get_item_display_value(state, "ocupacao_evento")),
        "tipo_vinculo": humanize_value(get_item_display_value(state, "tipo_vinculo")),
        "data_evento": get_item_display_value(state, "data_evento"),
        "houve_cirurgia": humanize_value(get_item_display_value(state, "houve_cirurgia")),
        "houve_sequela": humanize_value(get_item_display_value(state, "houve_sequela")),
        "reducao_capacidade": humanize_value(get_item_display_value(state, "reducao_capacidade")),
        "score": classification.score_total,
        "status": humanize_value(classification.commercial_status),
        "justificativas": classification.justification_bullets,
        "proximos_passos": classification.recommended_next_steps,
        "pendencias": state.derived.pending_items,
        "confirmacoes_pendentes": state.derived.confirmation_queue,
    }

    return summary

def is_case_complete(state: ConversationState) -> bool:
    if state.classification.commercial_status == "desqualificado":
        return True

    has_no_confirmation = len(state.derived.confirmation_queue) == 0

    required_core_slugs = [
        "nome_completo",
        "descricao_caso",
        "natureza_caso",
        "nexo_trabalho",
        "houve_sequela",
        "reducao_capacidade",
        "houve_cirurgia",
        "tipo_vinculo",
    ]

    tipo_vinculo = normalize_value(
        get_item_display_value(
            state,
            "tipo_vinculo",
            default=""
        )
    )

    if tipo_vinculo == "desempregado":
        required_core_slugs.append("tempo_desempregado")

    missing_required = [
        slug
        for slug in required_core_slugs
        if get_item_display_value(
            state,
            slug,
            default=None
        ) is None
    ]

    if missing_required:
        return False

    has_commercial_signal = (
        state.classification.score_total >= 60
        or state.classification.commercial_status in [
            "quente",
            "morno"
        ]
    )

    return bool(
        has_no_confirmation
        and has_commercial_signal
    )


def build_completion_message(state: ConversationState) -> str:
    status = humanize_value(state.classification.commercial_status)

    if state.classification.commercial_status == "desqualificado":
        motivo = (
            state.classification.disqualification_reason
            or "O caso informado parece estar fora do foco desta triagem."
        )

        return (
            "Obrigado pelas informações.\n\n"
            "Pelo que você informou, este caso parece não se enquadrar "
            "no foco inicial da triagem.\n\n"
            f"Motivo: {motivo}"
        )

    return f"""
    Obrigado. Já tenho informações suficientes para gerar uma análise preliminar.

    Para analisar melhor o potencial da sua ação, preciso do seu Extrato Previdenciário (CNIS).

    Como obter o CNIS:

    1. Acesse o portal Meu INSS:
    https://meu.inss.gov.br

    2. Faça login utilizando seu CPF e senha do GOV.BR.

    3. No menu principal clique em:
    "Extrato Previdenciário (CNIS)"

    4. Confirme sua identidade caso solicitado.

    5. Quando o extrato abrir, clique em:
    "Baixar PDF"

    6. Anexe o PDF diretamente nesta conversa.

    Após o envio, um advogado especializado poderá analisar seu histórico previdenciário e o potencial do seu caso.
    """

def display_value(value: Any) -> str:
    if value is None:
        return ""

    value = str(value)

    if re.match(r"^\d{4}_\d{2}_\d{2}$", value):
        return value.replace("_", "-")

    return value


def get_item_display_value(
    state: ConversationState,
    slug: str,
    default: str = "Não informado"
) -> str:

    item = state.items.get(slug)

    if item is None or item.value_normalized is None:
        return default

    return display_value(item.value_normalized)



