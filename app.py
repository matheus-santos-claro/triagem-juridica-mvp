import streamlit as st
from dotenv import load_dotenv
import html
import pandas as pd
from datetime import datetime

from engine_core import (
    load_csv,
    OpenAIExtractor,
    StateUpdateEngine,
    RuleEngine,
    PlannerEngine,
    DSLInterpreter,
    ConversationEngine,
    generate_case_summary,
)

from cnis_service import (
    extract_text_from_pdf,
    analyze_cnis_with_llm,
    build_cnis_summary_card,
)

from logger import save_interaction

load_dotenv()

st.set_page_config(
    page_title="Triagem Acidente de Trabalho",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .chat-container {
        background-color: #ECE5DD;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #ddd;
        margin-bottom: 16px;
    }

    .chat-row {
        display: flex;
        margin-bottom: 10px;
    }

    .chat-row.user {
        justify-content: flex-end;
    }

    .chat-row.assistant {
        justify-content: flex-start;
    }

    .chat-bubble {
        max-width: 78%;
        padding: 10px 14px;
        border-radius: 14px;
        font-size: 15px;
        line-height: 1.4;
        box-shadow: 0 1px 2px rgba(0,0,0,0.12);
        white-space: pre-wrap;
    }

    .chat-bubble.user {
        background-color: #DCF8C6;
        color: #111;
        border-bottom-right-radius: 4px;
    }

    .chat-bubble.assistant {
        background-color: #FFFFFF;
        color: #111;
        border-bottom-left-radius: 4px;
    }

    .chat-header {
        background-color: #075E54;
        color: white;
        padding: 12px 16px;
        border-radius: 14px 14px 0 0;
        font-weight: 600;
        margin-bottom: 0;
    }

    .chat-subheader {
        font-size: 13px;
        opacity: 0.85;
        font-weight: 400;
    }

    .summary-card {
    background: #ffffff;
    border: 1px solid #e6e6e6;
    border-radius: 16px;
    padding: 18px;
    margin-top: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.summary-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
}

.status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 12px;
}

.status-quente {
    background: #d8f5df;
    color: #167a32;
}

.status-morno {
    background: #fff3cd;
    color: #8a6500;
}

.status-frio {
    background: #f8d7da;
    color: #842029;
}

.status-desqualificado {
    background: #e2e3e5;
    color: #41464b;
}

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px 18px;
    margin-top: 10px;
}

.info-item {
    font-size: 14px;
}

.info-label {
    font-weight: 700;
    color: #555;
}

.info-value {
    color: #111;
}

.section-subtitle {
    font-weight: 700;
    margin-top: 14px;
    margin-bottom: 6px;
}
    </style>
    """,
    unsafe_allow_html=True
)

schema_df = load_csv("schema_itens.csv")
rules_df = load_csv("regras_dominio.csv")

def render_chat_message(role: str, content: str):
    safe_content = html.escape(content).replace("\n", "<br>")

    role_class = "user" if role == "user" else "assistant"

    st.markdown(
        f"""
        <div class="chat-row {role_class}">
            <div class="chat-bubble {role_class}">
                {safe_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_status_badge(status: str):
    normalized = str(status).lower()

    css_class = {
        "quente": "status-quente",
        "morno": "status-morno",
        "frio": "status-frio",
        "desqualificado": "status-desqualificado",
    }.get(normalized, "status-frio")

    return f'<span class="status-badge {css_class}">{status.upper()}</span>'


def render_info_item(label: str, value: str):
    return (
        f'<div class="info-item">'
        f'<div class="info-label">{label}</div>'
        f'<div class="info-value">{value or "Não informado"}</div>'
        f'</div>'
    )


def create_engine():
    dsl = DSLInterpreter()
    extractor = OpenAIExtractor(schema_df)
    updater = StateUpdateEngine()
    rule_engine = RuleEngine(dsl)
    planner = PlannerEngine(schema_df, dsl)

    return ConversationEngine(
        schema_df=schema_df,
        rules_df=rules_df,
        extractor=extractor,
        updater=updater,
        rule_engine=rule_engine,
        planner=planner,
    )

def get_current_state():
    return st.session_state.engine.get_state()


def get_current_classification():
    state = get_current_state()
    return state.classification


def get_status_color(status: str) -> str:
    colors = {
        "quente": "🟢",
        "morno": "🟡",
        "frio": "🔴",
        "desqualificado": "⚫",
    }
    return colors.get(status, "⚪")

def build_instagram_prefill_message(
    nome: str,
    telefone: str,
    relato: str,
    data_aproximada: str,
    origem: str = "instagram"
) -> str:

    partes = [
        f"Origem do lead: {origem}"
    ]

    if nome:
        partes.append(f"Nome do lead: {nome}")

    if telefone:
        partes.append(f"Telefone informado: {telefone}")

    if relato:
        partes.append(f"Relato inicial: {relato}")

    if data_aproximada:
        partes.append(f"Data aproximada informada: {data_aproximada}")

    return "\n".join(partes)


if "engine" not in st.session_state:
    st.session_state.engine = create_engine()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Olá! Sou o assistente de triagem. Me conte brevemente o que aconteceu no seu caso."
        }
    ]

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "cnis_analysis" not in st.session_state:
    st.session_state.cnis_analysis = None

if "cnis_summary" not in st.session_state:
    st.session_state.cnis_summary = None


st.title(
    "⚖️ Assistente Jurídico"
)

st.caption(
    "Triagem inicial de acidentes de trabalho e doenças ocupacionais"
)

with st.expander("📥 Simular dados recebidos do Instagram", expanded=False):

    col_ig1, col_ig2 = st.columns(2)

    with col_ig1:
        ig_nome = st.text_input(
            "Nome",
            placeholder="Ex: João da Silva"
        )

        ig_telefone = st.text_input(
            "Telefone",
            placeholder="Ex: (11) 99999-9999"
        )

    with col_ig2:
        ig_data = st.text_input(
            "Data aproximada",
            placeholder="Ex: 01-março de 2025, 2023, não lembro"
        )

    ig_relato = st.text_area(
        "Relato inicial",
        placeholder="Ex: Dor nas costas depois de anos carregando peso no trabalho..."
    )

    if st.button("Iniciar triagem com dados do Instagram"):

        prefill_message = build_instagram_prefill_message(
            nome=ig_nome,
            telefone=ig_telefone,
            relato=ig_relato,
            data_aproximada=ig_data
        )

        response = st.session_state.engine.process(
            prefill_message,
            debug=True
        )

        save_interaction({
            "timestamp": datetime.now().isoformat(),
            "source": "instagram_prefill",
            "user_input": prefill_message,
            "assistant_output": response["assistant_message"],
            "classification": response["classification"],
            "planner": response["planner"],
        })
        st.session_state.last_response = response

        mensagem_visivel = (
            "📥 Dados recebidos do formulário\n\n"
            f"**Nome:** {ig_nome or 'Não informado'}  \n"
            f"**Telefone:** {ig_telefone or 'Não informado'}  \n"
            f"**Data informada:** {ig_data or 'Não informada'}"
        )

        if ig_relato:
            mensagem_visivel += f"  \n**Relato inicial:** {ig_relato}"

        st.session_state.messages.append({
            "role": "user",
            "content": mensagem_visivel
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["assistant_message"]
        })

        st.rerun()

col_chat, col_side = st.columns([2, 1])


with col_chat:
    st.subheader("Assistente de Qualificação")

    st.markdown(
        """
        <div class="chat-header">
            ⚖️ Assistente Jurídico
            <div class="chat-subheader">
                Triagem inicial online
            </div>
        </div>
        <div class="chat-container">
        """,
        unsafe_allow_html=True
    )

    for msg in st.session_state.messages:
        render_chat_message(
            msg["role"],
            msg["content"]
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    state = get_current_state()
    classification = state.classification

    triagem_pronta_para_cnis = (
        classification.commercial_status in ["quente", "morno"]
        and state.planner.next_question_text is None
    )

    if triagem_pronta_para_cnis:

        st.divider()

        st.markdown("### 📎 Anexar CNIS")

        uploaded_cnis = st.file_uploader(
            "Selecione o PDF do CNIS",
            type=["pdf"],
            key="cnis_uploader_chat"
        )

        if uploaded_cnis is not None:

            if st.button("Analisar CNIS", key="btn_analisar_cnis_chat"):

                with st.spinner("Extraindo e analisando CNIS..."):

                    cnis_text = extract_text_from_pdf(
                        uploaded_cnis
                    )

                    cnis_analysis = analyze_cnis_with_llm(
                        cnis_text
                    )

                    cnis_summary = build_cnis_summary_card(
                        cnis_analysis
                    )

                    st.session_state.cnis_analysis = cnis_analysis
                    st.session_state.cnis_summary = cnis_summary

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        "Recebi seu CNIS com sucesso.\n\n"
                        "Obrigado por enviar as informações.\n\n"
                        "Seu caso foi encaminhado para análise preliminar junto "
                        "com o histórico previdenciário informado.\n\n"
                        "Um advogado especializado poderá avaliar:\n\n"
                        "• Existência de vínculo previdenciário\n"
                        "• Qualidade de segurado\n"
                        "• Benefícios recebidos\n"
                        "• Indicadores de pendência\n"
                        "• Potencial jurídico do caso\n\n"
                        "Abaixo apresento um resumo consolidado do CNIS enviado."
                    )
                })

                st.rerun()

    if st.session_state.cnis_summary:

        cnis = st.session_state.cnis_summary
        summary = generate_case_summary(state)

        st.divider()

        st.subheader("📄 Resumo do CNIS")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Vínculos", cnis.get("total_vinculos", 0))

        with c2:
            st.metric("Benefícios", cnis.get("total_beneficios", 0))

        with c3:
            st.metric("Confiança", cnis.get("confianca_extracao", ""))

        with st.container(border=True):
            st.markdown("#### Dados identificados")
            st.write(f"**Nome:** {cnis.get('nome')}")
            st.write(f"**CPF:** {cnis.get('cpf')}")
            st.write(f"**Nascimento:** {cnis.get('data_nascimento')}")
            st.write(f"**Competência mais antiga:** {cnis.get('competencia_mais_antiga')}")
            st.write(f"**Competência mais recente:** {cnis.get('competencia_mais_recente')}")
            st.write(f"**Última remuneração:** {cnis.get('ultima_remuneracao_identificada')}")

        if cnis.get("resumo_cnis"):
            with st.container(border=True):
                st.markdown("#### Resumo preliminar do CNIS")
                st.write(cnis.get("resumo_cnis"))

        if cnis.get("pontos_atencao"):
            with st.container(border=True):
                st.markdown("#### Pontos de atenção no CNIS")
                for ponto in cnis.get("pontos_atencao"):
                    st.write(f"- {ponto}")

        st.divider()

        st.subheader("⚖️ Resumo Executivo Final")

        status = summary["status"]
        score = summary["score"]

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Classificação", status)

        with c2:
            st.metric("Score", score)

        with st.container(border=True):
            st.markdown("#### Informações da triagem")
            st.write(f"**Lead:** {summary['nome']}")
            st.write(f"**Natureza:** {summary['natureza_caso']}")
            st.write(f"**Nexo com trabalho:** {summary['nexo_trabalho']}")
            st.write(f"**Vínculo:** {summary['tipo_vinculo']}")
            st.write(f"**Data:** {summary['data_evento']}")
            st.write(f"**Cirurgia:** {summary['houve_cirurgia']}")
            st.write(f"**Sequela:** {summary['houve_sequela']}")
            st.write(f"**Redução de capacidade:** {summary['reducao_capacidade']}")

        if summary["justificativas"]:
            with st.container(border=True):
                st.markdown("#### Justificativas")
                for item in summary["justificativas"]:
                    st.write(f"- {item}")

        if summary["proximos_passos"]:
            with st.container(border=True):
                st.markdown("#### Próximos passos")
                for item in summary["proximos_passos"]:
                    st.write(f"- {item}")

        with st.container(border=True):
            st.markdown("#### Complemento CNIS")
            st.write(cnis.get("resumo_cnis", ""))

            if cnis.get("pontos_atencao"):
                st.markdown("**Pontos CNIS:**")
                for ponto in cnis.get("pontos_atencao"):
                    st.write(f"- {ponto}")

    user_input = st.chat_input("Digite sua mensagem...")

    if user_input:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        response = st.session_state.engine.process(
            user_input,
            debug=True
        )

        save_interaction({
            "timestamp": datetime.now().isoformat(),
            "source": "chat",
            "user_input": user_input,
            "assistant_output": response["assistant_message"],
            "classification": response["classification"],
            "planner": response["planner"],
        })

        state_after = st.session_state.engine.get_state()

        st.session_state.debug_score_after_process = (
            state_after.classification.score_total
        )

        st.session_state.debug_status_after_process = (
            state_after.classification.commercial_status
        )

        st.session_state.debug_items_after_process = {
            slug: item.value_normalized
            for slug, item in state_after.items.items()
            if item.value_normalized is not None
        }

        st.session_state.last_response = response

        assistant_message = response["assistant_message"]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_message
            }
        )

        st.rerun()


with col_side:
    st.subheader("Painel de Controle - Resultado da Triagem")

    if st.button(
        "🔄 Nova Triagem",
        use_container_width=False
    ):
        st.session_state.clear()
        st.rerun()   

    state = get_current_state()
    classification = state.classification

    status = classification.commercial_status
    score = classification.score_total

    st.metric(
        "Status",
        f"{get_status_color(status)} {status.upper()}"
    )

    st.metric(
        "Score",
        score
    )

    st.divider()

    st.markdown("### Dados extraídos")

    extracted_items = {
        slug: item
        for slug, item in state.items.items()
        if item.value_normalized is not None
    }

    if not extracted_items:
        st.info("Nenhum dado extraído ainda.")
    else:
        for slug, item in extracted_items.items():
            st.write(
                f"**{slug}**: {item.value_normalized} "
                f"({item.status}, conf. {item.confidence})"
            )

    st.divider()

    st.markdown("### Próxima pergunta")
    st.write(
        state.planner.next_question_text
        or
        "Sem próxima pergunta."
    )

    st.divider()

    st.markdown("### Resumo Executivo")

    summary = generate_case_summary(state)

    st.markdown(f"**Status:** {summary['status']}")
    st.markdown(f"**Score:** {summary['score']}")

    st.markdown("**Dados principais:**")
    st.write(f"Nome: {summary['nome']}")
    st.write(f"Natureza: {summary['natureza_caso']}")
    st.write(f"Nexo com trabalho: {summary['nexo_trabalho']}")
    st.write(f"Vínculo: {summary['tipo_vinculo']}")
    st.write(f"Data: {summary['data_evento']}")
    st.write(f"Cirurgia: {summary['houve_cirurgia']}")
    st.write(f"Sequela: {summary['houve_sequela']}")
    st.write(f"Redução de capacidade: {summary['reducao_capacidade']}")

    st.divider()

    

    if summary["justificativas"]:
        st.markdown("**Justificativas:**")
        for item in summary["justificativas"]:
            st.write(f"- {item}")

    if summary["proximos_passos"]:
        st.markdown("**Próximos passos:**")
        for item in summary["proximos_passos"]:
            st.write(f"- {item}")

    if summary["pendencias"]:
        st.markdown("**Pendências:**")
        for item in summary["pendencias"]:
            st.write(f"- {item}")

    st.divider()


