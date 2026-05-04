from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from tse_youtube_notion_core import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_NEWS_GEMINI_MODEL,
    DEFAULT_NOTION_DATABASE_URL,
    GeminiSessionExtractor,
    NotionSessoesClient,
    RunArtifacts,
    build_preview_rows,
    build_runtime_context,
    enrich_preview_rows_with_process_metadata,
    enrich_preview_rows_with_theme_punchline,
    enrich_preview_rows_with_news,
    publish_preview_rows,
    rows_from_editor_records,
    rows_to_editor_records,
)


LOGGER = logging.getLogger(__name__)


def ensure_session_defaults() -> None:
    st.session_state.setdefault("preview_records", [])
    st.session_state.setdefault("analysis_payload", {})
    st.session_state.setdefault("publish_results", [])
    st.session_state.setdefault("artifact_dir", "")
    st.session_state.setdefault("last_error", "")


def load_notion_client_and_schema() -> tuple[Any, Any, str]:
    runtime = build_runtime_context()
    notion_key = runtime["notion_api_key"]
    if not notion_key:
        return None, None, "NOTION_API_KEY/NOTION_TOKEN não encontrado. A análise pode rodar, mas a publicação ficará desabilitada."
    try:
        client = NotionSessoesClient(
            api_key=notion_key,
            data_source_id=runtime["notion_data_source_id"],
        )
        schema = client.fetch_schema()
        return client, schema, ""
    except Exception as exc:  # pragma: no cover - depende de ambiente externo
        return None, None, f"Falha ao validar o data source do Notion: {exc}"


def analyze_video(youtube_url: str, model_name: str) -> None:
    runtime = build_runtime_context()
    gemini_key = runtime["gemini_api_key"]
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY não encontrado.")

    artifact_store = RunArtifacts.for_youtube_url(youtube_url)
    notion_client, notion_schema, notion_warning = load_notion_client_and_schema()

    extractor = GeminiSessionExtractor(
        api_key=gemini_key,
        model=model_name,
        artifact_store=artifact_store,
        logger=LOGGER,
    )
    analysis = extractor.analyze_session(youtube_url)
    rows = build_preview_rows(
        analysis,
        youtube_url=youtube_url,
        notion_schema=notion_schema,
        notion_client=notion_client,
    )
    rows = enrich_preview_rows_with_process_metadata(
        rows,
        api_key=gemini_key,
        model=model_name,
        artifact_store=artifact_store,
        logger=LOGGER,
        notion_schema=notion_schema,
    )
    rows = enrich_preview_rows_with_theme_punchline(
        rows,
        api_key=gemini_key,
        model=model_name,
        artifact_store=artifact_store,
        logger=LOGGER,
        notion_schema=notion_schema,
    )

    artifact_store.write_json("03_analysis.json", analysis.model_dump(mode="json"))
    artifact_store.write_json(
        "04_preview_rows.json",
        [row.model_dump(mode="json") for row in rows],
    )

    st.session_state["preview_records"] = rows_to_editor_records(rows)
    st.session_state["analysis_payload"] = analysis.model_dump(mode="json")
    st.session_state["publish_results"] = []
    st.session_state["artifact_dir"] = str(artifact_store.root_dir)
    st.session_state["last_error"] = notion_warning


def publish_current_rows() -> None:
    notion_client, notion_schema, notion_warning = load_notion_client_and_schema()
    if notion_warning:
        raise RuntimeError(notion_warning)
    if notion_client is None or notion_schema is None:
        raise RuntimeError("Cliente/schema do Notion indisponível.")

    rows = rows_from_editor_records(st.session_state["preview_records"], notion_schema)
    results = publish_preview_rows(rows, notion_client, notion_schema)
    st.session_state["publish_results"] = results

    artifact_dir = st.session_state.get("artifact_dir", "")
    if artifact_dir:
        RunArtifacts(root_dir=Path(st.session_state["artifact_dir"])).write_json(
            "05_publish_results.json",
            results,
        )


def enrich_current_rows_with_news() -> None:
    runtime = build_runtime_context()
    gemini_key = runtime["gemini_api_key"]
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY não encontrado.")

    _, notion_schema, _ = load_notion_client_and_schema()
    rows = rows_from_editor_records(st.session_state["preview_records"], notion_schema)

    artifact_dir = st.session_state.get("artifact_dir", "")
    artifact_store = RunArtifacts(root_dir=Path(artifact_dir)) if artifact_dir else None
    enriched_rows = enrich_preview_rows_with_news(
        rows,
        api_key=gemini_key,
        model=DEFAULT_NEWS_GEMINI_MODEL,
        artifact_store=artifact_store,
        logger=LOGGER,
    )
    st.session_state["preview_records"] = rows_to_editor_records(enriched_rows)
    st.session_state["publish_results"] = []
    if artifact_store:
        artifact_store.write_json(
            "04b_enriched_preview_rows.json",
            [row.model_dump(mode="json") for row in enriched_rows],
        )


def render_header() -> None:
    runtime = build_runtime_context()
    st.set_page_config(page_title="TSE YouTube > Notion", layout="wide")
    st.title("TSE YouTube > Notion")
    st.caption(
        "Extrai julgamentos de sessões do TSE via Gemini, gera prévia editável e publica no Notion."
    )
    st.markdown(
        f"Data source alvo: `{runtime['notion_data_source_id']}`  \n"
        f"Banco: {DEFAULT_NOTION_DATABASE_URL}"
    )


def render_controls() -> None:
    col_url, col_action = st.columns([6, 1])
    with col_url:
        youtube_url = st.text_input(
            "URL pública do YouTube",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input",
        )
        st.caption(f"Modelo Gemini fixo: `{DEFAULT_GEMINI_MODEL}`")
    with col_action:
        analyze_clicked = st.button("Analisar", use_container_width=True)

    if analyze_clicked:
        if not youtube_url.strip():
            st.session_state["last_error"] = "Informe a URL pública do YouTube."
        else:
            try:
                analyze_video(youtube_url.strip(), DEFAULT_GEMINI_MODEL)
            except Exception as exc:  # pragma: no cover - depende de ambiente externo
                st.session_state["last_error"] = str(exc)


def render_status() -> None:
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])
    artifact_dir = st.session_state.get("artifact_dir")
    if artifact_dir:
        st.info(f"Artefatos da execução atual: `{artifact_dir}`")


def render_preview() -> None:
    preview_records = st.session_state.get("preview_records", [])
    if not preview_records:
        st.warning("Nenhuma extração disponível. Informe uma URL e clique em Analisar.")
        return

    df = pd.DataFrame(preview_records)
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "warnings": st.column_config.TextColumn(width="medium"),
            "errors": st.column_config.TextColumn(width="medium"),
            "noticia_TSE": st.column_config.LinkColumn(display_text="Notícia TSE"),
            "noticia_TRE": st.column_config.LinkColumn(display_text="Notícia TRE"),
            "noticias_gerais": st.column_config.TextColumn(width="large"),
            "blocked": st.column_config.CheckboxColumn(disabled=True),
            "page_id": st.column_config.TextColumn(disabled=True),
            "action": st.column_config.SelectboxColumn(options=["create", "update"]),
        },
        key="preview_editor_grid",
    )
    st.session_state["preview_records"] = edited_df.to_dict("records")

    total_rows = len(st.session_state["preview_records"])
    blocked_rows = sum(1 for record in st.session_state["preview_records"] if record.get("blocked"))
    update_rows = sum(1 for record in st.session_state["preview_records"] if record.get("action") == "update")
    create_rows = total_rows - update_rows
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Linhas", total_rows)
    col_b.metric("Criar", create_rows)
    col_c.metric("Atualizar", update_rows)
    if blocked_rows:
        st.warning(f"{blocked_rows} linha(s) bloqueadas por validação. Corrija antes de publicar.")

    col_enrich, col_publish = st.columns([1, 1])
    with col_enrich:
        enrich_clicked = st.button(
            "Enriquecer Notícias Web",
            use_container_width=True,
        )
    with col_publish:
        publish_clicked = st.button(
            "Publicar no Notion",
            type="primary",
            use_container_width=True,
        )
    if enrich_clicked:
        try:
            enrich_current_rows_with_news()
        except Exception as exc:  # pragma: no cover - depende de ambiente externo
            st.session_state["last_error"] = str(exc)
    if publish_clicked:
        try:
            publish_current_rows()
        except Exception as exc:  # pragma: no cover - depende de ambiente externo
            st.session_state["last_error"] = str(exc)


def render_results() -> None:
    publish_results = st.session_state.get("publish_results", [])
    analysis_payload = st.session_state.get("analysis_payload", {})

    if publish_results:
        st.subheader("Resultado da Publicação")
        st.dataframe(pd.DataFrame(publish_results), use_container_width=True, hide_index=True)

    if analysis_payload:
        with st.expander("JSON da Extração", expanded=False):
            st.code(json.dumps(analysis_payload, ensure_ascii=False, indent=2), language="json")


def main() -> None:
    ensure_session_defaults()
    render_header()
    render_controls()
    render_status()
    render_preview()
    render_results()


if __name__ == "__main__":
    main()
