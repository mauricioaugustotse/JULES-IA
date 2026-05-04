from rewrite_notion_tema_punchline import (
    clean_theme,
    fallback_punchline,
    fallback_theme,
    punchline_invalid,
    theme_invalid,
)
from tse_youtube_notion_core import PublishPreviewRow


def test_clean_theme_removes_process_number() -> None:
    assert clean_theme("0600314-53 - recomposição de lista tríplice para o TRE") == "Recomposição de lista tríplice para o TRE"


def test_theme_validation_rejects_generic_decision_sentence() -> None:
    row = PublishPreviewRow(numero_processo="0600314-53")
    notes = theme_invalid("TSE decide controvérsia sobre propaganda eleitoral", row)
    assert "tema parece generico ou decisorio" in notes


def test_punchline_validation_rejects_formulaic_text() -> None:
    notes = punchline_invalid(
        "TSE mantém decisão sobre desaprovação de contas de campanha.",
        "Desaprovação de contas de campanha",
    )
    assert "punchline curta demais" in notes
    assert "punchline parece formulaica" in notes


def test_fallbacks_produce_distinct_theme_and_punchline() -> None:
    row = PublishPreviewRow(
        classe_processo="REspe",
        tema="Julgamento",
        resultado="Desprovido",
        analise_do_conteudo_juridico=(
            "O processo trata de recurso especial eleitoral sobre propaganda eleitoral antecipada "
            "em publicações na internet."
        ),
        raciocinio_juridico=(
            "O relator concluiu que as mensagens não continham pedido explícito de voto e manteve "
            "a decisão regional."
        ),
    )

    theme = fallback_theme(row)
    punchline = fallback_punchline(row, theme)

    assert "propaganda eleitoral antecipada" in theme.lower()
    assert theme.lower() not in punchline.lower() or len(punchline) > len(theme) + 60
