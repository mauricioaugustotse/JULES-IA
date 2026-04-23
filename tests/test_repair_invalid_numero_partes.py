from repair_invalid_numero_partes import (
    _extract_page_rich_text_property,
    _is_textual_process_descriptor,
)


def test_is_textual_process_descriptor_flags_recurso_without_digits():
    assert _is_textual_process_descriptor(
        "Recursos Ordinários de Luiz Augusto Barcelos Lara e Divaldo Vieira Lara"
    )
    assert _is_textual_process_descriptor(
        "Recurso Especial do Ministério Público Eleitoral e da Coligação Independência e Luta para Mudar o Rio Grande"
    )
    assert not _is_textual_process_descriptor("0603457-70.2020.6.21.0000")


def test_extract_page_rich_text_property_reads_raw_notion_text():
    page_payload = {
        "properties": {
            "numero_processo": {
                "rich_text": [
                    {"plain_text": "Recursos Ordinários de Luiz Augusto "},
                    {"plain_text": "Barcelos Lara e Divaldo Vieira Lara"},
                ]
            }
        }
    }

    assert (
        _extract_page_rich_text_property(page_payload, "numero_processo")
        == "Recursos Ordinários de Luiz Augusto Barcelos Lara e Divaldo Vieira Lara"
    )
