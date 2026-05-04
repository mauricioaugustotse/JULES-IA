from cleanup_notion_relator_pedido_duplicates import canonical_person_select, planned_remaining_options
from tse_normalization import normalize_ministro_name, normalize_pedido_vista_value


class _Prop:
    def __init__(self, prop_type: str) -> None:
        self.type = prop_type


class _Schema:
    def __init__(self) -> None:
        self.properties = {"relator": _Prop("select")}
        self.raw_payload = {
            "properties": {
                "relator": {
                    "type": "select",
                    "select": {
                        "options": [
                            {"name": "Min. Edson Fachin", "color": "red"},
                            {"name": "Min. Luís Edson Fachin", "color": "blue"},
                            {"name": "Min. Sem Uso", "color": "default"},
                        ]
                    },
                }
            }
        }


def _page(value: str) -> dict:
    return {"properties": {"relator": {"type": "select", "select": {"name": value} if value else None}}}


def test_canonical_person_select_unifies_known_duplicate_names() -> None:
    assert canonical_person_select("Min. Luís Edson Fachin", "relator") == (
        "Min. Edson Fachin",
        "relator unificado em etiqueta canonica do mesmo ministro",
    )
    assert canonical_person_select("Min. Luiz Edson Fachin", "pedido_vista")[0] == "Min. Edson Fachin"
    assert canonical_person_select("Min. Luis Salomão", "pedido_vista")[0] == "Min. Luís Felipe Salomão"
    assert canonical_person_select("Min. Tarcísio Vieira de Carvalho", "relator")[0] == "Min. Tarcísio Vieira de Carvalho Neto"


def test_canonical_person_select_clears_non_person_relator_label() -> None:
    assert canonical_person_select("Min. Presidência", "relator") == (
        "",
        "relator continha etiqueta sem pessoa identificada; campo limpo",
    )


def test_core_minister_normalization_prevents_duplicate_recreation() -> None:
    assert normalize_ministro_name("Luís Edson Fachin") == "Min. Edson Fachin"
    assert normalize_ministro_name("Henrique Neves") == "Min. Henrique Neves da Silva"
    assert normalize_ministro_name("Paulo Tarso Sanseverino") == "Min. Paulo de Tarso Sanseverino"
    assert normalize_pedido_vista_value("Maria Claudia Bucchianeri Pinheiro") == "Min. Maria Cláudia Bucchianeri"


def test_planned_remaining_options_removes_unused_and_preserves_used_color() -> None:
    remaining, unused, missing = planned_remaining_options(
        _Schema(),
        [_page("Min. Edson Fachin"), _page("Min. Novo Ministro")],
        "relator",
    )

    assert remaining == [
        {"name": "Min. Edson Fachin", "color": "red"},
        {"name": "Min. Novo Ministro", "color": "default"},
    ]
    assert unused == ["Min. Luís Edson Fachin", "Min. Sem Uso"]
    assert missing == ["Min. Novo Ministro"]
