"""Guards do preenchimento automatico de pedido_vista (anti-gravar-relator/alucinacao)."""
from fill_pedido_vista_via_grounding import _mesmo_ministro


def test_mesmo_ministro_tolera_nome_do_meio_e_sufixo():
    # relator pode vir com nome do meio expandido; o proposto (forma canonica/alias) tem de
    # casar para que o guard NUNCA grave o relator em pedido_vista.
    assert _mesmo_ministro("Min. Carlos Horbach", "Min. Carlos Bastide Horbach")
    assert _mesmo_ministro("Min. Carlos Bastide Horbach", "Min. Carlos Horbach")
    assert _mesmo_ministro("Min. Sebastião Reis", "Min. Sebastião Reis Júnior")
    assert _mesmo_ministro("Min. Dias Toffoli", "Min. Dias Toffoli")


def test_mesmo_ministro_distingue_ministros_diferentes():
    assert not _mesmo_ministro("Min. Dias Toffoli", "Min. André Mendonça")
    assert not _mesmo_ministro("Min. Alexandre de Moraes", "Min. André Ramos Tavares")
    assert not _mesmo_ministro("", "Min. André Mendonça")
    assert not _mesmo_ministro("Min. André Mendonça", "")
