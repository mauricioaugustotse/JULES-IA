"""Geometria da janela da Consulta SADP: nasce inteira dentro da area util e centrada.

Cobre a funcao pura `_encaixar_na_area` — nenhum teste aqui abre janela (harness que abre
GUI visivel atrapalha a sessao de quem estiver na maquina).

Historico: a GUI chamava geometry("1180x680") sem posicao, o Windows empilhava as janelas
em cascata (+26 px por abertura) e a partir da terceira o rodape nascia atras da barra de
tarefas; alem disso 1180 px eram menos que os ~1264 px que as sete colunas da tabela pedem.
"""
import sadp_consulta_gui as gui

DEC_X, DEC_Y = 16, 39          # bordas laterais + (barra de titulo + borda de baixo)
MIN = (880, 480)
CONTEUDO = (1264, 680)         # o que os widgets pedem hoje


def _frame(larg, alt):
    return larg + DEC_X, alt + DEC_Y


def test_cabe_e_centraliza_no_monitor_do_notebook():
    area = (0, 0, 1536, 816)   # 1536x864 com barra de tarefas embaixo
    larg, alt, x, y = gui._encaixar_na_area(*CONTEUDO, DEC_X, DEC_Y, area, MIN)
    fw, fh = _frame(larg, alt)
    assert (larg, alt) == CONTEUDO          # conteudo inteiro: nada de coluna cortada
    assert x >= 0 and y >= 0 and x + fw <= 1536 and y + fh <= 816
    assert abs(x - (1536 - x - fw)) <= 1    # centrada na horizontal
    assert abs(y - (816 - y - fh)) <= 1     # centrada na vertical


def test_monitor_com_origem_negativa():
    """Monitor secundario acima do primario: rcWork comeca em y = -1080."""
    area = (0, -1080, 1920, 1032)
    larg, alt, x, y = gui._encaixar_na_area(*CONTEUDO, DEC_X, DEC_Y, area, MIN)
    fw, fh = _frame(larg, alt)
    assert -1080 <= y and y + fh <= -1080 + 1032
    assert 0 <= x and x + fw <= 1920


def test_encolhe_para_caber_em_tela_pequena():
    area = (0, 0, 1024, 728)
    larg, alt, x, y = gui._encaixar_na_area(*CONTEUDO, DEC_X, DEC_Y, area, MIN)
    fw, fh = _frame(larg, alt)
    assert fw <= 1024 and fh <= 728
    assert larg >= MIN[0] and alt >= MIN[1]


def test_nunca_abaixo_do_minimo_e_encosta_no_canto_quando_nao_cabe():
    area = (0, 0, 800, 480)                  # menor que o minsize nos dois eixos
    larg, alt, x, y = gui._encaixar_na_area(*CONTEUDO, DEC_X, DEC_Y, area, MIN)
    assert (larg, alt) == MIN                # o Tk imporia o minsize de qualquer forma
    assert (x, y) == (0, 0)                  # topo esquerdo visivel (campos de busca)


def test_tamanho_pedido_nunca_fica_abaixo_do_conteudo_quando_a_tela_permite():
    area = (0, 0, 2560, 1400)
    larg, alt, _, _ = gui._encaixar_na_area(*CONTEUDO, DEC_X, DEC_Y, area, MIN)
    assert (larg, alt) == CONTEUDO           # tela grande nao infla a janela


# --------------------------------------------------------------------- dialogo "Localizar no DJe"
MAE = (136, 79, 1264, 680)      # (x, y, larg, alt) da janela principal ja centralizada
DIALOGO = (473, 347)


def test_dialogo_nasce_sobre_a_janela_mae():
    x, y = gui._encaixar_dialogo(*DIALOGO, MAE, (0, 0, 1536, 816))
    assert x == 136 + (1264 - 473) // 2
    assert y == 79 + (680 - 347) // 3
    assert x >= 136 and x + 473 <= 136 + 1264      # dentro da mae na horizontal


def test_dialogo_preso_a_area_quando_a_mae_esta_na_borda():
    """Mae empurrada para a direita: o dialogo encosta na borda em vez de sair da tela."""
    x, y = gui._encaixar_dialogo(*DIALOGO, (1300, 700, 1264, 680), (0, 0, 1536, 816))
    assert x + 473 <= 1536 and y + 347 <= 816
    assert x >= 0 and y >= 0


def test_dialogo_maior_que_a_area_encosta_no_canto_visivel():
    x, y = gui._encaixar_dialogo(2000, 1500, MAE, (0, 0, 1536, 816))
    assert (x, y) == (gui.MARGEM_TELA, gui.MARGEM_TELA)


def test_dialogo_ignora_o_monitor_do_cursor_e_segue_a_mae(monkeypatch):
    """Regressao: com o cursor no monitor de cima (origem Y negativa) e a mae no primario,
    misturar os dois referenciais jogava o dialogo MODAL para a outra tela."""
    area_cursor = (0, -1080, 1920, 1080)     # secundario, acima do primario
    area_mae = (0, 0, 1536, 816)

    def fake_area_util(win, ponto=None):
        return area_cursor if ponto is None else area_mae

    monkeypatch.setattr(gui, "_area_util", fake_area_util)

    class FakeWin:
        def __init__(self):
            self.geo = ""

        def update_idletasks(self):
            pass

        def winfo_reqwidth(self):
            return DIALOGO[0]

        def winfo_reqheight(self):
            return DIALOGO[1]

        def geometry(self, g):
            self.geo = g

    class FakeMae:
        winfo_rootx = staticmethod(lambda: MAE[0])
        winfo_rooty = staticmethod(lambda: MAE[1])
        winfo_width = staticmethod(lambda: MAE[2])
        winfo_height = staticmethod(lambda: MAE[3])

    win = FakeWin()
    gui._centralizar_sobre(win, FakeMae())
    x, y = (int(v) for v in win.geo.lstrip("+").replace("+-", "+@-").split("+"))
    assert y > 0, f"dialogo foi parar no monitor de cima: {win.geo}"
    assert (x, y) == gui._encaixar_dialogo(*DIALOGO, MAE, area_mae)


def test_area_util_do_ponto_contem_o_ponto():
    """A area devolvida para um ponto e a do monitor que contem aquele ponto."""
    class DummyWin:
        winfo_screenwidth = staticmethod(lambda: 1920)
        winfo_screenheight = staticmethod(lambda: 1080)

    w, h = DummyWin.winfo_screenwidth(), DummyWin.winfo_screenheight()
    px, py = w // 2, h // 2
    ax, ay, aw, ah = gui._area_util(DummyWin(), ponto=(px, py))
    assert ax <= px < ax + aw and ay <= py < ay + ah
    assert aw > 0 and ah > 0
