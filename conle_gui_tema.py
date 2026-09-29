# -*- coding: utf-8 -*-
"""Tema visual do Gerador para GUIs Tkinter, sem dependências de fluxo.

Cada projeto mantém uma cópia local para que seus atalhos abram de forma
independente dos demais projetos.
"""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk

VERDE = "#3E6F30"
VERDE_ESCURO = "#2c4f22"
CINZA = "#f4f5f2"


def _area_util(root):
    """Usa a área de trabalho do monitor do cursor quando disponível."""
    try:
        import ctypes
        from ctypes import wintypes

        class MonitorInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        user = ctypes.windll.user32
        ponto = wintypes.POINT()
        user.GetCursorPos(ctypes.byref(ponto))
        user.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        user.MonitorFromPoint.restype = wintypes.HANDLE
        monitor = user.MonitorFromPoint(ponto, 2)
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(info)
        user.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
        if user.GetMonitorInfoW(monitor, ctypes.byref(info)):
            r = info.rcWork
            return r.left, r.top, r.right - r.left, r.bottom - r.top
    except (AttributeError, OSError, ValueError):
        pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def _decoracao_janela():
    """Estima bordas e barra de título nas mesmas unidades usadas pelo Tk."""
    try:
        import ctypes

        user = ctypes.windll.user32
        borda = user.GetSystemMetrics(32) + user.GetSystemMetrics(92)
        titulo = user.GetSystemMetrics(4)
        return 2 * borda, 2 * borda + titulo
    except (AttributeError, OSError, ValueError):
        return 0, 0


def _posicao_centralizada(area, largura, altura, decoracao=(0, 0)):
    esquerda, topo, disponivel_w, disponivel_h = area
    moldura_w, moldura_h = decoracao
    w = min(int(largura), max(1, disponivel_w - moldura_w - 32))
    h = min(int(altura), max(1, disponivel_h - moldura_h - 32))
    x = esquerda + (disponivel_w - w - moldura_w) // 2
    y = topo + (disponivel_h - h - moldura_h) // 2
    return w, h, x, y


def geometria_centralizada(area, largura, altura, decoracao=(0, 0)):
    """Calcula a geometria inicial no centro da área útil de um monitor."""
    w, h, x, y = _posicao_centralizada(area, largura, altura, decoracao)
    return f"{w}x{h}{x:+d}{y:+d}"


def centralizar_janela(root, largura, altura):
    """Abre a janela no centro do monitor do cursor, fora da barra de tarefas."""
    w, h, x, y = _posicao_centralizada(
        _area_util(root), largura, altura, _decoracao_janela())
    root.geometry(f"{w}x{h}{x:+d}{y:+d}")
    return w, h


def aplicar_tema(root, titulo, icone=None, largura=940, altura=850):
    root.title(titulo)
    w, h = centralizar_janela(root, largura, altura)
    root.minsize(min(620, w), min(460, h))
    root.configure(bg=CINZA)
    if icone and Path(icone).is_file():
        try:
            root.iconbitmap(str(icone))
        except tk.TclError:
            pass
    return aplicar_estilo(root)


def aplicar_estilo(root):
    """Aplica as cores e fontes do Gerador sem alterar título ou tamanho."""
    root.configure(bg=CINZA)
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    st.configure("TFrame", background=CINZA)
    st.configure("TLabel", background=CINZA, font=("Segoe UI", 10))
    st.configure("Cab.TLabel", background=CINZA, foreground=VERDE_ESCURO,
                 font=("Segoe UI Semibold", 16))
    st.configure("Sub.TLabel", background=CINZA, foreground="#555", font=("Segoe UI", 9))
    st.configure("TLabelframe", background=CINZA)
    st.configure("TLabelframe.Label", background=CINZA, foreground=VERDE_ESCURO,
                 font=("Segoe UI Semibold", 10))
    st.configure("Sec.TLabelframe", background=CINZA)
    st.configure("Sec.TLabelframe.Label", background=CINZA, foreground=VERDE_ESCURO,
                 font=("Segoe UI Semibold", 10))
    for nome in ("TCheckbutton", "TRadiobutton"):
        st.configure(nome, background=CINZA, font=("Segoe UI", 10))
    st.configure("TButton", font=("Segoe UI", 9), padding=4)
    for nome in ("Gerar.TButton", "Acao.TButton"):
        st.configure(nome, font=("Segoe UI Semibold", 11), padding=8)
    st.configure("TNotebook", background=CINZA)
    st.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=(12, 6))
    st.configure("TEntry", font=("Segoe UI", 10))
    st.configure("TCombobox", font=("Segoe UI", 10))
    st.configure("Treeview", font=("Segoe UI", 9), rowheight=24)
    st.configure("Treeview.Heading", font=("Segoe UI Semibold", 9))
    return st


class Tooltip:
    """Ajuda contextual discreta, fechada ao sair, clicar ou destruir o controle."""

    _ativo = None

    def __init__(self, widget, texto, espera=450):
        self.widget = widget
        self.texto = texto
        self.espera = espera
        self.janela = None
        self.agendado = None
        self.vigia = None
        widget.bind("<Enter>", self._agendar, add="+")
        widget.bind("<Leave>", self._fechar, add="+")
        widget.bind("<ButtonPress>", self._fechar, add="+")
        widget.bind("<Destroy>", self._fechar, add="+")
        widget.bind("<Unmap>", self._fechar, add="+")

    def _agendar(self, _evento=None):
        self._cancelar()
        self.agendado = self.widget.after(self.espera, self._mostrar)

    def _cancelar(self):
        if self.agendado is not None:
            try:
                self.widget.after_cancel(self.agendado)
            except tk.TclError:
                pass
            self.agendado = None

    def _mostrar(self):
        self.agendado = None
        if not self.texto or not self.widget.winfo_exists():
            return
        if Tooltip._ativo is not None and Tooltip._ativo is not self:
            Tooltip._ativo._fechar()
        self.janela = tk.Toplevel(self.widget)
        self.janela.wm_overrideredirect(True)
        tk.Label(self.janela, text=self.texto, justify="left", wraplength=420,
                 bg="#fffbe6", fg="#333333", relief="solid", borderwidth=1,
                 font=("Segoe UI", 9), padx=8, pady=6).pack()
        self.janela.update_idletasks()
        x = self.widget.winfo_pointerx() + 14
        y = self.widget.winfo_pointery() + 18
        ax, ay, aw, ah = _area_util(self.widget)
        x = min(x, ax + aw - self.janela.winfo_width() - 8)
        y = min(y, ay + ah - self.janela.winfo_height() - 8)
        self.janela.wm_geometry(f"{max(x, ax + 8):+d}{max(y, ay + 8):+d}")
        Tooltip._ativo = self
        self.vigia = self.widget.after(250, self._vigiar)

    def _vigiar(self):
        self.vigia = None
        try:
            w = self.widget
            px, py = w.winfo_pointerx(), w.winfo_pointery()
            dentro = (w.winfo_viewable() and w.focus_displayof() is not None
                      and w.winfo_rootx() <= px < w.winfo_rootx() + w.winfo_width()
                      and w.winfo_rooty() <= py < w.winfo_rooty() + w.winfo_height())
        except tk.TclError:
            dentro = False
        if not dentro:
            self._fechar()
        else:
            self.vigia = self.widget.after(250, self._vigiar)

    def _fechar(self, _evento=None):
        self._cancelar()
        if self.vigia is not None:
            try:
                self.widget.after_cancel(self.vigia)
            except tk.TclError:
                pass
            self.vigia = None
        if self.janela is not None:
            self.janela.destroy()
            self.janela = None
        if Tooltip._ativo is self:
            Tooltip._ativo = None


def dica(widget, texto):
    """Anexa uma explicação do efeito do controle e devolve o widget."""
    widget._conle_tooltip = Tooltip(widget, texto)
    return widget


def dicas_por_texto(root, textos):
    """Anexa dicas a botões/opções pelo rótulo, inclusive em abas aninhadas."""
    def percorrer(widget):
        for filho in widget.winfo_children():
            try:
                rotulo = filho.cget("text")
            except tk.TclError:
                rotulo = ""
            if rotulo in textos and not hasattr(filho, "_conle_tooltip"):
                dica(filho, textos[rotulo])
            percorrer(filho)

    percorrer(root)


def ajuda(parent, texto):
    label = ttk.Label(parent, text=texto, style="Sub.TLabel", justify="left", wraplength=500)
    label.pack(anchor="w", fill="x", pady=(3, 8))
    label.bind("<Configure>", lambda e: label.configure(wraplength=max(120, e.width)))
    return label


def cabecalho(root, titulo, subtitulo):
    topo = ttk.Frame(root, padding=(18, 12, 18, 4))
    topo.pack(fill="x")
    label = ttk.Label(topo, text=titulo, style="Cab.TLabel", wraplength=700)
    label.pack(anchor="w", fill="x")
    label.bind("<Configure>", lambda e: label.configure(wraplength=max(120, e.width)))
    ajuda(topo, subtitulo)
    return topo


def aba_rolavel(notebook, titulo):
    aba = ttk.Frame(notebook)
    notebook.add(aba, text=titulo)
    canvas = tk.Canvas(aba, bg=CINZA, highlightthickness=0, yscrollincrement=24)
    barra = ttk.Scrollbar(aba, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=barra.set)
    barra.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    corpo = ttk.Frame(canvas, padding=(14, 12, 14, 16))
    janela = canvas.create_window((0, 0), window=corpo, anchor="nw")
    corpo.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(janela, width=e.width))
    if not hasattr(notebook, "_conle_canvases"):
        notebook._conle_canvases = {}

        def roda(evento):
            try:
                if evento.widget.winfo_class() in ("Text", "Listbox", "TCombobox", "Treeview"):
                    return
                ativo = notebook._conle_canvases.get(notebook.select())
                if ativo is None or ativo.yview() == (0.0, 1.0):
                    return
                num = getattr(evento, "num", None)
                passos = (-1 if num == 4 else 1) if num in (4, 5) else -int(evento.delta / 120)
                if passos:
                    ativo.yview_scroll(passos, "units")
            except tk.TclError:
                pass

        for evento in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            notebook.winfo_toplevel().bind(evento, roda, add="+")
    notebook._conle_canvases[str(aba)] = canvas
    return aba, corpo
