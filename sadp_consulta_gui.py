"""Consulta SADP — TSE (Acompanhamento Processual): GUI independente para garimpar processos.

Digite um número curto (ex.: "23287" ou "232-87") ou um CNJ-20 e veja TODOS os processos
que o SADP Push (https://sadppush.tse.jus.br/sadpPush/, publico, SEM captcha) retorna para
aquele numero — inclusive os de OUTRAS origens (homonimos) e os recursos relacionados ao
mesmo CNJ — para identificar o caso certo.

As partes/advogados/relator de cada linha sao carregadas automaticamente em segundo plano
(uma coluna "Partes" + um painel de detalhe). Informe "Local" (municipio/UF) e/ou "Ano" para
destacar o melhor palpite (heuristica de casamento por municipio+UF+ano). Independente do
fluxo "TSE YouTube > Notion" — reusa apenas a camada de consulta de sadp_lookup.py.

Rode:  .venv-win\\Scripts\\python.exe sadp_consulta_gui.py
"""
from __future__ import annotations

import re
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox

import sadp_lookup as sadp

DETALHE_URL = sadp.SADP_BASE + "ExibirDadosProcesso.do?comboTribunal=tse&nprot={nprot}"

# cores das tags (precedencia: palpite > realce > zebra-por-grupo)
COR_PALPITE = "#bfe3c0"   # verde — melhor palpite (municipio/UF/ano)
COR_REALCE = "#fff3a3"    # amarelo — bate o termo do campo "Realcar"
COR_GRP_A = "#ffffff"
COR_GRP_B = "#eef3f8"


def _campos_da_linha(rec: dict) -> dict:
    """Extrai os campos de exibicao de um registro do parse_results.

    A tabela do SADP tem 7 colunas estaveis (a 1a costuma vir vazia):
    [_, Protocolo, Origem, Situacao, Identificacao, Numeracao Unica, Natureza].
    Usamos posicao a partir do fim (robusto a 1a celula '' ou '&nbsp;') e caimos
    nos campos ja parseados por sadp_lookup quando a linha vier curta.
    """
    cells = rec.get("cells", []) or []
    cnj = rec.get("cnj", "") or ""
    if len(cells) >= 6:
        protocolo, origem, situacao = cells[-6], cells[-5], cells[-4]
        ident, num_unico, natureza = cells[-3], cells[-2], cells[-1]
    else:
        protocolo, natureza, num_unico = "", "", cnj
        origem = rec.get("origem", "")
        situacao = rec.get("situacao", "")
        ident = rec.get("identificacao", "")
    return {
        "protocolo": protocolo,
        "origem": origem,
        "situacao": situacao,
        "ident": ident,
        "cnj": cnj,
        "num_unico": num_unico,
        "natureza": natureza,
        "nprot": rec.get("nprot", "") or "",
        "rec": rec,
    }


class SadpConsultaApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.session: "sadp.requests.Session | None" = None
        self._net_lock = threading.Lock()
        self._search_seq = 0          # invalida buscas/cargas obsoletas
        self.rows: dict[str, dict] = {}   # iid -> campos + searchtext + grupo
        self.detalhes: dict[str, dict] = {}  # nprot -> detalhe (cache)
        self.match_iids: set[str] = set()
        self.filtro_termo = ""
        root.title("Consulta SADP — TSE (Acompanhamento Processual)")
        root.geometry("1180x680")
        root.minsize(880, 480)
        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        topo = ttk.Frame(self.root, padding=(10, 8, 10, 4))
        topo.pack(fill="x")
        ttk.Label(topo, text="Número:").grid(row=0, column=0, sticky="w")
        self.var_num = tk.StringVar()
        e_num = ttk.Entry(topo, textvariable=self.var_num, width=18)
        e_num.grid(row=0, column=1, padx=(4, 14))
        e_num.focus_set()
        ttk.Label(topo, text="Local (município/UF):").grid(row=0, column=2, sticky="w")
        self.var_local = tk.StringVar()
        e_local = ttk.Entry(topo, textvariable=self.var_local, width=26)
        e_local.grid(row=0, column=3, padx=(4, 14))
        ttk.Label(topo, text="Ano:").grid(row=0, column=4, sticky="w")
        self.var_ano = tk.StringVar()
        e_ano = ttk.Entry(topo, textvariable=self.var_ano, width=7)
        e_ano.grid(row=0, column=5, padx=(4, 14))
        self.btn_buscar = ttk.Button(topo, text="Buscar", command=self.buscar)
        self.btn_buscar.grid(row=0, column=6)
        ttk.Label(
            topo,
            text='Ex.: "23287" ou "232-87" (número curto) ou um CNJ-20. Local/Ano realçam o melhor palpite.',
            foreground="#666",
        ).grid(row=1, column=0, columnspan=7, sticky="w", pady=(4, 0))
        e_num.bind("<Return>", lambda _e: self.buscar())
        # Local/Ano: recomputam o melhor palpite AO VIVO (sem refazer a consulta de rede),
        # pois best_match opera sobre os registros já carregados em self.rows.
        for e in (e_local, e_ano):
            e.bind("<KeyRelease>", self._on_palpite_change)
            e.bind("<Return>", self._on_palpite_change)

        filtro = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        filtro.pack(fill="x")
        ttk.Label(filtro, text="Realçar (parte/origem/classe):").pack(side="left")
        self.var_filtro = tk.StringVar()
        e_filtro = ttk.Entry(filtro, textvariable=self.var_filtro)
        e_filtro.pack(side="left", fill="x", expand=True, padx=(6, 0))
        e_filtro.bind("<KeyRelease>", self._on_filtro)

        paned = ttk.PanedWindow(self.root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        # tabela de resultados
        tree_wrap = ttk.Frame(paned)
        cols = ("ident", "origem", "situacao", "cnj", "protocolo", "dje", "partes")
        titulos = {
            "ident": "Identificação",
            "origem": "Origem",
            "situacao": "Situação",
            "cnj": "CNJ-20 / Nº",
            "protocolo": "Protocolo",
            "dje": "DJe (publicação)",
            "partes": "Partes",
        }
        larguras = {"ident": 160, "origem": 165, "situacao": 165, "cnj": 170,
                    "protocolo": 90, "dje": 195, "partes": 280}
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=titulos[c])
            self.tree.column(c, width=larguras[c], anchor="w", stretch=(c in ("origem", "partes")))
        vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_wrap, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_wrap.rowconfigure(0, weight=1)
        tree_wrap.columnconfigure(0, weight=1)
        paned.add(tree_wrap, weight=3)
        self.tree.tag_configure("palpite", background=COR_PALPITE)
        self.tree.tag_configure("realce", background=COR_REALCE)
        self.tree.tag_configure("grp_a", background=COR_GRP_A)
        self.tree.tag_configure("grp_b", background=COR_GRP_B)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda _e: self.abrir_no_sadp())
        self.tree.bind("<Button-3>", self._popup_menu)

        # painel de detalhe
        det_wrap = ttk.LabelFrame(paned, text="Detalhe do processo (partes, advogados, relator, assunto, fase, publicações no DJe)")
        self.txt = tk.Text(det_wrap, height=10, wrap="word", state="disabled", font=("Segoe UI", 9))
        dsb = ttk.Scrollbar(det_wrap, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=dsb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        dsb.pack(side="right", fill="y")
        self.txt.tag_configure("lbl", font=("Segoe UI", 9, "bold"), foreground="#1a3c6e")
        self.txt.tag_configure("h", font=("Segoe UI", 10, "bold"))
        self.txt.tag_configure("link", foreground="#0a58ca", underline=True)
        self.txt.tag_bind("link", "<Button-1>", lambda _e: self._abrir_link_atual())
        self.txt.tag_bind("link", "<Enter>", lambda _e: self.txt.configure(cursor="hand2"))
        self.txt.tag_bind("link", "<Leave>", lambda _e: self.txt.configure(cursor=""))
        self.txt.tag_configure("djelink", foreground="#0a58ca", underline=True)
        self.txt.tag_bind("djelink", "<Button-1>", lambda _e: webbrowser.open(sadp.DJE_CONSULTA_URL))
        self.txt.tag_bind("djelink", "<Enter>", lambda _e: self.txt.configure(cursor="hand2"))
        self.txt.tag_bind("djelink", "<Leave>", lambda _e: self.txt.configure(cursor=""))
        self._link_atual = ""
        paned.add(det_wrap, weight=2)

        # barra inferior
        barra = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        barra.pack(fill="x")
        ttk.Button(barra, text="Abrir no SADP", command=self.abrir_no_sadp).pack(side="left")
        ttk.Button(barra, text="Copiar CNJ", command=self.copiar_cnj).pack(side="left", padx=(6, 0))
        ttk.Button(barra, text="Copiar linha", command=self.copiar_linha).pack(side="left", padx=(6, 0))
        ttk.Button(barra, text="Copiar link", command=self.copiar_link).pack(side="left", padx=(6, 0))
        ttk.Separator(barra, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(barra, text="Abrir consulta DJe", command=self.abrir_consulta_dje).pack(side="left")
        ttk.Button(barra, text="Copiar ref. DJe", command=self.copiar_ref_dje).pack(side="left", padx=(6, 0))
        self.var_status = tk.StringVar(value="Pronto.")
        ttk.Label(barra, textvariable=self.var_status, foreground="#444").pack(side="right")

        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="Abrir no SADP", command=self.abrir_no_sadp)
        self.menu.add_command(label="Copiar CNJ", command=self.copiar_cnj)
        self.menu.add_command(label="Copiar linha", command=self.copiar_linha)
        self.menu.add_command(label="Copiar link", command=self.copiar_link)
        self.menu.add_separator()
        self.menu.add_command(label="Abrir consulta DJe", command=self.abrir_consulta_dje)
        self.menu.add_command(label="Copiar ref. DJe", command=self.copiar_ref_dje)

    # ------------------------------------------------------------- busca
    def buscar(self) -> None:
        raw = self.var_num.get()
        dig = re.sub(r"\D", "", raw or "")
        if not dig:
            messagebox.showwarning("Consulta SADP", "Informe um número (ex.: 23287 ou 232-87).")
            return
        self._search_seq += 1
        seq = self._search_seq
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        self.match_iids.clear()
        self._set_detalhe_texto("")
        self.btn_buscar.configure(state="disabled")
        self.var_status.set(f"Consultando o SADP para {dig}…")
        threading.Thread(target=self._do_search, args=(seq, dig), daemon=True).start()

    def _ensure_session(self):
        with self._net_lock:
            if self.session is None:
                self.session = sadp.make_session()
            return self.session

    def _do_search(self, seq: int, dig: str) -> None:
        try:
            sess = self._ensure_session()
            # requests.Session NÃO é thread-safe e a busca compartilha a sessão com os fetches de
            # detalhe; o lock evita requisições concorrentes (e trocar o processo fixado na sessão).
            with self._net_lock:
                results = sadp.search_numunico(sess, dig) if len(dig) >= 20 else sadp.search_number(sess, dig)
            err = None
        except Exception as exc:  # rede/timeout
            results, err = [], str(exc)
        self.root.after(0, self._on_search_done, seq, results, err)

    def _on_search_done(self, seq: int, results: list[dict], err) -> None:
        if seq != self._search_seq:
            return
        self.btn_buscar.configure(state="normal")
        if err:
            self.var_status.set("Falha na consulta.")
            messagebox.showerror("Consulta SADP", f"Não foi possível consultar o SADP:\n{err}")
            return
        if not results:
            self.var_status.set("Nenhum resultado.")
            return
        prev_key = None
        grp = 0
        for rec in results:
            f = _campos_da_linha(rec)
            cnj_disp = f["cnj"] or f["num_unico"]
            key = f["cnj"] or ("id:" + sadp._fold(f["ident"]) if f["ident"] else "og:" + sadp._fold(f["origem"]))
            if key != prev_key:
                grp += 1
                prev_key = key
            iid = self.tree.insert(
                "", "end",
                values=(f["ident"], f["origem"], f["situacao"], cnj_disp, f["protocolo"], "⏳", "⏳ carregando…"),
            )
            f["grupo"] = grp
            f["partes_txt"] = ""
            f["publicacoes"] = []
            f["searchtext"] = sadp._fold(" ".join([f["ident"], f["origem"], f["situacao"], cnj_disp]))
            self.rows[iid] = f
        self._compute_palpite()
        self._recompute_all_tags()
        self.var_status.set(f"{len(results)} resultado(s). Carregando partes/advogados…")
        # carrega primeiro as linhas do melhor palpite, depois as demais
        ordem = sorted(self.rows.keys(), key=lambda i: 0 if i in self.match_iids else 1)
        threading.Thread(target=self._load_detalhes, args=(seq, ordem), daemon=True).start()

    # --------------------------------------------------- detalhes (async)
    def _load_detalhes(self, seq: int, iids: list[str]) -> None:
        import time
        sess = self._ensure_session()
        total = len(iids)
        for i, iid in enumerate(iids, 1):
            if seq != self._search_seq:
                return
            f = self.rows.get(iid)
            nprot = f["nprot"] if f else ""
            det = self.detalhes.get(nprot)
            if det is None:
                try:
                    # GET+POST atômicos: o GET fixa o processo na sessão e o POST lê o andamento dele;
                    # o lock evita que uma re-busca concorrente intercale e troque o processo fixado.
                    with self._net_lock:
                        det = sadp.fetch_detail_e_publicacoes(sess, nprot) if nprot else {}
                except Exception:
                    det = {}
                # só cacheia detalhe válido (flag 'ok' = GET 200); falha transitória não fica grudada
                if nprot and det.get("ok"):
                    self.detalhes[nprot] = det
            self.root.after(0, self._apply_detalhe, seq, iid, det, i, total)
            time.sleep(0.35)  # cortesia com o SADP
        self.root.after(0, lambda: self.var_status.set(f"{total} resultado(s) — partes carregadas."))

    def _apply_detalhe(self, seq: int, iid: str, det: dict, i: int, total: int) -> None:
        if seq != self._search_seq or iid not in self.rows:
            return
        partes = det.get("partes") or []
        resumo = "; ".join(partes) if partes else ("(sem partes no SADP)" if det else "(detalhe indisponível)")
        self.tree.set(iid, "partes", resumo)
        f = self.rows[iid]
        f["partes_txt"] = "; ".join(partes)
        f["det"] = det
        pubs = det.get("publicacoes_dje") or []
        f["publicacoes"] = pubs
        self.tree.set(iid, "dje", self._resumo_dje(pubs) if det else "—")
        f["searchtext"] = sadp._fold(
            " ".join([f["ident"], f["origem"], f["situacao"], f["cnj"] or f["num_unico"],
                      f["partes_txt"], " ".join(det.get("advogados") or []), det.get("relator", ""),
                      " ".join(self._fmt_pub(p) for p in pubs)])
        )
        self._recompute_row_tag(iid)
        if i < total:
            self.var_status.set(f"{total} resultado(s) — carregando partes {i}/{total}…")
        if iid in self.tree.selection():
            self._mostrar_detalhe(iid)

    # ------------------------------------------------------ realce/tags
    def _compute_palpite(self) -> None:
        self.match_iids.clear()
        local = self.var_local.get().strip()
        ano = re.sub(r"\D", "", self.var_ano.get() or "")
        if not (local or ano):
            return
        ufm = re.search(r"[-/\s]([A-Za-z]{2})\s*$", local)
        uf = ufm.group(1) if ufm else ""
        muni = local[:ufm.start()].strip() if uf else local.strip()  # corta exatamente a UF casada
        cands = [f["rec"] for f in self.rows.values()]
        match = sadp.best_match(cands, muni, uf, ano, "")
        alvo = match.get("cnj") if match else ""
        if alvo:
            for iid, f in self.rows.items():
                if f["cnj"] == alvo:
                    self.match_iids.add(iid)
        elif muni and len(sadp._fold(muni)) > 2:  # fallback conservador: município (+UF/ano) na origem
            mf, uf_f = sadp._fold(muni), sadp._fold(uf)
            cands_fb = []
            for iid, f in self.rows.items():
                ogf = sadp._fold(f["origem"])
                if mf not in ogf:
                    continue
                if uf_f and not ogf.endswith("-" + uf_f):  # respeita a UF informada
                    continue
                if ano and f["cnj"] and f["cnj"][11:15] != ano:  # respeita o ano informado
                    continue
                cands_fb.append(iid)
            # só destaca se não houver ambiguidade entre origens distintas (homônimos)
            if cands_fb and len({sadp._fold(self.rows[i]["origem"]) for i in cands_fb}) == 1:
                self.match_iids.update(cands_fb)

    def _tag_da_linha(self, iid: str) -> str:
        if iid in self.match_iids:
            return "palpite"
        if self.filtro_termo and self.filtro_termo in self.rows[iid]["searchtext"]:
            return "realce"
        return "grp_a" if self.rows[iid]["grupo"] % 2 else "grp_b"

    def _recompute_row_tag(self, iid: str) -> None:
        self.tree.item(iid, tags=(self._tag_da_linha(iid),))

    def _recompute_all_tags(self) -> None:
        for iid in self.rows:
            self._recompute_row_tag(iid)

    def _on_filtro(self, _evt=None) -> None:
        self.filtro_termo = sadp._fold(self.var_filtro.get())
        self._recompute_all_tags()

    def _on_palpite_change(self, _evt=None) -> None:
        # recálculo leve do melhor palpite ao editar Local/Ano — sem tocar na rede
        if not self.rows:
            return
        self._compute_palpite()
        self._recompute_all_tags()

    # ------------------------------------------------------------- DJe
    @staticmethod
    def _fmt_pub(p: dict) -> str:
        """Resumo de uma publicação: 'Acórdão 27/10/2017 · ed.209 · p.74/75'."""
        cab = p.get("ato") or ("Disponibilização" if p.get("evento") == "disponibilizacao" else "Publicação")
        partes = [cab, p.get("data", "")]
        extra = []
        if p.get("edicao"):
            extra.append("ed." + p["edicao"])
        if p.get("pagina"):
            extra.append("p." + p["pagina"])
        s = " ".join(x for x in partes if x)
        return s + (" · " + " · ".join(extra) if extra else "")

    @classmethod
    def _fmt_pub_longo(cls, p: dict) -> str:
        s = cls._fmt_pub(p)
        return s + (f" (ato de {p['data_ato']})" if p.get("data_ato") else "")

    @staticmethod
    def _pub_preferida(pubs: list) -> "dict | None":
        """Publicação a destacar na coluna: o acórdão de mérito (o mais ANTIGO, tipicamente o do
        recurso principal) em vez do acórdão de embargos/recursos posteriores; sem acórdão, a
        publicação mais recente; sem publicação, o evento mais recente. (pubs vem desc por data.)"""
        acordaos = [p for p in pubs if p.get("evento") == "publicacao" and "acord" in sadp._fold(p.get("ato", ""))]
        if acordaos:
            return acordaos[-1]  # mais antigo
        publicacoes = [p for p in pubs if p.get("evento") == "publicacao"]
        return (publicacoes or pubs or [None])[0]

    def _resumo_dje(self, pubs: list) -> str:
        pref = self._pub_preferida(pubs)
        if not pref:
            return "—"
        total = len(pubs)  # (+N) reflete TODOS os eventos listados no painel, não só publicações
        return self._fmt_pub(pref) + (f"  (+{total - 1})" if total > 1 else "")

    # --------------------------------------------------------- detalhe
    def _on_select(self, _evt=None) -> None:
        sel = self.tree.selection()
        if sel:
            self._mostrar_detalhe(sel[0])

    def _mostrar_detalhe(self, iid: str) -> None:
        f = self.rows.get(iid)
        if not f:
            return
        det = f.get("det")
        linhas: list[tuple[str, str]] = []
        cabec = f"{f['ident'] or '(sem identificação)'}    —    {f['origem']}"
        self._link_atual = DETALHE_URL.format(nprot=f["nprot"]) if f["nprot"] else ""
        if det is None:
            corpo = "Carregando detalhe…"
        elif not det:
            corpo = "Detalhe indisponível no SADP para este protocolo."
        else:
            linhas = [
                ("CNJ-20", det.get("cnj") or f["cnj"] or f["num_unico"]),
                ("Município", det.get("municipio") or f["origem"]),
                ("Situação", f["situacao"]),
                ("Protocolo", f["protocolo"]),
                ("Relator", det.get("relator", "")),
                ("Assunto", det.get("assunto", "")),
                ("Fase atual", det.get("fase", "")),
                ("Partes", "\n   • " + "\n   • ".join(det.get("partes") or []) if det.get("partes") else "(nenhuma)"),
                ("Advogados", "\n   • " + "\n   • ".join(det.get("advogados") or []) if det.get("advogados") else "(nenhum)"),
            ]
            corpo = None
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("end", cabec + "\n", "h")
        if corpo is not None:
            self.txt.insert("end", "\n" + corpo + "\n")
        else:
            self.txt.insert("end", "\n")
            for lbl, val in linhas:
                self.txt.insert("end", f"{lbl}: ", "lbl")
                self.txt.insert("end", f"{val}\n")
        pubs = f.get("publicacoes") or []
        if pubs:
            self.txt.insert("end", "\nPublicações no DJe (Diário da Justiça Eletrônico):\n", "lbl")
            for p in pubs:
                self.txt.insert("end", "   • " + self._fmt_pub_longo(p) + "   ")
                self.txt.insert("end", "[abrir consulta DJe]\n", "djelink")
        elif det:
            self.txt.insert("end", "\nPublicações no DJe: ", "lbl")
            self.txt.insert("end", "nenhuma registrada neste protocolo.\n")
        if self._link_atual:
            self.txt.insert("end", "\nLink (SADP): ", "lbl")
            self.txt.insert("end", self._link_atual + "\n", "link")
        self.txt.configure(state="disabled")

    def _set_detalhe_texto(self, txt: str) -> None:
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        if txt:
            self.txt.insert("end", txt)
        self.txt.configure(state="disabled")

    # ----------------------------------------------------------- ações
    def _linha_selecionada(self) -> dict | None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Consulta SADP", "Selecione uma linha primeiro.")
            return None
        return self.rows.get(sel[0])

    def abrir_no_sadp(self) -> None:
        f = self._linha_selecionada()
        if not f:
            return
        if not f["nprot"]:
            messagebox.showinfo("Consulta SADP", "Esta linha não tem protocolo para abrir.")
            return
        webbrowser.open(DETALHE_URL.format(nprot=f["nprot"]))

    def copiar_cnj(self) -> None:
        f = self._linha_selecionada()
        if not f:
            return
        valor = f["cnj"] or f["num_unico"]
        self.root.clipboard_clear()
        self.root.clipboard_append(valor)
        self.var_status.set(f"Copiado: {valor}")

    def copiar_linha(self) -> None:
        f = self._linha_selecionada()
        if not f:
            return
        pref = self._pub_preferida(f.get("publicacoes") or [])
        valor = "\t".join([f["ident"], f["origem"], f["situacao"], f["cnj"] or f["num_unico"],
                           f["protocolo"], (self._fmt_pub(pref) if pref else ""), f.get("partes_txt", "")])
        self.root.clipboard_clear()
        self.root.clipboard_append(valor)
        self.var_status.set("Linha copiada para a área de transferência.")

    def abrir_consulta_dje(self) -> None:
        webbrowser.open(sadp.DJE_CONSULTA_URL)
        self.var_status.set("Abrindo a consulta oficial do DJe — informe a data/edição mostrada na coluna DJe.")

    def copiar_ref_dje(self) -> None:
        f = self._linha_selecionada()
        if not f:
            return
        pref = self._pub_preferida(f.get("publicacoes") or [])
        if not pref:
            messagebox.showinfo("Consulta SADP", "Esta linha não tem publicação no DJe registrada no SADP.")
            return
        cnj = f["cnj"] or f["num_unico"]
        cauda = " ".join(x for x in (f["ident"], cnj) if x)  # evita espaço duplo quando não há identificação
        ref = f"DJe — {self._fmt_pub_longo(pref)}" + (f" — {cauda}" if cauda else "")
        self.root.clipboard_clear()
        self.root.clipboard_append(ref)
        self.var_status.set("Referência do DJe copiada: " + ref[:55] + ("…" if len(ref) > 55 else ""))

    def copiar_link(self) -> None:
        f = self._linha_selecionada()
        if not f:
            return
        if not f["nprot"]:
            messagebox.showinfo("Consulta SADP", "Esta linha não tem protocolo (sem link no SADP).")
            return
        url = DETALHE_URL.format(nprot=f["nprot"])
        self.root.clipboard_clear()
        self.root.clipboard_append(url)
        self.var_status.set("Link copiado para a área de transferência.")

    def _abrir_link_atual(self) -> None:
        if self._link_atual:
            webbrowser.open(self._link_atual)

    def _popup_menu(self, evt) -> None:
        iid = self.tree.identify_row(evt.y)
        if iid:
            self.tree.selection_set(iid)
            try:
                self.menu.tk_popup(evt.x_root, evt.y_root)
            finally:
                self.menu.grab_release()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")  # tema nativo do Windows quando disponível
    except tk.TclError:
        pass
    SadpConsultaApp(root)
    root.mainloop()


if __name__ == "__main__":
    # Lançado pelo atalho via pythonw.exe (sem console): em erro de inicialização, mostrar uma
    # janela de erro e gravar um log ao lado do script, em vez de falhar silenciosamente.
    try:
        main()
    except Exception:
        import traceback, os
        msg = traceback.format_exc()
        log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sadp_consulta_gui_erro.log")
        try:
            with open(log, "w", encoding="utf-8") as fh:
                fh.write(msg)
        except Exception:
            log = "(não foi possível gravar o log)"
        try:
            from tkinter import messagebox
            messagebox.showerror("Consulta SADP — erro ao iniciar", f"{msg}\n\nLog: {log}")
        except Exception:
            pass
        raise
