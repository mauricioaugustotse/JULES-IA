#!/usr/bin/env python3
"""Driver auto-dirigido (CDP) para padronizar etiquetas em `default` e excluir órfãs,
SEM depender da caixa de busca (que não persiste no painel) nem do foco do usuário.

Estratégia: abre o painel de Opções via CDP, e percorre a lista (rolando com a roda do
mouse) clicando em cada etiqueta-ALVO -> abre o submenu da opção -> "Padrão" (recolor)
ou "Excluir" (órfã). Como cliques via CDP não tiram o foco da janela, o popover não
fecha durante a execução.

Uso:
    python notion_labels_drive.py --property partes --limit 3            # teste recolor
    python notion_labels_drive.py --property partes                      # recolor total
    python notion_labels_drive.py --property partes --apply-deletes      # + exclui órfãs
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("artifacts") / "notion_labels_default"
CDP_URL = "http://127.0.0.1:9222"
DEFAULT_COLOR_LABELS = ("Padrão", "Padrao", "Default")
DELETE_LABELS = ("Excluir", "Remover", "Delete", "Remove")
LOGGER = logging.getLogger("notion_labels_drive")

VIS = """
const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();
const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};
"""


def load_targets(prop: str, apply_deletes: bool, only_deletes: bool = False) -> list[dict[str, str]]:
    path = OUTPUT_DIR / f"{prop}_plano_manual.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
    out = []
    for r in rows:
        if r["coluna"].strip().casefold() != prop.casefold():
            continue
        name = r["etiqueta"].strip()
        orphan = r["remover_se_apply"].strip() in {"1", "true", "sim"}
        nondefault = (r["cor_atual"].strip() or "default") != "default"
        if orphan and apply_deletes:
            out.append({"name": name, "action": "delete"})
        elif nondefault and not orphan and not only_deletes:  # --only-deletes: NAO recolore (ex.: composicao)
            out.append({"name": name, "action": "recolor"})
        elif orphan and not apply_deletes:
            continue
    return out


def pick_page(browser):
    for ctx in browser.contexts:
        for pg in ctx.pages:
            if "notion." in pg.url:
                return pg
    return browser.contexts[0].pages[0]


def _topmost_rect(page, text):
    return page.evaluate(
        "({t})=>{" + VIS + "const tt=norm(t);let best=null,top=1e9;const w=document.createTreeWalker(document.body,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||'')!==tt)continue;if(Array.from(el.children).some(c=>vis(c)&&norm(c.innerText||c.textContent||'')===tt))continue;const r=el.getBoundingClientRect();if(r.top<top){top=r.top;best={x:r.left+r.width/2,y:r.top+r.height/2};}}return best;}",
        {"t": text},
    )


def _rect_in_last_menu(page, text):
    return page.evaluate(
        "({t})=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const root=c.length?c[c.length-1]:document.body;const tt=norm(t);const w=document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||'')!==tt)continue;if(Array.from(el.children).some(ch=>vis(ch)&&norm(ch.innerText||ch.textContent||'')===tt))continue;const r=el.getBoundingClientRect();return {x:r.left+Math.min(20,r.width/2),y:r.top+r.height/2};}return null;}",
        {"t": text},
    )


def _options_dialog_info(page):
    return page.evaluate(
        "()=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const opt=[...c].reverse().find(d=>norm(d.innerText||'').includes('Opções'));if(!opt)return null;const r=opt.getBoundingClientRect();return {x:Math.round(r.left),y:Math.round(r.top),w:Math.round(r.width),h:Math.round(r.height)};}"
    )


def _submenu_open(page):
    return bool(page.evaluate(
        "()=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);if(!c.length)return false;const t=norm(c[c.length-1].innerText||'').toLowerCase();return t.includes('cores')||t.includes('excluir');}"
    ))


def _submenu_is_for(page, label) -> bool:
    """O submenu aberto (detalhe da opcao) e da etiqueta `label` (confere pelo campo de
    nome/renomear da opcao). Evita recolorir/excluir a opcao vizinha errada."""
    return bool(page.evaluate(
        "({t})=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);if(!c.length)return false;const sub=c[c.length-1];const tx=norm(sub.innerText||'').toLowerCase();if(!(tx.includes('cores')||tx.includes('excluir')))return false;const inp=sub.querySelector('input,textarea');if(!inp)return false;const tt=norm(t);const v=norm(inp.value);const ph=norm(inp.getAttribute('placeholder')||inp.getAttribute('aria-label')||'');return v===tt||ph===tt;}",
        {"t": label},
    ))


def _panel_is_for(page, prop) -> bool:
    """O painel aberto e da coluna `prop` (confere pelo campo 'Nome da propriedade')."""
    return bool(page.evaluate(
        "({p})=>{const norm=(v)=>String(v||'').replace(/\\s+/g,' ').trim().toLowerCase();const inp=Array.from(document.querySelectorAll('input')).find(e=>norm(e.getAttribute('placeholder')||'')==='nome da propriedade');return inp?norm(inp.value)===norm(p):false;}",
        {"p": prop},
    ))


def _visible_targets(page, pending: list[str]) -> list[str]:
    """Quais dos `pending` estão visíveis na lista de opções (texto exato)."""
    return page.evaluate(
        "({names})=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const opt=[...c].reverse().find(d=>norm(d.innerText||'').includes('Opções'));if(!opt)return [];const set=new Set(names.map(norm));const found=new Set();const w=document.createTreeWalker(opt,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;const t=norm(el.innerText||el.textContent||'');if(!set.has(t))continue;if(Array.from(el.children).some(ch=>vis(ch)&&norm(ch.innerText||ch.textContent||'')===t))continue;found.add(t);}return [...found];}",
        {"names": pending},
    )


def _exact_present(page, label) -> bool:
    return bool(page.evaluate(
        "({t})=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const opt=[...c].reverse().find(d=>norm(d.innerText||'').includes('Opções'))||document.body;const tt=norm(t);const w=document.createTreeWalker(opt,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||'')!==tt)continue;if(Array.from(el.children).some(ch=>vis(ch)&&norm(ch.innerText||ch.textContent||'')===tt))continue;return true;}return false;}",
        {"t": label},
    ))


# Encontra o input da busca de opções pela PLACEHOLDER, em qualquer lugar do documento
# (sem exigir que esteja no diálogo "Opções" — a estrutura muda apos cada acao).
_FIND_SEARCH_JS = (
    "const _ph=(el)=>String(el.getAttribute('placeholder')||el.getAttribute('data-placeholder')||'').toLowerCase();"
    "const _isSearch=(el)=>{const p=_ph(el);return (p.includes('nova op')||p.includes('opç')||p.includes('opc')||p.includes('digite uma'))&&!p.includes('nome da propriedade');};"
    "const _find=()=>Array.from(document.querySelectorAll('input,textarea')).find(_isSearch)||null;"
)


def _has_search(page) -> bool:
    return bool(page.evaluate("()=>{" + _FIND_SEARCH_JS + "return Boolean(_find());}"))


def _panel_open(page) -> bool:
    """A lista de Opcoes esta aberta (com ou sem a busca revelada)."""
    return bool(page.evaluate(
        "()=>{" + VIS + "return Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis).some(d=>norm(d.innerText||'').includes('Opções'));}"
    ))


def _search_active(page) -> bool:
    """A caixa de busca esta VISIVEL e utilizavel (apos clicar no '+')."""
    return bool(page.evaluate(
        "()=>{" + VIS + _FIND_SEARCH_JS + "const inp=_find();return inp?vis(inp):false;}"
    ))


def _click_plus(page, delay) -> bool:
    """Clica o '+' (aria-label 'Adicionar uma opção') para revelar a busca. O '+' fica
    no topo da lista, que pode estar rolada para muito longe (lista de 50k px) — por
    isso fazemos scrollIntoView ANTES de clicar e usamos as coords pos-scroll."""
    rect = page.evaluate(
        "()=>{" + VIS + "const cand=Array.from(document.querySelectorAll('[aria-label]')).find(e=>{const al=String(e.getAttribute('aria-label')||'').toLowerCase();return al.includes('adicionar')&&(al.includes('op')||al.includes('opç'));});"
        "if(!cand)return null;cand.scrollIntoView({block:'center',inline:'nearest'});const r=cand.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};}"
    )
    if isinstance(rect, dict) and 0 < rect.get("y", -1) < 1400:
        page.mouse.click(rect["x"], rect["y"], delay=30)
        time.sleep(delay)
        return _search_active(page)
    return False


def _target_in_viewport(page, label) -> bool:
    """A opcao-alvo esta VISIVEL na viewport (lista filtrada nela), nao so no DOM."""
    return bool(page.evaluate(
        "({t})=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const opt=[...c].reverse().find(d=>norm(d.innerText||'').includes('Opções'))||document.body;const tt=norm(t);const w=document.createTreeWalker(opt,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||'')!==tt)continue;if(Array.from(el.children).some(ch=>vis(ch)&&norm(ch.innerText||ch.textContent||'')===tt))continue;const r=el.getBoundingClientRect();return r.top>40&&r.top<window.innerHeight-20;}return false;}",
        {"t": label},
    ))


def _confirm_present(page) -> bool:
    """Diálogo de confirmação de exclusão ('Quer mesmo excluir esta opção?')."""
    return bool(page.evaluate(
        "()=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);if(!c.length)return false;const t=norm(c[c.length-1].innerText||'').toLowerCase();return t.includes('quer mesmo')||t.includes('cancelar');}"
    ))


def _submenu_usage(page) -> int:
    val = page.evaluate(
        "()=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);if(!c.length)return -1;const t=norm(c[c.length-1].innerText||'').toLowerCase();const m=t.match(/(\\d+)\\s*(p[aá]gina|registro|page)/);return m?parseInt(m[1],10):-1;}"
    )
    try:
        return int(val)
    except Exception:
        return -1


def _open_option_submenu(page, label, delay):
    """Abre o submenu da opção clicando NA PROPRIA opção (UI atual: clicar a linha abre
    rename+cores+Excluir). Hover antes do clique p/ o '...' renderizar (mouse real)."""
    pts = page.evaluate(
        "({t})=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const opt=[...c].reverse().find(d=>norm(d.innerText||'').includes('Op'))||document.body;const tt=norm(t);let chosen=null;const w=document.createTreeWalker(opt,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||'')!==tt)continue;if(Array.from(el.children).some(ch=>vis(ch)&&norm(ch.innerText||ch.textContent||'')===tt))continue;chosen=el;break;}if(!chosen)return null;chosen.scrollIntoView({block:'center'});const r=chosen.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2,right:r.right,left:r.left};}",
        {"t": label},
    )
    if not isinstance(pts, dict):
        return False
    # clicamos NA opcao exata (achada por texto) -> o submenu (Cores/Excluir) e dela.
    # A UI atual nao tem campo de renomear no submenu, entao _submenu_is_for nao se aplica.
    for cx in (pts["right"] - 12, pts["x"], pts["left"] + 20):
        page.mouse.move(cx, pts["y"])
        time.sleep(0.10)
        page.mouse.click(cx, pts["y"], delay=20)
        # POLLING: prossegue assim que o submenu abrir (em vez de esperar 'delay' fixo).
        for _ in range(int(max(delay, 0.30) / 0.03) + 1):
            if _submenu_open(page):
                return True
            time.sleep(0.03)
    return False


def _click_in_submenu(page, labels) -> bool:
    """Clica um item do submenu por MOUSE REAL (mantem o foco no popover)."""
    for lab in labels:
        rect = page.evaluate(
            "({t})=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const root=c.length?c[c.length-1]:document.body;const tt=norm(t);const w=document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||'')!==tt)continue;if(Array.from(el.children).some(ch=>vis(ch)&&norm(ch.innerText||ch.textContent||'')===tt))continue;const r=el.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};}return null;}",
            {"t": lab},
        )
        if isinstance(rect, dict):
            page.mouse.click(rect["x"], rect["y"], delay=15)
            return True
    return False


def _click_header(page, prop, delay) -> bool:
    """Clica o cabecalho da coluna (rolando-o para a tela se preciso)."""
    rect = page.evaluate(
        "({t})=>{" + VIS + "const tt=norm(t);let best=null,top=1e9;const w=document.createTreeWalker(document.body,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||'')!==tt)continue;if(Array.from(el.children).some(c=>vis(c)&&norm(c.innerText||c.textContent||'')===tt))continue;const r=el.getBoundingClientRect();if(r.top<top){top=r.top;best=el;}}if(!best)return null;best.scrollIntoView({block:'nearest',inline:'center'});const r=best.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};}",
        {"t": prop},
    )
    if isinstance(rect, dict) and 0 < rect.get("y", -1) < 300 and 0 < rect.get("x", -1) < 2400:
        page.mouse.click(rect["x"], rect["y"], delay=30)
        time.sleep(delay)
        return True
    return False


def ensure_panel(page, prop, delay, logger) -> bool:
    """Abre o painel de Opcoes da COLUNA `prop` via CDP (cabecalho -> Editar propriedade),
    verificando que e a coluna certa (nao reaproveita painel de outra coluna)."""
    for attempt in range(6):
        if _panel_open(page) and _panel_is_for(page, prop):
            return True
        for _ in range(3):
            page.keyboard.press("Escape"); time.sleep(0.15)
        if not _click_header(page, prop, delay):
            logger.warning("Cabecalho '%s' nao encontrado (tentativa %s).", prop, attempt + 1)
            time.sleep(0.4); continue
        ed = page.evaluate(
            "()=>{" + VIS + "const tt='editar propriedade';let best=null,top=1e9;const w=document.createTreeWalker(document.body,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||'').toLowerCase()!==tt)continue;if(Array.from(el.children).some(c=>vis(c)&&norm(c.innerText||c.textContent||'').toLowerCase()===tt))continue;const r=el.getBoundingClientRect();if(r.top<top){top=r.top;best={x:r.left+Math.min(20,r.width/2),y:r.top+r.height/2};}}return best;}"
        )
        if isinstance(ed, dict):
            page.mouse.click(ed["x"], ed["y"], delay=30)
            time.sleep(delay + 0.4)
        for _ in range(5):
            if _panel_open(page) and _panel_is_for(page, prop):
                return True
            time.sleep(0.3)
        logger.warning("Abrir painel de '%s' falhou (tentativa %s).", prop, attempt + 1)
    return _panel_open(page) and _panel_is_for(page, prop)


def wait_for_panel(page, timeout_s, logger):
    """Espera (polling, SEM tocar na UI) o usuario abrir a lista de Opcoes."""
    waited = 0
    while waited < timeout_s:
        if _panel_open(page):
            return True
        if waited == 0 or waited % 10 == 0:
            logger.info("Aguardando voce abrir o painel de Opcoes (partes -> Editar propriedade)... %ss/%ss", waited, timeout_s)
        time.sleep(1.0)
        waited += 1
    return _panel_open(page)


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    targets = load_targets(args.property, args.apply_deletes, getattr(args, "only_deletes", False))
    if args.limit:
        targets = targets[: args.limit]
    target_names = [t["name"] for t in targets]
    action_by = {t["name"]: t["action"] for t in targets}
    LOGGER.info("[%s] alvos: %s (recolor=%s, delete=%s)", args.property, len(targets),
                sum(1 for t in targets if t["action"] == "recolor"),
                sum(1 for t in targets if t["action"] == "delete"))
    delay = args.delay_ms / 1000.0
    ck_path = OUTPUT_DIR / f"{args.property}.drive.checkpoint.json"
    done = set(json.loads(ck_path.read_text(encoding="utf-8")).get("done", [])) if ck_path.exists() else set()
    shots = OUTPUT_DIR / "ui_debug"; shots.mkdir(parents=True, exist_ok=True)

    cdp_url = getattr(args, "cdp_url", None) or CDP_URL
    with sync_playwright() as p:
        browser = None
        for tentativa in range(1, 4):  # retry: Edge pode estar abrindo
            try:
                browser = p.chromium.connect_over_cdp(cdp_url)
                break
            except Exception as exc:
                LOGGER.warning("CDP %s indisponivel (tentativa %s/3): %s", cdp_url, tentativa, exc)
                time.sleep(2.0)
        if browser is None:
            LOGGER.error("Edge nao encontrado em %s. Abra o Edge com --remote-debugging-port=9222 "
                         "(ou use 'Preparar Edge'). Etiquetas NAO recoloridas.", cdp_url)
            return 2
        page = pick_page(browser)
        LOGGER.info("Pagina: %s", page.url)

        def acquire_panel() -> bool:
            if args.auto_open:
                return ensure_panel(page, args.property, delay, LOGGER)
            return wait_for_panel(page, args.wait_seconds, LOGGER)

        if not acquire_panel():
            LOGGER.error("Painel de Opcoes de %s nao foi aberto.", args.property)
            return 1
        LOGGER.info("Painel de Opcoes detectado (auto_open=%s).", args.auto_open)
        if getattr(args, "sort_only", False):
            # "Editar propriedade > Ordenar": clica a linha 'Ordenar [Manual] >' (painel de opcoes, a
            # mais a direita) -> submenu Manual/Alfabetica/Alfabetica reversa -> clica 'Alfabética'.
            ords = page.evaluate(
                "()=>{" + VIS + "const out=[];const all=Array.from(document.querySelectorAll('*')).filter(vis);"
                "for(const el of all){const t=norm(el.innerText||el.textContent||'');if(!/^ordenar/i.test(t)||t.length>22)continue;"
                "if(Array.from(el.children).some(c=>vis(c)&&/^ordenar/i.test(norm(c.innerText||c.textContent||''))))continue;"
                "const r=el.getBoundingClientRect();out.push({x:r.left+r.width/2,y:r.top+r.height/2});}return out;}")
            if not ords:
                LOGGER.warning("[%s] linha 'Ordenar' nao encontrada.", args.property); return 1
            row = max(ords, key=lambda e: e["x"])  # painel de opcoes fica a direita do menu da propriedade
            page.mouse.move(row["x"], row["y"]); time.sleep(0.12)
            page.mouse.click(row["x"], row["y"], delay=30)
            alfa = None
            for _ in range(15):  # poll: submenu (Manual/Alfabetica/reversa) demora mais em painel grande
                time.sleep(0.15)
                alfa = page.evaluate(
                    "(oy)=>{" + VIS + "const all=Array.from(document.querySelectorAll('*')).filter(vis);let best=null;"
                    "for(const el of all){const t=norm(el.innerText||el.textContent||'');if(t!=='Alfabética')continue;"
                    "if(Array.from(el.children).some(c=>vis(c)&&norm(c.innerText||c.textContent||'')==='Alfabética'))continue;"
                    "const r=el.getBoundingClientRect();const cy=r.top+r.height/2;if(cy<=oy+4)continue;"
                    "if(!best||cy<best.y)best={x:r.left+r.width/2,y:cy};}return best;}", row["y"])
                if isinstance(alfa, dict):
                    break
            if isinstance(alfa, dict):
                page.mouse.move(alfa["x"], alfa["y"]); time.sleep(0.12)
                page.mouse.click(alfa["x"], alfa["y"], delay=30); time.sleep(0.5)
                LOGGER.info("[%s] ordenado: Alfabética.", args.property); return 0
            LOGGER.warning("[%s] opcao 'Alfabética' do submenu nao encontrada.", args.property); return 1
        if args.diag:
            diag = page.evaluate(
                "({names})=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const opt=[...c].reverse().find(d=>norm(d.innerText||'').includes('Opções'));if(!opt)return {err:'no opt'};"
                "const fields=Array.from(opt.querySelectorAll('input,textarea,[contenteditable],[role=\\\"textbox\\\"]')).map(el=>{const at={};for(const a of el.attributes)at[a.name]=String(a.value).slice(0,40);return {tag:el.tagName,attrs:at};}).slice(0,8);"
                "const leaves=[];const w=document.createTreeWalker(opt,NodeFilter.SHOW_ELEMENT);let cnt=0;while(w.nextNode()&&cnt<2500){const el=w.currentNode;const t=norm(el.innerText||el.textContent||'');if(!t||t.length>60)continue;if(Array.from(el.children).some(ch=>norm(ch.innerText||ch.textContent||'')===t))continue;leaves.push(t);cnt++;}"
                "const set=new Set(leaves);const exact=names.map(n=>({n,found:set.has(norm(n))}));"
                "const contains=names.map(n=>({n,c:leaves.filter(t=>t.includes(norm(n))||norm(n).includes(t)).slice(0,2)}));"
                "return {fields, sampleLeaves:leaves.slice(0,18), totalLeaves:leaves.length, exact, contains};}",
                {"names": target_names + ["Abel Salvador Mesquita Junior"]},
            )
            print(json.dumps(diag, ensure_ascii=False, indent=2))
            return 0
        def focus_search() -> bool:
            return bool(page.evaluate(
                "()=>{" + _FIND_SEARCH_JS + "const inp=_find();if(!inp)return false;inp.focus();if(inp.select)inp.select();return true;}"
            ))

        def set_search(text: str) -> bool:
            """Define o valor da busca INSTANTANEAMENTE via setter nativo do React e
            dispara o evento 'input' (sem digitar caractere a caractere)."""
            return bool(page.evaluate(
                "({txt})=>{" + _FIND_SEARCH_JS + "const inp=_find();if(!inp)return false;inp.focus();const setter=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;setter.call(inp,txt);inp.dispatchEvent(new Event('input',{bubbles:true}));return true;}",
                {"txt": text},
            ))

        if args.probe_plus:
            lbl = target_names[0]
            if not _search_active(page):
                _click_plus(page, delay)
            set_search(lbl); time.sleep(delay)
            _open_option_submenu(page, lbl, delay); time.sleep(delay)
            _click_in_submenu(page, DEFAULT_COLOR_LABELS); time.sleep(delay)
            if _submenu_open(page):
                page.keyboard.press("Escape"); time.sleep(delay)
            page.screenshot(path=str(shots / "plus_collapsed.png"))
            LOGGER.info("estado colapsado: search_active=%s panel_open=%s", _search_active(page), _panel_open(page))
            dump = page.evaluate(
                "()=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);const opt=[...c].reverse().find(d=>norm(d.innerText||'').includes('Opções'));if(!opt)return null;"
                "let hdr=null;const w=document.createTreeWalker(opt,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||'')!=='Opções')continue;if(Array.from(el.children).some(ch=>vis(ch)&&norm(ch.innerText||'')==='Opções'))continue;hdr=el.getBoundingClientRect();break;}"
                "const near=Array.from(opt.querySelectorAll('*')).filter(vis).filter(e=>{const r=e.getBoundingClientRect();return hdr&&Math.abs(r.top-hdr.top)<26&&r.left>=hdr.left-4;}).slice(0,30).map(e=>{const r=e.getBoundingClientRect();const at={};for(const a of e.attributes)at[a.name]=String(a.value).slice(0,30);return {tag:e.tagName,role:e.getAttribute('role')||'',al:e.getAttribute('aria-label')||'',t:norm(e.innerText||'').slice(0,16),x:Math.round(r.left),y:Math.round(r.top),w:Math.round(r.width),h:Math.round(r.height)};});"
                "return {hdr:hdr?{x:Math.round(hdr.left),y:Math.round(hdr.top),w:Math.round(hdr.width)}:null, near};}"
            )
            print(json.dumps(dump, ensure_ascii=False, indent=2))
            return 0

        if args.probe_delete:
            orphan = next((t["name"] for t in targets if t["action"] == "delete"), target_names[0])
            if not _search_active(page):
                _click_plus(page, delay)
            set_search(orphan); time.sleep(delay)
            _open_option_submenu(page, orphan, delay); time.sleep(delay)
            before = page.evaluate("()=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);return c.map(d=>norm(d.innerText||'').slice(0,60));}")
            _click_in_submenu(page, DELETE_LABELS); time.sleep(delay + 0.3)
            after = page.evaluate("()=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);return c.map(d=>({snippet:norm(d.innerText||'').slice(0,70),btns:Array.from(d.querySelectorAll('[role=\\\"button\\\"],button,[role=\\\"menuitem\\\"]')).filter(vis).map(b=>norm(b.innerText||b.getAttribute('aria-label')||'').slice(0,20)).filter(Boolean).slice(0,12)}));}")
            page.screenshot(path=str(shots / "delete_after.png"))
            LOGGER.info("orphan=%s", orphan)
            LOGGER.info("ANTES Excluir: %s", json.dumps(before, ensure_ascii=False))
            LOGGER.info("DEPOIS Excluir: %s", json.dumps(after, ensure_ascii=False))
            return 0

        # UI atual NAO usa busca via '+': as opcoes ja ficam renderizadas na lista.
        # (best-effort; nao e fatal se nao houver caixa de busca.)
        if not _search_active(page):
            _click_plus(page, delay)

        def dialogs_state():
            return page.evaluate(
                "()=>{" + VIS + "const c=Array.from(document.querySelectorAll('[role=\\\"dialog\\\"],[role=\\\"menu\\\"]')).filter(vis);return {count:c.length, items:c.map(d=>({role:d.getAttribute('role'),x:Math.round(d.getBoundingClientRect().left),snippet:norm(d.innerText||'').slice(0,55)}))};}"
            )

        if args.probe_submenu:
            lbl = target_names[0]
            set_search(lbl); time.sleep(delay)
            print("apos busca:", json.dumps(dialogs_state(), ensure_ascii=False))
            _open_option_submenu(page, lbl, delay); time.sleep(delay)
            print("submenu aberto:", json.dumps(dialogs_state(), ensure_ascii=False))
            page.screenshot(path=str(shots / "submenu_open.png"))
            _click_in_submenu(page, DEFAULT_COLOR_LABELS); time.sleep(delay)
            print("apos clicar Padrao:", json.dumps(dialogs_state(), ensure_ascii=False), "| has_search=", _has_search(page))
            page.keyboard.press("Escape"); time.sleep(delay)
            print("apos 1 ESC:", json.dumps(dialogs_state(), ensure_ascii=False), "| has_search=", _has_search(page))
            page.screenshot(path=str(shots / "submenu_after_esc.png"))
            return 0

        def clear_search():
            set_search("")
            time.sleep(0.08)

        def reopen_list() -> bool:
            """Reabre a lista de Opcoes clicando 'Editar propriedade' no menu da
            propriedade (que persiste ao lado). Evita pedir reabertura ao usuario."""
            if _has_search(page):
                return True
            for _ in range(3):
                rect = _topmost_rect(page, "Editar propriedade")
                if rect:
                    page.mouse.click(rect["x"], rect["y"], delay=30)
                    time.sleep(delay + 0.2)
                    if _has_search(page):
                        return True
                time.sleep(0.2)
            return _has_search(page)

        def do_label(label, action):
            if not (_panel_open(page) and (not args.auto_open or _panel_is_for(page, args.property))):
                if not acquire_panel():
                    raise RuntimeError("painel nao reaberto")
            # UI atual: opcoes renderizadas direto -> abre o submenu na lista (sem busca).
            if _search_active(page):  # se houver busca, filtra (ajuda em listas enormes)
                set_search(label); time.sleep(delay * 0.5)
            if not _open_option_submenu(page, label, delay):
                raise RuntimeError("submenu nao abriu")
            if action == "delete":
                usage = _submenu_usage(page)
                if usage > 0:
                    page.keyboard.press("Escape")
                    raise RuntimeError(f"ABORTADO: UI indica {usage} pagina(s) usando '{label}' (nao e orfa)")
                if not _click_in_submenu(page, DELETE_LABELS):  # "Excluir" no detalhe
                    raise RuntimeError("nao clicou Excluir (detalhe)")
                confirmed = False
                for _ in range(12):
                    time.sleep(0.2)
                    if _confirm_present(page):
                        _click_in_submenu(page, ("Excluir", "Delete", "Remover", "Sim", "OK"))
                        confirmed = True
                        break
                if not confirmed:
                    raise RuntimeError("confirmacao de exclusao nao apareceu")
                time.sleep(delay)
                if _exact_present(page, label):
                    raise RuntimeError("etiqueta ainda presente apos exclusao")
            else:
                if not _click_in_submenu(page, DEFAULT_COLOR_LABELS):
                    raise RuntimeError("nao clicou Padrao")
                time.sleep(0.04)
                if _submenu_open(page):
                    page.keyboard.press("Escape"); time.sleep(0.05)

        processed = 0
        failed = 0
        for t in targets:
            label = t["name"]; action = t["action"]
            if label in done:
                continue
            ok_label = False
            last_err = None
            for attempt in range(3):
                try:
                    do_label(label, action)
                    ok_label = True
                    break
                except Exception as exc:
                    last_err = exc
                    LOGGER.warning("[%s] tentativa %s/3 '%s': %s", args.property, attempt + 1, label, exc)
                    for _ in range(3):
                        page.keyboard.press("Escape"); time.sleep(0.18)
                    if not _panel_open(page):
                        acquire_panel()
            if ok_label:
                done.add(label); processed += 1
                LOGGER.info("[%s] %s OK :: %s | %s/%s", args.property, action, label, processed, len(targets))
                ck_path.write_text(json.dumps({"done": sorted(done)}, ensure_ascii=False), encoding="utf-8")
            else:
                failed += 1
                LOGGER.error("[%s] FALHOU '%s' (3 tentativas): %s", args.property, label, last_err)
                try:
                    page.screenshot(path=str(shots / f"falha_{args.property}_{failed}.png"))
                except Exception:
                    pass
                if args.stop_on_error:
                    LOGGER.info("Parado no erro (processadas=%s, falhas=%s).", processed, failed)
                    return 1
        clear_search()
        page.keyboard.press("Escape")
        LOGGER.info("CONCLUIDO [%s] | processadas=%s | falhas=%s", args.property, processed, failed)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", required=True, choices=["partes", "advogados", "origem", "composicao"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay-ms", type=int, default=350)
    ap.add_argument("--apply-deletes", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--wait-seconds", type=int, default=90)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--probe-submenu", action="store_true")
    ap.add_argument("--probe-plus", action="store_true")
    ap.add_argument("--auto-open", action="store_true", help="Abre o painel da coluna via CDP (sem o usuario).")
    ap.add_argument("--probe-delete", action="store_true")
    ap.add_argument("--cdp-url", default=CDP_URL, help="URL do CDP do Edge (default 127.0.0.1:9222).")
    ap.add_argument("--sort-only", action="store_true", help="So clica 'Ordenar Alfabética' (sem recolorir/excluir).")
    ap.add_argument("--only-deletes", action="store_true", help="So exclui orfas (NAO recolore) — p/ composicao.")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
