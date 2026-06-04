"""Baixa em lote os CSVs da pesquisa de jurisprudência do TSE (já aberta no Edge CDP,
com o captcha resolvido por você). Por página: marca 'selecionar tudo' (#checkAll),
clica 'Exportar decisões', captura o CSV e salva; depois vai para a próxima página.

Uso:
    python tse_baixar_csvs.py --test            # processa SÓ a página atual (1 CSV)
    python tse_baixar_csvs.py --paginas 200      # processa N páginas a partir da atual
"""
import argparse, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

CDP = "http://127.0.0.1:9222"
OUT = Path("artifacts") / "jurisprudencia_csv"


def pick(browser):
    for ctx in browser.contexts:
        for pg in ctx.pages:
            if "jurisprudencia.tse" in pg.url:
                return pg
    return None


def page_num(pg) -> int:
    m = re.search(r"[?&]pag=(\d+)", pg.url)
    return int(m.group(1)) if m else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--watch", action="store_true", help="Voce navega (>/F5/captcha); o script baixa cada pagina sozinho.")
    ap.add_argument("--inicio", type=int, default=1, help="Pagina inicial.")
    ap.add_argument("--paginas", type=int, default=200)
    ap.add_argument("--delay-ms", type=int, default=1500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    delay = args.delay_ms / 1000.0
    downloads = []

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP)
        pg = pick(browser)
        if not pg:
            print("NAO achei a aba da jurisprudencia (abra a pesquisa no Edge debug)."); return 1
        print("aba:", pg.url[:90])

        search_resps = []

        def on_resp(r):
            u = r.url.lower()
            if "download" in u or "export" in u or "csv" in u:
                downloads.append(r)
            elif u.rstrip("/").endswith("/public/pesquisa"):
                search_resps.append(r)
        pg.on("response", on_resp)

        JS_CSV_RECT = ("()=>{const v=(e)=>{const s=getComputedStyle(e);const r=e.getBoundingClientRect();"
                       "return s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0;};const it=Array.from("
                       "document.querySelectorAll('[role=menuitem],.mat-menu-item,button,a,li,span'))"
                       ".filter(v).find(e=>(e.innerText||'').trim().toUpperCase()==='CSV');"
                       "if(!it)return null;const r=it.getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};}")

        def rect_of(css):
            return pg.evaluate("(c)=>{const e=document.querySelector(c);if(!e)return null;e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();const s=getComputedStyle(e);if(s.display==='none'||r.width<=0)return null;return {x:r.left+r.width/2,y:r.top+r.height/2};}", css)

        def next_rect():
            # o '>' do paginador do TOPO, visivel e dentro da viewport, habilitado
            return pg.evaluate(
                """()=>{const H=window.innerHeight;const inv=(e)=>{const s=getComputedStyle(e);const r=e.getBoundingClientRect();return s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0&&r.top>=0&&r.top<H-10;};
                const bs=Array.from(document.querySelectorAll('button[aria-label=\\\"Próxima página\\\"]')).filter(inv).filter(b=>!b.disabled&&b.getAttribute('aria-disabled')!=='true').sort((a,b)=>a.getBoundingClientRect().top-b.getBoundingClientRect().top);
                if(!bs.length)return null;const r=bs[0].getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2};}"""
            )

        def wait_ready():
            for _ in range(40):
                if pg.evaluate("()=>{const c=document.querySelector('#checkAll');const b=document.querySelector('button[aria-label=\\\"Exportar decisões\\\"]');const cards=document.querySelectorAll('.card-resultado,[class*=card],mat-card');return !!(c&&b)&&cards.length>0;}"):
                    return True
                time.sleep(0.3)
            return False

        def export_one(N):
            downloads.clear()
            for attempt in range(4):
                pg.keyboard.press("Escape"); time.sleep(0.3)
                # garante SELEÇÃO: clica 'tudo' (mouse real) ate marcar as linhas.
                # IMPORTANTE: as linhas sao input.check-icone (NAO mat-checkbox); o clique
                # em #checkAll ALTERNA, entao so clicamos enquanto ainda nao marcou.
                selected = False
                for _ in range(6):
                    n = pg.evaluate("()=>document.querySelectorAll('input.check-icone:checked').length")
                    if n and n >= 2:
                        selected = True; break
                    ca = rect_of('#checkAll')
                    if ca:
                        pg.mouse.click(ca["x"], ca["y"])
                    time.sleep(0.5)
                if not selected:
                    time.sleep(0.5); continue
                exp = rect_of('button[aria-label="Exportar decisões"]')
                if not exp:
                    time.sleep(0.5); continue
                pg.mouse.click(exp["x"], exp["y"]); time.sleep(0.5)  # mouse REAL -> abre menu Material
                csv_rect = None
                for _ in range(16):
                    time.sleep(0.3)
                    csv_rect = pg.evaluate(JS_CSV_RECT)
                    if csv_rect:
                        break
                if not csv_rect:
                    continue
                try:
                    with pg.expect_download(timeout=15000) as dl_info:
                        pg.mouse.click(csv_rect["x"], csv_rect["y"])  # mouse REAL no item CSV
                    dest = OUT / f"tse_pag_{N:03d}.csv"
                    dl_info.value.save_as(str(dest))
                    print(f"  pag {N}: OK -> {dest.name} ({dest.stat().st_size} bytes)")
                    return True
                except Exception:
                    time.sleep(delay)
                    body = next((b for r in downloads for b in [_safe_body(r)] if b and len(b) > 200), None)
                    if body:
                        dest = OUT / f"tse_pag_{N:03d}.csv"
                        dest.write_bytes(body)
                        print(f"  pag {N}: OK (resp) -> {dest.name} ({len(body)} bytes)")
                        return True
            return False

        def wait_search(n_before, ticks=45):
            for _ in range(ticks):
                if len(search_resps) > n_before:
                    return True
                time.sleep(0.4)
            return False

        def results_present():
            return bool(pg.evaluate("()=>{const c=document.querySelector('#checkAll');const b=document.querySelector('button[aria-label=\\\"Exportar decisões\\\"]');return !!(c&&b)&&document.querySelectorAll('input.check-icone').length>5;}"))

        if args.watch:
            done = {int(f.stem.split('_')[-1]) for f in OUT.glob("tse_pag_*.csv") if f.stem.split('_')[-1].isdigit()}
            print(f"MODO WATCH — ja baixadas: {sorted(done) if done else 'nenhuma'}")
            print("Navegue voce (seta '>' / F5 / resolva o captcha quando aparecer). Eu baixo cada pagina sozinho.")
            print("Pare com Ctrl+C quando terminar.\n")
            idle = 0
            while True:
                try:
                    cur = page_num(pg)
                    if results_present() and cur not in done:
                        if export_one(cur):
                            done.add(cur); idle = 0
                            print(f"    [{len(done)} baixadas] -> navegue para a proxima.")
                        else:
                            time.sleep(1.5)
                    else:
                        idle += 1
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    break
                except Exception:
                    time.sleep(1.0)
            print(f"WATCH encerrado | total na pasta: {len(list(OUT.glob('tse_pag_*.csv')))}")
            return 0

        npages = 1 if args.test else args.paginas
        saved = 0
        falhas = []
        done = 0
        while done < npages:
            wait_ready()
            cur = page_num(pg)
            if export_one(cur):
                saved += 1
            else:
                falhas.append(cur)
                print(f"  pag {cur}: FALHOU export")
            done += 1
            if args.test or done >= npages:
                break
            # F5 (destrava a pagina apos o download), espera recarregar
            nb = len(search_resps)
            try:
                pg.reload(wait_until="domcontentloaded")
            except Exception:
                pass
            wait_search(nb); wait_ready(); time.sleep(delay)
            # avanca com a seta '>' (SPA quente passa o captcha)
            pg.keyboard.press("Escape"); time.sleep(0.3)
            nb = len(search_resps)
            advanced = False
            for _ in range(3):
                nr = next_rect()
                if not nr:
                    break
                pg.mouse.move(nr["x"], nr["y"]); time.sleep(0.1)
                pg.mouse.click(nr["x"], nr["y"])
                for _ in range(20):
                    time.sleep(0.4)
                    if page_num(pg) != cur:
                        advanced = True; break
                if advanced:
                    break
            if not advanced:
                print(f"  fim: nao avancou de {cur} (ultima pagina?)."); break
            wait_search(nb); wait_ready(); time.sleep(delay)
        msg = f"CONCLUIDO | salvos: {saved} | total na pasta: {len(list(OUT.glob('tse_pag_*.csv')))}"
        if falhas:
            msg += f" | FALHAS: {falhas}"
        print(msg)
    return 0


def _safe_body(r):
    try:
        return r.body()
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
