"""Probe/driver temporário: conecta ao Edge logado (CDP) e navega a UI do Notion
passo a passo para descobrir os selectors do painel de Opções da propriedade.

Uso:
  python _notion_ui_probe.py report            # estado atual (titulo + dialogs)
  python _notion_ui_probe.py esc               # ESC x4 e report
  python _notion_ui_probe.py header partes     # clica no cabecalho (topmost) e dump menu
  python _notion_ui_probe.py click "Editar propriedade"   # clica texto no ultimo menu/dialog e dump
"""
import json
import sys
from playwright.sync_api import sync_playwright

CDP = "http://127.0.0.1:9222"

DUMP_JS = """
() => {
  const norm = (v) => String(v||"").replace(/\\s+/g," ").trim();
  const vis = (el) => { if(!el||!(el instanceof Element))return false; const s=getComputedStyle(el); const r=el.getBoundingClientRect(); return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0; };
  const h1 = document.querySelector('h1, [role="textbox"][placeholder]');
  const containers = Array.from(document.querySelectorAll('[role="dialog"],[role="menu"],[role="listbox"]')).filter(vis).map(d=>({
    role:d.getAttribute('role'),
    aria:d.getAttribute('aria-label')||"",
    text: norm(d.innerText||d.textContent||"").slice(0,600),
    inputs: Array.from(d.querySelectorAll('input,textarea,[contenteditable="true"]')).filter(vis).map(i=>({ph:i.getAttribute('placeholder')||"",val:norm(i.value||i.innerText||"")})),
    menuitems: Array.from(d.querySelectorAll('[role="menuitem"],[role="option"],[role="button"],button')).filter(vis).slice(0,40).map(b=>norm(b.innerText||b.textContent||"")).filter(Boolean),
  }));
  return { url: location.href, title: h1?{ph:h1.getAttribute('placeholder')||"",text:norm(h1.innerText||h1.textContent||"")}:null, dialogCount: containers.length, containers: containers.slice(0,8) };
}
"""

CLICK_TOPMOST_JS = """
({ targetText }) => {
  const norm = (v) => String(v||"").replace(/\\s+/g," ").trim();
  const vis = (el) => { if(!el||!(el instanceof Element))return false; const s=getComputedStyle(el); const r=el.getBoundingClientRect(); return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0; };
  const target = norm(targetText);
  let best=null, bestTop=1e9;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
  while (walker.nextNode()) {
    const el = walker.currentNode;
    if (!vis(el)) continue;
    const t = norm(el.innerText||el.textContent||"");
    if (t !== target) continue;
    const childSame = Array.from(el.children).some((c)=>vis(c)&&norm(c.innerText||c.textContent||"")===target);
    if (childSame) continue;
    const top = el.getBoundingClientRect().top;
    if (top < bestTop) { bestTop = top; best = el; }
  }
  if (!best) return false;
  best.scrollIntoView({block:'center'});
  best.click();
  return true;
}
"""

CLICK_IN_LAST_JS = """
({ targetText }) => {
  const norm = (v) => String(v||"").replace(/\\s+/g," ").trim();
  const vis = (el) => { if(!el||!(el instanceof Element))return false; const s=getComputedStyle(el); const r=el.getBoundingClientRect(); return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0; };
  const cont = Array.from(document.querySelectorAll('[role="dialog"],[role="menu"],[role="listbox"]')).filter(vis);
  const root = cont.length ? cont[cont.length-1] : document.body;
  const target = norm(targetText);
  const cands=[];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  while (walker.nextNode()) {
    const el = walker.currentNode;
    if (!vis(el)) continue;
    const t = norm(el.innerText||el.textContent||"");
    if (t !== target) continue;
    const childSame = Array.from(el.children).some((c)=>vis(c)&&norm(c.innerText||c.textContent||"")===target);
    if (childSame) continue;
    cands.push(el);
  }
  if (!cands.length) return false;
  cands[0].click();
  return true;
}
"""


def pick(browser):
    for ctx in browser.contexts:
        for pg in ctx.pages:
            if "notion." in pg.url:
                return pg
    return browser.contexts[0].pages[0]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(CDP)
        page = pick(browser)
        if cmd == "esc":
            for _ in range(4):
                page.keyboard.press("Escape"); page.wait_for_timeout(150)
        elif cmd == "header":
            ok = page.evaluate(CLICK_TOPMOST_JS, {"targetText": arg})
            print("click header ->", ok); page.wait_for_timeout(600)
        elif cmd == "click":
            ok = page.evaluate(CLICK_IN_LAST_JS, {"targetText": arg})
            print("click ->", ok); page.wait_for_timeout(600)
        elif cmd == "find":
            res = page.evaluate(
                """
                ({ targetText }) => {
                  const norm = (v) => String(v||"").replace(/\\s+/g," ").trim();
                  const vis = (el) => { if(!el||!(el instanceof Element))return false; const s=getComputedStyle(el); const r=el.getBoundingClientRect(); return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0; };
                  const target = norm(targetText).toLowerCase();
                  const out=[];
                  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
                  while (walker.nextNode()) {
                    const el = walker.currentNode;
                    if (!vis(el)) continue;
                    const t = norm(el.innerText||el.textContent||"");
                    if (t.toLowerCase() !== target) continue;
                    const childSame = Array.from(el.children).some((c)=>vis(c)&&norm(c.innerText||c.textContent||"")===norm(el.innerText||el.textContent||""));
                    if (childSame) continue;
                    const r = el.getBoundingClientRect();
                    out.push({tag:el.tagName, role:el.getAttribute('role')||"", aria:el.getAttribute('aria-label')||"", x:Math.round(r.left), y:Math.round(r.top), w:Math.round(r.width), h:Math.round(r.height)});
                  }
                  return out.slice(0,20);
                }
                """,
                {"targetText": arg},
            )
            print(json.dumps(res, ensure_ascii=False, indent=2)); return
        elif cmd == "openpanel":
            prop = arg or "partes"
            for _ in range(2):
                page.keyboard.press("Escape"); page.wait_for_timeout(200)
            rect_top = page.evaluate(
                """({targetText})=>{const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};const t=norm(targetText);let best=null,top=1e9;const w=document.createTreeWalker(document.body,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||"")!==t)continue;if(Array.from(el.children).some(c=>vis(c)&&norm(c.innerText||c.textContent||"")===t))continue;const r=el.getBoundingClientRect();if(r.top<top){top=r.top;best=r;}}return best?{x:best.left+best.width/2,y:best.top+best.height/2}:null;}""",
                {"targetText": prop},
            )
            ok1 = False
            if rect_top:
                page.mouse.click(rect_top["x"], rect_top["y"]); ok1 = True
            page.wait_for_timeout(700)
            rect_ed = page.evaluate(
                """({targetText})=>{const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};const cont=Array.from(document.querySelectorAll('[role="dialog"],[role="menu"]')).filter(vis);const root=cont.length?cont[cont.length-1]:document.body;const t=norm(targetText);const w=document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT);while(w.nextNode()){const el=w.currentNode;if(!vis(el))continue;if(norm(el.innerText||el.textContent||"")!==t)continue;if(Array.from(el.children).some(c=>vis(c)&&norm(c.innerText||c.textContent||"")===t))continue;const r=el.getBoundingClientRect();return {x:r.left+Math.min(20,r.width/2),y:r.top+r.height/2};}return null;}""",
                {"targetText": "Editar propriedade"},
            )
            ok2 = False
            if rect_ed:
                page.mouse.click(rect_ed["x"], rect_ed["y"]); ok2 = True
            page.wait_for_timeout(900)
            # TESTE: a busca vem auto-focada? digita e vê se a lista filtra.
            page.keyboard.type("Abelardo", delay=40)
            page.wait_for_timeout(700)
            try:
                page.screenshot(path="artifacts/notion_labels_default/ui_debug/diag_typed.png", full_page=False)
            except Exception:
                pass
            after = page.evaluate(
                """()=>{const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};const cont=Array.from(document.querySelectorAll('[role="dialog"],[role="menu"]')).filter(vis);const opt=[...cont].reverse().find(d=>norm(d.innerText||"").includes('Opções'))||cont[cont.length-1];if(!opt)return {found:false};const rows=Array.from(opt.querySelectorAll('*')).filter(e=>vis(e)).map(e=>norm(e.innerText||"")).filter(t=>t&&t.length<60);const uniq=[...new Set(rows)].slice(0,12);return {found:true,snippet:norm(opt.innerText||"").slice(0,70),sample:uniq};}"""
            )
            print("after_type:", json.dumps(after, ensure_ascii=False))
            return
            scrolled = page.evaluate(
                """()=>{const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};const cont=Array.from(document.querySelectorAll('[role="dialog"],[role="menu"]')).filter(vis);const opt=[...cont].reverse().find(d=>norm(d.innerText||"").includes('Opções'));if(!opt)return false;let n=0;opt.querySelectorAll('*').forEach(e=>{if(e.scrollHeight>e.clientHeight+10){e.scrollTop=0;n++;}});return n;}"""
            )
            page.wait_for_timeout(400)
            try:
                page.screenshot(path="artifacts/notion_labels_default/ui_debug/diag_panel.png", full_page=False)
            except Exception as e:
                print("screenshot fail", e)
            allf = page.evaluate(
                """()=>{const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};return Array.from(document.querySelectorAll('input,textarea,[contenteditable="true"],[role="textbox"],[role="searchbox"]')).filter(vis).map(el=>{const r=el.getBoundingClientRect();return {tag:el.tagName,role:el.getAttribute('role')||"",ph:el.getAttribute('placeholder')||el.getAttribute('data-placeholder')||el.getAttribute('aria-label')||"",x:Math.round(r.left),y:Math.round(r.top),w:Math.round(r.width)};}).filter(f=>f.x>1000);}"""
            )
            print("scrolled_containers:", scrolled)
            print("fields_x>1000:", json.dumps(allf, ensure_ascii=False))
            containers = page.evaluate(
                """
                () => {
                  const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();
                  const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};
                  const cont=Array.from(document.querySelectorAll('[role="dialog"],[role="menu"],[role="listbox"]')).filter(vis);
                  return cont.map((d,i)=>{const r=d.getBoundingClientRect();const flds=Array.from(d.querySelectorAll('input,textarea,[contenteditable="true"]')).filter(vis).map(e=>e.getAttribute('placeholder')||e.getAttribute('data-placeholder')||e.getAttribute('aria-label')||('<'+e.tagName+'>'));return {i,role:d.getAttribute('role'),x:Math.round(r.left),w:Math.round(r.width),snippet:norm(d.innerText||"").slice(0,60),fields:flds};});
                }
                """
            )
            print(json.dumps({"clicked_header":ok1,"clicked_editar":ok2,"containers":containers}, ensure_ascii=False, indent=2)); return
            inp = page.evaluate(
                """
                () => {
                  const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();
                  const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};
                  const cont=Array.from(document.querySelectorAll('[role="dialog"],[role="menu"],[role="listbox"]')).filter(vis);
                  const opt=[...cont].reverse().find(d=>norm(d.innerText||"").toLowerCase().includes('opções')||norm(d.innerText||"").toLowerCase().includes('opcoes'));
                  const all=Array.from(document.querySelectorAll('input,textarea,[contenteditable="true"]')).filter(vis).map(el=>{
                    const r=el.getBoundingClientRect();
                    const ph=el.getAttribute('placeholder')||el.getAttribute('data-placeholder')||el.getAttribute('aria-label')||"";
                    return {tag:el.tagName,ce:el.getAttribute('contenteditable'),ph,val:norm(el.value||el.innerText||"").slice(0,30),x:Math.round(r.left),y:Math.round(r.top),inOpt: opt?opt.contains(el):false};
                  }).filter(f=>f.ph||f.tag!=="INPUT");
                  return {optDialogFound:Boolean(opt), optText:norm((opt||{innerText:""}).innerText||"").slice(0,80), allFields:all};
                }
                """
            )
            print(json.dumps({"clicked_header":ok1,"clicked_editar":ok2,**inp}, ensure_ascii=False, indent=2)); return
        elif cmd == "allpages":
            for ci, ctx in enumerate(browser.contexts):
                for pi, pg in enumerate(ctx.pages):
                    try:
                        snap = pg.evaluate(
                            """()=>{
                              const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();
                              const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};
                              const cont=Array.from(document.querySelectorAll('[role="dialog"],[role="menu"],[role="listbox"]')).filter(vis);
                              return {n:cont.length, hasOpc:cont.some(d=>norm(d.innerText||"").toLowerCase().includes('op')&&norm(d.innerText||"").includes('Opções')), texts:cont.map(d=>norm(d.innerText||"").slice(0,40))};
                            }"""
                        )
                    except Exception as e:
                        snap = {"err": str(e)[:60]}
                    print(f"ctx{ci} pg{pi} url={pg.url[:60]} :: {json.dumps(snap, ensure_ascii=False)}")
            return
        elif cmd == "watch":
            import sys as _sys
            for i in range(25):
                snap = page.evaluate(
                    """()=>{
                      const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();
                      const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};
                      const cont=Array.from(document.querySelectorAll('[role="dialog"],[role="menu"],[role="listbox"]')).filter(vis);
                      const texts=cont.map(d=>norm(d.innerText||"").slice(0,45));
                      const hasOpc=cont.some(d=>norm(d.innerText||"").includes('Opções'));
                      return {n:cont.length, hasOpc, texts};
                    }"""
                )
                _sys.stdout.write(f"t={i:2d} dialogs={snap['n']} opcoes={snap['hasOpc']} :: {snap['texts']}\n")
                _sys.stdout.flush()
                page.wait_for_timeout(1000)
            return
        elif cmd == "waitdump":
            # Espera (polling) o usuario abrir o painel de Opcoes e dumpa a estrutura real.
            found = None
            for _ in range(120):
                found = page.evaluate(
                    """()=>{
                      const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();
                      const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};
                      const cont=Array.from(document.querySelectorAll('[role="dialog"],[role="menu"],[role="listbox"]')).filter(vis);
                      const opt=[...cont].reverse().find(d=>norm(d.innerText||"").includes('Opções'));
                      if(!opt)return null;
                      const fields=Array.from(opt.querySelectorAll('input,textarea,[contenteditable],[role="textbox"],[role="searchbox"]')).filter(vis).map(el=>{
                        const r=el.getBoundingClientRect();
                        const at={}; for(const a of el.attributes){at[a.name]=String(a.value).slice(0,40);}
                        return {tag:el.tagName, role:el.getAttribute('role')||"", attrs:at, x:Math.round(r.left), y:Math.round(r.top), w:Math.round(r.width), h:Math.round(r.height)};
                      });
                      return {optX:Math.round(opt.getBoundingClientRect().left), snippet:norm(opt.innerText||"").slice(0,80), fieldCount:fields.length, fields};
                    }"""
                )
                if found:
                    break
                page.wait_for_timeout(1000)
            if found:
                try:
                    page.screenshot(path="artifacts/notion_labels_default/ui_debug/waitdump.png", full_page=False)
                except Exception:
                    pass
                pagewide = page.evaluate(
                    """()=>{const norm=(v)=>String(v||"").replace(/\\s+/g," ").trim();const vis=(el)=>{if(!el||!(el instanceof Element))return false;const s=getComputedStyle(el);const r=el.getBoundingClientRect();return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0;};return Array.from(document.querySelectorAll('input,textarea,[contenteditable="true"],[role="textbox"],[role="searchbox"]')).filter(vis).map(el=>{const r=el.getBoundingClientRect();const at={};for(const a of el.attributes){if(['placeholder','data-placeholder','aria-label','role','type','class'].includes(a.name))at[a.name]=String(a.value).slice(0,50);}return {tag:el.tagName,attrs:at,x:Math.round(r.left),y:Math.round(r.top),w:Math.round(r.width)};}).filter(f=>f.x>900);}"""
                )
                found["pagewide_fields_xgt900"] = pagewide
            print(json.dumps(found, ensure_ascii=False, indent=2) if found else "TIMEOUT: painel nao aberto em 120s")
            return
        elif cmd == "inputs":
            res = page.evaluate(
                """
                () => {
                  const norm = (v) => String(v||"").replace(/\\s+/g," ").trim();
                  const vis = (el) => { if(!el||!(el instanceof Element))return false; const s=getComputedStyle(el); const r=el.getBoundingClientRect(); return s.visibility!=="hidden"&&s.display!=="none"&&r.width>0&&r.height>0; };
                  return Array.from(document.querySelectorAll('input,textarea,[contenteditable="true"],[contenteditable=""]')).filter(vis).map(el=>{
                    const r=el.getBoundingClientRect();
                    let anc=el, roleAnc="";
                    while(anc){ const ro=anc.getAttribute&&anc.getAttribute('role'); if(ro){roleAnc=ro;break;} anc=anc.parentElement; }
                    return {tag:el.tagName, ce:el.getAttribute('contenteditable'), ph:el.getAttribute('placeholder')||el.getAttribute('data-placeholder')||el.getAttribute('aria-label')||"", val:norm(el.value||el.innerText||""), roleAnc, x:Math.round(r.left), y:Math.round(r.top), w:Math.round(r.width), h:Math.round(r.height)};
                  }).slice(0,30);
                }
                """
            )
            print(json.dumps(res, ensure_ascii=False, indent=2)); return
        print(json.dumps(page.evaluate(DUMP_JS), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
