#!/usr/bin/env python3
"""Automação via UI do Notion (Playwright/CDP) para padronizar etiquetas em `default`
e EXCLUIR etiquetas órfãs nas colunas `partes`, `advogados` e `origem`.

Por que UI e não API: a API oficial não recolore opção existente ("Cannot update color
of select") e PATCH em `options` é destrutivo (REPLACE + limite de 100). Esta automação
opera no próprio navegador logado, via Chrome DevTools Protocol.

Fluxo:
1) Abrir Edge/Chrome do Windows com remote debugging (launcher `prepare`).
2) Logar no Notion, abrir a base e deixar o painel de Opções da propriedade-alvo aberto.
3) `--mode dump` inspeciona os controles visíveis (debug).
4) `--mode apply` aplica `default` opção a opção (pulando as já-default) e, com
   `--apply-deletes`, exclui as órfãs marcadas no CSV. Checkpoint local p/ retomar.

Baseado no engine validado de `api_keywords`, estendido com exclusão de órfãs e leitura
do CSV por coluna (etiqueta + cor_atual + remover_se_apply).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CDP_URL = "http://127.0.0.1:9222"
DEFAULT_PROPERTY_NAME = "partes"
DEFAULT_OUTPUT_DIR = Path("artifacts") / "notion_labels_default"
DEFAULT_COLOR_LABELS = ("Padrão", "Padrao", "Default")
DELETE_LABELS = ("Excluir", "Remover", "Delete", "Remove")
DELETE_CONFIRM_LABELS = ("Excluir", "Delete", "Remove", "Sim", "Yes", "OK")
COLOR_NAMES = (
    "Padrão", "Padrao", "Cinza", "Marrom", "Laranja", "Amarelo", "Verde", "Azul",
    "Roxo", "Rosa", "Vermelho", "Default", "Gray", "Brown", "Orange", "Yellow",
    "Green", "Blue", "Purple", "Pink", "Red",
)
WINDOWS_EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
WINDOWS_CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

PlaywrightError = Exception


def _normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truthy(value: Any) -> bool:
    return _normalize_ws(value).lower() in {"1", "true", "sim", "s", "yes", "y"}


def load_plan_rows(path: Path, *, property_name: str) -> list[dict[str, str]]:
    """Lê o CSV do plano e devolve, para a coluna-alvo, dicts com etiqueta/cor_atual/
    cor_alvo/remover e a ação derivada: 'delete' (órfã), 'recolor' (cor != default) ou
    'skip' (já default e não órfã)."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"coluna", "etiqueta", "cor_atual", "cor_alvo", "remover_se_apply"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"CSV inválido: faltam colunas {', '.join(sorted(missing))}")
        target = _normalize_ws(property_name).casefold()
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in reader:
            if _normalize_ws(row.get("coluna")).casefold() != target:
                continue
            label = _normalize_ws(row.get("etiqueta"))
            if not label or label in seen:
                continue
            seen.add(label)
            cor_atual = _normalize_ws(row.get("cor_atual")) or "default"
            cor_alvo = _normalize_ws(row.get("cor_alvo")) or "default"
            remover = _truthy(row.get("remover_se_apply"))
            if remover:
                action = "delete"
            elif cor_atual != cor_alvo:
                action = "recolor"
            else:
                action = "skip"
            rows.append({"etiqueta": label, "cor_atual": cor_atual, "cor_alvo": cor_alvo,
                         "remover": "1" if remover else "0", "action": action})
    if not rows:
        raise RuntimeError(f"Nenhuma etiqueta para a coluna '{property_name}' em {path}")
    return rows


def select_items_for_run(
    rows: list[dict[str, str]],
    *,
    completed: set[str] | None = None,
    apply_deletes: bool = False,
    start_label: str = "",
    limit: int = 0,
    pending_limit: int = 0,
) -> list[dict[str, str]]:
    completed = completed or set()
    actionable = []
    for row in rows:
        action = row["action"]
        if action == "skip":
            continue
        if action == "delete" and not apply_deletes:
            continue
        actionable.append(row)
    if start_label:
        labels = [r["etiqueta"] for r in actionable]
        try:
            start_index = labels.index(start_label)
        except ValueError as exc:
            raise RuntimeError(f"--start-label não encontrado: {start_label}") from exc
        actionable = actionable[start_index:]
    if pending_limit > 0:
        actionable = [r for r in actionable if r["etiqueta"] not in completed][:pending_limit]
    elif limit > 0:
        actionable = actionable[:limit]
    return actionable


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"completed": [], "failed": {}, "started_at": time.time(), "updated_at": 0.0}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": [], "failed": {}, "started_at": time.time(), "updated_at": 0.0}
    payload.setdefault("completed", [])
    payload.setdefault("failed", {})
    payload.setdefault("started_at", time.time())
    payload.setdefault("updated_at", 0.0)
    return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def configure_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("notion_labels_default_playwright")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


def require_playwright():
    global PlaywrightError
    try:
        from playwright.sync_api import Error as _PlaywrightError
        from playwright.sync_api import sync_playwright as _sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright não instalado neste Python. Rode os launchers `.cmd` deste "
            "repositório (eles criam o venv e instalam o playwright)."
        ) from exc
    PlaywrightError = _PlaywrightError
    return _sync_playwright


def dump_visible_controls(page: Any) -> dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
          const visible = (el) => {
            if (!el || !(el instanceof Element)) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
          };
          const summarize = (selector, limit = 50) => {
            return Array.from(document.querySelectorAll(selector))
              .filter(visible)
              .slice(0, limit)
              .map((el) => ({
                tag: el.tagName,
                role: el.getAttribute("role") || "",
                aria: el.getAttribute("aria-label") || "",
                placeholder: el.getAttribute("placeholder") || "",
                text: norm(el.innerText || el.textContent || "").slice(0, 160),
              }));
          };
          return {
            url: location.href,
            title: document.title,
            dialogs: summarize('[role="dialog"]'),
            textboxes: summarize('input, textarea, [contenteditable="true"]'),
            buttons: summarize('button, [role="button"], [role="menuitem"], [role="option"]', 120),
          };
        }
        """
    )


_JS_VISIBLE = """
const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
const visible = (el) => {
  if (!el || !(el instanceof Element)) return false;
  const style = window.getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
};
"""


def _js_has_exact_text(page: Any, text: str, *, dialog_scope: str = "last") -> bool:
    return bool(
        page.evaluate(
            """
            ({ targetText, dialogScope }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const visible = (el) => {
                if (!el || !(el instanceof Element)) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
              };
              const visibleDialogs = Array.from(document.querySelectorAll('[role="dialog"]')).filter(visible);
              const optionsDialog = [...visibleDialogs].reverse().find((el) => {
                const text = norm(el.innerText || el.textContent || "").toLowerCase();
                return text.includes('opções') || text.includes('opcoes');
              }) || null;
              let root = document.body;
              if (dialogScope === 'last' && visibleDialogs.length) {
                root = visibleDialogs[visibleDialogs.length - 1];
              } else if (dialogScope === 'options' && optionsDialog) {
                root = optionsDialog;
              }
              const target = norm(targetText);
              const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
              while (walker.nextNode()) {
                const el = walker.currentNode;
                if (!(el instanceof Element) || !visible(el)) continue;
                const text = norm(el.innerText || el.textContent || "");
                if (text !== target) continue;
                const childSame = Array.from(el.children).some((child) => visible(child) && norm(child.innerText || child.textContent || "") === target);
                if (childSame) continue;
                return true;
              }
              return false;
            }
            """,
            {"targetText": text, "dialogScope": dialog_scope},
        )
    )


def _js_click_exact_text(page: Any, text: str, *, dialog_scope: str = "last") -> bool:
    return bool(
        page.evaluate(
            """
            ({ targetText, dialogScope }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const visible = (el) => {
                if (!el || !(el instanceof Element)) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
              };
              const visibleDialogs = Array.from(document.querySelectorAll('[role="dialog"]')).filter(visible);
              const optionsDialog = [...visibleDialogs].reverse().find((el) => {
                const text = norm(el.innerText || el.textContent || "").toLowerCase();
                return text.includes('opções') || text.includes('opcoes');
              }) || null;
              let root = document.body;
              if (dialogScope === 'last' && visibleDialogs.length) {
                root = visibleDialogs[visibleDialogs.length - 1];
              } else if (dialogScope === 'options' && optionsDialog) {
                root = optionsDialog;
              }
              const score = (el) => {
                const role = (el.getAttribute("role") || "").toLowerCase();
                let points = 0;
                if (role === "button" || role === "option" || role === "menuitem" || role === "menuitemradio") points += 20;
                if (el.tagName === "BUTTON") points += 15;
                if (el.closest('[role="dialog"]')) points += 8;
                points -= Math.min(norm(el.innerText || el.textContent || "").length, 200) / 20;
                return points;
              };
              const target = norm(targetText);
              const candidates = [];
              const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
              while (walker.nextNode()) {
                const el = walker.currentNode;
                if (!(el instanceof Element) || !visible(el)) continue;
                const text = norm(el.innerText || el.textContent || "");
                if (text !== target) continue;
                const childSame = Array.from(el.children).some((child) => visible(child) && norm(child.innerText || child.textContent || "") === target);
                if (childSame) continue;
                candidates.push(el);
              }
              candidates.sort((a, b) => score(b) - score(a));
              const chosen = candidates[0];
              if (!chosen) return false;
              chosen.click();
              return true;
            }
            """,
            {"targetText": text, "dialogScope": dialog_scope},
        )
    )


def _js_click_option_row(page: Any, text: str) -> bool:
    return bool(
        page.evaluate(
            """
            ({ targetText }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
              const visible = (el) => {
                if (!el || !(el instanceof Element)) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
              };
              const dialogs = Array.from(document.querySelectorAll('[role="dialog"]')).filter(visible);
              const root = [...dialogs].reverse().find((el) => {
                const text = norm(el.innerText || el.textContent || "").toLowerCase();
                return text.includes('opções') || text.includes('opcoes');
              }) || document.body;
              const target = norm(targetText);
              const candidates = [];
              const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
              while (walker.nextNode()) {
                const el = walker.currentNode;
                if (!(el instanceof Element) || !visible(el)) continue;
                const text = norm(el.innerText || el.textContent || "");
                if (text !== target) continue;
                const childSame = Array.from(el.children).some((child) => visible(child) && norm(child.innerText || child.textContent || "") === target);
                if (childSame) continue;
                candidates.push(el);
              }
              const chosen = candidates[0];
              if (!chosen) return false;
              const chosenRect = chosen.getBoundingClientRect();
              let clickable = chosen;
              let node = chosen.parentElement;
              while (node && node !== root.parentElement) {
                if (!visible(node)) { node = node.parentElement; continue; }
                const role = (node.getAttribute('role') || '').toLowerCase();
                const rect = node.getBoundingClientRect();
                if (role === 'button' || role === 'option' || role === 'menuitem' || role === 'menuitemradio') { clickable = node; break; }
                if (rect.width > chosenRect.width + 40 && rect.height >= chosenRect.height) { clickable = node; }
                node = node.parentElement;
              }
              clickable.click();
              return true;
            }
            """,
            {"targetText": text},
        )
    )


def _js_open_option_details(page: Any, text: str) -> bool:
    points = page.evaluate(
        """
        ({ targetText }) => {
          const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim();
          const visible = (el) => {
            if (!el || !(el instanceof Element)) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
          };
          const dialogs = Array.from(document.querySelectorAll('[role="dialog"]')).filter(visible);
          const root = [...dialogs].reverse().find((el) => {
            const text = norm(el.innerText || el.textContent || "").toLowerCase();
            return text.includes('opções') || text.includes('opcoes');
          }) || document.body;
          const target = norm(targetText);
          const candidates = [];
          const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
          while (walker.nextNode()) {
            const el = walker.currentNode;
            if (!(el instanceof Element) || !visible(el)) continue;
            const text = norm(el.innerText || el.textContent || "");
            if (text !== target) continue;
            const childSame = Array.from(el.children).some((child) => visible(child) && norm(child.innerText || child.textContent || "") === target);
            if (childSame) continue;
            candidates.push(el);
          }
          const chosen = candidates[0];
          if (!chosen) return null;
          chosen.scrollIntoView({ block: 'nearest', inline: 'nearest' });
          let clickable = chosen;
          let node = chosen.parentElement;
          const chosenRect = chosen.getBoundingClientRect();
          while (node && node !== root.parentElement) {
            if (!visible(node)) { node = node.parentElement; continue; }
            const rect = node.getBoundingClientRect();
            if (rect.width >= chosenRect.width + 40 && rect.height >= chosenRect.height) { clickable = node; }
            node = node.parentElement;
          }
          const rect = clickable.getBoundingClientRect();
          return {
            rowX: Math.max(1, Math.min(window.innerWidth - 2, rect.left + Math.min(24, rect.width / 4))),
            rowY: Math.max(1, Math.min(window.innerHeight - 2, rect.top + rect.height / 2)),
            chevronX: Math.max(1, Math.min(window.innerWidth - 2, rect.right - 12)),
            chevronY: Math.max(1, Math.min(window.innerHeight - 2, rect.top + rect.height / 2)),
          };
        }
        """,
        {"targetText": text},
    )
    if not isinstance(points, dict):
        return False
    attempts = [
        (float(points.get("chevronX", 0.0)), float(points.get("chevronY", 0.0))),
        (float(points.get("rowX", 0.0)), float(points.get("rowY", 0.0))),
    ]
    for x, y in attempts:
        if x <= 0 or y <= 0:
            continue
        page.mouse.click(x, y, delay=50)
        time.sleep(0.08)
        if _js_is_option_submenu_open(page):
            return True
    return False


def _js_is_option_submenu_open(page: Any) -> bool:
    return bool(
        page.evaluate(
            """
            ({ labels }) => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim().toLowerCase();
              const visible = (el) => {
                if (!el || !(el instanceof Element)) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
              };
              const dialogs = Array.from(document.querySelectorAll('[role="dialog"]')).filter(visible);
              if (!dialogs.length) return false;
              const last = dialogs[dialogs.length - 1];
              const text = norm(last.innerText || last.textContent || "");
              if (text.includes('cores') || text.includes('excluir')) return true;
              return labels.some((label) => text.includes(norm(label)));
            }
            """,
            {"labels": list(DEFAULT_COLOR_LABELS) + list(COLOR_NAMES)},
        )
    )


def _js_submenu_page_usage(page: Any) -> int:
    """Tenta ler quantas páginas usam a opção (texto tipo 'usada em N páginas' no
    submenu/confirmação). Retorna -1 se não houver indicação."""
    value = page.evaluate(
        """
        () => {
          const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim().toLowerCase();
          const visible = (el) => {
            if (!el || !(el instanceof Element)) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
          };
          const dialogs = Array.from(document.querySelectorAll('[role="dialog"]')).filter(visible);
          if (!dialogs.length) return -1;
          const last = dialogs[dialogs.length - 1];
          const text = norm(last.innerText || last.textContent || "");
          const m = text.match(/(\\d+)\\s*(p[aá]gina|page|registro|row)/);
          return m ? parseInt(m[1], 10) : -1;
        }
        """
    )
    try:
        return int(value)
    except Exception:
        return -1


def _js_focus_search_input(page: Any) -> bool:
    return bool(
        page.evaluate(
            """
            () => {
              const norm = (value) => String(value || "").replace(/\\s+/g, " ").trim().toLowerCase();
              const visible = (el) => {
                if (!el || !(el instanceof Element)) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
              };
              // SEGURANÇA: só operamos DENTRO de um dialog/menu/listbox aberto — NUNCA no
              // corpo da página (senão digitaríamos no título/bloco). Sem painel -> false.
              const dialogs = Array.from(document.querySelectorAll('[role="dialog"], [role="menu"], [role="listbox"]')).filter(visible);
              if (!dialogs.length) return false;
              const optionsDialog = [...dialogs].reverse().find((el) => {
                const text = norm(el.innerText || el.textContent || "");
                return text.includes('opções') || text.includes('opcoes');
              }) || null;
              const roots = optionsDialog ? [optionsDialog, ...dialogs.filter((el) => el !== optionsDialog).reverse()] : [...dialogs].reverse();
              for (const root of roots) {
                const fields = Array.from(root.querySelectorAll('input, textarea, [contenteditable="true"]')).filter(visible);
                if (!fields.length) continue;
                // NUNCA o campo "Nome da propriedade" (renomearia a coluna).
                const candidates = fields.filter((el) => norm(el.getAttribute('placeholder')) !== 'nome da propriedade');
                if (!candidates.length) continue;
                const preferred = candidates.find((el) => {
                  const placeholder = norm(el.getAttribute('placeholder'));
                  return placeholder.includes('search') || placeholder.includes('buscar') || placeholder.includes('opç') || placeholder.includes('option');
                });
                const fallback = preferred || candidates[0];
                if (fallback) {
                  fallback.focus();
                  if (typeof fallback.select === 'function') { fallback.select(); }
                  else {
                    const selection = window.getSelection?.();
                    const range = document.createRange();
                    range.selectNodeContents(fallback);
                    selection?.removeAllRanges();
                    selection?.addRange(range);
                  }
                  return true;
                }
              }
              return false;
            }
            """
        )
    )


def find_search_input(page: Any, *, selector: str = ""):
    if selector:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                return locator
        except PlaywrightError:
            pass
    if _js_focus_search_input(page):
        return None
    raise RuntimeError(
        "Não foi possível localizar a caixa de busca no painel. Abra a configuração da "
        "propriedade-alvo e deixe o campo de busca de opções visível."
    )


def clear_and_fill(page: Any, target: Any, text: str) -> None:
    if target is None:
        if not _js_focus_search_input(page):
            raise RuntimeError("Não foi possível focar o campo de busca.")
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        page.keyboard.type(text, delay=20)
        return
    try:
        target.click()
    except PlaywrightError:
        target = find_search_input(page)
        if target is None:
            if not _js_focus_search_input(page):
                raise RuntimeError("Não foi possível refocar o campo de busca.")
        else:
            target.click()
    try:
        if target is not None:
            target.fill("")
            target.fill(text)
            return
    except PlaywrightError:
        pass
    try:
        target.press("Control+A")
        target.press("Backspace")
    except PlaywrightError:
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
    page.keyboard.type(text, delay=20)


def close_menus(page: Any, *, times: int = 1) -> None:
    for _ in range(max(1, times)):
        page.keyboard.press("Escape")
        time.sleep(0.12)


def try_click_default(page: Any, *, logger: logging.Logger) -> bool:
    for _ in range(3):
        for candidate_text in DEFAULT_COLOR_LABELS:
            try:
                if _js_click_exact_text(page, candidate_text, dialog_scope="last"):
                    logger.debug("Clique em '%s' executado.", candidate_text)
                    return True
            except PlaywrightError:
                continue
        time.sleep(0.2)
    return False


def try_open_color_menu(page: Any, *, logger: logging.Logger) -> bool:
    for label in COLOR_NAMES:
        try:
            if _js_click_exact_text(page, label, dialog_scope="last"):
                logger.debug("Possível menu de cor aberto via '%s'.", label)
                return True
        except PlaywrightError:
            continue
    return False


def capture_debug_artifacts(page: Any, *, label: str, screenshot_dir: Path, logger: logging.Logger) -> None:
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", label)[:80] or "etiqueta"
    png_path = screenshot_dir / f"{token}.png"
    json_path = screenshot_dir / f"{token}.controls.json"
    try:
        page.screenshot(path=str(png_path), full_page=False)
        json_path.write_text(json.dumps(dump_visible_controls(page), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Artefatos de debug salvos em %s e %s", png_path, json_path)
    except Exception as exc:
        logger.warning("Falha ao salvar artefatos de debug para %s: %s", label, exc)


def pick_target_page(browser: Any, *, database_url: str) -> Any:
    for context in browser.contexts:
        for page in context.pages:
            url = _normalize_ws(page.url)
            if database_url and database_url in url:
                return page
            if "notion.so" in url or "notion.site" in url or "notion.com" in url:
                return page
    context = browser.contexts[0]
    page = context.new_page()
    if database_url:
        page.goto(database_url, wait_until="domcontentloaded")
    return page


def _open_option_submenu(page: Any, *, label: str, delay_seconds: float, screenshot_dir: Path, logger: logging.Logger) -> None:
    search_input = find_search_input(page, selector="")
    clear_and_fill(page, search_input, label)
    time.sleep(delay_seconds)
    if not _js_has_exact_text(page, label, dialog_scope="options"):
        capture_debug_artifacts(page, label=label, screenshot_dir=screenshot_dir, logger=logger)
        raise RuntimeError(f"Etiqueta não encontrada na UI atual: {label}")
    for _ in range(3):
        if _js_open_option_details(page, label):
            time.sleep(delay_seconds)
            if _js_is_option_submenu_open(page):
                return
        if _js_click_option_row(page, label):
            time.sleep(delay_seconds)
            if _js_is_option_submenu_open(page):
                return
    capture_debug_artifacts(page, label=label, screenshot_dir=screenshot_dir, logger=logger)
    raise RuntimeError(f"Não foi possível abrir o submenu da etiqueta: {label}")


def apply_default_to_label(page: Any, *, label: str, delay_seconds: float, screenshot_dir: Path, logger: logging.Logger) -> None:
    _open_option_submenu(page, label=label, delay_seconds=delay_seconds, screenshot_dir=screenshot_dir, logger=logger)
    if try_click_default(page, logger=logger):
        time.sleep(delay_seconds)
        close_menus(page)
        return
    if try_open_color_menu(page, logger=logger):
        time.sleep(delay_seconds)
        if try_click_default(page, logger=logger):
            time.sleep(delay_seconds)
            close_menus(page)
            return
    capture_debug_artifacts(page, label=label, screenshot_dir=screenshot_dir, logger=logger)
    raise RuntimeError(f"Não foi possível aplicar a cor default para: {label}")


def delete_orphan_label(page: Any, *, label: str, delay_seconds: float, screenshot_dir: Path, logger: logging.Logger) -> None:
    """Abre o submenu da opção e clica em Excluir. GUARDA contra perda de dados: se o
    submenu/confirmação indicar que a opção é usada por >0 páginas, ABORTA (a órfã
    deveria ter 0 usos). Verifica que a opção sumiu ao final."""
    _open_option_submenu(page, label=label, delay_seconds=delay_seconds, screenshot_dir=screenshot_dir, logger=logger)
    usage = _js_submenu_page_usage(page)
    if usage > 0:
        capture_debug_artifacts(page, label=label, screenshot_dir=screenshot_dir, logger=logger)
        close_menus(page, times=2)
        raise RuntimeError(f"Abortado: a UI indica {usage} página(s) usando '{label}' — não é órfã. NÃO excluído.")
    clicked = False
    for candidate in DELETE_LABELS:
        try:
            if _js_click_exact_text(page, candidate, dialog_scope="last"):
                clicked = True
                logger.debug("Clique em '%s' (excluir) executado.", candidate)
                break
        except PlaywrightError:
            continue
    if not clicked:
        capture_debug_artifacts(page, label=label, screenshot_dir=screenshot_dir, logger=logger)
        raise RuntimeError(f"Não foi possível acionar Excluir no submenu de: {label}")
    time.sleep(delay_seconds)
    # confirmação só se a opção ainda existir (algumas versões pedem confirmar)
    search_input = find_search_input(page, selector="")
    clear_and_fill(page, search_input, label)
    time.sleep(delay_seconds)
    if _js_has_exact_text(page, label, dialog_scope="options"):
        usage2 = _js_submenu_page_usage(page)
        if usage2 > 0:
            capture_debug_artifacts(page, label=label, screenshot_dir=screenshot_dir, logger=logger)
            close_menus(page, times=2)
            raise RuntimeError(f"Abortado na confirmação: {usage2} página(s) usando '{label}'.")
        for candidate in DELETE_CONFIRM_LABELS:
            try:
                if _js_click_exact_text(page, candidate, dialog_scope="last"):
                    break
            except PlaywrightError:
                continue
        time.sleep(delay_seconds)
        clear_and_fill(page, search_input, label)
        time.sleep(delay_seconds)
        if _js_has_exact_text(page, label, dialog_scope="options"):
            capture_debug_artifacts(page, label=label, screenshot_dir=screenshot_dir, logger=logger)
            raise RuntimeError(f"Etiqueta ainda presente após tentativa de exclusão: {label}")
    close_menus(page)
    clear_and_fill(page, search_input, "")
    logger.info("Etiqueta órfã excluída: %s", label)


def run_dump(page: Any, *, dump_output: Path) -> int:
    payload = dump_visible_controls(page)
    dump_output.parent.mkdir(parents=True, exist_ok=True)
    dump_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    message = f"Dump salvo em: {dump_output}\n"
    try:
        sys.stdout.write(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(message.encode("utf-8", errors="replace"))
    return 0


def run_apply(args: argparse.Namespace, logger: logging.Logger) -> int:
    manual_plan = Path(args.manual_plan_csv)
    if not manual_plan.is_absolute():
        manual_plan = DEFAULT_OUTPUT_DIR / f"{args.property_name}_plano_manual.csv"
    rows = load_plan_rows(manual_plan, property_name=args.property_name)

    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else (DEFAULT_OUTPUT_DIR / f"{args.property_name}.checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)
    completed = set(checkpoint.get("completed", []))
    failed = dict(checkpoint.get("failed", {}))
    items = select_items_for_run(
        rows,
        completed=completed,
        apply_deletes=args.apply_deletes,
        start_label=args.start_label,
        limit=args.limit,
        pending_limit=args.pending_limit,
    )
    n_recolor = sum(1 for r in items if r["action"] == "recolor")
    n_delete = sum(1 for r in items if r["action"] == "delete")
    logger.info("[%s] itens nesta execução: %s (recolor=%s, excluir=%s) | apply_deletes=%s",
                args.property_name, len(items), n_recolor, n_delete, args.apply_deletes)

    sync_playwright = require_playwright()
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.connect_over_cdp(args.cdp_url)
        except Exception as exc:
            if "ECONNREFUSED" in str(exc) or "retrieving websocket url" in str(exc):
                raise RuntimeError(
                    "Nenhum navegador com remote debugging em 127.0.0.1:9222. "
                    "Abra primeiro com `NOTION_labels_prepare_windows.cmd`."
                ) from exc
            raise
        page = pick_target_page(browser, database_url=_normalize_ws(args.database_url))
        logger.info("Página alvo: %s", page.url)
        if args.mode == "dump":
            dump_output = Path(args.dump_output) if args.dump_output else (DEFAULT_OUTPUT_DIR / "ultimo_dump.json")
            return run_dump(page, dump_output=dump_output)

        screenshot_dir = Path(args.screenshot_dir) if args.screenshot_dir else (DEFAULT_OUTPUT_DIR / "ui_debug")
        delay = args.delay_ms / 1000.0
        processed = 0
        for row in items:
            label = row["etiqueta"]
            action = row["action"]
            if label in completed:
                continue
            logger.info("[%s] %s -> %s", args.property_name, action, label)
            try:
                if args.dry_run:
                    logger.info("Dry-run: nenhuma alteração para %s (%s)", label, action)
                elif action == "delete":
                    delete_orphan_label(page, label=label, delay_seconds=delay, screenshot_dir=screenshot_dir, logger=logger)
                else:
                    apply_default_to_label(page, label=label, delay_seconds=delay, screenshot_dir=screenshot_dir, logger=logger)
                completed.add(label)
                failed.pop(label, None)
                checkpoint["completed"] = sorted(completed)
                checkpoint["failed"] = failed
                save_checkpoint(checkpoint_path, checkpoint)
                processed += 1
            except Exception as exc:
                failed[label] = str(exc)
                checkpoint["completed"] = sorted(completed)
                checkpoint["failed"] = failed
                save_checkpoint(checkpoint_path, checkpoint)
                logger.error("Falha em %s: %s", label, exc)
                if args.stop_on_error:
                    raise
            time.sleep(delay)
        logger.info("Execução concluída | processadas=%d | concluídas=%d | falhas=%d", processed, len(completed), len(failed))
        return 0


def maybe_launch_browser(args: argparse.Namespace, logger: logging.Logger) -> None:
    if not args.launch_browser:
        return
    match = re.search(r":(\d+)(?:/|$)", args.cdp_url)
    port = int(match.group(1)) if match else 9222
    start_url = _normalize_ws(args.database_url) or "https://www.notion.so"
    exe = WINDOWS_EDGE_PATH if args.browser == "edge" else WINDOWS_CHROME_PATH
    browser_args = [
        f"--remote-debugging-port={int(port)}",
        f"--user-data-dir={args.user_data_dir}",
        "--new-window",
        start_url,
    ]
    logger.info("Abrindo navegador: %s %s", exe, " ".join(browser_args))
    if os.name == "nt":
        subprocess.Popen([exe, *browser_args], close_fds=True)
    time.sleep(2.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automação UI do Notion: recolorir default + excluir órfãs.")
    parser.add_argument("--mode", choices=("dump", "apply"), default="apply")
    parser.add_argument("--property-name", default=DEFAULT_PROPERTY_NAME, help="Coluna-alvo: partes | advogados | origem.")
    parser.add_argument("--cdp-url", default=DEFAULT_CDP_URL)
    parser.add_argument("--database-url", default=os.getenv("NOTION_DATABASE_URL", "").strip())
    parser.add_argument("--manual-plan-csv", default="")
    parser.add_argument("--checkpoint-file", default="")
    parser.add_argument("--screenshot-dir", default="")
    parser.add_argument("--dump-output", default="")
    parser.add_argument("--start-label", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pending-limit", type=int, default=0)
    parser.add_argument("--delay-ms", type=int, default=450)
    parser.add_argument("--apply-deletes", action="store_true", help="Habilita a EXCLUSÃO das etiquetas órfãs marcadas no CSV.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--launch-browser", action="store_true")
    parser.add_argument("--browser", choices=("edge", "chrome"), default="edge")
    parser.add_argument("--user-data-dir", default=r"C:\Users\mauri\AppData\Local\Temp\notion-cdp-profile")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    logger = configure_logging(bool(args.debug))
    try:
        maybe_launch_browser(args, logger)
        return run_apply(args, logger)
    except Exception as exc:
        logger.error("Falha de execução: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
