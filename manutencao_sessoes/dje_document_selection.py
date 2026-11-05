"""Select a DJe judgment by date and procedural decision, without network access."""
from datetime import date
import re


def _date(value):
    try:
        return date.fromisoformat(str(value or '')[:10])
    except ValueError:
        return None


def _embargos(document):
    judicial_class = (document.get('classe') or '').upper().strip()
    if judicial_class:
        return judicial_class.startswith(('ED', 'EMBARGOS'))
    # The source class was not imported in older snapshots. Examine only the
    # dispositive opening, rather than mentions in a long ementa/report.
    return 'embargos' in (document.get('decisao') or '').lower()[:150]


def choose_document(documents, session_date, page_result, page_class, result_parser, *, allow_window=False):
    """Return only a unique source whose outcome and ED phase match the page.

    Equal CNJ and calendar date do not identify a procedural decision: the four
    TV cases of28Oct2022 have morning relief and evening homologation. Unknown
    outcomes and distinct matching sources stay pending for certificate review.
    """
    day = _date(session_date)
    if not day or not documents:
        return None, 'sem_data_ou_sem_acordao'
    dated = [(doc, (_date(doc.get('data')) - day).days) for doc in documents if _date(doc.get('data'))]
    if not dated:
        return None, 'acordao_sem_data'
    exact = [(doc, delta) for doc, delta in dated if delta == 0]
    candidates = exact or ([(doc, delta) for doc, delta in dated if -5 <= delta <= 60] if allow_window else [])
    if not candidates:
        return None, 'sem_acordao_no_dia_{:+d}d'.format(min((delta for _, delta in dated), key=abs))
    page_ed = str(page_class or '').upper().strip().startswith(('ED', 'EMBARGOS'))
    matches = []
    for doc, delta in candidates:
        decision = doc.get('decisao') or ''
        result, _ = result_parser(decision)
        if result and page_result and result == page_result and page_ed == _embargos(doc):
            matches.append((doc, delta))
    if not matches:
        return None, 'data_exata_sem_confirmacao_de_fase' if exact else 'janela_sem_confirmacao'
    # Multiple source rows with byte-equivalent legal text are one document.
    # Otherwise equal outcomes still do not prove that two decisions are equal.
    signatures = {(delta, re.sub(r'\s+', ' ', doc.get('ementa') or '').strip(),
                   re.sub(r'\s+', ' ', doc.get('decisao') or '').strip()) for doc, delta in matches}
    if len(signatures) != 1:
        return None, 'acordaos_distintos_exigem_revisao_de_fase'
    doc, delta = matches[0]
    return doc, 'data_exata_resultado_e_fase_conferem' if exact else 'janela_{:+d}d_resultado_confere'.format(delta)
