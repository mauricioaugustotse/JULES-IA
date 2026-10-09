"""Recover scan exclusions contradicted by the independently dated TSE inventory.

Recovery schedules a video reread; it never fabricates a judgment or approves a row.
"""
from copy import deepcopy

from tse_official_session import _classification, _cnj


def recover_ignored_windows(session, inventory, *, confirmed_date):
    result = deepcopy(session)
    audit = {'session_date': confirmed_date, 'corrections': []}
    if (not confirmed_date or inventory.get('status') != 'available'
            or inventory.get('session_date') != confirmed_date):
        return result, audit
    processes = inventory.get('processes') or []
    for index, window in enumerate(result.get('judgments') or []):
        if not window.get('should_ignore'):
            continue
        cores = {_cnj(n)[0] for n in window.get('mentioned_process_numbers', [])} - {''}
        candidates = [p for p in processes if _cnj(p.get('numeroProcesso'))[0] in cores]
        # Multiple identities, secret/unknown statuses and collective lists are
        # not enough evidence to reverse an exclusion.
        if len(cores) != 1 or len(candidates) != 1:
            continue
        process = candidates[0]
        if _classification(process) not in {'judged', 'suspended'}:
            continue
        before = deepcopy(window)
        window.update(should_ignore=False, ignore_reason='')
        audit['corrections'].append({
            'index': index, 'numero_processo': process['numeroProcesso'],
            'before': before, 'after': deepcopy(window),
            'reason': 'O inventário oficial confirma julgamento individual ou pedido de vista; reler o vídeo.',
        })
    # A wrong date propagated through all details in the October 8 extraction.
    # The caller supplies a date read from the exact video title, never the model.
    if result.get('data_sessao') != confirmed_date:
        audit['date_correction'] = {
            'before': result.get('data_sessao'), 'after': confirmed_date,
            'reason': 'Data confirmada pelo título do vídeo e pelo inventário oficial.',
        }
    result['data_sessao'] = confirmed_date
    return result, audit
