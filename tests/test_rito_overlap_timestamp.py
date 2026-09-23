from tse_youtube_notion_core import TranscriptSnippet, detect_rito_events, count_individual_apregoamentos


def test_chamada_no_segundo_snippet_apos_pausa_nao_duplica():
    snippets = [
        TranscriptSnippet('tríplice ao poder executivo para escolha nomeação, nos termos do voto da relatora.', 1528, 1536),
        TranscriptSnippet('Chamo para o julgamento o processo administrativo', 1549, 1551),
        TranscriptSnippet('0674204 Natal Rio Grande do Norte da minha relatoria.', 1551, 1556),
    ]
    events = detect_rito_events(snippets)
    calls = [event for event in events if event.kind == 'apregoamento']
    assert len(calls) == 1
    assert calls[0].start_seconds == 1549
    assert calls[0].end_seconds == 1551
    assert count_individual_apregoamentos(events) == 1


def test_chamada_dividida_entre_legendas_mantem_inicio_real():
    snippets = [
        TranscriptSnippet('Chamo para', 100, 102),
        TranscriptSnippet('julgamento o processo administrativo', 103, 105),
        TranscriptSnippet('Chamo para julgamento outro feito', 130, 134),
    ]
    calls = [event for event in detect_rito_events(snippets) if event.kind == 'apregoamento']
    assert [event.start_seconds for event in calls] == [100, 130]
