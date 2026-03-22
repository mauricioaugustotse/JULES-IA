# JULES-IA

App e utilitários para extrair julgamentos de sessões do TSE no YouTube via Gemini e publicar no Notion.

## Requisitos

- Python com ambiente virtual ativo.
- `GEMINI_API_KEY` ou `GOOGLE_API_KEY`.
- `NOTION_API_KEY` ou `NOTION_TOKEN`.

As chaves podem estar no `.env` ou em arquivos locais como `Chave_Gemini.txt` e `Chave_Notion.txt`.

## Instalação

```bash
python -m pip install -r requirements.txt
```

No Windows, se estiver usando o `.venv` do projeto:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Executar a interface

```bash
streamlit run tse_youtube_notion_app.py
```

No Windows com o `.venv` do projeto:

```bash
.venv\Scripts\streamlit.exe run tse_youtube_notion_app.py
```

## O que a app faz

1. Recebe a URL pública de uma sessão do TSE no YouTube.
2. Usa Gemini em duas passagens para segmentar a sessão e extrair os julgamentos.
3. Normaliza campos jurídicos e valida contra o schema real do Notion.
4. Mostra uma prévia editável.
5. Cria ou atualiza páginas no data source configurado.

## Arquivos principais

- `tse_youtube_notion_app.py`: interface Streamlit.
- `tse_youtube_notion_core.py`: extração Gemini, preview e cliente Notion.
- `tse_normalization.py`: normalizações reaproveitadas do domínio jurídico-eleitoral.
- `tse_backfill_2025_notion.py`: backfill em lote do ano de 2025.
- `tse_backfill_monitor.py`: monitor de terminal para acompanhar o backfill.

## Monitorar o backfill

Snapshot único:

```bash
python tse_backfill_monitor.py
```

Monitor contínuo:

```bash
python tse_backfill_monitor.py --watch
```

No Windows com o `.venv` do projeto:

```bash
.venv\Scripts\python.exe tse_backfill_monitor.py --watch
```

## Retomar o backfill em modo econômico

Perfil recomendado para reduzir custo sem desligar a publicação:

```bash
python tse_backfill_2025_notion.py --resume --no-trash-unmatched-precedents --auto-scale --initial-workers 3 --max-workers 4
```

No Windows com o `.venv` do projeto:

```bash
.venv\Scripts\python.exe tse_backfill_2025_notion.py --resume --no-trash-unmatched-precedents --auto-scale --initial-workers 3 --max-workers 4
```

Observações:

- o backfill principal continua sem notícias por padrão;
- o grounding para `origem` isoladamente fica desligado por padrão para economizar;
- o grounding continua ativo quando falta o número CNJ integral.
