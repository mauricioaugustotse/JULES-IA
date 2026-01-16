## Purpose
Give targeted guidance for AI coding agents working on this repository (JULES-IA). Focus on the actual runtime flows, file-level conventions, and safe edit areas so automated edits are useful and low-risk.

## Quick architecture summary
- Repository contains a collection of Python ETL-style scripts that convert legal texts (TXT, DOCX, ementas) into structured CSVs using LLMs.
- Major scripts:
  - `TEMAS_SELC_txt_csv_v8.py` — split/julgados extraction, per-judgment LLM calls, checkpointing per `.txt` file -> `.csv` output.
  - `DOD_txt_to_csv_viaAPI_universal.py` — generalized batch processor with robust prompt templates and multi-tribunal support (STF/STJ).
  - `TSJE_docx_to_csv_viaAPI.py` — asynchronous docx → CSV pipeline, concurrent API calls with semaphore and quorum normalization.
  - `processar_teses.py` — Google Gemini batch worker that reads `copia_RG.csv` and writes processed thesis columns; uses checkpointing.

## Important patterns & conventions (use these as “contracts”)
- Checkpoints: scripts save progress to JSON files.
  - Per-file checkpoint: `<input>.checkpoint.json` (e.g., `document.txt.checkpoint.json`) — used by `TEMAS_*` scripts.
  - Global checkpoint: `processamento_checkpoint.json` (used by `processar_teses.py`).
  - Behavior: if output CSV exists AND checkpoint is absent, the file is treated as fully processed and usually skipped.
- Output naming: each input file maps to an output CSV using the same basename (e.g., `foo.txt` -> `foo.csv`, `file.docx` -> `file.csv`).
- Prompt location: prompts are embedded as `PROMPT_*` constants inside each script. Edits to prompt text should be implemented in-place in those files.
- API client compatibility: some scripts detect and support both the old `openai` client and the new `OpenAI`/`client` pattern. Keep edits compatible with both styles when modifying API-calls.

## Environment & dependencies
- Install dependencies:
  - pip install -r requirements.txt
  - `requirements.txt` currently lists: `pandas`, `google-generativeai`, `python-dotenv` (scripts also use `openai`, `python-docx`, `tqdm` — some are imported conditionally).
- Required env vars:
  - `OPENAI_API_KEY` for scripts that call OpenAI (TEMAS, DOD, TSJE variants).
  - `GOOGLE_API_KEY` for `processar_teses.py` (Gemini).
  - Use a `.env` file or system environment; scripts call `dotenv.load_dotenv()` in several places.

## Running / developer workflows (concrete commands)
- Run a single script locally (example):
  - python TEMAS_SELC_txt_to_csv_v8.py  # processes all `*.txt` in CWD
  - python DOD_txt_to_csv_viaAPI_universal.py --input "*.txt" --model gpt-5-mini
  - python TSJE_docx_to_csv_viaAPI.py    # processes all `*.docx` in script dir (async)
  - python processar_teses.py            # expects `copia_RG.csv` and GOOGLE_API_KEY
- Install deps then run in project root. Watch for pauses/rate-limits: scripts include pauses (e.g., `PAUSA_ENTRE_CHAMADAS_SEG`, `time.sleep`) and batch sizes (`TAMANHO_LOTE=20`).

## Rate-limit & concurrency notes
- Scripts intentionally throttle requests and use checkpoints to recover from partial runs. Default patterns to keep:
  - `TEMAS*`: small `pausa_s` between calls (e.g., 1–2s) and per-judgment checkpointing.
  - `processar_teses.py`: uses `TAMANHO_LOTE=20` and `PAUSA_ENTRE_CHAMADAS_SEG=6.1` to avoid quota overruns.
  - `TSJE_docx_to_csv_viaAPI.py`: concurrency controlled by `CONCURRENT_REQUESTS` semaphore (default 10).

## Where to edit (safe/unsafe areas)
- Safe to edit:
  - Prompt text (`PROMPT_*`) to improve extraction schema or formatting.
  - Timeouts, batch sizes, pause durations (to tune rate-limit behavior).
  - Extra logging, debug prints, and minor parsing robustness in helper functions.
- Be cautious editing:
  - Checkpoint naming and save/load logic — changing format will break resumability for in-flight runs.
  - The API-calling wrappers: preserve both new/old client compatibility or migrate consistently across repo.
  - Column order and CSV write logic — consumers may depend on specific CSV column names.

## Key examples to reference
- Regex-based split: `PADRAO_FINAL_JULGADO` in `TEMAS_SELC_txt_csv_v8.py` — used to split texts into individual judgments.
- JSON parsing helper: `_parse_and_normalize_json_response` (TEMAS) and `_strip_code_fences` + `_coerce_lists` (DOD) — use these when normalizing model output.
- Output write: `df.to_csv(saida_csv, index=False, encoding='utf-8-sig')` pattern is common; tests/tools expect UTF-8 with BOM for Excel compatibility.

## Integration & external dependencies
- Primary external services: OpenAI (gpt-5-mini, gpt-4o etc.) and Google Gemini (`gemini-2.5-flash`).
- If adding new models or clients, update both the client detection code and any hard-coded model names used in CLI defaults.

## Small checklist for PRs that change extraction logic
1. Update the PROMPT constant and add a small unit test or representative `.txt` fixture if possible.
2. Run the script locally on 1–2 small input files and verify output CSV and checkpoint behavior.
3. Preserve backward-compatible checkpoint format or provide a migration path.

## If you need clarification
- Ask which script is the authoritative source for a specific dataset (TEMAS vs DOD vs TSJE). Use `README.md` and this file as first references.

---
If this needs expansion or you want examples for tests/fixtures, tell me which script to target and I will add a small test harness and a README snippet.
