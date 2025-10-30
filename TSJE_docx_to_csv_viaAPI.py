# -*- coding: utf-8 -*-
from __future__ import annotations
import re, csv, os, json, time, sys, traceback, asyncio
from pathlib import Path
from typing import List, Tuple, Dict, Any

# --- Bloco de Cores para o Terminal (para um log mais sofisticado) ---
class C:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
# --------------------------------------------------------------------

# --- Bloco de Instalação de Dependências ---
try:
    from docx import Document
except ImportError:
    raise SystemExit(f"{C.RED}ERRO: python-docx é necessário. Execute 'pip install python-docx'{C.END}")

try:
    import openai
except ImportError:
    raise SystemExit(f"{C.RED}ERRO: openai é necessário. Execute 'pip install openai'{C.END}")

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    raise SystemExit(f"{C.RED}ERRO: tqdm é necessário. Execute 'pip install tqdm'{C.END}")
# -----------------------------------------

# --- Configuração da API OpenAI ---
# MANTENHA A SUA CHAVE DE API AQUI
if not api_key:
    raise SystemExit(f"{C.RED}A chave da API da OpenAI não foi inserida no código.{C.END}")
client = openai.AsyncOpenAI(api_key=api_key)
# -----------------------------------

# --- Constantes e Listas Globais ---
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5

MINISTROS_LISTA_OFICIAL = sorted([
    "Affonso Celso de Assis Figueiredo Júnior", "Affonso Augusto Moreira Penna Júnior",
    "Antônio de Sampaio Dória", "Armando Prado", "Arthur Quadros Collares Moreira",
    "Cândido Luiz Maria de Oliveira Filho", "Eduardo Espinola", "Francisco Carneiro Monteiro de Salles",
    "Francisco Cezário Alvim", "Hermenegildo Rodrigues de Barros", "João Chrisóstomo da Rocha Cabral",
    "João Martins de Carvalho Mourão", "José de Miranda Valverde", "José Linhares",
    "José Lisboa", "José Maria Mac Dowell da Costa", "José Ovídio Marcondes Romeiro",
    "José Soriano de Souza Filho", "Laudo Ferreira de Camargo", "Levi Fernandes Carneiro",
    "Mário da Lima Rocha", "Plínio de Castro Casado", "Prudente de Moraes Filho",
    "Renato de Carvalho Tavares"
])

MODEL_COLS = ["classe","punchline","decisão","numero_do_processo","data_da_decisão","numero_da_sessão","UF",
              "relator","parte_recorrente","parte_recorrida","interessado","inteiro_teor","ementa","quorum", "votação"]

# --- Funções Auxiliares ---
def _norm_spaces(s: str) -> str: return re.sub(r"\s{2,}", " ", (s or "").replace("\xa0", " ")).strip()

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions, deletions, substitutions = previous_row[j + 1] + 1, current_row[j] + 1, previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def find_best_match(name: str, name_list: List[str]) -> str:
    if not name or not name_list: return ""
    return min(name_list, key=lambda official_name: levenshtein_distance(name.upper(), official_name.upper()))

def normalize_date_pt(text: str) -> str:
    meses = {"JANEIRO":1,"FEVEREIRO":2,"MARÇO":3,"MARCO":3,"ABRIL":4,"MAIO":5,"JUNHO":6,"JULHO":7, "AGOSTO":8,"SETEMBRO":9,"OUTUBRO":10,"NOVEMBRO":11,"DEZEMBRO":12}
    m = re.search(r"(\d{1,2})\s*(?:DE)?\s*([A-ZÇÃÕÉÍÓÚÂÊÔÜÀ]+)\s*(?:DE)?\s*(\d{4})", (text or "").upper())
    if not m: return ""
    d, mes, ano = int(m.group(1)), m.group(2), int(m.group(3))
    mesn = meses.get(mes)
    return f"{d:02d}/{mesn:02d}/{ano:04d}" if mesn else ""

def extract_sessao_num(text: str) -> str:
    uc_text = (text or "").upper()
    if "SESSÃO PREPARATÓRIA" in uc_text: return "sessão preparatória"
    if "SESSÃO DE INSTALAÇÃO" in uc_text: return "sessão de instalação"
    m_extra = re.search(r"(\d+)\s*ª\s*SESSÃO\s+EXTRAORDINÁRIA", uc_text)
    if m_extra: return f"{m_extra.group(1)}E"
    m_ord = re.search(r"(\d+)\s*ª\s*SESSÃO\s+ORDINÁRIA", uc_text)
    if m_ord: return m_ord.group(1)
    m_any = re.search(r"(\d+)\s*ª\s*SESS[ÃA]O", uc_text)
    if m_any: return m_any.group(1)
    return "sessão s/nº"

def extract_sessions_from_docx(path:Path)->list[str]:
    doc=Document(str(path));sessions,current=[],[]
    session_heading_pattern=re.compile(r"^\s*(?:ATA DA\s+)?(?:\d+ª\s*)?SESS[ÃA]O.*",re.I)
    for p in doc.paragraphs:
        t=(p.text or "").strip();
        if not t:continue
        is_sep="***"in t;is_head=session_heading_pattern.match(t)
        if(is_sep or is_head)and current:sessions.append("\n".join(current));current=[]
        if not is_sep:current.append(p.text)
    if current:sessions.append("\n".join(current))
    return[s.strip()for s in sessions if len(s.strip())>50]

async def extract_data_with_llm(session_index: int, session_text: str, session_date: str, session_num: str, ministros_list: List[str]) -> List[Dict[str, str]]:
    """Função assíncrona que envia a sessão inteira para a IA decompor e extrair."""
    ministros_prompt_part = f"O nome do relator DEVE ser um dos seguintes: [{', '.join(ministros_list)}]. Corrija para o nome mais próximo da lista." if ministros_list else ""

    date_instruction = f'Use o valor "{session_date}".'
    if not session_date:
        date_instruction = 'Extraia a data completa por extenso (ex: "30 de julho de 1932") do "TEXTO COMPLETO DA SESSÃO" e formate-a estritamente como DD/MM/AAAA (formato brasileiro Dia/Mês/Ano). Por exemplo, "30 de julho de 1932" deve se tornar "30/07/1932". Cuidado para não registrar no padrão americano MM/DD/AAAA. Muita atenção a isso! Estou no Brasil e quero a data no formato brasileiro.'

    prompt = f"""
    Sua tarefa principal é analisar o TEXTO COMPLETO DE UMA SESSÃO do Tribunal Superior Eleitoral de 1932 e decompor seu conteúdo.

    TEXTO COMPLETO DA SESSÃO:
    ---
    {session_text}
    ---
    INSTRUÇÃO PRINCIPAL:
    1. DECOMPONHA O TEXTO: Leia o texto completo e identifique CADA deliberação, processo, consulta ou ato administrativo individual. Cada assunto distinto deve se tornar um registro separado.
    2. CRIE UM OBJETO JSON PARA CADA DELIBERAÇÃO: Para cada assunto que você identificar, crie um objeto JSON.
    3. RETORNE UMA LISTA: Sua resposta final DEVE ser um objeto JSON com uma chave "deliberacoes", cujo valor é a LISTA de todos os objetos JSON que você criou.

    Para CADA objeto JSON, extraia os seguintes campos:
    - "classe": (ESSENCIAL) Use "CONSULTA", "REPRESENTAÇÃO", etc. para processos formais. Para outros atos, discussões ou deliberações, use "ATOS E EXPEDIENTES".
    - "punchline": (ESSENCIAL) Um resumo de UMA ÚNICA FRASE, clara e direta, da deliberação.
    - "decisão": (ESSENCIAL E DETALHADO) Um resumo completo e informativo, compreensível por si só.
    - "numero_do_processo": Apenas os dígitos do número do processo, se houver.
    - "data_da_decisão": (ESSENCIAL) {date_instruction}
    - "numero_da_sessão": Use o valor "{session_num}".
    - "UF": A sigla da Unidade Federativa (UF), se mencionada.
    - "relator": Identifique o nome do ministro que relata o processo, apresenta a consulta ou propõe a deliberação principal. O nome deve ser completo. {ministros_prompt_part}. Se nenhum ministro for claramente identificado nesta função, deixe este campo em branco.
    - "parte_recorrente": Identifique o nome da pessoa, partido ou entidade que inicia ou apresenta o processo (recorrente, representante, consulente, etc). Em 'ATOS E EXPEDIENTES', pode ser o proponente principal da matéria, se houver. Se não houver ou não for claro, deixe este campo em branco. Não acrescente  '(', '/', '-' ou outros detalhes ao nome detectado.
    - "parte_recorrida": Identifique o nome da pessoa, partido ou entidade contra quem o processo é apresentado (recorrido, representado, impetrado, etc). Este campo geralmente não se aplica a 'CONSULTAS' ou 'ATOS E EXPEDIENTES'. Se não houver ou não for aplicável, deixe este campo em branco. Não acrescente  '(', '/', '-' ou outros detalhes ao nome detectado.
    - "interessado": Liste os nomes de quaisquer outras partes que participam do processo mas não são recorrente ou recorrida (ex: terceiros interessados, litisconsortes, ou outras pessoas/entidades indetificadas no feito). Separe múltiplos nomes por vírgula. Se não houver, deixe este campo em branco. Não confundir com os juízes/ministros responsáveis pela deliberação.
    - "ementa": (ESSENCIAL) O trecho de texto exato e literal, extraído do "TEXTO COMPLETO DA SESSÃO" acima, que corresponde especificamente a esta única deliberação.
    - "quorum": (ESSENCIAL E CONSISTENTE) Encontre a lista de todos os ministros presentes na sessão (geralmente no início do texto) e REPITA essa mesma lista de nomes para TODOS os objetos JSON que você gerar. Todos os registros desta sessão devem ter o mesmo quorum. NECESSARIAMENTE individualize os nomes e separe múltiplos nomes por vírgula. Normalize os nomes para o formato abreviado, por exemplo, "Carvalho Mourão". Ignore títulos como "Dr.", "Sr.", "Ministro", "desembargador", etc. Todas as sessões devem contar com o nome do presidente "Hermenegildo de Barros", que sempre presidia as sessões.
    - "votação": (SE APLICÁVEL) Se o texto indicar claramente como cada ministro votou (por exemplo, "o ministro X votou contra" ou "o desembargador Y divergiu"), liste os nomes abreviados e padronizados dos que votaram contra, separados por vírgula. Caso a decisão tenha sido por unanimidade ou sem divergência, registre "Unânime". Se não houver detalhes de votação ou tratar-se de um simples registro sem cunho decisório, deixe este campo em branco.
    """
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "Você é um especialista em documentos jurídicos antigos do Brasil. Sua tarefa principal é decompor o texto completo de uma ata de sessão em múltiplas deliberações individuais e retornar um array JSON sob a chave 'deliberacoes'."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("deliberacoes", [])
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            delay = INITIAL_RETRY_DELAY * (2 ** attempt)
            tqdm_asyncio.write(f"{C.YELLOW}AVISO: Erro de conexão/timeout na API para a sessão #{session_index+1}. Tentativa {attempt + 1}/{MAX_RETRIES}. Tentando novamente em {delay}s...{C.END}")
            await asyncio.sleep(delay)
        except Exception as e:
            tqdm_asyncio.write(f"{C.RED}ERRO INESPERADO na API para a sessão #{session_index+1}: {e}{C.END}")
            return [{"error": f"{type(e).__name__}: {e}", "session_text": session_text}]
    
    return [{"error": f"Erro de API: A requisição falhou após {MAX_RETRIES} tentativas.", "session_text": session_text}]


async def process_session_worker(job: Dict[str, Any], semaphore: asyncio.Semaphore, ministros_list: List[str]) -> Dict[str, Any]:
    """Worker que processa uma sessão inteira."""
    async with semaphore:
        session_text = job['session_text']
        llm_results = await extract_data_with_llm(job['index'], session_text, job['date'], job['num'], ministros_list)
        
        # Pós-processamento para garantir quórum consistente e limpo
        if llm_results and "error" not in llm_results[0]:
            master_quorum = ""
            for res in llm_results:
                quorum_value = res.get("quorum")
                if quorum_value:
                    if isinstance(quorum_value, list):
                        # Converte lista para string
                        master_quorum = ", ".join(map(str, quorum_value))
                    else:
                        # Garante que é uma string limpa
                        master_quorum = _norm_spaces(str(quorum_value))
                    break  # Para após encontrar o primeiro quórum válido
            
            if master_quorum:
                # Aplica o quórum limpo e consistente a todos os resultados da sessão
                for res in llm_results:
                    res["quorum"] = master_quorum
        
        processed_rows = []
        for llm_data in llm_results:
            if "error" in llm_data:
                 return {"original_index": job["index"], "rows": [], "error": llm_data["error"]}

            extracted_relator = llm_data.get("relator", "")
            official_relator = find_best_match(extracted_relator, ministros_list) if extracted_relator else ""
            
            final_date = llm_data.get("data_da_decisão") or job['date'] or "Data não encontrada"

            row_data = {
                "classe": llm_data.get("classe", ""),
                "punchline": llm_data.get("punchline", ""),
                "decisão": llm_data.get("decisão", ""),
                "numero_do_processo": llm_data.get("numero_do_processo", ""),
                "data_da_decisão": final_date,
                "numero_da_sessão": llm_data.get("numero_da_sessão", job['num']),
                "UF": llm_data.get("UF", ""), "relator": official_relator,
                "parte_recorrente": llm_data.get("parte_recorrente", ""),
                "parte_recorrida": llm_data.get("parte_recorrida", ""),
                "interessado": llm_data.get("interessado", ""),
                "inteiro_teor": _norm_spaces(session_text),
                "ementa": _norm_spaces(llm_data.get("ementa", "")),
                "quorum": llm_data.get("quorum", ""),
                "votação": llm_data.get("votação", ""),
            }
            processed_rows.append(row_data)

        return {"original_index": job["index"], "rows": processed_rows, "error": None}


async def process_docx_to_csv(input_path: Path, out_csv: Path, ministros_list: List[str]) -> None:
    print(f"{C.CYAN}  → Lendo e preparando as sessões do arquivo...{C.END}")
    sessions = extract_sessions_from_docx(input_path)
    
    jobs = []
    for i, sess in enumerate(sessions):
        session_date = normalize_date_pt(sess)
        session_num = extract_sessao_num(sess)
        jobs.append({"index": i, "session_text": sess, "date": session_date, "num": session_num})

    if not jobs:
        print(f"{C.YELLOW}  AVISO: Nenhuma sessão encontrada no arquivo.{C.END}"); return

    CONCURRENT_REQUESTS = 10
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    print(f"\n{C.BOLD}{C.BLUE}--- INICIANDO ANÁLISE CONCORRENTE DE {len(jobs)} SESSÕES ---{C.END}")
    print(f"{C.CYAN}  → Serão feitas até {CONCURRENT_REQUESTS} chamadas à API em paralelo.{C.END}")

    start_time = time.time()
    tasks = [process_session_worker(job, semaphore, ministros_list) for job in jobs]
    results = await tqdm_asyncio.gather(*tasks, desc="Analisando sessões")
    end_time = time.time()

    print(f"\n{C.GREEN}✔ Análise de todas as sessões concluída em {end_time - start_time:.2f} segundos.{C.END}")

    results.sort(key=lambda r: r["original_index"])
    
    final_rows = []
    errors_found = 0
    for result in results:
        if result["error"]:
            tqdm_asyncio.write(f"{C.RED}  ERRO na sessão {result['original_index']+1}: {result['error']}{C.END}")
            errors_found += 1
        final_rows.extend(result["rows"])
        
    print(f"{C.CYAN}  → Salvando {len(final_rows)} registros em {C.BOLD}{out_csv.name}{C.END}...")
    if errors_found > 0:
        print(f"{C.YELLOW}  AVISO: {errors_found} sessões falharam e não foram incluídas no CSV.{C.END}")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_COLS)
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"{C.GREEN}  ✔ Arquivo salvo com sucesso.{C.END}")


async def main():
    start_total_time = time.time()
    print(f"{C.BOLD}{C.BLUE}="*70)
    print(f"AUTOMATIZADOR DE ANÁLISE DE ATAS JURÍDICAS v26.0 (Quorum Limpo)")
    print(f"{C.BLUE}-{C.END}"*70)

    script_dir = Path(__file__).parent
    ministros_list = MINISTROS_LISTA_OFICIAL
    print(f"{C.GREEN}INFO: Usando lista interna com {len(ministros_list)} nomes de ministros.{C.END}")

    docx_files = list(script_dir.glob("*.docx"))
    if not docx_files:
        print(f"{C.YELLOW}AVISO: Nenhum arquivo .docx encontrado.{C.END}"); return

    print(f"{C.GREEN}INFO: {len(docx_files)} arquivo(s) .docx encontrado(s).{C.END}")
    
    for idx, docx_path in enumerate(docx_files):
        print(f"\n{C.BOLD}{C.CYAN}Iniciando processamento do arquivo {idx+1}/{len(docx_files)}: {docx_path.name}{C.END}")
        csv_path = docx_path.with_suffix(".csv")
        try:
            await process_docx_to_csv(docx_path, csv_path, ministros_list)
        except Exception as e:
            print(f"\n{C.RED}ERRO INESPERADO NO FLUXO PRINCIPAL: {e}{C.END}")
            traceback.print_exc()
    
    end_total_time = time.time()
    print(f"\n{C.BLUE}-{C.END}"*70)
    print(f"{C.BOLD}{C.GREEN}Processamento de todos os arquivos concluído!{C.END}")
    print(f"Tempo total de execução: {end_total_time - start_total_time:.2f} segundos.")
    print(f"{C.BOLD}{C.BLUE}="*70 + C.END)

if __name__ == "__main__":
    asyncio.run(main())