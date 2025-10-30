import os
import re
import json
import time
import glob
import argparse
from typing import List, Any, Dict, Pattern

import pandas as pd

# ---- OpenAI (compatível com versões antigas e novas) ----
try:
    from dotenv import load_dotenv  # opcional
    load_dotenv()
except ImportError:
    print("dotenv não instalado, pulando o carregamento de variáveis de .env")
except Exception as e:
    print(f"Erro ao carregar .env: {e}")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não encontrado nas variáveis de ambiente (.env).")

# Tenta importar a nova versão da biblioteca openai, senão usa a antiga
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    import openai
    openai.api_key = OPENAI_API_KEY
    client = None # Flag para usar a sintaxe antiga


# ===================== PROMPTS =====================
PROMPT_STF = """
# PAPEL E OBJETIVO
Você é um assistente de jurimetria treinado para analisar informativos do STF e extrair metadados estruturados em JSON.

# TAREFA
Analise o texto a seguir (um bloco do informativo) e extraia os dados abaixo.
Se o bloco contiver mais de um acórdão (ex.: duas linhas de referência final distintas, como “RE 1.057.258/MG” e “RE 1.037.396/SP”), RETORNE uma LISTA de objetos JSON, um por acórdão.
Se houver apenas um acórdão, retorne um ÚNICO objeto JSON.

# ESQUEMA (CHAVES E TIPOS)
Cada objeto deve conter EXATAMENTE estas chaves:
- "tipo_de_processo": string | null                      # Classe processual (ex.: "ADI", "ADPF", "RE", "ARE")
- "numero_processo": string | null                       # Apenas número (ex.: "7.459/ES" será "7.459", "631.363/SP" será "631.363", "1.057.258/MG" será "1.057.258")
- "UF_Origem_do_caso": string | null                     # Sigla da UF ou origem do caso (ex.: "ES", "MG", "SP", "DF"); null se não houver
- "tribunal": "STF" | "STJ" | "TST" | "TSE" | "STM" | null
- "orgao_julgador": string | null                        # Plenário, 1ª Turma, 2ª Turma etc.
- "relator": string | null                               # Ex.: "Min. Dias Toffoli"
- "data_julgamento": string | null                       # Formato DD-MM-AAAA (converta de DD/MM/AAAA se preciso)
- "informativo": string | null                           # Ex.: "Info 1184"
- "titulo": string | null                                # Título principal (até 20 palavras)
- "tema_repercussao": string | null                      # Ex.: "Tema 284", "Tema 533", "Tema 987" (se houver)
- "contexto": string | null                              # Um resumo robusto do caso prático ou da situação que deu origem à discussão, com menção à legislação citada e à jurisprudência/precedentes mencionados
- "tese": string | null                                  # Regra jurídica/ratio decidendi (clareza e concisão)
- "modulacao": string | null                             # Se houver modulação/limitação de efeitos, resuma; senão null
- "dispositivo": string | null                           # Comandos finais (se existentes); senão null
- "referencias_legais": array of string                  # Normas citadas normalizadas (ex.: "CF/88, art. 170"; "Lei 12.965/2014, art. 19")
- "precedentes_citados": array of string                 # Ex.: "ADPF 165", "RE 631.363". Liste aqui os precedentes mencionados que serviram como panorama/fundamento do caso, explicando-o brevemente no contexto do julgado principal.
- "ramo_direito": string | null                           # Ex.: "Direito Constitucional", "Direito Civil", "Direito Administrativo", "Direito Penal", "Direito Tributário" etc.

# REGRAS DE NORMALIZAÇÃO
- Extraia "tipo_de_processo" e "numero_processo" a partir da(s) linha(s) de referência final do STF (que começam com "STF. ...").
- "tribunal" e "orgao_julgador" saem da mesma linha (ex.: "STF. Plenário. ...").
- Converta datas no formato DD/MM/AAAA para DD-MM-AAAA.
- "informativo": normalize como "Info NNNN" quando houver "(Info NNNN)".
- "tema_repercussao": capte "Tema NNN" quando houver “(Repercussão Geral – Tema NNN)”.
- "referencias_legais": normalize diploma+dispositivo (ex.: "CF/88, art. 5º, XXXV"; "Lei 12.965/2014, art. 19").
- "precedentes_citados": inclua apenas referências explícitas.
- Se não houver informação inequívoca para uma chave, use null (exceto arrays, que devem ser []).

# REGRAS DE FOCO ESPECIAL
- Os campos "tipo_de_processo", "numero_processo", "tribunal", "orgao_julgador", "relator", "data_julgamento", "informativo" e "UF_Origem_do_caso" são muito importantes. Eles quase sempre estão juntos na referência final do julgado. Analise essa referência com atenção máxima.
- Se você não encontrar um valor claro para eles, e apenas para eles, faça uma segunda verificação no texto antes de retornar `null`.

# DIFERENÇA ENTRE "CONTEXTO" E "TESE"
- "contexto" = narrativa fático-processual (o que motivou a controvérsia).
- "tese"     = a regra/ratio decidendi aplicada pelo STF, preferindo formulação geral.

# FORMATO DE SAÍDA
- Se houver mais de um acórdão no bloco: retorne uma LISTA de objetos.
- Caso contrário: retorne um ÚNICO objeto.
- SEM texto fora do JSON.
- A saída DEVE ser JSON válido.

--- TEXTO PARA ANÁLISE ---
{TEXTO_PARA_ANALISE}
--- FIM ---
"""

PROMPT_STJ = """
# PAPEL E OBJETIVO
Você é um assistente de jurimetria treinado para analisar informativos do STJ e extrair metadados estruturados em JSON.

# TAREFA
Analise o texto a seguir (um bloco do informativo) e extraia os dados abaixo.
Se o bloco contiver mais de um acórdão (ex.: várias linhas de referência final do tipo “STJ. 2ª Turma. REsp 2.029.719-RJ, ... (Info 857).”), RETORNE uma LISTA de objetos JSON, um por acórdão.
Se houver apenas um acórdão, retorne um ÚNICO objeto JSON.

# ESQUEMA (CHAVES E TIPOS)
Cada objeto deve conter EXATAMENTE estas chaves:
- "tipo_de_processo": string | null                      # Classe processual (ex.: "REsp", "AREsp", "AgInt no REsp", "AgInt no AREsp", "EDcl", "AgRg")
- "numero_processo": string | null                       # Apenas número (ex.: "2.029.719-RJ" será "2.029.719", "2.163.612-PR" será "2.163.612")
- "UF_Origem_do_caso": string | null                     # Sigla da UF ou origem do caso (ex.: "RJ", "PR", "AL", "SP", "DF"); null se não houver
- "tribunal": "STF" | "STJ" | "TST" | "TSE" | "STM" | null
- "orgao_julgador": string | null                        # Corte Especial, 1ª/2ª/3ª/4ª Turma etc.
- "relator": string | null                               # Ex.: "Min. Marco Aurélio Bellizze"
- "data_julgamento": string | null                       # Formato DD-MM-AAAA (converta de DD/MM/AAAA ou de 'DJe de DD/MM/AAAA')
- "informativo": string | null                           # Ex.: "Info 857"
- "titulo": string | null                                # Título principal (até 20 palavras)
- "tema_repetitivo": string | null                       # Ex.: "Tema 1201" (se houver “Recurso Repetitivo – Tema NNN”). Se não houver, insira a classe processual do recurso (ex.: "REsp", "AREsp" etc.) e número do processo (ex.: "2.029.719")
- "contexto": string | null                              # Um resumo robusto do caso prático ou da situação que deu origem à discussão, com menção à legislação citada e à jurisprudência/precedentes mencionados
- "tese": string | null                                  # Regra jurídica/ratio decidendi
- "modulacao": string | null                             # Se houver modulação/limitação de efeitos, resuma; senão null
- "dispositivo": string | null                           # Comandos finais (se existentes); senão null
- "referencias_legais": array of string                  # Normas citadas normalizadas (ex.: "CF/88, art. 170"; "Lei 8.666/1993, art. 25, III"; "DL 911/1969, art. 2º")
- "precedentes_citados": array of string                 # Ex.: "EDcl no AgInt no AREsp 1.252.262", "REsp 1.924.164". Liste aqui os precedentes mencionados que serviram como panorama/fundamento do caso, explicando-o brevemente no contexto do julgado principal.
- "ramo_direito": string | null                           # Ex.: "Direito Constitucional", "Direito Civil", "Direito Administrativo", "Direito Penal", "Direito Tributário" etc.

# REGRAS DE EXTRAÇÃO E NORMALIZAÇÃO
- Extraia "tipo_de_processo" e "numero_processo" a partir da(s) linha(s) final(is) do bloco (as que começam com "STJ. ...").
- Em "numero_processo", traga exatamente o número.
- "tribunal" e "orgao_julgador" estão na mesma linha (ex.: "STJ. 2ª Turma. ...", "STJ. Corte Especial. ...").
- Converta datas para DD-MM-AAAA, inclusive quando vierem como "DJe de DD/MM/AAAA".
- "informativo": normalize como "Info XXX" quando houver "(Info XXX)".
- "tema_repetitivo": capte "Tema NNN" quando houver “(Recurso Repetitivo – Tema NNN)” ou forma equivalente.
- "referencias_legais": normalize cada diploma+dispositivo.
- "precedentes_citados": inclua apenas referências jurisprudenciais explicitamente citadas.
- Se não houver informação inequívoca para uma chave, use null (exceto arrays, que devem ser []).

# REGRAS DE FOCO ESPECIAL
- Os campos "tipo_de_processo", "numero_processo", "tribunal", "orgao_julgador", "relator", "data_julgamento", "informativo" e "UF_Origem_do_caso" são muito importantes. Eles quase sempre estão juntos na referência final do julgado. Analise essa referência com atenção máxima.
- Se você não encontrar um valor claro para eles, e apenas para eles, faça uma segunda verificação no texto antes de retornar `null`.

# DIFERENÇA ENTRE "CONTEXTO" E "TESE"
- "contexto" = narrativa fático-processual (o que motivou a controvérsia).
- "tese"     = a regra/ratio decidendi aplicada pelo STJ, preferindo formulação geral.

# FORMATO DE SAÍDA
- Se houver mais de um acórdão no bloco: retorne uma LISTA de objetos.
- Caso contrário: retorne um ÚNICO objeto.
- SEM texto fora do JSON.
- A saída DEVE ser JSON válido.

--- TEXTO PARA ANÁLISE ---
{TEXTO_PARA_ANALISE}
--- FIM ---
"""

# ===================== PADRÕES DE REFERÊNCIA FINAL =====================
PADRAO_FINAL_STF: Pattern[str] = re.compile(
    r"""
    ^[\s\d\.\-—]*STF\.\s+
    (?P<orgao>[^.]+)\.\s+
    (?P<classe>[\w\s]+?)\s+
    (?P<numero>[\d\.\-]+(?:/\w{2})?)\s*
    ,\s*Rel\.\s*[^,]+,\s*
    julgado\s+em\s+(?P<data>\d{1,2}/\d{1,2}/\d{4})
    [^()]*?
    \(Info\s*\d+\)
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE
)

PADRAO_FINAL_STJ: Pattern[str] = re.compile(
    r"""
    ^[\s\d\.\-—]*STJ\.\s+
    (?P<orgao>[^.]+)\.?\s*\.\s+
    (?P<texto>.+?)
    \(
        (?:(?:Recurso\s+Repetitivo\s*[-–]\s*Tema\s*\d+)\s*\)\s*\()?
        Info\s*\d+
    \)\.?\s*$
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE
)


# ===================== Utilitários =====================
def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 1)[1]
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()

def _coerce_lists(obj: Dict[str, Any]) -> None:
    for k in ("referencias_legais", "precedentes_citados"):
        if k in obj:
            if obj[k] is None:
                obj[k] = []
            elif isinstance(obj[k], str):
                parts = [p.strip() for p in re.split(r"[;,#]|\s{2,}|,\s*", obj[k]) if p.strip()]
                obj[k] = parts

# --- CORREÇÃO APLICADA AQUI ---
def _call_openai(prompt: str, model: str = "gpt-5-mini") -> str:
    """Chama a API da OpenAI removendo o parâmetro 'temperature' para usar o padrão do modelo."""
    try:
        # A mensagem de erro indicava que o modelo não suporta temperature=0.
        # O parâmetro foi removido para usar o valor padrão da API.
        if client: # Nova sintaxe (openai >= 1.0)
            resp = client.chat.completions.create(
                model=model,
                # temperature=... foi removido daqui
                messages=[
                    {"role": "system", "content": "Responda APENAS com JSON válido."},
                    {"role": "user", "content": prompt},
                ],
            )
            txt = resp.choices[0].message.content
        else: # Sintaxe antiga (openai < 1.0)
            resp = openai.ChatCompletion.create(
                model=model,
                # temperature=... foi removido daqui
                messages=[
                    {"role": "system", "content": "Responda APENAS com JSON válido."},
                    {"role": "user", "content": prompt},
                ],
            )
            txt = resp["choices"][0]["message"]["content"]

        if txt:
            return txt
    except Exception as e:
        raise RuntimeError(f"Falha na chamada à API OpenAI: {e}")
    raise RuntimeError("Resposta vazia da OpenAI.")
# --- FIM DA CORREÇÃO ---

def _auto_detect_tribunal(conteudo: str) -> str:
    stf_hits = len(list(PADRAO_FINAL_STF.finditer(conteudo)))
    stj_hits = len(list(PADRAO_FINAL_STJ.finditer(conteudo)))
    if stf_hits == 0 and stj_hits == 0:
        if "supremo tribunal federal" in conteudo.lower() or "\nSTF." in conteudo:
            return "STF"
        if "superior tribunal de justiça" in conteudo.lower() or "\nSTJ." in conteudo:
            return "STJ"
        return "STF" # Default
    return "STF" if stf_hits >= stj_hits else "STJ"

def _split_em_blocos(conteudo: str, tribunal: str) -> List[str]:
    padrao = PADRAO_FINAL_STF if tribunal == "STF" else PADRAO_FINAL_STJ
    matches = list(padrao.finditer(conteudo))
    if not matches:
        return [conteudo.strip()] if conteudo.strip() else []

    blocos = []
    cursor = 0
    for match in matches:
        # Adiciona o texto ANTES da referência final encontrada
        bloco_anterior = conteudo[cursor:match.start()].strip()
        if bloco_anterior:
            blocos.append(bloco_anterior)
        cursor = match.start()

    # Adiciona o último bloco, que contém a última referência final
    ultimo_bloco = conteudo[cursor:].strip()
    if ultimo_bloco:
        blocos.append(ultimo_bloco)

    # Lógica para agrupar múltiplos acórdãos no mesmo bloco
    blocos_agrupados = []
    temp_bloco = ""
    for bloco in blocos:
        if temp_bloco and not padrao.search(temp_bloco):
            temp_bloco += "\n\n" + bloco
        else:
            if temp_bloco:
                blocos_agrupados.append(temp_bloco)
            temp_bloco = bloco
    if temp_bloco:
        blocos_agrupados.append(temp_bloco)

    return blocos_agrupados

def processar_arquivo_txt(caminho_txt: str,
                          modelo: str = "gpt-5-mini",
                          pausa_segundos: float = 1.0,
                          tribunal_cli: str = "auto") -> List[Dict[str, Any]]:
    with open(caminho_txt, "r", encoding="utf-8") as f:
        bruto = f.read()

    tribunal_detectado = tribunal_cli.upper()
    if tribunal_detectado == "AUTO":
        tribunal_detectado = _auto_detect_tribunal(bruto)

    prompt_base = PROMPT_STF if tribunal_detectado == "STF" else PROMPT_STJ

    blocos = _split_em_blocos(bruto, tribunal_detectado)
    resultados: List[Dict[str, Any]] = []

    print(f"  Tribunal: {tribunal_detectado} | Blocos: {len(blocos)}")

    for i, bloco in enumerate(blocos, 1):
        if not bloco.strip(): continue

        prompt = prompt_base.replace("{TEXTO_PARA_ANALISE}", bloco)
        print(f"  - Chamando API para bloco {i}/{len(blocos)}...")
        raw = _call_openai(prompt, model=modelo) # Chamada sem temperature
        raw = _strip_code_fences(raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"    AVISO: Falha ao decodificar JSON do bloco {i}. Tentando limpar...")
            raw_clean = raw.strip().rstrip(",").replace("'", '"')
            try:
                parsed = json.loads(raw_clean)
            except Exception as e:
                print(f"    ERRO: Limpeza falhou. Bloco ignorado. Erro: {e}\n    Trecho recebido:\n{raw[:500]}")
                continue

        if isinstance(parsed, dict):
            _coerce_lists(parsed)
            resultados.append(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    _coerce_lists(item)
                    resultados.append(item)
        else:
            print(f"    AVISO: Bloco {i} retornou tipo inesperado: {type(parsed)}. Ignorado.")

        time.sleep(pausa_segundos)

    return resultados

def processar_em_lote(
    input_glob: str,
    modelo: str = "gpt-5-mini",
    pausa_segundos: float = 1.0,
    tribunal: str = "auto",
) -> None:
    caminhos = sorted(glob.glob(input_glob))
    if not caminhos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado para o padrão: '{input_glob}' no diretório '{os.getcwd()}'")

    print(f"Encontrados {len(caminhos)} arquivos para processar.")

    for caminho in caminhos:
        print(f"\nProcessando arquivo: {caminho}")
        try:
            res_arquivo = processar_arquivo_txt(
                caminho, modelo=modelo, pausa_segundos=pausa_segundos, tribunal_cli=tribunal
            )

            if not res_arquivo:
                print(f"  => Nenhum registro extraído de '{caminho}'.")
                continue

            print(f"  => {len(res_arquivo)} registro(s) extraído(s).")

            df = pd.DataFrame(res_arquivo)
            schema_cols = [
                "Tipo_de_processo","numero_processo","tribunal","orgao_julgador",
                "relator","data_julgamento","informativo","tema",
                "tema_repercussao","tema_repetitivo",
                "contexto","tese","modulacao","dispositivo",
                "referencias_legais","precedentes_citados","fonte_referencia_final"]

            outras_cols = [c for c in df.columns if c not in schema_cols]
            df = df[[c for c in schema_cols if c in df.columns] + outras_cols]

            def _list_to_str(v: Any) -> str:
                if isinstance(v, list):
                    return ", ".join(map(str, v))
                return v

            for col in ("referencias_legais", "precedentes_citados"):
                if col in df.columns:
                    df[col] = df[col].apply(_list_to_str)

            output_csv = os.path.splitext(caminho)[0] + ".csv"

            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"  => Salvo em: {output_csv}")

        except Exception as e:
            print(f"   ERRO GERAL ao processar '{caminho}': {e}")

    print("\nProcessamento em lote concluído.")


def main():
    parser = argparse.ArgumentParser(description="Extrai metadados de Informativos (STF/STJ) e gera um CSV por arquivo.")
    parser.add_argument("--input", required=False, default="*.txt",
                        help="Caminho ou glob para arquivos .txt (ex.: 'Informativo*.txt' ou 'C:/docs/*.txt')")
    parser.add_argument("--model", required=False, default="gpt-5-mini",
                        help="Modelo OpenAI (ex.: gpt-5-mini, gpt-4o)")
    parser.add_argument("--sleep", required=False, type=float, default=1.0,
                        help="Pausa (segundos) entre chamadas à API")
    parser.add_argument("--tribunal", required=False, default="auto", choices=["auto","STF","STJ"],
                        help="Força o padrão de análise: STF, STJ ou auto-detecção.")
    args = parser.parse_args()

    processar_em_lote(
        input_glob=args.input,
        modelo=args.model,
        pausa_segundos=args.sleep,
        tribunal=args.tribunal
    )

if __name__ == "__main__":
    main()