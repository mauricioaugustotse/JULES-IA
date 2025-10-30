# -*- coding: utf-8 -*-
"""
TEMAS_SELC_txt_to_csv_v8.py

Foco: Correção na lógica de retomada (checkpoint).
A v8 ajusta a condição que determinava se um arquivo deveria ser pulado.
Agora, o script só ignora um arquivo .txt se o .csv de resultado correspondente
já existir E não houver um arquivo de checkpoint pendente. Isso garante que
processamentos interrompidos (onde ambos os arquivos podem existir) sejam
corretamente retomados em vez de pulados.

Uso:
    python TEMAS_SELC_txt_to_csv_v8.py
"""

import os
import re
import json
import pandas as pd
import openai
import time
from dotenv import load_dotenv
import glob
from typing import List, Dict, Any, Tuple
import traceback

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------- Regex de final de julgado ------------------------------
PADRAO_FINAL_JULGADO = re.compile(
    r"""
    \(
      \s*
      (?:Ac\.|Res\.|EDcl|Embargos?|AgR(?:-[A-Za-z]+)?|MS|REspE?|REspe|REsp|RO-?El|RMS|RCEd|AE|TutCautAnt|MC|AAg|Rec\.?)
      [\s\S]*?
      rel\.\s*(?:Min\.|Des\.|Juiz(?:a)?|in\.)[\s\S]*?
    \)
    \s*
    """,
    re.IGNORECASE | re.VERBOSE
)

# -------------------- Prompt para IA (v6) -----------------------------------
PROMPT_EXTRACAO = """
Você é um assistente de jurimetria especializado em jurisprudência eleitoral brasileira.
Sua tarefa é extrair informações de um trecho de um julgado do TSE.

REGRA GERAL:
1.  Analise a citação final do julgado. Retorne uma LISTA de objetos JSON, um para cada processo encontrado. Cada objeto deve ter: "tipo_de_processo", "numero_processo", "relator", "data_julgamento". Se houver apenas um, retorne uma lista com um único objeto. PREENCHA TODOS OS CAMPOS PARA CADA CASO.

REGRAS ESPECÍFICAS (SCHEMA JSON):
1.  **tipo_de_processo**: A classificação processual do expediente (CAMPO CRÍTICO). É uma sigla (ex: "AREspE", "ED-REspe", "AgRgAg", "AgRgREspe") que aparece próximo ao número do processo na referência final. (Tipo: string)
2.  **numero_processo**: O número do processo. (CAMPO CRÍTICO). Procure por um padrão numérico, como "060023641", "82911", etc. Está na referência final entre parênteses. (Tipo: string)
3.  **relator**: O nome do relator ou relatora da decisão. Sempre usar "Min." antes do nome (ex.: Min. Luiz Fux). (Tipo: string)
4.  **data_julgamento**: A data do julgamento, no formato DD-MM-AAAA. (Tipo: string)
5.  **subramo**: Liste os subramos jurídicos discutidos, com base na taxonomia fornecida. Separe-os por vírgulas. (Tipo: string)
6.  **contexto**: Descreva o caso concreto, com menção à legislação citada e à jurisprudência/precedentes mencionados. (Tipo: string)
7.  **tese**: Extraia a tese jurídica principal ou a 'ratio decidendi', com a referência à legislação e à jurisprudência adotada. (Tipo: string)
8.  **bullet_points**: Liste os principais pontos e fundamentos legais em 3 a 5 bullets. (Tipo: string)
9.  **punchline**: Crie uma frase de efeito, curta e de alto impacto, resumindo a decisão. (Tipo: string)
10. **tema**: O título principal que resume o assunto tratado em até 20 palavras. (Tipo: string)
11. **texto_original**: O texto exato do julgado. (Tipo: string)

# REGRAS DE FORMATAÇÃO E EXCEÇÕES
1. Preste atenção especial para distinguir "contexto" (a história do caso) de "tese" (a decisão/regra jurídica).

# REGRAS DE FOCO ESPECIAL
- Os campos `tipo_de_processo`, `numero_processo`, `relator` e `data_julgamento` são muito importantes. Eles quase sempre estão juntos na referência final do julgado, dentro dos parênteses. Analise essa referência com atenção máxima.
- Se você não encontrar um valor claro para eles, e apenas para eles, faça uma segunda verificação no texto antes de retornar `null`.

Julgado a ser analisado:
---
{texto_julgado}
---

Retorne a resposta APENAS em formato JSON.
Exemplo de saída para um julgado com múltiplas citações:
{{
  "casos": [
    {{
      "tipo_de_processo": "AgR-MS",
      "numero_processo": "57264",
      "relator": "Marcelo Ribeiro",
      "data_julgamento": "12-05-2011"
    }},
    {{
      "tipo_de_processo": "MS",
      "numero_processo": "168383",
      "relator": "Cármen Lúcia",
      "data_julgamento": "14-02-2012"
    }}
  ],
  "tema": "Mitigação de prazos em eleições suplementares",
  "subramo": ["Prazo processual"],
  "contexto": "Descrição do caso...",
  "tese": "A tese jurídica firmada foi...",
  "bullet_points": "• Ponto 1\\n• Ponto 2",
  "punchline": "Frase de efeito.",
  "texto_original": "Texto exato do julgado..."
}}
"""

# -------------------- Funções de Checkpoint -------------------------

def carregar_checkpoint(filepath: str) -> Tuple[int, List[Dict]]:
    """
    Carrega o estado de um arquivo de checkpoint.
    Retorna o índice do próximo julgado a ser processado e a lista de registros já salvos.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ultimo_indice = data.get('ultimo_julgado_processado_idx', -1)
                registros = data.get('registros_salvos', [])
                print(f"  -> Checkpoint encontrado. Retomando do julgado #{ultimo_indice + 2}.")
                return ultimo_indice + 1, registros
        except (json.JSONDecodeError, IOError) as e:
            print(f"  -> AVISO: Não foi possível ler o checkpoint '{filepath}'. Começando do zero. Erro: {e}")
            return 0, []
    return 0, []

def salvar_checkpoint(filepath: str, indice: int, registros: List[Dict]):
    """Salva o progresso atual em um arquivo de checkpoint."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            checkpoint_data = {
                'ultimo_julgado_processado_idx': indice,
                'registros_salvos': registros
            }
            json.dump(checkpoint_data, f, indent=4)
    except IOError as e:
        print(f"  -> ERRO: Falha ao salvar o checkpoint em '{filepath}'. Erro: {e}")

# -------------------- Funções Auxiliares ------------------------------------

def carregar_taxonomia(filepath: str) -> Dict[str, str]:
    """Lê o arquivo taxonomia.json e cria um mapa de 'filho' -> 'mãe'."""
    child_to_parent_map = {}
    print(f"Carregando taxonomia de '{filepath}'...")

    def _traverse_json(data_node, parent=None):
        if isinstance(data_node, dict):
            for key, value in data_node.items():
                if parent:
                    child_to_parent_map[key] = parent
                _traverse_json(value, parent=key)
        elif isinstance(data_node, list):
            if parent:
                for child in data_node:
                    child_to_parent_map[child] = parent

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            taxonomia_data = json.load(f)
        _traverse_json(taxonomia_data)
    except FileNotFoundError:
        print(f"AVISO: Arquivo de taxonomia '{filepath}' não encontrado.")
        return {}
    except json.JSONDecodeError:
        print(f"AVISO: Erro ao decodificar o arquivo JSON '{filepath}'.")
        return {}

    if not child_to_parent_map:
        print(f"AVISO: Nenhum mapeamento de taxonomia foi carregado.")
    else:
        print(f"Carregamento da taxonomia concluído. {len(child_to_parent_map)} mapeamentos carregados.")
    return child_to_parent_map

def corrigir_subramo_com_taxonomia(subramos: List[str], taxonomia_mapa: Dict[str, str]) -> List[List[str]]:
    """Corrige a lista de subramos usando o mapa de taxonomia."""
    if not isinstance(subramos, list): return []
    subramos_corrigidos, pais_ja_incluidos = [], set()
    for subramo in subramos:
        pai = taxonomia_mapa.get(subramo)
        item_final = [pai, subramo] if pai else [subramo]
        if pai: pais_ja_incluidos.add(pai)
        if not (len(item_final) == 1 and item_final[0] in pais_ja_incluidos):
            subramos_corrigidos.append(item_final)
    return [list(item) for item in set(map(tuple, [tuple(i) for i in subramos_corrigidos]))]

def extrair_julgados(texto: str) -> List[str]:
    """Divide o texto em julgados individuais."""
    julgados, ultimo_ponto_corte = [], 0
    for match in PADRAO_FINAL_JULGADO.finditer(texto):
        ponto_corte_atual = match.end()
        julgado_completo = texto[ultimo_ponto_corte:ponto_corte_atual].strip()
        if julgado_completo: julgados.append(julgado_completo)
        ultimo_ponto_corte = ponto_corte_atual
    return [j for j in julgados if j]

def _parse_and_normalize_json_response(raw_text: str) -> Dict:
    """Localiza, analisa e normaliza um JSON dentro de um texto."""
    if not raw_text: return {}
    json_str = ""
    if "```json" in raw_text:
        try: json_str = raw_text.split("```json")[1].split("```")[0].strip()
        except IndexError: pass
    if not json_str:
        start, end = raw_text.find('{'), raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = raw_text[start:end+1]
        else:
            print("      -> AVISO: Nenhum objeto JSON encontrado na resposta da IA.")
            return {}
    try: data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"      -> AVISO: Falha no parsing do JSON. Erro: {e}. Conteúdo: '{json_str[:200]}...'")
        return {}
    if not isinstance(data, dict):
        print(f"      -> AVISO: Resposta da IA não é um dicionário. Tipo: {type(data)}")
        return {}
    def normalize_keys(obj: Any) -> Any:
        if isinstance(obj, dict): return {str(k).strip().lower(): normalize_keys(v) for k, v in obj.items()}
        if isinstance(obj, list): return [normalize_keys(elem) for elem in obj]
        return obj
    normalized_data = normalize_keys(data)
    if 'casos' in normalized_data:
        casos_value = normalized_data['casos']
        if isinstance(casos_value, dict): normalized_data['casos'] = [casos_value]
        elif not isinstance(casos_value, list):
            print(f"      -> AVISO: O campo 'casos' não é válido. Removendo. Conteúdo: {casos_value}")
            normalized_data.pop('casos')
    return normalized_data

def extrair_dados_com_ia(julgado: str) -> Dict:
    """Envia um julgado para a API e retorna os dados extraídos."""
    mensagens = [
        {"role": "system", "content": "Você é um assistente de jurimetria especializado em jurisprudência eleitoral brasileira."},
        {"role": "user", "content": PROMPT_EXTRACAO.format(texto_julgado=julgado)}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-5-mini", # Recomendo usar um modelo estável
            messages=mensagens,
            response_format={"type": "json_object"},
        )
        raw_response = response.choices[0].message.content or ""
        if raw_response:
            return _parse_and_normalize_json_response(raw_response)
    except Exception as e:
        print(f"      -> ERRO: Chamada à API falhou: {e}")
    return {}

# -------------------- Main (com lógica de checkpoint corrigida) -----------------------
def main():
    """Função principal para orquestrar o processo."""
    pausa_s = 2
    taxonomia_mapa = carregar_taxonomia("taxonomia.json")

    arquivos_txt = glob.glob("*.txt")
    if "taxonomia.txt" in [os.path.basename(f) for f in arquivos_txt]:
        arquivos_txt = [f for f in arquivos_txt if os.path.basename(f) != "taxonomia.txt"]

    for arquivo_txt in arquivos_txt:
        print(f"\n--- Processando arquivo: {arquivo_txt} ---")
        saida_csv = os.path.splitext(arquivo_txt)[0] + ".csv"
        checkpoint_file = arquivo_txt + ".checkpoint.json"
        
        # LÓGICA DE RETOMADA CORRIGIDA:
        # Pula o arquivo SOMENTE se o CSV de resultado já existe E não há um checkpoint.
        # A ausência do checkpoint indica que o processo foi concluído com sucesso.
        if os.path.exists(saida_csv) and not os.path.exists(checkpoint_file):
            print(f"  -> AVISO: Arquivo de resultado '{saida_csv}' já foi gerado com sucesso anteriormente. Pulando...")
            continue

        try:
            with open(arquivo_txt, "r", encoding="utf-8") as f:
                conteudo = f.read()
        except Exception as e:
            print(f"  -> ERRO: Não foi possível ler o arquivo '{arquivo_txt}'. Erro: {e}")
            continue

        if not conteudo or not conteudo.strip():
            print(f"  -> AVISO: O arquivo '{arquivo_txt}' está vazio. Pulando...")
            continue
        
        julgados = extrair_julgados(conteudo)
        print(f"Encontrados {len(julgados)} julgados para análise.")
        
        # Carrega o progresso anterior (se houver)
        indice_inicial, registros_finais = carregar_checkpoint(checkpoint_file)

        # Itera a partir do índice inicial
        for i in range(indice_inicial, len(julgados)):
            julgado = julgados[i]
            print(f" - Processando Julgado #{i + 1}/{len(julgados)}...")
            try:
                dados = extrair_dados_com_ia(julgado)

                if not dados or 'casos' not in dados or not isinstance(dados.get('casos'), list) or not dados['casos']:
                    print("   -> AVISO: Nenhum caso válido foi extraído pela IA para este julgado.")
                    if dados: print(f"     -> Debug: Chaves recebidas: {list(dados.keys())}")
                    continue
                
                dados_comuns = {k: v for k, v in dados.items() if k != 'casos'}
                
                casos_processados_neste_julgado = 0
                for caso_especifico in dados['casos']:
                    if not isinstance(caso_especifico, dict): continue

                    novo_registro = dados_comuns.copy()
                    novo_registro.update(caso_especifico)
                    
                    subramo_val = novo_registro.get("subramo", [])
                    if isinstance(subramo_val, str): novo_registro["subramo"] = [subramo_val]
                    elif not isinstance(subramo_val, list): novo_registro["subramo"] = []

                    novo_registro["subramo"] = corrigir_subramo_com_taxonomia(novo_registro.get("subramo",[]), taxonomia_mapa)
                    
                    registros_finais.append(novo_registro)
                    casos_processados_neste_julgado += 1

                print(f"   -> OK Julgado #{i + 1} | Extraídos {casos_processados_neste_julgado} casos.")
                
                # Salva o progresso após cada julgado bem-sucedido
                salvar_checkpoint(checkpoint_file, i, registros_finais)

            except Exception as e:
                print(f"   -> ERRO CRÍTICO no Julgado #{i + 1}: Um erro inesperado ocorreu.")
                print(f"   -> Tipo de Erro: {type(e).__name__}, Mensagem: {e}")
                traceback.print_exc()

            time.sleep(pausa_s)

        if not registros_finais:
            print(f"Nenhum dado foi extraído com sucesso do arquivo '{arquivo_txt}'.")
            # Se não há registros, mas o checkpoint existe, remove-o para evitar confusão.
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            continue

        df = pd.DataFrame(registros_finais)
        col_order_lower = [
            "tipo_de_processo", "numero_processo", "relator", "data_julgamento",
            "tema", "subramo", "contexto", "tese", "bullet_points", "punchline", "texto_original"
        ]
        
        for col in col_order_lower:
            if col not in df.columns: df[col] = None

        df = df[col_order_lower + [c for c in df.columns if c not in col_order_lower]]
        df.to_csv(saida_csv, index=False, encoding="utf-8-sig")
        print(f"\nArquivo '{saida_csv}' gerado com sucesso!")

        # Limpa o arquivo de checkpoint após o sucesso
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

if __name__ == "__main__":
    main()