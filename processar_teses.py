import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json

# --- 1. CONFIGURAÇÕES ---
MODELO_API = "gemini-2.5-flash"
PAUSA_ENTRE_CHAMADAS_SEG = 6.1
TAMANHO_LOTE = 20
# --- Configuração Inicial ---
def configurar_api():
    """
    Carrega as variáveis de ambiente e configura a API do Google Gemini.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente GOOGLE_API_KEY não foi encontrada. Crie um arquivo .env ou defina a variável de ambiente.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

# --- 2. PROMPT PARA PROCESSAMENTO EM LOTE ---
PROMPT_LOTE = """
# PAPEL E OBJETIVO
Você é um assistente de jurimetria de elite, especializado em analisar e estruturar dados da jurisprudência do STF em lote.

# TAREFA
Eu fornecerei uma lista de objetos JSON, cada um contendo um `id` e o texto de uma `ementa`. Sua tarefa é processar CADA objeto na lista e retornar uma lista de objetos JSON com os resultados, mantendo o `id` original para cada um.

Para cada ementa, você deve realizar três tarefas:
1.  **Extrair a Tese (chave: "tese"):** Extraia a tese jurídica exata da ementa. Se nenhuma tese for explicitamente fixada, retorne "N/A".
2.  **Gerar Justificativa (chave: "justificativa"):** Crie uma explicação concisa e de impacto (60-80 palavras) sobre a importância da tese. Se a tese for "N/A", retorne uma string vazia ("").
3.  **Classificar Resultado (chave: "resultado"):** Use UMA das três tags canônicas a seguir, com base nestas regras precisas:
    - **"Tese fixada":** Use esta tag APENAS quando a ementa apresentar uma tese jurídica clara e definida sobre um tema, sem menções a um simples reconhecimento de repercussão geral.
    - **"Aguardando julgamento":** Use esta tag quando a ementa indicar que "a repercussão foi reconhecida" (ou expressão similar), mas AINDA NÃO houver uma tese de mérito explícita. Isso significa que o tema será julgado no futuro.
    - **"Infraconstitucional":** Use esta tag quando a ementa declarar que o tema "não tem repercussão geral" ou "não possui caráter constitucional" (ou expressões equivalentes).

# REGRAS DE FORMATAÇÃO DA SAÍDA
- A sua resposta deve ser APENAS o objeto JSON contendo uma única chave "resultados", cujo valor é a lista de objetos processados.
- Mantenha a ordem original dos `id`s.
- Não inclua NENHUM texto, explicação ou ```json ``` fora do objeto JSON de resposta.

# EXEMPLO
## ENTRADA:
{{
  "casos": [
    {{"id": 101, "ementa": "Ementa que fixa a tese X..."}},
    {{"id": 102, "ementa": "Ementa que não possui tese de RG..."}}
  ]
}}

## SAÍDA ESPERADA:
{{
  "resultados": [
    {{
      "id": 101,
      "tese": "Tese X...",
      "justificativa": "Esta decisão é crucial pois define...",
      "resultado": "Tese fixada"
    }},
    {{
      "id": 102,
      "tese": "N/A",
      "justificativa": "",
      "resultado": "Infraconstitucional"
    }}
  ]
}}

# DADOS PARA PROCESSAR
---
{json_lote}
---
"""

# --- 3. FUNÇÕES DE CHECKPOINT ---
def carregar_checkpoint(filepath: str) -> tuple[int, list]:
    """Carrega o progresso de um arquivo de checkpoint."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ultimo_indice = data.get('ultimo_indice_processado', -1)
                registros = data.get('registros_salvos', [])
                print(f"Checkpoint encontrado. Retomando do índice {ultimo_indice + 1}.")
                return ultimo_indice + 1, registros
        except (json.JSONDecodeError, IOError) as e:
            print(f"AVISO: Não foi possível ler o checkpoint '{filepath}'. Começando do zero. Erro: {e}")
    return 0, []

def salvar_checkpoint(filepath: str, indice: int, registros: list):
    """Salva o progresso atual em um arquivo de checkpoint."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            checkpoint_data = {
                'ultimo_indice_processado': indice,
                'registros_salvos': registros
            }
            json.dump(checkpoint_data, f, indent=4)
    except IOError as e:
        print(f"ERRO: Falha ao salvar o checkpoint em '{filepath}'. Erro: {e}")

# --- 4. FUNÇÕES DE PROCESSAMENTO ---
def configurar_api():
    """Configura a API do Google Gemini."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("A GOOGLE_API_KEY não foi encontrada. Defina-a em um arquivo .env.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODELO_API)

def analisar_lote_com_gemini(model, lote_json: str) -> list | None:
    """Envia um lote para a API e retorna a lista de resultados."""
    prompt = PROMPT_LOTE.format(json_lote=lote_json)
    for _ in range(3):  # Retentativas
        try:
            response = model.generate_content(prompt)
            # Limpeza para garantir que apenas o JSON seja processado
            json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(json_text)
            return data.get("resultados", [])
        except Exception as e:
            print(f"Erro na chamada da API ou parsing do JSON: {e}. Tentando novamente em {PAUSA_ENTRE_CHAMADAS_SEG}s...")
            time.sleep(PAUSA_ENTRE_CHAMADAS_SEG)
    return None

def processar_dataframe(df: pd.DataFrame, model) -> pd.DataFrame:
    """Função compatível para evitar NameError: retorna o DataFrame recebido.
    
    Observação: os resultados já foram mesclados ao DataFrame antes desta chamada,
    portanto aqui fazemos apenas uma passagem que permite estender a lógica no futuro.
    """
    # Nenhum processamento adicional é necessário no momento; apenas devolve o DataFrame.
    return df

def main():
    """Função principal que orquestra todo o processo."""
    print("Iniciando o script de processamento de teses em lote...")

    try:
        model = configurar_api()

        caminho_csv = 'copia_RG.csv'
        df = pd.read_csv(caminho_csv)

        checkpoint_file = 'processamento_checkpoint.json'
        indice_inicial, todos_os_resultados = carregar_checkpoint(checkpoint_file)

        df_para_processar = df.iloc[indice_inicial:]

        for i in range(0, len(df_para_processar), TAMANHO_LOTE):
            lote_df = df_para_processar.iloc[i:i + TAMANHO_LOTE]

            if lote_df.empty:
                continue

            indice_inicio_lote_global = indice_inicial + i
            print(f"\nProcessando lote de {len(lote_df)} linhas (Índice Global: {indice_inicio_lote_global} a {indice_inicio_lote_global + len(lote_df) - 1})...")

            lote_para_api = []
            resultados_lote_atual = {}

            # Pré-processamento foi removido para centralizar a lógica na API.
            for index, row in lote_df.iterrows():
                ementa = str(row.get('Ementa', ''))
                lote_para_api.append({"id": int(index), "ementa": ementa})

            # Chamada à API apenas se houver itens no lote
            if lote_para_api:
                json_para_api = json.dumps({"casos": lote_para_api}, indent=2)
                resultados_api = analisar_lote_com_gemini(model, json_para_api)

                if resultados_api:
                    for res in resultados_api:
                        original_id = res.get('id')
                        if original_id is not None:
                            resultados_lote_atual[original_id] = {
                                "Tese": res.get("tese", "ERRO_API"),
                                "Justificativa": res.get("justificativa", ""),
                                "Resultado": res.get("resultado", "Infraconstitucional")
                            }
                else:
                    print(f"ERRO: O lote a partir do índice {indice_inicio_lote_global} falhou após múltiplas tentativas.")
                    # Marcar itens com erro para não serem perdidos
                    for item in lote_para_api:
                        resultados_lote_atual[item['id']] = {"Tese": "ERRO_LOTE", "Justificativa": "", "Resultado": "Infraconstitucional"}

            # Adicionar resultados processados à lista principal
            for index, data in resultados_lote_atual.items():
                data['original_index'] = index
                todos_os_resultados.append(data)

            # Salvar checkpoint ao final de cada lote processado
            ultimo_indice_processado_neste_lote = indice_inicio_lote_global + len(lote_df) - 1
            salvar_checkpoint(checkpoint_file, ultimo_indice_processado_neste_lote, todos_os_resultados)
            print(f"  -> Lote processado. Checkpoint salvo. Pausando por {PAUSA_ENTRE_CHAMADAS_SEG} segundos...")
            time.sleep(PAUSA_ENTRE_CHAMADAS_SEG)

        # Processo final: criar e salvar o CSV
        print("\nProcessamento de todos os lotes concluído. Gerando CSV final...")

        if not todos_os_resultados:
            print("Nenhum resultado foi gerado.")
            return

        # Criar um DataFrame a partir dos resultados
        res_df = pd.DataFrame(todos_os_resultados)
        res_df = res_df.set_index('original_index')

        # Juntar os resultados com o DataFrame original
        df.loc[res_df.index, 'Tese'] = res_df['Tese']
        df.loc[res_df.index, 'Justificativa'] = res_df['Justificativa']
        df.loc[res_df.index, 'Resultado'] = res_df['Resultado']
        # --- ADICIONE ESTAS LINHAS AQUI (PARA CORRIGIR O AVISO) ---
        print("Garantindo que as colunas de saída sejam do tipo 'texto'...")
        df['Tese'] = df['Tese'].astype(object)         # Garante que a Coluna G (Tese) aceite texto
        df['Resultado'] = df['Resultado'].astype(object)   # Garante que a Coluna F (Resultado) aceite texto
        df['Justificativa'] = pd.Series(dtype=object)  # Cria a Coluna M (Justificativa) como texto
        # --- FIM DA ADIÇÃO ---

        # Mapeamento de colunas (H -> Ementa, G -> Tese, F -> Resultado, M -> Justificativa)
        # O script assume que as colunas já existem ou as cria.
        # A lógica de manipulação de colunas específicas por letra (F, G, H, M) é mais
        # segura se feita pelo nome da coluna. O código está adaptado para usar nomes.

        # 3. Processar o DataFrame
        print("Iniciando a análise com a API do Gemini...")
        df_processado = processar_dataframe(df, model)

        caminho_saida = 'teses_processadas.csv'
        df.to_csv(caminho_saida, index=False, encoding='utf-8-sig')
        print(f"Arquivo '{caminho_saida}' salvo com sucesso!")

        # Limpar o arquivo de checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("Arquivo de checkpoint removido.")

    except FileNotFoundError:
        print(f"ERRO: O arquivo '{caminho_csv}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()
