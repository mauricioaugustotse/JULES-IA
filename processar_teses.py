import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time

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

# --- Prompts para a IA ---
PROMPT_EXTRACAO_TESE = """
# PAPEL E OBJETIVO
Você é um assistente jurídico altamente qualificado, especializado em análise de jurisprudência do STF. Sua tarefa é analisar o texto de uma ementa de Repercussão Geral e extrair a tese jurídica exata.

# TAREFA
Analise o texto da "Ementa" fornecido abaixo e extraia, **verbatim**, a tese de repercussão geral fixada. A tese é a regra jurídica final e consolidada pelo tribunal.

# REGRAS DE EXTRAÇÃO
1.  **Exatidão Absoluta:** Retorne apenas o texto exato da tese, sem adicionar introduções, explicações ou qualquer texto que não faça parte da tese oficial.
2.  **Foco na Tese:** Ignore o resumo dos fatos, o relatório, os votos dos ministros e outras partes da ementa. Foque exclusivamente na declaração final que começa com "Fixada a seguinte tese:", "Tese:", ou formulação similar.
3.  **Sem Tese Explícita:** Se a ementa não contiver uma tese de repercussão geral claramente fixada (por exemplo, o julgamento ainda está pendente ou a ementa apenas discute a existência de repercussão geral), retorne a string "N/A".

# TEXTO DA EMENTA PARA ANÁLISE
---
{ementa}
---

# SAÍDA ESPERADA
A tese jurídica exata ou "N/A".
"""

PROMPT_GERACAO_JUSTIFICATIVA = """
# PAPEL E OBJETIVO
Você é um advogado e comunicador brilhante. Sua especialidade é traduzir jargão jurídico complexo em explicações claras, concisas e impactantes para um público leigo.

# TAREFA
Com base na "Tese Jurídica" fornecida, crie uma "Justificativa". A justificativa deve ser uma punchline de impacto, explicando a importância e o significado da tese em 60 a 80 palavras.

# REGRAS PARA A JUSTIFICATIVA
1.  **Linguagem Simples:** Evite termos técnicos. Explique o conceito como se estivesse conversando com alguém sem formação em direito.
2.  **Foco no Impacto:** Responda à pergunta: "Por que essa decisão é importante para o cidadão comum ou para a sociedade?"
3.  **Estrutura de Punchline:** Comece com uma afirmação forte e desenvolva a ideia de forma direta e memorável.
4.  **Contagem de Palavras:** Mantenha-se estritamente entre 60 e 80 palavras.

# TESE JURÍDICA
---
{tese}
---

# SAÍDA ESPERADA
Um parágrafo único contendo a justificativa (punchline).
"""

# --- Funções de Processamento ---

def analisar_texto_com_gemini(model, prompt_template: str, **kwargs) -> str:
    """
    Envia um prompt para a API do Gemini e retorna a resposta.
    Inclui tratamento de erros e retentativas.
    """
    prompt = prompt_template.format(**kwargs)
    for _ in range(3):  # Tentar até 3 vezes
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Erro ao chamar a API do Gemini: {e}. Tentando novamente em 5 segundos...")
            time.sleep(5)
    return "ERRO_API"

def processar_dataframe(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Itera sobre o DataFrame, processa cada linha e preenche as colunas necessárias.
    """
    # Adiciona as novas colunas se não existirem
    if "Tese" not in df.columns:
        df['Tese'] = ''
    if "Justificativa" not in df.columns:
        df['Justificativa'] = ''
    if "Resultado" not in df.columns:
        df['Resultado'] = ''

    # Itera sobre as linhas do DataFrame
    for index, row in df.iterrows():
        print(f"Processando linha {index + 1}/{len(df)}...")
        ementa = row['Ementa']

        # 6. Regra Especial
        if 'há repercussão geral' in str(ementa).lower():
            df.loc[index, 'Resultado'] = "Aguardando julgamento"
            df.loc[index, 'Tese'] = "Repercussão geral reconhecida, mas mérito pendente de julgamento."
            df.loc[index, 'Justificativa'] = ""
            continue

        # 3. Extrair a Tese
        tese_extraida = analisar_texto_com_gemini(model, PROMPT_EXTRACAO_TESE, ementa=ementa)
        df.loc[index, 'Tese'] = tese_extraida

        # 4. Gerar Justificativa e 5. Classificar Resultado
        if tese_extraida not in ["N/A", "ERRO_API"]:
            justificativa = analisar_texto_com_gemini(model, PROMPT_GERACAO_JUSTIFICATIVA, tese=tese_extraida)
            df.loc[index, 'Justificativa'] = justificativa
            df.loc[index, 'Resultado'] = "Tese fixada"
        else:
            df.loc[index, 'Justificativa'] = ""
            df.loc[index, 'Resultado'] = "Infraconstitucional" # Ou outra classificação padrão

    return df

# --- Função Principal ---

def main():
    """
    Orquestra o processo de leitura, processamento e salvamento dos dados.
    """
    print("Iniciando o script de processamento de teses...")
    try:
        # 1. Configurar API
        model = configurar_api()

        # 2. Ler o arquivo CSV
        caminho_csv = 'copia_RG.csv'
        print(f"Lendo o arquivo '{caminho_csv}'...")
        df = pd.read_csv(caminho_csv)

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

        # 7. Salvar o novo arquivo CSV
        caminho_saida = 'teses_processadas.csv'
        print(f"Salvando o resultado em '{caminho_saida}'...")
        df_processado.to_csv(caminho_saida, index=False, encoding='utf-8-sig')

        print("Processamento concluído com sucesso!")

    except FileNotFoundError:
        print(f"ERRO: O arquivo 'copia_RG.csv' não foi encontrado. Verifique se o arquivo está no mesmo diretório do script.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    main()
