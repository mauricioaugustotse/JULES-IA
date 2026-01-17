import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Carrega as variáveis do arquivo .env (onde sua senha está guardada)
load_dotenv()

# 2. Pega a chave pelo nome que você definiu no arquivo .env
api_key = os.getenv("GEMINI_API_KEY")

# Verifica se a chave foi carregada corretamente
if not api_key:
    print("ERRO: A chave não foi encontrada. Verifique se o arquivo .env está na mesma pasta.")
else:
    # Configura o Gemini com a chave segura
    genai.configure(api_key=api_key)

    print("--- INICIANDO LISTAGEM ---")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Modelo encontrado: {m.name}")
    except Exception as e:
        print(f"Erro ao conectar: {e}")

    print("--- FIM DA LISTAGEM ---")