import google.generativeai as genai
import os
from local_secrets import get_secret, load_local_secrets

# 1. Carrega as variáveis do .env e dos arquivos locais ignorados pelo Git
load_local_secrets(base_dir=os.path.dirname(os.path.abspath(__file__)))

# 2. Pega a chave pelo nome configurado no ambiente ou em arquivo local
api_key = get_secret(
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    base_dir=os.path.dirname(os.path.abspath(__file__)),
)

# Verifica se a chave foi carregada corretamente
if not api_key:
    print(
        "ERRO: A chave não foi encontrada. Verifique o .env ou um arquivo local "
        "como Chave_Gemini.txt."
    )
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
