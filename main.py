import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai.errors import APIError

# 🎯 Importa as funções de comunicação com o Redis/Gemini
from chat_memory import send_message_with_history, reset_chat_session, get_chat_history_from_redis 

key = os.getenv("GEMINI_AK") 
if not key:
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada.")

# --- Configuração de Chave - local ---
# KEY_FILE_PATH = "keys.txt"
MODEL_NAME = "gemini-2.5-flash"

# --- Função de Carregamento da Chave ---
def load_api_key(file_path: str) -> str:
    """Carrega a chave da API do Gemini a partir de um arquivo de texto."""
    try:
        with open(file_path, 'r') as f:
            key = f.read().strip()
        if not key:
            raise ValueError("O arquivo keys.txt está vazio.")
        return key
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo de chave não encontrado: {file_path}")
    except Exception as e:
        raise Exception(f"Erro ao ler o arquivo de chave: {e}")

# --- Configuração do FastAPI ---
app = FastAPI(title="Gemini Chat API",
              description="Back-end com Memória Persistente (Redis) e limite de 10 interações.")

# --- NOVO: Configuração CORS (Permite que o navegador se comunique) ---
from fastapi.middleware.cors import CORSMiddleware

# Permitir todas as origens (ideal para desenvolvimento)
origins = [
    "*", # Permite qualquer domínio (incluindo o seu arquivo local "file://")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Lista de origens permitidas
    allow_credentials=True,
    allow_methods=["*"], # Permitir todos os métodos (GET, POST, DELETE)
    allow_headers=["*"],
)
# --- Fim da Configuração CORS ---              

# --- Inicialização do Cliente Gemini ---
client = None
API_KEY_LOAD_ERROR = None

try:
    api_key = load_api_key(KEY_FILE_PATH)
    client = genai.Client(api_key=api_key)
except Exception as e:
    API_KEY_LOAD_ERROR = str(e)
    # A mensagem de erro será impressa na inicialização do chat_memory


# --- Configuração do Modelo de Dados para a Requisição ---
class PromptRequest(BaseModel):
    """Modelo para receber a pergunta e o ID da sessão do front-end."""
    pergunta_cliente: str
    session_id: str


# --- Endpoint 1: POST /chat (Chat com Memória/Redis) ---
@app.post("/chat")
async def chat_with_gemini(request: PromptRequest):
    """
    Processa a requisição do chat usando o Redis para manter o histórico compartilhado.
    """
    if client is None:
        raise HTTPException(status_code=500, 
                            detail=f"Erro de configuração: O serviço Gemini não pôde ser inicializado. Detalhe: {API_KEY_LOAD_ERROR}")
            
    pergunta = request.pergunta_cliente
    session_id = request.session_id
    
    if not pergunta or not session_id:
        raise HTTPException(status_code=400, detail="A pergunta e o ID da sessão não podem estar vazios.")

    try:
        # Chama a função que lida com o Redis e o Gemini
        resposta_gemini = send_message_with_history(
            session_id=session_id,
            client=client,
            new_prompt=pergunta
        )

        # Trata erros retornados pela função de memória
        if resposta_gemini.startswith("Erro"):
            raise HTTPException(status_code=500, detail=resposta_gemini)
                
        return {
            "status": "success",
            "session_id": session_id,
            "resposta_ia": resposta_gemini
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"ERRO INTERNO NO CHAT: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno no servidor durante a comunicação.")


# --- Endpoint 2: GET /chat/history (Obter Histórico) ---
@app.get("/chat/history")
async def get_history(session_id: str):
    """
    Retorna o histórico de mensagens (limitado a 10 interações) para a sessão.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="O ID da sessão não pode estar vazio.")

    history = get_chat_history_from_redis(session_id)
    
    return {
        "status": "success",
        "session_id": session_id,
        "history": history
    }


# --- Endpoint 3: DELETE /chat/reset (Resetar Sessão) ---
@app.delete("/chat/reset")
async def reset_chat(session_id: str):
    """
    Remove uma sessão de chat específica, apagando seu histórico.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="O ID da sessão não pode estar vazio.")

    session_deleted = reset_chat_session(session_id)
    
    if session_deleted:
        return {
            "status": "success",
            "message": f"Sessão {session_id} resetada com sucesso."
        }
    else:
        return {
            "status": "success",
            "message": f"Sessão {session_id} não encontrada. Nenhuma ação necessária."
        }