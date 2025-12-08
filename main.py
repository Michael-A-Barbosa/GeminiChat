import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai.errors import APIError

# 🎯 Importa a CLASSE de gerenciamento de chat
from chat_manager import GeminiChatManager 

# --- Configuração do FastAPI ---
app = FastAPI(title="Gemini Chat API",
              description="Back-end com Memória Persistente (Redis) e limite de 10 interações.")

# --- Configuração CORS (Permite que o navegador se comunique) ---
from fastapi.middleware.cors import CORSMiddleware
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)
# --- Fim da Configuração CORS ---              

# --- INICIALIZAÇÃO CENTRALIZADA DO SERVIÇO DE CHAT ---
try:
    # Lê as variáveis de ambiente (MELHOR PRÁTICA)
    GEMINI_API_KEY = os.getenv("GEMINI_AK") 
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    if not GEMINI_API_KEY:
        raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada.")
        
    # Inicializa a classe, passando as dependências
    chat_manager = GeminiChatManager(
        api_key=GEMINI_API_KEY, 
        redis_url=REDIS_URL
    )
    
except Exception as e:
    # Se houver erro na inicialização (chave ou Redis), a variável armazena o erro
    chat_manager = None
    API_KEY_LOAD_ERROR = str(e)


# --- Configuração do Modelo de Dados para a Requisição ---
class PromptRequest(BaseModel):
    """Modelo para receber a pergunta e o ID da sessão do front-end."""
    pergunta_cliente: str
    session_id: str


# --- Endpoint 1: POST /chat (Chat com Memória/Redis) ---
@app.post("/chat")
async def chat_with_gemini(request: PromptRequest):
    """
    Processa a requisição do chat usando a classe GeminiChatManager.
    """
    if chat_manager is None:
        raise HTTPException(status_code=500, 
                            detail=f"Erro de configuração: O serviço Gemini não pôde ser inicializado. Detalhe: {API_KEY_LOAD_ERROR}")
            
    pergunta = request.pergunta_cliente
    session_id = request.session_id
    
    if not pergunta or not session_id:
        raise HTTPException(status_code=400, detail="A pergunta e o ID da sessão não podem estar vazios.")

    try:
        # Chama o MÉTODO da classe
        resposta_gemini = chat_manager.send_message(
            session_id=session_id,
            new_prompt=pergunta
        )

        if resposta_gemini.startswith("Erro de Serviço"):
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
        # Isto agora captura erros de API do Gemini (e a chave já está correta se o deploy funcionar)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno no servidor durante a comunicação.")


# --- Endpoint 2: GET /chat/history (Obter Histórico) ---
@app.get("/chat/history")
async def get_history(session_id: str):
    """
    Retorna o histórico de mensagens para a sessão.
    """
    if chat_manager is None:
        raise HTTPException(status_code=500, detail="Serviço de chat indisponível.")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="O ID da sessão não pode estar vazio.")

    # Chama o MÉTODO da classe
    history = chat_manager.get_chat_history_from_redis(session_id)
    
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
    if chat_manager is None:
        raise HTTPException(status_code=500, detail="Serviço de chat indisponível.")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="O ID da sessão não pode estar vazio.")

    # Chama o MÉTODO da classe
    session_deleted = chat_manager.reset_chat_session(session_id)
    
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