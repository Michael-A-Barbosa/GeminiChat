import redis
import json
from google import genai
from google.genai import types
from google.genai.types import Content

# --- Configuração do Redis e Constantes ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
MAX_MESSAGES = 20  # Limite para 10 interações (20 mensagens)
SYSTEM_INSTRUCTION = "Você é um assistente de atendimento ao cliente amigável e prestativo. Mantenha as respostas concisas e focadas no tópico."

# Conexão com o Redis (Compartilhada pelos workers)
R = None
try:
    R = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True) 
    R.ping() 
    print(f"Conexão com Redis estabelecida em {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.ConnectionError as e:
    print(f"ERRO: Não foi possível conectar ao Redis. Detalhe: {e}")
    R = None

MODEL_NAME = "gemini-2.5-flash"

# --- Funções de Conversão ---
def serialize_content(contents: list[Content]) -> list[str]:
    """Serializa uma lista de objetos Content para strings JSON."""
    serialized = []
    for content in contents:
        # Pega o texto da primeira parte (assumindo apenas partes de texto)
        text_part = content.parts[0].text if content.parts and content.parts[0].text else ""
        msg_data = {
            "role": content.role,
            "parts": [{"text": text_part}]
        }
        serialized.append(json.dumps(msg_data))
    return serialized

def deserialize_content(serialized_history: list[str]) -> list[Content]:
    """Deserializa strings JSON de volta para Content objects."""
    history = []
    for item in serialized_history:
        data = json.loads(item)
        content = Content(
            role=data["role"],
            parts=[types.Part(text=part["text"]) for part in data["parts"]]
        )
        history.append(content)
    return history

# --- Função Principal de Comunicação com o Gemini (CORRIGIDA) ---
def send_message_with_history(session_id: str, client: genai.Client, new_prompt: str) -> str:
    """
    Busca o histórico no Redis, envia ao Gemini com a System Instruction via config
    e atualiza o histórico no Redis com limite.
    """
    if R is None:
        return "Erro de Serviço: O banco de dados Redis não está acessível."
        
    redis_key = f"chat:{session_id}"

    # 1. Recuperar Histórico do Redis
    serialized_history = R.lrange(redis_key, 0, -1)
    history = deserialize_content(serialized_history)
    
    # 2. FILTRA o histórico para garantir que apenas roles "user" e "model" sejam passados
    # Isso corrige o erro 'INVALID_ARGUMENT' ao enviar role="system" no histórico
    history = [item for item in history if item.role in ["user", "model"]]
    
    # 3. Prepara o Payload para o Gemini (Histórico + Novo Prompt)
    contents_to_send = history + [Content(role="user", parts=[types.Part(text=new_prompt)])]

    # 4. Define a Configuração, incluindo a System Instruction (MÉTODO CORRETO)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION
    )

    try:
        # 5. Chama a API, passando a instrução do sistema via 'config'
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents_to_send,
            config=config # <--- MUDANÇA CRÍTICA AQUI
        )
        
        response_text = response.text
        
        # 6. Salvar Novo Histórico no Redis (Transação Atômica)
        new_history_items = serialize_content([
            Content(role="user", parts=[types.Part(text=new_prompt)]),
            Content(role="model", parts=[types.Part(text=response_text)])
        ])

        with R.pipeline() as pipe:
            pipe.rpush(redis_key, *new_history_items) # Adiciona no final da lista
            
            # 7. Limitar o Histórico (LTRIM)
            pipe.ltrim(redis_key, -MAX_MESSAGES, -1) 
            pipe.execute()
            
        return response_text

    except Exception as e:
        print(f"ERRO API/REDIS: {e}")
        # Lança a exceção para que o main.py a trate e retorne o erro 500
        raise e

# --- Funções Auxiliares (Reset e Histórico) ---
def reset_chat_session(session_id: str) -> bool:
    """Remove a chave do histórico do Redis."""
    if R is None:
        return False
    
    redis_key = f"chat:{session_id}"
    deleted_count = R.delete(redis_key)
    
    if deleted_count > 0:
        print(f"Sessão de chat {session_id} resetada e removida do Redis.")
        return True
    return False

def get_chat_history_from_redis(session_id: str) -> list[dict]:
    """Busca o histórico no Redis e retorna em um formato amigável para o front-end."""
    if R is None:
        return [{"role": "system", "text": "Erro: Redis não está conectado."}]
        
    redis_key = f"chat:{session_id}"
    serialized_history = R.lrange(redis_key, 0, -1)
    history_content = deserialize_content(serialized_history)
    
    formatted_history = []
    
    for content in history_content:
        if content.role == "system":
            continue
            
        text = content.parts[0].text if content.parts and content.parts[0].text else ""
        
        formatted_history.append({
            "role": content.role,
            "text": text
        })
        
    return formatted_history