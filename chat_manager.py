import os
import redis
import json
from google import genai
from google.genai import types
from google.genai.types import Content

# --- Configurações Comuns ---
MAX_MESSAGES = 20  # Limite para 10 interações (20 mensagens)
MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = "Você é um assistente de atendimento ao cliente amigável e prestativo. Mantenha as respostas concisas e focadas no tópico."

# --- Classe Central de Gerenciamento do Chat ---
class GeminiChatManager:
    """Gerencia a conexão com o Gemini e o histórico de chat no Redis."""

    def __init__(self, api_key: str, redis_url: str):
        self.redis_url = redis_url
        self.client = None
        self.R = None
        
        # 1. Inicializa o Cliente Gemini
        try:
            self.client = genai.Client(api_key=api_key)
            # A API KEY está carregada, mas não há um 'ping' de autenticação aqui.
        except Exception as e:
            # Captura qualquer erro de inicialização do cliente
            raise ValueError(f"Falha na inicialização do Cliente Gemini: {e}")

        # 2. Inicializa o Redis
        try:
            self.R = redis.from_url(self.redis_url, decode_responses=True)
            self.R.ping() 
            print("Conexão com Redis estabelecida com sucesso via URL.")
        except redis.exceptions.ConnectionError as e:
            print(f"ERRO: Não foi possível conectar ao Redis. Histórico não será persistido. Detalhe: {e}")
            self.R = None # Se falhar, o Redis fica desabilitado


    # --- Funções Internas de Serialização (Mantidas) ---
    def _serialize_content(self, contents: list[Content]) -> list[str]:
        """Serializa uma lista de objetos Content para strings JSON."""
        serialized = []
        for content in contents:
            text_part = content.parts[0].text if content.parts and content.parts[0].text else ""
            msg_data = {"role": content.role, "parts": [{"text": text_part}]}
            serialized.append(json.dumps(msg_data))
        return serialized

    def _deserialize_content(self, serialized_history: list[str]) -> list[Content]:
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
    
    # --- MÉTODOS PÚBLICOS PARA USO NO FASTAPI ---
    
    def send_message(self, session_id: str, new_prompt: str) -> str:
        """
        Busca o histórico, envia ao Gemini e atualiza o histórico no Redis.
        """
        if self.R is None:
            return "Erro de Serviço: O banco de dados Redis não está acessível."
        
        redis_key = f"chat:{session_id}"

        # 1. Recuperar Histórico do Redis
        serialized_history = self.R.lrange(redis_key, 0, -1)
        history = self._deserialize_content(serialized_history)
        
        # 2. FILTRA o histórico (garantindo que roles "user" e "model" sejam passados)
        history = [item for item in history if item.role in ["user", "model"]]
        
        # 3. Prepara o Payload para o Gemini (Histórico + Novo Prompt)
        contents_to_send = history + [Content(role="user", parts=[types.Part(text=new_prompt)])]

        # 4. Define a Configuração (System Instruction)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
        )

        try:
            # 5. Chama a API
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=contents_to_send,
                config=config
            )
            
            response_text = response.text
            
            # 6. Salvar Novo Histórico no Redis (Transação Atômica)
            new_history_items = self._serialize_content([
                Content(role="user", parts=[types.Part(text=new_prompt)]),
                Content(role="model", parts=[types.Part(text=response_text)])
            ])

            with self.R.pipeline() as pipe:
                pipe.rpush(redis_key, *new_history_items) # Adiciona no final
                pipe.ltrim(redis_key, -MAX_MESSAGES, -1) # Limita
                pipe.execute()
                
            return response_text

        except Exception as e:
            # Lança a exceção para que o main.py a trate e retorne o erro 500
            print(f"ERRO API/REDIS: {e}")
            raise e

    def reset_chat_session(self, session_id: str) -> bool:
        """Remove a chave do histórico do Redis."""
        if self.R is None:
            return False
        
        redis_key = f"chat:{session_id}"
        deleted_count = self.R.delete(redis_key)
        
        if deleted_count > 0:
            print(f"Sessão de chat {session_id} resetada e removida do Redis.")
            return True
        return False

    def get_chat_history_from_redis(self, session_id: str) -> list[dict]:
        """Busca o histórico no Redis e retorna em um formato amigável."""
        if self.R is None:
            return [{"role": "system", "text": "Erro: Redis não está conectado."}]
            
        redis_key = f"chat:{session_id}"
        serialized_history = self.R.lrange(redis_key, 0, -1)
        history_content = self._deserialize_content(serialized_history)
        
        formatted_history = []
        for content in history_content:
            if content.role == "system":
                continue
            text = content.parts[0].text if content.parts and content.parts[0].text else ""
            formatted_history.append({"role": content.role, "text": text})
            
        return formatted_history