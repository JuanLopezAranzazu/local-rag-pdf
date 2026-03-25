LLM_MODEL    = "phi3"
LLM_TEMPERATURE = 0.1
EMBED_MODEL  = "nomic-embed-text"
CHROMA_DIR   = "./chroma_db"
CHUNK_SIZE   = 1000
CHUNK_OVERLAP = 200
K_DOCS       = 4

PROMPT = """Eres un asistente que responde preguntas sobre un documento PDF.
Usa ÚNICAMENTE el siguiente contexto para responder.
Si no encuentras la respuesta, di "No encontré esa información en el documento."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""