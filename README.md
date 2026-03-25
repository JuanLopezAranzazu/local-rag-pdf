# RAG PDF CLI

CLI para hacer preguntas a documentos PDF utilizando Retrieval-Augmented Generation (RAG) con LangChain y Ollama.

## Diagrama de Arquitectura RAG

```mermaid
flowchart LR
    subgraph Ingesta
        A[Fuentes de datos\nPDF, Docs, Web] --> B[Loader]
        B --> C[Text Splitter]
        C --> D[Embeddings]
        D --> E[(Vector DB)]
    end

    subgraph Consulta
        F[Usuario / Pregunta] --> G[Embeddings]
        G --> H[Retriever]
        E --> H
        H --> I[Contexto relevante]
        I --> J[LLM]
        J --> K[Respuesta]
    end
```

Este diagrama muestra el flujo de datos desde la carga del PDF, pasando por el procesamiento y almacenamiento de los embeddings, hasta la generación de respuestas a las preguntas del usuario.

## Entorno virtual

```bash
python -m venv venv
# En Windows
venv\Scripts\activate
# En Unix o MacOS
source venv/bin/activate
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py ruta/al/documento.pdf
```
