# Documento Técnico

## A. Análisis del requerimiento

- Se requiere transcribir audios en formato `.wav` o `.mp3`.
- La información debe estructurarse para consultas posteriores mediante un chatbot.
- Se eligió **Azure OpenAI** para transcripción, embeddings y respuestas porque está disponible y ofrece modelos de alta calidad.
- Se usa **FastAPI** para exponer endpoints REST.
- Se emplea **Qdrant** como base vectorial en memoria para almacenar las transcripciones y sus metadatos.

## B. Arquitectura propuesta

```
[Cliente] --(audio)-->/transcribe --(transcripción + extracción)--> [VectorStore]
[Cliente] --(pregunta)-->/chat --(búsqueda+LLM)--> respuesta
```

- **FastAPI** actúa como orquestador.
- **Azure OpenAI** Whisper transcribe el audio.
- **Qdrant** almacena embeddings y metadatos.
- El endpoint `/chat` recupera información relevante y genera la respuesta con un modelo de chat.

## C. Plan de desarrollo

### Entrega MVP
- Endpoint `/transcribe` funcional con almacenamiento en Qdrant.
- Endpoint `/chat` que responde preguntas usando las transcripciones almacenadas.
- Pruebas unitarias básicas y ejemplo de prueba en vivo contra Azure OpenAI.

### Funcionalidades futuras
- Transcripción en tiempo real y diferenciación de hablantes.
- Soporte para documentos PDF e imágenes con OCR.
- Autenticación y autorización de usuarios.
- Persistencia de Qdrant en disco o servicio gestionado.
- Pipelines de MLOps para versionar modelos y datos.

### Plan de producción
- Despliegue en contenedores (Docker/Kubernetes).
- Uso de Azure OpenAI administrado con claves seguras y rotación.
- Qdrant administrado o PostgreSQL con `pgvector` para persistencia.
- Integración de monitoreo, logging y métricas.
- Configuración de CI/CD y pruebas automatizadas.
