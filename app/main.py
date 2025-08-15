"""FastAPI application exposing transcription and chat endpoints."""

from fastapi import FastAPI, File, HTTPException, UploadFile

from .audio import transcribe_audio
from .models import ChatRequest, ChatResponse, TranscriptionResponse
from .vectorstore import VectorStore
from .openai_utils import get_openai_client
from .config import settings

app = FastAPI(title="ElSol Challenge API")

# Vector store compartido en memoria (usa embeddings si están configurados; si no, TF-IDF)
store = VectorStore()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/models")
def debug_models():
    # Útil para verificar que el .env se esté leyendo bien
    return {
        "chat": settings.azure_openai_chat_deployment,
        "transcribe": settings.azure_openai_transcribe_deployment,
        "embeddings": settings.azure_openai_embeddings_deployment,
        "endpoint": settings.azure_openai_api_endpoint,
        "api_version": settings.azure_openai_api_version,
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    """Transcribe an uploaded audio file (.wav or .mp3)."""
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Formato no soportado (usa .wav o .mp3)")
    # transcribe_audio espera (file, store)
    return transcribe_audio(file, store)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer medical questions based on stored transcripts."""
    results = store.query(request.question)
    context = "\n".join(point.payload.get("text", "") for point in results)

    model_name = settings.azure_openai_chat_deployment
    if not model_name:
        raise HTTPException(
            status_code=500,
            detail="Falta configurar 'azure_openai_chat_deployment' en el .env",
        )

    client = get_openai_client()
    messages = [
        {"role": "system", "content": "Eres un asistente médico. Responde de forma clara y segura."},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {request.question}"},
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,  # NOMBRE DEL DEPLOYMENT de chat
            messages=messages,
            temperature=0.7,
        )
        answer = completion.choices[0].message.content or ""
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar Azure OpenAI: {e!s}")

    return ChatResponse(answer=answer)
