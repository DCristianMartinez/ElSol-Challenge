"""Audio processing utilities with robust local fallback (Whisper) when Azure transcription is unavailable."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from fastapi import UploadFile

from openai import NotFoundError, BadRequestError, OpenAIError

from .openai_utils import get_openai_client
from .vectorstore import VectorStore
from .models import TranscriptionResponse
from .config import settings


# Parsers muy simples para extraer campos estructurados por línea tipo "Clave: Valor"
_name_re = re.compile(r"(?im)^\s*nombre\s*:\s*(.+?)\s*$")
_age_re = re.compile(r"(?im)^\s*edad\s*:\s*([0-9]+)\s*$")
_dx_re = re.compile(r"(?im)^\s*(diagn[oó]stico|dx)\s*:\s*(.+?)\s*$")


def extract_structured_info(text: str) -> dict[str, str]:
    """Extrae datos básicos (nombre, edad, diagnóstico) del texto transcrito."""
    data: dict[str, str] = {}

    m = _name_re.search(text or "")
    if m:
        data["nombre"] = m.group(1).strip()

    m = _age_re.search(text or "")
    if m:
        data["edad"] = m.group(1).strip()

    m = _dx_re.search(text or "")
    if m:
        data["diagnostico"] = m.group(2).strip()

    return data


def _save_temp(file: UploadFile) -> str:
    """Guarda el UploadFile en un archivo temporal y retorna la ruta."""
    suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            file.file.seek(0)
        except Exception:
            pass
        tmp.write(file.file.read())
        return tmp.name


# ---------------- Azure Transcriptions ----------------

def _transcribe_via_azure(tmp_path: str, model_name: str) -> str:
    """Usa la API nativa de transcripción de Azure (si el deployment existe y es compatible)."""
    client = get_openai_client()
    with open(tmp_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=model_name,  
            file=audio_file,
        )
    return getattr(result, "text", "") or ""

def _transcribe_via_local_whisper(tmp_path: str) -> str:
    """
    Fallback offline usando Whisper local.
    Requiere:
      - pip install -U openai-whisper
      - ffmpeg instalado y en PATH
    """
    try:
        import whisper  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Fallback local no disponible: instala 'openai-whisper' y ffmpeg. "
            "Ejemplo: pip install -U openai-whisper"
        ) from e

    model_name = os.getenv("WHISPER_MODEL", "base")
    try:
        model = whisper.load_model(model_name) 
        result = model.transcribe(tmp_path, language=None)  
        return (result.get("text") or "").strip()
    except Exception as e:
        raise RuntimeError(f"Whisper local falló: {e!s}")


# ---------------- Orquestación ----------------

def transcribe_audio(file: UploadFile, store: VectorStore) -> TranscriptionResponse:
    """
    Transcribe un archivo de audio y guarda el resultado en el VectorStore.

    Flujo:
      1) Si hay 'azure_openai_transcribe_deployment', intenta la API nativa de Azure.
         - Si 404 (DeploymentNotFound) o 400 OperationNotSupported -> fallback local Whisper.
      2) Si no hay, usa directamente el fallback local Whisper.
    """
    tmp_path = _save_temp(file)
    transcript = ""
    modo = ""

    try:
        transcribe_depl = (settings.azure_openai_transcribe_deployment or "").strip() or None

        if transcribe_depl:
            try:
                transcript = _transcribe_via_azure(tmp_path, transcribe_depl)
                modo = "azure-transcriptions"
            except NotFoundError:
                # Deployment no existe -> fallback local
                transcript = _transcribe_via_local_whisper(tmp_path)
                modo = "local-whisper"
            except BadRequestError as e:
                # Modelo incompatible para audio (OperationNotSupported) -> fallback local
                if "OperationNotSupported" in str(e):
                    transcript = _transcribe_via_local_whisper(tmp_path)
                    modo = "local-whisper"
                else:
                    raise
            except OpenAIError:
                # Cualquier otra falla del servicio -> fallback local
                transcript = _transcribe_via_local_whisper(tmp_path)
                modo = "local-whisper"
        else:
            # No hay deployment de transcripción configurado -> fallback directo
            transcript = _transcribe_via_local_whisper(tmp_path)
            modo = "local-whisper"

    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

    structured = extract_structured_info(transcript)
    structured["__modo__"] = modo
    doc_id = str(uuid.uuid4())

    store.add_document(doc_id, transcript, structured)

    return TranscriptionResponse(id=doc_id, transcript=transcript, structured_data=structured)



