from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    azure_openai_api_key: str | None = None
    azure_openai_api_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_transcribe_deployment: str | None = None
    azure_openai_embeddings_deployment: str | None = None

    # Compatibilidad hacia atrás (fallback si aún usas uno solo)
    azure_openai_deployment: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
