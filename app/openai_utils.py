"""Utility helpers to interact with Azure OpenAI."""

from openai import AzureOpenAI

from .config import settings


def get_openai_client() -> AzureOpenAI:
    """Create and return an Azure OpenAI client using environment settings."""

    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_api_endpoint,
    )
