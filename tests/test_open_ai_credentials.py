import os
import pytest
from openai import AzureOpenAI


@pytest.mark.skipif(not os.getenv("AZURE_OPENAI_API_KEY"), reason="API key not configured")
def test_open_ai_credentials():
    """Simple live test to ensure OpenAI credentials work."""

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    )

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "Eres un asistente médico."},
            {"role": "user", "content": "¿Qué síntomas tiene la fiebre?"},
        ],
        temperature=0.7,
    )

    assert response.choices[0].message.content
