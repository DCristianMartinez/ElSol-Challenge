import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT")
)

response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    messages=[
        {"role": "system", "content": "Eres un asistente médico."},
        {"role": "user", "content": "¿Qué síntomas tiene la fiebre?"}
    ],
    temperature=0.7
)

print("✅ Respuesta:")
print(response.choices[0].message.content)
