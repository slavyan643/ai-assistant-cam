import os
from openai import OpenAI

# Беремо ключ з змінної середовища OPENAI_API_KEY
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def ask_ai(text: str) -> str:
    """
    Повертає коротку відповідь (1 речення), українською або російською
    залежно від контексту/побажання в запиті.
    """
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Ти дружній AI-асистент для камери. "
                    "Відповідай дуже коротко (1 речення). "
                    "Якщо користувач/запит українською — відповідай українською. "
                    "Якщо російською — російською. "
                    "Без зайвих пояснень."
                )
            },
            {"role": "user", "content": text},
        ],
        max_output_tokens=80,
    )
    return (resp.output_text or "").strip()
