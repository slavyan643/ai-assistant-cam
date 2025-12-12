import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def ask_ai(user_text: str) -> str:
    r = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Ти дружній асистент для камери. "
                    "Відповідай дуже коротко (1 речення). "
                    "Мова користувача = мова відповіді (укр/рос)."
                )
            },
            {"role": "user", "content": user_text},
        ],
        max_output_tokens=80,
    )
    return (r.output_text or "").strip()
