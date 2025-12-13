import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def ask_ai(text: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Ти AI-асистент. "
                    "Відповідай коротко (1 речення). "
                    "Використовуй мову користувача (українська або російська)."
                )
            },
            {"role": "user", "content": text},
        ],
        max_output_tokens=80,
    )
    return (response.output_text or "").strip()