import os
import sys

from dotenv import load_dotenv
from groq import Groq


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY" )
    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    if not api_key:
        print(
            "[ERREUR] GROQ_API_KEY est absente du fichier .env."
        )
        sys.exit(1)

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Réponds uniquement par le texte : "
                        "CONNEXION_OK"
                    ),
                }
            ],
            temperature=0,
            max_tokens=20,
        )

        answer = (
            response.choices[0]
            .message.content
            .strip()
        )

        print("[OK] Connexion à Groq réussie.")
        print(f"[OK] Modèle utilisé {model}")
        print(f"[RÉPONSE] {answer}")

    except Exception as error:
        print("[ERREUR] Connexion Groq impossible.")
        print(f"[DÉTAIL] {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()