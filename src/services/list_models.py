"""
List the Gemini models your API key can use, to pick llm.model / llm.judge_model.

    python -m repopilot.services.list_models

Model names and free-tier availability change often, so this asks the API
instead of relying on a hard-coded list.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    # .env is in the project root: src/llm/list_models.py → parents[2]
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GOOGLE_API_KEY in the project's .env first (get one at https://aistudio.google.com/apikey).")

    from google import genai
    client = genai.Client(api_key=key)

    chat = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" in actions:
            chat.append(model.name.removeprefix("models/"))

    print("Gemini chat models available to your key (use in config.yaml → llm.model):\n")
    for name in sorted(chat):
        print(f"  {name}")
    print("\nFree tier: Flash and Flash-Lite models. Check limits at https://aistudio.google.com/ (rate limits are per project).")


if __name__ == "__main__":
    main()
