from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=dotenv_path)

local_llm_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
client: OpenAI = OpenAI(api_key="not-needed", base_url=local_llm_url)
logger = logging.getLogger(__name__)

PRIMARY_MODEL = "qwen2.5-coder-7b-instruct"
MAX_TOKENS = 512


def _build_messages(prompt: str, system_instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
    ]


def _get_response_content(response: Any) -> str:
    if not response:
        return ""

    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content:
                return content
        content = getattr(first_choice, "content", None)
        if content:
            return content

    if hasattr(response, "message"):
        message = getattr(response, "message")
        return getattr(message, "content", "") or ""

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            first_choice = choices[0]
            message = first_choice.get("message", {})
            if message:
                return message.get("content", "") or first_choice.get("content", "")

    return ""


def _generate_content_with_model(prompt: str, system_instruction: str) -> str:
    messages = _build_messages(prompt, system_instruction)
    response = client.chat.completions.create(
        model=PRIMARY_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.25,
        top_p=0.95,
    )
    return _get_response_content(response).strip()


def classify_email(subject: str, body: str) -> str:
    """Classify an email into one of [Spam, Work, Study].

    The function instructs the model to return exactly one word from the
    set {Spam, Work, Study} and nothing else.
    """
    if not subject and not body:
        raise ValueError("Both subject and body are empty; cannot classify an empty email.")

    system_instructions = (
        "You are a strict classifier. Only output one of the following words: "
        "Spam, Work, Study. Return exactly that single word with no punctuation, no explanation, "
        "and no extra whitespace."
    )

    prompt = (
        f"Email subject:\n{subject}\n\n"
        f"Email body:\n{body}\n\n"
        "Respond with a single word: Spam, Work, or Study."
    )

    try:
        text = _generate_content_with_model(prompt, system_instructions)
        if not text:
            raise RuntimeError("No text returned from LM Studio model during classification.")

        # Extract the first token and normalize
        category = text.strip().split()[0]
        if category not in {"Spam", "Work", "Study"}:
            raise RuntimeError(f"Model returned invalid category: {category}")

        return category
    except Exception as exc:
        raise RuntimeError("Failed to classify email with LM Studio.") from exc


def generate_reply(subject: str, body: str, category: str) -> str:
    """Generate a professional, polite reply in the same language as the input.

    Returns an empty string when `category` is 'Spam'.
    """
    if category == "Spam":
        return ""

    if not subject and not body:
        raise ValueError("Both subject and body are empty; cannot generate a reply.")

    system_instructions = (
        "You are an assistant that writes professional and polite email replies. "
        "Write a concise, courteous response appropriate to the email content. "
        "Reply in the same language as the incoming email. Do not include signatures beyond a short closing."
    )

    prompt = (
        f"Email category: {category}\n\n"
        f"Email subject:\n{subject}\n\n"
        f"Email body:\n{body}\n\n"
        "Write the full reply below:\n"
    )

    try:
        text = _generate_content_with_model(prompt, system_instructions)
        if text is None:
            raise RuntimeError("No text returned from LM Studio model when generating reply.")

        return text.strip()

    except Exception as exc:
        raise RuntimeError("Failed to generate reply with LM Studio.") from exc
