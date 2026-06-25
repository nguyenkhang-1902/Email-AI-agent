from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=dotenv_path)

local_llm_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
client: OpenAI = OpenAI(api_key="not-needed", base_url=local_llm_url)
logger = logging.getLogger(__name__)

# Model configuration - Load from .env with sensible defaults
MODEL_CLASSIFY = os.getenv("MODEL_CLASSIFY", "llama-3.2-1b-instruct")
MODEL_REASONING = os.getenv("MODEL_REASONING", "qwen2.5-coder-7b-instruct")
MAX_TOKENS = 512

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _load_text_file(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {path}. Using fallback prompt.")
        return fallback


def _load_json_file(path: Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if fallback is None:
        fallback = {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to load JSON profile from {path}: {exc}")
        return fallback


def _render_prompt(template: str, **kwargs: Any) -> str:
    """Render a prompt template by replacing only specified placeholder keys."""
    rendered = template
    for key, value in kwargs.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    return rendered


def _get_profile_description() -> str:
    if not USER_PROFILE:
        return ""

    profile_lines: list[str] = []
    profile_summary = USER_PROFILE.get("profile_summary", {})

    if isinstance(profile_summary, dict):
        if full_name := profile_summary.get("full_name"):
            profile_lines.append(f"Full name: {full_name}")
        if age := profile_summary.get("age"):
            profile_lines.append(f"Age: {age}")
        if background := profile_summary.get("background"):
            profile_lines.append(f"Background: {background}")
        if current_role := profile_summary.get("current_role"):
            profile_lines.append(f"Current role: {current_role}")
        if technical_skills := profile_summary.get("technical_skills"):
            profile_lines.append("Technical skills:")
            for skill in technical_skills:
                profile_lines.append(f"- {skill}")
        if notable_projects := profile_summary.get("notable_projects"):
            profile_lines.append("Notable projects:")
            for project in notable_projects:
                profile_lines.append(f"- {project}")
    elif isinstance(profile_summary, str) and profile_summary.strip():
        profile_lines.append(f"Profile summary: {profile_summary.strip()}")

    schedule_preferences = USER_PROFILE.get("schedule_preferences", {})
    if schedule_preferences:
        profile_lines.append("Schedule preferences:")
        for key, value in schedule_preferences.items():
            profile_lines.append(f"- {key}: {value}")

    rewrite_preferences = USER_PROFILE.get("rewrite_preferences", {}).get("rules", [])
    if rewrite_preferences:
        profile_lines.append("Rewrite preferences:")
        for rule in rewrite_preferences:
            profile_lines.append(f"- {rule}")

    signature = USER_PROFILE.get("signature", {})
    if isinstance(signature, dict):
        profile_lines.append("Signature preferences:")
        for key, value in signature.items():
            profile_lines.append(f"- {key}: {value}")

    return "\n".join(profile_lines)


def _extract_json_payload(text: str) -> str:
    """Normalize LLM output and extract the first JSON payload from markdown or plain text."""
    if not text:
        return ""

    normalized = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\s*```", normalized, re.IGNORECASE)
    if fenced_match:
        normalized = fenced_match.group(1).strip()

    if normalized.startswith("`") and normalized.endswith("`"):
        normalized = normalized[1:-1].strip()

    first_brace = normalized.find("{")
    if first_brace != -1:
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(normalized[first_brace:], start=first_brace):
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == '"' and not escaped:
                in_string = not in_string
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return normalized[first_brace:index + 1].strip()
            escaped = False

    return normalized


CLASSIFY_PROMPT = _load_text_file(
    PROMPT_DIR / "classify_email_prompt.txt",
    "You are an email classifier. Classify email content into Spam, Work, or Study. Return exactly one word."
)

REPLY_PROMPT = _load_text_file(
    PROMPT_DIR / "reply_email_prompt.txt",
    "You are a polished email response assistant. Transform bullet points into a complete, polite reply."
)

CALENDAR_PROMPT = _load_text_file(
    PROMPT_DIR / "calendar_prompt.txt",
    "You are a scheduling assistant. Extract meeting request details from email content."
)

def extract_scheduling_intent(subject: str, body: str, current_datetime: str) -> dict[str, Any]:
    """Extract scheduling intent from email text using the calendar prompt structure."""
    calendar_prompt = _render_prompt(CALENDAR_PROMPT, current_datetime=current_datetime)
    system_instructions = (
        f"{calendar_prompt}\n\n"
        "Pay close attention to relative date phrases like 'chiều mai', 'thứ hai tuần tới', or 'sáng ngày mai'. "
        "Convert them to exact ISO 8601 format with timezone +07:00. "
        "If no duration is specified, assume a default meeting length of 30 minutes. "
        "Respond ONLY with valid JSON matching the schema."
    )

    prompt = (
        f"Current Datetime: {current_datetime}\n"
        f"Email subject:\n{subject}\n\n"
        f"Email body:\n{body}\n"
        "\n\nRespond with raw JSON only."
    )

    response_text = _generate_content_with_model(prompt, system_instructions, model=MODEL_REASONING)
    cleaned_response = _extract_json_payload(response_text)

    if not cleaned_response:
        return {"has_intent": False, "title": "", "start": "", "end": ""}

    try:
        parsed = json.loads(cleaned_response)
        return {
            "has_intent": bool(parsed.get("has_intent", False)),
            "title": parsed.get("title", "") or "",
            "start": parsed.get("start", "") or "",
            "end": parsed.get("end", "") or "",
        }
    except json.JSONDecodeError:
        logger.warning("Failed to parse scheduling intent JSON response: %s", cleaned_response)
        return {"has_intent": False, "title": "", "start": "", "end": ""}


USER_PROFILE = _load_json_file(
    PROMPT_DIR / "user_profile.json",
    {
        "name": "Unknown",
        "role": "AI User",
        "industry": "General",
        "tone_preferences": {},
        "email_preferences": {},
        "personal_notes": [],
    },
)

logger.info(f"Classification model: {MODEL_CLASSIFY}")
logger.info(f"Reasoning model: {MODEL_REASONING}")


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


def _generate_content_with_model(prompt: str, system_instruction: str, model: str | None = None) -> str:
    """Generate content using the specified model or reasoning model by default.
    
    Args:
        prompt: The user prompt
        system_instruction: The system instruction
        model: Optional model name. If not provided, uses MODEL_REASONING
    
    Returns:
        Generated content as string
    """
    if model is None:
        model = MODEL_REASONING
    
    messages = _build_messages(prompt, system_instruction)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.25,
            top_p=0.95,
        )
    except Exception as exc:
        logger.error("LLM request failed: %s", exc)
        raise
    return _get_response_content(response).strip()


def classify_email(subject: str, body: str, history_context: str = "") -> str:
    """Classify an email into one of [Spam, Work, Study].

    The function instructs the model to return exactly one word from the
    set {Spam, Work, Study} and nothing else.

    Args:
        subject: Email subject line.
        body: Email body text.
        history_context: Optional prior conversation context to inform classification.
    """
    if not subject and not body:
        raise ValueError("Both subject and body are empty; cannot classify an empty email.")

    system_instructions = (
        f"{CLASSIFY_PROMPT}\n\n"
        "You are a strict classifier. Only output one of the following words: Spam, Work, Study. "
        "Return exactly that single word with no punctuation, no explanation, and no extra whitespace."
    )

    history_section = f"\n{history_context}\n" if history_context else ""

    prompt = (
        f"Email subject:\n{subject}\n\n"
        f"Email body:\n{body}\n"
        f"{history_section}"
        "Respond with a single word: Spam, Work, or Study."
    )

    try:
        text = _generate_content_with_model(prompt, system_instructions, model=MODEL_CLASSIFY)
        if not text:
            raise RuntimeError("No text returned from LM Studio model during classification.")

        # Extract the first token and normalize
        category = text.strip().split()[0]
        if category not in {"Spam", "Work", "Study"}:
            raise RuntimeError(f"Model returned invalid category: {category}")

        return category
    except Exception as exc:
        raise RuntimeError("Failed to classify email with LM Studio.") from exc


def _detect_language(subject: str, body: str) -> str:
    """Detect the language of the email.

    Returns a language identifier (e.g., 'English', 'Vietnamese', 'Spanish').
    """
    if not subject and not body:
        return "English"  # Default to English if both are empty

    system_instructions = (
        "You are a language detector. Identify the primary language of the given text. "
        "Return only the language name (e.g., English, Vietnamese, Spanish, French, German, etc.) "
        "with no punctuation or explanation."
    )

    prompt = (
        f"Subject:\n{subject}\n\n"
        f"Body:\n{body}\n\n"
        "Respond with only the language name:"
    )

    try:
        text = _generate_content_with_model(prompt, system_instructions, model=MODEL_REASONING)
        if not text:
            return "English"  # Default fallback
        return text.strip().split()[0]  # Extract first token
    except Exception:
        return "English"  # Default fallback on error


def generate_reply_points(subject: str, body: str, category: str, history_context: str = "") -> str:
    """Generate core factual bullet points needed for the reply (Reasoning phase).

    Returns an empty string when `category` is 'Spam'.

    Args:
        subject: Email subject line.
        body: Email body text.
        category: Email category (Work, Study, Spam, etc.).
        history_context: Optional prior conversation context to inform reasoning.
    """
    if category == "Spam":
        return ""

    if not subject and not body:
        raise ValueError("Both subject and body are empty; cannot generate reply points.")

    system_instructions = (
        "You are a reasoning assistant that extracts key points from emails. "
        "Generate ONLY core factual bullet points that need to be addressed in a reply. "
        "Use concise bullet format (one line per bullet). "
        "Do not write full sentences or complete thoughts. "
        "Focus on the essential information needed to craft a response. "
        "If prior conversation context is provided, ensure consistency with previous discussions."
    )

    history_section = f"\n{history_context}\n" if history_context else ""

    prompt = (
        f"Email category: {category}\n\n"
        f"Email subject:\n{subject}\n\n"
        f"Email body:\n{body}\n"
        f"{history_section}"
        "Extract the key points that need to be addressed:\n"
    )

    try:
        text = _generate_content_with_model(prompt, system_instructions, model=MODEL_REASONING)
        if text is None:
            raise RuntimeError("No text returned from LM Studio model when generating reply points.")

        return text.strip()

    except Exception as exc:
        raise RuntimeError("Failed to generate reply points with LM Studio.") from exc


def polish_email_style(outline_points: str, original_language: str, history_context: str = "") -> str:
    """Rewrite bullet points into a professional, polite email (Styling phase).

    Takes the output from generate_reply_points and transforms it into a complete,
    well-formatted email response in the original language.
    """
    if not outline_points:
        return ""

    system_instructions = _render_prompt(
        REPLY_PROMPT,
        user_profile=_get_profile_description(),
        history_context=history_context,
        subject="",
        body="",
        reply_outline=outline_points,
    )

    system_instructions += (
        f"\n\nYou are a professional email writer. Transform the provided bullet points into a "
        f"complete, professional, and polite email reply. Write in {original_language}. "
        f"Use proper email formatting with a greeting and closing. "
        f"Be concise but courteous. Do not include sender information or signatures."
    )

    prompt = (
        f"Transform these bullet points into a professional email reply in {original_language}:\n\n"
        f"{outline_points}\n\n"
        "Write the complete email reply below:\n"
    )

    try:
        text = _generate_content_with_model(prompt, system_instructions, model=MODEL_REASONING)
        if text is None:
            raise RuntimeError("No text returned from LM Studio model when polishing email style.")

        return text.strip()

    except Exception as exc:
        raise RuntimeError("Failed to polish email style with LM Studio.") from exc


def generate_reply(subject: str, body: str, category: str, history_context: str = "", reply_outline: str = "") -> str:
    """Generate a professional, polite reply in the same language as the input.

    This is the main orchestration function that chains together:
    1. generate_reply_points() - Extract key points (Reasoning phase)
    2. polish_email_style() - Transform into polished email (Styling phase)

    Returns an empty string when `category` is 'Spam'.

    Args:
        subject: Email subject line.
        body: Email body text.
        category: Email category (Work, Study, Spam, etc.).
        history_context: Optional prior conversation context.
        reply_outline: Optional bullet outline or calendar status notes for the final prompt.
    """
    if category == "Spam":
        return ""

    if not subject and not body:
        raise ValueError("Both subject and body are empty; cannot generate a reply.")

    try:
        if reply_outline:
            # If caller already provided a polished outline or calendar status note,
            # reuse it directly and avoid regenerating the reasoning phase.
            reply_points = reply_outline
        else:
            # Phase 1: Generate core bullet points (Reasoning) with historical context
            reply_points = generate_reply_points(subject, body, category, history_context)
            if not reply_points:
                raise RuntimeError("Failed to generate reply points.")

        # Detect the original language for styling phase
        original_language = _detect_language(subject, body)

        # Phase 2: Polish into professional email (Styling)
        reply_text = polish_email_style(reply_points, original_language, history_context)
        if not reply_text:
            raise RuntimeError("Failed to polish email style.")

        return reply_text

    except Exception as exc:
        raise RuntimeError("Failed to generate reply with LM Studio.") from exc
