"""Thread-based conversation memory management.

This module provides a lightweight JSON-based storage system for tracking
email conversation history by threadId, enabling context-aware responses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the thread memory storage file
THREAD_MEMORY_FILE = Path(__file__).resolve().parents[1] / "thread_memory.json"


def _ensure_memory_file_exists() -> None:
    """Ensure the thread memory JSON file exists."""
    if not THREAD_MEMORY_FILE.exists():
        THREAD_MEMORY_FILE.write_text(json.dumps({}, indent=2))


def load_thread_history(thread_id: str) -> list[dict[str, Any]]:
    """Load conversation history for a given threadId.

    Args:
        thread_id: The Gmail threadId to look up.

    Returns:
        A list of past message summaries/QA pairs, or empty list if not found.
    """
    try:
        _ensure_memory_file_exists()
        data = json.loads(THREAD_MEMORY_FILE.read_text())
        history = data.get(thread_id, [])
        return history if isinstance(history, list) else []
    except Exception as e:
        logger.warning(f"Failed to load thread history for {thread_id}: {e}")
        return []


def format_history_context(history: list[dict[str, Any]]) -> str:
    """Format conversation history into a clean summary string for prompt context.

    Args:
        history: List of past message dictionaries with 'summary', 'question', 'answer', etc.

    Returns:
        A formatted string suitable for inclusion in prompts, or empty string if no history.
    """
    if not history:
        return ""

    lines = ["Previous conversation context:"]
    for i, entry in enumerate(history, 1):
        if "question" in entry and "answer" in entry:
            lines.append(f"{i}. Q: {entry['question']}")
            lines.append(f"   A: {entry['answer']}")
        elif "summary" in entry:
            lines.append(f"{i}. {entry['summary']}")

    return "\n".join(lines) if len(lines) > 1 else ""


def append_to_thread_history(thread_id: str, entry: dict[str, Any]) -> None:
    """Append a new response summary to the thread history.

    Args:
        thread_id: The Gmail threadId to update.
        entry: A dictionary with response summary (e.g., {'summary': '...', 'question': '...', 'answer': '...'}).
    """
    try:
        _ensure_memory_file_exists()
        data = json.loads(THREAD_MEMORY_FILE.read_text())

        if thread_id not in data:
            data[thread_id] = []

        if not isinstance(data[thread_id], list):
            data[thread_id] = []

        data[thread_id].append(entry)

        # Keep only the last 20 messages per thread to avoid unbounded growth
        data[thread_id] = data[thread_id][-20:]

        THREAD_MEMORY_FILE.write_text(json.dumps(data, indent=2))
        logger.debug(f"Appended entry to thread history for {thread_id}")
    except Exception as e:
        logger.error(f"Failed to append to thread history for {thread_id}: {e}")


def clear_thread_history(thread_id: str) -> None:
    """Clear all history for a given threadId.

    Args:
        thread_id: The Gmail threadId to clear.
    """
    try:
        _ensure_memory_file_exists()
        data = json.loads(THREAD_MEMORY_FILE.read_text())

        if thread_id in data:
            del data[thread_id]
            THREAD_MEMORY_FILE.write_text(json.dumps(data, indent=2))
            logger.debug(f"Cleared history for thread {thread_id}")
    except Exception as e:
        logger.error(f"Failed to clear thread history for {thread_id}: {e}")
