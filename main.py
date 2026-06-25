import base64
from email.message import EmailMessage
from typing import Any

from src.gmail_service import get_gmail_service
from src.lm_agent import classify_email, generate_reply


def _get_header(headers: list[dict[str, str]], name: str) -> str | None:
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value")
    return None


def _get_plain_text(payload: dict[str, Any]) -> str:
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data + "=").decode("utf-8", errors="replace")

    for part in payload.get("parts", []) or []:
        text = _get_plain_text(part)
        if text:
            return text
    return ""


def _create_draft(service: Any, to_address: str, subject: str, body_text: str, thread_id: str) -> str:
    message = EmailMessage()
    message["To"] = to_address
    message["Subject"] = subject
    message["In-Reply-To"] = thread_id
    message["References"] = thread_id
    message.set_content(body_text)

    raw_bytes = message.as_bytes()
    raw_encoded = base64.urlsafe_b64encode(raw_bytes).decode("utf-8")

    draft_body = {
        "message": {
            "raw": raw_encoded,
            "threadId": thread_id,
        }
    }
    draft = service.users().drafts().create(userId="me", body=draft_body).execute()
    draft_id = draft.get("id")
    print(f"Draft created successfully: {draft_id}")
    return draft_id


def _process_message(service: Any, message_id: str) -> None:
    message = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    headers = message.get("payload", {}).get("headers", [])
    thread_id = message.get("threadId")
    subject = _get_header(headers, "Subject") or ""
    sender = _get_header(headers, "Reply-To") or _get_header(headers, "From") or ""
    body_text = _get_plain_text(message.get("payload", {}))

    if not sender or not thread_id:
        print(f"Skipping message {message_id}: missing sender or threadId.")
        return

    category = classify_email(subject, body_text)
    if category not in {"Work", "Study"}:
        print(f"Skipping message {message_id}: classified as {category}.")
        return

    reply_text = generate_reply(subject, body_text, category)
    if not reply_text:
        print(f"Skipping draft creation for message {message_id}: no reply generated.")
        return

    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    _create_draft(service, sender, reply_subject, reply_text, thread_id)


def process_unread_emails() -> None:
    service = get_gmail_service()
    print("Đang khởi tạo kết nối với Gmail API...")
    if not service:
        raise RuntimeError("Failed to initialize Gmail service.")

    response = service.users().messages().list(userId="me", q="is:unread", maxResults=20).execute()
    messages = response.get("messages", []) or []
    print(f"Found {len(messages)} unread messages.")

    for message in messages:
        message_id = message.get("id")
        if message_id:
            try:
                _process_message(service, message_id)
            except Exception as exc:
                print(f"Failed to process message {message_id}: {exc}")


if __name__ == "__main__":
    process_unread_emails()