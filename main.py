import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Optional

from src.gmail_service import get_gmail_service, check_calendar_busy, create_calendar_event
from src.lm_agent import (
    classify_email,
    extract_scheduling_intent,
    generate_reply,
    generate_reply_points,
)
from src.thread_memory import load_thread_history, format_history_context, append_to_thread_history

logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION - Inspired by LangGraph StateGraph
# ============================================================================
@dataclass
class AgentState:
    """Global state for the email AI agent pipeline.
    
    This dataclass represents the mutable state that flows through the
    state graph, carrying data between nodes.
    """
    current_email: dict[str, Any] = field(default_factory=dict)
    category: str = ""
    scheduling_intent: dict[str, Any] = field(default_factory=dict)
    calendar_note: str = ""
    reply_outline: str = ""
    final_reply: str = ""
    draft_id: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for logging."""
        return {
            "current_email": {
                "message_id": self.current_email.get("message_id", ""),
                "subject": self.current_email.get("subject", "")[:50],
                "thread_id": self.current_email.get("thread_id", ""),
            },
            "category": self.category,
            "calendar_note_length": len(self.calendar_note),
            "reply_outline_length": len(self.reply_outline),
            "final_reply_length": len(self.final_reply),
            "draft_id": self.draft_id,
            "error": self.error,
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _get_header(headers: list[dict[str, str]], name: str) -> str | None:
    """Extract header value from email headers list."""
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value")
    return None


def _get_plain_text(payload: dict[str, Any]) -> str:
    """Recursively extract plain text from email payload."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data + "=").decode("utf-8", errors="replace")

    for part in payload.get("parts", []) or []:
        text = _get_plain_text(part)
        if text:
            return text
    return ""




# ============================================================================
# STATE GRAPH NODES
# ============================================================================
def node_ingest_and_triage(state: AgentState, service: Any, message_id: str) -> AgentState:
    """Node 1: Ingest and Triage
    
    Fetches the email from Gmail API and extracts essential information.
    Determines if the email has required fields (sender, threadId).
    
    Returns:
        Updated AgentState with current_email populated or error set.
    """
    try:
        logger.info(f"[NODE: INGEST_AND_TRIAGE] Processing message {message_id}")
        
        message = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        headers = message.get("payload", {}).get("headers", [])
        thread_id = message.get("threadId")
        subject = _get_header(headers, "Subject") or ""
        sender = _get_header(headers, "Reply-To") or _get_header(headers, "From") or ""
        body_text = _get_plain_text(message.get("payload", {}))
        
        if not sender or not thread_id:
            state.error = f"Missing sender ({sender}) or threadId ({thread_id})"
            logger.warning(f"[NODE: INGEST_AND_TRIAGE] {state.error}")
            return state
        
        # Load thread history for context
        history = load_thread_history(thread_id)
        history_context = format_history_context(history)
        
        state.current_email = {
            "message_id": message_id,
            "subject": subject,
            "body": body_text,
            "sender": sender,
            "thread_id": thread_id,
            "history_context": history_context,
        }
        
        logger.info(f"[NODE: INGEST_AND_TRIAGE] Email ingested successfully. Subject: {subject[:50]}")
        return state
        
    except Exception as exc:
        state.error = f"Ingest failed: {str(exc)}"
        logger.error(f"[NODE: INGEST_AND_TRIAGE] {state.error}")
        return state


def node_classify(state: AgentState) -> AgentState:
    """Node 2: Classify
    
    Calls the LM agent to classify the email into [Spam, Work, Study].
    Uses historical context from thread memory for better accuracy.
    
    Returns:
        Updated AgentState with category set or error populated.
    """
    try:
        if state.error:
            logger.info(f"[NODE: CLASSIFY] Skipping due to prior error: {state.error}")
            return state
        
        logger.info(f"[NODE: CLASSIFY] Classifying email: {state.current_email.get('subject', '')[:50]}")
        
        subject = state.current_email.get("subject", "")
        body = state.current_email.get("body", "")
        history_context = state.current_email.get("history_context", "")
        
        category = classify_email(subject, body, history_context)
        state.category = category
        
        logger.info(f"[NODE: CLASSIFY] Email classified as: {category}")
        return state
        
    except Exception as exc:
        state.error = f"Classification failed: {str(exc)}"
        logger.error(f"[NODE: CLASSIFY] {state.error}")
        return state


def node_handle_calendar(state: AgentState) -> AgentState:
    """Node 2.5: Handle Calendar Scheduling
    
    Checks if the email contains scheduling intent (meeting request with dates/times).
    If scheduling intent detected:
    - Checks calendar availability
    - Books event if free, or notes conflict if busy
    - Updates reply_outline with calendar status
    
    Skipped if category is 'Spam' or on prior errors.
    
    Returns:
        Updated AgentState with scheduling_intent and reply_outline modified.
    """
    try:
        if state.error:
            logger.info(f"[NODE: HANDLE_CALENDAR] Skipping due to prior error: {state.error}")
            return state
        
        if state.category not in {"Work", "Study"}:
            logger.info(f"[NODE: HANDLE_CALENDAR] Skipping non-actionable category: {state.category}")
            return state
        
        logger.info(f"[NODE: HANDLE_CALENDAR] Analyzing scheduling intent")
        
        subject = state.current_email.get("subject", "")
        body = state.current_email.get("body", "")
        current_datetime = datetime.now(timezone.utc).astimezone().isoformat()
        
        scheduling_data = extract_scheduling_intent(subject, body, current_datetime)
        state.scheduling_intent = scheduling_data
        logger.debug(f"[NODE: HANDLE_CALENDAR] Extracted scheduling data: {scheduling_data}")
        
        if scheduling_data.get("has_intent", False):
            title = scheduling_data.get("title", "Meeting")
            start_iso = scheduling_data.get("start", "")
            end_iso = scheduling_data.get("end", "")
            
            if not start_iso or not end_iso:
                logger.warning(f"[NODE: HANDLE_CALENDAR] Incomplete scheduling data: {scheduling_data}")
                return state
            
            logger.info(f"[NODE: HANDLE_CALENDAR] Scheduling intent detected: {title} ({start_iso} - {end_iso})")
            
            try:
                is_busy = check_calendar_busy(start_iso, end_iso)

                if is_busy:
                    logger.info(f"[NODE: HANDLE_CALENDAR] Calendar conflict detected")
                    state.calendar_note = "\n\n[Calendar Status] Conflict detected on your Calendar. Need to propose a different time."
                else:
                    logger.info(f"[NODE: HANDLE_CALENDAR] Calendar slot is free, booking event")
                    event = create_calendar_event(
                        summary=title,
                        start_iso=start_iso,
                        end_iso=end_iso,
                        description=f"Meeting scheduled from email: {subject}"
                    )
                    logger.info(f"[NODE: HANDLE_CALENDAR] Event created: {event.get('id')}")
                    state.calendar_note = f"\n\n[Calendar Status] Appointment successfully booked on your Calendar: {event.get('htmlLink', 'Calendar')}"
            except Exception as exc:
                logger.error(f"[NODE: HANDLE_CALENDAR] Calendar operation failed: {str(exc)}")
                state.calendar_note = "\n\n[Calendar Status] Unable to access calendar at this time."

        return state
        
    except Exception as exc:
        state.error = f"Calendar handling failed: {str(exc)}"
        logger.error(f"[NODE: HANDLE_CALENDAR] {state.error}")
        return state


def node_generate_content(state: AgentState) -> AgentState:
    """Node 3: Generate Content
    
    Runs the two-step reasoning and styling pipeline:
    1. generate_reply_points() - Extract bullet points (Reasoning)
    2. generate_reply() internally calls polish_email_style() (Styling)
    
    Skipped if category is 'Spam' or on prior errors.
    
    Returns:
        Updated AgentState with reply_outline and final_reply populated.
    """
    try:
        if state.error:
            logger.info(f"[NODE: GENERATE_CONTENT] Skipping due to prior error: {state.error}")
            return state
        
        if state.category not in {"Work", "Study"}:
            logger.info(f"[NODE: GENERATE_CONTENT] Skipping non-actionable category: {state.category}")
            return state
        
        logger.info(f"[NODE: GENERATE_CONTENT] Generating reply for category: {state.category}")
        
        subject = state.current_email.get("subject", "")
        body = state.current_email.get("body", "")
        history_context = state.current_email.get("history_context", "")
        
        # Phase 1: Generate bullet points (Reasoning)
        reply_outline = generate_reply_points(subject, body, state.category, history_context)
        if state.calendar_note:
            reply_outline = (reply_outline or "") + state.calendar_note
        state.reply_outline = reply_outline
        logger.info(f"[NODE: GENERATE_CONTENT] Reply outline generated ({len(reply_outline)} chars)")
        
        # Phase 2: Polish into full email (Reasoning + Styling via generate_reply)
        final_reply = generate_reply(subject, body, state.category, history_context, reply_outline)
        state.final_reply = final_reply
        logger.info(f"[NODE: GENERATE_CONTENT] Final reply generated ({len(final_reply)} chars)")
        
        return state
        
    except Exception as exc:
        state.error = f"Content generation failed: {str(exc)}"
        logger.error(f"[NODE: GENERATE_CONTENT] {state.error}")
        return state


def node_create_gmail_draft(state: AgentState, service: Any) -> AgentState:
    """Node 4: Create Gmail Draft
    
    Handles Gmail API interaction to save the generated reply as a draft.
    Appends the response to thread memory after successful creation.
    
    Returns:
        Updated AgentState with draft_id set or error populated.
    """
    try:
        if state.error:
            logger.info(f"[NODE: CREATE_DRAFT] Skipping due to prior error: {state.error}")
            return state
        
        if not state.final_reply:
            logger.info(f"[NODE: CREATE_DRAFT] No reply generated, skipping draft creation")
            return state
        
        logger.info(f"[NODE: CREATE_DRAFT] Creating Gmail draft")
        
        # Prepare draft metadata
        subject = state.current_email.get("subject", "")
        sender = state.current_email.get("sender", "")
        thread_id = state.current_email.get("thread_id", "")
        reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        
        # Create email message
        message = EmailMessage()
        message["To"] = sender
        message["Subject"] = reply_subject
        message["In-Reply-To"] = thread_id
        message["References"] = thread_id
        message.set_content(state.final_reply)
        
        # Encode and send to Gmail API
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
        state.draft_id = draft_id
        
        logger.info(f"[NODE: CREATE_DRAFT] Draft created successfully: {draft_id}")
        
        # Append to thread memory for future reference
        entry = {
            "question": f"{subject}: {state.current_email.get('body', '')[:100]}...",
            "answer": state.final_reply[:200] + "..." if len(state.final_reply) > 200 else state.final_reply,
            "category": state.category,
            "draft_id": draft_id,
        }
        append_to_thread_history(thread_id, entry)
        logger.info(f"[NODE: CREATE_DRAFT] Response appended to thread memory")
        
        return state
        
    except Exception as exc:
        state.error = f"Draft creation failed: {str(exc)}"
        logger.error(f"[NODE: CREATE_DRAFT] {state.error}")
        return state


# ============================================================================
# ROUTER FUNCTION - Conditional Routing Logic
# ============================================================================
def router_classify_to_next(state: AgentState) -> str:
    """Router function for conditional routing after classification.
    
    Determines the next node based on the email category and prior errors.
    
    Returns:
        Route name: "generate_content", "end", or "error"
    """
    if state.error:
        return "error"
    
    if state.category in {"Work", "Study"}:
        return "generate_content"
    else:
        # Spam emails: skip to end
        return "end"


def router_content_to_next(state: AgentState) -> str:
    """Router function for conditional routing after content generation.
    
    Determines the next node based on whether content was generated.
    
    Returns:
        Route name: "create_draft" or "end"
    """
    if state.error:
        return "error"
    
    if state.final_reply:
        return "create_draft"
    else:
        return "end"


# ============================================================================
# ORCHESTRATOR - Central State Graph Execution
# ============================================================================
def run_agent_graph(service: Any, message_id: str) -> AgentState:
    """Central orchestrator function that manages state transitions.
    
    Implements a state graph with the following flow:
    
    1. node_ingest_and_triage
            ↓
    2. node_classify
            ↓
    2.5. node_handle_calendar (NEW)
            ↓ [router_classify_to_next]
        ├─ generate_content (Work/Study) → create_draft
        ├─ end (Spam)
        └─ error
    
    3. node_generate_content
            ↓
    4. node_create_gmail_draft
            ↓
    5. end
    
    Args:
        service: Gmail service instance from get_gmail_service()
        message_id: The Gmail message ID to process
    
    Returns:
        Final AgentState with all pipeline results
    """
    logger.info(f"[ORCHESTRATOR] Starting agent graph for message {message_id}")
    
    # Initialize state
    state = AgentState()
    
    # ========== NODE 1: INGEST AND TRIAGE ==========
    logger.info("=== STATE NODE 1: INGEST_AND_TRIAGE ===")
    state = node_ingest_and_triage(state, service, message_id)
    logger.debug(f"State after node_ingest_and_triage: {state.to_dict()}")
    
    # ========== NODE 2: CLASSIFY ==========
    logger.info("=== STATE NODE 2: CLASSIFY ===")
    state = node_classify(state)
    logger.debug(f"State after node_classify: {state.to_dict()}")
    
    # ========== NODE 2.5: HANDLE CALENDAR (NEW) ==========
    logger.info("=== STATE NODE 2.5: HANDLE_CALENDAR ===")
    state = node_handle_calendar(state)
    logger.debug(f"State after node_handle_calendar: {state.to_dict()}")
    
    # ========== ROUTER: CLASSIFY_TO_NEXT ==========
    next_route = router_classify_to_next(state)
    logger.info(f"Router decision: {next_route}")
    
    if next_route == "error":
        logger.error(f"[ORCHESTRATOR] Halting due to error: {state.error}")
        return state
    
    if next_route == "end":
        logger.info(f"[ORCHESTRATOR] Email category '{state.category}' requires no action. Ending graph.")
        return state
    
    # ========== NODE 3: GENERATE CONTENT ==========
    if next_route == "generate_content":
        logger.info("=== STATE NODE 3: GENERATE_CONTENT ===")
        state = node_generate_content(state)
        logger.debug(f"State after node_generate_content: {state.to_dict()}")
        
        # ========== ROUTER: CONTENT_TO_NEXT ==========
        next_route = router_content_to_next(state)
        logger.info(f"Router decision: {next_route}")
        
        if next_route == "error":
            logger.error(f"[ORCHESTRATOR] Halting due to error: {state.error}")
            return state
        
        if next_route == "end":
            logger.info(f"[ORCHESTRATOR] No reply generated. Ending graph.")
            return state
        
        # ========== NODE 4: CREATE GMAIL DRAFT ==========
        if next_route == "create_draft":
            logger.info("=== STATE NODE 4: CREATE_GMAIL_DRAFT ===")
            state = node_create_gmail_draft(state, service)
            logger.debug(f"State after node_create_gmail_draft: {state.to_dict()}")
    
    logger.info(f"[ORCHESTRATOR] Graph execution completed successfully. Draft ID: {state.draft_id}")
    return state


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def process_unread_emails() -> None:
    """Main function that fetches unread emails and processes them through the agent graph."""
    service = get_gmail_service()
    print("Đang khởi tạo kết nối với Gmail API...")
    if not service:
        raise RuntimeError("Failed to initialize Gmail service.")

    response = service.users().messages().list(userId="me", q="is:unread", maxResults=5).execute()
    messages = response.get("messages", []) or []
    print(f"Found {len(messages)} unread messages.")

    for message in messages:
        message_id = message.get("id")
        if message_id:
            try:
                # Execute the state graph for this message
                final_state = run_agent_graph(service, message_id)
                
                # Log the outcome
                if final_state.error:
                    print(f"❌ Message {message_id} failed: {final_state.error}")
                elif final_state.draft_id:
                    print(f"✅ Message {message_id} processed successfully. Draft: {final_state.draft_id}")
                else:
                    print(f"⏭️  Message {message_id} skipped (category: {final_state.category})")
                    
            except Exception as exc:
                print(f"❌ Unexpected error processing message {message_id}: {exc}")
                logger.exception(f"Uncaught exception in run_agent_graph for {message_id}")


if __name__ == "__main__":
    process_unread_emails()