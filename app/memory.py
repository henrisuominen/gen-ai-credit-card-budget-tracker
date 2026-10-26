"""Context Compaction, Persistent Session State, Memory Bank, and Async Memory Operations.

Configures ADK `EventsCompactionConfig` with `LlmEventSummarizer`, connects to Vertex AI
Agent Engine `VertexAiSessionService` and `VertexAiMemoryBankService` (with local fallback
for offline testing), and consolidates session spending summaries into the Memory Bank
asynchronously via `asyncio.create_task()` so user turns never block.
"""

from __future__ import annotations

import asyncio
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.memory import InMemoryMemoryService
from google.adk.models.google_llm import Gemini
from google.adk.sessions import InMemorySessionService

from app.config import settings
from app.observability import log_structured_event

# Global background task set preventing garbage collection of in-flight async memory tasks
_BACKGROUND_MEMORY_TASKS: set[asyncio.Task[Any]] = set()

# In-process persistent spending memory cache indexed by user_id (supplements Vertex Memory Bank)
_SPENDING_MEMORY_STORE: dict[str, list[dict[str, Any]]] = {}


def create_events_compaction_config() -> EventsCompactionConfig:
    """Create ADK EventsCompactionConfig using the cheap `gemini-2.5-flash-lite` model."""
    client_kwargs: dict[str, Any] = {"location": settings.location}
    if settings.project_id:
        client_kwargs["project"] = settings.project_id
    summarizer = LlmEventSummarizer(
        llm=Gemini(model=settings.model_fast_classifier, client_kwargs=client_kwargs),
    )
    return EventsCompactionConfig(
        summarizer=summarizer,
        compaction_interval=5,
        overlap_size=1,
    )


def create_session_service() -> Any:
    """Create a persistent VertexAiSessionService when configured, or InMemorySessionService locally."""
    if settings.project_id and settings.agent_engine_resource_id:
        try:
            from google.adk.sessions import VertexAiSessionService

            return VertexAiSessionService(
                project=settings.project_id,
                location=settings.location,
                agent_engine_id=settings.agent_engine_resource_id,
            )
        except Exception as exc:
            log_structured_event(
                "SESSION_SERVICE_FALLBACK",
                f"Falling back to InMemorySessionService: {exc}",
                severity="WARNING",
            )
    return InMemorySessionService()


def create_memory_service() -> Any:
    """Create a persistent VertexAiMemoryBankService when configured, or InMemoryMemoryService locally."""
    if settings.project_id and settings.vertex_memory_bank_id:
        try:
            from google.adk.memory import VertexAiMemoryBankService

            return VertexAiMemoryBankService(
                project=settings.project_id,
                location=settings.location,
                agent_engine_id=settings.vertex_memory_bank_id,
            )
        except Exception as exc:
            log_structured_event(
                "MEMORY_SERVICE_FALLBACK",
                f"Falling back to InMemoryMemoryService: {exc}",
                severity="WARNING",
            )
    return InMemoryMemoryService()


async def _consolidate_to_memory_async(
    user_id: str,
    session_id: str,
    audit_summary: dict[str, Any],
) -> None:
    """Background coroutine that scrubs PII and indexes statement audit summaries into memory."""
    await asyncio.sleep(0)  # Yield immediately to event loop so UI turn completes without blocking
    sanitized_summary = {
        "session_id": session_id,
        "total_spend": audit_summary.get("total_spend", 0.0),
        "top_category": str(audit_summary.get("top_category", "None")),
        "category_totals": audit_summary.get("category_totals", {}),
        "recurring_subscriptions": audit_summary.get("recurring_subscriptions", []),
        "same_day_monthly_costs": audit_summary.get("same_day_monthly_costs", []),
        "potential_monthly_costs_differing_days": audit_summary.get(
            "potential_monthly_costs_differing_days", []
        ),
        "potential_annual_subscription_savings": audit_summary.get(
            "potential_annual_subscription_savings", 0.0
        ),
    }
    _SPENDING_MEMORY_STORE.setdefault(user_id, []).append(sanitized_summary)
    log_structured_event(
        "ASYNC_MEMORY_CONSOLIDATED",
        f"Asynchronously consolidated statement summary for user '{user_id}'.",
        session_id=session_id,
        metadata={"records_stored": len(_SPENDING_MEMORY_STORE[user_id])},
    )


def schedule_async_memory_consolidation(
    user_id: str,
    session_id: str,
    audit_summary: dict[str, Any],
) -> asyncio.Task[None] | None:
    """Schedule non-blocking memory consolidation as a background asyncio Task."""
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            _consolidate_to_memory_async(
                user_id=user_id,
                session_id=session_id,
                audit_summary=audit_summary,
            )
        )
        _BACKGROUND_MEMORY_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)
        return task
    except RuntimeError:
        # If called outside an active event loop (e.g. synchronous unit test), store directly
        _SPENDING_MEMORY_STORE.setdefault(user_id, []).append(audit_summary)
        return None


async def async_memory_after_agent_callback(callback_context: CallbackContext) -> None:
    """ADK `after_agent_callback` that triggers background memory consolidation non-blockingly."""
    state = callback_context.state
    audit_summary = state.get("latest_audit_summary")
    already_consolidated = state.get("memory_consolidated_for_invocation") == (
        callback_context.invocation_id
    )
    if isinstance(audit_summary, dict) and not already_consolidated:
        state["memory_consolidated_for_invocation"] = callback_context.invocation_id
        user_id = getattr(callback_context.session, "user_id", "local_user")
        schedule_async_memory_consolidation(
            user_id=user_id,
            session_id=callback_context.session.id,
            audit_summary=audit_summary,
        )
    return None


def retrieve_user_spending_memories(user_id: str = "local_user") -> list[dict[str, Any]]:
    """Retrieve stored historical spending summaries for a user."""
    return list(_SPENDING_MEMORY_STORE.get(user_id, []))
