"""Multi-Agent Orchestration, ADK App Compaction, and Vertex AI Agent Runtime (AdkApp) Entry Point.

Provides the Coordinator LlmAgent, SequentialAgent preprocessing pipeline, Synthesis Advisor
LlmAgent, `EventsCompactionConfig` on the ADK `App`, and `create_agent_engine_app()` wrapping
the agent hierarchy in `AdkApp` for Vertex AI Agent Runtime deployment.
"""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps.app import App
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.genai import types

from app.config import settings
from app.guardrails import EnterpriseGuardrailPlugin
from app.memory import (
    async_memory_after_agent_callback,
    create_events_compaction_config,
    create_memory_service,
    create_session_service,
)
from app.observability import ObservabilityPlugin, log_structured_event
from app.prompts import (
    BUDGET_COORDINATOR_INSTRUCTION,
    INGESTION_ANONYMIZER_INSTRUCTION,
    SPENDING_SYNTHESIS_ADVISOR_INSTRUCTION,
    SUBSCRIPTION_AUDIT_INSTRUCTION,
    TRANSACTION_CLASSIFIER_INSTRUCTION,
)
from app.tools import (
    classify_statement_transactions_batch,
    detect_recurring_monthly_subscriptions,
    extract_and_anonymize_statement_pdf,
    flag_subscription_confirmation_tool,
    query_historical_spending_memory_bank,
)

def _vertex_model(model_name: str) -> Gemini:
    """Create a Gemini model instance pinned to the configured regional Vertex AI endpoint."""
    client_kwargs: dict[str, Any] = {"location": settings.location}
    if settings.project_id:
        client_kwargs["project"] = settings.project_id
    return Gemini(model=model_name, client_kwargs=client_kwargs)


# 1. Sequential Preprocessing Sub-Agents (Fast Tier: gemini-2.5-flash-lite)
ingestion_anonymizer_agent = LlmAgent(
    name="ingestion_anonymizer_agent",
    model=_vertex_model(settings.model_fast_classifier),
    description=(
        "Extracts text from credit card statement PDFs via Document AI and scrubs all PII "
        "using Cloud Sensitive Data Protection (DLP)."
    ),
    instruction=INGESTION_ANONYMIZER_INSTRUCTION,
    tools=[extract_and_anonymize_statement_pdf],
)

transaction_classifier_agent = LlmAgent(
    name="transaction_classifier_agent",
    model=_vertex_model(settings.model_fast_classifier),
    description=(
        "Classifies anonymized transaction lines into normalized merchants, broad spending "
        "categories, and amounts using gemini-2.5-flash-lite."
    ),
    instruction=TRANSACTION_CLASSIFIER_INSTRUCTION,
    tools=[classify_statement_transactions_batch],
)

subscription_audit_agent = LlmAgent(
    name="subscription_audit_agent",
    model=_vertex_model(settings.model_fast_classifier),
    description=(
        "Executes deterministic rule-based detection of monthly recurring charges (same merchant "
        "and amount) and aggregates category spending totals."
    ),
    instruction=SUBSCRIPTION_AUDIT_INSTRUCTION,
    tools=[detect_recurring_monthly_subscriptions],
)

statement_preprocessing_pipeline = SequentialAgent(
    name="statement_preprocessing_pipeline",
    description=(
        "Sequential preprocessing pipeline that ingests and anonymizes a statement PDF, "
        "classifies all transactions with gemini-2.5-flash-lite, and runs rule-based monthly "
        "subscription detection."
    ),
    sub_agents=[
        ingestion_anonymizer_agent,
        transaction_classifier_agent,
        subscription_audit_agent,
    ],
)

# 2. Financial Synthesis & Action Advisor Sub-Agent (Reasoning Tier: gemini-2.5-pro)
spending_synthesis_advisor_agent = LlmAgent(
    name="spending_synthesis_advisor_agent",
    model=_vertex_model(settings.model_synthesis_pro),
    description=(
        "Senior financial advisor agent running on gemini-2.5-pro that synthesizes category "
        "spending, compares historical memory, recommends concrete savings actions, and triggers "
        "Human-in-the-Loop confirmation before flagging subscriptions for cancellation."
    ),
    instruction=SPENDING_SYNTHESIS_ADVISOR_INSTRUCTION,
    tools=[
        query_historical_spending_memory_bank,
        flag_subscription_confirmation_tool,
    ],
    after_agent_callback=async_memory_after_agent_callback,
)

# 3. Root Coordinator Agent (Coordinator Tier: gemini-2.5-flash)
budget_coordinator_agent = LlmAgent(
    name="budget_coordinator_agent",
    model=_vertex_model(settings.model_coordinator),
    description=(
        "Coordinator agent for the Credit Card Budget Tracker on Gemini Enterprise Agent Platform. "
        "Coordinates statement preprocessing and conversational spend-reduction synthesis."
    ),
    instruction=BUDGET_COORDINATOR_INSTRUCTION,
    tools=[
        extract_and_anonymize_statement_pdf,
        classify_statement_transactions_batch,
        detect_recurring_monthly_subscriptions,
        query_historical_spending_memory_bank,
        flag_subscription_confirmation_tool,
    ],
    sub_agents=[
        statement_preprocessing_pipeline,
        spending_synthesis_advisor_agent,
    ],
    after_agent_callback=async_memory_after_agent_callback,
)

# Standard ADK export
root_agent = budget_coordinator_agent

# 4. ADK App with Context Compaction, Pre-Ingestion PII Gatekeeper, and Observability Plugins
app = App(
    name="app",
    root_agent=root_agent,
    events_compaction_config=create_events_compaction_config(),
    plugins=[
        EnterpriseGuardrailPlugin(),
        ObservabilityPlugin(),
    ],
)


def create_agent_engine_app() -> Any:
    """Wrap the ADK root_agent in Vertex AI Agent Engine `AdkApp` for Gemini Enterprise."""
    from vertexai.preview.reasoning_engines import AdkApp  # type: ignore[import-untyped]

    return AdkApp(
        agent=root_agent,
        enable_tracing=settings.enable_cloud_trace,
    )


async def run_chat() -> None:
    """Run a local-only interactive CLI session with Pre-Ingestion PII Gatekeeper protection."""
    runner = Runner(
        app=app,
        session_service=create_session_service(),
        memory_service=create_memory_service(),
    )
    session = await runner.session_service.create_session(
        app_name="app",
        user_id="local_user",
    )

    log_structured_event(
        "LOCAL_CLI_STARTED",
        "Started local-only Credit Card Budget Tracker CLI session.",
        session_id=session.id,
    )
    print("Credit Card Budget Tracker Agent (Local Only - type 'exit' to quit)")
    print("Upload a statement with '/file /path/to/statement.pdf' or ask a question.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break

        parts: list[types.Part] = []
        if user_input.startswith("/file "):
            file_path = Path(user_input.removeprefix("/file ").strip()).expanduser()
            if not file_path.is_file():
                print(f"File not found: {file_path}")
                continue
            mime_type, _ = mimetypes.guess_type(file_path)
            # Attach bytes; EnterpriseGuardrailPlugin intercepts & scrubs PII in memory before LLM
            parts.append(
                types.Part.from_bytes(
                    data=file_path.read_bytes(),
                    mime_type=mime_type or "application/pdf",
                )
            )
            parts.append(
                types.Part.from_text(
                    text=(
                        "Please classify all transactions in this statement, identify monthly "
                        "recurring subscriptions, and suggest ways to reduce my spend."
                    )
                )
            )
        else:
            parts.append(types.Part.from_text(text=user_input))

        content = types.Content(role="user", parts=parts)
        async for event in runner.run_async(
            user_id="local_user",
            session_id=session.id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"\nAgent: {part.text}\n")


def main() -> None:
    """CLI entry point for `budget-tracker` or `python3 -m app.agents`."""
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
