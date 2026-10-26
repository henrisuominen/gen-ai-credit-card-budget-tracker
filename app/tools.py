"""ADK Function Tools for Credit Card Statement Preprocessing, Classification, and Synthesis.

Provides Google-style documented tools with strict Pydantic input/output validation,
structured `GuidedToolResult` error recovery instructions, and Human-in-the-Loop confirmation
via `FunctionTool(..., require_confirmation=True)` and `tool_context.request_confirmation(...)`.
"""

from pathlib import Path
from typing import Any, Optional

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.memory import retrieve_user_spending_memories, schedule_async_memory_consolidation
from app.preprocessing import (
    classify_transactions_with_flash_lite,
    detect_monthly_recurring_transactions,
    extract_and_anonymize_bytes,
)
from app.schemas import ClassifiedTransaction, GuidedToolResult


class ExtractStatementInputSchema(BaseModel):
    """Strict input schema for `extract_and_anonymize_statement_pdf`."""

    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(
        default="",
        description="Local path to the statement PDF or text file, or empty if uploaded in UI.",
    )


class ClassifyTransactionsInputSchema(BaseModel):
    """Strict input schema for `classify_statement_transactions_batch`."""

    model_config = ConfigDict(extra="forbid")
    anonymized_statement_text: str = Field(
        default="",
        description="PII-redacted statement text to classify.",
    )


class FlagSubscriptionInputSchema(BaseModel):
    """Strict input schema for `flag_subscription_for_cancellation_with_approval`."""

    model_config = ConfigDict(extra="forbid")
    merchant: str = Field(..., min_length=1, description="Canonical merchant name.")
    monthly_amount: float = Field(..., ge=0.0, description="Non-negative monthly amount.")
    reason: str = Field(
        default="User requested spend reduction action",
        description="Reason for flagging subscription.",
    )


def extract_and_anonymize_statement_pdf(
    file_path: str = "",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Extract text from a credit card statement PDF using Document AI and scrub PII via Cloud SDP.

    Use this tool when the user provides a file path to a PDF or text statement, OR when
    retrieving statement text that was intercepted and anonymized by the Pre-Ingestion PII
    Gatekeeper upon upload.

    Args:
        file_path: Optional local file system path to the statement PDF or text file.
            Leave empty if the user attached the file directly in the chat UI (the gatekeeper
            automatically stores the anonymized text in session state).
        tool_context: ADK ToolContext providing access to session state.

    Returns:
        A dictionary matching `GuidedToolResult` containing:
        - `status`: 'success' or 'error'
        - `data.anonymized_text`: The PII-redacted statement text ready for classification.
        - `data.character_count`: Length of the anonymized text.
        - `recovery_instruction`: Guidance if the file is missing or unreadable.
    """
    try:
        validated_in = ExtractStatementInputSchema(file_path=file_path)
        clean_path = validated_in.file_path.strip()

        if not clean_path and tool_context is not None:
            cached_text = tool_context.state.get("latest_anonymized_statement_text", "")
            if cached_text:
                return GuidedToolResult(
                    status="success",
                    data={
                        "anonymized_text": cached_text,
                        "character_count": len(cached_text),
                        "source": "pre_ingestion_pii_gatekeeper",
                    },
                    message="Retrieved PII-cleaned statement text from Pre-Ingestion Gatekeeper.",
                ).model_dump()

        if not clean_path:
            return GuidedToolResult(
                status="error",
                error_code="MISSING_STATEMENT_INPUT",
                message="No statement file path or uploaded statement attachment was found.",
                recovery_instruction=(
                    "Ask the user to attach their credit card statement PDF using the paperclip "
                    "button in the chat UI, or provide a valid local file path."
                ),
            ).model_dump()

        resolved = Path(clean_path).expanduser()
        if not resolved.is_file():
            return GuidedToolResult(
                status="error",
                error_code="FILE_NOT_FOUND",
                message="Statement file does not exist at the specified path.",
                recovery_instruction=(
                    "Verify the file path with the user or ask them to upload the PDF directly "
                    "through the chat attachment button."
                ),
            ).model_dump()

        mime_type = (
            "application/pdf" if resolved.suffix.lower() == ".pdf" else "text/plain"
        )
        anonymized_text = extract_and_anonymize_bytes(
            file_bytes=resolved.read_bytes(),
            mime_type=mime_type,
        )
        if not anonymized_text.strip():
            return GuidedToolResult(
                status="error",
                error_code="EMPTY_DOCUMENT_TEXT",
                message="Document AI extracted zero readable text from the provided file.",
                recovery_instruction=(
                    "Ask the user to check that the PDF is not password-protected or corrupted, "
                    "and upload a readable statement."
                ),
            ).model_dump()

        if tool_context is not None:
            tool_context.state["latest_anonymized_statement_text"] = anonymized_text

        return GuidedToolResult(
            status="success",
            data={
                "anonymized_text": anonymized_text,
                "character_count": len(anonymized_text),
                "source": "document_ai_and_cloud_sdp",
            },
            message="Successfully extracted and anonymized statement text.",
        ).model_dump()
    except Exception as exc:
        return GuidedToolResult(
            status="error",
            error_code="EXTRACTION_FAILED",
            message=f"Unexpected error during PDF extraction and anonymization: {exc}",
            recovery_instruction=(
                "Ask the user to provide the statement transactions as plain text or upload "
                "a standard PDF credit card statement."
            ),
        ).model_dump()


def classify_statement_transactions_batch(
    anonymized_statement_text: str = "",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Classify anonymized statement lines into merchants, broad categories, and amounts.

    Uses the low-cost `gemini-2.5-flash-lite` model with strict Pydantic structured outputs
    to normalize merchant descriptions (e.g. 'UBER LLC #12NYC' -> 'UBER') and categorize
    each purchase.

    Args:
        anonymized_statement_text: The PII-redacted text of the credit card statement.
            If omitted, reads `latest_anonymized_statement_text` from session state.
        tool_context: ADK ToolContext used to read/write structured transaction lists in state.

    Returns:
        A dictionary matching `GuidedToolResult` containing:
        - `status`: 'success' or 'error'
        - `data.transactions`: List of validated `ClassifiedTransaction` dicts.
        - `data.transaction_count`: Number of classified transactions.
        - `recovery_instruction`: Instructions if no valid transactions could be parsed.
    """
    try:
        validated_in = ClassifyTransactionsInputSchema(
            anonymized_statement_text=anonymized_statement_text
        )
        text_to_classify = validated_in.anonymized_statement_text.strip()
        if not text_to_classify and tool_context is not None:
            cached_classified = tool_context.state.get("latest_classified_transactions")
            if isinstance(cached_classified, list) and cached_classified:
                return GuidedToolResult(
                    status="success",
                    data={
                        "transactions": cached_classified,
                        "transaction_count": len(cached_classified),
                    },
                    message=f"Retrieved {len(cached_classified)} background-classified transactions.",
                ).model_dump()
            text_to_classify = str(
                tool_context.state.get("latest_anonymized_statement_text", "")
            ).strip()

        if not text_to_classify:
            return GuidedToolResult(
                status="error",
                error_code="NO_ANONYMIZED_TEXT_AVAILABLE",
                message="Cannot classify transactions because anonymized statement text is empty.",
                recovery_instruction=(
                    "Call `extract_and_anonymize_statement_pdf` first or ask the user to upload "
                    "a credit card statement containing transaction lines."
                ),
            ).model_dump()

        classified = classify_transactions_with_flash_lite(text_to_classify)
        if not classified:
            return GuidedToolResult(
                status="error",
                error_code="NO_TRANSACTIONS_DETECTED",
                message="No recognizable transaction lines (date, merchant, amount) were found.",
                recovery_instruction=(
                    "Ask the user to confirm the uploaded document is a credit card statement "
                    "with itemized purchases (e.g. 'MM/DD MERCHANT $AMOUNT')."
                ),
            ).model_dump()

        serialized = [tx.model_dump(mode="json") for tx in classified]
        if tool_context is not None:
            tool_context.state["latest_classified_transactions"] = serialized

        return GuidedToolResult(
            status="success",
            data={
                "transactions": serialized,
                "transaction_count": len(serialized),
            },
            message=f"Classified {len(serialized)} transactions.",
        ).model_dump()
    except Exception as exc:
        return GuidedToolResult(
            status="error",
            error_code="CLASSIFICATION_ERROR",
            message=f"Transaction classification failed: {exc}",
            recovery_instruction=(
                "Retry calling `classify_statement_transactions_batch` with a smaller batch of "
                "statement lines."
            ),
        ).model_dump()


def detect_recurring_monthly_subscriptions(
    transactions: Optional[list[dict[str, Any]]] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Run the deterministic rule-based engine to identify monthly recurring transactions.

    Flags recurring charges sharing the same normalized merchant and amount across billing
    dates (as well as fixed-rate monthly subscriptions), calculates annualized costs, and
    aggregates spending by broad category.

    Args:
        transactions: Optional list of classified transaction dicts. If omitted, reads
            `latest_classified_transactions` from `tool_context.state`.
        tool_context: ADK ToolContext used to store the resulting `StatementAuditSummary`.

    Returns:
        A dictionary matching `GuidedToolResult` with `data.audit_summary` containing
        `total_spend`, `category_totals`, `top_category`, `recurring_subscriptions`,
        and `potential_annual_subscription_savings`.
    """
    try:
        raw_tx_list = transactions
        if not raw_tx_list and tool_context is not None:
            cached_summary = tool_context.state.get("latest_audit_summary")
            if isinstance(cached_summary, dict) and cached_summary:
                return GuidedToolResult(
                    status="success",
                    data={"audit_summary": cached_summary},
                    message="Retrieved pre-computed monthly recurring audit summary from background pipeline.",
                ).model_dump()
            raw_tx_list = tool_context.state.get("latest_classified_transactions", [])

        if not raw_tx_list:
            return GuidedToolResult(
                status="error",
                error_code="MISSING_CLASSIFIED_TRANSACTIONS",
                message="No classified transactions found to analyze for recurring subscriptions.",
                recovery_instruction=(
                    "Invoke `classify_statement_transactions_batch` before calling "
                    "`detect_recurring_monthly_subscriptions`."
                ),
            ).model_dump()

        validated_txs = [ClassifiedTransaction.model_validate(item) for item in raw_tx_list]
        summary = detect_monthly_recurring_transactions(validated_txs)
        summary_dict = summary.model_dump(mode="json")

        if tool_context is not None:
            tool_context.state["latest_audit_summary"] = summary_dict
            schedule_async_memory_consolidation(
                user_id="local_user",
                session_id=tool_context.invocation_id,
                audit_summary=summary_dict,
            )

        return GuidedToolResult(
            status="success",
            data={"audit_summary": summary_dict},
            message=(
                f"Identified {len(summary.recurring_subscriptions)} monthly recurring subscriptions "
                f"with ${summary.potential_annual_subscription_savings:.2f}/year potential savings."
            ),
        ).model_dump()
    except ValidationError as val_err:
        return GuidedToolResult(
            status="error",
            error_code="SCHEMA_VALIDATION_ERROR",
            message=f"Transaction list failed strict schema validation: {val_err}",
            recovery_instruction=(
                "Ensure each transaction includes valid `date`, `raw_description`, `merchant`, "
                "`category`, and non-negative `amount` fields."
            ),
        ).model_dump()
    except Exception as exc:
        return GuidedToolResult(
            status="error",
            error_code="RECURRING_ENGINE_ERROR",
            message=f"Failed to run recurring subscription rule engine: {exc}",
            recovery_instruction="Re-run `classify_statement_transactions_batch` and try again.",
        ).model_dump()


def query_historical_spending_memory_bank(
    user_id: str = "local_user",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Retrieve historical anonymized spending summaries from the Agent Engine Memory Bank.

    Use this tool during synthesis to compare the current statement against prior months'
    spending totals, top categories, and previously detected subscriptions.

    Args:
        user_id: Identifier for the user whose anonymized spending history should be retrieved.
        tool_context: Optional ADK ToolContext to merge current session summary if present.

    Returns:
        A `GuidedToolResult` dictionary containing `data.historical_summaries` and
        `data.latest_session_summary`.
    """
    try:
        memories = retrieve_user_spending_memories(user_id=user_id)
        latest_summary = (
            tool_context.state.get("latest_audit_summary") if tool_context is not None else None
        )
        return GuidedToolResult(
            status="success",
            data={
                "historical_summaries": memories,
                "historical_count": len(memories),
                "latest_session_summary": latest_summary,
            },
            message=f"Retrieved {len(memories)} historical spending records from memory.",
        ).model_dump()
    except Exception as exc:
        return GuidedToolResult(
            status="error",
            error_code="MEMORY_RETRIEVAL_ERROR",
            message=f"Could not query historical spending memory bank: {exc}",
            recovery_instruction=(
                "Proceed with synthesizing recommendations using the current session's "
                "`latest_audit_summary`."
            ),
        ).model_dump()


def flag_subscription_for_cancellation_with_approval(
    merchant: str,
    monthly_amount: float,
    reason: str = "User requested spend reduction action",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
    """Flag a recurring monthly subscription for cancellation and set a budget alert.

    HIGH-STAKES ACTION (Human-in-the-Loop Hook): Modifying the user's active cancellation
    watchlist and budget alert thresholds requires explicit human confirmation.

    Args:
        merchant: Normalized canonical merchant name to flag for cancellation (e.g. 'NETFLIX').
        monthly_amount: Monthly recurring charge amount in statement currency (must be >= 0).
        reason: Explanation for why this subscription is being flagged for cancellation.
        tool_context: ADK ToolContext used to request and verify human confirmation.

    Returns:
        A `GuidedToolResult` dictionary indicating whether the action is awaiting human
        confirmation (`pending_confirmation`) or has been confirmed and applied (`success`).
    """
    try:
        validated_in = FlagSubscriptionInputSchema(
            merchant=merchant,
            monthly_amount=monthly_amount,
            reason=reason,
        )
    except ValidationError:
        return GuidedToolResult(
            status="error",
            error_code="INVALID_SUBSCRIPTION_INPUT",
            message="Monthly subscription amount must be non-negative and merchant non-empty.",
            recovery_instruction="Provide a positive `monthly_amount` and valid `merchant` name.",
        ).model_dump()

    normalized_merchant = validated_in.merchant.strip().upper()
    annual_savings = round(validated_in.monthly_amount * 12, 2)

    if tool_context is not None:
        confirmation = tool_context.tool_confirmation
        if confirmation is None:
            tool_context.request_confirmation(
                hint=(
                    f"Please confirm: Flag recurring subscription '{normalized_merchant}' "
                    f"(${validated_in.monthly_amount:.2f}/mo, ${annual_savings:.2f}/yr) "
                    "for cancellation?"
                ),
                payload={
                    "merchant": normalized_merchant,
                    "monthly_amount": validated_in.monthly_amount,
                    "annual_savings": annual_savings,
                    "reason": validated_in.reason,
                },
            )
            return GuidedToolResult(
                status="pending_confirmation",
                data={
                    "merchant": normalized_merchant,
                    "monthly_amount": validated_in.monthly_amount,
                    "annual_savings": annual_savings,
                },
                message=(
                    f"Awaiting human approval to flag '{normalized_merchant}' "
                    f"(${validated_in.monthly_amount:.2f}/month) for cancellation."
                ),
            ).model_dump()

        if not confirmation.confirmed:
            return GuidedToolResult(
                status="error",
                error_code="USER_REJECTED_ACTION",
                message=f"User declined flagging '{normalized_merchant}' for cancellation.",
                recovery_instruction=(
                    "Acknowledge the user's decision to keep the subscription and offer "
                    "alternative spending reduction tips."
                ),
            ).model_dump()

        watchlist = list(tool_context.state.get("cancellation_watchlist", []))
        watchlist.append(
            {
                "merchant": normalized_merchant,
                "monthly_amount": validated_in.monthly_amount,
                "annual_savings": annual_savings,
                "reason": validated_in.reason,
            }
        )
        tool_context.state["cancellation_watchlist"] = watchlist

    return GuidedToolResult(
        status="success",
        data={
            "merchant": normalized_merchant,
            "monthly_amount": validated_in.monthly_amount,
            "annual_savings": annual_savings,
            "status": "flagged_for_cancellation",
        },
        message=(
            f"Confirmed and flagged '{normalized_merchant}' for cancellation "
            f"(estimated savings: ${annual_savings:.2f}/year)."
        ),
    ).model_dump()


# Wrap high-stakes tool in ADK FunctionTool with require_confirmation=True
flag_subscription_confirmation_tool = FunctionTool(
    func=flag_subscription_for_cancellation_with_approval,
    require_confirmation=True,
)
