"""Unit tests for Fail-Closed Cloud SDP PII anonymization, Document AI extraction,
Pre-Ingestion Gatekeeper, strict Pydantic schemas, recurring rules, and HITL hooks.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import types
import pytest
from pydantic import ValidationError

from app.guardrails import EnterpriseGuardrailPlugin
from app.observability import CloudSdpPiiScrubber, PiiAnonymizationError
from app.preprocessing import (
    DocumentExtractionError,
    TransactionClassificationError,
    classify_transactions_with_flash_lite,
    detect_monthly_recurring_transactions,
    extract_and_anonymize_bytes,
    extract_text_with_document_ai,
)
from app.schemas import (
    ClassifiedTransaction,
    MonthlyCostClassification,
    SpendingCategory,
    TransactionBatchResponse,
)
from app.tools import (
    classify_statement_transactions_batch,
    detect_recurring_monthly_subscriptions,
    extract_and_anonymize_statement_pdf,
    flag_subscription_for_cancellation_with_approval,
)

SAMPLE_RAW_STATEMENT = """
Cardholder Name: Jane Doe
Account Number: 4111-2222-3333-4444
09/01 NETFLIX.COM #1029 $15.99
10/01 NETFLIX.COM #2048 $15.99
09/14 UBER LLC #12NYC $24.50
"""

SAMPLE_SDP_ANONYMIZED_STATEMENT = """
Cardholder Name: [PERSON_NAME]
Account Number: [CREDIT_CARD_NUMBER]
09/01 NETFLIX.COM #1029 $15.99
10/01 NETFLIX.COM #2048 $15.99
09/14 UBER LLC #12NYC $24.50
"""


def test_cloud_sdp_fails_closed_when_unconfigured_or_api_errors() -> None:
    """Verify CloudSdpPiiScrubber NEVER falls back to regex and raises PiiAnonymizationError."""
    with patch("app.observability.settings", SimpleNamespace(project_id="", sdp_deidentify_template_id="")):
        with pytest.raises(PiiAnonymizationError, match="Fail-closed security halt"):
            CloudSdpPiiScrubber.anonymize_text(SAMPLE_RAW_STATEMENT)


def test_document_ai_fails_closed_when_unconfigured() -> None:
    """Verify Document AI extraction NEVER falls back to local parsing and raises DocumentExtractionError."""
    with patch(
        "app.preprocessing.settings",
        SimpleNamespace(project_id="", document_ai_processor_id="", document_ai_location="us"),
    ):
        with pytest.raises(DocumentExtractionError, match="Fail-closed halt"):
            extract_text_with_document_ai(b"%PDF-1.4 raw bytes")


def test_pre_ingestion_gatekeeper_fails_closed_when_cloud_sdp_unavailable() -> None:
    """Ensure uploaded PDF bytes are discarded and turn is halted if Cloud SDP/DocAI fails."""
    plugin = EnterpriseGuardrailPlugin()
    user_msg = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=b"%PDF-1.4 secret", mime_type="application/pdf"),
            types.Part.from_text(text="Summarize my statement"),
        ],
    )
    fake_context = SimpleNamespace(session=SimpleNamespace(state={}))

    with patch(
        "app.preprocessing.settings",
        SimpleNamespace(project_id="", document_ai_processor_id="", document_ai_location="us"),
    ):
        halted_msg = asyncio.run(
            plugin.on_user_message_callback(
                invocation_context=fake_context,  # type: ignore[arg-type]
                user_message=user_msg,
            )
        )

    assert halted_msg is not None
    assert halted_msg.parts is not None
    for part in halted_msg.parts:
        assert getattr(part, "inline_data", None) is None
    assert "[SECURITY_HALT_UNSAFE_TO_PROCEED]" in (halted_msg.parts[0].text or "")
    assert "security_halt_reason" in fake_context.session.state


def test_pre_ingestion_gatekeeper_succeeds_with_document_ai_and_cloud_sdp() -> None:
    """Verify Gatekeeper replaces PDF bytes with Cloud SDP anonymized text when GCP APIs succeed."""
    plugin = EnterpriseGuardrailPlugin()
    user_msg = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(data=b"%PDF-1.4 raw statement", mime_type="application/pdf"),
        ],
    )
    fake_context = SimpleNamespace(session=SimpleNamespace(state={}))

    with (
        patch("app.preprocessing.extract_text_with_document_ai", return_value=SAMPLE_RAW_STATEMENT),
        patch(
            "app.observability.CloudSdpPiiScrubber.anonymize_text",
            return_value=SAMPLE_SDP_ANONYMIZED_STATEMENT,
        ),
    ):
        sanitized_msg = asyncio.run(
            plugin.on_user_message_callback(
                invocation_context=fake_context,  # type: ignore[arg-type]
                user_message=user_msg,
            )
        )

    assert sanitized_msg is not None
    assert sanitized_msg.parts is not None
    assert getattr(sanitized_msg.parts[0], "inline_data", None) is None
    text_out = sanitized_msg.parts[0].text or ""
    assert "[ANONYMIZED_STATEMENT_TEXT]" in text_out
    assert "4111-2222-3333-4444" not in text_out
    assert "[CREDIT_CARD_NUMBER]" in text_out


def test_flash_lite_classification_and_rule_based_recurring_detection() -> None:
    """Verify Flash-Lite structured output parsing and deterministic monthly recurring detection."""
    mock_batch = TransactionBatchResponse(
        transactions=[
            ClassifiedTransaction(
                date="2026-09-01",
                raw_description="NETFLIX.COM #1029",
                merchant="NETFLIX",
                category=SpendingCategory.SUBSCRIPTIONS_AND_UTILITIES,
                amount=15.99,
            ),
            ClassifiedTransaction(
                date="2026-10-01",
                raw_description="NETFLIX.COM #2048",
                merchant="NETFLIX",
                category=SpendingCategory.SUBSCRIPTIONS_AND_UTILITIES,
                amount=15.99,
            ),
            ClassifiedTransaction(
                date="2026-09-14",
                raw_description="UBER LLC #12NYC",
                merchant="UBER",
                category=SpendingCategory.TRANSPORTATION,
                amount=24.50,
            ),
            ClassifiedTransaction(
                date="2026-09-05",
                raw_description="SPOTIFY USA #11",
                merchant="SPOTIFY",
                category=SpendingCategory.SUBSCRIPTIONS_AND_UTILITIES,
                amount=11.99,
            ),
            ClassifiedTransaction(
                date="2026-10-12",
                raw_description="SPOTIFY USA #22",
                merchant="SPOTIFY",
                category=SpendingCategory.SUBSCRIPTIONS_AND_UTILITIES,
                amount=11.99,
            ),
        ]
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text=mock_batch.model_dump_json()
    )

    with (
        patch(
            "app.preprocessing.settings",
            SimpleNamespace(
                project_id="test-project",
                location="global",
                use_vertexai=True,
                model_fast_classifier="gemini-3.1-flash-lite",
            ),
        ),
        patch("app.preprocessing.genai.Client", return_value=mock_client),
    ):
        transactions = classify_transactions_with_flash_lite(SAMPLE_SDP_ANONYMIZED_STATEMENT)

    assert len(transactions) == 5
    uber_tx = next(t for t in transactions if t.merchant == "UBER")
    assert uber_tx.category == SpendingCategory.TRANSPORTATION
    assert uber_tx.amount == 24.50

    summary = detect_monthly_recurring_transactions(transactions)
    assert summary.total_spend == pytest.approx(80.46, rel=1e-3)
    assert len(summary.same_day_monthly_costs) == 1
    assert summary.same_day_monthly_costs[0].merchant == "NETFLIX"
    assert (
        summary.same_day_monthly_costs[0].monthly_cost_type
        == MonthlyCostClassification.SAME_DAY_MONTHLY
    )
    assert summary.same_day_monthly_costs[0].billing_days_of_month == [1]
    assert summary.same_day_monthly_costs[0].annualized_cost == pytest.approx(191.88, rel=1e-3)

    assert len(summary.potential_monthly_costs_differing_days) == 1
    assert summary.potential_monthly_costs_differing_days[0].merchant == "SPOTIFY"
    assert (
        summary.potential_monthly_costs_differing_days[0].monthly_cost_type
        == MonthlyCostClassification.POTENTIAL_MONTHLY_DIFFERING_DAYS
    )
    assert summary.potential_monthly_costs_differing_days[0].billing_days_of_month == [5, 12]
    assert summary.potential_monthly_costs_differing_days[0].annualized_cost == pytest.approx(
        143.88, rel=1e-3
    )


def test_explicit_json_schema_forbids_invalid_fields_and_negative_amounts() -> None:
    """Verify strict Pydantic schemas reject negative amounts or unexpected properties."""
    with pytest.raises(ValidationError):
        ClassifiedTransaction(
            date="2026-09-14",
            raw_description="UBER LLC #12NYC",
            merchant="UBER",
            category=SpendingCategory.TRANSPORTATION,
            amount=-10.0,
        )

    with pytest.raises(ValidationError):
        ClassifiedTransaction(
            date="2026-09-14",
            raw_description="UBER LLC #12NYC",
            merchant="UBER",
            category=SpendingCategory.TRANSPORTATION,
            amount=24.50,
            unexpected_field="not_allowed",  # type: ignore[call-arg]
        )


def test_guided_error_handling_returns_recovery_instructions() -> None:
    """Verify tools return descriptive recovery instructions when operations fail closed."""
    missing_file_res = extract_and_anonymize_statement_pdf(file_path="/nonexistent/statement.pdf")
    assert missing_file_res["status"] == "error"
    assert missing_file_res["error_code"] == "FILE_NOT_FOUND"
    assert missing_file_res["recovery_instruction"]

    empty_classify_res = classify_statement_transactions_batch(anonymized_statement_text="")
    assert empty_classify_res["status"] == "error"
    assert empty_classify_res["error_code"] == "NO_ANONYMIZED_TEXT_AVAILABLE"
    assert empty_classify_res["recovery_instruction"]

    empty_recurring_res = detect_recurring_monthly_subscriptions(transactions=[])
    assert empty_recurring_res["status"] == "error"
    assert empty_recurring_res["error_code"] == "MISSING_CLASSIFIED_TRANSACTIONS"


def test_human_in_the_loop_confirmation_hook() -> None:
    """Verify high-stakes subscription cancellation requests confirmation before executing."""
    requested_hints: list[str] = []

    class DummyContext:
        tool_confirmation = None
        state: dict[str, object] = {}

        def request_confirmation(self, *, hint: str, payload: dict[str, object]) -> None:
            requested_hints.append(hint)

    ctx = DummyContext()
    res = flag_subscription_for_cancellation_with_approval(
        merchant="Netflix",
        monthly_amount=15.99,
        tool_context=ctx,  # type: ignore[arg-type]
    )
    assert res["status"] == "pending_confirmation"
    assert len(requested_hints) == 1
    assert "NETFLIX" in requested_hints[0]
