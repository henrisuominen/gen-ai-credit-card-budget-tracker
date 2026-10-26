"""Preprocessing pipeline: Document AI OCR, Cloud SDP anonymization, Flash-Lite classification,
and deterministic rule-based monthly recurring transaction detection.

SECURITY POLICY:
- If Document AI, Cloud Sensitive Data Protection (DLP), or Vertex AI
  cannot safely process the document, execution fails closed immediately.
"""

from __future__ import annotations

from collections import defaultdict
import re

from google import genai
from google.genai import types

from app.config import settings
from app.observability import CloudSdpPiiScrubber, PiiAnonymizationError, tracer
from app.schemas import (
    ClassifiedTransaction,
    MonthlyCostClassification,
    RecurringSubscription,
    SpendingCategory,
    StatementAuditSummary,
    TransactionBatchResponse,
)


class DocumentExtractionError(RuntimeError):
    """Raised when Google Cloud Document AI cannot extract text from a statement PDF."""


class TransactionClassificationError(RuntimeError):
    """Raised when `gemini-3.1-flash-lite` cannot classify anonymized statement transactions."""


_CACHED_DOCAI_PROCESSOR_NAME: str = ""


def _resolve_or_create_docai_processor(
    client: object,
    project_id: str,
    location: str,
    configured_processor_id: str,
) -> str:
    """Resolve an existing Document AI OCR processor or auto-provision one in the project."""
    global _CACHED_DOCAI_PROCESSOR_NAME
    from google.cloud import documentai_v1 as documentai  # type: ignore[import-untyped]

    doc_client: documentai.DocumentProcessorServiceClient = client  # type: ignore[assignment]
    if configured_processor_id:
        if configured_processor_id.startswith("projects/"):
            return configured_processor_id
        return str(doc_client.processor_path(project_id, location, configured_processor_id))

    if _CACHED_DOCAI_PROCESSOR_NAME:
        return _CACHED_DOCAI_PROCESSOR_NAME

    parent = doc_client.common_location_path(project_id, location)
    for proc in doc_client.list_processors(parent=parent):
        if proc.type_ == "OCR_PROCESSOR" and proc.state == documentai.Processor.State.ENABLED:
            _CACHED_DOCAI_PROCESSOR_NAME = str(proc.name)
            return _CACHED_DOCAI_PROCESSOR_NAME

    created = doc_client.create_processor(
        parent=parent,
        processor=documentai.Processor(
            display_name="budget-tracker-statement-ocr",
            type_="OCR_PROCESSOR",
        ),
    )
    _CACHED_DOCAI_PROCESSOR_NAME = str(created.name)
    return _CACHED_DOCAI_PROCESSOR_NAME


def extract_text_with_document_ai(
    file_bytes: bytes,
    mime_type: str = "application/pdf",
) -> str:
    """Read a PDF statement in memory using Google Cloud Document AI Online Processing.

    Fails closed by raising `DocumentExtractionError` if Document AI is not configured
    or if the API call fails.

    Args:
        file_bytes: Raw bytes of the uploaded statement PDF.
        mime_type: MIME type of the document (defaults to 'application/pdf').

    Returns:
        Extracted plain text of the statement from Document AI.

    Raises:
        DocumentExtractionError: If Document AI is unconfigured or fails to extract text.
    """
    with tracer.start_as_current_span("preprocessing.extract_text_with_document_ai") as span:
        span.set_attribute("document.mime_type", mime_type)
        span.set_attribute("document.byte_length", len(file_bytes))

        if not file_bytes:
            raise DocumentExtractionError("Fail-closed halt: Uploaded file bytes are empty.")

        if not settings.project_id or settings.project_id == "your-gcp-project-id":
            raise DocumentExtractionError(
                "Fail-closed halt: GOOGLE_CLOUD_PROJECT must be configured to read PDF "
                "statements via Google Cloud Document AI."
            )

        try:
            from google.cloud import documentai_v1 as documentai  # type: ignore[import-untyped]

            opts = {
                "api_endpoint": f"{settings.document_ai_location}-documentai.googleapis.com",
                "quota_project_id": settings.project_id,
            }
            client = documentai.DocumentProcessorServiceClient(client_options=opts)
            name = _resolve_or_create_docai_processor(
                client=client,
                project_id=settings.project_id,
                location=settings.document_ai_location,
                configured_processor_id=settings.document_ai_processor_id,
            )
            raw_document = documentai.RawDocument(
                content=file_bytes,
                mime_type=mime_type,
            )
            request = documentai.ProcessRequest(
                name=name,
                raw_document=raw_document,
            )
            result = client.process_document(request=request)
            if not result.document or not result.document.text or not result.document.text.strip():
                raise DocumentExtractionError(
                    "Fail-closed halt: Document AI returned empty text for the PDF."
                )
            return str(result.document.text)
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                f"Fail-closed halt: Google Cloud Document AI failed to read PDF: {exc}"
            ) from exc


def anonymize_text_with_cloud_sdp(raw_text: str) -> str:
    """Anonymize statement text strictly using Google Cloud Sensitive Data Protection (DLP).

    Fails closed (`PiiAnonymizationError`) if Cloud SDP cannot anonymize the text.
    """
    with tracer.start_as_current_span("preprocessing.anonymize_text_with_cloud_sdp"):
        return CloudSdpPiiScrubber.anonymize_text(raw_text)


def extract_and_anonymize_bytes(
    file_bytes: bytes,
    mime_type: str = "application/pdf",
) -> str:
    """Statelessly extract text via Document AI and immediately scrub PII via Cloud SDP.

    If either Document AI or Cloud SDP fails, raw data is discarded and an exception is raised.
    """
    normalized_mime = (mime_type or "application/pdf").lower().strip()
    if normalized_mime.startswith("text/") or normalized_mime in {
        "application/csv",
        "application/json",
    }:
        try:
            raw_text = file_bytes.decode("utf-8")
        except Exception as exc:
            raise DocumentExtractionError(
                f"Fail-closed halt: Could not decode text document bytes: {exc}"
            ) from exc
    else:
        raw_text = extract_text_with_document_ai(file_bytes=file_bytes, mime_type=normalized_mime)
    try:
        anonymized_text = anonymize_text_with_cloud_sdp(raw_text)
    finally:
        del raw_text
    return anonymized_text


def classify_transactions_with_flash_lite(
    anonymized_text: str,
) -> list[ClassifiedTransaction]:
    """Classify statement purchases using `gemini-2.5-flash-lite` with Pydantic structured output.

    Fails closed by raising `TransactionClassificationError` if Vertex AI is unconfigured
    or if the model does not return valid structured transactions.
    """
    with tracer.start_as_current_span("preprocessing.classify_transactions_with_flash_lite") as span:
        span.set_attribute("llm.model", settings.model_fast_classifier)

        if not anonymized_text or not anonymized_text.strip():
            raise TransactionClassificationError(
                "Cannot classify transactions: anonymized statement text is empty."
            )

        if not settings.project_id or settings.project_id == "your-gcp-project-id":
            raise TransactionClassificationError(
                "Fail-closed halt: GOOGLE_CLOUD_PROJECT is not configured for Vertex AI."
            )

        prompt = (
            "Extract and classify every purchase transaction from the following Cloud SDP "
            "anonymized credit card statement text.\n"
            "For each transaction:\n"
            "1. Normalize the merchant name into a clean uppercase canonical name "
            "(for example, 'UBER LLC #12NYC' -> 'UBER', 'NETFLIX.COM #8812' -> 'NETFLIX').\n"
            "2. Assign the most appropriate broad SpendingCategory.\n"
            "3. Extract the transaction date and positive numeric amount.\n\n"
            f"ANONYMIZED STATEMENT TEXT:\n{anonymized_text}"
        )

        try:
            client = genai.Client(
                vertexai=settings.use_vertexai,
                project=settings.project_id,
                location=settings.location,
            )
            response = client.models.generate_content(
                model=settings.model_fast_classifier,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=TransactionBatchResponse,
                ),
            )
            if not response.text:
                raise TransactionClassificationError(
                    "Model returned an empty response during transaction classification."
                )
            parsed = TransactionBatchResponse.model_validate_json(response.text)
            return parsed.transactions
        except TransactionClassificationError:
            raise
        except Exception as exc:
            raise TransactionClassificationError(
                f"Failed to classify transactions with {settings.model_fast_classifier}: {exc}"
            ) from exc


def _extract_day_of_month(date_str: str) -> int | None:
    """Extract the calendar day of the month (1-31) from a transaction date string."""
    cleaned = date_str.strip()
    if not cleaned:
        return None

    # 1. ISO format YYYY-MM-DD or YYYY/MM/DD
    iso_match = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", cleaned)
    if iso_match:
        day = int(iso_match.group(3))
        return day if 1 <= day <= 31 else None

    # 2. Dot-separated European format DD.MM.YYYY or DD.MM
    dot_match = re.match(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?$", cleaned)
    if dot_match:
        day = int(dot_match.group(1))
        return day if 1 <= day <= 31 else None

    # 3. Slash or hyphen format MM/DD(/YYYY) or DD/MM(/YYYY)
    slash_match = re.match(r"^(\d{1,2})[-/](\d{1,2})(?:[-/](\d{2,4}))?$", cleaned)
    if slash_match:
        first, second = int(slash_match.group(1)), int(slash_match.group(2))
        day = first if first > 12 else second
        return day if 1 <= day <= 31 else None

    # 4. Fallback for text dates (e.g. 'Sep 14, 2026' or '14 Sep')
    nums = [int(n) for n in re.findall(r"\d+", cleaned) if 1 <= int(n) <= 31]
    return nums[0] if nums else None


def detect_monthly_recurring_transactions(
    transactions: list[ClassifiedTransaction],
) -> StatementAuditSummary:
    """Rule-based engine qualifying monthly recurring charges into two categories.

    Qualification Categories:
    1. `Monthly cost happening on the same day` (`SAME_DAY_MONTHLY`):
       - Identical normalized `(merchant, amount)` occurring >= 2 times across distinct
         billing dates where every occurrence falls on the exact same day of the month
         (e.g., 09/01 and 10/01 both fall on day 1).
    2. `Potential monthly costs (happening on differing days)` (`POTENTIAL_MONTHLY_DIFFERING_DAYS`):
       - Identical normalized `(merchant, amount)` occurring >= 2 times across distinct
         billing dates but on differing days of the month (e.g., 09/05 and 10/12), OR
       - Subscription/utility merchants (`Subscriptions & Utilities` or known monthly
         providers) that occur on differing days or only once in a single-cycle upload.

    Args:
        transactions: List of classified credit card transactions.

    Returns:
        Validated `StatementAuditSummary` with category totals and split monthly cost lists.
    """
    with tracer.start_as_current_span("preprocessing.detect_monthly_recurring_transactions"):
        grouped_by_merchant_amount: dict[tuple[str, float], list[ClassifiedTransaction]] = (
            defaultdict(list)
        )
        category_totals: dict[str, float] = defaultdict(float)
        total_spend = 0.0

        for tx in transactions:
            normalized_merchant = tx.merchant.strip().upper()
            rounded_amount = round(float(tx.amount), 2)
            grouped_by_merchant_amount[(normalized_merchant, rounded_amount)].append(tx)
            category_totals[tx.category.value] = round(
                category_totals[tx.category.value] + rounded_amount, 2
            )
            total_spend = round(total_spend + rounded_amount, 2)

        same_day_list: list[RecurringSubscription] = []
        potential_differing_days_list: list[RecurringSubscription] = []
        known_subscription_merchants = {
            "NETFLIX",
            "SPOTIFY",
            "HULU",
            "OPENAI",
            "CHATGPT",
            "DISNEY PLUS",
            "APPLE SERVICES",
            "ICLOUD",
            "YOUTUBE PREMIUM",
            "FITNESS CLUB",
        }

        for (merchant, amount), matching_txs in grouped_by_merchant_amount.items():
            count = len(matching_txs)
            sample_cat = matching_txs[0].category
            distinct_dates = {t.date.strip() for t in matching_txs}
            extracted_days = sorted(
                {
                    day
                    for t in matching_txs
                    if (day := _extract_day_of_month(t.date)) is not None
                }
            )

            if count >= 2 and len(distinct_dates) >= 2:
                if len(extracted_days) == 1:
                    day_num = extracted_days[0]
                    same_day_list.append(
                        RecurringSubscription(
                            merchant=merchant,
                            category=sample_cat,
                            monthly_cost_type=MonthlyCostClassification.SAME_DAY_MONTHLY,
                            billing_days_of_month=extracted_days,
                            monthly_amount=amount,
                            occurrence_count=count,
                            annualized_cost=round(amount * 12, 2),
                            detection_reason=(
                                f"Monthly cost happening on the same day: identical merchant "
                                f"'{merchant}' and amount ${amount:.2f} charged on day {day_num} "
                                f"of the month across {count} billing dates."
                            ),
                        )
                    )
                else:
                    days_label = (
                        ", ".join(str(d) for d in extracted_days)
                        if extracted_days
                        else ", ".join(sorted(distinct_dates))
                    )
                    potential_differing_days_list.append(
                        RecurringSubscription(
                            merchant=merchant,
                            category=sample_cat,
                            monthly_cost_type=(
                                MonthlyCostClassification.POTENTIAL_MONTHLY_DIFFERING_DAYS
                            ),
                            billing_days_of_month=extracted_days,
                            monthly_amount=amount,
                            occurrence_count=count,
                            annualized_cost=round(amount * 12, 2),
                            detection_reason=(
                                f"Potential monthly cost (differing days): identical merchant "
                                f"'{merchant}' and amount ${amount:.2f} appeared {count} times "
                                f"on differing days of the month (days: {days_label})."
                            ),
                        )
                    )
            elif (
                sample_cat == SpendingCategory.SUBSCRIPTIONS_AND_UTILITIES
                or merchant in known_subscription_merchants
            ):
                days_label = (
                    ", ".join(str(d) for d in extracted_days)
                    if extracted_days
                    else next(iter(distinct_dates), "unknown")
                )
                potential_differing_days_list.append(
                    RecurringSubscription(
                        merchant=merchant,
                        category=sample_cat,
                        monthly_cost_type=(
                            MonthlyCostClassification.POTENTIAL_MONTHLY_DIFFERING_DAYS
                        ),
                        billing_days_of_month=extracted_days,
                        monthly_amount=amount,
                        occurrence_count=count,
                        annualized_cost=round(amount * 12, 2),
                        detection_reason=(
                            f"Potential monthly cost: subscription/utility merchant '{merchant}' "
                            f"at ${amount:.2f} (billing day: {days_label}; not repeated on the "
                            "same day across multiple months)."
                        ),
                    )
                )

        same_day_list.sort(key=lambda item: item.annualized_cost, reverse=True)
        potential_differing_days_list.sort(key=lambda item: item.annualized_cost, reverse=True)
        recurring_list = [*same_day_list, *potential_differing_days_list]

        top_category = (
            max(category_totals.items(), key=lambda kv: kv[1])[0]
            if category_totals
            else "None"
        )
        annual_savings = round(sum(r.annualized_cost for r in recurring_list), 2)

        return StatementAuditSummary(
            total_spend=total_spend,
            transaction_count=len(transactions),
            category_totals=dict(category_totals),
            top_category=top_category,
            transactions=transactions,
            recurring_subscriptions=recurring_list,
            same_day_monthly_costs=same_day_list,
            potential_monthly_costs_differing_days=potential_differing_days_list,
            potential_annual_subscription_savings=annual_savings,
        )
