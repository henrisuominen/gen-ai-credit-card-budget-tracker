"""Enterprise Guardrail Plugin: Fail-Closed Pre-Ingestion PII Gatekeeper, Model Armor, and Self-Eval.

SECURITY POLICY (Fail-Closed):
- Whenever a user uploads a PDF or sends text, this plugin requires Google Cloud Document AI
  (for PDFs) and Google Cloud Sensitive Data Protection (DLP) to succeed before any content
  is stored in Session State or sent to an LLM.
- If Document AI or Cloud SDP fails or is unavailable, the raw attachment/input is immediately
  discarded from memory and the turn is halted with a safe error response.
"""

from __future__ import annotations

import re

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

from app.config import settings
from app.observability import (
    CloudSdpPiiScrubber,
    PiiAnonymizationError,
    log_structured_event,
    tracer,
)
from app.preprocessing import (
    DocumentExtractionError,
    TransactionClassificationError,
    classify_transactions_with_flash_lite,
    detect_monthly_recurring_transactions,
    extract_and_anonymize_bytes,
)

_PROMPT_INJECTION_PATTERNS = re.compile(
    r"(?i)(ignore\s+all\s+previous\s+instructions|system\s+override|reveal\s+system\s+prompt|"
    r"bypass\s+pii\s+redaction|output\s+raw\s+credit\s+card)"
)

_SECURITY_HALT_PREFIX = "[SECURITY_HALT_UNSAFE_TO_PROCEED]"


def _screen_with_model_armor(text: str) -> tuple[bool, str]:
    """Screen user input against prompt injection and Google Cloud Model Armor API."""
    if _PROMPT_INJECTION_PATTERNS.search(text):
        return False, "Blocked potential prompt injection or guardrail bypass attempt."

    if not settings.project_id or not settings.model_armor_template_id:
        return True, ""

    try:
        import json
        import urllib.request
        import google.auth  # type: ignore[import-untyped]
        from google.auth.transport.requests import Request  # type: ignore[import-untyped]

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        url = (
            f"https://modelarmor.{settings.location}.rep.googleapis.com/v1/"
            f"{settings.model_armor_template_id}:sanitizeUserPrompt"
        )
        req_body = json.dumps({"userPromptData": {"text": text}}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_body,
            headers={
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            match_state = (
                payload.get("sanitizationResult", {}).get("filterMatchState", "NO_MATCH_FOUND")
            )
            if match_state == "MATCH_FOUND":
                return False, "Blocked by Google Cloud Model Armor policy template."
    except Exception as exc:
        return False, f"Fail-closed halt: Model Armor verification failed: {exc}"

    return True, ""


def _sanitize_content_parts_fail_closed(content: types.Content | None) -> types.Content | None:
    """Anonymize all PDF/document attachments and text parts via Cloud SDP or fail closed."""
    if not content or not content.parts:
        return content

    sanitized_parts: list[types.Part] = []
    for part in content.parts:
        if getattr(part, "inline_data", None) and part.inline_data and part.inline_data.data:
            mime_type = part.inline_data.mime_type or "application/pdf"
            raw_bytes = part.inline_data.data
            try:
                anonymized_text = extract_and_anonymize_bytes(
                    file_bytes=raw_bytes,
                    mime_type=mime_type,
                )
            finally:
                del raw_bytes

            log_structured_event(
                "PRE_INGESTION_PII_GATEKEEPER",
                "Intercepted uploaded document in memory and replaced with Cloud SDP anonymized text.",
                metadata={"original_mime_type": mime_type},
            )
            sanitized_parts.append(
                types.Part.from_text(
                    text=(
                        "[ANONYMIZED_STATEMENT_TEXT]\n"
                        f"{anonymized_text}\n"
                        "[/ANONYMIZED_STATEMENT_TEXT]"
                    )
                )
            )
        elif getattr(part, "text", None) and part.text:
            if part.text.startswith(_SECURITY_HALT_PREFIX):
                sanitized_parts.append(part)
            elif (
                "[ANONYMIZED_STATEMENT_TEXT]" in part.text
                or "[PRECOMPUTED_STATEMENT_AUDIT_SUMMARY]" in part.text
            ):
                # Already anonymized by on_user_message_callback
                sanitized_parts.append(part)
            else:
                scrubbed_text = CloudSdpPiiScrubber.anonymize_text(part.text)
                sanitized_parts.append(types.Part.from_text(text=scrubbed_text))
        else:
            sanitized_parts.append(part)

    return types.Content(role=content.role, parts=sanitized_parts)


class EnterpriseGuardrailPlugin(BasePlugin):
    """Fail-Closed ADK Plugin enforcing Pre-Ingestion PDF Anonymization, Model Armor, and Self-Eval."""

    def __init__(self) -> None:
        super().__init__(name="enterprise_guardrail_plugin")

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Intercept user uploads BEFORE session persistence or agent execution.

        If Document AI or Cloud SDP fails, the raw upload is discarded in memory and replaced
        with a `_SECURITY_HALT_PREFIX` marker so raw PII is never stored in the session.
        """
        with tracer.start_as_current_span("guardrail.on_user_message_callback"):
            try:
                sanitized = _sanitize_content_parts_fail_closed(user_message)
                if sanitized and sanitized.parts:
                    extra_parts: list[types.Part] = []
                    for part in sanitized.parts:
                        if part.text and "[ANONYMIZED_STATEMENT_TEXT]" in part.text:
                            extracted = (
                                part.text.split("[ANONYMIZED_STATEMENT_TEXT]", 1)[1]
                                .split("[/ANONYMIZED_STATEMENT_TEXT]", 1)[0]
                                .strip()
                            )
                            invocation_context.session.state[
                                "latest_anonymized_statement_text"
                            ] = extracted
                            # Run Steps 1 & 2 (Flash-Lite classification + rule-based monthly detection)
                            # silently in the background on initial upload so the user only gets Response 3.
                            try:
                                classified = classify_transactions_with_flash_lite(extracted)
                                if classified:
                                    serialized_txs = [
                                        tx.model_dump(mode="json") for tx in classified
                                    ]
                                    invocation_context.session.state[
                                        "latest_classified_transactions"
                                    ] = serialized_txs
                                    summary = detect_monthly_recurring_transactions(classified)
                                    summary_dict = summary.model_dump(mode="json")
                                    invocation_context.session.state[
                                        "latest_audit_summary"
                                    ] = summary_dict
                                    extra_parts.append(
                                        types.Part.from_text(
                                            text=(
                                                "[PRECOMPUTED_STATEMENT_AUDIT_SUMMARY]\n"
                                                f"{summary.model_dump_json(indent=2)}\n"
                                                "[/PRECOMPUTED_STATEMENT_AUDIT_SUMMARY]"
                                            )
                                        )
                                    )
                            except TransactionClassificationError as classify_exc:
                                log_structured_event(
                                    "BACKGROUND_PREPROCESSING_SKIPPED",
                                    f"Deferred transaction classification to tool execution: {classify_exc}",
                                    severity="WARNING",
                                )
                    if extra_parts:
                        sanitized = types.Content(
                            role=sanitized.role,
                            parts=[*sanitized.parts, *extra_parts],
                        )
                return sanitized
            except (PiiAnonymizationError, DocumentExtractionError) as exc:
                log_structured_event(
                    "GUARDRAIL_FAIL_CLOSED_INGESTION",
                    f"Halted ingestion because safe anonymization could not be guaranteed: {exc}",
                    severity="ERROR",
                )
                invocation_context.session.state["security_halt_reason"] = str(exc)
                return types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=f"{_SECURITY_HALT_PREFIX} {exc}"
                        )
                    ],
                )

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Fail-closed check before any LLM invocation."""
        with tracer.start_as_current_span("guardrail.before_model_callback"):
            halt_reason = callback_context.state.get("security_halt_reason")
            if halt_reason:
                callback_context.state["security_halt_reason"] = None
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text=(
                                    "Security Halt (Fail-Closed): Unable to process your document "
                                    "or message because Google Cloud Document AI / Cloud Sensitive "
                                    "Data Protection could not verify a PII-free payload. Your raw "
                                    f"data was discarded in memory and never sent to the model. ({halt_reason})"
                                )
                            )
                        ],
                    )
                )

            if llm_request.contents:
                sanitized_contents: list[types.Content] = []
                total_items = len(llm_request.contents)
                for idx, content_item in enumerate(llm_request.contents):
                    is_latest_turn = idx == (total_items - 1)
                    try:
                        cleaned_item = _sanitize_content_parts_fail_closed(content_item)
                    except (PiiAnonymizationError, DocumentExtractionError) as exc:
                        log_structured_event(
                            "GUARDRAIL_FAIL_CLOSED_BEFORE_MODEL",
                            f"Blocked model call due to anonymization failure: {exc}",
                            severity="ERROR",
                            agent_name=callback_context.agent_name,
                        )
                        return LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part.from_text(
                                        text=(
                                            "Security Halt (Fail-Closed): Cannot invoke the LLM "
                                            "without verified Cloud SDP PII anonymization. "
                                            f"Details: {exc}"
                                        )
                                    )
                                ],
                            )
                        )

                    if cleaned_item and cleaned_item.parts:
                        kept_parts: list[types.Part] = []
                        for part in cleaned_item.parts:
                            if part.text:
                                if part.text.startswith(_SECURITY_HALT_PREFIX):
                                    if is_latest_turn:
                                        return LlmResponse(
                                            content=types.Content(
                                                role="model",
                                                parts=[
                                                    types.Part.from_text(
                                                        text=(
                                                            "Security Halt (Fail-Closed): Request aborted "
                                                            f"because PII anonymization did not succeed ({part.text})."
                                                        )
                                                    )
                                                ],
                                            )
                                        )
                                    # Skip failed messages from earlier turns so they do not poison the session
                                    continue
                                is_safe, reason = _screen_with_model_armor(part.text)
                                if not is_safe:
                                    log_structured_event(
                                        "GUARDRAIL_BLOCKED_INPUT",
                                        reason,
                                        severity="WARNING",
                                        agent_name=callback_context.agent_name,
                                    )
                                    return LlmResponse(
                                        content=types.Content(
                                            role="model",
                                            parts=[
                                                types.Part.from_text(
                                                    text=(
                                                        "Request blocked by enterprise security "
                                                        f"guardrail: {reason}"
                                                    )
                                                )
                                            ],
                                        )
                                    )
                            kept_parts.append(part)
                        if kept_parts:
                            sanitized_contents.append(
                                types.Content(role=cleaned_item.role, parts=kept_parts)
                            )
                llm_request.contents = sanitized_contents
            return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Fail-closed output guardrail verifying model response via Cloud SDP."""
        with tracer.start_as_current_span("guardrail.after_model_callback"):
            if not llm_response.content or not llm_response.content.parts:
                return None

            has_function_call = any(
                getattr(p, "function_call", None) is not None for p in llm_response.content.parts
            )
            suppress_intermediate_text = (
                callback_context.agent_name
                in {"ingestion_anonymizer_agent", "transaction_classifier_agent"}
                and not has_function_call
            )

            new_parts: list[types.Part] = []
            for part in llm_response.content.parts:
                if part.text:
                    if suppress_intermediate_text:
                        continue
                    try:
                        scrubbed = CloudSdpPiiScrubber.anonymize_text(part.text)
                    except PiiAnonymizationError as exc:
                        log_structured_event(
                            "GUARDRAIL_FAIL_CLOSED_OUTPUT",
                            f"Blocked model output because Cloud SDP output verification failed: {exc}",
                            severity="ERROR",
                            agent_name=callback_context.agent_name,
                        )
                        return LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part.from_text(
                                        text=(
                                            "Security Halt (Fail-Closed): Response withheld because "
                                            "Cloud SDP output safety verification could not complete."
                                        )
                                    )
                                ],
                            )
                        )
                    new_parts.append(types.Part.from_text(text=scrubbed))
                else:
                    new_parts.append(part)

            return LlmResponse(
                content=types.Content(role=llm_response.content.role, parts=new_parts)
            )
