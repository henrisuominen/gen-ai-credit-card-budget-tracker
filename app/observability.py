"""Observability, OpenTelemetry Distributed Tracing, Structured JSON Logging, and Fail-Closed Cloud SDP PII Redaction.

Emits structured JSON log payloads with trace/session context, captures `AGENT_INTENT` / `TOOL_INTENT`
before execution and `AGENT_OUTCOME` / `TOOL_OUTCOME` after execution via an ADK `BasePlugin`,
instruments agent and tool lifecycles with OpenTelemetry spans, and scrubs payloads strictly via
Google Cloud Sensitive Data Protection (DLP). Fails closed (`PiiAnonymizationError`) if Cloud SDP
cannot guarantee a safe response.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings


class PiiAnonymizationError(RuntimeError):
    """Raised when Google Cloud Sensitive Data Protection (DLP) cannot anonymize content safely."""


class CloudSdpPiiScrubber:
    """Strict Fail-Closed PII Anonymizer backed exclusively by Google Cloud Sensitive Data Protection (DLP)."""

    DEFAULT_INFO_TYPES = [
        {"name": "PERSON_NAME"},
        {"name": "CREDIT_CARD_NUMBER"},
        {"name": "FINANCIAL_ACCOUNT_NUMBER"},
        {"name": "IBAN_CODE"},
        {"name": "US_SOCIAL_SECURITY_NUMBER"},
        {"name": "STREET_ADDRESS"},
        {"name": "PHONE_NUMBER"},
        {"name": "EMAIL_ADDRESS"},
    ]

    CUSTOM_INFO_TYPES = [
        {
            "info_type": {"name": "CREDIT_CARD_OR_ACCOUNT_NUMBER"},
            "regex": {"pattern": r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{1,7}"},
        },
    ]

    _dlp_clients: dict[str, Any] = {}
    _verified_scrub_cache: dict[tuple[str, str, str], str] = {}

    @classmethod
    def anonymize_text(cls, text: str) -> str:
        """De-identify text strictly using Google Cloud Sensitive Data Protection (`deidentify_content`).

        Fails closed by raising `PiiAnonymizationError` if `GOOGLE_CLOUD_PROJECT` is not configured
        or if the Cloud SDP API call fails for any reason. Never falls back to regex.

        Args:
            text: Raw input text that may contain PII.

        Returns:
            PII-redacted text returned by Google Cloud Sensitive Data Protection.

        Raises:
            PiiAnonymizationError: If Cloud SDP is unconfigured, unreachable, or returns an error.
        """
        if not text or not text.strip():
            return ""

        if not settings.project_id or settings.project_id == "your-gcp-project-id":
            raise PiiAnonymizationError(
                "Fail-closed security halt: GOOGLE_CLOUD_PROJECT is not configured. "
                "Cannot anonymize PII via Google Cloud Sensitive Data Protection."
            )

        cache_key = (settings.project_id, settings.sdp_deidentify_template_id, text)
        cached = cls._verified_scrub_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from google.cloud import dlp_v2  # type: ignore[import-untyped]

            dlp_client = cls._dlp_clients.get(settings.project_id)
            if dlp_client is None:
                dlp_client = dlp_v2.DlpServiceClient(
                    client_options={"quota_project_id": settings.project_id}
                )
                cls._dlp_clients[settings.project_id] = dlp_client

            parent = f"projects/{settings.project_id}/locations/global"

            request: dict[str, Any] = {
                "parent": parent,
                "item": {"value": text},
                "inspect_config": {
                    "info_types": cls.DEFAULT_INFO_TYPES,
                    "custom_info_types": cls.CUSTOM_INFO_TYPES,
                    "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
                },
            }
            if settings.sdp_deidentify_template_id:
                request["deidentify_template_name"] = settings.sdp_deidentify_template_id
            else:
                request["deidentify_config"] = {
                    "info_type_transformations": {
                        "transformations": [
                            {
                                "primitive_transformation": {
                                    "replace_with_info_type_config": {}
                                }
                            }
                        ]
                    }
                }

            response = dlp_client.deidentify_content(request=request)
            if response is None or response.item is None:
                raise PiiAnonymizationError(
                    "Fail-closed security halt: Cloud SDP returned an empty de-identification response."
                )
            scrubbed = str(response.item.value)
            cls._verified_scrub_cache[cache_key] = scrubbed
            cls._verified_scrub_cache[
                (settings.project_id, settings.sdp_deidentify_template_id, scrubbed)
            ] = scrubbed
            return scrubbed
        except PiiAnonymizationError:
            raise
        except Exception as exc:
            raise PiiAnonymizationError(
                f"Fail-closed security halt: Google Cloud Sensitive Data Protection API failed: {exc}"
            ) from exc


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON without logging raw user payloads."""

    def format(self, record: logging.LogRecord) -> str:
        span = trace.get_current_span()
        span_ctx = span.get_span_context() if span else None
        trace_id = (
            format(span_ctx.trace_id, "032x")
            if span_ctx and span_ctx.is_valid
            else getattr(record, "trace_id", None)
        )

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "event_type": getattr(record, "event_type", "APPLICATION_LOG"),
            "agent_name": getattr(record, "agent_name", None),
            "tool_name": getattr(record, "tool_name", None),
            "session_id": getattr(record, "session_id", None),
            "invocation_id": getattr(record, "invocation_id", None),
            "trace_id": trace_id,
        }
        extra_metadata = getattr(record, "metadata", None)
        if isinstance(extra_metadata, dict):
            payload["metadata"] = extra_metadata

        return json.dumps({k: v for k, v in payload.items() if v is not None}, default=str)


def configure_observability() -> tuple[logging.Logger, trace.Tracer]:
    """Configure structured JSON logging and OpenTelemetry distributed tracing."""
    logger = logging.getLogger("budget_tracker_agent")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "gen-ai-credit-card-budget-tracker",
                "service.version": "0.1.0",
            }
        )
    )
    if settings.enable_cloud_trace and settings.project_id:
        try:
            from opentelemetry.exporter.cloud_trace import (  # type: ignore[import-untyped]
                CloudTraceSpanExporter,
            )

            exporter = CloudTraceSpanExporter(project_id=settings.project_id)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except Exception:
            pass

    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("budget_tracker_agent.tracer")
    return logger, tracer


structured_logger, tracer = configure_observability()


def log_structured_event(
    event_type: str,
    message: str,
    *,
    severity: str = "INFO",
    agent_name: str | None = None,
    tool_name: str | None = None,
    session_id: str | None = None,
    invocation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a structured JSON log event."""
    level = getattr(logging, severity.upper(), logging.INFO)
    structured_logger.log(
        level,
        message,
        extra={
            "event_type": event_type,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "session_id": session_id,
            "invocation_id": invocation_id,
            "metadata": metadata or {},
        },
    )


class ObservabilityPlugin(BasePlugin):
    """ADK Plugin capturing explicit Intent vs. Outcome records and OpenTelemetry spans."""

    def __init__(self) -> None:
        super().__init__(name="observability_intent_outcome_plugin")
        self._tool_start_times: dict[str, float] = {}

    async def before_agent_callback(
        self, *, agent: Any, callback_context: CallbackContext
    ) -> None:
        log_structured_event(
            "AGENT_INTENT",
            f"Agent '{agent.name}' starting turn execution.",
            agent_name=agent.name,
            session_id=callback_context.session.id,
            invocation_id=callback_context.invocation_id,
            metadata={"intended_action": "process_user_or_pipeline_turn"},
        )
        return None

    async def after_agent_callback(
        self, *, agent: Any, callback_context: CallbackContext
    ) -> None:
        log_structured_event(
            "AGENT_OUTCOME",
            f"Agent '{agent.name}' completed turn execution.",
            agent_name=agent.name,
            session_id=callback_context.session.id,
            invocation_id=callback_context.invocation_id,
            metadata={"outcome_status": "completed"},
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        call_key = f"{tool_context.invocation_id}:{tool.name}"
        self._tool_start_times[call_key] = time.perf_counter()
        # Only log argument keys (never raw argument values) to prevent accidental PII logging
        log_structured_event(
            "TOOL_INTENT",
            f"Intending to execute tool '{tool.name}'.",
            agent_name=tool_context.agent_name,
            tool_name=tool.name,
            invocation_id=tool_context.invocation_id,
            metadata={"intended_arg_keys": list(tool_args.keys())},
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        call_key = f"{tool_context.invocation_id}:{tool.name}"
        started = self._tool_start_times.pop(call_key, time.perf_counter())
        duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
        status = result.get("status", "success") if isinstance(result, dict) else "success"
        log_structured_event(
            "TOOL_OUTCOME",
            f"Completed tool '{tool.name}' with status '{status}' in {duration_ms}ms.",
            agent_name=tool_context.agent_name,
            tool_name=tool.name,
            invocation_id=tool_context.invocation_id,
            metadata={
                "status": status,
                "duration_ms": duration_ms,
                "intended_arg_keys": list(tool_args.keys()),
            },
        )
        return None
