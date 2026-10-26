"""Configuration and Google Cloud Secret Manager loader for Budget Tracker Agent.

All runtime resource IDs and sensitive settings are loaded dynamically via
Application Default Credentials (ADC) and Google Cloud Secret Manager without
any hardcoded API keys.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)


def _detect_default_project() -> str:
    """Resolve the active GCP project from environment or ADC."""
    env_project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if env_project and env_project != "your-gcp-project-id":
        return env_project
    try:
        import google.auth  # type: ignore[import-untyped]

        _, project_id = google.auth.default()
        return project_id or ""
    except Exception:
        return env_project


def load_secret_payload(project_id: str, secret_id: str, version: str = "latest") -> dict[str, Any]:
    """Fetch JSON runtime configuration from Google Cloud Secret Manager.

    Args:
        project_id: The Google Cloud project identifier.
        secret_id: The Secret Manager secret name.
        version: Secret version to access (defaults to 'latest').

    Returns:
        Parsed dictionary of secret configuration values, or an empty dict if
        Secret Manager is unreachable (e.g., during offline unit tests).
    """
    if not project_id or not secret_id or project_id == "your-gcp-project-id":
        return {}

    try:
        from google.cloud import secretmanager  # type: ignore[import-untyped]

        client = secretmanager.SecretManagerServiceClient(
            client_options={"quota_project_id": project_id}
        )
        resource_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
        response = client.access_secret_version(request={"name": resource_name})
        payload_text = response.payload.data.decode("utf-8")
        parsed = json.loads(payload_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


@dataclass(frozen=True)
class AgentSettings:
    """Runtime settings with strategic model routing and GCP resource identifiers."""

    project_id: str = field(default_factory=_detect_default_project)
    location: str = field(default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    gemini_enterprise_location: str = field(
        default_factory=lambda: os.getenv("GEMINI_ENTERPRISE_LOCATION", "global")
    )
    use_vertexai: bool = field(
        default_factory=lambda: os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "true"
    )
    use_enterprise: bool = field(
        default_factory=lambda: os.getenv("GOOGLE_GENAI_USE_ENTERPRISE", "true").lower() == "true"
    )

    # Strategic Model Routing Tiers
    model_fast_classifier: str = field(
        default_factory=lambda: os.getenv("MODEL_FAST_CLASSIFIER", "gemini-3.1-flash-lite")
    )
    model_coordinator: str = field(
        default_factory=lambda: os.getenv("MODEL_COORDINATOR", "gemini-3.5-flash")
    )
    model_synthesis_pro: str = field(
        default_factory=lambda: os.getenv("MODEL_SYNTHESIS_PRO", "gemini-3.1-pro-preview")
    )

    # Secret Manager pointer
    config_secret_id: str = field(
        default_factory=lambda: os.getenv(
            "SECRET_MANAGER_CONFIG_SECRET_ID", "budget-tracker-runtime-config"
        )
    )

    # Resource IDs populated from Secret Manager or environment overrides
    document_ai_location: str = "us"
    document_ai_processor_id: str = ""
    sdp_deidentify_template_id: str = ""
    model_armor_template_id: str = ""
    vertex_memory_bank_id: str = ""
    vertex_search_data_store_id: str = ""
    gemini_enterprise_app_id: str = ""
    agent_engine_resource_id: str = ""
    agent_staging_bucket: str = ""

    # Observability flags
    enable_cloud_trace: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CLOUD_TRACE", "true").lower() == "true"
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @classmethod
    def from_environment_and_secret_manager(cls) -> "AgentSettings":
        """Build settings by merging Secret Manager values with environment overrides."""
        base = cls()
        secret_cfg = load_secret_payload(base.project_id, base.config_secret_id)

        def _resolve(env_key: str, secret_key: str, default: str = "") -> str:
            val = os.getenv(env_key, "").strip()
            if val and not val.startswith("your-"):
                return val
            return str(secret_cfg.get(secret_key, default))

        return cls(
            project_id=base.project_id,
            location=base.location,
            gemini_enterprise_location=base.gemini_enterprise_location,
            use_vertexai=base.use_vertexai,
            use_enterprise=base.use_enterprise,
            model_fast_classifier=base.model_fast_classifier,
            model_coordinator=base.model_coordinator,
            model_synthesis_pro=base.model_synthesis_pro,
            config_secret_id=base.config_secret_id,
            document_ai_location=_resolve("DOCUMENT_AI_LOCATION", "document_ai_location", "us"),
            document_ai_processor_id=_resolve(
                "DOCUMENT_AI_PROCESSOR_ID", "document_ai_processor_id"
            ),
            sdp_deidentify_template_id=_resolve(
                "SDP_DEIDENTIFY_TEMPLATE_ID", "sdp_deidentify_template_id"
            ),
            model_armor_template_id=_resolve(
                "MODEL_ARMOR_TEMPLATE_ID", "model_armor_template_id"
            ),
            vertex_memory_bank_id=_resolve("VERTEX_MEMORY_BANK_ID", "vertex_memory_bank_id"),
            vertex_search_data_store_id=_resolve(
                "VERTEX_SEARCH_DATA_STORE_ID", "vertex_search_data_store_id"
            ),
            gemini_enterprise_app_id=_resolve(
                "GEMINI_ENTERPRISE_APP_ID", "gemini_enterprise_app_id"
            ),
            agent_engine_resource_id=_resolve(
                "AGENT_ENGINE_RESOURCE_ID", "agent_engine_resource_id"
            ),
            agent_staging_bucket=_resolve("AGENT_STAGING_BUCKET", "agent_staging_bucket"),
            enable_cloud_trace=base.enable_cloud_trace,
            log_level=base.log_level,
        )


settings = AgentSettings.from_environment_and_secret_manager()
