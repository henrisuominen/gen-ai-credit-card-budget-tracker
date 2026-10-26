"""Deploy the Credit Card Budget Tracker ADK Agent to Vertex AI Agent Runtime (Agent Engine).

Usage:
    python3 deployment/deploy_agent_engine.py
"""

from __future__ import annotations

import vertexai  # type: ignore[import-untyped]
from vertexai import agent_engines  # type: ignore[import-untyped]

from app.agents import create_agent_engine_app
from app.config import settings
from app.observability import log_structured_event


def deploy_to_agent_runtime() -> str:
    """Package and deploy the ADK agent hierarchy to Vertex AI Agent Runtime."""
    if not settings.project_id or not settings.agent_staging_bucket:
        raise ValueError(
            "Both GOOGLE_CLOUD_PROJECT and AGENT_STAGING_BUCKET (via Secret Manager or env) "
            "must be configured before deploying to Vertex AI Agent Runtime."
        )

    vertexai.init(
        project=settings.project_id,
        location=settings.location,
        staging_bucket=settings.agent_staging_bucket,
    )

    adk_app = create_agent_engine_app()
    remote_agent = agent_engines.create(
        adk_app,
        display_name="credit-card-budget-tracker-agent",
        description=(
            "Anonymizes credit card statements with Cloud SDP & Document AI, classifies spend "
            "with gemini-2.5-flash-lite, detects monthly subscriptions, and synthesizes savings."
        ),
        requirements=[
            "google-adk>=1.0.0",
            "google-genai>=1.0.0",
            "google-cloud-aiplatform[adk,agent_engines]>=1.70.0",
            "google-cloud-documentai>=2.29.0",
            "google-cloud-dlp>=3.21.0",
            "google-cloud-secret-manager>=2.20.0",
            "google-cloud-logging>=3.10.0",
            "opentelemetry-api>=1.25.0",
            "opentelemetry-sdk>=1.25.0",
            "opentelemetry-exporter-gcp-trace>=1.6.0",
            "pydantic>=2.7.0",
        ],
        extra_packages=["./app"],
    )

    resource_name = str(remote_agent.resource_name)
    log_structured_event(
        "AGENT_RUNTIME_DEPLOYED",
        f"Deployed ADK agent to Vertex AI Agent Runtime: {resource_name}",
        metadata={"resource_name": resource_name},
    )
    return resource_name


if __name__ == "__main__":
    deployed_resource = deploy_to_agent_runtime()
    print(f"AGENT_ENGINE_RESOURCE_ID={deployed_resource}")
