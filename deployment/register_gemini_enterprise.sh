#!/usr/bin/env bash
# Register the deployed Vertex AI Agent Runtime (AdkApp) instance with Gemini Enterprise
# via the Agents CLI (`agents-cli`) or Discovery Engine REST API.

set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
APP_ID="${GEMINI_ENTERPRISE_APP_ID:?Set GEMINI_ENTERPRISE_APP_ID}"
LOCATION="${GEMINI_ENTERPRISE_LOCATION:-global}"
AGENT_ENGINE_RESOURCE="${AGENT_ENGINE_RESOURCE_ID:?Set AGENT_ENGINE_RESOURCE_ID}"

echo "Registering Agent Runtime resource with Gemini Enterprise Agent Platform..."

if command -v agents-cli >/dev/null 2>&1; then
  agents-cli publish gemini-enterprise \
    --project "${PROJECT_ID}" \
    --location "${LOCATION}" \
    --app-id "${APP_ID}" \
    --agent-engine "${AGENT_ENGINE_RESOURCE}" \
    --display-name "Credit Card Budget Tracker" \
    --description "Anonymizes credit card statements, classifies spend, detects monthly subscriptions, and suggests savings."
else
  API_HOST="discoveryengine.googleapis.com"
  if [[ "${LOCATION}" != "global" ]]; then
    API_HOST="${LOCATION}-discoveryengine.googleapis.com"
  fi

  curl -sS -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    -H "X-Goog-User-Project: ${PROJECT_ID}" \
    "https://${API_HOST}/v1alpha/projects/${PROJECT_ID}/locations/${LOCATION}/collections/default_collection/engines/${APP_ID}/assistants/default_assistant/agents" \
    -d "{
      \"displayName\": \"Credit Card Budget Tracker\",
      \"description\": \"Anonymizes credit card statements, classifies purchases, detects monthly subscriptions, and recommends spend reductions.\",
      \"adkAgentDefinition\": {
        \"toolSettings\": {
          \"toolDescription\": \"Use this agent to analyze credit card statements, audit recurring monthly subscriptions, and create budget reduction plans.\"
        },
        \"provisionedReasoningEngine\": {
          \"reasoningEngine\": \"${AGENT_ENGINE_RESOURCE}\"
        }
      }
    }"
fi

echo "Registration complete."
