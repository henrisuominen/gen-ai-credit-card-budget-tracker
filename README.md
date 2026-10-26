# GenAI Credit Card Budget Tracker

A conversational multi-agent personal finance and credit card statement analyzer built with **Google Agent Development Kit (`google-adk`)**, **Terraform**, and the **Gemini Enterprise Agent Platform**.

## Key Features

- **Zero-Raw-PDF Pre-Ingestion PII Gatekeeper**: Uploaded PDF credit card statements are intercepted in memory before reaching the session store or LLM, extracted via **Google Cloud Document AI** (`process_document`), and anonymized via **Cloud Sensitive Data Protection (Cloud DLP)** (`deidentify_content`) so the agent only ever sees PII-cleaned text.
- **Strategic Model Routing**:
  - `gemini-3.1-flash-lite`: Low-cost, high-throughput batch transaction classification and merchant normalization (e.g., `UBER LLC #12NYC` $\rightarrow$ `Transportation`, `UBER`) plus session history compaction.
  - `gemini-3.5-flash`: Interactive Coordinator routing (`budget_coordinator_agent`).
  - `gemini-3.1-pro-preview`: Deep spending synthesis and action planning (`spending_synthesis_advisor_agent`).
- **Rule-Based Monthly Subscription Detection**: Deterministic Python rule engine identifying recurring monthly transactions (same normalized merchant and amount) and calculating annualized savings.
- **Enterprise Guardrails & Human-in-the-Loop**: Model Armor prompt screening, output PII scrubbing, and explicit human confirmation (`require_confirmation=True`) before flagging subscriptions for cancellation.
- **Observability & Tracing**: Structured JSON logging (`TOOL_INTENT` vs. `TOOL_OUTCOME`) and OpenTelemetry distributed tracing with Cloud Trace.

---

## Local Usage (Bound Strictly to `127.0.0.1` Loopback)

The interactive UI is designed to run locally on your machine over `127.0.0.1` so it is never exposed to the public internet or external networks.

### 1. Authenticate & Configure
```bash
gcloud auth application-default login
cp .env.example .env
# Edit .env and set GOOGLE_CLOUD_PROJECT to your GCP project ID
```

### 2. Launch the Local Browser UI (`adk web`)
```bash
adk web --host 127.0.0.1 --port 8000
```
- Open **`http://127.0.0.1:8000`** in your browser and select the `app` agent.
- Click the **Attachment / Paperclip icon** in the chat bar to upload a credit card statement PDF. The Pre-Ingestion PII Gatekeeper will extract and scrub PII in memory before the agent processes your transactions.

### 3. Alternative Local CLI
```bash
adk run app
# or
python3 -m app.agents
```

---

## Provisioning Infrastructure with Terraform

All required GCP APIs, the **Document AI OCR Processor**, the **Cloud SDP De-identification Template**, **Google Cloud Secret Manager** runtime config, and least-privilege Service Account IAM bindings are provisioned via Terraform:

```bash
cd deployment/terraform
terraform init
terraform apply -var="project_id=YOUR_GCP_PROJECT_ID"
```

---

## Deploying & Registering with Gemini Enterprise Agent Platform

### 1. Deploy to Vertex AI Agent Runtime (`AdkApp`)
```bash
python3 deployment/deploy_agent_engine.py
# Or using the ADK CLI:
adk deploy agent_engine --project="$GOOGLE_CLOUD_PROJECT" --region="us-central1" --staging_bucket="$AGENT_STAGING_BUCKET" app
```

### 2. Register with Gemini Enterprise (`agents-cli` / Discovery Engine API)
```bash
chmod +x deployment/register_gemini_enterprise.sh
./deployment/register_gemini_enterprise.sh
```

---

## Running Tests & Automated Evaluation Suite

```bash
# Run unit tests (PII redaction, Pre-Ingestion Gatekeeper, schemas, recurring rules, HITL hooks)
pytest

# Run the ADK automated evaluation suite against the golden dataset
adk eval app tests/eval/budget_tracker.evalset.json --config_file_path=tests/eval/test_config.json
```
