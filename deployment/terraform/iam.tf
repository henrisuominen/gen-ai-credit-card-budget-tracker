# Least-Privilege Service Account & Private IAM Bindings
# Safe for isolated sandbox environments:
# - Zero exported Service Account keys (uses ADC / Workload Identity)
# - Zero public (`allUsers`) or broad domain (`domain:`) bindings

resource "google_service_account" "budget_agent_sa" {
  project      = var.project_id
  account_id   = "budget-tracker-agent-sa"
  display_name = "Credit Card Budget Tracker Agent Service Account"
  description  = "Least-privilege identity for Document AI, Cloud SDP, Secret Manager, and Vertex AI Agent Runtime."

  depends_on = [google_project_service.enabled_apis]
}

locals {
  agent_sa_roles = [
    "roles/aiplatform.user",
    "roles/documentai.apiUser",
    "roles/dlp.user",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectAdmin",
    "roles/logging.logWriter",
    "roles/cloudtrace.agent",
  ]
}

resource "google_project_iam_member" "agent_sa_bindings" {
  for_each = toset(local.agent_sa_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.budget_agent_sa.email}"
}
