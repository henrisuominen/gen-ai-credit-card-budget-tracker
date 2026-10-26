output "document_ai_processor_id" {
  description = "Resource name of the provisioned Document AI OCR processor."
  value       = google_document_ai_processor.statement_ocr.name
}

output "sdp_deidentify_template_id" {
  description = "Full resource ID of the Cloud Sensitive Data Protection de-identify template."
  value       = google_data_loss_prevention_deidentify_template.statement_pii_scrubber.id
}

output "runtime_config_secret_id" {
  description = "Secret Manager secret ID containing the runtime configuration JSON."
  value       = google_secret_manager_secret.runtime_config.secret_id
}

output "agent_staging_bucket" {
  description = "Private GCS staging bucket URI for Vertex AI Agent Runtime."
  value       = "gs://${google_storage_bucket.agent_staging.name}"
}

output "agent_service_account_email" {
  description = "Email of the least-privilege service account for the agent."
  value       = google_service_account.budget_agent_sa.email
}
