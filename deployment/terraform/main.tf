terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_apis = [
    "aiplatform.googleapis.com",
    "discoveryengine.googleapis.com",
    "documentai.googleapis.com",
    "dlp.googleapis.com",
    "modelarmor.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
  ]
}

# 1. Enable Required Google Cloud APIs
resource "google_project_service" "enabled_apis" {
  for_each           = toset(local.required_apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# 2. Private Staging Bucket for Vertex AI Agent Runtime (Public Access Explicitly Enforced Off)
resource "google_storage_bucket" "agent_staging" {
  name                        = "${var.project_id}-budget-agent-staging"
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  depends_on = [google_project_service.enabled_apis]
}

# 3. Google Cloud Document AI OCR Processor for PDF Statement Reading
resource "google_document_ai_processor" "statement_ocr" {
  project      = var.project_id
  location     = var.document_ai_location
  display_name = "credit-card-statement-ocr-processor"
  type         = "OCR_PROCESSOR"

  depends_on = [google_project_service.enabled_apis]
}

# 4. Google Cloud Sensitive Data Protection (DLP) De-identification Template
resource "google_data_loss_prevention_deidentify_template" "statement_pii_scrubber" {
  parent       = "projects/${var.project_id}/locations/global"
  display_name = "Credit Card Statement PII Anonymizer"
  description  = "Replaces credit card numbers, account numbers, names, addresses, phones, and emails with infoType tokens."

  deidentify_config {
    info_type_transformations {
      transformations {
        info_types {
          name = "PERSON_NAME"
        }
        info_types {
          name = "CREDIT_CARD_NUMBER"
        }
        info_types {
          name = "FINANCIAL_ACCOUNT_NUMBER"
        }
        info_types {
          name = "IBAN_CODE"
        }
        info_types {
          name = "US_SOCIAL_SECURITY_NUMBER"
        }
        info_types {
          name = "STREET_ADDRESS"
        }
        info_types {
          name = "PHONE_NUMBER"
        }
        info_types {
          name = "EMAIL_ADDRESS"
        }

        primitive_transformation {
          replace_with_info_type_config {}
        }
      }
    }
  }

  depends_on = [google_project_service.enabled_apis]
}

# 5. Google Cloud Secret Manager for Secure Runtime Configuration Injection (Zero Hardcoded Keys)
resource "google_secret_manager_secret" "runtime_config" {
  project   = var.project_id
  secret_id = var.config_secret_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.enabled_apis]
}

resource "google_secret_manager_secret_version" "runtime_config_v1" {
  secret = google_secret_manager_secret.runtime_config.id

  secret_data = jsonencode({
    document_ai_location       = var.document_ai_location
    document_ai_processor_id   = google_document_ai_processor.statement_ocr.name
    sdp_deidentify_template_id = google_data_loss_prevention_deidentify_template.statement_pii_scrubber.id
    model_armor_template_id    = var.model_armor_template_id
    agent_staging_bucket       = "gs://${google_storage_bucket.agent_staging.name}"
    gemini_enterprise_app_id   = var.gemini_enterprise_app_id
  })
}
