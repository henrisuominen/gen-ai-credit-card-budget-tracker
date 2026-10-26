variable "project_id" {
  description = "Google Cloud Project ID where the Budget Tracker resources are provisioned."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Vertex AI Agent Runtime and storage."
  type        = string
  default     = "us-central1"
}

variable "document_ai_location" {
  description = "Multi-region location for Google Cloud Document AI processor ('us' or 'eu')."
  type        = string
  default     = "us"
}

variable "config_secret_id" {
  description = "Secret Manager secret identifier storing runtime resource IDs."
  type        = string
  default     = "budget-tracker-runtime-config"
}

variable "gemini_enterprise_app_id" {
  description = "Optional Gemini Enterprise (Discovery Engine) Application ID for agent registration."
  type        = string
  default     = ""
}

variable "model_armor_template_id" {
  description = "Optional Google Cloud Model Armor template resource ID for prompt sanitization."
  type        = string
  default     = ""
}
