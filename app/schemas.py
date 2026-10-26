"""Strict Pydantic v2 JSON schemas for tool inputs, outputs, and LLM structured generation.

Enforces strict types, value constraints, and forbidden extra fields across all agent boundaries.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SpendingCategory(str, Enum):
    """Broad spending taxonomy for classifying credit card transactions."""

    TRANSPORTATION = "Transportation"
    DINING_AND_GROCERIES = "Dining & Groceries"
    SHOPPING = "Shopping"
    ENTERTAINMENT = "Entertainment"
    SUBSCRIPTIONS_AND_UTILITIES = "Subscriptions & Utilities"
    TRAVEL = "Travel"
    HEALTH_AND_WELLNESS = "Health & Wellness"
    FEES_AND_INTEREST = "Fees & Interest"
    OTHER = "Other"


class MonthlyCostClassification(str, Enum):
    """Two-tier qualification for monthly recurring costs."""

    SAME_DAY_MONTHLY = "Monthly cost happening on the same day"
    POTENTIAL_MONTHLY_DIFFERING_DAYS = "Potential monthly costs (happening on differing days)"


class ClassifiedTransaction(BaseModel):
    """A single normalized and categorized credit card transaction."""

    model_config = ConfigDict(extra="forbid")

    date: str = Field(
        ...,
        description="Transaction date in ISO-8601 format (YYYY-MM-DD) or MM/DD format.",
        min_length=4,
    )
    raw_description: str = Field(
        ...,
        description="Original anonymized merchant line from the statement (e.g. 'UBER LLC #12NYC').",
        min_length=1,
    )
    merchant: str = Field(
        ...,
        description="Canonical normalized merchant name in uppercase (e.g. 'UBER').",
        min_length=1,
    )
    category: SpendingCategory = Field(
        ...,
        description="Broad spending category assigned to the purchase.",
    )
    amount: float = Field(
        ...,
        description="Transaction amount in statement currency (positive number).",
        ge=0.0,
    )


class TransactionBatchResponse(BaseModel):
    """Structured output schema returned by gemini-2.5-flash-lite during batch classification."""

    model_config = ConfigDict(extra="forbid")

    transactions: list[ClassifiedTransaction] = Field(
        default_factory=list,
        description="List of all classified transactions extracted from the anonymized statement.",
    )


class RecurringSubscription(BaseModel):
    """Rule-based detection record for a monthly recurring charge or subscription."""

    model_config = ConfigDict(extra="forbid")

    merchant: str = Field(
        ...,
        description="Normalized canonical merchant name (e.g. 'NETFLIX', 'SPOTIFY').",
    )
    category: SpendingCategory = Field(
        ...,
        description="Assigned spending category for this recurring merchant.",
    )
    monthly_cost_type: MonthlyCostClassification = Field(
        default=MonthlyCostClassification.SAME_DAY_MONTHLY,
        description=(
            "Qualification tier: 'Monthly cost happening on the same day' vs. "
            "'Potential monthly costs (happening on differing days)'."
        ),
    )
    billing_days_of_month: list[int] = Field(
        default_factory=list,
        description="Sorted day(s) of the month on which the transaction occurred (1-31).",
    )
    monthly_amount: float = Field(
        ...,
        description="Recurring monthly transaction amount.",
        ge=0.0,
    )
    occurrence_count: int = Field(
        ...,
        description="Number of occurrences matching the same merchant and amount.",
        ge=1,
    )
    annualized_cost: float = Field(
        ...,
        description="Projected 12-month cost (monthly_amount * 12).",
        ge=0.0,
    )
    detection_reason: str = Field(
        ...,
        description="Rule-based explanation for why this transaction was flagged as monthly recurring.",
    )


class StatementAuditSummary(BaseModel):
    """Complete aggregated summary of an anonymized credit card statement."""

    model_config = ConfigDict(extra="forbid")

    total_spend: float = Field(
        ...,
        description="Sum of all classified transaction amounts.",
        ge=0.0,
    )
    transaction_count: int = Field(
        ...,
        description="Total number of transactions processed.",
        ge=0,
    )
    category_totals: dict[str, float] = Field(
        default_factory=dict,
        description="Total spend aggregated by broad spending category.",
    )
    top_category: str = Field(
        default="None",
        description="The spending category with the highest total spend.",
    )
    transactions: list[ClassifiedTransaction] = Field(
        default_factory=list,
        description="List of individual classified transactions.",
    )
    recurring_subscriptions: list[RecurringSubscription] = Field(
        default_factory=list,
        description="All detected monthly and potential monthly recurring charges.",
    )
    same_day_monthly_costs: list[RecurringSubscription] = Field(
        default_factory=list,
        description="Category 1: Monthly costs happening on the exact same day of the month.",
    )
    potential_monthly_costs_differing_days: list[RecurringSubscription] = Field(
        default_factory=list,
        description="Category 2: Potential monthly costs that happen on differing days.",
    )
    potential_annual_subscription_savings: float = Field(
        default=0.0,
        description="Total annualized cost of all identified recurring subscriptions.",
        ge=0.0,
    )


class GuidedToolResult(BaseModel):
    """Standardized tool response envelope with explicit recovery instructions on error."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error", "pending_confirmation"] = Field(
        ...,
        description="Execution status of the tool call.",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured output payload when status is 'success'.",
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error identifier when status is 'error'.",
    )
    message: str | None = Field(
        default=None,
        description="Human-readable summary of the outcome or error.",
    )
    recovery_instruction: str | None = Field(
        default=None,
        description="Actionable instructions telling the LLM how to recover from an error.",
    )
