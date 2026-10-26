"""Automated Evaluation & Multi-Agent Architecture Verification Suite.

Validates the evaluation dataset (`tests/eval/budget_tracker.evalset.json`), multi-agent routing,
context compaction config, and observability plugins.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agents import (
    app,
    budget_coordinator_agent,
    root_agent,
    spending_synthesis_advisor_agent,
    statement_preprocessing_pipeline,
)
from app.config import settings


def _model_name(agent_model: object) -> str:
    return getattr(agent_model, "model", str(agent_model))


def test_multi_agent_hierarchy_and_strategic_model_routing() -> None:
    """Verify Coordinator, Sequential Pipeline, and Synthesis Advisor use strategic model tiers."""
    assert root_agent.name == "budget_coordinator_agent"
    assert _model_name(budget_coordinator_agent.model) == settings.model_coordinator
    assert _model_name(spending_synthesis_advisor_agent.model) == settings.model_synthesis_pro

    pipeline_sub_names = [a.name for a in statement_preprocessing_pipeline.sub_agents]
    assert pipeline_sub_names == [
        "ingestion_anonymizer_agent",
        "transaction_classifier_agent",
        "subscription_audit_agent",
    ]
    for fast_agent in statement_preprocessing_pipeline.sub_agents:
        assert _model_name(fast_agent.model) == settings.model_fast_classifier


def test_app_compaction_and_plugins_registered() -> None:
    """Verify ADK App has EventsCompactionConfig, EnterpriseGuardrailPlugin, and ObservabilityPlugin."""
    assert app.events_compaction_config is not None
    assert app.events_compaction_config.compaction_interval == 5
    plugin_names = [p.name for p in app.plugins]
    assert "enterprise_guardrail_plugin" in plugin_names
    assert "observability_intent_outcome_plugin" in plugin_names


def test_golden_eval_set_and_thresholds_are_valid() -> None:
    """Validate golden evaluation dataset and threshold files for `adk eval`."""
    eval_dir = Path(__file__).parent / "eval"
    evalset_path = eval_dir / "budget_tracker.evalset.json"
    config_path = eval_dir / "test_config.json"

    assert evalset_path.is_file()
    assert config_path.is_file()

    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    assert evalset["eval_set_id"] == "credit_card_budget_tracker_golden_eval_set"
    assert len(evalset["eval_cases"]) >= 1

    first_case = evalset["eval_cases"][0]
    tool_uses = first_case["conversation"][0]["intermediate_data"]["tool_uses"]
    tool_names = [t["name"] for t in tool_uses]
    assert "classify_statement_transactions_batch" in tool_names
    assert "detect_recurring_monthly_subscriptions" in tool_names

    eval_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "tool_trajectory_avg_score" in eval_config["criteria"]
    assert "response_match_score" in eval_config["criteria"]
