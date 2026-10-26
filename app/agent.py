"""ADK standard `agent.py` entrypoint re-exporting `app` and `root_agent` from `app.agents`."""

from app.agents import app, root_agent

__all__ = ["app", "root_agent"]
