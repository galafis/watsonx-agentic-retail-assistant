"""Audit logger for tracking every agent action, tool call, and response."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class AuditLogger:
    """Logs all agentic interactions for governance, auditing, and compliance.

    Tracks agent actions, tool invocations, responses, and quality metrics.
    Integrates with IBM Watsonx Governance for enterprise deployments.
    For local development, logs to structured JSON files.
    """

    def __init__(self, log_path: str | None = None) -> None:
        gov_config = settings.governance
        self.enabled: bool = gov_config.get("enabled", True)
        self.log_prompts: bool = gov_config.get("log_prompts", True)
        self.log_responses: bool = gov_config.get("log_responses", True)
        self.log_tool_calls: bool = gov_config.get("log_tool_calls", True)

        self.log_path = Path(log_path or gov_config.get("log_path", "./logs/governance"))
        if self.enabled:
            self.log_path.mkdir(parents=True, exist_ok=True)

        self._entries: list[dict[str, Any]] = []
        self._interaction_count: int = 0

    def log_agent_action(
        self,
        agent_name: str,
        action_type: str,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Log an agent action to the audit trail.

        Args:
            agent_name: Name of the agent performing the action.
            action_type: Type of action (e.g., 'intent_classification', 'tool_call', 'response').
            input_data: Input data for the action.
            output_data: Output/result of the action.
            metadata: Additional metadata.

        Returns:
            The log entry ID, or None if logging is disabled.
        """
        if not self.enabled:
            return None

        self._interaction_count += 1
        timestamp = datetime.now(timezone.utc).isoformat()
        entry_id = (
            f"audit_{timestamp.replace(':', '-').replace('.', '-')}_{self._interaction_count}"
        )

        entry: dict[str, Any] = {
            "id": entry_id,
            "timestamp": timestamp,
            "agent": agent_name,
            "action_type": action_type,
        }

        if self.log_prompts and input_data:
            entry["input"] = input_data

        if self.log_responses and output_data:
            entry["output"] = output_data

        entry["metadata"] = metadata or {}
        self._entries.append(entry)

        # Write to daily log file
        self._write_entry(entry)

        logger.info(
            "audit_logged",
            entry_id=entry_id,
            agent=agent_name,
            action=action_type,
        )
        return entry_id

    def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: dict[str, Any],
        duration_ms: float | None = None,
    ) -> str | None:
        """Log a tool invocation by an agent.

        Args:
            agent_name: The agent that called the tool.
            tool_name: Name of the tool invoked.
            tool_input: Input parameters to the tool.
            tool_output: Output from the tool.
            duration_ms: Execution time in milliseconds.

        Returns:
            The log entry ID.
        """
        return self.log_agent_action(
            agent_name=agent_name,
            action_type="tool_call",
            input_data={"tool": tool_name, "parameters": tool_input},
            output_data=tool_output,
            metadata={"duration_ms": duration_ms} if duration_ms else None,
        )

    def log_conversation_turn(
        self,
        user_message: str,
        assistant_response: str,
        agent_used: str,
        tools_called: list[str] | None = None,
        quality_score: float | None = None,
    ) -> str | None:
        """Log a complete conversation turn for audit purposes.

        Args:
            user_message: The user's input message.
            assistant_response: The assistant's response.
            agent_used: Which agent handled the query.
            tools_called: List of tools invoked.
            quality_score: Response quality assessment score.

        Returns:
            The log entry ID.
        """
        return self.log_agent_action(
            agent_name="orchestrator",
            action_type="conversation_turn",
            input_data={"user_message": user_message},
            output_data={"response": assistant_response},
            metadata={
                "agent_used": agent_used,
                "tools_called": tools_called or [],
                "quality_score": quality_score,
            },
        )

    def get_recent_entries(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve recent audit log entries from memory."""
        return self._entries[-limit:]

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregated audit metrics."""
        if not self._entries:
            return {
                "total_interactions": 0,
                "agents_used": {},
                "tools_invoked": 0,
                "avg_quality_score": 0.0,
            }

        agents_used: dict[str, int] = {}
        tools_invoked = 0
        quality_scores: list[float] = []

        for entry in self._entries:
            agent = entry.get("agent", "unknown")
            agents_used[agent] = agents_used.get(agent, 0) + 1

            if entry.get("action_type") == "tool_call":
                tools_invoked += 1

            qs = entry.get("metadata", {}).get("quality_score")
            if qs is not None:
                quality_scores.append(qs)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            "total_interactions": self._interaction_count,
            "agents_used": agents_used,
            "tools_invoked": tools_invoked,
            "avg_quality_score": round(avg_quality, 3),
        }

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a log entry to the daily log file."""
        try:
            log_file = self.log_path / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error("audit_write_failed", error=str(e))


# Module-level singleton
audit_logger = AuditLogger()
