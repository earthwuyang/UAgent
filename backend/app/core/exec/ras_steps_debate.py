"""RAS step handler for multi-agent debate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ...debate import DebateManager, DebateConfig, DebaterConfig


class RASDebateStepHandler:
    """Callable plugin that executes a debate via DebateManager."""

    def __init__(self, debate_manager: DebateManager | None) -> None:
        self.debate_manager = debate_manager

    async def __call__(self, ctx: Dict[str, Any], step_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.debate_manager:
            raise RuntimeError("DebateManager unavailable")

        payload = step_payload.get("with") or {}
        topic = payload.get("topic", "General inquiry")
        context = payload.get("context") or {}
        raw_cfg = payload.get("config") or {}
        debaters_raw = payload.get("debaters") or []

        cfg = DebateConfig(
            num_agents=int(raw_cfg.get("num_agents", 3)),
            num_rounds=int(raw_cfg.get("num_rounds", 2)),
            groups=int(raw_cfg.get("groups", 1)),
            rubric=str(raw_cfg.get("rubric", "")),
            max_tokens=int(raw_cfg.get("max_tokens", 800)),
            temperature=float(raw_cfg.get("temperature", 0.6)),
        )
        debaters = [
            DebaterConfig(
                role=str(item.get("role", f"debater_{idx}")),
                style=str(item.get("style", "concise")),
                model=item.get("model"),
                temperature=float(item.get("temperature", cfg.temperature)),
            )
            for idx, item in enumerate(debaters_raw)
        ] or [
            DebaterConfig(role="proposer", style="thorough"),
            DebaterConfig(role="critic", style="concise"),
            DebaterConfig(role="safety", style="conservative"),
        ]

        session_id = step_payload.get("session_id")
        result = await self.debate_manager.run(topic, context, cfg, debaters, session_id=session_id)
        return result


def create_debate_plugins(debate_manager: DebateManager | None) -> Dict[str, Any]:
    return {
        "multi_agent_debate": RASDebateStepHandler(debate_manager),
    }
