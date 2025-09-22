"""Multi-agent debate orchestration utilities."""

from __future__ import annotations

import asyncio
import json
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from ..memory import AgentMemory
from ..core.llm_client import LLMClient
from ..core.websocket_manager import WebSocketConnectionManager


@dataclass
class DebaterConfig:
    """Configuration for an individual debating agent."""

    role: str
    style: str
    model: Optional[str] = None
    temperature: float = 0.6


@dataclass
class DebateConfig:
    """Run-level configuration for a debate."""

    num_agents: int = 3
    num_rounds: int = 2
    groups: int = 1
    rubric: str = ""
    max_tokens: int = 800
    temperature: float = 0.6


class DebateManager:
    """Coordinates multi-agent debates with optional grouping (GroupDebate style)."""

    def __init__(
        self,
        llm_client: LLMClient,
        memory: Optional[AgentMemory] = None,
        websocket_manager: Optional[WebSocketConnectionManager] = None,
    ) -> None:
        self._llm = llm_client
        self._memory = memory
        self._ws = websocket_manager

    async def run(
        self,
        topic: str,
        context: Dict[str, Any],
        cfg: DebateConfig,
        debaters: List[DebaterConfig],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the debate and return structured verdict + transcripts."""

        start = time.time()
        groups = self._build_groups(debaters, cfg.groups)
        transcripts: List[Dict[str, Any]] = []
        interim_summaries: Optional[List[Dict[str, Any]]] = None

        memory_context = await self._memory_context(topic, context)

        for round_idx in range(cfg.num_rounds):
            round_record = {"round": round_idx + 1, "groups": []}
            for group_idx, group in enumerate(groups):
                messages = await self._run_group_round(
                    topic,
                    context,
                    memory_context,
                    group,
                    round_idx,
                    group_idx,
                    cfg,
                    interim_summaries,
                    session_id=session_id,
                )
                round_record["groups"].append({"group": group_idx + 1, "messages": messages})
            transcripts.append(round_record)
            interim_summaries = await self._summarise_groups(topic, context, round_record, cfg)

        verdict = await self._judge(topic, context, transcripts, cfg)
        elapsed = time.time() - start

        if self._memory:
            await self._memory.save_debate(topic, transcripts, verdict)

        result = {
            "topic": topic,
            "verdict": verdict,
            "transcripts": transcripts,
            "elapsed_seconds": elapsed,
        }
        await self._emit(session_id, "debate_complete", result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_groups(debaters: List[DebaterConfig], group_count: int) -> List[List[DebaterConfig]]:
        if group_count <= 1:
            return [debaters]
        return [debaters[i::group_count] for i in range(group_count)]

    async def _run_group_round(
        self,
        topic: str,
        context: Dict[str, Any],
        memory_context: Dict[str, List[Dict[str, Any]]],
        group: List[DebaterConfig],
        round_idx: int,
        group_idx: int,
        cfg: DebateConfig,
        interim: Optional[List[Dict[str, Any]]],
        session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for debater in group:
            prompt = self._compose_debater_prompt(
                topic=topic,
                context=context,
                memory_context=memory_context,
                debater=debater,
                round_idx=round_idx,
                group_idx=group_idx,
                interim=interim,
            )
            reply = await self._llm.generate(
                prompt,
                temperature=debater.temperature,
                max_tokens=cfg.max_tokens,
            )
            message = {
                "role": debater.role,
                "style": debater.style,
                "content": reply.strip(),
            }
            messages.append(message)
            await self._emit(session_id, "debate_message", {
                "round": round_idx + 1,
                "group": group_idx + 1,
                "message": message,
            })
        return messages

    async def _summarise_groups(
        self,
        topic: str,
        context: Dict[str, Any],
        round_record: Dict[str, Any],
        cfg: DebateConfig,
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for group_entry in round_record["groups"]:
            group_messages = group_entry["messages"]
            transcript = "\n".join(
                f"[{msg['role']}] {msg['content']}" for msg in group_messages
            )
            prompt = textwrap.dedent(
                f"""
                You are a debate summariser.
                Topic: {topic}
                Context: {json.dumps(context)}
                Transcript:
                {transcript}

                Provide a concise summary (<=80 words) capturing key agreements, disagreements, and evidence.
                Respond with plain text.
                """
            )
            summary_text = await self._llm.generate(prompt, temperature=0.3, max_tokens=200)
            summaries.append({
                "group": group_entry["group"],
                "summary": summary_text.strip(),
            })
        return summaries

    async def _judge(
        self,
        topic: str,
        context: Dict[str, Any],
        transcripts: List[Dict[str, Any]],
        cfg: DebateConfig,
    ) -> Dict[str, Any]:
        rubric = cfg.rubric or DEFAULT_RUBRIC
        transcript_text = json.dumps(transcripts, ensure_ascii=False)
        prompt = textwrap.dedent(
            f"""
            You are the final judge in a multi-agent debate.
            Topic: {topic}
            Context: {json.dumps(context)}
            Debate transcripts: {transcript_text}

            Using the rubric below, score each dimension between 0 and 1 (inclusive) and provide an overall recommendation.
            Rubric:
            {rubric}

            Respond in JSON with the schema:
            {{
              "summary": str,
              "scores": {{"factuality": float, "feasibility": float, "safety": float, "expected_gain": float}},
              "recommendation": str,
              "supporting_points": [str]
            }}
            """
        )
        raw = await self._llm.generate(prompt, temperature=0.2, max_tokens=400)
        verdict = self._safe_json_parse(raw)
        if not verdict:
            verdict = {
                "summary": raw.strip(),
                "scores": {},
                "recommendation": "undecided",
                "supporting_points": [],
            }
        await self._emit(None, "debate_verdict", verdict)
        return verdict

    async def _memory_context(self, topic: str, context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        if not self._memory:
            return {}
        query_terms = topic
        if context:
            query_terms += " " + " ".join(str(v) for v in context.values() if isinstance(v, (str, int)))
        return await self._memory.retrieve_context(query_terms, k=5)

    @staticmethod
    def _compose_debater_prompt(
        *,
        topic: str,
        context: Dict[str, Any],
        memory_context: Dict[str, List[Dict[str, Any]]],
        debater: DebaterConfig,
        round_idx: int,
        group_idx: int,
        interim: Optional[List[Dict[str, Any]]],
    ) -> str:
        mem_lines: List[str] = []
        for key, memories in (memory_context or {}).items():
            if not memories:
                continue
            mem_lines.append(f"{key.upper()} MEMORY:")
            for mem in memories[:3]:
                content = mem.get("content", "").strip()
                importance = mem.get("importance_score")
                if importance is not None:
                    mem_lines.append(f"- ({importance:.2f}) {content}")
                else:
                    mem_lines.append(f"- {content}")
        memory_text = "\n".join(mem_lines) if mem_lines else "(no retrieved memory)"

        interim_text = ""
        if interim:
            parts = [f"Group {entry['group']}: {entry['summary']}" for entry in interim]
            interim_text = "\n".join(parts)

        return textwrap.dedent(
            f"""
            You are the {debater.role} in a multi-agent debate (round {round_idx + 1}, group {group_idx + 1}).
            Debate topic: {topic}
            Context: {json.dumps(context, ensure_ascii=False)}

            Retrieved memory:
            {memory_text}

            Interim summaries from other groups:
            {interim_text or '(none yet)'}

            Debate style directive: {debater.style}.
            Provide a reasoned argument or critique grounded in the evidence above.
            Make explicit references to artifacts, data, or prior experience where possible.
            Avoid speculation without signalling uncertainty.
            End with a concise action recommendation or key takeaway (<=20 words).
            """
        ).strip()

    @staticmethod
    def _safe_json_parse(raw: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # try to extract JSON substring
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    async def _emit(self, session_id: Optional[str], event: str, payload: Dict[str, Any]) -> None:
        if not self._ws or not session_id:
            return
        try:
            await self._ws.send_to_research_session(session_id, {
                "type": event,
                "payload": payload,
            })
        except Exception:  # pragma: no cover - telemetry best-effort
            pass


DEFAULT_RUBRIC = textwrap.dedent(
    """
    1. Factuality: Are the claims supported by cited data, calculations, or artifacts?
    2. Feasibility: Are resources, time, and dependencies realistic? Are risks surfaced?
    3. Safety & Compliance: Does the proposal avoid dangerous actions, policy violations, or data leaks?
    4. Expected Gain: Will adopting the recommendation significantly improve research outcomes vs. cost/time?
    Weight scores equally. Penalise hand-wavy or uncited assertions.
    """
)
