"""Integration bridge for invoking RepoMaster workflows from UAgent."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RepoMasterBridgeResult:
    """Structured result returned by RepoMasterBridge."""

    query: str
    final_answer: str
    report_markdown: str
    repositories: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RepoMasterBridge:
    """Thin wrapper that embeds RepoMaster inside the UAgent runtime."""

    def __init__(self, vendor_root: Optional[Path] = None) -> None:
        if vendor_root is not None:
            resolved_root = Path(vendor_root)
        else:
            file_path = Path(__file__).resolve()
            backend_dir = file_path.parents[2]
            app_dir = file_path.parents[1]
            candidates = [
                backend_dir / "vendor" / "repomaster",
                app_dir / "vendor" / "repomaster",
            ]
            resolved_root = next((path for path in candidates if path.exists()), candidates[0])

        self.vendor_root = resolved_root
        self._ensure_import_path()

        # Lazy imports so the path logic above occurs before import resolution.
        from configs.mode_config import ModeConfigManager  # type: ignore
        from src.core.agent_scheduler import RepoMasterAgent  # type: ignore
        from src.utils.utils_config import AppConfig  # type: ignore

        self._mode_manager_cls = ModeConfigManager
        self._agent_cls = RepoMasterAgent
        self._app_config = AppConfig.get_instance()

    def _ensure_import_path(self) -> None:
        """Ensure the RepoMaster vendor directory is on sys.path."""

        if not self.vendor_root.exists():
            raise FileNotFoundError(
                f"RepoMaster vendor directory not found at {self.vendor_root}."
            )

        vendor_path = str(self.vendor_root)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

    async def run_task(
        self,
        session_id: str,
        query: str,
        workspace: Path,
        *,
        repository_hint: Optional[str] = None,
        input_data: Optional[List[Dict[str, Any]]] = None,
        progress_handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        llm_handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> RepoMasterBridgeResult:
        """Execute a RepoMaster workflow within the provided workspace."""

        workspace.mkdir(parents=True, exist_ok=True)

        # Ensure RepoMaster uses DashScope unless caller overrides.
        os.environ.setdefault("DEFAULT_API_PROVIDER", "litellm")

        # Prep RepoMaster configuration.
        manager = self._mode_manager_cls()
        manager.create_config(
            mode="backend",
            backend_mode="repository_agent",
            work_dir=str(workspace),
        )
        llm_config = manager.get_llm_config(api_type="dashscope")
        execution_config = manager.get_execution_config()

        # Ensure RepoMaster's AppConfig session aligns with the UAgent session/workspace.
        self._app_config.create_session(session_id, work_dir=str(workspace))

        loop = asyncio.get_running_loop()
        collected_events: List[Dict[str, Any]] = []
        collected_messages: List[Dict[str, Any]] = []

        def _schedule_handler(
            handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]],
            payload: Dict[str, Any],
        ) -> None:
            if handler is None:
                return
            try:
                future = asyncio.run_coroutine_threadsafe(handler(payload), loop)

                def _log_future_result(fut: asyncio.Future) -> None:
                    try:
                        fut.result()
                    except Exception as exc:  # pragma: no cover - diagnostic logging
                        logger.warning("RepoMaster handler dispatch failed: %s", exc)

                future.add_done_callback(_log_future_result)
            except Exception as exc:  # pragma: no cover - handler best effort
                logger.debug("RepoMaster handler dispatch failed: %s", exc)

        def _progress_callback(event: Dict[str, Any]) -> None:
            event = dict(event)
            event.setdefault("session_id", session_id)
            collected_events.append(event)
            _schedule_handler(progress_handler, event)

        def _llm_callback(message: Dict[str, Any]) -> None:
            message = dict(message)
            message.setdefault("session_id", session_id)
            collected_messages.append(message)
            _schedule_handler(llm_handler, message)

        agent = self._agent_cls(
            llm_config=llm_config,
            code_execution_config=execution_config,
            progress_callback=_progress_callback,
            llm_callback=_llm_callback,
        )

        def _execute() -> str:
            if repository_hint:
                json_payload = input_data
                if input_data and not isinstance(input_data, str):
                    import json

                    json_payload = json.dumps(input_data, ensure_ascii=False)
                return agent.run_repository_agent(
                    task_description=query,
                    repository=repository_hint,
                    input_data=json_payload,
                )
            return agent.solve_task_with_repo(query)

        final_answer: str = await asyncio.to_thread(_execute)

        report_markdown = final_answer if final_answer else ""
        metadata = {
            "progress_events": collected_events,
            "llm_messages": collected_messages,
            "workspace": str(workspace),
            "repository_hint": repository_hint,
        }

        return RepoMasterBridgeResult(
            query=query,
            final_answer=final_answer,
            report_markdown=report_markdown,
            repositories=[],
            metadata=metadata,
        )
