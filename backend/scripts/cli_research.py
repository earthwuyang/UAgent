"""Headless CLI monitor for UAgent research sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import aiohttp


@dataclass
class TreeNode:
    node_id: str
    title: str = ""
    phase: str = ""
    engine: str = ""
    status: str = "pending"
    progress: Optional[float] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def add_child(self, child_id: str) -> None:
        if child_id not in self.children:
            self.children.append(child_id)


class ResearchTree:
    def __init__(self, session_id: str, primary_engine: str) -> None:
        self.session_id = session_id
        self.primary_engine = primary_engine
        self.nodes: Dict[str, TreeNode] = {}
        self.completed = False
        self.last_render = 0.0
        self.spinner = cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        self.latest_events: List[Dict[str, Any]] = []
        self.latest_llm: List[str] = []

    def ensure_node(self, node_id: str) -> TreeNode:
        if node_id not in self.nodes:
            self.nodes[node_id] = TreeNode(node_id=node_id)
        return self.nodes[node_id]

    def apply_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("event_type", "").lower()
        data = event.get("data", {})
        engine = data.get("engine") or event.get("source") or "unknown"
        metadata = data.get("metadata") or {}
        message = event.get("message") or metadata.get("title") or event_type
        node_id = metadata.get("node_id")

        if not node_id:
            # fall back to engine-specific root
            node_id = f"{self.session_id}-{engine}-root"
            metadata.setdefault("node_type", "engine")
            metadata.setdefault("title", f"{engine.replace('_', ' ').title()} Engine")

        node = self.ensure_node(node_id)
        node.title = metadata.get("title", message)
        node.phase = metadata.get("phase", data.get("phase", node.phase))
        node.engine = engine
        node.metadata = {**node.metadata, **metadata}
        node.updated_at = time.time()

        parent_id = metadata.get("parent_id")
        if parent_id:
            parent = self.ensure_node(parent_id)
            node.parent_id = parent_id
            parent.add_child(node_id)
        elif node_id != f"{self.session_id}-{engine}-root":
            # attach to engine root if explicit parent missing
            root_id = f"{self.session_id}-{engine}-root"
            root = self.ensure_node(root_id)
            root.engine = engine
            root.title = f"{engine.replace('_', ' ').title()} Engine"
            node.parent_id = root_id
            root.add_child(node_id)

        progress = event.get("progress_percentage")
        if progress is not None:
            node.progress = progress

        status = metadata.get("status")
        if not status:
            if event_type == "research_error":
                status = "error"
            elif event_type == "research_completed" or (progress is not None and progress >= 100):
                status = "completed"
            else:
                status = "running"
        node.status = status

        if event_type == "research_completed" and engine == self.primary_engine:
            self.completed = True

        # track latest events for CLI display
        self.latest_events.append({
            "timestamp": event.get("timestamp"),
            "message": message,
            "engine": engine,
            "status": status,
        })
        self.latest_events = self.latest_events[-10:]

    def record_llm(self, payload: Dict[str, Any]) -> None:
        content = payload.get("prompt") or payload.get("response") or ""
        if not content:
            return
        prefix = "PROMPT" if payload.get("prompt") else "RESPONSE"
        entry = f"[{prefix}] {content.strip()}"
        self.latest_llm.append(entry)
        self.latest_llm = self.latest_llm[-5:]

    def running_nodes(self) -> List[str]:
        return [node_id for node_id, node in self.nodes.items() if node.status == "running"]

    def root_ids(self) -> List[str]:
        candidates = []
        referenced = {node.parent_id for node in self.nodes.values() if node.parent_id}
        for node_id, node in self.nodes.items():
            if node.parent_id is None or node_id not in referenced:
                candidates.append(node_id)
        return sorted(set(candidates))

    def render(self) -> None:
        now = time.time()
        if now - self.last_render < 0.2:
            return
        self.last_render = now

        spinner_char = next(self.spinner)
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Session: {self.session_id} | Engine: {self.primary_engine.upper()} | Running nodes: {len(self.running_nodes())}")
        print("=" * 80)

        for root_id in self.root_ids():
            self._render_node(root_id, 0, spinner_char)

        print("\nRecent events:")
        if not self.latest_events:
            print("  (no events yet)")
        else:
            for evt in self.latest_events[-5:]:
                status = evt["status"].upper()
                msg = evt["message"]
                print(f"  [{status:>9}] {msg}")

        if self.latest_llm:
            print("\nRecent LLM messages:")
            for line in self.latest_llm[-3:]:
                shortened = line if len(line) < 200 else line[:197] + "..."
                print(f"  {shortened}")

        if self.completed and not self.running_nodes():
            print("\n✅ Research completed.")

    def _render_node(self, node_id: str, depth: int, spinner_char: str) -> None:
        node = self.nodes.get(node_id)
        if not node:
            return
        indent = "  " * depth
        status = node.status.upper()
        progress = f" {node.progress:.0f}%" if node.progress is not None else ""
        marker = spinner_char if node.status == "running" else "*" if node.status == "completed" else "-"
        title = node.title or node.node_id
        print(f"{indent}{marker} {title} [{status}{progress}]")
        if node.metadata.get("details"):
            details = node.metadata["details"]
            exit_code = details.get("exit_code")
            if exit_code is not None:
                print(f"{indent}    exit_code: {exit_code}")
            if details.get("output"):
                output = details["output"].strip()
                preview = output if len(output) <= 200 else output[:197] + "..."
                for line in preview.splitlines():
                    print(f"{indent}    {line}")
        for child_id in node.children:
            self._render_node(child_id, depth + 1, spinner_char)


def http_to_ws(url: str) -> str:
    parsed = urlparse(url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "", "", "", ""))


async def start_research(session: aiohttp.ClientSession, http_base: str, query: str) -> Dict[str, Any]:
    payload = {"user_request": query}
    async with session.post(f"{http_base}/api/router/route-and-execute", json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_full_result(session: aiohttp.ClientSession, http_base: str, session_id: str) -> Optional[Dict[str, Any]]:
    async with session.get(f"{http_base}/api/research/sessions/{session_id}/full") as resp:
        if resp.status != 200:
            return None
        try:
            return await resp.json()
        except aiohttp.ContentTypeError:
            return None


async def monitor_session(http_base: str, session_id: str, primary_engine: str) -> None:
    ws_base = http_to_ws(http_base)
    tree = ResearchTree(session_id=session_id, primary_engine=primary_engine)

    async with aiohttp.ClientSession() as session:
        research_ws = await session.ws_connect(f"{ws_base}/ws/research/{session_id}")
        llm_ws = await session.ws_connect(f"{ws_base}/ws/llm/{session_id}")

        async def consume_research():
            async for msg in research_ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    payload = json.loads(msg.data)
                    if payload.get("type") == "research_event":
                        tree.apply_event(payload["event"])
                        tree.render()
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break

        async def consume_llm():
            async for msg in llm_ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    payload = json.loads(msg.data)
                    if payload.get("type", "").startswith("llm_"):
                        tree.record_llm(payload)
                        tree.render()
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break

        async def refresh():
            while True:
                await asyncio.sleep(0.5)
                tree.render()
                if tree.completed and not tree.running_nodes():
                    await research_ws.close()
                    await llm_ws.close()
                    break

        await asyncio.gather(consume_research(), consume_llm(), refresh())

        result = await fetch_full_result(session, http_base, session_id)
        if result:
            execution = result.get("execution", {})
            markdown = execution.get("report_markdown")
            if markdown:
                print("\nFinal Report (markdown preview):\n")
                preview = markdown if len(markdown) < 1000 else markdown[:997] + "..."
                print(preview)


async def run_cli(args: argparse.Namespace) -> None:
    http_base = args.host.rstrip("/")
    session_id = args.session
    primary_engine = "code_research"

    async with aiohttp.ClientSession() as session:
        if not session_id:
            response = await start_research(session, http_base, args.query)
            session_id = response.get("session_id")
            classification = response.get("classification", {})
            primary_engine = classification.get("primary_engine", primary_engine)
            print(f"Started research session {session_id} (engine={primary_engine})")
        else:
            print(f"Attaching to existing session {session_id}")

    await monitor_session(http_base, session_id, primary_engine)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI research monitor for UAgent")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Backend host URL (HTTP)")
    parser.add_argument("--session", help="Attach to an existing session id")
    parser.add_argument("--query", help="Start a new research session with the provided prompt")
    args = parser.parse_args(argv)

    if not args.session and not args.query:
        parser.error("either --query or --session must be provided")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except aiohttp.ClientError as exc:
        print(f"HTTP/WebSocket error: {exc}")
        print("Hint: ensure the backend is running and reachable. Use --host to target the correct port (e.g. http://127.0.0.1:8012 if you run on 8012).")
        sys.exit(1)


if __name__ == "__main__":
    main()
