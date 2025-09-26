from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..integrations.openhands_runtime import OpenHandsClientV2


@dataclass
class ProxyResponse:
    ok: bool
    status: str               # "SUCCEEDED" | "FAILED"
    engine: str               # "duckdb" | "postgres"
    sql: str
    rows: Optional[List[Dict[str, Any]]]
    rows_count: int
    elapsed_ms: Optional[int]
    error_type: Optional[str]
    error_message: Optional[str]


class ProxySQLTool:
    """First-class SQL proxy executor with JSONL ledger.

    Uses a short Python stdlib HTTP snippet inside the OpenHands job runner to
    post to the local research proxy. Does not require external tools.
    """

    def __init__(self, client: OpenHandsClientV2, workspace_dir: Path, base_url: str = "http://127.0.0.1:7890/sql") -> None:
        self.client = client
        self.ws = workspace_dir
        self.base_url = base_url
        self.ledger_fp = self.ws / "logs" / "proxy_calls.jsonl"
        self.ledger_fp.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, rec: Dict[str, Any]) -> None:
        rec = dict(rec)
        rec["ts"] = time.time()
        with self.ledger_fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def execute(
        self,
        sql: str,
        engine: str,
        dataset: Optional[str] = None,
        timeout_sec: int = 120,
        retries: int = 2,
        backoff: float = 1.5,
    ) -> ProxyResponse:
        payload = {"engine": engine, "sql": sql}
        if dataset:
            payload["dataset"] = dataset
        body = json.dumps(payload).replace('"', r'\"')
        py_snippet = f"""python - <<'PY'\nimport json, sys, urllib.request, urllib.error, time, os\nurl = {self.base_url!r}\npayload = {payload!r}\nreq = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={{'Content-Type':'application/json'}})\nstart=time.time()\nstatus=-1; body=''; err=None\ntry:\n    with urllib.request.urlopen(req, timeout={timeout_sec}) as resp:\n        status = resp.getcode(); body = resp.read().decode('utf-8','replace')\nexcept urllib.error.HTTPError as e:\n    status = e.code; body = e.read().decode('utf-8','replace'); err=str(e)\nexcept Exception as e:\n    err=str(e)\nelapsed_ms=int((time.time()-start)*1000)\ntry:\n    obj=json.loads(body) if body else {{}}\nexcept Exception:\n    obj={{'error': body[:1024]}}\nrec={{'phase':'proxy_exec','engine': {engine!r}, 'status':'SUCCEEDED' if (200<=status<300 and not obj.get('error')) else 'FAILED', 'elapsed_ms': elapsed_ms, 'http_status': status, 'error': obj.get('error')}}\nprint(json.dumps(rec))\ntry:\n    os.makedirs('logs', exist_ok=True)\n    with open('logs/proxy_calls.jsonl','a') as fp: fp.write(json.dumps(rec)+'\n')\nexcept Exception: pass\nPY\n"""

        attempt = 0
        last_err: Optional[str] = None
        started = time.time()
        while attempt <= retries:
            jid = await self.client.start_job(py_snippet, timeout_sec=timeout_sec + 10, dedup=False)
            res = await self.client.wait_job(jid, max_wait_sec=timeout_sec + 20)
            if not res.success:
                last_err = (res.error.message if res.error else "proxy job failed")
                self._log({"phase": "proxy_exec", "engine": engine, "status": "FAILED", "exit_code": res.exit_code, "error": last_err})
            else:
                try:
                    parsed = json.loads((res.stdout or "").strip())
                    status = parsed.get("status") or parsed.get("http_status")
                    elapsed_ms = parsed.get("elapsed_ms")
                    if parsed.get("status") == "SUCCEEDED" or (isinstance(status, int) and 200 <= status < 300 and not parsed.get("error")):
                        self._log(parsed)
                        return ProxyResponse(True, "SUCCEEDED", engine, sql, None, 0, elapsed_ms, None, None)
                    last_err = parsed.get("error") or f"HTTP {status}"
                    self._log({"phase": "proxy_exec", "engine": engine, "status": "FAILED", "error": last_err, "http_status": status})
                except Exception as e:
                    last_err = f"CLIENT_PARSE_ERROR: {e}"
                    self._log({"phase": "proxy_exec", "engine": engine, "status": "FAILED", "error": last_err})

            attempt += 1
            if attempt <= retries:
                await asyncio.sleep(backoff ** attempt)

        elapsed_total = int((time.time() - started) * 1000)
        return ProxyResponse(False, "FAILED", engine, sql, None, 0, elapsed_total, "NETWORK_PROXY", last_err or "unknown")
