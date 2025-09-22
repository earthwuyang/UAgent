import asyncio
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.memory import AgentMemory, AVDBConfig


def test_agent_memory_add_and_query(tmp_path):
    cfg = AVDBConfig(db_path=str(tmp_path / "agent_db.lance"), dim=64)
    memory = AgentMemory(cfg)

    async def _run():
        await memory.add_episodic("Test event about DuckDB vs Postgres", importance=0.6, tags={"topic": "duckdb"})
        results = await memory.query("episodic", "duckdb", k=1)
        assert results
        await memory.prune()

    asyncio.run(_run())

    assert Path(cfg.db_path).exists()
