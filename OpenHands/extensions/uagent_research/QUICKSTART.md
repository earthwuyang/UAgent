# UAgent Research - Quick Start Guide

Quick start guide for using the UAgent Research extension.

See PHASE1_COMPLETION_SUMMARY.md for full documentation.

## Installation

```bash
cd /home/wuy/AI/UAgent/OpenHands
poetry add playwright beautifulsoup4 cachetools
poetry run playwright install chromium
```

## Basic Usage

```python
import asyncio
from extensions.uagent_research.orchestrator.tree_orchestrator import TreeSearchOrchestrator
from extensions.uagent_research.adapters.base.agent_adapter import adapter_registry
from extensions.uagent_research.adapters.deepresearch.adapter import DeepResearchAdapter
from extensions.uagent_research.adapters.repomaster.adapter import RepoMasterAdapter
from extensions.uagent_research.adapters.codeact.adapter import CodeActAdapter

async def main():
    # Register adapters
    adapter_registry.register(DeepResearchAdapter())
    adapter_registry.register(RepoMasterAdapter())
    adapter_registry.register(CodeActAdapter())

    # Create orchestrator
    orchestrator = TreeSearchOrchestrator(max_parallel=3)

    # Run tree search
    tree = await orchestrator.run(
        goal="Research neural architecture search",
        max_iterations=5
    )

    print(f"Completed with {len(tree.nodes)} nodes")

asyncio.run(main())
```

## Testing

```bash
cd extensions/uagent_research
python tests/test_tools_integration.py
```
