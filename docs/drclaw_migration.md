# UAgent-DrClaw Migration Analysis

## Scope
This note summarizes how the current repository is wired to OpenHands and where to decouple for a standalone autonomous research loop.

## OpenHands-specific components identified

1. **Session-centric orchestration**
   - `WebSession` in `OpenHands/openhands/server/session/session.py` owns lifecycle for web socket events, runtime interaction, and research hooks.
   - It eagerly constructs `AgentSession` and conditionally integrates research middleware, coupling UI/session concerns with research execution.

2. **Coordinator wrapper around existing research stack**
   - `MultiAgentCoordinator` in `OpenHands/openhands/server/session/multi_agent_coordinator.py` manages sub-agent lifecycle and proxies to `TreeSearchOrchestrator` from `uagent_research`.
   - Current flow is tightly tied to OpenHands event buses (`EventBus`, `ControlBus`, `MessageBus`) and streaming observations.

3. **Execution flow currently in server path**
   - Research execution is started from session-level message handling, then tracked as background tasks by the coordinator, then reported back to session streams.
   - This makes autonomous experiment optimization hard to run headless/reproducibly without server bootstrapping.

## Migration direction for DrClaw

- Extract a **headless experiment loop** independent of web session and runtime abstractions.
- Keep extensibility by defining module boundaries (`planner`, `executor`, `evaluator`, `memory`, `experiments`).
- Preserve iterative orchestration semantics but simplify to deterministic experiment runs and file-based outputs for reproducibility.
- Decouple policy logic from execution so future LLM/RL planners can be swapped in without touching simulator or memory layers.

## Assumptions

- DrClaw is an experimental prototype and can run as a standalone Python entrypoint (`experiment_runner.py`).
- Existing OpenHands runtime/event infra remains intact and unchanged; DrClaw is additive and does not break legacy behavior.
