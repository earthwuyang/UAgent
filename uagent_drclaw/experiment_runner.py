from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from uagent_drclaw.evaluator import MetricsEvaluator
from uagent_drclaw.executor import ExperimentExecutor
from uagent_drclaw.executor.pipeline_simulator import PipelineSimulator
from uagent_drclaw.memory import ExperimentMemory
from uagent_drclaw.planner import HeuristicPlanner
from uagent_drclaw.types import PipelineConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        # Tiny fallback parser for this known config shape
        data: dict[str, Any] = {}
        section: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":"):
                section = line[:-1]
                data[section] = {}
                continue
            if ":" in line and section is not None:
                k, v = [p.strip() for p in line.split(":", 1)]
                if v.lower() in {"true", "false"}:
                    parsed: Any = v.lower() == "true"
                else:
                    try:
                        parsed = int(v)
                    except ValueError:
                        try:
                            parsed = float(v)
                        except ValueError:
                            parsed = v
                data[section][k] = parsed
        return data


def _now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def run_agent_loop(config_path: Path, output_root: Path) -> Path:
    cfg = _load_yaml(config_path)
    exp_cfg = cfg["experiment"]
    search = cfg["search_space"]

    initial = PipelineConfig.from_dict(cfg["initial_config"])
    planner = HeuristicPlanner(
        max_batch_size=int(search["max_batch_size"]),
        max_workers=int(search["max_workers"]),
        max_prefetch=int(search["max_prefetch_size"]),
        max_parallelism=int(search["max_parallelism"]),
    )
    executor = ExperimentExecutor(
        simulator=PipelineSimulator(seed=int(exp_cfg.get("random_seed", 7)))
    )
    evaluator = MetricsEvaluator()
    memory = ExperimentMemory()

    run_dir = output_root / f"{exp_cfg['name']}_{_now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logs: list[str] = [
        "[assumption] Pipeline metrics are simulator-derived approximations for research prototyping",
        "[assumption] Convergence is no-improvement patience based, not statistical significance",
    ]
    metrics_log: list[dict[str, Any]] = []

    max_iterations = int(exp_cfg["max_iterations"])
    patience = int(exp_cfg["no_improvement_patience"])
    no_improvement = 0

    current = initial
    last_metrics = None

    for i in range(1, max_iterations + 1):
        decision = planner.propose(current, last_metrics, memory)
        candidate = decision.candidate

        logs.append(f"[iter={i}] PLAN config={candidate.to_dict()} reason={decision.reason}")

        result = executor.execute(candidate)
        score = evaluator.score(result)

        memory.record(candidate, result, score)

        improved = memory.best_config == candidate and not result.failed
        if improved:
            no_improvement = 0
        else:
            no_improvement += 1

        logs.append(
            f"[iter={i}] EVAL score={score:.3f} improved={improved} "
            f"throughput={result.throughput_samples_per_sec} "
            f"gpu={result.gpu_utilization} latency={result.latency_ms_per_batch} "
            f"failed={result.failed}"
        )

        metrics_log.append(
            {
                "iteration": i,
                "config": candidate.to_dict(),
                "decision_reason": decision.reason,
                "reverted": decision.reverted,
                "metrics": asdict(result),
                "score": score,
                "is_best": improved,
            }
        )

        current = memory.best_config or current
        last_metrics = result

        if no_improvement >= patience:
            logs.append(f"[iter={i}] STOP no improvement for {patience} iterations")
            break

    best = memory.best_config
    summary = {
        "experiment_name": exp_cfg["name"],
        "iterations_executed": len(metrics_log),
        "best_config": best.to_dict() if best else None,
        "best_score": memory.best_score,
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")
    (run_dir / "logs.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")

    summary_lines = [
        f"# Experiment Summary: {exp_cfg['name']}",
        "",
        f"- Iterations executed: {summary['iterations_executed']}",
        f"- Best score: {summary['best_score']:.3f}",
        f"- Best config: `{summary['best_config']}`",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UAgent-DrClaw autonomous optimization loop")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("uagent_drclaw/experiments/config.yaml"),
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("uagent_drclaw/results"),
        help="Folder for experiment outputs",
    )
    args = parser.parse_args()

    run_dir = run_agent_loop(args.config, args.output_root)
    print(f"DrClaw run complete. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
