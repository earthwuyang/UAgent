from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .ras import ResearchActionSpec, Step

__all__ = [
    "RASValidationError",
    "validate_research_action_spec",
]


class RASValidationError(RuntimeError):
    """Raised when a ResearchActionSpec fails structural validation."""


def validate_research_action_spec(
    spec: ResearchActionSpec,
    *,
    design_context: Optional[Dict[str, Any]] = None,
    min_steps: int = 3,
) -> None:
    """Validate that a ResearchActionSpec is sufficiently concrete.

    The checks are intentionally generic: they only require that multi-stage
    workflows are described with multiple step kinds and that evidence
    assertions are present. Domain-specific logic remains with the planner.
    """

    errors: List[str] = []
    if spec.version < 1:
        errors.append("`version` must be >= 1")

    if not isinstance(spec.run, dict) or not spec.run:
        errors.append("`run` block must be a non-empty object")

    steps: Sequence[Step] = spec.steps or []
    if len(steps) < max(1, min_steps):
        errors.append(f"spec must contain at least {max(1, min_steps)} steps")

    step_ids: Set[str] = set()
    unique_kinds: Set[str] = set()
    for idx, step in enumerate(steps):
        step_id = (step.id or "").strip()
        if not step_id:
            errors.append(f"step #{idx} is missing an id")
        if step_id in step_ids:
            errors.append(f"duplicate step id '{step_id}'")
        step_ids.add(step_id)
        unique_kinds.add(step.kind)

        payload = getattr(step, "with_", None)
        if payload is None:
            errors.append(f"step '{step_id or step.kind}' is missing required `with` config")
            continue
        if not isinstance(payload, dict):
            errors.append(f"step '{step_id or step.kind}' has invalid `with` payload (must be object)")
            continue

        if step.kind == "run_commands":
            commands = payload.get("commands")
            if not isinstance(commands, list) or not commands:
                errors.append(f"step '{step_id}' must declare one or more commands")
            if commands and not _commands_are_concrete(commands):
                errors.append(
                    f"step '{step_id}' commands must be strings or lists of strings"
                )
        elif step.kind == "fetch_repo":
            if not payload.get("url") or not payload.get("dest"):
                errors.append(f"step '{step_id}' must provide both `url` and `dest`")
        elif step.kind == "collect_artifacts":
            patterns = payload.get("patterns")
            if not isinstance(patterns, list) or not patterns:
                errors.append(f"step '{step_id}' must declare artifact patterns")

    assertions = spec.assertions or []
    if not assertions:
        errors.append("spec must declare at least one assertion")
    else:
        for idx, assertion in enumerate(assertions):
            if not isinstance(assertion, dict):
                errors.append(f"assertion #{idx} must be an object with a `type` field")
                continue
            if not assertion.get("type"):
                errors.append(f"assertion #{idx} is missing `type`")

    flags = _derive_design_flags(design_context)
    requires_code_edit = flags["requires_code_edit"]
    requires_build = flags["requires_build"]
    requires_multiphase = flags["requires_multiphase"]

    if requires_code_edit and "code_edit" not in unique_kinds:
        errors.append(
            "design calls for source modifications but spec omits a `code_edit` step"
        )

    if requires_build and not _has_build_stage(steps):
        errors.append(
            "design references build/compile work but spec lacks a build step or command"
        )

    if requires_multiphase and len(unique_kinds) < 2:
        errors.append(
            "spec must enumerate multiple step kinds (e.g. fetch, edit, build, run) "
            "for multi-phase workflows"
        )

    if errors:
        raise RASValidationError("Invalid ResearchActionSpec:\n- " + "\n- ".join(dict.fromkeys(errors)))


def _commands_are_concrete(commands: Iterable[Any]) -> bool:
    for cmd in commands:
        if isinstance(cmd, str):
            continue
        if isinstance(cmd, list) and all(isinstance(part, str) for part in cmd):
            continue
        return False
    return True


def _derive_design_flags(context: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    text = _context_to_text(context)
    lower = text.lower()

    requires_code_edit = any(
        token in lower
        for token in {"modify", "instrument", "patch", "source code", "hook", "edit"}
    )

    requires_build = any(token in lower for token in {"build", "compile", "cmake", "make", "install"})

    requires_data = any(
        token in lower
        for token in {"collect", "benchmark", "measure", "run query", "execute query", "latency"}
    )
    requires_training = any(token in lower for token in {"train", "model", "learning", "fit"})

    requires_multiphase = requires_code_edit or requires_build or (requires_data and requires_training)

    return {
        "requires_code_edit": requires_code_edit,
        "requires_build": requires_build,
        "requires_multiphase": requires_multiphase,
    }


def _context_to_text(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return ""
    parts: List[str] = []
    for value in context.values():
        parts.append(_value_to_text(value))
    return " \n".join(parts)


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{k} {_value_to_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_value_to_text(item) for item in value)
    return str(value)


def _has_build_stage(steps: Sequence[Step]) -> bool:
    build_keywords = {"make", "cmake", "ninja", "configure", "build", "compile"}
    for step in steps:
        if step.kind == "build":
            return True
        if step.kind == "run_commands":
            payload = getattr(step, "with_", {}) or {}
            commands = payload.get("commands") or []
            for cmd in commands:
                cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
                if any(keyword in cmd_str for keyword in build_keywords):
                    return True
    return False
