"""Ad-hoc verification of the OpenHands runtime integration.

The script launches the OpenHands action execution server through
`OpenHandsActionServerRunner`, mirrors a couple of CodeAct style
operations (run + str_replace_editor calls), and asserts that the
relative paths the agent would emit are rewritten correctly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


# Ensure repository root is importable when running as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.integrations.openhands_runtime import OpenHandsActionServerRunner


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _exercise_runtime() -> None:
    runner = OpenHandsActionServerRunner()
    _assert(runner.is_available, "OpenHands runtime is not available in this environment")

    with TemporaryDirectory(prefix="uagent_openhands_test_") as tmp_dir:
        workspace = Path(tmp_dir).resolve()

        # Prime the workspace with a code/ directory via a shell command that the
        # agent would send. This doubles as a sanity-check that `run` actions
        # execute relative to the workspace root.
        session = await runner.open_session(workspace, enable_browser=False)
        try:
            mkdir_result = await session.run_cmd('bash -lc "mkdir -p code"', timeout=60)
            _assert(
                mkdir_result.execution_result.success,
                f"mkdir command failed: {mkdir_result.execution_result.stdout}",
            )

            target_rel_path = "code/normalize_data.py"
            initial_content = "print('hello from CodeAct runtime')\n"

            # Issue a create command using the relative path the LLM would emit.
            create_result = await session.file_edit(
                target_rel_path,
                command="create",
                file_text=initial_content,
                timeout=60,
            )
            create_obs = create_result.raw_observation
            create_output = str(create_obs.get("content", ""))
            _assert(
                not create_output.startswith("ERROR"),
                f"create command failed: {create_result.raw_observation}",
            )

            expected_path = workspace / target_rel_path
            _assert(
                expected_path.exists(),
                f"Expected file {expected_path} was not created",
            )
            _assert(
                expected_path.read_text() == initial_content,
                "Created file contents did not match",
            )

            # Replace the line using str_replace to ensure edits also succeed.
            replacement = "print('runtime edit succeeded')\n"
            replace_result = await session.file_edit(
                target_rel_path,
                command="str_replace",
                old_str=initial_content.rstrip("\n"),
                new_str=replacement.rstrip("\n"),
                timeout=60,
            )
            replace_obs = replace_result.raw_observation
            replace_output = str(replace_obs.get("content", ""))
            _assert(
                not replace_output.startswith("ERROR"),
                f"str_replace command failed: {replace_result.raw_observation}",
            )
            _assert(
                expected_path.read_text() == replacement,
                "Edited file contents did not match replacement",
            )

            # Finally run the script to ensure execute_bash works with relative paths.
            run_result = await session.run_cmd(
                "bash -lc 'python code/normalize_data.py'",
                timeout=60,
            )
            _assert(
                run_result.execution_result.success,
                f"python execution failed: {run_result.execution_result.stdout}",
            )
        finally:
            await session.close()

    print("OpenHands runtime integration test passed")


def main() -> None:
    try:
        asyncio.run(_exercise_runtime())
    except AssertionError as exc:
        print(f"Test failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - unexpected runtime errors
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
