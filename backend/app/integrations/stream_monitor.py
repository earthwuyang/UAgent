"""Real-time streaming monitor for OpenHands commands."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any


class CommandStreamMonitor:
    """Monitor and stream command output in real-time."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.log_dir = workspace_path / "logs" / "commands"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.active_monitors = {}

    def create_log_file(self, action: str, command: Optional[str] = None) -> Path:
        """Create a log file for streaming output."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]

        # Create descriptive filename
        safe_action = action.replace("/", "_").replace(" ", "_")
        if command:
            # Extract key part of command for filename
            cmd_parts = command.split()
            if "pip" in command:
                log_name = f"{timestamp}_pip_install"
            elif "npm" in command:
                log_name = f"{timestamp}_npm_install"
            elif "apt" in command:
                log_name = f"{timestamp}_apt_install"
            elif "docker" in command:
                log_name = f"{timestamp}_docker_{cmd_parts[1] if len(cmd_parts) > 1 else 'cmd'}"
            else:
                cmd_preview = command[:30].replace("/", "_").replace(" ", "_").replace("|", "_")
                log_name = f"{timestamp}_{safe_action}_{cmd_preview}"
        else:
            log_name = f"{timestamp}_{safe_action}"

        log_path = self.log_dir / f"{log_name}.log"

        # Write header
        with log_path.open("w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"OpenHands Command Execution Log\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Action: {action}\n")
            if command:
                f.write(f"Command: {command}\n")
            f.write(f"Log File: {log_path}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Monitor this file in real-time with:\n")
            f.write(f"  tail -f {log_path}\n\n")
            f.write("Or watch for errors:\n")
            f.write(f"  tail -f {log_path} | grep -E 'ERROR|FAIL|timeout'\n\n")
            f.write("=" * 80 + "\n\n")

        return log_path

    def write_to_log(self, log_path: Path, content: str, section: Optional[str] = None):
        """Write content to log file with optional section header."""
        try:
            with log_path.open("a", encoding="utf-8") as f:
                if section:
                    f.write(f"\n--- {section} [{datetime.now().strftime('%H:%M:%S')}] ---\n")
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")
                f.flush()  # Force immediate write
        except Exception as e:
            print(f"Warning: Failed to write to log {log_path}: {e}")

    def write_json_to_log(self, log_path: Path, data: Dict[str, Any], section: str):
        """Write JSON data to log in readable format."""
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n--- {section} [{datetime.now().strftime('%H:%M:%S')}] ---\n")
                f.write(json.dumps(data, indent=2, default=str))
                f.write("\n")
                f.flush()
        except Exception as e:
            print(f"Warning: Failed to write JSON to log {log_path}: {e}")

    async def monitor_command_async(
        self,
        log_path: Path,
        process_check: Callable[[], bool],
        update_interval: int = 2,
        max_duration: int = 3600
    ):
        """Monitor a command execution asynchronously."""
        start_time = datetime.now()
        last_size = 0

        while process_check():
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > max_duration:
                self.write_to_log(log_path, f"Monitoring stopped after {max_duration} seconds", "TIMEOUT")
                break

            try:
                stat = log_path.stat()
                if stat.st_size != last_size:
                    last_size = stat.st_size
                    self.write_to_log(
                        log_path,
                        f"File size: {stat.st_size} bytes, Elapsed: {int(elapsed)}s",
                        "PROGRESS"
                    )
            except FileNotFoundError:
                pass

            await asyncio.sleep(update_interval)

        self.write_to_log(
            log_path,
            f"Monitoring ended. Total duration: {int((datetime.now() - start_time).total_seconds())}s",
            "COMPLETE"
        )

    def create_monitoring_script(self, log_path: Path) -> str:
        """Create a shell script to monitor the log file."""
        script = f"""#!/bin/bash
# Monitor script for {log_path.name}
LOG_FILE="{log_path}"

echo "Monitoring OpenHands command output..."
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Check if file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "Waiting for log file to be created..."
    while [ ! -f "$LOG_FILE" ]; do
        sleep 1
    done
fi

# Monitor with tail and highlight important patterns
tail -f "$LOG_FILE" | while read LINE; do
    # Highlight different types of output
    case "$LINE" in
        *ERROR*|*FAIL*|*Exception*)
            echo -e "\\033[31m$LINE\\033[0m"  # Red for errors
            ;;
        *SUCCESS*|*COMPLETE*|*installed*)
            echo -e "\\033[32m$LINE\\033[0m"  # Green for success
            ;;
        *WARNING*|*timeout*)
            echo -e "\\033[33m$LINE\\033[0m"  # Yellow for warnings
            ;;
        *"pip install"*|*"npm install"*|*downloading*|*installing*)
            echo -e "\\033[36m$LINE\\033[0m"  # Cyan for install progress
            ;;
        *)
            echo "$LINE"  # Default
            ;;
    esac
done
"""

        # Save script
        script_path = log_path.with_suffix(".monitor.sh")
        with script_path.open("w") as f:
            f.write(script)
        script_path.chmod(0o755)

        return f"bash {script_path}"

    def get_latest_logs(self, count: int = 10) -> list:
        """Get the latest log files."""
        if not self.log_dir.exists():
            return []

        logs = sorted(self.log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        return logs[:count]

    def print_monitoring_instructions(self, log_path: Path):
        """Print instructions for monitoring the log file."""
        print("\n" + "=" * 80)
        print("REAL-TIME COMMAND MONITORING")
        print("=" * 80)
        print(f"Log file created: {log_path}")
        print("\nMonitor in real-time with any of these commands:")
        print(f"  1. tail -f {log_path}")
        print(f"  2. tail -f {log_path} | grep -E 'ERROR|SUCCESS|pip'")
        print(f"  3. watch -n 1 'tail -20 {log_path}'")
        print(f"  4. {self.create_monitoring_script(log_path)}")
        print("\nLatest logs directory:")
        print(f"  ls -lt {self.log_dir} | head")
        print("=" * 80 + "\n")