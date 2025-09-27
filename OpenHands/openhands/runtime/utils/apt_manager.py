"""APT package manager with lock handling and retry logic."""

import logging
import os
import subprocess
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class APTLockError(Exception):
    """Exception raised when APT lock cannot be acquired."""
    pass


class APTManager:
    """APT package manager with intelligent lock handling."""

    def __init__(
        self,
        max_retries: int = 10,
        retry_delay: int = 5,
        lock_timeout: int = 300,
    ):
        """Initialize APT manager.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            lock_timeout: Maximum time to wait for lock in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.lock_timeout = lock_timeout

    def _check_lock_status(self) -> Optional[str]:
        """Check if APT locks are held by other processes.

        Returns:
            Process holding the lock or None if no lock
        """
        lock_files = [
            '/var/lib/dpkg/lock',
            '/var/lib/dpkg/lock-frontend',
            '/var/cache/apt/archives/lock'
        ]

        for lock_file in lock_files:
            try:
                result = subprocess.run(
                    ['lsof', lock_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    # Extract process info
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        process_line = lines[1]  # Skip header
                        parts = process_line.split()
                        if len(parts) >= 2:
                            return f"{parts[0]} (PID: {parts[1]})"
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                continue

        return None

    def _wait_for_lock_release(self) -> bool:
        """Wait for APT locks to be released.

        Returns:
            True if locks are released, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < self.lock_timeout:
            lock_holder = self._check_lock_status()
            if not lock_holder:
                logger.info("APT locks released, proceeding with operation")
                return True

            logger.debug(f"APT lock held by {lock_holder}, waiting...")
            time.sleep(self.retry_delay)

        logger.error(f"APT lock timeout after {self.lock_timeout} seconds")
        return False

    def _kill_hanging_apt_processes(self) -> bool:
        """Kill hanging APT processes as last resort.

        Returns:
            True if processes were killed, False otherwise
        """
        try:
            # Find hanging apt processes
            result = subprocess.run(
                ['pgrep', '-f', '(apt|dpkg)'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                logger.warning(f"Killing hanging APT processes: {pids}")

                for pid in pids:
                    try:
                        subprocess.run(['kill', '-9', pid], timeout=5)
                    except subprocess.SubprocessError:
                        continue

                # Wait for cleanup
                time.sleep(2)
                return True

        except subprocess.SubprocessError:
            pass

        return False

    def execute_apt_command(
        self,
        command: List[str],
        timeout: int = 300,
        force_unlock: bool = False
    ) -> subprocess.CompletedProcess:
        """Execute APT command with lock handling.

        Args:
            command: APT command to execute
            timeout: Command timeout in seconds
            force_unlock: Whether to force unlock if needed

        Returns:
            Completed process result

        Raises:
            APTLockError: If lock cannot be acquired
        """
        # Ensure DEBIAN_FRONTEND=noninteractive for automated installs
        env = os.environ.copy()
        env['DEBIAN_FRONTEND'] = 'noninteractive'

        for attempt in range(self.max_retries):
            try:
                # Check for locks before attempting
                lock_holder = self._check_lock_status()
                if lock_holder:
                    logger.warning(f"APT lock held by {lock_holder} (attempt {attempt + 1}/{self.max_retries})")

                    if not self._wait_for_lock_release():
                        if force_unlock and attempt == self.max_retries - 1:
                            logger.warning("Force unlocking APT as last resort")
                            self._kill_hanging_apt_processes()
                        else:
                            time.sleep(self.retry_delay)
                            continue

                # Execute the command
                logger.info(f"Executing APT command: {' '.join(command)}")
                result = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode == 0:
                    logger.info(f"APT command completed successfully")
                    return result

                # Handle lock-related errors
                if ('could not get lock' in result.stderr.lower() or
                    'dpkg: error' in result.stderr.lower()):
                    logger.warning(f"APT lock error on attempt {attempt + 1}: {result.stderr[:200]}")
                    time.sleep(self.retry_delay)
                    continue

                # Other errors, return immediately
                logger.error(f"APT command failed: {result.stderr}")
                return result

            except subprocess.TimeoutExpired:
                logger.error(f"APT command timed out after {timeout} seconds")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise

        raise APTLockError(f"Failed to execute APT command after {self.max_retries} attempts")

    def update_packages(self) -> bool:
        """Update package lists.

        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.execute_apt_command(['apt-get', 'update'])
            return result.returncode == 0
        except (APTLockError, subprocess.SubprocessError) as e:
            logger.error(f"Failed to update packages: {e}")
            return False

    def install_packages(self, packages: List[str], force_unlock: bool = False) -> bool:
        """Install packages.

        Args:
            packages: List of package names to install
            force_unlock: Whether to force unlock if needed

        Returns:
            True if successful, False otherwise
        """
        if not packages:
            return True

        command = ['apt-get', 'install', '-y'] + packages
        try:
            result = self.execute_apt_command(command, force_unlock=force_unlock)
            return result.returncode == 0
        except (APTLockError, subprocess.SubprocessError) as e:
            logger.error(f"Failed to install packages {packages}: {e}")
            return False


# Global APT manager instance
apt_manager = APTManager()