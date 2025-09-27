from dataclasses import dataclass, field
from typing import MutableMapping
import os

import httpx

from openhands.core.logger import openhands_logger as logger

# Create client that bypasses proxy for localhost connections
# This fixes the issue where proxy interferes with local OpenHands server connections
def _create_client_with_localhost_bypass():
    """Create httpx client that bypasses proxy for localhost"""
    original_no_proxy = os.environ.get('NO_PROXY', os.environ.get('no_proxy', ''))
    localhost_exclusions = 'localhost,127.0.0.1,0.0.0.0,::1'

    # Build comprehensive NO_PROXY list
    if original_no_proxy:
        # Check if localhost is already in the list (case-insensitive)
        existing_entries = [entry.strip().lower() for entry in original_no_proxy.split(',')]
        new_entries = [entry for entry in localhost_exclusions.split(',')
                      if entry.lower() not in existing_entries]
        if new_entries:
            no_proxy_value = f"{original_no_proxy},{','.join(new_entries)}"
        else:
            no_proxy_value = original_no_proxy
    else:
        no_proxy_value = localhost_exclusions

    # Set environment variables for the client creation
    os.environ['NO_PROXY'] = no_proxy_value
    os.environ['no_proxy'] = no_proxy_value

    # Create client with updated proxy settings
    client = httpx.Client()

    # Restore original environment if it existed, otherwise keep the localhost bypass
    if original_no_proxy and original_no_proxy != no_proxy_value:
        # Keep the localhost bypass in place for other processes
        pass  # Don't restore, keep localhost bypass active

    return client

CLIENT = _create_client_with_localhost_bypass()


@dataclass
class HttpSession:
    """request.Session is reusable after it has been closed. This behavior makes it
    likely to leak file descriptors (Especially when combined with tenacity).
    We wrap the session to make it unusable after being closed
    """

    _is_closed: bool = False
    headers: MutableMapping[str, str] = field(default_factory=dict)

    def request(self, *args, **kwargs):
        if self._is_closed:
            logger.error(
                'Session is being used after close!', stack_info=True, exc_info=True
            )
            self._is_closed = False
        headers = kwargs.get('headers') or {}
        headers = {**self.headers, **headers}
        kwargs['headers'] = headers
        return CLIENT.request(*args, **kwargs)

    def stream(self, *args, **kwargs):
        if self._is_closed:
            logger.error(
                'Session is being used after close!', stack_info=True, exc_info=True
            )
            self._is_closed = False
        headers = kwargs.get('headers') or {}
        headers = {**self.headers, **headers}
        kwargs['headers'] = headers
        return CLIENT.stream(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.request('GET', *args, **kwargs)

    def post(self, *args, **kwargs):
        return self.request('POST', *args, **kwargs)

    def patch(self, *args, **kwargs):
        return self.request('PATCH', *args, **kwargs)

    def put(self, *args, **kwargs):
        return self.request('PUT', *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.request('DELETE', *args, **kwargs)

    def options(self, *args, **kwargs):
        return self.request('OPTIONS', *args, **kwargs)

    def close(self) -> None:
        self._is_closed = True
