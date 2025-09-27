from logging import DEBUG
from typing import Any

from litellm import ChatCompletionMessageToolCall
from litellm.types.utils import ModelResponse

from openhands.core.logger import llm_prompt_logger, llm_response_logger, concise_llm_logger, _setup_concise_llm_logger
from openhands.core.logger import openhands_logger as logger
import os

MESSAGE_SEPARATOR = '\n\n----------\n\n'


class DebugMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_concise_logging()

    def _init_concise_logging(self):
        """Initialize concise logging attributes"""
        if not hasattr(self, '_concise_logger'):
            self._concise_logger = None
        if not hasattr(self, '_pending_prompt'):
            self._pending_prompt = None

    def _get_concise_logger(self):
        """Get or create the concise LLM logger"""
        # Ensure attributes are initialized
        if not hasattr(self, '_concise_logger'):
            self._init_concise_logging()

        if self._concise_logger is None:
            workspace_dir = os.environ.get('WORKSPACE_BASE', '/tmp')
            self._concise_logger = _setup_concise_llm_logger(workspace_dir)
        return self._concise_logger

    def log_prompt(self, messages: list[dict[str, Any]] | dict[str, Any]) -> None:
        if not logger.isEnabledFor(DEBUG):
            # Don't use memory building message string if not logging.
            return
        if not messages:
            logger.debug('No completion messages!')
            return

        messages = messages if isinstance(messages, list) else [messages]
        debug_message = MESSAGE_SEPARATOR.join(
            self._format_message_content(msg)
            for msg in messages
            if msg['content'] is not None
        )

        if debug_message:
            llm_prompt_logger.debug(debug_message)
            # Store for concise logging (exclude system prompts)
            if not hasattr(self, '_pending_prompt'):
                self._init_concise_logging()
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            if user_messages:
                self._pending_prompt = self._format_message_content(user_messages[-1])
        else:
            logger.debug('No completion messages!')

    def log_response(self, resp: ModelResponse) -> None:
        if not logger.isEnabledFor(DEBUG):
            # Don't use memory building message string if not logging.
            return
        message_back: str = resp['choices'][0]['message']['content'] or ''
        tool_calls: list[ChatCompletionMessageToolCall] = resp['choices'][0][
            'message'
        ].get('tool_calls', [])
        if tool_calls:
            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments
                message_back += f'\nFunction call: {fn_name}({fn_args})'

        if message_back:
            llm_response_logger.debug(message_back)

            # Log concise combined interaction
            try:
                concise_logger = self._get_concise_logger()
                if self._pending_prompt and concise_logger:
                    # Format: USER: prompt | ASSISTANT: response
                    combined_log = f"USER: {self._pending_prompt.strip()}\nASSISTANT: {message_back.strip()}"
                    concise_logger.info(combined_log)
                    self._pending_prompt = None  # Clear after logging
            except Exception as e:
                logger.error(f"Failed to log concise LLM interaction: {e}")

    def _format_message_content(self, message: dict[str, Any]) -> str:
        content = message['content']
        if isinstance(content, list):
            return '\n'.join(
                self._format_content_element(element) for element in content
            )
        return str(content)

    def _format_content_element(self, element: dict[str, Any] | Any) -> str:
        if isinstance(element, dict):
            if 'text' in element:
                return str(element['text'])
            if (
                self.vision_is_active()
                and 'image_url' in element
                and 'url' in element['image_url']
            ):
                return str(element['image_url']['url'])
        return str(element)

    # This method should be implemented in the class that uses DebugMixin
    def vision_is_active(self) -> bool:
        raise NotImplementedError
