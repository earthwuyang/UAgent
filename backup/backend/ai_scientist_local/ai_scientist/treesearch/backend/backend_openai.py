import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")

_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@once
def _setup_openai_client():
    global _client
    _client = openai.OpenAI(max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            raw_args = choice.message.tool_calls[0].function.arguments
            output = json.loads(raw_args)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            # Try to sanitize control characters and retry
            try:
                import re
                raw_args = choice.message.tool_calls[0].function.arguments

                # More aggressive sanitization
                # 1. Remove problematic control characters except \n, \t
                sanitized_args = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', raw_args)

                # 2. Escape any remaining backslashes that aren't part of valid escapes
                sanitized_args = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', sanitized_args)

                # 3. Fix common JSON issues
                # Replace unescaped newlines in strings with \\n
                sanitized_args = re.sub(r'(?<!\\)(\n)', r'\\n', sanitized_args)
                # Replace unescaped tabs in strings with \\t
                sanitized_args = re.sub(r'(?<!\\)(\t)', r'\\t', sanitized_args)

                logger.warning(f"Attempting to parse sanitized arguments: {sanitized_args[:200]}...")
                output = json.loads(sanitized_args)
                logger.info("Successfully parsed after sanitization")
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse even after sanitization: {e2}")
                logger.error(f"Original error: {e}")
                logger.error(f"Problematic string (first 500 chars): {raw_args[:500]}")

                # Last resort: try to extract a minimal valid response
                try:
                    logger.warning("Attempting last resort fallback response")
                    func_name = choice.message.tool_calls[0].function.name

                    # Create a minimal valid response based on function name
                    if "review" in func_name.lower():
                        output = {
                            "review": "Error parsing LLM response. Code execution may have issues.",
                            "is_buggy": True,
                            "reasoning": "Failed to parse LLM analysis due to control characters in response."
                        }
                    elif "metric" in func_name.lower():
                        output = {
                            "value": None,
                            "maximize": None,
                            "name": "parsing_error",
                            "description": "Failed to parse metrics due to control characters"
                        }
                    else:
                        # Generic fallback
                        output = {"error": "Failed to parse LLM response", "fallback": True}

                    logger.warning(f"Using fallback response: {output}")
                except Exception as e3:
                    logger.error(f"Even fallback failed: {e3}")
                    raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
