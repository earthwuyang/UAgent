# OpenHands V3 LLM Interaction Logging Guide

## Overview
OpenHands V3 bridge now logs every detailed interaction with the LLM (Kimi model) for debugging and analysis purposes.

## Log Locations

### 1. Workspace Logs (Per Experiment)
When running experiments, logs are saved in the workspace directory:
```
/tmp/test_v3_simple/logs/
├── openhands_stdout.log     # General OpenHands output + LLM verbose logs
├── openhands_stderr.log     # Detailed debug logs + LLM interactions
└── openhands_llm_interactions.log  # Dedicated LLM log (if configured)
```

### 2. Global OpenHands Logs
Detailed prompt/response logs are saved in:
```
/home/wuy/AI/UAgent/OpenHands/logs/llm/default/
├── prompt_001.log      # Full prompt sent to LLM
├── response_001.log    # LLM response
├── prompt_002.log      # Next interaction prompt
└── response_002.log    # Next interaction response
```

## What Information is Logged

### In stderr.log:
- **Model Configuration**: `"model": "openai/kimi-k2-turbo-preview"`
- **API Endpoint**: `"base_url": "https://api.moonshot.cn/v1"`
- **Token Usage**: Input/output tokens, total tokens, cached tokens
- **Response Latency**: Time taken for each LLM call
- **Raw LLM Response**: Complete JSON response from Kimi API
- **Agent Decision Process**: How OpenHands processes LLM responses

### In stdout.log:
- **Provider Information**: LiteLLM provider details
- **LLM Verbose Output**: Raw request/response data
- **Auto-confirmation**: `⚠️ DEBUG_LLM enabled automatically (headless mode)`

### In prompt_XXX.log:
- **Complete System Prompt**: Full instructions sent to LLM
- **Context**: Previous conversation history
- **User Task**: The specific goal being executed

### In response_XXX.log:
- **LLM Response**: Exact response from the model
- **Function Calls**: Tools/functions called by the LLM
- **Agent Actions**: What actions the agent decided to take

## Configuration Added

The V3 bridge automatically enables:
```python
env["DEBUG_LLM"] = "true"                    # Enable LLM debugging
env["DEBUG_LLM_AUTO_CONFIRM"] = "true"       # Skip interactive confirmation
env["LOG_TO_FILE"] = "true"                  # Write logs to files
env["LOG_LEVEL"] = "DEBUG"                   # Maximum verbosity
```

## Example Log Snippets

### Model Configuration:
```
[92m10:04:35 - openhands:DEBUG[0m: llm.py:483 - Model info: {
  "model": "openai/kimi-k2-turbo-preview",
  "base_url": "https://api.moonshot.cn/v1"
}
```

### Token Usage:
```
[92m10:04:54 - openhands:DEBUG[0m: llm.py:664 - Response Latency: 1.431 seconds
Input tokens: 9744 | Output tokens: 57
```

### Raw LLM Response:
```
RAW RESPONSE:
{"id": "chatcmpl-68d7464527894c282afedf99", "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "...", "role": "assistant"}}], "model": "kimi-k2-turbo-preview", "usage": {"completion_tokens": 57, "prompt_tokens": 9744, "total_tokens": 9801, "cached_tokens": 9472}}
```

## Usage

The logging is automatically enabled for all V3 experiments. To view logs:

```bash
# View real-time LLM interactions
tail -f /tmp/test_v3_simple/logs/openhands_stderr.log | grep -A5 -B5 "RAW RESPONSE\|kimi-k2-turbo"

# View latest prompt sent to LLM
cat /home/wuy/AI/UAgent/OpenHands/logs/llm/default/prompt_001.log

# View latest LLM response
cat /home/wuy/AI/UAgent/OpenHands/logs/llm/default/response_001.log

# Search for token usage
grep "Input tokens\|Output tokens" /tmp/test_v3_simple/logs/openhands_stderr.log
```

## Benefits

- **Full Transparency**: See exactly what prompts are sent to Kimi
- **Performance Analysis**: Track token usage and response times
- **Debugging**: Understand why the agent made specific decisions
- **Cost Tracking**: Monitor token consumption per experiment
- **Model Verification**: Confirm the correct model (Kimi) is being used