# LLM Configuration Fix for UAgent Research Extension

## Problem

The research extension's CodeAct agents were failing with authentication errors:

```
litellm.AuthenticationError: AuthenticationError: OpenAIException - The api_key client option must be set
```

### Root Cause

The research agents were using hardcoded default LLM configuration (`model="gpt-4o", api_key=None`) instead of reading from environment variables like the main OpenHands application does.

## Solution

Created a unified configuration loading mechanism that reads LLM settings from environment variables, consistent with main OpenHands.

### Files Modified

1. **`extensions/uagent_research/utils/config_utils.py`** (NEW)
   - Added `load_llm_config_from_env()` function
   - Reads configuration from environment variables:
     - `LLM_MODEL`: Model identifier (e.g., "openai/qwen3-coder-plus")
     - `LLM_API_KEY`: API key for the LLM provider
     - `LLM_BASE_URL`: Base URL for the LLM API
     - `LLM_NUM_RETRIES`: Number of retries
     - `LLM_RETRY_MIN_WAIT`: Minimum wait time between retries  
     - `LLM_RETRY_MAX_WAIT`: Maximum wait time between retries
     - `LLM_TIMEOUT`: Request timeout in seconds
     - `LLM_TEMPERATURE`: Temperature for sampling
   - Added `get_llm_config_for_research()` helper function

2. **`extensions/uagent_research/adapters/codeact/adapter.py`** (MODIFIED)
   - Updated `CodeActAdapter._get_llm_config()` method
   - Now calls `load_llm_config_from_env()` instead of creating hardcoded config
   - Provides detailed logging of loaded configuration

3. **`extensions/uagent_research/utils/__init__.py`** (NEW)
   - Created empty init file for utils module

## Usage

### Environment Variables

Set the following in your `.env` file or export them:

```bash
# Required
LLM_MODEL=openai/qwen3-coder-plus
LLM_API_KEY=${DASHSCOPE_API_KEY}
LLM_BASE_URL=${DASHSCOPE_BASE_URL}

# Optional (with defaults shown)
LLM_NUM_RETRIES=8
LLM_RETRY_MIN_WAIT=15
LLM_RETRY_MAX_WAIT=120
LLM_TIMEOUT=600
LLM_TEMPERATURE=0.0
```

### With litellm Providers

The configuration works with any litellm-supported provider:

#### OpenAI
```bash
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-...
```

#### Anthropic
```bash
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-...
```

#### Qwen (via OpenRouter or DashScope)
```bash
LLM_MODEL=openai/qwen3-coder-plus
LLM_API_KEY=sk-...
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

#### OpenRouter
```bash
LLM_MODEL=openai/gpt-4o
LLM_API_KEY=sk-or-v1-...
LLM_BASE_URL=https://openrouter.ai/api/v1
```

## Testing

After applying this fix, research agents should:

1. ✅ Load LLM configuration from environment variables
2. ✅ Use the configured model and API key
3. ✅ Successfully authenticate with the LLM provider
4. ✅ Execute research tasks without authentication errors

### Verify Configuration Loading

Check the logs for:

```
Loading LLM config from environment:
  Model: openai/qwen3-coder-plus
  Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
  API Key: set
  Retries: 8
  Timeout: 600
```

### Verify Research Execution

When research is triggered, you should see:

```
🔬 Research mode triggered
Launching a scientific research experiment now...
```

Without the previous authentication errors.

## Implementation Details

### Config Loading Flow

1. `CodeActAdapter` initializes without `llm_config`
2. When a research task runs, `_get_llm_config()` is called
3. If no custom config provided, calls `load_llm_config_from_env()`
4. Environment variables are read and converted to `LLMConfig` object
5. Config is passed to `HeadlessAgentSession`
6. Session creates agent with proper LLM authentication

### Consistency with Main OpenHands

The implementation mirrors main OpenHands config loading:

- Uses `openhands.core.config.LLMConfig` (same class)
- Reads same environment variables (`LLM_*`)
- Uses `dotenv` for `.env` file support
- Converts API keys to `SecretStr` for security
- Provides same defaults and validation

## Related Files

- `openhands/core/config/llm_config.py`: LLMConfig class definition
- `openhands/core/config/utils.py`: Main OpenHands config loading
- `openhands/llm/llm.py`: LLM initialization with litellm
- `.env`: Environment configuration file

## Notes

- API keys are securely handled using Pydantic's `SecretStr`
- Configuration is logged (without exposing sensitive data)
- The fix is backward compatible - custom configs still work
- Uses `load_dotenv()` to support `.env` files
- Works with all litellm-supported providers

## Future Improvements

1. Support for per-agent LLM configuration
2. Dynamic model selection based on task type
3. Cost tracking and budget limits
4. Fallback models for reliability
5. Configuration validation and helpful error messages
