# Max Tokens Configuration - Latest Settings

## Overview
Due to very long function calls with file contents (18000-20000+ characters), we've significantly increased max_tokens across all components.

## Current Token Limits

### CodeAct Runner (`codeact_runner.py`)
- **Primary Generation**: `20000` tokens (line 230)
  - Used for main LLM tool call generation
  - Handles function calls with extensive file contents

- **Retry Generation**: `15000` tokens (line 240)
  - Used when first attempt fails
  - Stricter prompt for focused response

### OpenHands Bridge (`openhands_bridge.py`)
- **Plan Generation**: `4000` tokens (line 142)
  - Generates multi-step execution plans
  - JSON structured response

### Scientific Research Engine (`scientific_research.py`)
- **Hypothesis Generation**: `6000` tokens (lines 1872, 3657)
  - Complex scientific hypotheses with multiple variables
  - Detailed experimental predictions

- **Evaluation**: `4000` tokens (line 2595)
  - Comprehensive evaluation of research results
  - Detailed analysis and metrics

### Deep Research Engine (`deep_research.py`)
- **Plan Response**: `4000` tokens (line 301)
  - Research plan with multiple steps

- **Reflection**: `3000` tokens (line 482)
  - Gap analysis and improvement suggestions

- **Synthesis**: `5000` tokens (line 811)
  - Final research synthesis with all findings

## Why These Limits?

### Observed Response Sizes
- Typical tool calls: 5000-10000 characters
- Complex file operations: 15000-18000 characters
- Maximum observed: 18544 characters

### Token to Character Ratio
- Approximately 1 token ≈ 4 characters
- 20000 tokens ≈ 80000 characters
- Provides ample buffer for longest responses

## Impact

### Benefits
✅ No more truncation errors
✅ Complete function calls with file contents
✅ Full JSON responses for complex operations
✅ Detailed analysis and synthesis

### Trade-offs
⚠️ Higher API costs (more tokens consumed)
⚠️ Slightly longer response times
⚠️ Increased memory usage

## Monitoring

Watch for these log messages:
```
WARNING - First parse attempt failed: Function call appears truncated
```

If you see response lengths exceeding 20000 characters, further increases may be needed.

## Configuration Guidelines

### Development/Debug
- Current settings are appropriate
- Monitor for truncation warnings

### Production
- Consider implementing dynamic token sizing
- Use smaller limits for simple operations
- Scale up only when needed

## Future Improvements

1. **Dynamic Sizing**: Adjust tokens based on operation complexity
2. **Streaming**: Use streaming for very large responses
3. **Chunking**: Split large operations into smaller parts
4. **Compression**: Minimize prompt size to leave room for response