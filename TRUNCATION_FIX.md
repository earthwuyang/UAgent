# Fix for LLM Response Truncation Issues

## Problem
LLM responses were being truncated, causing:
1. "Function call appears truncated (missing </function> tag)" errors
2. "OpenHands plan generation returned non-dict payload" errors
3. Incomplete tool calls and failed JSON parsing
4. Response lengths of 14761+ characters being cut off

## Root Cause
The `max_tokens` parameter was set too low for complex responses that include:
- Detailed analysis with multiple steps
- Long file contents in tool calls
- Complex JSON structures in plans
- Multiple hypotheses in scientific research

## Solution

### Increased max_tokens across all engines (Latest Update):

#### 1. CodeAct Runner (`backend/app/services/codeact_runner.py`)
- **Primary generation**: 4000 → 8000 → **20000 tokens**
- **Retry generation**: 3000 → 6000 → **15000 tokens**
- **Reason**: Tool calls with file content can be 18000-20000+ characters

#### 2. OpenHands Bridge (`backend/app/integrations/openhands_bridge.py`)
- **Plan generation**: 700 → 2000 → **4000 tokens**
- **Added fallback**: Attempts to repair truncated JSON
- **Error reporting**: Now includes response length in error messages

#### 3. Scientific Research Engine (`backend/app/core/research_engines/scientific_research.py`)
- **Hypothesis generation**: 1200 → 3000 → **6000 tokens**
- **Evaluation**: 700 → 2000 → **4000 tokens**
- **Reason**: Complex scientific hypotheses need more space

#### 4. Deep Research Engine (`backend/app/core/research_engines/deep_research.py`)
- **Plan response**: 800 → 2000 → **4000 tokens**
- **Reflection**: 600 → 1500 → **3000 tokens**
- **Synthesis**: 900 → 2500 → **5000 tokens**
- **Reason**: Comprehensive research plans and synthesis need more space

## Implementation Details

### Enhanced Error Handling
```python
# In openhands_bridge.py
if not isinstance(parsed, dict):
    # Try to extract JSON from potentially truncated response
    import re
    json_match = re.search(r'\{[\s\S]*', response)
    if json_match:
        try:
            import json
            parsed = json.loads(json_match.group(0) + ']}')  # Try to close truncated JSON
        except:
            pass
```

### Logging Improvements
- Added response length to error messages
- Log warnings for first parse attempts
- More detailed error reporting

## Testing
Monitor the backend logs for:
```bash
grep "truncated\|parse attempt failed" /path/to/logs
```

## Impact
- Eliminates most truncation errors
- Allows complex multi-step plans
- Supports longer file contents in tool calls
- Enables detailed scientific hypotheses

## Trade-offs
- Higher token usage (cost)
- Slightly longer response times
- More memory usage for responses

## Future Improvements
1. Dynamic token sizing based on complexity
2. Streaming responses for very long outputs
3. Response chunking for extremely large operations