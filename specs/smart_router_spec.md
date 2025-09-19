# Smart Router Specification

## Overview

The Smart Router is an intelligent routing system that automatically classifies user research requests and routes them to the appropriate research engines. It uses LLM-based classification to determine whether a request should go to Deep Research, Code Research, or Scientific Research engines. This component is critical for the UAgent system's ability to automatically handle diverse research requests without manual categorization.

## Requirements

### Functional Requirements
- Classify user requests into three categories: DEEP_RESEARCH, SCIENTIFIC_RESEARCH, CODE_RESEARCH
- Provide confidence scores for classifications
- Handle mixed or ambiguous requests
- Support manual routing override
- Identify sub-components required for complex requests

### Non-functional Requirements
- Classification accuracy: >95%
- Response time: <1 second for classification
- Support concurrent requests
- Graceful fallback for classification failures

### Performance Requirements
- Process up to 100 concurrent classification requests
- Cache classification results for identical requests
- Minimize LLM API calls through intelligent caching

## Interface

### Input Parameters
```python
@dataclass
class ClassificationRequest:
    user_request: str
    context: Optional[Dict[str, Any]] = None
    override_engine: Optional[str] = None
    confidence_threshold: float = 0.7
```

### Output Format
```python
@dataclass
class ClassificationResult:
    primary_engine: str  # DEEP_RESEARCH, SCIENTIFIC_RESEARCH, CODE_RESEARCH
    confidence_score: float  # 0.0 to 1.0
    sub_components: Dict[str, bool]  # Required sub-engines
    reasoning: str  # Explanation of classification decision
    workflow_plan: Dict[str, Any]  # Detailed execution plan
```

### Error Conditions
- `ClassificationError`: When LLM fails to classify
- `InvalidRequestError`: When input is malformed
- `ThresholdError`: When confidence is below threshold

## Behavior

### Normal Operation Flow
1. Receive user request
2. Check cache for identical request
3. If not cached, call LLM for classification
4. Analyze confidence score against threshold
5. Generate workflow plan based on classification
6. Cache result and return

### Classification Priority Logic
1. **Scientific Research** (highest priority):
   - Keywords: hypothesis, experiment, test, methodology, analysis, validate
   - Requires multi-engine coordination
   - Includes iterative feedback loops

2. **Code Research**:
   - Keywords: implementation, repository, code, analyze, pattern, library
   - Focus on existing code understanding

3. **Deep Research** (default):
   - General information gathering
   - Literature review without experimentation
   - Market analysis and trends

### Edge Cases
- **Ambiguous requests**: Route to most complex applicable engine (Scientific > Code > Deep)
- **Multi-type requests**: Create workflow with multiple sub-engines
- **Low confidence**: Request clarification from user
- **Empty/invalid requests**: Return error with suggested examples

### Error Handling
- LLM API failures: Retry up to 3 times with exponential backoff
- Network timeouts: Use cached fallback classification
- Invalid responses: Use rule-based classification as fallback

## Testing

### Test Scenarios
1. **Classification Accuracy Tests**:
   - 100 labeled examples for each engine type
   - Measure accuracy, precision, recall
   - Target: >95% accuracy

2. **Performance Tests**:
   - Concurrent request handling (100 simultaneous)
   - Response time under load
   - Cache hit ratio measurement

3. **Edge Case Tests**:
   - Ambiguous requests
   - Very long requests (>1000 words)
   - Non-English requests
   - Empty or malformed input

4. **Integration Tests**:
   - End-to-end workflow generation
   - Sub-engine coordination
   - Error propagation and recovery

### Success Criteria
- Classification accuracy >95% on test dataset
- Average response time <1 second
- Cache hit ratio >70% for repeated requests
- Zero data loss during failures
- Graceful degradation under high load

## Implementation Notes

### LLM Integration
- Primary: Anthropic Claude for classification
- Fallback: OpenAI GPT for redundancy
- Prompt engineering for consistent classification
- Temperature set to 0.1 for deterministic results

### Caching Strategy
- Redis for classification result caching
- Key: SHA-256 hash of normalized request
- TTL: 24 hours for classification results
- Invalidation on engine updates

### Monitoring and Metrics
- Classification accuracy tracking
- Response time monitoring
- Error rate alerting
- Cache performance metrics