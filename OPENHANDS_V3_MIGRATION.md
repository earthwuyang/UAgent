# OpenHands V3 Migration Guide

## Overview

The OpenHands V3 integration introduces a headless subprocess execution model for deterministic scientific experiments, replacing the previous action-server based approach.

## Key Changes

### 1. New Files Added

- `backend/app/integrations/openhands_codeact_bridge_v3.py` - V3 bridge implementation
- `backend/requirements-openhands.txt` - Additional dependencies for headless execution
- `test_openhands_v3.py` - Test suite for V3 functionality

### 2. Modified Files

- `backend/app/core/research_engines/scientific_research.py` - Added V3 execution path (feature-flagged)

### 3. Architecture Changes

**V2 (Previous)**:
- Action execution server with HTTP API
- UAgent orchestrates CodeAct loop
- Complex state management between UAgent and OpenHands

**V3 (New)**:
- Headless subprocess execution
- OpenHands owns entire CodeAct loop
- Deterministic artifact-based communication
- One subprocess per experiment

## Configuration

### Environment Variables

Configure V3 behavior:

```bash
# V3 headless mode is ENABLED by default
# To disable V3 and use V2 legacy mode:
export UAGENT_OPENHANDS_V3=0
# or
export UAGENT_OPENHANDS_V3=false

# Configure execution limits
export UAGENT_OPENHANDS_MAX_STEPS=80      # Max CodeAct steps (default: 80)
export UAGENT_OPENHANDS_MAX_MINUTES=30     # Max execution time (default: 30)
export UAGENT_OPENHANDS_DISABLE_BROWSER=true  # Disable browser (default: true)

# LLM configuration (inherited from UAgent environment)
export LLM_MODEL=gpt-4
export LLM_API_KEY=your-key
export LLM_BASE_URL=https://api.openai.com/v1

# Or use LiteLLM environment variables
export LITELLM_MODEL=gpt-4
export LITELLM_API_KEY=your-key
export LITELLM_API_BASE=https://api.openai.com/v1
```

### Workspace Structure

V3 creates a structured workspace for each experiment:

```
workspace/
├── README_UAGENT.md          # Contract documentation
├── code/                     # Generated code
├── data/                     # Data files
├── experiments/
│   └── <experiment_id>/
│       └── results/
│           └── final.json    # Required output artifact
├── logs/
│   ├── openhands_stdout.log
│   └── openhands_stderr.log
└── workspace/                # Working directory
```

## Artifact Contract

Each experiment must produce `experiments/<id>/results/final.json`:

```json
{
  "success": true,
  "data": {
    "raw_measurements": [...]
  },
  "analysis": {
    "mean": 1.23,
    "std": 0.45,
    "statistical_tests": {...}
  },
  "conclusions": [
    "Conclusion 1",
    "Conclusion 2"
  ],
  "measurements": [1.1, 1.2, 1.3],
  "errors": []
}
```

## Installation

1. Ensure OpenHands source is available:
```bash
# If not already present, clone OpenHands
git clone https://github.com/All-Hands-AI/OpenHands.git OpenHands
```

2. Install additional dependencies:
```bash
pip install -r backend/requirements-openhands.txt
```

3. Verify system dependencies:
```bash
# On Ubuntu/Debian
sudo apt-get install tmux

# On macOS
brew install tmux
```

## Testing

Run the V3 test suite:

```bash
# Basic test (V3 is now default)
python test_openhands_v3.py

# Test with V2 legacy mode
UAGENT_OPENHANDS_V3=0 python test_openhands_v2_legacy.py

# Full integration test
pytest test/integration/test_scientific_research.py -v
```

## Migration Path

### Phase 1: Default Enabled (Current)
- V3 enabled by default
- Set `UAGENT_OPENHANDS_V3=0` to disable and use V2
- V2 remains as fallback on V3 failures

### Phase 2: Gradual Rollout
- Monitor V3 success rates
- Tune timeout and step limits
- Gather performance metrics

### Phase 3: Default Switch
- Make V3 default after validation
- Keep V2 code for one release

### Phase 4: Cleanup
- Remove V2 code paths
- Simplify configuration

## Rollback

To rollback to V2:
```bash
unset UAGENT_OPENHANDS_V3
# or
export UAGENT_OPENHANDS_V3=0
```

## Benefits of V3

1. **Deterministic Execution**: One process per experiment with clear start/end
2. **Better Isolation**: Each experiment runs in isolated subprocess
3. **Simpler Debugging**: Standalone logs and artifacts
4. **Reduced Complexity**: No action-server orchestration
5. **Improved Reliability**: Process-level timeout and cleanup

## Known Limitations

1. Requires `tmux` for LocalRuntime on Unix systems
2. No Docker support in V3 (uses local runtime only)
3. Browser automation disabled by default
4. Subprocess overhead for small tasks

## Troubleshooting

### Issue: "OpenHands directory not found"
**Solution**: Ensure OpenHands is cloned at `./OpenHands/`

### Issue: "tmux: command not found"
**Solution**: Install tmux for your OS (see Installation section)

### Issue: "No final.json found"
**Solution**: Check `workspace/logs/openhands_stdout.log` for execution errors

### Issue: Timeout errors
**Solution**: Increase `UAGENT_OPENHANDS_MAX_MINUTES` for long-running experiments

## Support

For issues or questions:
1. Check `workspace/logs/` for execution logs
2. Review the artifact contract in `README_UAGENT.md`
3. File issues with V3-specific tag in the repository