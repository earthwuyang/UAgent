# Parallel UAgent Research - CLI Unit Test

This document describes how to test the parallel UAgent research functionality in CLI mode without needing the full web server.

## Overview

The test suite includes:
1. **Basic component tests** - Verify imports and basic functionality
2. **Full parallel research test** - Complete end-to-end test of the research tree orchestrator

## Files

- `test_parallel_research_cli.py` - Main Python test script
- `run_research_test.sh` - Bash wrapper script for easy execution
- `PARALLEL_RESEARCH_TEST_README.md` - This file

## Quick Start

### Run Basic Tests

```bash
./run_research_test.sh --basic
```

This will:
- Import core modules
- Create test nodes
- Verify PUCT calculations
- Validate budget creation

### Run Full Parallel Research Test

```bash
./run_research_test.sh
```

This will:
- Initialize the research orchestrator
- Create a research tree
- Run parallel exploration with PUCT-based node selection
- Track events via event bus
- Display comprehensive statistics

## Usage

### Using the Bash Wrapper (Recommended)

```bash
# Basic tests
./run_research_test.sh --basic

# Full test
./run_research_test.sh

# Show help
./run_research_test.sh --help
```

### Using Python Directly

```bash
# Basic tests
python3 test_parallel_research_cli.py --basic-test

# Full test
python3 test_parallel_research_cli.py --full-test

# Show help
python3 test_parallel_research_cli.py --help
```

## Configuration

### Environment Variables

```bash
# Set custom database URL
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./my_test.db"
./run_research_test.sh

# Or inline
RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./my_test.db" ./run_research_test.sh
```

### Test Parameters

Edit `test_parallel_research_cli.py` to modify:

```python
config = {
    'max_iterations': 10,      # Maximum iterations to run
    'max_cost': 5.0,           # Maximum cost in dollars
    'max_parallel': 3,         # Number of parallel tasks
    'max_tokens': 50000,       # Maximum tokens to use
}
```

## What Gets Tested

### 1. Module Imports
- TreeOrchestrator
- Event bus system
- Research tree models
- Budget management
- Database models (if available)

### 2. Event System
- Event subscription
- Event publishing
- Event type tracking

### 3. Tree Orchestrator
- Initialization
- Parallel task execution
- PUCT-based node selection
- Budget management
- Graceful shutdown

### 4. Statistics Collection
- Node counts by type
- PUCT scores
- Iteration counts
- Cost tracking
- Token usage

## Expected Output

### Basic Test Output

```
🔧 Testing basic component initialization...
  ✅ Created test node: test_node_1
  ✅ Created budget: max_iterations=10
  ✅ PUCT calculation: 0.623
```

### Full Test Output

```
================================================================================
🧪 Parallel UAgent Research - CLI Unit Test
================================================================================

📦 Importing required modules...
✅ Modules imported successfully

🗄️  Initializing database...
✅ Database initialized: sqlite+aiosqlite:///./test_research.db

📡 Setting up event bus...
✅ Event bus configured

⚙️  Configuring research parameters...
   - Max iterations: 10
   - Max parallel: 3
   - Budget: $5.0, 50000 tokens

🎯 Research Goal:
   Investigate machine learning optimization techniques for large language models

🎭 Creating TreeOrchestrator...
✅ Orchestrator created

🚀 Starting parallel research exploration...
--------------------------------------------------------------------------------
📬 Event: research_started - Research exploration started
📬 Event: node_created - Created root node
📬 Event: iteration_complete - Iteration 1 complete
...
--------------------------------------------------------------------------------
✅ Research completed successfully!

📊 Research Results:
================================================================================
Duration: 45.23 seconds

Tree Statistics:
  - Total nodes: 23
  - Root node: root_123abc
  - Depth levels: 4

Node Types:
  - root: 1
  - idea: 8
  - hypothesis: 10
  - experiment: 4

Top 5 Nodes by PUCT Score:
  1. node_abc123... (PUCT: 2.456, visits: 5)
  2. node_def456... (PUCT: 2.234, visits: 3)
  ...

Execution Statistics:
  - Iterations: 10
  - Completed nodes: 20
  - Failed nodes: 3
  - Total cost: $2.3456
  - Total tokens: 12,345

Events Received: 47
  - node_created: 23
  - node_updated: 15
  - iteration_complete: 10
  ...

================================================================================
✅ ALL TESTS PASSED
================================================================================
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Make sure you're in the correct directory
cd /home/wuy/AI/UAgent/OpenHands

# Check Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Try running with explicit path
python3 ./test_parallel_research_cli.py --basic-test
```

### Database Errors

If database initialization fails:

```bash
# Run without database (memory-only mode)
# The script will automatically fall back to memory mode

# Or specify a different database
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./test_db_$(date +%s).db"
./run_research_test.sh
```

### Timeout Issues

If the test times out:

1. Reduce iterations in the script:
   ```python
   config = {
       'max_iterations': 5,  # Reduced from 10
       ...
   }
   ```

2. Or increase timeout:
   ```python
   result = await asyncio.wait_for(
       orchestrator.run(...),
       timeout=600  # Increased from 300 seconds
   )
   ```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test Parallel Research

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run basic tests
        run: ./run_research_test.sh --basic
      - name: Run full tests
        run: ./run_research_test.sh
```

## Performance Benchmarks

Expected performance (approximate):

| Test Type | Duration | Nodes Created | Iterations |
|-----------|----------|---------------|------------|
| Basic     | < 1s     | 3             | 0          |
| Full (minimal) | 30-60s | 10-20 | 5-10 |
| Full (standard) | 2-5min | 20-40 | 10-20 |

## Next Steps

After running tests successfully:

1. **Modify research goal** - Edit the `research_goal` variable to test different scenarios
2. **Adjust parameters** - Change `max_parallel`, `max_iterations`, etc.
3. **Add custom event handlers** - Extend `event_listener` function for custom logging
4. **Integrate with your workflow** - Use these tests in your development pipeline

## Support

For issues or questions:
1. Check the logs in the output
2. Review `test_research.db` for data inspection
3. Enable debug logging:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

## License

Same as OpenHands main project.
