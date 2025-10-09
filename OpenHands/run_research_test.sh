#!/bin/bash
# Wrapper script to run the parallel research CLI test

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Parallel UAgent Research - CLI Test Runner"
echo "════════════════════════════════════════════════════════════"
echo ""

# Change to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Check if virtual environment exists
if [ -d "../.venv" ]; then
    echo "✓ Activating virtual environment..."
    source ../.venv/bin/activate
elif [ -d ".venv" ]; then
    echo "✓ Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠ No virtual environment found, using system Python"
fi

# Set environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"
export RESEARCH_DATABASE_URL="${RESEARCH_DATABASE_URL:-sqlite+aiosqlite:///./test_research.db}"

# Parse arguments
TEST_TYPE="full"
if [ "$1" == "--basic" ]; then
    TEST_TYPE="basic"
    shift
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    python3 test_parallel_research_cli.py --help
    exit 0
fi

echo "Configuration:"
echo "  - Python: $(which python3)"
echo "  - Database: $RESEARCH_DATABASE_URL"
echo "  - Test type: $TEST_TYPE"
echo ""

# Run the test
if [ "$TEST_TYPE" == "basic" ]; then
    echo "Running basic component tests..."
    python3 test_parallel_research_cli.py --basic-test "$@"
else
    echo "Running full parallel research test..."
    python3 test_parallel_research_cli.py --full-test "$@"
fi

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "✅ All tests passed!"
    echo "════════════════════════════════════════════════════════════"
else
    echo "════════════════════════════════════════════════════════════"
    echo "❌ Tests failed with exit code $exit_code"
    echo "════════════════════════════════════════════════════════════"
fi

exit $exit_code
