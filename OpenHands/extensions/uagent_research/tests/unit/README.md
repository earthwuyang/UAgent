# Unit Tests

This directory contains unit tests for the UAgent Research extension.

## Structure

```
tests/unit/
├── __init__.py
├── README.md (this file)
├── conftest.py  # Shared fixtures
├── test_research_tree_models.py  # Tests for research tree models
└── test_control_bus.py  # Tests for control bus (NEW)
```

## Running Tests

### Run all unit tests
```bash
pytest tests/unit/
```

### Run specific test file
```bash
pytest tests/unit/test_research_tree_models.py
```

### Run specific test class
```bash
pytest tests/unit/test_research_tree_models.py::TestResearchNode
```

### Run specific test
```bash
pytest tests/unit/test_research_tree_models.py::TestResearchNode::test_node_creation_minimal
```

### Run with coverage
```bash
pytest tests/unit/ --cov=uagent_research.models --cov-report=html
```

### Run with verbose output
```bash
pytest tests/unit/ -v
```

### Run only unit tests (using marker)
```bash
pytest -m unit
```

## Test Conventions

### Naming
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<what_is_being_tested>()`

### Organization
- Group related tests in classes
- Use descriptive test names that explain the scenario
- One assertion per test when possible
- Use fixtures for common setup

### Markers
- `@pytest.mark.unit`: Mark as unit test
- `@pytest.mark.asyncio`: Mark as async test (if needed)
- `@pytest.mark.parametrize`: Parameterize tests for multiple inputs

### Fixtures

Common fixtures are available in test files:
- `sample_artifact`: Factory for creating Artifact instances
- `sample_node`: Factory for creating ResearchNode instances
- `sample_tree`: Factory for creating ResearchTree instances
- `populated_tree`: Pre-populated tree for testing traversal

## Control Bus Testing

### Test Classes
- `TestControlMessage`: Tests for ControlMessage Pydantic model
- `TestControlBus`: Core functionality tests
- `TestControlBusEdgeCases`: Edge cases and error conditions
- `TestControlBusConcurrency`: Concurrent operations and stress tests
- `TestControlBusSingleton`: Singleton pattern tests
- `TestControlBusIntegration`: Integration scenarios

### Fixtures
- `sample_control_message`: Factory for creating ControlMessage instances
- `control_bus`: Fresh ControlBus instance for each test
- `experiment_id`: Standard experiment ID for tests
- `subscriber_helper`: Helper for creating test subscribers

### Running Control Bus Tests
```bash
# Run all control bus tests
pytest tests/unit/test_control_bus.py

# Run specific test class
pytest tests/unit/test_control_bus.py::TestControlMessage

# Run with coverage
pytest tests/unit/test_control_bus.py --cov=control.control_bus --cov-report=html
```

### Coverage Goals
- Control bus module: >85%

### Example Test

```python
import pytest
from uagent_research.models.research_tree import ResearchNode, NodeType, NodeStatus

@pytest.mark.unit
class TestResearchNode:
    def test_node_creation_minimal(self, sample_node):
        """Test creating a node with minimal required fields."""
        node = sample_node(
            node_id="test-1",
            node_type=NodeType.IDEA,
            title="Test Idea",
            content="Test content"
        )
        
        assert node.id == "test-1"
        assert node.type == NodeType.IDEA
        assert node.status == NodeStatus.PENDING
        assert node.visits == 0
        assert node.prior == 0.5
```

## Coverage Goals

- Overall: >80%
- Models module: >90%
- Control bus module: >85% (NEW)
- Critical paths: 100%

## Best Practices

1. **Test behavior, not implementation**: Focus on what the code does, not how it does it
2. **Keep tests independent**: Each test should be able to run in isolation
3. **Use fixtures for setup**: Avoid duplicating setup code
4. **Test edge cases**: Empty collections, None values, boundary conditions
5. **Test error conditions**: Invalid inputs, exceptions
6. **Keep tests fast**: Unit tests should run in milliseconds
7. **Use descriptive assertions**: Make failures easy to understand
8. **Document complex tests**: Add docstrings explaining the scenario

## Troubleshooting

### Import errors
Ensure the package is installed in development mode:
```bash
pip install -e .
```

### Fixture not found
Check that fixtures are defined in the test file or `conftest.py`.

### Coverage not measured
Ensure `pytest-cov` is installed:
```bash
pip install pytest-cov
```
