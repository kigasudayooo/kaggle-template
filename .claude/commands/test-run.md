# Test Execution (Universal)

## Run Tests

```bash
# All tests
uv run pytest

# Specific file
uv run pytest tests/test_module.py

# Specific test
uv run pytest tests/test_module.py::test_function

# With coverage
uv run pytest --cov=src --cov-report=html
```

## Test Patterns

```python
import pytest

def test_basic():
    """Basic test example."""
    result = my_function(input_data)
    assert result == expected

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_parameterized(input, expected):
    """Parameterized test."""
    assert double(input) == expected

@pytest.fixture
def sample_data():
    """Test fixture."""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """Test using fixture."""
    assert sample_data["key"] == "value"
```

## Debugging

```bash
# Stop on first failure
uv run pytest -x

# Verbose output
uv run pytest -v

# Show print statements
uv run pytest -s
```
