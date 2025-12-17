# Pytest Testing Guide

## Basic Test Structure

```python
# tests/test_calculator.py
import pytest
from src.calculator import add, divide

def test_add():
    """Test addition function."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_divide():
    """Test division function."""
    assert divide(10, 2) == 5
    assert divide(7, 2) == 3.5

def test_divide_by_zero():
    """Test division by zero raises exception."""
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)
```

## Fixtures

Fixtures provide reusable test data or setup.

```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com"
    }

def test_with_fixture(sample_data):
    """Test using fixture data."""
    assert sample_data["name"] == "Alice"
    assert sample_data["age"] == 30

@pytest.fixture
def temp_file(tmp_path):
    """Create temporary file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    return file_path

def test_file_reading(temp_file):
    """Test file operations."""
    content = temp_file.read_text()
    assert content == "test content"
```

## Parametrization

Test multiple inputs efficiently.

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (0, 0),
    (-1, -2),
])
def test_double(input, expected):
    """Test doubling function with multiple inputs."""
    assert double(input) == expected

@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (10, -5, 5),
    (0, 0, 0),
])
def test_add_parametrized(a, b, expected):
    """Test addition with multiple cases."""
    assert add(a, b) == expected
```

## Mocking

Mock external dependencies.

```python
from unittest.mock import Mock, patch

def test_api_call():
    """Test function that calls external API."""
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"key": "value"}

        result = fetch_data("https://api.example.com")
        assert result["key"] == "value"
        mock_get.assert_called_once()

@patch('src.module.expensive_function')
def test_with_mock(mock_func):
    """Test with mocked expensive function."""
    mock_func.return_value = 42
    result = process_data()
    assert result == 42
```

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_unit/               # Unit tests
│   ├── test_models.py
│   ├── test_utils.py
│   └── test_data.py
├── test_integration/        # Integration tests
│   └── test_pipeline.py
└── test_e2e/                # End-to-end tests
    └── test_workflow.py
```

## Coverage

```bash
# Run with coverage
uv run pytest --cov=src --cov-report=html

# View HTML report
open htmlcov/index.html

# Show missing lines
uv run pytest --cov=src --cov-report=term-missing
```

## Common Patterns

### Testing Exceptions

```python
def test_invalid_input():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        validate_number(-1)
```

### Testing Logging

```python
def test_logging(caplog):
    """Test that function logs correctly."""
    process_item(item)
    assert "Processing item" in caplog.text
```

### Testing Files

```python
def test_file_creation(tmp_path):
    """Test file creation."""
    output_file = tmp_path / "output.txt"
    create_file(output_file)
    assert output_file.exists()
    assert output_file.read_text() == "expected content"
```

## Best Practices

1. **One assertion per test** (when possible)
2. **Clear test names**: `test_<what>_<condition>_<expected>`
3. **Arrange-Act-Assert pattern**
4. **Use fixtures** for common setup
5. **Mock external dependencies**
6. **Test edge cases** (empty, null, boundary values)
7. **Keep tests fast** (< 1 second each)
