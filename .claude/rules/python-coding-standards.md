---
paths:
  - "**/*.py"
  - tests/**/*.py
---

# Python Coding Standards (Universal)

## Type Hints & Docstrings
- All public functions: type hints required
- Public functions: Google-style docstrings
- Variable names: descriptive, minimum 3 characters

## Best Practices
- Use context managers (`with` statements)
- Avoid mutable default arguments
- Use f-strings for formatting
- Prefer pathlib over os.path

## Common Patterns

```python
# File I/O
from pathlib import Path
import yaml

def read_config(path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    with path.open("r") as f:
        return yaml.safe_load(f)

# Error handling
def process_data(data: list[dict]) -> list[dict]:
    """Process data with proper error handling.

    Args:
        data: List of dictionaries to process.

    Returns:
        Processed data.

    Raises:
        KeyError: If required key is missing.
    """
    try:
        return [transform(item) for item in data]
    except KeyError as e:
        logger.error(f"Missing key: {e}")
        raise

# Type hints for complex types
from typing import Optional, Union

def fetch_data(
    url: str,
    timeout: Optional[int] = None
) -> Union[dict, None]:
    """Fetch data from URL with optional timeout."""
    ...
```

**Note**: Style guidelines (line length, quotes, etc.) are handled by Ruff. DO NOT include style rules here.
