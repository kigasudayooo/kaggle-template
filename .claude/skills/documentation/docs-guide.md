# Documentation Guide

## Google-Style Docstrings

### Functions

```python
def calculate_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    """Calculate evaluation metrics for predictions.

    Args:
        predictions: Model predictions as list of floats.
        targets: Ground truth targets as list of floats.

    Returns:
        Dictionary containing metric names and values.
        Keys: 'mse', 'rmse', 'mae', 'r2'.

    Raises:
        ValueError: If predictions and targets have different lengths.

    Example:
        >>> predictions = [1.0, 2.0, 3.0]
        >>> targets = [1.1, 2.1, 2.9]
        >>> metrics = calculate_metrics(predictions, targets)
        >>> print(metrics['rmse'])
        0.1
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    # Implementation...
```

### Classes

```python
class DataProcessor:
    """Process and transform raw data for model training.

    This class handles data loading, cleaning, and feature engineering
    for the competition dataset.

    Attributes:
        config: Configuration dictionary containing processing parameters.
        scaler: StandardScaler instance for feature normalization.

    Example:
        >>> processor = DataProcessor(config={'normalize': True})
        >>> X_train = processor.fit_transform(raw_data)
        >>> X_test = processor.transform(test_data)
    """

    def __init__(self, config: dict):
        """Initialize processor with configuration.

        Args:
            config: Configuration dictionary with keys:
                - 'normalize': bool, whether to normalize features
                - 'feature_cols': list of feature column names
        """
        self.config = config
        self.scaler = StandardScaler() if config.get('normalize') else None
```

## README Structure

```markdown
# Project Name

Brief description of the project in 1-2 sentences.

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

\`\`\`bash
# Clone repository
git clone https://github.com/username/repo.git
cd repo

# Install dependencies
uv sync
\`\`\`

## Quick Start

\`\`\`python
from src.module import MyClass

# Basic usage example
obj = MyClass(config)
result = obj.process(data)
\`\`\`

## Usage

### Training

\`\`\`bash
uv run python src/scripts/train.py --config configs/baseline.yaml
\`\`\`

### Evaluation

\`\`\`bash
uv run python src/scripts/evaluate.py --model models/best.pth
\`\`\`

## Project Structure

\`\`\`
project/
├── src/
│   ├── data/         # Data loading and processing
│   ├── models/       # Model definitions
│   └── utils/        # Utility functions
├── configs/          # Configuration files
├── tests/            # Unit tests
└── docs/             # Documentation
\`\`\`

## Configuration

Edit `configs/config.yaml`:

\`\`\`yaml
model:
  name: resnet50
  pretrained: true

training:
  epochs: 100
  batch_size: 32
\`\`\`

## Development

\`\`\`bash
# Run tests
uv run pytest

# Run linting
uv run ruff check src/

# Format code
uv run ruff format src/
\`\`\`

## License

MIT License

## Contact

- Author: Your Name
- Email: your.email@example.com
- GitHub: @username
\`\`\`

## API Documentation

### REST API

\`\`\`markdown
### GET /api/predict

Make prediction on input data.

**Request:**

\`\`\`json
{
  "data": [1.0, 2.0, 3.0]
}
\`\`\`

**Response:**

\`\`\`json
{
  "prediction": 0.85,
  "confidence": 0.92
}
\`\`\`

**Errors:**

- `400 Bad Request`: Invalid input format
- `500 Internal Server Error`: Model prediction failed
\`\`\`

## Inline Comments

Use comments sparingly, only when code isn't self-explanatory:

```python
# Good: Explains WHY, not WHAT
# Use exponential moving average to smooth noisy predictions
smoothed = 0.9 * current + 0.1 * previous

# Bad: States the obvious
# Loop through items
for item in items:
    ...

# Good: Clarifies complex logic
# Binary search requires sorted array, but maintaining sort order
# is expensive, so we sort once and cache the result
if not self._sorted_cache:
    self._sorted_cache = sorted(self.data)
```

## Type Hints

Always use type hints for function signatures:

```python
from typing import Optional, Union
from pathlib import Path

def load_model(
    path: Path,
    device: str = 'cuda',
    strict: bool = True
) -> Optional[torch.nn.Module]:
    """Load trained model from checkpoint."""
    ...

def process_batch(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module
) -> tuple[torch.Tensor, dict[str, float]]:
    """Process a single batch and return predictions and metrics."""
    ...
```

## Best Practices

1. **Docstrings for all public functions/classes**
2. **Update docs when code changes**
3. **Include examples in docstrings**
4. **Keep README up-to-date**
5. **Document assumptions and limitations**
6. **Use type hints consistently**
7. **Explain WHY, not WHAT in comments**
